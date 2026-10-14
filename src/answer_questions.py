from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import sys

import chromadb
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_DIR / "outputs" / "chroma_db"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "final_output.csv"
EVIDENCE_PATH = PROJECT_DIR / "outputs" / "route_evidence.csv"
USAGE_LOG_PATH = PROJECT_DIR / "outputs" / "qa_usage_log.jsonl"

COLLECTION_NAME = "shipping_context_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Keep the model that you have already tested successfully.
GEMINI_MODEL = "gemini-3.5-flash-lite"

load_dotenv(PROJECT_DIR / ".env")


# --------------------------------------------------
# Load data and clients
# --------------------------------------------------

def load_sources():
    """Load calculated results, validated evidence and AI clients."""

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            "Run the pipeline first: outputs/final_output.csv is missing."
        )

    if not EVIDENCE_PATH.exists():
        raise FileNotFoundError(
            "Run the evidence pipeline first: "
            "outputs/route_evidence.csv is missing."
        )

    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            "Vector database is missing. Run src/build_vector_store.py first."
        )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Add it to the project-root .env file."
        )

    results = pd.read_csv(OUTPUT_PATH).fillna("")
    evidence = pd.read_csv(EVIDENCE_PATH).fillna("")

    required_result_columns = {
        "route",
        "week_of",
        "cost_per_tonne_km",
        "flagged",
        "matched_note_id",
        "reason",
    }

    required_evidence_columns = {
        "origin",
        "destination",
        "week_of",
        "verdict",
        "matched_note_id",
        "reason",
    }

    missing_results = required_result_columns - set(results.columns)
    missing_evidence = required_evidence_columns - set(evidence.columns)

    if missing_results:
        raise ValueError(
            f"final_output.csv is missing columns: {sorted(missing_results)}"
        )

    if missing_evidence:
        raise ValueError(
            f"route_evidence.csv is missing columns: "
            f"{sorted(missing_evidence)}"
        )

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load ChromaDB collection '{COLLECTION_NAME}'. "
            "Run src/build_vector_store.py first."
        ) from exc

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    llm = genai.Client(api_key=api_key)

    return results, evidence, collection, embedding_model, llm


# --------------------------------------------------
# Context-note retrieval
# --------------------------------------------------

def retrieve_notes(question, collection, embedding_model):
    """
    Retrieve semantically relevant notes.

    Retrieved notes are candidates only. They do not prove that a note
    applies to a route or week.
    """

    count = collection.count()

    if count == 0:
        return []

    embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    retrieved = collection.query(
        query_embeddings=[embedding],
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )

    notes = []

    metadatas = retrieved.get("metadatas", [[]])[0]
    documents = retrieved.get("documents", [[]])[0]
    distances = retrieved.get("distances", [[]])[0]

    for metadata, document, distance in zip(
        metadatas, documents, distances
    ):
        metadata = metadata or {}

        notes.append({
            "note_id": str(metadata.get("note_id", "Unknown")),
            "date": str(metadata.get("date", "")),
            "applies_to": str(metadata.get("applies_to", "")),
            "note": str(document or metadata.get("note", "")),
            "distance": float(distance),
        })

    return notes


def format_notes(notes):
    if not notes:
        return "No context notes were retrieved."

    return "\n".join(
        f"- Note ID: {note['note_id']} | "
        f"Date: {note['date']} | "
        f"Scope: {note['applies_to']} | "
        f"Content: {note['note']}"
        for note in notes
    )


# --------------------------------------------------
# Parse route and time period from the question
# --------------------------------------------------

MONTH_PATTERN = (
    r"\b(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)"
    r"\s+(\d{4})\b"
)

ISO_DATE_PATTERN = r"\b(\d{4}-\d{2}-\d{2})\b"


def parse_requested_route(question, known_routes):
    """
    Find a route explicitly mentioned in the question.

    Route names are derived from the actual output data rather than
    assuming that every question follows one particular naming format.
    """

    question_lower = question.casefold()

    matches = []

    for route in known_routes:
        normalized_route = str(route).strip()

        if not normalized_route:
            continue

        # Match the complete route text, case-insensitively.
        if normalized_route.casefold() in question_lower:
            matches.append(normalized_route)

    if not matches:
        return None

    # Prefer the longest matching route if route names overlap.
    return max(matches, key=len)


def filter_by_period(question, records):
    """Filter records to a requested month or exact ISO date, if provided."""

    filtered = records.copy()

    date_column = "week_of"

    if date_column not in filtered.columns:
        return filtered

    dates = pd.to_datetime(filtered[date_column], errors="coerce")

    date_match = re.search(ISO_DATE_PATTERN, question)

    if date_match:
        requested_date = pd.to_datetime(
            date_match.group(1), errors="coerce"
        )

        if pd.notna(requested_date):
            filtered = filtered[
                dates.dt.strftime("%Y-%m-%d")
                == requested_date.strftime("%Y-%m-%d")
            ]

        return filtered

    month_match = re.search(MONTH_PATTERN, question, re.IGNORECASE)

    if month_match:
        month_name, year = month_match.groups()

        month_number = pd.to_datetime(
            f"{month_name} {year}",
            format="%B %Y",
        ).month

        filtered = filtered[
            (dates.dt.year == int(year))
            & (dates.dt.month == month_number)
        ]

    return filtered


def find_relevant_results(question, results):
    """
    Select route-week records explicitly requested by the user.

    If a route is named but doesn't exist, return no records instead of
    falling back to unrelated routes.
    """

    known_routes = results["route"].astype(str).unique().tolist()
    requested_route = parse_requested_route(question, known_routes)

    if requested_route is not None:
        filtered = results[
            results["route"].astype(str).str.casefold()
            == requested_route.casefold()
        ].copy()
    else:
        # Do not use every route when a route-specific question cannot
        # be resolved. A broad question can be handled separately later.
        return results.iloc[0:0].copy()

    filtered = filter_by_period(question, filtered)

    return filtered.sort_values("week_of").tail(30)


# --------------------------------------------------
# Match calculated records to validated evidence
# --------------------------------------------------

def normalize_text(value):
    return str(value).strip().casefold()


def find_matching_evidence(route_results, evidence):
    """
    Match evidence on route endpoints and week.

    final_output.csv uses a combined route such as Chennai-Bangalore.
    route_evidence.csv stores origin and destination separately.
    """

    if route_results.empty or evidence.empty:
        return evidence.iloc[0:0].copy()

    required_result_columns = {"route", "week_of"}
    required_evidence_columns = {
        "origin",
        "destination",
        "week_of",
    }

    if not required_result_columns.issubset(route_results.columns):
        raise ValueError(
            "final_output.csv must contain route and week_of."
        )

    if not required_evidence_columns.issubset(evidence.columns):
        raise ValueError(
            "route_evidence.csv must contain origin, destination and week_of."
        )

    result_keys = set()

    for _, row in route_results.iterrows():
        route = str(row["route"]).strip().casefold()
        week = pd.to_datetime(row["week_of"], errors="coerce")

        if pd.notna(week):
            result_keys.add(
                (route, week.strftime("%Y-%m-%d"))
            )

    evidence_routes = (
        evidence["origin"].astype(str).str.strip()
        + "-"
        + evidence["destination"].astype(str).str.strip()
    ).str.casefold()

    evidence_dates = pd.to_datetime(
        evidence["week_of"], errors="coerce"
    )

    evidence_keys = []

    for route, week in zip(evidence_routes, evidence_dates):
        normalized_week = (
            week.strftime("%Y-%m-%d")
            if pd.notna(week)
            else None
        )

        evidence_keys.append((route, normalized_week))

    mask = [key in result_keys for key in evidence_keys]

    matched = evidence.loc[mask].copy()

    # Fail visibly if an expected route-week has no evidence row.
    matched_keys = set(
        zip(
            (
                matched["origin"].astype(str).str.strip()
                + "-"
                + matched["destination"].astype(str).str.strip()
            ).str.casefold(),
            pd.to_datetime(
                matched["week_of"], errors="coerce"
            ).dt.strftime("%Y-%m-%d"),
        )
    )

    missing_keys = result_keys - matched_keys

    if missing_keys:
        print(
            "\nWarning: validated evidence is missing for "
            f"{len(missing_keys)} selected route-week record(s)."
        )

    return matched


# --------------------------------------------------
# Token and cost logging
# --------------------------------------------------

def log_usage(question, model, input_tokens, output_tokens):
    """
    Append usage data to outputs/qa_usage_log.jsonl.

    Configure optional per-million-token rates in .env to calculate
    estimated cost. Without configured rates, cost is recorded as null.
    """

    input_rate = os.getenv("GEMINI_INPUT_COST_PER_MILLION_USD")
    output_rate = os.getenv("GEMINI_OUTPUT_COST_PER_MILLION_USD")

    estimated_cost = None

    if input_rate is not None and output_rate is not None:
        try:
            estimated_cost = (
                input_tokens * float(input_rate)
                + output_tokens * float(output_rate)
            ) / 1_000_000
        except ValueError:
            print(
                "Warning: invalid Gemini pricing values in .env. "
                "Estimated cost will be left blank."
            )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "question": question,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": estimated_cost,
    }

    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with USAGE_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# --------------------------------------------------
# Question answering
# --------------------------------------------------

def answer_question(question):
    results, evidence, collection, embedding_model, llm = load_sources()

    notes = retrieve_notes(
        question,
        collection,
        embedding_model,
    )

    route_results = find_relevant_results(question, results)

    matching_evidence = find_matching_evidence(
        route_results,
        evidence,
    )

    note_context = format_notes(notes)

    result_context = (
        route_results.to_csv(index=False)
        if not route_results.empty
        else "No matching calculated route-week records were found."
    )

    evidence_context = (
        matching_evidence.to_csv(index=False)
        if not matching_evidence.empty
        else "No matching validated evidence rows were found."
    )

    # Prevent the model from using other routes when route filtering fails.
    if route_results.empty:
        print("\n--- Answer ---\n")
        print(
            "I couldn't find matching route-week records for this question. "
            "Please check the route name and requested period against the "
            "available output data."
        )
        return

    prompt = f"""
You are a shipping-cost analysis assistant.

Answer the user's question using only the source records supplied below.
Never invent routes, dates, costs, percentage changes, verdicts or note IDs.

SOURCE-OF-TRUTH RULES:

1. Calculated route-week records are authoritative for cost values,
   historical comparisons, peer comparisons and anomaly flags.

2. Validated evidence records are authoritative for whether a note
   supports a specific route and week.

3. Retrieved context notes are semantic search candidates only.
   A retrieved note is NOT proof that the note applies to a route or week.

4. Claim that a note explains a route-week only when the corresponding
   validated evidence row has verdict "Justified" and its matched_note_id
   identifies that note.

5. If the evidence verdict is "Unexplained", state that the anomaly is
   unexplained by the available validated notes. Do not cite a retrieved
   note as its explanation.

6. If no matching validated evidence row exists for a route-week, say that
   validated evidence is unavailable for that record. Do not infer a verdict
   from semantic notes alone.

7. A flagged anomaly is not automatically an explained anomaly.

8. Do not claim that Gemini made the deterministic anomaly decision.

9. Use only the supplied route-week rows. If a value is absent, say so.

10. Be concise. Explain which weeks were flagged, their cost values, and
    whether an available note was validated for each week.

USER QUESTION:
{question}

RETRIEVED CONTEXT NOTES (CANDIDATES ONLY):
{note_context}

MATCHING CALCULATED ROUTE-WEEK RECORDS:
{result_context}

MATCHING VALIDATED EVIDENCE RECORDS:
{evidence_context}
"""

    response = llm.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 700,
        },
    )

    print("\n--- Answer ---\n")
    print(response.text or "No answer was returned.")

    usage = getattr(response, "usage_metadata", None)

    input_tokens = (
        getattr(usage, "prompt_token_count", 0) or 0
        if usage else 0
    )

    output_tokens = (
        getattr(usage, "candidates_token_count", 0) or 0
        if usage else 0
    )

    log_usage(
        question=question,
        model=GEMINI_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    print("\n--- Usage ---")
    print("Input tokens:", input_tokens)
    print("Output tokens:", output_tokens)
    print("Total tokens:", input_tokens + output_tokens)
    print("Usage log:", USAGE_LOG_PATH)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = input("Ask about shipping costs: ").strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        answer_question(question)
    except Exception as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()
