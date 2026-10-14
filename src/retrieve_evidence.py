
from pathlib import Path
import re

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTES_PATH = PROJECT_DIR / "data" / "context_notes.csv"
ANOMALIES_PATH = PROJECT_DIR / "outputs" / "route_anomalies.csv"
VECTOR_DB_PATH = PROJECT_DIR / "outputs" / "chroma_db"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "route_evidence.csv"

COLLECTION_NAME = "shipping_context_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10

NON_SUPPORTING_PATTERNS = [
    r"costs were not significantly affected",
    r"costs were not affected",
    r"no impact on costs",
    r"not part of this dataset",
    r"returned to normal",
    r"normal conditions",
    r"costs? (?:remained|remain) stable",
]

COST_DRIVER_PATTERN = (
    r"higher trip costs|higher costs|"
    r"costs? (?:rose|increased|increasing)|"
    r"pushing up transportation costs|"
    r"temporary surcharge|surcharge applied|"
    r"forcing longer detours|"
    r"fuel prices? (?:rose|increased)|"
    r"toll (?:increase|increased|plaza)|"
    r"disrupted normal truck movement"
)

MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)

DATE_RANGE_PATTERN = re.compile(
    rf"\b(?:from\s+)?({MONTHS})\s+(\d{{1,2}})\s+to\s+"
    rf"({MONTHS})\s+(\d{{1,2}})",
    flags=re.IGNORECASE,
)


def normalize_route(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def route_is_applicable(applies_to, origin, destination):
    scope = str(applies_to).strip()

    if scope.casefold() == "all routes":
        return True

    expected = normalize_route(f"{origin}-{destination}")
    return normalize_route(scope) == expected


def explicit_date_range(note_text, anchor_date):
    match = DATE_RANGE_PATTERN.search(str(note_text))
    if not match:
        return None

    start_month, start_day, end_month, end_day = match.groups()
    year = pd.Timestamp(anchor_date).year

    start = pd.to_datetime(
        f"{start_month} {start_day} {year}", errors="coerce"
    )
    end = pd.to_datetime(
        f"{end_month} {end_day} {year}", errors="coerce"
    )

    if pd.isna(start) or pd.isna(end):
        return None

    if end < start:
        end += pd.DateOffset(years=1)

    return start.normalize(), end.normalize()


def note_overlaps_week(note, week_start):
    week_start = pd.Timestamp(week_start).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    anchor_date = pd.Timestamp(note["date"]).normalize()
    text = str(note["note"])

    date_range = explicit_date_range(text, anchor_date)

    if date_range is not None:
        start, end = date_range
        return start <= week_end and end >= week_start

    # Notes without an explicit duration are limited to their dated week.
    anchor_week = anchor_date - pd.Timedelta(days=anchor_date.weekday())

    return week_start == anchor_week


def supports_cost_increase(note_text):
    text = str(note_text)

    if any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in NON_SUPPORTING_PATTERNS
    ):
        return False

    return bool(re.search(COST_DRIVER_PATTERN, text, re.IGNORECASE))


def retrieve_candidates(collection, model, route, query_text):
    """Use semantic search to retrieve candidates; never decide verdicts here."""
    query = (
        f"Route: {route['origin']}-{route['destination']}. "
        f"Route type: {route['route_type']}. "
        f"Week beginning: {route['week_of']}. "
        f"Observed cost per tonne-km: {route['cost_per_tonne_km']}. "
        f"Question: What documented real-world event could explain this "
        f"shipping cost increase? {query_text}"
    )

    embedding = model.encode(
        query, normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(TOP_K, collection.count()),
        include=["metadatas", "distances"],
    )

    candidates = []

    for metadata, distance in zip(
        result["metadatas"][0],
        result["distances"][0],
    ):
        candidates.append({
            **metadata,
            "distance": float(distance),
        })

    return candidates


def retrieve_evidence():
    notes = pd.read_csv(NOTES_PATH).fillna("")
    routes = pd.read_csv(
        ANOMALIES_PATH, parse_dates=["week_of"]
    )

    notes["date"] = pd.to_datetime(notes["date"], errors="raise")

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Only flagged route-weeks need evidence decisions.
    flagged = routes[routes["flagged"]].copy()
    results = []

    for _, route in flagged.iterrows():
        route_label = f"{route['origin']}-{route['destination']}"

        candidates = retrieve_candidates(
            collection,
            model,
            route,
            "Find a dated fuel, toll, weather, disruption, or "
            "festival note that explicitly supports higher costs.",
        )

        eligible = []

        for candidate in candidates:
            # Semantic similarity alone never establishes applicability.
            if not route_is_applicable(
                candidate["applies_to"],
                route["origin"],
                route["destination"],
            ):
                continue

            source_note = notes[
                notes["note_id"].astype(str)
                == str(candidate["note_id"])
            ]

            if source_note.empty:
                continue

            note = source_note.iloc[0]

            if not note_overlaps_week(note, route["week_of"]):
                continue

            if not supports_cost_increase(note["note"]):
                continue

            eligible.append({
                "note_id": str(note["note_id"]),
                "date": note["date"],
                "note": str(note["note"]),
                "distance": candidate["distance"],
            })

        # Stable ordering makes the selected note deterministic.
        eligible.sort(
            key=lambda item: (
                item["date"],
                item["note_id"],
                item["distance"],
            )
        )

        if eligible:
            selected = eligible[0]
            verdict = "Justified"
            matched_note_id = selected["note_id"]
            reason = (
                f"Cost increase supported by {matched_note_id}: "
                f"{selected['note']}"
            )
        else:
            verdict = "Unexplained"
            matched_note_id = ""
            reason = (
                "No retrieved note passed route, date, and "
                "cost-impact validation for this route and week."
            )

        results.append({
            "origin": route["origin"],
            "destination": route["destination"],
            "route_type": route["route_type"],
            "week_of": route["week_of"].date().isoformat(),
            "cost_per_tonne_km": route["cost_per_tonne_km"],
            "flagged": bool(route["flagged"]),
            "verdict": verdict,
            "matched_note_id": matched_note_id,
            "reason": reason,
        })

    evidence_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    evidence_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Flagged route-weeks: {len(flagged)}")
    print(
        "Justified:",
        int((evidence_df["verdict"] == "Justified").sum()),
    )
    print(
        "Unexplained:",
        int((evidence_df["verdict"] == "Unexplained").sum()),
    )
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    retrieve_evidence()
