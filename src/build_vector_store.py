
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTES_PATH = PROJECT_DIR / "data" / "context_notes.csv"
VECTOR_DB_PATH = PROJECT_DIR / "outputs" / "chroma_db"

COLLECTION_NAME = "shipping_context_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_vector_store():
    notes = pd.read_csv(NOTES_PATH).fillna("")

    required_columns = {"note_id", "date", "applies_to", "note"}
    missing = required_columns - set(notes.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if notes["note_id"].duplicated().any():
        raise ValueError("Duplicate note IDs found in context_notes.csv")

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    # Rebuild deterministically from the source CSV on every run.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "FreightTiger shipping context notes"},
    )

    if notes.empty:
        print("No context notes found.")
        return

    documents = [
        f"Route scope: {row.applies_to}. "
        f"Date: {row.date}. Context: {row.note}"
        for row in notes.itertuples(index=False)
    ]

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=notes["note_id"].astype(str).tolist(),
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "note_id": str(row.note_id),
                "date": str(row.date),
                "applies_to": str(row.applies_to),
                "note": str(row.note),
            }
            for row in notes.itertuples(index=False)
        ],
    )

    print("\nVector store created successfully!")
    print(f"Notes indexed: {collection.count()}")
    print(f"Database path: {VECTOR_DB_PATH}")

    # Smoke test: retrieve notes related to a shipping cost increase.
    results = collection.query(
        query_embeddings=[
            model.encode(
                "Fuel price increase causing higher transportation costs",
                normalize_embeddings=True,
            ).tolist()
        ],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    print("\nSample semantic retrieval:")
    for metadata, distance in zip(
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(
            f"{metadata['note_id']} | "
            f"{metadata['applies_to']} | "
            f"distance={distance:.4f}"
        )
        print(f"  {metadata['note']}")


if __name__ == "__main__":
    build_vector_store()
