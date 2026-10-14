
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_DIR / "outputs" / "chroma_db"
COLLECTION_NAME = "shipping_context_notes"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

TEST_CASES = [
    {
        "case_id": "R001",
        "query": "Heavy flooding and detours increased shipping costs on Chennai Bangalore",
        "expected_note_id": "N001",
    },
    {
        "case_id": "R002",
        "query": "Festival demand caused transporter surcharge on Ahmedabad Mumbai",
        "expected_note_id": "N002",
    },
    {
        "case_id": "R003",
        "query": "Diesel prices increased nationwide and raised transport costs",
        "expected_note_id": "N003",
    },
    {
        "case_id": "R004",
        "query": "Mumbai Delhi highway maintenance but costs were not significantly affected",
        "expected_note_id": "N005",
    },
]


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    passed = 0

    for case in TEST_CASES:
        embedding = model.encode(
            case["query"],
            normalize_embeddings=True,
        ).tolist()

        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(TOP_K, collection.count()),
            include=["metadatas", "distances"],
        )

        retrieved = [
            metadata["note_id"]
            for metadata in result["metadatas"][0]
        ]

        success = case["expected_note_id"] in retrieved
        passed += int(success)

        print(
            f"{case['case_id']} | "
            f"Expected: {case['expected_note_id']} | "
            f"Retrieved: {retrieved} | "
            f"{'PASS' if success else 'FAIL'}"
        )

    print(f"\nRecall@{TOP_K}: {passed}/{len(TEST_CASES)} "
          f"({passed / len(TEST_CASES) * 100:.1f}%)")


if __name__ == "__main__":
    main()
