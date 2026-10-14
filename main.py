
from pathlib import Path
import pandas as pd
import csv

DATA_DIR = Path("data")

files = [
    "shipment_records.csv",
    "context_notes.csv",
    "sample_output_format_v2.csv",
]


def inspect_csv(path):
    # First, inspect the raw CSV structure
    print("\nChecking CSV structure...")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)

            expected_fields = None

            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue

                if expected_fields is None:
                    expected_fields = len(row)
                    print(f"Expected fields per row: {expected_fields}")

                if len(row) != expected_fields:
                    print(
                        f"Malformed row at line {line_number}: "
                        f"expected {expected_fields} fields, "
                        f"found {len(row)}"
                    )
                    print("Row content:", row)

    except UnicodeDecodeError:
        print("Encoding issue: unable to inspect this file as UTF-8.")
        return

    # Next, load and inspect the dataframe
    print("\nLoading dataframe...")

    try:
        df = pd.read_csv(path)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        print("\nShape:", df.shape)

        print("\nMissing values:")
        print(df.isnull().sum())

    except Exception as error:
        print(f"Could not load dataframe: {error}")


for filename in files:
    path = DATA_DIR / filename

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")
    print("=" * 60)

    if not path.exists():
        print("File not found:", path)
        continue

    inspect_csv(path)
