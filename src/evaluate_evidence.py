
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "evaluation_cases.csv"
ANOMALIES_PATH = ROOT / "outputs" / "route_anomalies.csv"
EVIDENCE_PATH = ROOT / "outputs" / "route_evidence.csv"
OUTPUT_PATH = ROOT / "outputs" / "evidence_evaluation.csv"

KEYS = ["origin", "destination", "week_of"]


def main():
    cases = pd.read_csv(CASES_PATH, dtype=str).fillna("")
    anomalies = pd.read_csv(
        ANOMALIES_PATH, parse_dates=["week_of"]
    )
    evidence = pd.read_csv(
        EVIDENCE_PATH, parse_dates=["week_of"]
    ).fillna("")

    cases["week_of"] = pd.to_datetime(
        cases["week_of"], errors="raise"
    )
    anomalies["week_of"] = pd.to_datetime(
        anomalies["week_of"], errors="raise"
    )
    evidence["week_of"] = pd.to_datetime(
        evidence["week_of"], errors="raise"
    )

    # Normalize join keys to avoid date-format mismatches.
    for frame in (cases, anomalies, evidence):
        frame["origin"] = frame["origin"].str.strip()
        frame["destination"] = frame["destination"].str.strip()
        frame["week_of"] = frame["week_of"].dt.normalize()

    anomaly_columns = KEYS + ["flagged"]
    evidence_columns = KEYS + ["verdict", "matched_note_id"]

    results = cases.merge(
        anomalies[anomaly_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    results = results.merge(
        evidence[evidence_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    results["actual_status"] = results.apply(
        lambda row: (
            "Missing route-week"
            if pd.isna(row["flagged"])
            else "Flagged"
            if bool(row["flagged"])
            else "Not flagged"
        ),
        axis=1,
    )

    results["actual_verdict"] = results["verdict"].replace(
        "", "No evidence decision"
    )
    results["actual_note_id"] = results["matched_note_id"].replace(
        "", "None"
    )

    results["verdict_correct"] = (
        results["expected_verdict"] == results["verdict"]
    )
    results["note_id_correct"] = (
        results["expected_note_id"] == results["matched_note_id"]
    )

    # A case passes only if the route-week exists, is flagged,
    # and both the verdict and note ID match expectations.
    results["case_passed"] = (
        results["flagged"].eq(True)
        & results["verdict_correct"]
        & results["note_id_correct"]
    )

    total = len(results)
    passed = int(results["case_passed"].sum())

    print(f"Evaluation cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Exact-match accuracy: {passed / total:.1%}" if total else
          "Exact-match accuracy: N/A")

    print("\nCase details:")
    print(
        results[
            [
                "case_id",
                "actual_status",
                "expected_verdict",
                "actual_verdict",
                "expected_note_id",
                "actual_note_id",
                "case_passed",
            ]
        ].to_string(index=False)
    )

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved detailed results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
