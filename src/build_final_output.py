
from pathlib import Path

import pandas as pd


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"

ANOMALIES_FILE = OUTPUT_DIR / "route_anomalies.csv"
EVIDENCE_FILE = OUTPUT_DIR / "route_evidence.csv"
FINAL_FILE = OUTPUT_DIR / "final_output.csv"

# Exact required column order
FINAL_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

JOIN_KEYS = ["origin", "destination", "route_type", "week_of"]


def format_percentage(value, description):
    """Format a percentage change for the final report."""
    if pd.isna(value):
        return f"Not available ({description})"

    return f"{value:+.1f}% " + description


def main():
    anomalies = pd.read_csv(ANOMALIES_FILE)
    evidence = pd.read_csv(EVIDENCE_FILE)

    # Validate the intermediate files.
    required_anomaly_columns = JOIN_KEYS + [
        "cost_per_tonne_km",
        "history_change_pct",
        "peer_change_pct",
        "flagged",
    ]
    required_evidence_columns = JOIN_KEYS + [
        "verdict",
        "matched_note_id",
        "reason",
    ]

    missing_anomaly_columns = set(required_anomaly_columns) - set(
        anomalies.columns
    )
    missing_evidence_columns = set(required_evidence_columns) - set(
        evidence.columns
    )

    if missing_anomaly_columns:
        raise ValueError(
            f"Missing anomaly columns: {sorted(missing_anomaly_columns)}"
        )

    if missing_evidence_columns:
        raise ValueError(
            f"Missing evidence columns: {sorted(missing_evidence_columns)}"
        )

    # Each route-week must appear only once in each intermediate file.
    if anomalies.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate route-week records found in anomalies.")

    if evidence.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate route-week records found in evidence.")

    # Only flagged route-weeks should have evidence records.
    flagged_keys = anomalies.loc[anomalies["flagged"], JOIN_KEYS]
    evidence_check = evidence.merge(
        flagged_keys,
        on=JOIN_KEYS,
        how="left",
        indicator=True,
    )

    if (evidence_check["_merge"] != "both").any():
        raise ValueError(
            "Evidence contains a route-week that is not flagged."
        )

    # Join evidence onto all weekly anomaly records.
    merged = anomalies.merge(
        evidence[JOIN_KEYS + ["verdict", "matched_note_id", "reason"]],
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )

    merged["route"] = (
        merged["origin"].astype(str)
        + "-"
        + merged["destination"].astype(str)
    )

    merged["cost_per_tonne_km"] = (
        merged["cost_per_tonne_km"].round(2)
    )

    merged["vs_own_history"] = merged["history_change_pct"].apply(
        lambda value: format_percentage(
            value, "vs this route's past average"
        )
    )

    merged["vs_similar_routes"] = merged["peer_change_pct"].apply(
        lambda value: format_percentage(
            value, "vs similar-length routes this week"
        )
    )

    merged["flagged"] = merged["flagged"].map(
        {True: "Yes", False: "No", 1: "Yes", 0: "No"}
    )

    # Use evidence-based explanations for flagged records.
    # Unflagged records do not need a supporting note.
    def build_reason(row):
        if row["flagged"] == "No":
            return "No unusual cost increase detected."

        if pd.isna(row["verdict"]):
            raise ValueError(
                f"Missing evidence verdict for {row['route']} "
                f"week {row['week_of']}"
            )

        return row["reason"]

    merged["reason"] = merged.apply(build_reason, axis=1)

    # Keep the note ID blank when no note supports the increase.
    merged["matched_note_id"] = merged["matched_note_id"].fillna("")

    # Preserve the exact required column names and order.
    final_output = merged[FINAL_COLUMNS].copy()
    final_output = final_output.sort_values(
        ["week_of", "route"]
    ).reset_index(drop=True)

    # Final validation.
    if final_output.columns.tolist() != FINAL_COLUMNS:
        raise ValueError("Final output columns do not match the template.")

    if final_output.duplicated(["route", "week_of"]).any():
        raise ValueError("Duplicate route-week records in final output.")

    if final_output.isnull().any().any():
        raise ValueError("Unexpected missing values in final output.")

    final_output.to_csv(FINAL_FILE, index=False, encoding="utf-8")

    print(f"Final records: {len(final_output)}")
    print(f"Flagged records: {(final_output['flagged'] == 'Yes').sum()}")
    print(f"Unflagged records: {(final_output['flagged'] == 'No').sum()}")
    print(f"Saved to: {FINAL_FILE}")
    print("\nPreview:")
    print(final_output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
