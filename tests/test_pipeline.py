
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


# 1. Basic calculation tests

def test_weekly_cost_per_tonne_km():
    total_cost = 10000
    total_tonne_km = 5000

    result = total_cost / total_tonne_km

    assert result == 2.0


def test_week_starts_on_monday():
    dates = pd.to_datetime([
        "2025-02-24",
        "2025-02-26",
        "2025-03-02",
    ])

    week_starts = dates - pd.to_timedelta(dates.weekday, unit="D")

    assert week_starts.strftime("%Y-%m-%d").tolist() == [
        "2025-02-24",
        "2025-02-24",
        "2025-02-24",
    ]


def test_historical_baseline_uses_only_prior_weeks():
    costs = pd.Series([2.0, 4.0, 6.0])

    baseline = costs.shift(1).rolling(
        window=8, min_periods=1
    ).mean()

    assert pd.isna(baseline.iloc[0])
    assert baseline.iloc[1] == 2.0
    assert baseline.iloc[2] == 3.0


def test_peer_baseline_excludes_current_route():
    route_costs = pd.Series([2.0, 3.0, 4.0])

    peer_baseline = (
        route_costs.sum() - route_costs.iloc[1]
    ) / (len(route_costs) - 1)

    assert peer_baseline == 3.0


def test_threshold_flags_cost_increase():
    threshold = 10.0

    assert 12.0 >= threshold
    assert 10.0 >= threshold
    assert 9.9 < threshold


def test_final_output_schema_and_unique_route_weeks():
    output_path = ROOT / "outputs" / "final_output.csv"
    output = pd.read_csv(output_path)

    expected_columns = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
        "reason",
    ]

    assert output.columns.tolist() == expected_columns
    assert not output.duplicated(["route", "week_of"]).any()
    assert output["reason"].notna().all()
    assert set(output["flagged"].unique()).issubset({"Yes", "No"})


# 2. Evidence retrieval tests

def load_evidence():
    return pd.read_csv(ROOT / "outputs" / "route_evidence.csv")


def get_evidence_row(evidence, origin, destination, week_of):
    rows = evidence[
        (evidence["origin"] == origin)
        & (evidence["destination"] == destination)
        & (evidence["week_of"] == week_of)
    ]

    assert len(rows) == 1, (
        f"Expected exactly one record for "
        f"{origin}-{destination} in {week_of}"
    )

    return rows.iloc[0]


def test_flood_note_justifies_cost_increase_during_affected_week():
    evidence = load_evidence()

    row = get_evidence_row(
        evidence, "Chennai", "Bangalore", "2025-02-24"
    )

    assert row["verdict"] == "Justified"
    assert row["matched_note_id"] == "N001"


def test_flood_note_covers_the_following_week():
    evidence = load_evidence()

    row = get_evidence_row(
        evidence, "Chennai", "Bangalore", "2025-03-03"
    )

    assert row["verdict"] == "Justified"
    assert row["matched_note_id"] == "N001"


def test_return_to_normal_note_does_not_justify_later_increase():
    evidence = load_evidence()

    row = get_evidence_row(
        evidence, "Chennai", "Bangalore", "2025-03-10"
    )

    assert row["verdict"] == "Unexplained"
    assert pd.isna(row["matched_note_id"])


def test_festival_surcharge_note_matches_correct_route():
    evidence = load_evidence()

    row = get_evidence_row(
        evidence, "Ahmedabad", "Mumbai", "2025-01-20"
    )

    assert row["verdict"] == "Justified"
    assert row["matched_note_id"] == "N002"


def test_all_flagged_route_weeks_have_evidence_decisions():
    anomalies = pd.read_csv(
        ROOT / "outputs" / "route_anomalies.csv"
    )
    evidence = load_evidence()

    flagged = anomalies[anomalies["flagged"] == True]

    keys = ["origin", "destination", "route_type", "week_of"]

    flagged_keys = set(
        map(tuple, flagged[keys].astype(str).values)
    )
    evidence_keys = set(
        map(tuple, evidence[keys].astype(str).values)
    )

    assert flagged_keys == evidence_keys


def test_evidence_verdicts_are_valid():
    evidence = load_evidence()

    assert set(evidence["verdict"].dropna().unique()).issubset(
        {"Justified", "Unexplained"}
    )

    justified = evidence[evidence["verdict"] == "Justified"]
    assert justified["matched_note_id"].notna().all()

    unexplained = evidence[evidence["verdict"] == "Unexplained"]
    assert unexplained["matched_note_id"].isna().all()
