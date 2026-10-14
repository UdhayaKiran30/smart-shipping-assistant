
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "weekly_route_costs.csv"
OUTPUT_PATH = ROOT / "outputs" / "threshold_sensitivity.csv"

THRESHOLDS = [5.0, 10.0, 15.0]


def calculate_baselines(df):
    df = df.sort_values(
        ["origin", "destination", "route_type", "week_of"]
    ).copy()

    route_keys = ["origin", "destination", "route_type"]
    grouped = df.groupby(route_keys, sort=False)

    df["history_baseline"] = grouped[
        "cost_per_tonne_km"
    ].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean()
    )

    df["history_change_pct"] = (
        (df["cost_per_tonne_km"] / df["history_baseline"]) - 1
    ) * 100

    peer_keys = ["route_type", "week_of"]
    df["peer_baseline"] = df.groupby(peer_keys)[
        "cost_per_tonne_km"
    ].transform(
        lambda s: (s.sum() - s) / (len(s) - 1)
        if len(s) > 1 else float("nan")
    )

    df["peer_change_pct"] = (
        (df["cost_per_tonne_km"] / df["peer_baseline"]) - 1
    ) * 100

    return df


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["week_of"])
    df = calculate_baselines(df)

    results = []

    for threshold in THRESHOLDS:
        history_flag = df["history_change_pct"] >= threshold
        peer_flag = df["peer_change_pct"] >= threshold
        flagged = history_flag | peer_flag

        results.append({
            "threshold_pct": threshold,
            "total_route_weeks": len(df),
            "flagged_route_weeks": int(flagged.sum()),
            "unflagged_route_weeks": int((~flagged).sum()),
            "history_flags": int(history_flag.sum()),
            "peer_flags": int(peer_flag.sum()),
            "both_flags": int((history_flag & peer_flag).sum()),
        })

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
