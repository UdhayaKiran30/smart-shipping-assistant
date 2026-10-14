
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "outputs" / "weekly_route_costs.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "route_anomalies.csv"

# Initial thresholds for the prototype.
# These are engineering choices, not requirements from the brief.
HISTORY_INCREASE_THRESHOLD_PCT = 10.0
PEER_INCREASE_THRESHOLD_PCT = 10.0

ROUTE_COLUMNS = ["origin", "destination", "route_type"]


def detect_anomalies():
    df = pd.read_csv(INPUT_PATH, parse_dates=["week_of"])

    df = df.sort_values(ROUTE_COLUMNS + ["week_of"]).copy()

    route_group = df.groupby(ROUTE_COLUMNS, sort=False)[
        "cost_per_tonne_km"
    ]

    # Previous 8 observed route-weeks, excluding the current week.
    df["history_baseline"] = route_group.transform(
        lambda s: s.shift(1).rolling(window=8, min_periods=1).mean()
    )

    df["history_weeks_available"] = route_group.transform(
        lambda s: s.shift(1).notna().rolling(
            window=8, min_periods=1
        ).sum()
    ).fillna(0).astype(int)

    # Mean cost of all other routes of the same type in the same week.
    peer_group = df.groupby(["route_type", "week_of"])[
        "cost_per_tonne_km"
    ]

    df["peer_total"] = peer_group.transform("sum")
    df["peer_route_count"] = peer_group.transform("count")

    df["peer_baseline"] = (
        (df["peer_total"] - df["cost_per_tonne_km"])
        / (df["peer_route_count"] - 1)
    ).where(df["peer_route_count"] > 1)

    # Percentage change against each baseline.
    df["history_change_pct"] = (
        (df["cost_per_tonne_km"] / df["history_baseline"] - 1) * 100
    ).where(df["history_baseline"] > 0)

    df["peer_change_pct"] = (
        (df["cost_per_tonne_km"] / df["peer_baseline"] - 1) * 100
    ).where(df["peer_baseline"] > 0)

    # Initial rule: flag a rising cost that exceeds either baseline
    # by the configured percentage threshold.
    history_alert = (
        (df["history_change_pct"] >= HISTORY_INCREASE_THRESHOLD_PCT)
        & df["history_change_pct"].notna()
    )

    peer_alert = (
        (df["peer_change_pct"] >= PEER_INCREASE_THRESHOLD_PCT)
        & df["peer_change_pct"].notna()
    )

    df["flagged"] = history_alert | peer_alert

    df["flag_reason"] = "No unusual increase detected"

    df.loc[
        history_alert & peer_alert, "flag_reason"
    ] = "Above historical and peer baselines"

    df.loc[
        history_alert & ~peer_alert, "flag_reason"
    ] = "Above historical baseline"

    df.loc[
        ~history_alert & peer_alert, "flag_reason"
    ] = "Above peer baseline"

    # Remove intermediate calculation columns.
    df = df.drop(columns=["peer_total"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Anomaly detection completed!")
    print(f"Total route-weeks: {len(df)}")
    print(f"Flagged route-weeks: {int(df['flagged'].sum())}")
    print(f"Output saved to: {OUTPUT_PATH}")

    print("\nSample flagged records:")
    print(
        df[df["flagged"]][
            ROUTE_COLUMNS
            + [
                "week_of",
                "cost_per_tonne_km",
                "history_baseline",
                "history_change_pct",
                "peer_baseline",
                "peer_change_pct",
                "flag_reason",
            ]
        ].head(10).to_string(index=False)
    )

    return df


if __name__ == "__main__":
    detect_anomalies()
