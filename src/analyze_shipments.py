
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "shipment_records.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Columns required for analysis
REQUIRED_COLUMNS = [
    "origin",
    "destination",
    "route_type",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
]


def analyze_shipments():
    # 1. Load shipment data
    df = pd.read_csv(DATA_PATH)

    # 2. Validate required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 3. Convert shipment dates to datetime
    df["shipment_date"] = pd.to_datetime(
        df["shipment_date"], errors="raise"
    )

    # 4. Validate quantities and distances
    if (df["quantity_tonnes"] <= 0).any():
        raise ValueError("Quantity must be greater than zero.")

    if (df["distance_km"] <= 0).any():
        raise ValueError("Distance must be greater than zero.")

    # 5. Calculate tonne-kilometres for each shipment
    df["tonne_km"] = (
        df["quantity_tonnes"] * df["distance_km"]
    )

    # 6. Find the Monday of each shipment's week
    df["week_of"] = (
        df["shipment_date"]
        - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    ).dt.strftime("%Y-%m-%d")

    # 7. Group by route, route type, and week
    group_columns = [
        "origin",
        "destination",
        "route_type",
        "week_of",
    ]

    weekly = (
        df.groupby(group_columns, as_index=False)
        .agg(
            total_freight_cost_inr=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_date", "size"),
        )
    )

    # 8. Calculate the weighted weekly cost per tonne-km
    weekly["cost_per_tonne_km"] = (
        weekly["total_freight_cost_inr"]
        / weekly["total_tonne_km"]
    )

    # 9. Sort results for easy inspection
    weekly = weekly.sort_values(
        ["origin", "destination", "route_type", "week_of"]
    ).reset_index(drop=True)

    # 10. Save weekly results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "weekly_route_costs.csv"
    weekly.to_csv(output_path, index=False)

    # 11. Display summary
    print("Weekly shipment analysis completed!")
    print(f"Input shipment records: {len(df)}")
    print(f"Weekly route records: {len(weekly)}")
    print(f"Output saved to: {output_path}")

    print("\nFirst 10 weekly records:")
    print(weekly.head(10).to_string(index=False))

    return weekly


if __name__ == "__main__":
    analyze_shipments()
