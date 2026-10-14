
from pathlib import Path
import json
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

PIPELINE_STEPS = [
    "src/analyze_shipments.py",
    "src/detect_anomalies.py",
    "src/retrieve_evidence.py",
    "src/build_final_output.py",
]


def main():
    start = time.perf_counter()

    for script in PIPELINE_STEPS:
        print(f"Running {script}...")
        subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            check=True,
        )

    elapsed_seconds = time.perf_counter() - start

    shipments = pd.read_csv(ROOT / "data" / "shipment_records.csv")
    weekly = pd.read_csv(OUTPUT_DIR / "weekly_route_costs.csv")
    anomalies = pd.read_csv(OUTPUT_DIR / "route_anomalies.csv")
    evidence = pd.read_csv(OUTPUT_DIR / "route_evidence.csv")
    final_output = pd.read_csv(OUTPUT_DIR / "final_output.csv")

    flagged = final_output[final_output["flagged"] == "Yes"]

    metrics = {
        "shipment_records_processed": int(len(shipments)),
        "weekly_route_records": int(len(weekly)),
        "final_output_records": int(len(final_output)),
        "flagged_route_weeks": int(len(flagged)),
        "unflagged_route_weeks": int(
            (final_output["flagged"] == "No").sum()
        ),
        "justified_flags": int(
            (evidence["verdict"] == "Justified").sum()
        ),
        "unexplained_flags": int(
            (evidence["verdict"] == "Unexplained").sum()
        ),
        "execution_seconds": round(elapsed_seconds, 3),
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "api_cost_usd": 0.0,
        "llm_note": (
            "Current pipeline uses deterministic Python rules; "
            "no LLM API calls or token usage."
        ),
    }

    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n=== Evaluation Metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
