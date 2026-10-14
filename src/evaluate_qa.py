
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
FINAL_OUTPUT = OUTPUT_DIR / "final_output.csv"
EVIDENCE_OUTPUT = OUTPUT_DIR / "route_evidence.csv"
REPORT_PATH = OUTPUT_DIR / "qa_evaluation.csv"

QUESTIONS = [
    {
        "id": "QA001",
        "question": "Why did Chennai-Bangalore shipping costs rise in February 2025?",
        "expected_route": "Chennai-Bangalore",
        "expected_period": "2025-02",
    },
    {
        "id": "QA002",
        "question": "Explain the Chennai-Bangalore shipping costs in March 2025.",
        "expected_route": "Chennai-Bangalore",
        "expected_period": "2025-03",
    },
    {
        "id": "QA003",
        "question": "Why did Ahmedabad-Mumbai shipping costs rise in January 2025?",
        "expected_route": "Ahmedabad-Mumbai",
        "expected_period": "2025-01",
    },
]


def main():
    for path in (FINAL_OUTPUT, EVIDENCE_OUTPUT):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    results = []

    for case in QUESTIONS:
        command = [
            sys.executable,
            str(ROOT / "src" / "answer_questions.py"),
            case["question"],
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            passed = completed.returncode == 0

            # Basic safety checks on the answer's wording.
            lower_output = output.lower()
            has_error = any(
                phrase in lower_output
                for phrase in (
                    "traceback (most recent call last)",
                    "api key not found",
                    "invalid api key",
                )
            )

            results.append({
                "test_id": case["id"],
                "question": case["question"],
                "expected_route": case["expected_route"],
                "expected_period": case["expected_period"],
                "process_success": passed and not has_error,
                "answer_contains_route": (
                    case["expected_route"].lower() in lower_output
                ),
                "error": (
                    output[-1000:] if not passed or has_error else ""
                ),
                "note": (
                    "Basic smoke test only; manually verify factual grounding."
                ),
            })

            print(
                f'{case["id"]}: '
                f'{"PASS" if passed and not has_error else "FAIL"}'
            )

        except subprocess.TimeoutExpired:
            results.append({
                "test_id": case["id"],
                "question": case["question"],
                "expected_route": case["expected_route"],
                "expected_period": case["expected_period"],
                "process_success": False,
                "answer_contains_route": False,
                "error": "Timed out after 120 seconds",
                "note": "Basic smoke test only; manually verify factual grounding.",
            })
            print(f'{case["id"]}: FAIL (timeout)')

    report = pd.DataFrame(results)
    report.to_csv(REPORT_PATH, index=False)

    passed_count = int(report["process_success"].sum())
    total = len(report)

    print(f"\nQ&A smoke tests: {passed_count}/{total}")
    print(f"Report saved to: {REPORT_PATH}")

    if passed_count != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
