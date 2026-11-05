
import os
import sys
import json
import yaml

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_quality():

    print("=" * 55)
    print("FLAVORS OF CACAO - MODEL QUALITY GATE")
    print("=" * 55)

    # Load configuration
    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Read evaluation report
    report_path = ROOT / config["paths"]["evaluation_report"]

    if not report_path.exists():
        print("[ERROR] Evaluation report not found.")
        return False

    with open(report_path, "r") as file:
        evaluation = json.load(file)

    # Get quality gate settings
    metric_name = config["quality_gate"]["metric"]
    maximum_value = config["quality_gate"]["maximum_value"]

    actual_value = evaluation["metrics"][metric_name]

    # Check model quality
    passed = actual_value <= maximum_value

    # Create quality report
    quality_report = {
        "stage": "Lab 8 - Model Quality Gate",
        "metric": metric_name,
        "actual_value": actual_value,
        "maximum_allowed_value": maximum_value,
        "quality_gate_passed": passed
    }

    # Save report
    output_path = ROOT / config["paths"]["quality_gate_report"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(quality_report, file, indent=4)

    # Display results
    print("Metric:", metric_name)
    print("Actual value:", round(actual_value, 6))
    print("Maximum allowed:", maximum_value)

    if passed:
        print("[SUCCESS] Model passed the quality gate.")
    else:
        print("[ERROR] Model failed the quality gate.")

    print("Report saved:", output_path)

    return passed


if __name__ == "__main__":

    try:
        result = check_quality()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
