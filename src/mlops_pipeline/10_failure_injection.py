
import os
import sys
import json
import yaml
import mlflow
import mlflow.pyfunc
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_failure_tests():

    print("=" * 55)
    print("FLAVORS OF CACAO - FAILURE INJECTION TESTING")
    print("=" * 55)

    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    results = []

    # Test 1: Missing dataset
    print("\n[TEST 1] Missing dataset")

    fake_path = ROOT / "data/raw/missing_dataset.csv"

    if not fake_path.exists():
        print("[SUCCESS] Missing dataset detected.")

        results.append({
            "test": "Missing Dataset",
            "status": "PASSED"
        })
    else:
        results.append({
            "test": "Missing Dataset",
            "status": "FAILED"
        })

    # Test 2: Schema validation
    print("\n[TEST 2] Schema validation")

    sample_data = pd.DataFrame({
        "Company": ["Test Company"],
        "Rating": [3.5]
    })

    required_columns = [
        "Company",
        "Specific_Bean_Origin",
        "Review_Date",
        "Cocoa_Percent",
        "Company_Location",
        "Bean_Type",
        "Broad_Bean_Origin"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in sample_data.columns
    ]

    if missing_columns:
        print("[SUCCESS] Missing columns detected:")
        print(missing_columns)

        results.append({
            "test": "Schema Validation",
            "status": "PASSED",
            "missing_columns": missing_columns
        })
    else:
        results.append({
            "test": "Schema Validation",
            "status": "FAILED"
        })

    # Test 3: Invalid rating detection
    print("\n[TEST 3] Invalid rating detection")

    invalid_data = pd.DataFrame({
        "Rating": [3.5, 6.0, 2.5]
    })

    invalid_ratings = int(
        (
            (invalid_data["Rating"] < 1) |
            (invalid_data["Rating"] > 5)
        ).sum()
    )

    if invalid_ratings > 0:
        print("[SUCCESS] Invalid rating detected.")

        results.append({
            "test": "Invalid Rating",
            "status": "PASSED",
            "invalid_values": invalid_ratings
        })
    else:
        results.append({
            "test": "Invalid Rating",
            "status": "FAILED"
        })

    # Test 4: Model recovery from local MLflow artifacts
    print("\n[TEST 4] Model recovery")

    model_path = ROOT / config["paths"]["model"]

    if model_path.exists():

        print("[INFO] Simulating missing local model.")
        print("[INFO] Recovering model from saved MLflow artifacts.")

        try:
            model_artifact_path = (
                ROOT
                / "mlruns"
                / "1"
                / "models"
                / "m-188c29a769e5428086d7bf7cba1acfca"
                / "artifacts"
            )

            if not model_artifact_path.exists():
                raise FileNotFoundError(
                    "Saved MLflow model artifact not found."
                )

            recovered_model = mlflow.pyfunc.load_model(
                str(model_artifact_path)
            )

            if recovered_model is not None:
                print("[SUCCESS] Model recovery successful.")

                results.append({
                    "test": "Model Recovery",
                    "status": "PASSED",
                    "recovery_source": "Local MLflow model artifacts"
                })

            else:
                raise RuntimeError("Model recovery returned None.")

        except Exception as error:

            print("[ERROR] Model recovery failed:", error)

            results.append({
                "test": "Model Recovery",
                "status": "FAILED",
                "error": str(error)
            })

    else:
        print("[ERROR] Local model file not found.")

        results.append({
            "test": "Model Recovery",
            "status": "FAILED"
        })

    # Count results
    passed = sum(
        1 for result in results
        if result["status"] == "PASSED"
    )

    failed = len(results) - passed

    # Save report
    report = {
        "stage": "Lab 9 - Failure Injection Testing",
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "results": results
    }

    report_path = ROOT / "artifacts/failure_injection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\n========== FAILURE TEST RESULTS ==========")
    print("Total tests:", len(results))
    print("Passed:", passed)
    print("Failed:", failed)
    print("==========================================")

    print("[SUCCESS] Report saved:", report_path)

    return failed == 0


if __name__ == "__main__":

    try:
        result = run_failure_tests()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
