
import os
import sys
import json
import subprocess

from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "01_load_data.py",
    "02_validate_data.py",
    "03_preprocessing.py",
    "04_validate_outputs.py",
    "05_train_registry.py",
    "06_model_evaluation.py",
    "07_quality_gate.py",
    "08_automate_lifecycle.py",
    "09_inference.py",
    "10_failure_injection.py"
]


def run_pipeline():

    print("=" * 60)
    print("FLAVORS OF CACAO - COMPLETE MLOPS PIPELINE")
    print("=" * 60)

    start_time = datetime.now()

    results = []

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for script in SCRIPTS:

        script_path = ROOT / "src" / "mlops_pipeline" / script

        print("\n" + "=" * 60)
        print("Running:", script)
        print("=" * 60)

        if not script_path.exists():
            print("[ERROR] Script not found:", script_path)
            results.append({
                "script": script,
                "status": "NOT FOUND"
            })
            break

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )

        # Display script output
        print(result.stdout)

        if result.stderr:
            print("[STDERR]")
            print(result.stderr)

        if result.returncode == 0:

            print("[SUCCESS]", script, "completed.")

            results.append({
                "script": script,
                "status": "PASSED"
            })

        else:

            print("[FAILED]", script, "stopped the pipeline.")

            results.append({
                "script": script,
                "status": "FAILED",
                "error": result.stderr[-2000:]
            })

            break

    end_time = datetime.now()

    passed = sum(
        1 for item in results
        if item["status"] == "PASSED"
    )

    failed = len(results) - passed

    # Create final pipeline report
    report = {
        "project": "Flavors of Cacao Rating Prediction",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_stages": len(SCRIPTS),
        "executed_stages": len(results),
        "passed_stages": passed,
        "failed_stages": failed,
        "pipeline_status": (
            "SUCCESS"
            if passed == len(SCRIPTS)
            else "FAILED"
        ),
        "results": results
    }

    report_path = ROOT / "artifacts" / "complete_pipeline_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    # Save execution log
    log_path = log_dir / "complete_pipeline.log"

    with open(log_path, "a") as file:
        file.write("\n" + "=" * 60 + "\n")
        file.write("Pipeline execution: " + start_time.isoformat() + "\n")
        file.write(json.dumps(report, indent=4))
        file.write("\n")

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE SUMMARY")
    print("=" * 60)

    print("Total stages:", len(SCRIPTS))
    print("Executed stages:", len(results))
    print("Passed:", passed)
    print("Failed:", failed)

    if passed == len(SCRIPTS):
        print("[SUCCESS] COMPLETE PIPELINE EXECUTED SUCCESSFULLY.")
    else:
        print("[ERROR] PIPELINE EXECUTION FAILED.")

    print("Pipeline report:", report_path)
    print("Execution log:", log_path)

    return passed == len(SCRIPTS)


if __name__ == "__main__":

    try:
        success = run_pipeline()
        sys.exit(0 if success else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
