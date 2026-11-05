
import json
import sys
import yaml
import mlflow

from pathlib import Path
from mlflow import MlflowClient


ROOT = Path(__file__).resolve().parents[2]


def automate_lifecycle():

    print("=" * 55)
    print("FLAVORS OF CACAO - MODEL LIFECYCLE AUTOMATION")
    print("=" * 55)

    # Load configuration
    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    model_name = config["registry"]["model_name"]
    champion_alias = config["registry"]["alias"]

    # Check quality gate
    quality_path = ROOT / config["paths"]["quality_gate_report"]

    if not quality_path.exists():
        print("[ERROR] Quality gate report not found.")
        return False

    with open(quality_path, "r") as file:
        quality_report = json.load(file)

    if not quality_report["quality_gate_passed"]:
        print("[ERROR] Quality gate failed.")
        print("[INFO] Model will not be promoted.")
        return False

    print("[SUCCESS] Quality gate passed.")

    # Read latest training report
    training_path = ROOT / "artifacts" / "training_report.json"

    if not training_path.exists():
        print("[ERROR] Training report not found.")
        return False

    with open(training_path, "r") as file:
        training_report = json.load(file)

    new_run_id = training_report["mlflow_run_id"]

    # Find the newly registered model version
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    candidate = None

    for version in versions:
        if version.run_id == new_run_id:
            candidate = version
            break

    if candidate is None:
        print("[ERROR] Newly trained model version not found.")
        return False

    candidate_version = str(candidate.version)

    candidate_run = client.get_run(new_run_id)
    candidate_rmse = candidate_run.data.metrics.get("RMSE")

    if candidate_rmse is None:
        print("[ERROR] Candidate model RMSE not found.")
        return False

    print("[INFO] New model version:", candidate_version)
    print("[INFO] New model RMSE:", round(candidate_rmse, 6))

    # Check existing Champion
    old_champion_version = None
    old_champion_rmse = None

    try:
        old_champion = client.get_model_version_by_alias(
            model_name,
            champion_alias
        )

        old_champion_version = str(old_champion.version)

        old_run = client.get_run(old_champion.run_id)
        old_champion_rmse = old_run.data.metrics.get("RMSE")

        print("[INFO] Current Champion:", old_champion_version)
        print("[INFO] Current Champion RMSE:", old_champion_rmse)

    except Exception:
        print("[INFO] No existing Champion alias found.")

    # Compare models
    promoted = False

    if old_champion_version is None:
        print("[INFO] Assigning the first Champion model.")

        client.set_registered_model_alias(
            model_name,
            champion_alias,
            candidate_version
        )

        promoted = True

    elif old_champion_version == candidate_version:
        print("[INFO] This version is already the Champion.")

    elif old_champion_rmse is None or candidate_rmse < old_champion_rmse:

        print("[INFO] New model has better RMSE.")
        print("[INFO] Promoting new model...")

        client.set_registered_model_alias(
            model_name,
            champion_alias,
            candidate_version
        )

        promoted = True

    else:
        print("[INFO] Existing Champion performs better.")
        print("[INFO] Keeping the existing Champion.")

    # Read final Champion
    final_champion = client.get_model_version_by_alias(
        model_name,
        champion_alias
    )

    final_run = client.get_run(final_champion.run_id)
    final_rmse = final_run.data.metrics.get("RMSE")

    # Save lifecycle report
    report = {
        "stage": "Lab 8 - Model Lifecycle Automation",
        "model_name": model_name,
        "champion_alias": champion_alias,
        "candidate_version": candidate_version,
        "candidate_rmse": float(candidate_rmse),
        "previous_champion_version": old_champion_version,
        "previous_champion_rmse": old_champion_rmse,
        "promoted": promoted,
        "final_champion_version": str(final_champion.version),
        "final_champion_rmse": final_rmse
    }

    report_path = ROOT / "artifacts" / "lab8_model_lifecycle.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\n========== LIFECYCLE RESULTS ==========")
    print("Candidate version:", candidate_version)
    print("Promoted:", promoted)
    print("Final Champion version:", final_champion.version)
    print("Final Champion RMSE:", final_rmse)
    print("=======================================")

    print("[SUCCESS] Lifecycle report saved:", report_path)

    return True


if __name__ == "__main__":

    try:
        result = automate_lifecycle()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
