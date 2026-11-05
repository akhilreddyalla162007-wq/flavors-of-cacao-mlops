from pathlib import Path
import json

import mlflow
from mlflow import MlflowClient


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

client = MlflowClient()


# ============================================================
# MODEL NAME
# ============================================================

MODEL_NAME = (
    "Flavors_of_Cacao_Rating_Production_Model"
)

CHAMPION_ALIAS = "Champion"


# ============================================================
# LIFECYCLE AUTOMATION
# ============================================================

def automate_lifecycle():

    print("==========================================")
    print("[INFO] Automating Model Lifecycle")
    print("==========================================")

    print(
        f"[INFO] Model: {MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Get all registered versions
    # --------------------------------------------------------

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:

        print(
            "[ERROR] No registered model versions found."
        )

        return

    print(
        f"[INFO] Found {len(versions)} model version(s)."
    )

    # --------------------------------------------------------
    # Compare versions using RMSE
    # --------------------------------------------------------

    candidates = []

    for version in versions:

        run_id = version.run_id

        try:

            run = client.get_run(run_id)

            metrics = run.data.metrics

            rmse = metrics.get("RMSE")

            r2 = metrics.get("R2")

            if rmse is None:
                print(
                    f"[WARNING] Version {version.version} "
                    "does not contain RMSE."
                )
                continue

            candidates.append({
                "version": int(version.version),
                "run_id": run_id,
                "rmse": float(rmse),
                "r2": float(r2) if r2 is not None else None
            })

        except Exception as e:

            print(
                f"[WARNING] Could not read "
                f"version {version.version}: {e}"
            )

    if not candidates:

        print(
            "[ERROR] No valid model versions "
            "with RMSE found."
        )

        return

    # --------------------------------------------------------
    # Select lowest RMSE
    # --------------------------------------------------------

    champion = min(
        candidates,
        key=lambda x: x["rmse"]
    )

    champion_version = champion["version"]

    print()
    print("========== MODEL VERSIONS ==========")

    for candidate in sorted(
        candidates,
        key=lambda x: x["version"]
    ):

        print(
            f"Version {candidate['version']} | "
            f"RMSE: {candidate['rmse']:.6f} | "
            f"R2: {candidate['r2']}"
        )

    print("====================================")

    # --------------------------------------------------------
    # Assign Champion alias
    # --------------------------------------------------------

    print()
    print(
        f"[INFO] Promoting Version "
        f"{champion_version} to Champion..."
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
        str(champion_version)
    )

    print(
        "[SUCCESS] Champion alias assigned."
    )

    # --------------------------------------------------------
    # Save lifecycle information
    # --------------------------------------------------------

    lifecycle_dir = (
        ROOT / "artifacts"
    )

    lifecycle_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    lifecycle_report = {
        "model_name": MODEL_NAME,
        "champion_alias": CHAMPION_ALIAS,
        "champion_version": champion_version,
        "champion_run_id": champion["run_id"],
        "champion_rmse": champion["rmse"],
        "champion_r2": champion["r2"],
        "all_versions": candidates
    }

    report_path = (
        lifecycle_dir /
        "model_lifecycle.json"
    )

    with open(
        report_path,
        "w"
    ) as file:

        json.dump(
            lifecycle_report,
            file,
            indent=4
        )

    print(
        f"[SUCCESS] Lifecycle report saved to:"
    )

    print(
        f"         {report_path}"
    )

    print()
    print("==========================================")
    print("[SUCCESS] Model Lifecycle Completed")
    print(
        f"[SUCCESS] Champion Version: "
        f"{champion_version}"
    )
    print(
        f"[SUCCESS] Champion RMSE: "
        f"{champion['rmse']:.6f}"
    )
    print("==========================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    automate_lifecycle()