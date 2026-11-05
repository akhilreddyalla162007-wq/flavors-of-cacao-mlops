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
# MODEL
# ============================================================

MODEL_NAME = (
    "Flavors_of_Cacao_Rating_Production_Model"
)

CHAMPION_ALIAS = "Champion"


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report():

    print("==========================================")
    print("[INFO] Generating Model Registry Report")
    print("==========================================")

    # --------------------------------------------------------
    # Get Champion
    # --------------------------------------------------------

    try:

        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            CHAMPION_ALIAS
        )

    except Exception as e:

        print(
            "[ERROR] Champion model not found."
        )

        print(
            f"[ERROR] {e}"
        )

        return

    # --------------------------------------------------------
    # Get run information
    # --------------------------------------------------------

    run = client.get_run(
        champion.run_id
    )

    metrics = run.data.metrics
    params = run.data.params

    # --------------------------------------------------------
    # Deployment artifact
    # --------------------------------------------------------

    deployment_artifact = {

        "model_name": MODEL_NAME,

        "model_version": champion.version,

        "model_alias": CHAMPION_ALIAS,

        "run_id": champion.run_id,

        "model_uri": (
            f"models:/{MODEL_NAME}@"
            f"{CHAMPION_ALIAS}"
        ),

        "task": "Regression",

        "dataset": "flavors_of_cacao.csv",

        "target": "Rating",

        "metrics": {
            "MAE": metrics.get("MAE"),
            "MSE": metrics.get("MSE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("R2")
        },

        "parameters": params,

        "status": "READY_FOR_DEPLOYMENT"
    }

    # --------------------------------------------------------
    # Create artifacts directory
    # --------------------------------------------------------

    artifacts_dir = (
        ROOT / "artifacts"
    )

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    report_path = (
        artifacts_dir /
        "deployment_artifact.json"
    )

    with open(
        report_path,
        "w"
    ) as file:

        json.dump(
            deployment_artifact,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("========== DEPLOYMENT ARTIFACT ==========")

    print(
        f"Model Name : {MODEL_NAME}"
    )

    print(
        f"Version    : {champion.version}"
    )

    print(
        f"Alias      : {CHAMPION_ALIAS}"
    )

    print(
        f"RMSE       : {metrics.get('RMSE')}"
    )

    print(
        f"R2         : {metrics.get('R2')}"
    )

    print(
        f"Status     : READY_FOR_DEPLOYMENT"
    )

    print("=========================================")

    print()
    print(
        "[SUCCESS] Deployment artifact generated:"
    )

    print(
        f"{report_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_report()
    