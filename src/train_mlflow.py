from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline

from model_utils import (
    load_data,
    make_preprocessor
)


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# SQLite MLflow database
MLFLOW_DB = ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(
    "Flavors_of_Cacao_Rating_Prediction"
)


# ============================================================
# TRAIN AND TRACK MODEL
# ============================================================

def train_and_track():

    print("[INFO] --- Starting MLflow Experiment Tracking ---")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data(ROOT)

    # --------------------------------------------------------
    # Model parameters
    # --------------------------------------------------------

    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42,
        "min_samples_split": 2
    }

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="RandomForest_Experiment"
    ):

        # Log parameters
        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "dataset",
            "flavors_of_cacao.csv"
        )

        mlflow.log_param(
            "target",
            "Rating"
        )

        mlflow.log_param(
            "task",
            "Regression"
        )

        # ----------------------------------------------------
        # Create model pipeline
        # ----------------------------------------------------

        print("[INFO] Training RandomForestRegressor...")

        model = Pipeline([
            (
                "preprocessor",
                make_preprocessor()
            ),
            (
                "model",
                RandomForestRegressor(
                    **params,
                    n_jobs=-1
                )
            )
        ])

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        print("[SUCCESS] Model training completed.")

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Evaluation metrics
        # ----------------------------------------------------

        mse = mean_squared_error(
            y_test,
            predictions
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "MSE",
            mse
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print()
        print("========== MODEL RESULTS ==========")
        print(f"MAE  : {mae:.6f}")
        print(f"MSE  : {mse:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"R2   : {r2:.6f}")
        print("===================================")

        # ----------------------------------------------------
        # FIX FOR NEW MLflow / SKOPS
        # ----------------------------------------------------

        print("[INFO] Logging model to MLflow...")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        print("[SUCCESS] Model logged successfully.")

        # ----------------------------------------------------
        # Run information
        # ----------------------------------------------------

        run = mlflow.active_run()

        if run is not None:
            print()
            print("=========================================")
            print("[SUCCESS] MLflow Experiment Completed")
            print(f"[INFO] Run ID: {run.info.run_id}")
            print(f"[INFO] Experiment: Flavors_of_Cacao_Rating_Prediction")
            print("=========================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_and_track()