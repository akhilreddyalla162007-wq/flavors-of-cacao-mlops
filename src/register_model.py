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


# ============================================================
# MLFLOW SQLITE DATABASE
# ============================================================

MLFLOW_DB = ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)


# ============================================================
# MLFLOW EXPERIMENT
# ============================================================

mlflow.set_experiment(
    "Flavors_of_Cacao_Rating_Prediction"
)


# ============================================================
# TRAIN AND REGISTER MODEL
# ============================================================

def train_and_register_model():

    print(
        "=========================================="
    )

    print(
        "[INFO] Starting MLflow Model Registry"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    print(
        "[INFO] Loading processed dataset..."
    )

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
        run_name="RandomForest_Registry_V1"
    ) as run:

        print(
            f"[INFO] MLflow Run ID: {run.info.run_id}"
        )


        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

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

        print(
            "[INFO] Creating RandomForestRegressor..."
        )

        model = Pipeline(
            [
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
            ]
        )


        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        print(
            "[INFO] Training RandomForest model..."
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "[SUCCESS] Model training completed."
        )


        # ----------------------------------------------------
        # Make predictions
        # ----------------------------------------------------

        print(
            "[INFO] Evaluating model..."
        )

        pred = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        mse = mean_squared_error(
            y_test,
            pred
        )

        metrics = {

            "mae":
                mean_absolute_error(
                    y_test,
                    pred
                ),

            "mse":
                mse,

            "rmse":
                np.sqrt(mse),

            "r2":
                r2_score(
                    y_test,
                    pred
                )
        }


        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            metrics
        )


        print()
        print(
            "========== MODEL RESULTS =========="
        )

        print(
            f"MAE  : {metrics['mae']:.6f}"
        )

        print(
            f"MSE  : {metrics['mse']:.6f}"
        )

        print(
            f"RMSE : {metrics['rmse']:.6f}"
        )

        print(
            f"R2   : {metrics['r2']:.6f}"
        )

        print(
            "===================================="
        )


        # ----------------------------------------------------
        # Register model in MLflow
        # ----------------------------------------------------

        print(
            "[INFO] Registering model in MLflow..."
        )

        model_name = (
            "Flavors_of_Cacao_Rating_Production_Model"
        )


        # IMPORTANT:
        # MLflow/skops requires these specific types
        # to be explicitly trusted for RandomForest models.

        mlflow.sklearn.log_model(

            sk_model=model,

            name="random_forest_rating_model",

            registered_model_name=model_name,

            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )


        # ----------------------------------------------------
        # Success message
        # ----------------------------------------------------

        print()

        print(
            "=========================================="
        )

        print(
            "[SUCCESS] Model successfully registered!"
        )

        print(
            f"[SUCCESS] Model name: {model_name}"
        )

        print(
            f"[SUCCESS] RMSE: "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"[SUCCESS] R2: "
            f"{metrics['r2']:.4f}"
        )

        print(
            "=========================================="
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_and_register_model()