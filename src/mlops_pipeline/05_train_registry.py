
import os
import sys
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parents[2]


def train_and_register():

    print("=" * 55)
    print("FLAVORS OF CACAO - MODEL TRAINING AND REGISTRY")
    print("=" * 55)

    # Load configuration
    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load processed datasets
    train_path = ROOT / "data/processed/train.csv"
    test_path = ROOT / "data/processed/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Rating"])
    y_train = train_df["Rating"]

    X_test = test_df.drop(columns=["Rating"])
    y_test = test_df["Rating"]

    print("[INFO] Training data:", X_train.shape)
    print("[INFO] Testing data:", X_test.shape)

    # MLflow configuration
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    model_name = config["registry"]["model_name"]

    # Model parameters
    params = {
        "n_estimators": config["model"]["n_estimators"],
        "max_depth": config["model"]["max_depth"],
        "min_samples_split": config["model"]["min_samples_split"],
        "random_state": config["model"]["random_state"],
        "n_jobs": -1
    }

    # Create model
    model = RandomForestRegressor(**params)

    # Start MLflow run
    with mlflow.start_run(
        run_name="Lab8_Flavors_Regression_Registry"
    ) as run:

        print("[INFO] MLflow Run ID:", run.info.run_id)

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestRegressor")
        mlflow.log_param("dataset", "flavors_of_cacao.csv")
        mlflow.log_param("target", "Rating")
        mlflow.log_param("task", "Regression")

        # Train model
        print("[INFO] Training Random Forest model...")

        model.fit(X_train, y_train)

        print("[SUCCESS] Model training completed.")

        # Predictions
        predictions = model.predict(X_test)

        # Calculate regression metrics
        mse = mean_squared_error(y_test, predictions)

        metrics = {
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, predictions)
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        print("\n========== MODEL RESULTS ==========")
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")
        print("===================================")

        # Save model locally
        model_path = ROOT / "models/best_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)

        print("[SUCCESS] Model saved:", model_path)

        # Save training report
        report = {
            "stage": "Lab 8 - Model Training",
            "model_name": model_name,
            "model_family": "RandomForestRegressor",
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
            "features": int(X_train.shape[1]),
            "parameters": params,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
            },
            "mlflow_run_id": run.info.run_id
        }

        report_path = ROOT / "artifacts/training_report.json"

        with open(report_path, "w") as file:
            json.dump(report, file, indent=4)

        mlflow.log_artifact(str(report_path))

        # Register model in MLflow
        print("\n[INFO] Registering model in MLflow...")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_rating_model",
            registered_model_name=model_name,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\n[SUCCESS] Model registered successfully.")
        print("[INFO] Model name:", model_name)
        print("[INFO] MLflow experiment:", experiment_name)
        print("[INFO] MLflow Run ID:", run.info.run_id)
        print("[SUCCESS] Training report saved.")
        
    return True


if __name__ == "__main__":

    try:
        train_and_register()

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)

