
import os
import sys
import json
import joblib
import yaml
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


ROOT = Path(__file__).resolve().parents[2]


def evaluate_model():

    print("=" * 55)
    print("FLAVORS OF CACAO - MODEL EVALUATION")
    print("=" * 55)

    # Load configuration
    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load model
    model_path = ROOT / config["paths"]["model"]

    if not model_path.exists():
        print("[ERROR] Trained model not found:", model_path)
        return False

    model = joblib.load(model_path)

    # Load testing data
    test_path = ROOT / "data/processed/test.csv"

    if not test_path.exists():
        print("[ERROR] Testing dataset not found.")
        return False

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["Rating"])
    y_test = test_df["Rating"]

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, predictions))
    }

    # Save actual and predicted ratings
    prediction_df = pd.DataFrame({
        "Actual_Rating": y_test.values,
        "Predicted_Rating": predictions
    })

    prediction_path = ROOT / "outputs/evaluation_predictions.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(prediction_path, index=False)

    # Create evaluation report
    report = {
        "stage": "Lab 8 - Model Evaluation",
        "model": "RandomForestRegressor",
        "target": "Rating",
        "testing_rows": int(len(y_test)),
        "metrics": metrics
    }

    report_path = ROOT / "reports/evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    # Display results
    print("\n========== EVALUATION RESULTS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print("========================================")

    print("[SUCCESS] Evaluation completed.")
    print("[SUCCESS] Predictions saved:", prediction_path)
    print("[SUCCESS] Report saved:", report_path)

    return True


if __name__ == "__main__":

    try:
        result = evaluate_model()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)

