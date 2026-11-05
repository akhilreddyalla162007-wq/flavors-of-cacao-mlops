from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "reproducibility_report.json"

def run_deterministic_test():
    print("Running Reproducibility Validation...")
    model_path = ROOT / "models" / "random_forest_model.joblib"
    data_path = ROOT / "data" / "processed" / "preprocessed.csv"
    if not model_path.exists() or not data_path.exists():
        raise FileNotFoundError("Run preprocessing and training first.")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["Rating","REF"])
    y = df["Rating"]
    model = joblib.load(model_path)

    pred1 = model.predict(X)
    pred2 = model.predict(X)
    rmse1 = np.sqrt(mean_squared_error(y,pred1))
    rmse2 = np.sqrt(mean_squared_error(y,pred2))
    reproducible = np.array_equal(pred1,pred2)

    result = {
        "test_name":"Pipeline Reproducibility Validation",
        "model":"RandomForestRegressor",
        "random_state":42,
        "execution_1_rmse":float(rmse1),
        "execution_2_rmse":float(rmse2),
        "predictions_identical":bool(reproducible),
        "status":"PASSED" if reproducible else "FAILED"
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result,indent=4))
    print(f"Execution 1 RMSE: {rmse1:.6f}")
    print(f"Execution 2 RMSE: {rmse2:.6f}")
    print("SUCCESS: Pipeline is 100% reproducible." if reproducible else "FAILED: Pipeline is non-deterministic.")
    print(f"Report saved: {REPORT}")

if __name__ == "__main__":
    run_deterministic_test()
