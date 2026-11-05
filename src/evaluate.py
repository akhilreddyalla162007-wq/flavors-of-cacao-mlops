from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    model = joblib.load(ROOT/"models/best_model.joblib")
    test = pd.read_csv(ROOT/"outputs/test_data.csv")
    X = test.drop(columns=["Rating"])
    y = test["Rating"]
    pred = model.predict(X)

    metrics = {
        "Model":"Tuned Random Forest",
        "MAE":mean_absolute_error(y,pred),
        "MSE":mean_squared_error(y,pred),
        "RMSE":np.sqrt(mean_squared_error(y,pred)),
        "R2":r2_score(y,pred)
    }
    pd.DataFrame([metrics]).to_csv(ROOT/"outputs/evaluation_results.csv", index=False)
    pd.DataFrame({"Actual_Rating":y, "Predicted_Rating":pred}).to_csv(
        ROOT/"outputs/actual_vs_predicted.csv", index=False
    )
    print("\n========== EVALUATION ==========")
    for k,v in metrics.items():
        if isinstance(v,(int,float,np.floating)):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("[SUCCESS] Evaluation completed.")

if __name__ == "__main__":
    main()
