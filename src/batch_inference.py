from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "random_forest_model.joblib"
INPUT = ROOT / "data" / "processed" / "batch_input.csv"
OUTPUT = ROOT / "outputs" / "batch_predictions.csv"

def main():
    if not MODEL.exists():
        raise FileNotFoundError("Run src/train.py first.")
    if not INPUT.exists():
        raise FileNotFoundError(f"Create batch input at {INPUT}")
    model = joblib.load(MODEL)
    df = pd.read_csv(INPUT)
    df["Predicted_Rating"] = model.predict(df)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"[SUCCESS] Batch predictions saved to {OUTPUT}")

if __name__ == "__main__":
    main()
