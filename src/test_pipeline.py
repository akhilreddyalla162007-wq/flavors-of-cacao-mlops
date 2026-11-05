from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    required = [
        ROOT/"data/raw/flavors_of_cacao.csv",
        ROOT/"data/processed/preprocessed.csv",
        ROOT/"models/random_forest_model.joblib"
    ]
    for path in required:
        assert path.exists(), f"Missing required artifact: {path}"

    df = pd.read_csv(ROOT/"data/processed/preprocessed.csv")
    assert "Rating" in df.columns
    assert df["Rating"].notna().all()
    assert len(df) > 0
    print("[SUCCESS] Pipeline schema/artifact validation passed.")

if __name__ == "__main__":
    main()
