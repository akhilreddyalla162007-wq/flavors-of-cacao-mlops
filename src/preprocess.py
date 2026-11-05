import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "flavors_of_cacao.csv"
PROCESSED_FILE = ROOT / "data" / "processed" / "preprocessed.csv"
METADATA_FILE = ROOT / "data" / "processed" / "dataset_metadata.json"

def main():
    print("[INFO] Starting preprocessing...")
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    if len(df.columns) == 9:
        df.columns = [
            "Company", "Specific_Bean_Origin", "REF", "Review_Date",
            "Cocoa_Percent", "Company_Location", "Rating",
            "Bean_Type", "Broad_Bean_Origin"
        ]

    df = df.drop_duplicates()

    df["Cocoa_Percent"] = (
        df["Cocoa_Percent"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["Cocoa_Percent"] = pd.to_numeric(df["Cocoa_Percent"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Review_Date"] = pd.to_numeric(df["Review_Date"], errors="coerce")
    df = df.dropna(subset=["Rating"])

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    metadata = {
        "project_id": "241FA18335",
        "dataset": "flavors_of_cacao.csv",
        "task": "regression",
        "target": "Rating",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": [
            "Company", "Specific_Bean_Origin", "Review_Date",
            "Cocoa_Percent", "Company_Location", "Bean_Type",
            "Broad_Bean_Origin"
        ]
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=4))
    print(f"[SUCCESS] Preprocessing completed. Shape: {df.shape}")

if __name__ == "__main__":
    main()
