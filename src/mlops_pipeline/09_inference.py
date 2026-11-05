
import sys
import yaml
import mlflow
import mlflow.pyfunc
import pandas as pd

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FEATURES = [
    "Company",
    "Specific_Bean_Origin",
    "Review_Date",
    "Cocoa_Percent",
    "Company_Location",
    "Bean_Type",
    "Broad_Bean_Origin"
]


def run_inference():

    print("=" * 55)
    print("FLAVORS OF CACAO - CHAMPION MODEL INFERENCE")
    print("=" * 55)

    # Load configuration
    with open(ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Configure MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Load verified local model artifact
    model_path = (
        ROOT / "mlruns" / "1" / "models"
        / "m-188c29a769e5428086d7bf7cba1acfca"
        / "artifacts"
    )

    print("[INFO] Loading saved model...")

    if not model_path.exists():
        print("[ERROR] Model artifact not found:", model_path)
        return False

    model = mlflow.pyfunc.load_model(str(model_path))

    print("[SUCCESS] Model loaded.")

    # Load new data
    input_path = ROOT / config["data"]["new_data_path"]

    if not input_path.exists():
        print("[ERROR] New data file not found:", input_path)
        print("[INFO] Create data/raw/new_flavors.csv first.")
        return False

    df = pd.read_csv(input_path)

    # Rename original dataset columns if all 9 are present
    if len(df.columns) == 9:
        df.columns = [
            "Company",
            "Specific_Bean_Origin",
            "REF",
            "Review_Date",
            "Cocoa_Percent",
            "Company_Location",
            "Rating",
            "Bean_Type",
            "Broad_Bean_Origin"
        ]

    # Remove target and ID columns if present
    df = df.drop(
        columns=["Rating", "REF"],
        errors="ignore"
    )

    # Check required columns
    missing_columns = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_columns:
        print("[ERROR] Missing columns:", missing_columns)
        return False

    # Keep original 7 columns in correct order
    df = df[FEATURES].copy()

    # Clean numeric columns
    df["Cocoa_Percent"] = (
        df["Cocoa_Percent"]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    df["Cocoa_Percent"] = pd.to_numeric(
        df["Cocoa_Percent"], errors="coerce"
    )

    df["Review_Date"] = pd.to_numeric(
        df["Review_Date"], errors="coerce"
    )

    # IMPORTANT:
    # The saved MLflow model already contains its preprocessor.
    # Pass the original 7 columns directly to the model.
    print("[INFO] Predicting ratings...")

    predictions = model.predict(df)

    # Save predictions
    result = df.copy()
    result["Predicted_Rating"] = predictions

    output_path = ROOT / config["paths"]["inference_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)

    print("\n========== INFERENCE RESULTS ==========")
    print("Records predicted:", len(predictions))

    print("\nSample predictions:")
    print(result[["Predicted_Rating"]].head())

    print("\n[SUCCESS] Inference completed.")
    print("[SUCCESS] Predictions saved:", output_path)

    return True


if __name__ == "__main__":

    try:
        result = run_inference()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
