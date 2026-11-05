
import os
import sys
import json
import pandas as pd
import yaml

# Get the main project folder
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def load_data():

    print("=" * 50)
    print("FLAVORS OF CACAO - DATA LOADING")
    print("=" * 50)

    # Read configuration
    config_path = os.path.join(ROOT, "config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Read dataset
    data_path = os.path.join(ROOT, config["data"]["raw_path"])

    if not os.path.exists(data_path):
        print("[ERROR] Dataset not found:", data_path)
        return False

    df = pd.read_csv(data_path)

    # Rename original dataset columns
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

    target = config["data"]["target_column"]

    if target not in df.columns:
        print("[ERROR] Target column not found:", target)
        print("Available columns:", df.columns.tolist())
        return False

    # Create report
    report = {
        "stage": "Lab 7 - Data Loading",
        "dataset": config["data"]["raw_path"],
        "task": "Regression",
        "target_column": target,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

    # Save report
    report_dir = os.path.join(ROOT, "artifacts")
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(
        report_dir, "data_load_report.json"
    )

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("[SUCCESS] Dataset loaded successfully.")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Missing values:", report["missing_values"])
    print("Duplicate rows:", report["duplicate_rows"])
    print("[SUCCESS] Report saved:", report_path)

    return True


if __name__ == "__main__":

    try:
        result = load_data()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
