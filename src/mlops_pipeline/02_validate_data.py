
import os
import sys
import json
import pandas as pd
import yaml

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def validate_data():

    print("=" * 50)
    print("FLAVORS OF CACAO - DATA VALIDATION")
    print("=" * 50)

    # Read configuration
    with open(os.path.join(ROOT, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)

    # Load dataset
    data_path = os.path.join(ROOT, config["data"]["raw_path"])
    df = pd.read_csv(data_path)

    # Rename columns
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

    # Required columns
    required_columns = [
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

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    # Check target column
    target = config["data"]["target_column"]

    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    invalid_ratings = 0

    if target in df.columns:
        invalid_ratings = int(
            ((df[target] < 1) | (df[target] > 5)).sum()
        )

    # Missing values and duplicates
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Validation result
    validation_passed = (
        len(missing_columns) == 0
        and target in df.columns
        and invalid_ratings == 0
    )

    report = {
        "stage": "Lab 7 - Data Validation",
        "validation_passed": validation_passed,
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "invalid_ratings": invalid_ratings,
        "total_rows": int(len(df))
    }

    # Save validation report
    report_path = os.path.join(
        ROOT, "artifacts", "data_validation_report.json"
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    # Display results
    print("Missing columns:", missing_columns)
    print("Missing values:", missing_values)
    print("Duplicate rows:", duplicate_rows)
    print("Invalid ratings:", invalid_ratings)

    if validation_passed:
        print("[SUCCESS] Data validation passed.")
    else:
        print("[ERROR] Data validation failed.")

    print("Report saved:", report_path)

    return validation_passed


if __name__ == "__main__":

    try:
        result = validate_data()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
