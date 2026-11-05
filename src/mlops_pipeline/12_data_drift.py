
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = BASE_DIR / "data/raw/flavors_of_cacao.csv"
NEW_FILE = BASE_DIR / "data/raw/new_flavors.csv"

REPORT_DIR = BASE_DIR / "artifacts"
REPORT_DIR.mkdir(exist_ok=True)

# Features used by the prediction model
FEATURES = [
    "Company",
    "Specific_Bean_Origin",
    "Review_Date",
    "Cocoa_Percent",
    "Company_Location",
    "Bean_Type",
    "Broad_Bean_Origin"
]

NUMERIC_FEATURES = ["Review_Date", "Cocoa_Percent"]

# Clean column names

def clean_columns(df):
    import re

    # Remove special characters and extra spaces
    df.columns = [
        re.sub(r"\s+", " ", str(column).replace("\xa0", " ")).strip()
        for column in df.columns
    ]

    rename_columns = {
        "Company (Maker-if known)": "Company",
        "Specific Bean Origin or Bar Name": "Specific_Bean_Origin",
        "Review Date": "Review_Date",
        "Cocoa Percent": "Cocoa_Percent",
        "Company Location": "Company_Location",
        "Bean Type": "Bean_Type",
        "Broad Bean Origin": "Broad_Bean_Origin"
    }

    return df.rename(columns=rename_columns)



# Load datasets
print("Starting Data Drift Monitoring...")

train_data = pd.read_csv(TRAIN_FILE)
new_data = pd.read_csv(NEW_FILE)

train_data = clean_columns(train_data)
new_data = clean_columns(new_data)

# Remove target column from training data
if "Rating" in train_data.columns:
    train_data = train_data.drop(columns=["Rating"])

# Check required columns
missing_train = [c for c in FEATURES if c not in train_data.columns]
missing_new = [c for c in FEATURES if c not in new_data.columns]

if missing_train or missing_new:
    raise ValueError(
        f"Missing training columns: {missing_train}\n"
        f"Missing new-data columns: {missing_new}"
    )

train_data = train_data[FEATURES].copy()
new_data = new_data[FEATURES].copy()

# Convert numerical columns
for column in NUMERIC_FEATURES:
    train_data[column] = pd.to_numeric(
        train_data[column].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )

    new_data[column] = pd.to_numeric(
        new_data[column].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )


# Clean categorical columns
for column in FEATURES:
    if column not in NUMERIC_FEATURES:
        train_data[column] = (
            train_data[column].astype("string").fillna("Unknown").str.strip()
        )
        new_data[column] = (
            new_data[column].astype("string").fillna("Unknown").str.strip()
        )


drift_results = {}

# Numerical drift detection
for column in NUMERIC_FEATURES:
    reference_mean = train_data[column].mean()
    new_mean = new_data[column].mean()

    if pd.isna(reference_mean) or pd.isna(new_mean):
        drift_results[column] = {
            "type": "numerical",
            "status": "Not enough valid data",
            "reference_mean": None,
            "new_mean": None
        }
        continue

    if reference_mean != 0:
        change_percent = (
            abs(new_mean - reference_mean) / abs(reference_mean)
        ) * 100
    else:
        change_percent = 0 if new_mean == 0 else 100

    drift_results[column] = {
        "type": "numerical",
        "reference_mean": round(float(reference_mean), 2),
        "new_mean": round(float(new_mean), 2),
        "change_percent": round(float(change_percent), 2),
        "drift_detected": bool(change_percent > 20)
    }

# Categorical drift detection
for column in FEATURES:
    if column in NUMERIC_FEATURES:
        continue

    reference_distribution = (
        train_data[column].value_counts(normalize=True)
    )

    new_distribution = (
        new_data[column].value_counts(normalize=True)
    )

    all_categories = set(reference_distribution.index).union(
        set(new_distribution.index)
    )

    total_difference = sum(
        abs(
            reference_distribution.get(category, 0)
            - new_distribution.get(category, 0)
        )
        for category in all_categories
    )

    # Total Variation Distance
    drift_score = total_difference / 2

    drift_results[column] = {
        "type": "categorical",
        "drift_score": round(float(drift_score), 4),
        "drift_detected": bool(drift_score > 0.20)
    }

# Overall result
drifted_features = [
    column
    for column, result in drift_results.items()
    if result.get("drift_detected", False)
]

report = {
    "timestamp": datetime.now().isoformat(),
    "reference_dataset": str(TRAIN_FILE),
    "new_dataset": str(NEW_FILE),
    "reference_rows": len(train_data),
    "new_rows": len(new_data),
    "small_sample_warning": len(new_data) < 30,
    "total_features_checked": len(FEATURES),
    "drifted_features": drifted_features,
    "drift_detected": len(drifted_features) > 0,
    "feature_results": drift_results
}

# Save report
report_path = REPORT_DIR / "data_drift_report.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

# Display results
print("\n========== DATA DRIFT REPORT ==========")

print("Reference rows:", len(train_data))
print("New data rows:", len(new_data))

for column, result in drift_results.items():
    print(f"\nFeature: {column}")
    print("Result:", result)

print("\nDrifted features:", drifted_features)
print("Overall drift detected:", report["drift_detected"])

if report["small_sample_warning"]:
    print("WARNING: New dataset has fewer than 30 rows.")
    print("Collect more incoming records for reliable drift monitoring.")

print("\nReport saved:", report_path)
print("\nDATA DRIFT MONITORING COMPLETED!")
