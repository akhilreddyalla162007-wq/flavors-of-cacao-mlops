
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[2]

NEW_DATA_FILE = BASE_DIR / "data/raw/new_flavors.csv"
DRIFT_REPORT = BASE_DIR / "artifacts/data_drift_report.json"

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

print("Starting Model Retraining Readiness Monitoring...")

# Load incoming data
data = pd.read_csv(NEW_DATA_FILE)

# Check whether target column is available
target_available = "Rating" in data.columns

# Check minimum number of records
minimum_rows = 30
enough_data = len(data) >= minimum_rows

# Check whether drift was detected
drift_detected = False

if DRIFT_REPORT.exists():
    with open(DRIFT_REPORT, "r") as file:
        drift_report = json.load(file)

    drift_detected = drift_report.get("drift_detected", False)

# Decide retraining readiness
if target_available and enough_data:
    if drift_detected:
        status = "RETRAINING RECOMMENDED"
    else:
        status = "RETRAINING NOT REQUIRED"
else:
    status = "NOT READY FOR RETRAINING"

# Create report
report = {
    "timestamp": datetime.now().isoformat(),
    "incoming_rows": len(data),
    "target_column_available": target_available,
    "minimum_required_rows": minimum_rows,
    "enough_data": enough_data,
    "drift_detected": drift_detected,
    "retraining_status": status
}

# Save report
report_path = ARTIFACT_DIR / "retraining_readiness_report.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

# Display results
print("\nRETRAINING READINESS REPORT")
print("---------------------------")
print("Incoming rows:", len(data))
print("Rating column available:", target_available)
print("Enough data:", enough_data)
print("Drift detected:", drift_detected)
print("Retraining status:", status)

print("\nReport saved:", report_path)
print("\nRETRAINING READINESS MONITORING COMPLETED!")
