
import json
from pathlib import Path
from datetime import datetime

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Existing evaluation report
EVALUATION_FILE = BASE_DIR / "reports" / "evaluation_report.json"

# Output directory
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

print("Starting Model Performance Monitoring...")

# Check evaluation report
if not EVALUATION_FILE.exists():
    raise FileNotFoundError("Evaluation report not found!")

# Read existing model evaluation
with open(EVALUATION_FILE, "r") as file:
    evaluation = json.load(file)

metrics = evaluation["metrics"]

# Baseline thresholds
MAX_MAE = 0.40
MAX_RMSE = 0.60
MAX_MSE = 0.30
MIN_R2 = 0.10

# Check model performance
checks = {
    "MAE": {
        "value": metrics["MAE"],
        "threshold": MAX_MAE,
        "passed": metrics["MAE"] <= MAX_MAE
    },
    "MSE": {
        "value": metrics["MSE"],
        "threshold": MAX_MSE,
        "passed": metrics["MSE"] <= MAX_MSE
    },
    "RMSE": {
        "value": metrics["RMSE"],
        "threshold": MAX_RMSE,
        "passed": metrics["RMSE"] <= MAX_RMSE
    },
    "R2": {
        "value": metrics["R2"],
        "threshold": MIN_R2,
        "passed": metrics["R2"] >= MIN_R2
    }
}

# Overall result
failed_metrics = [
    name for name, result in checks.items()
    if not result["passed"]
]

performance_status = (
    "HEALTHY" if len(failed_metrics) == 0 else "DEGRADED"
)

# Create report
report = {
    "timestamp": datetime.now().isoformat(),
    "model": evaluation["model"],
    "testing_rows": evaluation["testing_rows"],
    "baseline_metrics": metrics,
    "threshold_checks": checks,
    "failed_metrics": failed_metrics,
    "performance_status": performance_status
}

# Save report
report_path = ARTIFACT_DIR / "model_performance_report.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

# Display results
print("\nMODEL PERFORMANCE REPORT")
print("------------------------")

for name, result in checks.items():
    status = "PASS" if result["passed"] else "FAIL"

    print(
        f"{name}: {result['value']:.4f} "
        f"| Threshold: {result['threshold']} "
        f"| {status}"
    )

print("\nOverall Status:", performance_status)
print("Failed Metrics:", failed_metrics)
print("\nReport saved:", report_path)
print("\nMODEL PERFORMANCE MONITORING COMPLETED!")