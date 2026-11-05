
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "src/mlops_pipeline/12_api_performance_monitoring.py",
    "src/mlops_pipeline/12_data_drift.py",
    "src/mlops_pipeline/13_model_performance_monitoring.py",
    "src/mlops_pipeline/14_retraining_readiness.py"
]

print("=" * 50)
print("FLAVORS OF CACAO - LAB 12 MONITORING PIPELINE")
print("=" * 50)

passed = 0
failed = 0

for script in SCRIPTS:
    script_path = BASE_DIR / script

    print("\nRunning:", script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode == 0:
        print("[SUCCESS]", script)
        passed += 1
    else:
        print("[FAILED]", script)
        failed += 1

print("\n" + "=" * 50)
print("LAB 12 PIPELINE SUMMARY")
print("=" * 50)

print("Total scripts:", len(SCRIPTS))
print("Successful:", passed)
print("Failed:", failed)

if failed == 0:
    print("\nLAB 12 MONITORING PIPELINE COMPLETED!")
else:
    print("\nLAB 12 PIPELINE COMPLETED WITH ERRORS")
