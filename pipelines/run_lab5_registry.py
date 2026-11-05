import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 5: MLflow Model Registry")
    print("[INFO] =========================================")
    scripts = [
        "src/preprocess.py",
        "src/register_model.py"
    ]
    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)
    print("\n[SUCCESS] Lab 5 Model Registry Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
