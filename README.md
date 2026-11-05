# 241FA18335 - Flavors of Cacao MLOps Project

Dataset: Flavors of Cacao
Target: Rating
Task: Regression

## Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Lab 3
python pipelines/run_lab3_baseline.py

## Lab 4
python pipelines/run_lab4_tracking.py

## Lab 5
python pipelines/run_lab5_registry.py

## Reproducibility
python src/validate_reproducibility.py

## Pipeline tests
python src/test_pipeline.py

## Batch inference
Create data/processed/batch_input.csv using the same feature columns, then:
python src/batch_inference.py

## MLflow UI
mlflow ui --backend-store-uri ./artifacts/mlruns
