from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from model_utils import load_data, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

def main():
    if not (ROOT/"data/processed/preprocessed.csv").exists():
        raise FileNotFoundError("Run src/preprocess.py first.")

    X_train, X_test, y_train, y_test = load_data(ROOT)
    preprocessor = make_preprocessor()
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        )
    }

    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        print(f"[INFO] Training {name}...")
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR/f"{name}_model.joblib")

    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }
    search = GridSearchCV(rf_pipe, grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    print("[INFO] Running Random Forest GridSearch...")
    search.fit(X_train, y_train)
    joblib.dump(search.best_estimator_, MODEL_DIR/"best_model.joblib")

    test = X_test.copy()
    test["Rating"] = y_test.values
    test.to_csv(OUTPUT_DIR/"test_data.csv", index=False)
    (OUTPUT_DIR/"training_info.json").write_text(json.dumps({
        "project_id":"241FA18335",
        "train_samples":len(X_train),
        "test_samples":len(X_test),
        "best_parameters":search.best_params_
    }, indent=4))
    print("[SUCCESS] Baseline training completed.")
    print("[INFO] Best parameters:", search.best_params_)

if __name__ == "__main__":
    main()
