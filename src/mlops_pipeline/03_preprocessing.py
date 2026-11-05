
import os
import json
import sys
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def preprocess_data():

    print("=" * 50)
    print("FLAVORS OF CACAO - PREPROCESSING")
    print("=" * 50)

    # Load configuration
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

    # Clean data
    df["Cocoa_Percent"] = (
        df["Cocoa_Percent"].astype(str)
        .str.replace("%", "", regex=False)
    )

    df["Cocoa_Percent"] = pd.to_numeric(
        df["Cocoa_Percent"], errors="coerce"
    )

    df["Review_Date"] = pd.to_numeric(
        df["Review_Date"], errors="coerce"
    )

    df["Rating"] = pd.to_numeric(
        df["Rating"], errors="coerce"
    )

    # Remove rows without target values
    df = df.dropna(subset=["Rating"])

    # Input and target
    X = df.drop(columns=["Rating", "REF"])
    y = df["Rating"]

    # Numeric and categorical columns
    numeric_columns = [
        "Review_Date",
        "Cocoa_Percent"
    ]

    categorical_columns = [
        "Company",
        "Specific_Bean_Origin",
        "Company_Location",
        "Bean_Type",
        "Broad_Bean_Origin"
    ]

    # Numeric preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Split dataset
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save processed datasets
    processed_dir = os.path.join(ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    joblib.dump(
        preprocessor,
        os.path.join(ROOT, "models", "preprocessor.pkl")
    )

    # Save training and testing data
    from scipy.sparse import issparse

    if issparse(X_train_processed):
        X_train_processed = X_train_processed.toarray()

    if issparse(X_test_processed):
        X_test_processed = X_test_processed.toarray()

    train_df = pd.DataFrame(X_train_processed)
    train_df["Rating"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_processed)
    test_df["Rating"] = y_test.reset_index(drop=True)

    train_df.to_csv(
        os.path.join(processed_dir, "train.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(processed_dir, "test.csv"),
        index=False
    )

    # Save preprocessing report
    report = {
        "stage": "Lab 7 - Preprocessing",
        "original_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_features": int(X_train_processed.shape[1]),
        "testing_features": int(X_test_processed.shape[1]),
        "target": "Rating"
    }

    report_path = os.path.join(
        ROOT, "artifacts", "preprocessing_report.json"
    )

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("[SUCCESS] Preprocessing completed.")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Features:", X_train_processed.shape[1])
    print("[SUCCESS] Preprocessor saved.")
    print("[SUCCESS] Train and test datasets saved.")
    print("[SUCCESS] Report saved:", report_path)


if __name__ == "__main__":

    try:
        preprocess_data()

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)
