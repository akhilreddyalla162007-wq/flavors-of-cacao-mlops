
import os
import sys
import json
import pandas as pd
import numpy as np


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def validate_outputs():

    print("=" * 50)
    print("FLAVORS OF CACAO - OUTPUT VALIDATION")
    print("=" * 50)

    train_path = os.path.join(
        ROOT, "data", "processed", "train.csv"
    )

    test_path = os.path.join(
        ROOT, "data", "processed", "test.csv"
    )

    # Check whether files exist
    if not os.path.exists(train_path):
        print("[ERROR] Training file not found.")
        return False

    if not os.path.exists(test_path):
        print("[ERROR] Testing file not found.")
        return False

    # Load files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Check required target
    if "Rating" not in train_df.columns:
        print("[ERROR] Rating column missing in training data.")
        return False

    if "Rating" not in test_df.columns:
        print("[ERROR] Rating column missing in testing data.")
        return False

    # Separate features and target
    X_train = train_df.drop(columns=["Rating"])
    X_test = test_df.drop(columns=["Rating"])

    y_train = train_df["Rating"]
    y_test = test_df["Rating"]

    # Check feature consistency
    same_features = list(X_train.columns) == list(X_test.columns)

    # Check missing and infinite values
    train_missing = int(train_df.isnull().sum().sum())
    test_missing = int(test_df.isnull().sum().sum())

    train_infinite = int(
        np.isinf(X_train.select_dtypes(include=np.number)).sum().sum()
    )

    test_infinite = int(
        np.isinf(X_test.select_dtypes(include=np.number)).sum().sum()
    )

    # Check target values
    invalid_train_ratings = int(
        ((y_train < 1) | (y_train > 5)).sum()
    )

    invalid_test_ratings = int(
        ((y_test < 1) | (y_test > 5)).sum()
    )

    # Final validation
    validation_passed = (
        same_features
        and train_missing == 0
        and test_missing == 0
        and train_infinite == 0
        and test_infinite == 0
        and invalid_train_ratings == 0
        and invalid_test_ratings == 0
    )

    # Create report
    report = {
        "stage": "Lab 7 - Output Validation",
        "validation_passed": validation_passed,
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "training_features": int(X_train.shape[1]),
        "testing_features": int(X_test.shape[1]),
        "same_features": same_features,
        "training_missing_values": train_missing,
        "testing_missing_values": test_missing,
        "training_infinite_values": train_infinite,
        "testing_infinite_values": test_infinite,
        "invalid_training_ratings": invalid_train_ratings,
        "invalid_testing_ratings": invalid_test_ratings
    }

    report_path = os.path.join(
        ROOT, "artifacts", "output_validation_report.json"
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    # Display results
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training features:", X_train.shape[1])
    print("Testing features:", X_test.shape[1])
    print("Same features:", same_features)
    print("Training missing values:", train_missing)
    print("Testing missing values:", test_missing)
    print("Training infinite values:", train_infinite)
    print("Testing infinite values:", test_infinite)

    if validation_passed:
        print("[SUCCESS] Output validation passed.")
    else:
        print("[ERROR] Output validation failed.")

    print("Report saved:", report_path)

    return validation_passed


if __name__ == "__main__":

    try:
        result = validate_outputs()
        sys.exit(0 if result else 1)

    except Exception as error:
        print("[ERROR]", error)
        sys.exit(1)

