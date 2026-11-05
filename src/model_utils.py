from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

NUMERIC = ["Review_Date", "Cocoa_Percent"]
CATEGORICAL = [
    "Company", "Specific_Bean_Origin", "Company_Location",
    "Bean_Type", "Broad_Bean_Origin"
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Rating"

def load_data(root: Path):
    df = pd.read_csv(root / "data" / "processed" / "preprocessed.csv")
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.20, random_state=42)

def make_preprocessor():
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", cat, CATEGORICAL)
    ])
