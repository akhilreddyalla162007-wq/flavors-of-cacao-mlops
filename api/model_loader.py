
from pathlib import Path
import mlflow.pyfunc


# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Existing trained MLflow model
MODEL_PATH = (
    BASE_DIR
    / "mlruns"
    / "1"
    / "models"
    / "m-188c29a769e5428086d7bf7cba1acfca"
    / "artifacts"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = mlflow.pyfunc.load_model(str(MODEL_PATH))

    print("Cocoa prediction model loaded successfully!")

    return model
