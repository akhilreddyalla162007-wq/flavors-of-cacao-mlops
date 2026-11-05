
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "api_predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("cocoa_api")


def log_prediction(input_data, prediction):
    logger.info(
        "Input: %s | Predicted rating: %s",
        input_data,
        prediction
    )
