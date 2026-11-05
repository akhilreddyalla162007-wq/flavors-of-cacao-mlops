
from fastapi import FastAPI, HTTPException
import pandas as pd

from api.schemas import CocoaInput, PredictionResponse
from api.model_loader import load_model
from api.monitoring_logger import log_prediction


app = FastAPI(
    title="Flavors of Cacao Prediction API",
    description="Predict chocolate ratings using a trained ML model",
    version="1.0"
)

# Load trained model once when the API starts
model = load_model()


@app.get("/")
def home():
    return {
        "message": "Welcome to Flavors of Cacao Prediction API",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_rating(data: CocoaInput):

    try:
        # Convert input into a DataFrame
        input_data = pd.DataFrame([{
            "Company": data.Company,
            "Specific_Bean_Origin": data.Specific_Bean_Origin,
            "Review_Date": data.Review_Date,
            "Cocoa_Percent": data.Cocoa_Percent,
            "Company_Location": data.Company_Location,
            "Bean_Type": data.Bean_Type,
            "Broad_Bean_Origin": data.Broad_Bean_Origin
        }])

        # Predict rating
        prediction = model.predict(input_data)

        predicted_rating = round(float(prediction[0]), 2)

        # Save prediction in log
        log_prediction(
            input_data.to_dict(orient="records"),
            predicted_rating
        )

        return {
            "predicted_rating": predicted_rating,
            "message": "Cocoa rating predicted successfully"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
