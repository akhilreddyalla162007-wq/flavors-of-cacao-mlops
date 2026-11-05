
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True


def test_prediction():
    data = {
        "Company": "A. Morin",
        "Specific_Bean_Origin": "Peru",
        "Review_Date": 2016,
        "Cocoa_Percent": 70,
        "Company_Location": "France",
        "Bean_Type": "",
        "Broad_Bean_Origin": ""
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "predicted_rating" in response.json()


def test_invalid_cocoa_percent():
    data = {
        "Company": "A. Morin",
        "Specific_Bean_Origin": "Peru",
        "Review_Date": 2016,
        "Cocoa_Percent": 150,
        "Company_Location": "France",
        "Bean_Type": "",
        "Broad_Bean_Origin": ""
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422
