# tests/test_app.py

import sys
import os
from fastapi.testclient import TestClient

# Ensure the backend module is available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from prediction import app  # Import your FastAPI app

client = TestClient(app)

def test_predict_success():
    with open("tests/test_image.jpg", "rb") as image:
        response = client.post("/predict", files={"file": ("test_image.jpg", image, "image/jpeg")})
    assert response.status_code == 200
    assert "prediction" in response.json()  # Adjust based on actual response structure

def test_predict_no_image():
    response = client.post("/predict", files={})
    assert response.status_code == 422  # Expecting a validation error
