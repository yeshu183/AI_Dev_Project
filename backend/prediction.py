import os
import io
import uuid
import torch
import uvicorn
from PIL import Image
print("Current Working Directory:", os.getcwd())
from utils.util_classes import *
from utils.util_model_classes import *
from utils.utils_test import *
import base64
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Initialize the app
app = FastAPI(title="LaTeX Recognition API")

# Image feedback storage paths
LOCAL_FEEDBACK_DIR = "/app/feedback_data"
LOCAL_IMAGES_DIR = os.path.join(LOCAL_FEEDBACK_DIR, "images")
LOCAL_LABELS_DIR = os.path.join(LOCAL_FEEDBACK_DIR, "labels")

# Ensure directories exist
os.makedirs(LOCAL_IMAGES_DIR, exist_ok=True)
os.makedirs(LOCAL_LABELS_DIR, exist_ok=True)

# Image cache
image_cache = {}

@app.on_event("startup")
async def load_model():
    global model, tokenizer, config
    checkpoint = torch.load("models/best_model.pth", map_location='cpu', weights_only=False)
    print("Model loaded successfully")

    tokenizer = checkpoint['tokenizer']
    config = checkpoint['config']
    config.device = 'cpu'

    model = HandwrittenMathRecognizer(config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Loading model and tokenizer...")

@app.get("/")
def read_root():
    return {"message": "LaTeX Recognition API - Upload an image to convert to LaTeX"}

@app.post("/predict")
async def predict_from_file(file: UploadFile = File(...)):
    """Endpoint for predicting from an uploaded image file"""
    global model, tokenizer, config, image, session_id, latex_prediction

    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(contents)).convert('L')
        # Generate unique ID for this prediction session
        session_id = str(uuid.uuid4())

        # Store image data in the cache with session ID
        image_cache[session_id] = {
            "image_data": contents,
            "filename": file.filename
        }
        latex_prediction = predict_image(model, tokenizer, image, config)
        return {
            "latex": latex_prediction,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/feedback")
async def save_feedback(
        latex_feedback: Optional[str] = Form(None)
    ):
    """
    Endpoint to receive feedback when the prediction is incorrect.
    Saves image and corrected LaTeX to `feedback_data/`.
    """
    try:
        # Retrieve image data from cache
        if session_id not in image_cache:
            raise HTTPException(status_code=400, detail="Session expired or invalid")
        
        cached_data = image_cache[session_id]
        image_data = cached_data["image_data"]
        original_filename = cached_data["filename"]
        
        # Generate unique filename for storing locally
        unique_filename = f"{uuid.uuid4()}{os.path.splitext(original_filename)[1]}"
        
        # Save image to local filesystem
        local_image_path = os.path.join(LOCAL_IMAGES_DIR, unique_filename)
        with open(local_image_path, "wb") as f:
            f.write(image_data)
        
        # Save LaTeX to local filesystem
        local_label_path = os.path.join(LOCAL_LABELS_DIR, f"{os.path.splitext(unique_filename)[0]}.txt")
        with open(local_label_path, "w") as f:
            f.write(latex_feedback or latex_prediction)
        
        return JSONResponse(
            content={
                "message": "Feedback saved successfully",
                "image_filename": unique_filename,
                "latex": latex_feedback or latex_prediction
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
