import os
import io
import uuid
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.util_classes import *
from utils.util_model_classes import *
from utils.utils_test import *

# Initialize the app
app = FastAPI(title="LaTeX Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    global model, tokenizer, config, image

    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(contents)).convert('L')
        latex = predict_image(model, tokenizer, image, config)
        print("latex_pred", latex)

        return {"latex": latex}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/feedback")
async def receive_feedback(correct_latex: str = Form(...)):
    """
    Endpoint to receive feedback when the prediction is incorrect.
    Saves image and corrected LaTeX to `feedback_data/`.
    """
    try:
        # contents = await file.read()
        # if not contents:
        #     raise HTTPException(status_code=400, detail="Empty file")

        #image = Image.open(io.BytesIO(contents)).convert('L')

        # Generate unique filename
        uid = str(uuid.uuid4())[:8]
        os.makedirs("feedback_data/images", exist_ok=True)
        os.makedirs("feedback_data/labels", exist_ok=True)

        image_path = f"feedback_data/images/{uid}.png"
        label_path = f"feedback_data/labels/{uid}.txt"

        image.save(image_path)
        with open(label_path, 'w') as f:
            f.write(correct_latex.strip())

        return {"message": "Feedback received", "id": uid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
