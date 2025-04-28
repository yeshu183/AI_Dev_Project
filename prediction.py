import os
import glob
import random
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from tqdm import tqdm
import re
import time
import argparse
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from util_classes import *
from util_model_classes import *
from utils_test import *
import io

#image_path = "datasets/filtered_basic_arithmetic/test/images/expr_000877.png"
#print(predict_image(model,tokenizer,image_path,config))


# Initialize the app
app = FastAPI(title="LaTeX Recognition API")

# Request model for raw image data
class ImageData(BaseModel):
    image_vector: list = []

@app.on_event("startup")
async def load_model():
    global model, tokenizer, config
    # Load the full checkpoint
    checkpoint = torch.load("best_model.pth", map_location='cpu', weights_only=False)
    print("Model loaded successfully")

    tokenizer = checkpoint['tokenizer']
    config = checkpoint['config']
    config.device = 'cpu'
    model = HandwrittenMathRecognizer(config, tokenizer.vocab_size)
    # Load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # For demonstration purposes:
    print("Loading model and tokenizer...")

@app.get("/")
def read_root():
    return {"message": "LaTeX Recognition API - Upload an image to convert to LaTeX"}

@app.post("/predict-vector")
async def predict_from_vector(data: ImageData):
    """Endpoint for predicting from a vector representation of an image"""
    global model, tokenizer, config
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not data.image_vector:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    try:
        # Convert vector to image
        # Assuming the vector is a flattened grayscale image
        img_size = int(np.sqrt(len(data.image_vector)))
        image_arr = np.array(data.image_vector).reshape(img_size, img_size)
        image = Image.fromarray(image_arr.astype('uint8'))
        
        # Get prediction
        latex = predict_image(model, tokenizer, image, config)
        return {"latex": latex}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    """Endpoint for predicting from an uploaded image file"""
    global model, tokenizer, config
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and validate the file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents)).convert('L')
        
        # Get prediction
        latex = predict_image(model, tokenizer, image, config)
        return {"latex": latex}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


