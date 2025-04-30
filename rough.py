import os
import io
import uuid
import torch
import uvicorn
from PIL import Image
print("Current Working Directory:", os.getcwd())
from backend.utils.util_classes import *
from backend.utils.util_model_classes import *

checkpoint = torch.load("models/best_model.pth", map_location='cpu', weights_only=False)
print("Model loaded successfully")
print(checkpoint.keys())

tokenizer = checkpoint['tokenizer']
config = checkpoint['config']
config.device = 'cpu'

model = HandwrittenMathRecognizer(config, tokenizer.vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])

print("Loading model and tokenizer...")