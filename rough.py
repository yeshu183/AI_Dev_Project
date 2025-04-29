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
from utils.util_classes import *
from utils.util_model_classes import *
from utils.utils_test import *
from utils.utils_train import *
import io

checkpoint = torch.load("models/best_model.pth", map_location='cpu', weights_only=False)
print("Model loaded successfully")

tokenizer = checkpoint['tokenizer']
config = checkpoint['config']
config.device = 'cpu'
model = HandwrittenMathRecognizer(config, tokenizer.vocab_size)
# Load the weights
model.load_state_dict(checkpoint['model_state_dict'])
print("Weights Loaded Successfully")
image = Image.open("filtered_basic_arithmetic/test/images/expr_000852.png").convert('L')
latex = predict_image(model, tokenizer, image, config)
print(latex)