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

if __name__=="__main__":
    # Load the full checkpoint
    checkpoint_model = torch.load("models/best_model.pth", map_location='cpu', weights_only=False)
    print("Model loaded successfully")

    tokenizer = checkpoint_model['tokenizer']
    config = checkpoint_model['config']
    config.device = 'cpu'
    config.num_epochs = 2
    model = HandwrittenMathRecognizer(config, tokenizer.vocab_size)
    # Load the weights
    model.load_state_dict(checkpoint_model['model_state_dict'])

    # For demonstration purposes:
    print("Loading model and tokenizer...")

    #retraining process
    checkpoint_path = None
    config.checkpoint_dir = "models"
    config.log_dir = "models/logs"
    config.data_root = "filtered_basic_arithmetic"
    if checkpoint_path:
            # Continue training from checkpoint
            print(f"Loading checkpoint from {checkpoint_path}...")
            model, tokenizer, loaded_config = load_checkpoint(checkpoint_path)
            
            # Update config with loaded config
            for key, value in vars(loaded_config).items():
                if key not in ['batch_size', 'num_epochs', 'learning_rate']:
                    setattr(config, key, value)
            
            model, tokenizer, metrics_history = train_model(model,tokenizer,config)
    else:
        # Train from scratch
        model, tokenizer,metrics_history = train_model(model,tokenizer,config)
