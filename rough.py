import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from tqdm import tqdm
import re
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from utils.util_model_classes import *
from utils.util_classes import *


# Load the full checkpoint
checkpoint = torch.load("models/best_model.pth", map_location='cpu', weights_only=False)
print("Model loaded successfully")

train_model(checkpoint)
# Print the checkpoint keys to see what's available
print(f"Checkpoint contains keys: {list(checkpoint.keys())}")

# If the checkpoint contains a model state dictionary
if 'model_state_dict' in checkpoint:
    # Print number of parameters
    num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
    print(f"Model has {num_params:,} parameters")
    
    # Print some layer names
    print("\nSome model layers:")
    for i, (name, _) in enumerate(list(checkpoint['model_state_dict'].items())[:5]):
        print(f"  {i+1}. {name}")
    print("  ...")
    
    # Print a sample tensor shape
    for name, tensor in list(checkpoint['model_state_dict'].items())[:3]:
        print(f"\nLayer: {name}")
        print(f"Shape: {tensor.shape}")
        print(f"Data type: {tensor.dtype}")
        
# If the checkpoint contains a config
if 'config' in checkpoint:
    print("\nModel config:")
    for key, value in vars(checkpoint['config']).items():
        print(f"  {key}: {value}")

# If the checkpoint contains a tokenizer
if 'tokenizer' in checkpoint:
    print("\nTokenizer info:")
    print(f"  Type: {type(checkpoint['tokenizer'])}")
    if hasattr(checkpoint['tokenizer'], 'vocab_size'):
        print(f"  Vocab size: {checkpoint['tokenizer'].vocab_size}")