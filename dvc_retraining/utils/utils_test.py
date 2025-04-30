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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
from utils.util_classes import *
from utils.util_model_classes import *

def clean_latex(latex_string):
    """Clean LaTeX expressions by removing unwanted duplicate symbols"""
    # List of symbols that shouldn't be repeated
    no_repeat_symbols = [
        '+', '=', '^', '-', '*', '/', '_', '.',
        '>', '<', '!', '(', ')', '[', ']', '{', '}', '|'
    ]
    
    # LaTeX commands that shouldn't be repeated
    no_repeat_latex_commands = [
        '\\times', '\\div', '\\rightarrow', '\\leftarrow', '\\Rightarrow', '\\Leftarrow',
        '\\leq', '\\geq', '\\approx', '\\sim', '\\cong', '\\neq', '\\|', '\\langle', '\\rangle'
    ]
    
    # First remove all whitespace
    cleaned = re.sub(r'\s+', '', latex_string)
    
    # Process one character symbols
    for sym in no_repeat_symbols:
        # Replace repeated symbols with a single instance
        # This simpler approach avoids regex escaping issues
        while sym + sym in cleaned:
            cleaned = cleaned.replace(sym + sym, sym)
    
    # Process LaTeX commands
    for cmd in no_repeat_latex_commands:
        # Replace repeated commands with a single instance
        while cmd + cmd in cleaned:
            cleaned = cleaned.replace(cmd + cmd, cmd)
    
    return cleaned.strip()



def calculate_metrics(predictions, targets, tokenizer):
    """Calculate evaluation metrics"""
    # Convert token indices to LaTeX strings
    pred_latex = [tokenizer.decode(pred.tolist()) for pred in predictions]
    target_latex = [tokenizer.decode(target.tolist()) for target in targets]
    
    # Clean LaTeX expressions
    pred_latex_clean = [clean_latex(latex) for latex in pred_latex]
    target_latex_clean = [clean_latex(latex) for latex in target_latex]
    
    # Calculate exact match accuracy with cleaned expressions
    exact_matches = sum(pred == target for pred, target in zip(pred_latex_clean, target_latex_clean))
    exact_match_accuracy = exact_matches / len(pred_latex) if len(pred_latex) > 0 else 0
    
    # Calculate token accuracy
    total_tokens = 0
    correct_tokens = 0
    
    for pred, target in zip(predictions, targets):
        # Find end index (based on END token or max sequence length)
        end_idx = tokenizer.config.max_seq_len
        for i, token in enumerate(target):
            if token.item() == tokenizer.token2idx[tokenizer.config.special_tokens['END']]:
                end_idx = i + 1
                break
        
        # Count correct tokens up to end index
        min_len = min(len(pred), end_idx)
        total_tokens += end_idx
        correct_tokens += (pred[:min_len] == target[:min_len]).sum().item()
    
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    # Calculate BLEU score
    smoother = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, target in zip(pred_latex_clean, target_latex_clean):
        # Convert strings to lists of characters for character-level BLEU
        pred_chars = list(pred)
        target_chars = list(target)
        
        # Calculate BLEU score with smoothing
        try:
            score = sentence_bleu([target_chars], pred_chars, smoothing_function=smoother)
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)
    
    # Average BLEU score
    bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    # Generate 5 random indices without replacement
    total_samples = len(pred_latex)
    random_indices = np.random.choice(total_samples, size=min(5, total_samples), replace=False)
    
    return {
        'exact_match': exact_match_accuracy,
        'token_accuracy': token_accuracy,
        'bleu': bleu_score,
        'pred_examples': [pred_latex_clean[i] for i in random_indices],  # Random sample predictions
        'target_examples': [target_latex_clean[i] for i in random_indices]  # Corresponding targets
    }

def evaluate(model, dataloader, criterion, tokenizer, config):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get batch data
            images = batch['image'].to(config.device)
            targets = batch['latex_tokens'].to(config.device)
            
            # Forward pass for loss calculation
            outputs = model(images, targets, teacher_forcing_ratio=0.0)
            
            # Reshape for loss calculation
            outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
            targets_flat = targets.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            # Generate predictions for accuracy calculation
            predictions = model.generate(images)
            
            # Store predictions and targets
            all_predictions.extend(predictions.detach().cpu())
            all_targets.extend(targets.detach().cpu())
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets, tokenizer)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def test_model(model, tokenizer, config, test_loader=None):
    """Test the model on the test set"""
    if test_loader is None:
        # Create test dataset and loader
        test_dataset = CROHMEDataset(config.data_root, tokenizer, config, split='test')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True
        )
    else:
        print("Test Loader Provided")
    # Evaluate
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx[config.special_tokens['PAD']])
    metrics = evaluate(model, test_loader, criterion, tokenizer, config)
    
    print("Test Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Exact Match: {metrics['exact_match']:.4f}")
    print(f"  Token Accuracy: {metrics['token_accuracy']:.4f}")
    
    # Sample predictions
    print("Sample predictions:")
    for i in range(min(5, len(metrics['pred_examples']))):
        print(f"  Pred: {metrics['pred_examples'][i]}")
        print(f"  True: {metrics['target_examples'][i]}")
        print()
    
    return metrics

def predict_image(model, tokenizer, image, config):
    """Predict LaTeX for a single image"""
    if image.mode != 'L':
        image = image.convert('L')
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = transform(image).unsqueeze(0).to(config.device)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        prediction = model.generate(image)
        latex = tokenizer.decode(prediction[0].tolist())
    
    return clean_latex(latex)

