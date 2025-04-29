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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import tempfile
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.util_classes import *
from utils.util_model_classes import *
from utils.utils_test import *
from utils.utils_mlflow import *
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
# import wandb
# wandb.login(key="f659082c2b19bf3ffaaceceb36c1e280541f6b11")

def train_epoch(model, dataloader, criterion, optimizer, config, epoch, tokenizer):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Teacher forcing ratio with decay
    teacher_forcing_ratio = config.teacher_forcing_ratio * (config.teacher_forcing_decay ** epoch)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    for batch in progress_bar:
        images = batch['image'].to(config.device)
        targets = batch['latex_tokens'].to(config.device)
        
        # Forward pass
        outputs = model(images, targets, teacher_forcing_ratio)
        
        # Flatten for loss
        outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
        targets_flat = targets.contiguous().view(-1)
        
        # Compute loss (ignore_index already set in criterion)
        loss = criterion(outputs_flat, targets_flat)
        
        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", tf_ratio=f"{teacher_forcing_ratio:.2f}")
        
        # Store predictions and targets for metric calculation
        with torch.no_grad():
            predictions = model.generate(images)
            all_predictions.extend(predictions.detach().cpu())
            all_targets.extend(targets.detach().cpu())
    
    # Calculate train metrics
    train_metrics = calculate_metrics(all_predictions, all_targets, tokenizer)
    train_metrics['loss'] = total_loss / len(dataloader)
    
    return train_metrics


def train_model(model, tokenizer, config):
    """Main training function with metrics storage for plotting"""
    print(f"Using device: {config.device}")
    
    # Set up MLflow
    run = setup_mlflow()
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow run ID: {run.info.run_id}")
    # Log parameters
    log_params(config)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Initialize metrics storage for plotting
    metrics_history = {
        'train_loss': [],
        'train_exact_match': [],
        'train_token_accuracy': [],
        'train_bleu': [],
        'val_loss': [],
        'val_exact_match': [],
        'val_token_accuracy': [],
        'val_bleu': [],
        'epoch': []
    }
    
    # Load dataset
    print("Loading dataset...")
    
    # Create datasets
    train_dataset = CROHMEDataset(config.data_root, tokenizer, config, split='train')
    val_dataset = CROHMEDataset(config.data_root, tokenizer, config, split='val')

    mlflow.log_param("train_dataset_size", len(train_dataset))
    mlflow.log_param("val_dataset_size", len(val_dataset))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    # Limit dataset size
    train_subset = Subset(train_dataset, list(range(min(50, len(train_dataset)))))
    val_subset = Subset(val_dataset, list(range(min(50, len(val_dataset)))))
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=0
    )
    
    # Create model
    # print("Creating model...")
    # model = HandwrittenMathRecognizer(config, tokenizer.vocab_size).to(config.device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx[config.special_tokens['PAD']])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config, epoch, tokenizer)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, tokenizer, config)
        val_loss = val_metrics['loss']
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics for plotting
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['train_exact_match'].append(train_metrics['exact_match'])
        metrics_history['train_token_accuracy'].append(train_metrics['token_accuracy']) 
        metrics_history['train_bleu'].append(train_metrics['bleu'])
        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['val_exact_match'].append(val_metrics['exact_match'])
        metrics_history['val_token_accuracy'].append(val_metrics['token_accuracy'])
        metrics_history['val_bleu'].append(val_metrics['bleu'])

        # Log metrics to MLflow
        log_metrics(train_metrics, step=epoch, prefix="train_")
        log_metrics(val_metrics, step=epoch, prefix="val_")
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config.num_epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Exact Match: {train_metrics['exact_match']:.4f}")
        print(f"  Train Token Accuracy: {train_metrics['token_accuracy']:.4f}")
        print(f"  Train BLEU Score: {train_metrics['bleu']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Exact Match: {val_metrics['exact_match']:.4f}")
        print(f"  Val Token Accuracy: {val_metrics['token_accuracy']:.4f}")
        print(f"  Val BLEU Score: {val_metrics['bleu']:.4f}")
        
        # Sample predictions
        print("Sample predictions:")
        for i in range(min(3, len(val_metrics['pred_examples']))):
            print(f"  Pred: {val_metrics['pred_examples'][i]}")
            print(f"  True: {val_metrics['target_examples'][i]}")
            print()
        
        # Save metrics history to JSON
        metrics_path = os.path.join(config.log_dir, 'metrics_history.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        
        # Save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'metrics_history': metrics_history,
                'tokenizer': tokenizer,
                'config': config
            }, checkpoint_path)
            
            print(f"Saved best model checkpoint to {checkpoint_path}")
            #Logging the model 
            mlflow.pytorch.log_model(model, "best_model")
            # Log the tokenizer and config as artifacts
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                temp_name = f.name

            torch.save({'tokenizer': tokenizer, 'config': config}, temp_name)
            mlflow.log_artifact(temp_name, "tokenizer_config")
            os.unlink(temp_name)

        # Make sure to save to Kaggle's output directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)  # This is a persistent location
        # Always save latest model
        checkpoint_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'metrics_history': metrics_history,
            'tokenizer': tokenizer,
            'config': config
        }, checkpoint_path)
    print("Training complete!")
    
    # Plot and save metrics graphs
    plot_metrics(metrics_history, config.log_dir)
    
    return model, tokenizer, metrics_history

def load_checkpoint(checkpoint_path, device=None):
    """Load model from checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config and tokenizer
    config = checkpoint['config']
    tokenizer = checkpoint['tokenizer']
    
    # Create model
    model = HandwrittenMathRecognizer(config, tokenizer.vocab_size).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer, config

def plot_metrics(metrics_history, save_dir):
    """Plot training and validation metrics and save the figures"""
    import matplotlib.pyplot as plt
    
    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(save_dir, 'metrics_plots')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    
    # Create subplots for each metric
    metrics_to_plot = [
        ('loss', 'Loss'),
        ('exact_match', 'Exact Match Accuracy'),
        ('token_accuracy', 'Token Accuracy'),
        ('bleu', 'BLEU Score')
    ]
    
    for metric_key, metric_title in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Plot training and validation metrics
        train_key = f'train_{metric_key}'
        val_key = f'val_{metric_key}'
        
        plt.plot(metrics_history['epoch'], metrics_history[train_key], 'b-', label=f'Training {metric_title}')
        plt.plot(metrics_history['epoch'], metrics_history[val_key], 'r-', label=f'Validation {metric_title}')
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_title)
        plt.title(f'Training and Validation {metric_title}')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        save_path = os.path.join(metrics_dir, f'{metric_key}_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined plot with all metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric_key, metric_title) in enumerate(metrics_to_plot):
        train_key = f'train_{metric_key}'
        val_key = f'val_{metric_key}'
        
        axes[i].plot(metrics_history['epoch'], metrics_history[train_key], 'b-', label=f'Training')
        axes[i].plot(metrics_history['epoch'], metrics_history[val_key], 'r-', label=f'Validation')
        
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric_title)
        axes[i].set_title(metric_title)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    combined_save_path = os.path.join(metrics_dir, 'combined_metrics_plot.png')
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(combined_save_path, f"metrics_plots")
    plt.show()
    plt.close()
    
    print(f"Metrics plots saved to {metrics_dir}")