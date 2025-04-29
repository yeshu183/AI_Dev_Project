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
import tempfile
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_classes import *
from util_model_classes import *
from utils_test import *
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

def setup_mlflow(experiment_name="handwritten-math-recognition"):
    """Set up MLflow experiment"""
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()

def log_params(config):
    """Log model parameters to MLflow"""
    # Log model architecture parameters
    mlflow.log_param("embed_dim", config.embed_dim)
    mlflow.log_param("hidden_dim", config.hidden_dim)
    mlflow.log_param("num_layers", config.num_layers)
    mlflow.log_param("dropout", config.dropout)
    mlflow.log_param("max_seq_len", config.max_seq_len)
    
    # Log training parameters
    mlflow.log_param("batch_size", config.batch_size)
    mlflow.log_param("num_epochs", config.num_epochs)
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("teacher_forcing_ratio", config.teacher_forcing_ratio)
    mlflow.log_param("teacher_forcing_decay", config.teacher_forcing_decay)
    mlflow.log_param("grad_clip", config.grad_clip)
    
    # Log image preprocessing parameters
    mlflow.log_param("img_height", config.img_height)
    mlflow.log_param("img_width", config.img_width)

def log_metrics(metrics, step=None, prefix=""):
    """Log metrics to MLflow"""
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(f"{prefix}{metric_name}", metric_value, step=step)

def log_model(model, tokenizer, config, artifact_name="model"):
    """Log PyTorch model to MLflow"""
    # Create a dictionary with model and necessary components
    model_info = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config
    }
    
    # Define a custom loader for the saved PyTorch model
    def _load_model(model_dir):
        import torch
        import os
        model_path = os.path.join(model_dir, "model.pth")
        model_config = torch.load(os.path.join(model_dir, "model_info.pth"))
        
        # Recreate model
        model = HandwrittenMathRecognizer(model_config["config"], model_config["tokenizer"].vocab_size)
        model.load_state_dict(torch.load(model_path))
        return model, model_config["tokenizer"], model_config["config"]
    
    # Save model artifacts
    with tempfile.TemporaryDirectory() as tmp_dir:
        torch.save(model.state_dict(), os.path.join(tmp_dir, "model.pth"))
        torch.save({"tokenizer": tokenizer, "config": config}, os.path.join(tmp_dir, "model_info.pth"))
        
        mlflow.pyfunc.log_model(
            artifact_path=artifact_name,
            python_model=None,
            artifacts={"model_dir": tmp_dir},
            loader_module="mlflow.pyfunc.model"
        )

def log_images(image_paths, predictions, true_labels, step=None):
    """Log sample images with predictions to MLflow"""
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
    
    for idx, (img_path, pred, true) in enumerate(zip(image_paths, predictions, true_labels)):
        plt.figure(figsize=(10, 6))
        plt.imshow(Image.open(img_path).convert('L'), cmap='gray')
        plt.title(f"Prediction: {pred}\nTrue: {true}")
        plt.axis('off')
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log figure
        mlflow.log_figure(plt.gcf(), f"sample_{idx}_epoch_{step}.png")
        plt.close()

def log_artifact(file_path):
    """Log an artifact file to MLflow"""
    mlflow.log_artifact(file_path)

def register_model_to_mlflow(model_path, model_name, tags=None):
    """Register a model in the MLflow model registry"""
    # Load the saved model
    loaded_model, tokenizer, config = load_checkpoint(model_path)
    
    # Start a new run for model registration
    with mlflow.start_run(run_name=f"register_{model_name}"):
        # Log model parameters
        log_params(config)
        
        # Log the model
        mlflow.pytorch.log_model(loaded_model, "model")
        
        # Register the model in the model registry
        registered_model = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            model_name
        )
        
        # Add tags if provided
        if tags:
            client = MlflowClient()
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=registered_model.version,
                    key=key,
                    value=value
                )
        
        print(f"Model registered with name: {model_name}, version: {registered_model.version}")
        
        # Log additional artifacts
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            torch.save({'tokenizer': tokenizer, 'config': config}, f.name)
            mlflow.log_artifact(f.name, "tokenizer_config")
            os.unlink(f.name)
    
    return registered_model

def load_model_from_registry(model_name, version=None, stage=None):
    """Load a model from the MLflow model registry"""
    client = MlflowClient()
    
    # Get the model version URI
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"
    
    # Load the PyTorch model
    loaded_model = mlflow.pytorch.load_model(model_uri)
    
    # Get the run ID for this model version
    if version is not None:
        model_version = client.get_model_version(model_name, version)
    else:
        # Get latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(versions, key=lambda x: int(x.version))
        model_version = latest_version
    
    run_id = model_version.run_id
    
    # Download the tokenizer and config
    artifacts_path = client.download_artifacts(run_id, "tokenizer_config")
    tokenizer_config = torch.load(os.path.join(artifacts_path, os.listdir(artifacts_path)[0]))
    
    return loaded_model, tokenizer_config['tokenizer'], tokenizer_config['config']