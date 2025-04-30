#!/usr/bin/env python
"""
Script to evaluate the performance of a retrained handwritten math recognition model.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Import utilities from your backend
import sys
from utils.util_classes import *
from utils.util_model_classes import *
from utils.utils_train import *
from utils.utils_test import *

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--status-file', required=True, help='JSON file containing feedback status')
    parser.add_argument('--model-path', required=True, help='Path to the trained model')
    parser.add_argument('--test-data', required=True, help='Directory containing test data')
    args = parser.parse_args()
    
    # Load status
    with open(args.status_file, 'r') as f:
        status = json.load(f)
    
    metrics = {}
    
    if not status.get('should_retrain', False):
        print("Model was not retrained. Skipping detailed evaluation.")
        metrics = {
            "retrained": False,
            "message": "Evaluation skipped because model was not retrained",
            "timestamp": "2025-04-30"
        }
    else:
        print("Model was retrained. Running evaluation...")
        try:
            # Load model checkpoint
            checkpoint = torch.load(args.model_path, map_location='cpu')
            
            # Extract components from checkpoint
            tokenizer = checkpoint['tokenizer']
            config = checkpoint['config']
            
            # Set device and test data path
            config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            config.data_root = args.test_data
            
            # Initialize model
            model = HandwrittenMathRecognizer(config, tokenizer.vocab_size)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(config.device)
            
            # Create test dataset and loader
            test_dataset = CROHMEDataset(config.data_root, tokenizer, config, split=None)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config.batch_size, 
                shuffle=False,
                pin_memory=True
            )
            
            # Evaluate model
            eval_metrics = test_model(model, tokenizer, config, test_loader)
            
            # Create output metrics
            metrics = {
                "retrained": True,
                "loss": float(eval_metrics['loss']),
                "exact_match": float(eval_metrics['exact_match']),
                "token_accuracy": float(eval_metrics['token_accuracy']),
                "bleu": float(eval_metrics['bleu']),
                "timestamp": "2025-04-30"
            }
            
            # If available, add training metrics history from the checkpoint
            if 'metrics_history' in checkpoint:
                metrics['training_history'] = checkpoint['metrics_history']
            
            # Generate and save sample visualizations
            if len(eval_metrics['pred_examples']) > 0:
                samples = []
                for i in range(min(5, len(eval_metrics['pred_examples']))):
                    samples.append({
                        "prediction": eval_metrics['pred_examples'][i],
                        "ground_truth": eval_metrics['target_examples'][i]
                    })
                metrics['samples'] = samples
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            metrics = {
                "retrained": True,
                "error": str(e),
                "message": "Evaluation failed due to an error",
                "timestamp": "2025-04-30"
            }
    
    # Save metrics to file
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to metrics.json")
    
    # Print a summary of the results
    if metrics.get('retrained', False) and 'error' not in metrics:
        print("\nEvaluation Results Summary:")
        print(f"Exact Match Accuracy: {metrics.get('exact_match', 0):.4f}")
        print(f"Token Accuracy: {metrics.get('token_accuracy', 0):.4f}")
        print(f"BLEU Score: {metrics.get('bleu', 0):.4f}")
        print(f"Loss: {metrics.get('loss', 0):.4f}")

if __name__ == "__main__":
    main()