#!/usr/bin/env python
"""
Script to conditionally retrain the Handwritten Math Recognizer model based on feedback status.
"""
import os
import json
import torch
import argparse
import sys
from utils.util_classes import *
from utils.util_model_classes import *
from utils.utils_train import *

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Conditionally retrain the Handwritten Math Recognizer model.")
    parser.add_argument('--status-file', type=str, required=True, help="JSON file containing feedback status")
    parser.add_argument('--train-data', type=str, required=True, help="Directory containing retraining data")
    parser.add_argument('--save-model-path', type=str, required=True, help="Path to save the retrained model")
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', 
                        help="Path to the current model checkpoint to use as starting point")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    
    # Load status to check if retraining should be performed
    with open(args.status_file, 'r') as f:
        status = json.load(f)
    
    if not status['should_retrain']:
        print("Feedback threshold not met. Skipping retraining.")
        
        # If source model exists, copy it to destination to maintain pipeline continuity
        if os.path.exists(args.model_path):
            print(f"Copying existing model to {args.save_model_path} without retraining")
            import shutil
            shutil.copy(args.model_path, args.save_model_path)
        else:
            # Create a dummy model file so the DVC pipeline doesn't fail
            print(f"Creating placeholder model at {args.save_model_path}")
            with open(args.save_model_path, 'w') as f:
                f.write("# No retraining performed - feedback threshold not met")
        
        return
    
    print(f"Feedback threshold exceeded ({status['feedback_count']} > {status['threshold']}). Starting retraining...")
    
    try:
        # Load the full checkpoint
        checkpoint_model = torch.load(args.model_path, map_location='cpu', weights_only=False)
        print("Model loaded successfully")

        tokenizer = checkpoint_model['tokenizer']
        config = checkpoint_model['config']
        config.device = 'cpu'
        config.num_epochs = 2
        model = HandwrittenMathRecognizer(config, tokenizer.vocab_size)
        
        # Load the weights
        model.load_state_dict(checkpoint_model['model_state_dict'])

        print("Loading model and tokenizer...")

        # retraining process
        checkpoint_path = None
        config.checkpoint_dir = "models"
        config.log_dir = "models/logs"
        config.data_root = args.train_data  # Use the provided training data directory
        use_val = False  # Should change this to check if the folder has a val folder

        if checkpoint_path:
            # Continue training from checkpoint
            print(f"Loading checkpoint from {checkpoint_path}...")
            model, tokenizer, loaded_config = load_checkpoint(checkpoint_path)

            # Update config with loaded config
            for key, value in vars(loaded_config).items():
                if key not in ['batch_size', 'num_epochs', 'learning_rate']:
                    setattr(config, key, value)

            model, tokenizer, metrics_history = train_model(model, tokenizer, config, use_validation=use_val)
        else:
            # Train from scratch
            model, tokenizer, metrics_history = train_model(model, tokenizer, config, use_validation=use_val)
            
        # Save the updated model after retraining
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'config': config,
            'metrics_history': metrics_history,
            'retrained': True,
            'feedback_count': status['feedback_count']
        }, args.save_model_path)  # Save to the specified model path
        
        print(f"Model retrained and saved to '{args.save_model_path}'")
        
    except Exception as e:
        print(f"Error during retraining: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()