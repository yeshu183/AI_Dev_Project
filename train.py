import os
import torch
import argparse
from utils.util_classes import *
from utils.util_model_classes import *
from utils.utils_train import *

def main(args):
    # Load the full checkpoint
    checkpoint_model = torch.load(args.model_path, map_location='cpu', weights_only=False)
    print("Model loaded successfully")

    tokenizer = checkpoint_model['tokenizer']
    config = checkpoint_model['config']
    config.device = 'cpu'
    config.num_epochs = 20
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
        'config': config
    }, args.save_model_path)  # Save to the specified model path
    
    print(f"Model retrained and saved to '{args.save_model_path}'")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Retrain the Handwritten Math Recognizer model.")
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', help="Path to the model checkpoint.")
    parser.add_argument('--train-data', type=str, default='retraining data', help="Directory containing retraining data.")
    parser.add_argument('--save-model-path', type=str, default='models/best_model.pth', help="Path to save the retrained model.")
    
    args = parser.parse_args()
    main(args)
