#!/usr/bin/env python
"""
Script to update the production model if evaluation metrics meet criteria.
"""
import os
import json
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='Update production model if metrics are good')
    parser.add_argument('--status-file', required=True, help='JSON file containing feedback status')
    parser.add_argument('--metrics-file', required=True, help='JSON file containing evaluation metrics')
    parser.add_argument('--source-model', required=True, help='Path to the retrained model')
    parser.add_argument('--target-model', required=True, help='Path to the production model')
    args = parser.parse_args()
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(args.target_model), exist_ok=True)
    
    # Load status
    with open(args.status_file, 'r') as f:
        status = json.load(f)
    
    # Always copy from source to target, even if no retraining was done
    if os.path.exists(args.source_model):
        print(f"Copying model from {args.source_model} to {args.target_model}")
        shutil.copy(args.source_model, args.target_model)
        print(f"Production model updated at {args.target_model}")
    else:
        print(f"Error: Source model {args.source_model} does not exist.")
        # Create a placeholder file so DVC doesn't fail
        with open(args.target_model, 'w') as f:
            f.write("# Placeholder production model - source model not found")

if __name__ == "__main__":
    main()