#!/usr/bin/env python
"""
Script to check feedback data and determine if retraining is needed.
"""
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Check feedback data and determine if retraining is needed')
    parser.add_argument('--feedback-dir', required=True, help='Directory containing feedback data')
    parser.add_argument('--threshold', type=int, required=True, help='Threshold count to trigger retraining')
    args = parser.parse_args()
    
    # Count feedback data files
    feedback_files = []
    if os.path.exists(args.feedback_dir):
        for root, _, files in os.walk(args.feedback_dir):
            for file in files:
                if file.endswith('.txt'):
                    feedback_files.append(os.path.join(root, file))
    
    feedback_count = len(feedback_files)
    
    # Determine if retraining is needed
    should_retrain = feedback_count >= args.threshold
    
    # Save status to JSON
    status = {
        "feedback_count": feedback_count,
        "threshold": args.threshold,
        "should_retrain": should_retrain,
        "feedback_files": feedback_files
    }
    
    # Ensure the output file is created regardless of other conditions
    with open('feedback_status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"Feedback Count: {feedback_count}")
    print(f"Threshold: {args.threshold}")
    print(f"Should retrain: {should_retrain}")
    print(f"Status saved to feedback_status.json")

if __name__ == "__main__":
    main()