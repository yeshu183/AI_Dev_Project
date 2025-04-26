import os
import json
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

def register_model_with_mlflow(model_path, model_name, run_name=None, run_params=None):
    """
    Register a model and its artifacts with MLflow
    
    Args:
        model_path (str): Path to the directory containing model artifacts
        model_name (str): Name to register the model as
        run_name (str, optional): Name for the MLflow run
        run_params (dict, optional): Parameters to log with the run
    
    Returns:
        str: MLflow run ID
    """
    # Set MLflow tracking URI if needed
    # mlflow.set_tracking_uri("your_mlflow_server_uri")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        if run_params:
            for param_name, param_value in run_params.items():
                mlflow.log_param(param_name, param_value)
        
        # Log metrics if available
        metrics_file = os.path.join(model_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        # Log history plot if available
        history_plot = os.path.join(model_path, "training_history.png")
        if os.path.exists(history_plot):
            mlflow.log_artifact(history_plot, "plots")
        
        # Log confusion matrix if available
        confusion_matrix = os.path.join(model_path, "confusion_matrix.png")
        if os.path.exists(confusion_matrix):
            mlflow.log_artifact(confusion_matrix, "plots")
        
        # Log example predictions if available
        example_predictions = os.path.join(model_path, "example_predictions.png")
        if os.path.exists(example_predictions):
            mlflow.log_artifact(example_predictions, "plots")
        
        # Log the model if available
        model_file = os.path.join(model_path, "model.h5")
        if os.path.exists(model_file):
            mlflow.keras.log_model(
                keras_model=load_model(model_file),
                artifact_path="model",
                registered_model_name=model_name
            )
        
        # Log any additional artifacts
        for file in os.listdir(model_path):
            if file not in ["model.h5", "metrics.json", "training_history.png", 
                          "confusion_matrix.png", "example_predictions.png"]:
                file_path = os.path.join(model_path, file)
                if os.path.isfile(file_path):
                    mlflow.log_artifact(file_path, "additional_artifacts")
        
        return run.info.run_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register model with MLflow")
    parser.add_argument("--model-path", required=True, help="Path to model artifacts")
    parser.add_argument("--model-name", required=True, help="Name to register the model as")
    parser.add_argument("--run-name", help="Name for the MLflow run")
    parser.add_argument("--params", help="JSON string of parameters to log")
    
    args = parser.parse_args()
    
    run_params = None
    if args.params:
        run_params = json.loads(args.params)
    
    register_model_with_mlflow(args.model_path, args.model_name, args.run_name, run_params)