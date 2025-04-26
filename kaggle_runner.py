import os
import time
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def run_kaggle_notebook(kernel_slug, output_path, parameters=None):
    """
    Run a Kaggle kernel and download its outputs
    
    Args:
        kernel_slug (str): The slug of the kernel (username/kernel-name)
        output_path (str): Path to save the outputs
        parameters (dict, optional): Parameters to pass to the kernel
    
    Returns:
        str: Path to the downloaded artifacts
    """
    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    print(f"Starting kernel: {kernel_slug}")
    
    # Run the kernel with optional parameters
    if parameters:
        run = api.kernel_run(kernel_slug, parameters=parameters)
    else:
        run = api.kernel_run(kernel_slug)
    
    run_id = run['id']
    print(f"Kernel run initiated with ID: {run_id}")
    
    # Monitor the kernel run
    status = 'running'
    while status in ['running', 'queued', 'starting']:
        print(f"Current status: {status}. Checking again in 30 seconds...")
        time.sleep(30)
        kernel_status = api.kernel_status(run_id)
        status = kernel_status['status']
    
    if status != 'complete':
        raise Exception(f"Kernel failed with status: {status}")
    
    print(f"Kernel completed successfully. Downloading artifacts to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    api.kernel_output(run_id, path=output_path)
    
    return output_path

def upload_dataset_to_kaggle(dataset_path, dataset_name, version_notes=None):
    """
    Upload a dataset to Kaggle
    
    Args:
        dataset_path (str): Path to the dataset directory
        dataset_name (str): Name of the dataset on Kaggle
        version_notes (str, optional): Notes for this version
    
    Returns:
        str: The dataset reference
    """
    api = KaggleApi()
    api.authenticate()
    
    if not version_notes:
        version_notes = f"Updated dataset {dataset_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Create or update the dataset
    api.dataset_create_version(
        folder=dataset_path,
        version_notes=version_notes,
        slug=dataset_name,
        convert_to_csv=False,
        delete_old_versions=False
    )
    
    username = api.get_config_value('username')
    return f"{username}/{dataset_name}"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kaggle notebook and download artifacts")
    parser.add_argument("--kernel", required=True, help="Kernel slug (username/kernel-name)")
    parser.add_argument("--output", default="./model_artifacts", help="Output directory for artifacts")
    parser.add_argument("--params", help="JSON string of parameters to pass to the kernel")
    
    args = parser.parse_args()
    
    parameters = None
    if args.params:
        parameters = json.loads(args.params)
    
    run_kaggle_notebook(args.kernel, args.output, parameters)