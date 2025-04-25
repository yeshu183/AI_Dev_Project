import argparse
import os
from crohme_processor import CROHMEProcessor, filter_basic_arithmetic

def main():
    parser = argparse.ArgumentParser(description='Process CROHME dataset for handwritten equation recognition')
    parser.add_argument('--crohme_dir', required=True, help='Directory containing CROHME dataset')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed data')
    parser.add_argument('--filter_basic', action='store_true', help='Filter to keep only basic arithmetic expressions')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample expressions')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the dataset
    processor = CROHMEProcessor(args.crohme_dir, args.output_dir)
    data_info = processor.process_dataset()
    
    print(f"Processed {data_info['processed_files']} files out of {data_info['total_files']}")
    
    # Visualize samples if requested
    if args.visualize:
        processor.visualize_samples(args.samples)
    
    # Filter to keep only basic arithmetic expressions if requested
    if args.filter_basic:
        filtered_dir = os.path.join(args.output_dir, "filtered_basic_arithmetic")
        filter_stats = filter_basic_arithmetic(args.output_dir, filtered_dir)

if __name__ == "__main__":
    main()