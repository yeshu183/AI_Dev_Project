import os
import glob
import numpy as np
import cv2
from xml.etree import ElementTree as ET
import shutil
from tqdm import tqdm
import pickle
import re
import matplotlib.pyplot as plt

class CROHMEProcessor:
    def __init__(self, crohme_dir, output_dir):
        """
        Initialize the CROHME dataset processor
        
        Args:
            crohme_dir: Directory containing the CROHME dataset (inkml files)
            output_dir: Directory to save processed data
        """
        self.crohme_dir = crohme_dir
        self.output_dir = output_dir
        
        # Create output directories if they don't exist
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Define which symbols we're interested in (digits and basic operators)
        self.target_symbols = set([str(i) for i in range(10)] + ['+', '-', '*', '/', '(', ')'])
        
    def _parse_inkml(self, inkml_file):
        """
        Parse an inkml file and extract strokes and ground truth
        
        Args:
            inkml_file: Path to the inkml file
            
        Returns:
            strokes: List of strokes, each stroke is a list of points
            label: Ground truth label for the expression
        """
        tree = ET.parse(inkml_file)
        root = tree.getroot()
        
        # Extract namespace
        ns = {'ns': root.tag.split('}')[0].strip('{')}
        
        # Extract all strokes (traces)
        strokes = []
        for trace in root.findall('ns:trace', ns):
            stroke = []
            trace_str = trace.text.strip()
            for point_str in trace_str.split(','):
                x, y = map(float, point_str.strip().split())
                stroke.append((x, y))
            strokes.append(stroke)
        
        # Extract ground truth
        label = None
        for annotation in root.findall('.//ns:annotation', ns):
            if annotation.get('type') == 'truth':
                label = annotation.text.strip()
                break
                
        # Also try to extract LaTeX representation if available
        latex = None
        for annotation in root.findall('.//ns:annotation', ns):
            if annotation.get('type') == 'LaTeX':
                latex = annotation.text.strip()
                break
        
        return strokes, label, latex
    
    def _normalize_strokes(self, strokes):
        """
        Normalize strokes to fit in a standard canvas
        
        Args:
            strokes: List of strokes
            
        Returns:
            normalized_strokes: Normalized strokes
        """
        # Find min and max coordinates
        all_points = [point for stroke in strokes for point in stroke]
        if not all_points:
            return []
            
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        
        # Calculate scale factor to fit in a 256x256 canvas
        width = max_x - min_x
        height = max_y - min_y
        scale = min(240 / width, 240 / height) if width > 0 and height > 0 else 1
        
        # Normalize strokes
        normalized_strokes = []
        for stroke in strokes:
            normalized_stroke = []
            for x, y in stroke:
                # Scale and translate to center
                new_x = (x - min_x) * scale + 8
                new_y = (y - min_y) * scale + 8
                normalized_stroke.append((new_x, new_y))
            normalized_strokes.append(normalized_stroke)
            
        return normalized_strokes
    
    def _strokes_to_image(self, strokes, size=(256, 256)):
        """
        Convert strokes to an image
        
        Args:
            strokes: List of strokes
            size: Size of the output image
            
        Returns:
            image: The rendered image
        """
        image = np.ones(size, dtype=np.uint8) * 255
        
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                pt1 = (int(stroke[i][0]), int(stroke[i][1]))
                pt2 = (int(stroke[i+1][0]), int(stroke[i+1][1]))
                cv2.line(image, pt1, pt2, 0, 2)
                
        return image
    
    def _is_basic_arithmetic(self, label):
        """
        Check if an expression contains only basic arithmetic operations
        
        Args:
            label: The expression label
            
        Returns:
            bool: True if the expression is basic arithmetic
        """
        # Filter out expressions with complex symbols
        for char in label:
            if char not in self.target_symbols and not char.isspace():
                return False
        return True
    
    def process_file(self, inkml_file, idx):
        """
        Process a single inkml file
        
        Args:
            inkml_file: Path to the inkml file
            idx: Index for naming the output file
            
        Returns:
            success: Whether processing was successful
            label: The ground truth label
        """
        try:
            strokes, label, latex = self._parse_inkml(inkml_file)
            
            # Skip if no label was found
            if label is None:
                return False, None
                
            # Skip if not basic arithmetic (optional)
            # if not self._is_basic_arithmetic(label):
            #     return False, None
            
            normalized_strokes = self._normalize_strokes(strokes)
            if not normalized_strokes:
                return False, None
                
            image = self._strokes_to_image(normalized_strokes)
            
            # Save image
            image_path = os.path.join(self.images_dir, f'expr_{idx:06d}.png')
            cv2.imwrite(image_path, image)
            
            # Save label
            label_path = os.path.join(self.labels_dir, f'expr_{idx:06d}.txt')
            with open(label_path, 'w') as f:
                f.write(label)
                if latex:
                    f.write('\n' + latex)
            
            return True, label
        except Exception as e:
            #print(f"Error processing {inkml_file}: {e}")
            return False, None
    
    def process_dataset(self):
        """
        Process the entire dataset
        
        Returns:
            data_info: Dictionary with dataset information
        """
        inkml_files = glob.glob(os.path.join(self.crohme_dir, '**', '*.inkml'), recursive=True)
        print(f"Found {len(inkml_files)} inkml files")
        
        success_count = 0
        processed_labels = []
        
        for idx, inkml_file in enumerate(tqdm(inkml_files)):
            success, label = self.process_file(inkml_file, idx)
            if success:
                success_count += 1
                processed_labels.append(label)
        
        # Create a data info file
        data_info = {
            'total_files': len(inkml_files),
            'processed_files': success_count,
            'symbol_statistics': self._compute_symbol_stats(processed_labels)
        }
        
        # Save data info
        with open(os.path.join(self.output_dir, 'data_info.pkl'), 'wb') as f:
            pickle.dump(data_info, f)
            
        # Create train/val/test splits
        self._create_data_splits()
        
        return data_info
    
    def _compute_symbol_stats(self, labels):
        """
        Compute statistics of symbols in the dataset
        
        Args:
            labels: List of expression labels
            
        Returns:
            stats: Dictionary with symbol statistics
        """
        stats = {}
        for label in labels:
            for char in label:
                if char.isspace():
                    continue
                stats[char] = stats.get(char, 0) + 1
        return stats
    
    def _create_data_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Create train/val/test splits
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
        """
        image_files = glob.glob(os.path.join(self.images_dir, '*.png'))
        np.random.shuffle(image_files)
        
        num_files = len(image_files)
        train_end = int(num_files * train_ratio)
        val_end = train_end + int(num_files * val_ratio)
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Create split directories and copy files
        for split_name, files in splits.items():
            split_dir = os.path.join(self.output_dir, split_name)
            split_img_dir = os.path.join(split_dir, 'images')
            split_label_dir = os.path.join(split_dir, 'labels')
            
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_label_dir, exist_ok=True)
            
            for img_path in files:
                filename = os.path.basename(img_path)
                basename = os.path.splitext(filename)[0]
                
                # Copy image
                shutil.copy(img_path, os.path.join(split_img_dir, filename))
                
                # Copy label
                label_path = os.path.join(self.labels_dir, f'{basename}.txt')
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(split_label_dir, f'{basename}.txt'))
        
        # Create split info file
        split_info = {
            'train_size': len(splits['train']),
            'val_size': len(splits['val']),
            'test_size': len(splits['test'])
        }
        
        with open(os.path.join(self.output_dir, 'split_info.pkl'), 'wb') as f:
            pickle.dump(split_info, f)
    
    def visualize_samples(self, num_samples=5):
        """
        Visualize some samples from the dataset
        
        Args:
            num_samples: Number of samples to visualize
        """
        image_files = glob.glob(os.path.join(self.images_dir, '*.png'))
        np.random.shuffle(image_files)
        
        plt.figure(figsize=(15, 3*num_samples))
        
        for i, img_path in enumerate(image_files[:num_samples]):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_dir, f'{basename}.txt')
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            label = "No label found"
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = f.readline().strip()
            
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Expression: {label}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_visualizations.png'))
        plt.close()

def filter_basic_arithmetic(input_dir, output_dir):
    """
    Filter dataset to keep only basic arithmetic expressions
    
    Args:
        input_dir: Directory with processed data
        output_dir: Directory to save filtered data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define allowed characters for basic arithmetic
    allowed_chars = set('0123456789+-*/() ')
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Process each split
    stats = {'total': 0, 'filtered': 0}
    
    for split in ['train', 'val', 'test']:
        label_dir = os.path.join(input_dir, split, 'labels')
        image_dir = os.path.join(input_dir, split, 'images')
        
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        
        for label_path in label_files:
            stats['total'] += 1
            
            # Read label
            with open(label_path, 'r') as f:
                label = f.readline().strip()
            
            # Check if it's basic arithmetic
            if all(c in allowed_chars for c in label):
                stats['filtered'] += 1
                
                # Copy label and image
                basename = os.path.basename(label_path)
                image_name = os.path.splitext(basename)[0] + '.png'
                
                shutil.copy(label_path, os.path.join(output_dir, split, 'labels', basename))
                shutil.copy(os.path.join(image_dir, image_name), 
                            os.path.join(output_dir, split, 'images', image_name))
    
    print(f"Filtered {stats['filtered']} basic arithmetic expressions out of {stats['total']} total")
    return stats

def main():
    # Example usage
    crohme_dir = "path/to/CROHME_dataset"  # Directory with inkml files
    output_dir = "path/to/processed_data"  # Output directory for processed data
    
    processor = CROHMEProcessor(crohme_dir, output_dir)
    data_info = processor.process_dataset()
    
    print(f"Processed {data_info['processed_files']} files out of {data_info['total_files']}")
    print("Symbol statistics:")
    for symbol, count in sorted(data_info['symbol_statistics'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {count}")
    
    # Visualize some samples
    processor.visualize_samples(5)
    
    # Filter to keep only basic arithmetic expressions (optional)
    filtered_dir = os.path.join(output_dir, "filtered_basic_arithmetic")
    filter_stats = filter_basic_arithmetic(output_dir, filtered_dir)

if __name__ == "__main__":
    main()