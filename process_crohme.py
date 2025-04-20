import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import json

def setup_logging(output_dir):
    log_path = os.path.join(output_dir, 'processing_log.txt')
    log_file = open(log_path, 'w')
    
    def write_log(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        log_file.write(log_entry)
        print(log_entry.strip())
        log_file.flush()
    
    return write_log, log_file

def process_inkml(inkml_file_path, output_image_path):
    # Parse the InkML file
    tree = ET.parse(inkml_file_path)
    root = tree.getroot()
    
    # Define XML namespace
    ns = {'inkml': 'http://www.w3.org/2003/InkML'}
    
    # Extract all traces
    traces = []
    for trace_tag in root.findall('.//inkml:trace', ns):
        trace_str = trace_tag.text.strip()
        points = []
        for point_str in trace_str.split(','):
            parts = point_str.strip().split()
            if len(parts) >= 2:
                x, y = map(float, parts[:2])
                points.append((x, y))
        if points:
            traces.append(points)
    
    if not traces:
        raise ValueError("No traces found in the InkML file")
    
    # Find bounding box
    min_x = min(point[0] for trace in traces for point in trace)
    max_x = max(point[0] for trace in traces for point in trace)
    min_y = min(point[1] for trace in traces for point in trace)
    max_y = max(point[1] for trace in traces for point in trace)
    
    # Add padding
    padding = 20
    width = int(max_x - min_x + 2 * padding)
    height = int(max_y - min_y + 2 * padding)
    
    # Create blank image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw traces
    for trace in traces:
        # Normalize points
        norm_trace = [(x - min_x + padding, y - min_y + padding) for x, y in trace]
        
        # Draw lines connecting the points
        if len(norm_trace) > 1:
            for i in range(len(norm_trace) - 1):
                draw.line([norm_trace[i], norm_trace[i+1]], fill='black', width=2)
    
    # Save image
    img.save(output_image_path)
    
    return img

def extract_latex_from_inkml(inkml_file_path):
    tree = ET.parse(inkml_file_path)
    root = tree.getroot()
    
    # Define XML namespace
    ns = {'inkml': 'http://www.w3.org/2003/InkML'}
    
    # Find LaTeX annotation
    annotation_paths = [
        './/inkml:annotation[@type="truth"]',
        './/inkml:annotation[@type="LaTeX"]',
        './/inkml:annotation'
    ]
    
    for path in annotation_paths:
        elements = root.findall(path, ns)
        for element in elements:
            if element.text and (
                element.get('type') in ['truth', 'LaTeX'] or 
                (element.text and ('math' in element.text or '\\' in element.text))
            ):
                return element.text.strip()
    
    return None

def process_crohme_dataset(input_dir, output_dir):
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    annotations_dir = os.path.join(output_dir, 'annotations')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Set up logging
    write_log, log_file = setup_logging(output_dir)
    
    try:
        # Find all inkml files
        inkml_files = glob.glob(os.path.join(input_dir, '**', '*.inkml'), recursive=True)
        write_log(f"Found {len(inkml_files)} InkML files to process")
        
        # Create a dataset index
        dataset_index = []
        
        # Process each file
        successful = 0
        for idx, inkml_path in enumerate(inkml_files):
            try:
                # Get filename without extension
                base_name = os.path.splitext(os.path.basename(inkml_path))[0]
                
                # Output paths
                image_path = os.path.join(images_dir, f"{base_name}.png")
                annotation_path = os.path.join(annotations_dir, f"{base_name}.txt")
                
                # Process image
                process_inkml(inkml_path, image_path)
                
                # Extract LaTeX
                latex = extract_latex_from_inkml(inkml_path)
                
                # Save LaTeX annotation
                if latex:
                    with open(annotation_path, 'w') as f:
                        f.write(latex)
                    
                    # Add to dataset index
                    dataset_index.append({
                        'id': base_name,
                        'image_path': os.path.relpath(image_path, output_dir),
                        'annotation_path': os.path.relpath(annotation_path, output_dir),
                        'latex': latex
                    })
                    
                    successful += 1
                else:
                    write_log(f"WARNING: No LaTeX annotation found for {base_name}")
                
                write_log(f"[{idx+1}/{len(inkml_files)}] Processed {base_name}")
            
            except Exception as e:
                write_log(f"ERROR processing {inkml_path}: {str(e)}")
        
        # Save dataset index
        index_path = os.path.join(output_dir, 'dataset_index.json')
        with open(index_path, 'w') as f:
            json.dump(dataset_index, f, indent=2)
        
        write_log(f"Processing complete. Successfully processed {successful}/{len(inkml_files)} files.")
    
    finally:
        log_file.close()

# Example usage
if __name__ == "__main__":
    # Update these paths to match your environment
    input_dir = 'path/to/CROHME_dataset'
    output_dir = '\processed_data'
    
    process_crohme_dataset(input_dir, output_dir)