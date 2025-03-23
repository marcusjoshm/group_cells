import os
import cv2
import numpy as np
import argparse

def main(input_dir, output_dir):
    # Dictionary to map common prefix to list of file paths
    groups = {}

    # List all .tif files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif'):
            # Extract common prefix: text before "_bin_"
            parts = filename.split("_bin_")
            if len(parts) < 2:
                print(f"Skipping file (unexpected naming): {filename}")
                continue
            prefix = parts[0]
            filepath = os.path.join(input_dir, filename)
            groups.setdefault(prefix, []).append(filepath)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each group: read, sum, and save
    for prefix, file_list in groups.items():
        sum_img = None
        for file_path in file_list:
            # Read image in unchanged mode (to keep original bit depth)
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Unable to read {file_path}. Skipping.")
                continue
            # Initialize sum image if needed
            if sum_img is None:
                sum_img = np.zeros_like(img, dtype=np.float64)
            # Add image to the sum
            sum_img += img.astype(np.float64)
        
        if sum_img is None:
            print(f"No valid images for group {prefix}.")
            continue
        
        # Clip to the maximum for 16-bit, then convert to uint16
        sum_img = np.clip(sum_img, 0, 255).astype(np.uint8)
        output_path = os.path.join(output_dir, f"{prefix}.tiff")
        cv2.imwrite(output_path, sum_img)
        print(f"Saved summed image for group {prefix} at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add together .tif files in a directory grouped by common prefix (before '_bin_')"
    )
    parser.add_argument("input_dir", help="Directory containing .tif files to be summed")
    parser.add_argument("output_dir", help="Directory to save summed .tiff files")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
