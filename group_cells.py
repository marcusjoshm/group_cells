import os
import cv2
import numpy as np
import argparse
from sklearn.mixture import GaussianMixture

def compute_auc(img):
    """Compute the area under the histogram curve of the image.
    Here we use the integrated intensity (sum of pixel values) as a proxy for the AUC.
    """
    # Alternatively, you could compute the histogram and integrate:
    # hist, _ = np.histogram(img, bins=256, range=(0,256))
    # auc = np.sum(hist * np.arange(256))
    # return auc
    return np.sum(img)

def main(input_dir, output_dir, num_bins):
    # Get all image file names (adjust extensions as needed)
    image_files = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    
    if not image_files:
        print("No images found in directory:", input_dir)
        return

    # List to store tuples of (filename, auc, image)
    image_data = []
    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read {image_file}. Skipping.")
            continue
        auc_value = compute_auc(img)
        image_data.append((image_file, auc_value, img))
    
    if not image_data:
        print("No valid images to process.")
        return

    # Prepare an array of AUC values for GMM clustering
    auc_values = np.array([data[1] for data in image_data]).reshape(-1, 1)

    # Fit Gaussian Mixture Model to cluster images based on AUC
    gmm = GaussianMixture(n_components=num_bins, random_state=0)
    gmm.fit(auc_values)
    labels = gmm.predict(auc_values)

    # Remap cluster labels so that they are ordered by increasing AUC
    cluster_means = {}
    for label in range(num_bins):
        cluster_values = auc_values[labels == label]
        if len(cluster_values) > 0:
            cluster_means[label] = cluster_values.mean()
        else:
            cluster_means[label] = float('inf')
    sorted_clusters = sorted(cluster_means, key=lambda k: cluster_means[k])
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}

    # Initialize list to hold summed images for each group
    sum_images = [None] * num_bins

    # Group images using the mapped labels and sum them
    for idx, (filename, auc_value, img) in enumerate(image_data):
        original_label = labels[idx]
        mapped_label = label_mapping[original_label]
        if sum_images[mapped_label] is None:
            sum_images[mapped_label] = np.zeros_like(img, dtype=np.float64)
        sum_images[mapped_label] += img.astype(np.float64)

    # Extract the base name from the input directory for naming the output files
    base_name = os.path.basename(os.path.normpath(input_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Save the summed images as 16-bit TIFF files
    for i, sum_img in enumerate(sum_images):
        if sum_img is None:
            print(f"No images in group {i+1}")
            continue
        # Normalize the summed image to 0-65535 for a 16-bit image
        norm_img = cv2.normalize(sum_img, None, 0, 65535, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint16)
        output_path = os.path.join(output_dir, f"{base_name}_bin_{i+1}_summed.tif")
        cv2.imwrite(output_path, norm_img)
        print(f"Saved summed image for group {i+1} at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Group cell images based on the AUC of their intensity histogram (integrated density) using GMM clustering and sum them."
    )
    parser.add_argument("input_dir", help="Directory containing cell images.")
    parser.add_argument("output_dir", help="Directory to save output summed images.")
    parser.add_argument("--bins", type=int, default=3, help="Number of groups to cluster images into (default: 3).")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.bins)
