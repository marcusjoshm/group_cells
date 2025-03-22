import os
import cv2
import numpy as np
import argparse

def compute_mean_intensity(img):
    """Compute the mean pixel intensity excluding zero values."""
    nonzero_pixels = img[img > 0]
    if nonzero_pixels.size > 0:
        return nonzero_pixels.mean()
    else:
        return 0

def main(input_dir, output_dir, num_bins):
    # Get all image file names (adjust extensions as needed)
    image_files = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    
    if not image_files:
        print("No images found in directory:", input_dir)
        return

    # List to store tuples of (filename, mean_intensity, image)
    image_data = []
    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read {image_file}. Skipping.")
            continue
        mean_intensity = compute_mean_intensity(img)
        image_data.append((image_file, mean_intensity, img))

    # If no valid images were found, exit
    if not image_data:
        print("No valid images to process.")
        return

    # Sort the images by mean intensity (lowest to highest)
    image_data.sort(key=lambda x: x[1])

    # Compute intensity range among images
    intensities = [d[1] for d in image_data]
    min_intensity = min(intensities)
    max_intensity = max(intensities)

    # Create bin boundaries using equally spaced intensity values
    bins = np.linspace(min_intensity, max_intensity, num_bins + 1)
    print(f"Using bins: {bins}")

    # Prepare a list to hold the summed images for each bin
    sum_images = [None] * num_bins

    # Assign each image to a bin and add it to the corresponding sum image
    for (filename, mean_intensity, img) in image_data:
        # If the mean equals the maximum, put it in the last bin
        if mean_intensity == max_intensity:
            bin_index = num_bins - 1
        else:
            # np.searchsorted finds the index where the mean_intensity fits in bins
            bin_index = np.searchsorted(bins, mean_intensity, side='right') - 1

        if sum_images[bin_index] is None:
            sum_images[bin_index] = np.zeros_like(img, dtype=np.float64)
        sum_images[bin_index] += img.astype(np.float64)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the summed images. Optionally, you can normalize for visualization.
    for i, sum_img in enumerate(sum_images):
        if sum_img is None:
            print(f"No images in bin {i+1}")
            continue
        # Normalize summed image to the 0-255 range for saving as an 8-bit image
        norm_img = cv2.normalize(sum_img, None, 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint8)
        output_path = os.path.join(output_dir, f"bin_{i+1}_summed.png")
        cv2.imwrite(output_path, norm_img)
        print(f"Saved summed image for bin {i+1} at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Group cell images based on mean intensity and sum them."
    )
    parser.add_argument("input_dir", help="Directory containing cell images.")
    parser.add_argument("output_dir", help="Directory to save output summed images.")
    parser.add_argument("--bins", type=int, default=3, help="Number of bins to group images into (default: 3).")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.bins)
