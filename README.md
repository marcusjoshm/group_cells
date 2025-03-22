# Group Cells

This project processes microscopy images of cells expressing a fluorescent protein. It groups cell images based on their total fluorescence intensity (calculated as the sum of all pixel values) using a Gaussian Mixture Model (GMM) for clustering. The images in each group are then summed together and exported as 16-bit TIFF files.

## Features

- **Intensity Calculation:** Calculates the total fluorescence intensity (sum of pixel values) for each cell image.
- **Clustering:** Uses a Gaussian Mixture Model to group images with similar fluorescence intensities.
- **Image Summation:** Sums images in each group to create a composite image.
- **16-bit TIFF Output:** Outputs the resulting summed images as 16-bit TIFF files.

## Requirements

- Python 3.x
- numpy
- opencv-python
- scikit-learn

Install the dependencies using:

    pip install -r requirements.txt

## Usage

Run the script from the command line:

    python group_cells.py <input_directory> <output_directory> --bins <number_of_groups>

For example:

    python group_cells.py /path/to/cell_images /path/to/output --bins 3

The script will group cell images based on their fluorescence intensity, sum the images in each group, and save them with names that include the base name of the input directory and the group number (e.g., R_1_45min_bin_1_summed.tif).

## License

This project is licensed under the MIT License.
