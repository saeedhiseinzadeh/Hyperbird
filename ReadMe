# Hyperspectral Image Processing with SAM and PlantCV

## Overview

This project processes hyperspectral images to segment and analyze specific regions of interest using HSV color space, PlantCV for hyperspectral image calibration, and SAM (Segment Anything Model) for object segmentation. The script calibrates hyperspectral images, identifies regions of interest, segments these regions, and calculates average spectral data.

## Requirements

The script requires the following Python packages:

- Python 3.6+
- OpenCV
- Spectral
- PlantCV
- Multiprocessing
- CSV
- Torch
- Matplotlib
- Pillow
- Segment Anything Model (SAM)

## Installation

Install the required packages using pip:

```sh
pip install opencv-python spectral plantcv torch matplotlib pillow segment-anything
```

## Usage

1. **Prepare your files**:
   - Place your `.raw` hyperspectral image files in the working directory.
   - Ensure you have corresponding `.hdr` files for each `.raw` file.
   - Prepare white and black reference images in ENVI format.

2. **Run the script**:
   - The script takes the following arguments:
     - `white_ref`: Path to the white reference image.
     - `black_ref`: Path to the black reference image.
     - `num_objects`: Number of objects to detect in the image.
     - `percentage`: Percentage to shrink the mask.
     - `num_processes`: Number of processes to use for multiprocessing (optional, defaults to the number of CPU cores).

   ```sh
   python script.py [white_ref] [black_ref] [num_objects] [percentage] [num_processes]
   ```

   Example:

   ```sh
   python script.py white_ref.raw black_ref.raw 5 2 4
   ```

## Detailed Steps

### 1. Initial Setup

- **Directory Setup**: The script creates directories for saving processed images and segmented data if they do not exist.

### 2. Process Each File

- **Read Hyperspectral Data**: The script reads the hyperspectral image and its corresponding HDR file.
- **Calibration**: The image is calibrated using white and black reference images.
- **RGB Conversion**: Specific bands are selected and gamma correction is applied to create an RGB image.
- **HSV Conversion**: The RGB image is converted to HSV color space.

### 3. User Input for HSV Bounds

- **Mouse Clicks**: The user clicks on the image to select points. The average HSV values of these points are used to determine the segmentation bounds.

### 4. Segmentation

- **Mask Creation**: A mask is created using the HSV bounds.
- **Contour Detection**: The script detects contours in the mask to identify regions of interest.
- **Center Calculation**: The center of each detected region is calculated and marked.

### 5. SAM Segmentation

- **SAM Model Loading**: The SAM model is loaded, and the image is segmented based on the detected centers.
- **Best Mask Selection**: The mask with the highest score is selected and shrunk by the specified percentage.

### 6. Data Calculation and Saving

- **Spectral Data Calculation**: The average spectral data is calculated for the segmented region.
- **CSV Output**: The data is saved in a CSV file with the wavelength headers extracted from the HDR file.

### 7. Multiprocessing

- **Parallel Processing**: The script uses multiprocessing to process multiple files in parallel.

## Example Output

The script generates the following outputs:
- **Segmented Images**: Saved in the `disk_images` directory.
- **Marked Images**: Saved in the `centers` directory.
- **CSV File**: Contains the average spectral data for each processed image.

## Error Handling

The script includes error handling to skip files that cannot be processed and continue with the remaining files.

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to reach out with any questions or issues you encounter while using this script. Happy processing!
