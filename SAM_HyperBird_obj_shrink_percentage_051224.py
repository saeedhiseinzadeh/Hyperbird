"""
Saeed Hosseinzadeh
Last Edited: 05/14/2024
GrapeSpec Lab, Cornell AgriTech
Dr. Katie Gold
Contact: sh2387@cornell.edu
"""

# load the library

import sys
import os
import cv2
import cv2 as cv
import spectral as spec
import numpy as np
from plantcv import plantcv as pcv
from multiprocessing import Pool
import csv
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Enable support for ENVI files with non-lowercase parameters in the spectral library
spec.settings.envi_support_nonlowercase_params = True


# Function to shrink a binary mask by a given percentage
def shrink_mask(mask, percentage):
    mask = (mask * 255).astype(np.uint8)
    
    # Calculate the number of pixels to shrink based on the percentage
    height, width = mask.shape
    pixels_height = int(height * percentage / 100)
    pixels_width = int(width * percentage / 100)
    
    # Create a kernel with the calculated dimensions
    kernel = np.ones((pixels_height, pixels_width), np.uint8)
    
    # Erode the mask using the kernel
    shrunk_mask = cv2.erode(mask, kernel, iterations=1)
    
    # Convert back to binary mask
    shrunk_mask = (shrunk_mask > 127).astype(np.uint8)
    
    return shrunk_mask

# Function to extract wavelengths from a .hdr file
def extract_wavelengths(hdr_file_path):
    wavelengths = []
    with open(hdr_file_path, 'r') as file:
        reading_wavelengths = False
        for line in file:
            if line.strip().lower().startswith('wavelength ='):
                reading_wavelengths = True
                continue
            if reading_wavelengths and '}' in line:
                break
            if reading_wavelengths:
                values = line.replace(',', '').strip().split()
                wavelengths.extend([float(value) for value in values])
    return wavelengths

# Function to calculate the average of non-zero values in a data slice
def average_slice_excluding_zeros(slice_data):
    non_zero_values = slice_data[slice_data != 0]
    if non_zero_values.size == 0:
        return 0
    return np.mean(non_zero_values)

# Function to display a mask on an image plot
def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Function to display points on an image plot
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Function to display a bounding box on an image plot
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Initialize the bounds for HSV values
lower_bound = None
upper_bound = None
bounds_calculated = False

# Create directory to save images if it doesn't exist
output_dir = './centers'
os.makedirs(output_dir, exist_ok=True)


# Function to process a single file
def process_file(file_name, white_ref, black_ref, num_objects, percentage, disk_images):
    full_file_path = os.path.join('.', file_name)
    hdr_file_path = full_file_path.replace('.raw', '.hdr')
    png_filename = file_name.replace('.raw', '.png')
    white_ref = pcv.readimage(filename=white_ref, mode='envi')
    dark_ref = pcv.readimage(filename=black_ref, mode='envi')

    # Check if HDR file exists and PNG file does not exist
    if os.path.exists(hdr_file_path) and not os.path.exists(os.path.join(disk_images, png_filename)):
        try:
            # Read and calibrate hyperspectral data
            raw_file = pcv.readimage(full_file_path, mode='envi')
            print(f"Now Processing {file_name}")

            Calibrated_data = pcv.hyperspectral.calibrate(raw_data=raw_file, white_reference=white_ref, dark_reference=dark_ref)
            Cal_64 = Calibrated_data.array_data
            Cal_data = Cal_64.astype('float16')
            hyperspectral_image = Cal_data

            # Select specific bands for RGB channels to convert the 3dnp arry to image
            band_red = 120
            band_green = 61
            band_blue = 29

            R = hyperspectral_image[:, :, band_red]
            G = hyperspectral_image[:, :, band_green]
            B = hyperspectral_image[:, :, band_blue]

            # Apply gamma correction to RGB channels to have better image
            gamma = 2.2
            R_scaled = np.uint8(255 * ((R / R.max()) ** (1 / gamma)))
            G_scaled = np.uint8(255 * ((G / G.max()) ** (1 / gamma)))
            B_scaled = np.uint8(255 * ((B / B.max()) ** (1 / gamma)))

            brightness_factor = 1
            R_bright = np.clip(R_scaled * brightness_factor, 0, 255).astype(np.uint8)
            G_bright = np.clip(G_scaled * brightness_factor, 0, 255).astype(np.uint8)
            B_bright = np.clip(B_scaled * brightness_factor, 0, 255).astype(np.uint8)

            RGB_bright = np.stack((R_bright, G_bright, B_bright), axis=-1)


            # Save the RGB image
            img = Image.fromarray(RGB_bright)
            img.save('./temp.jpg', 'JPEG', quality=95)
            image = cv2.imread('./temp.jpg')


            # Convert the image to HSV color space
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            global bounds_calculated, lower_bound, upper_bound
            if not bounds_calculated:
                clicked_points = []

                # Mouse click event to collect HSV values from the image
                def on_mouse_click(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        if len(clicked_points) < 5:
                            clicked_points.append((x, y))
                            print(f'Point {len(clicked_points)}: ({x}, {y})')
                            if len(clicked_points) == 5:
                                cv2.destroyAllWindows()

                cv2.imshow('saeed Image', image_hsv)
                cv2.setMouseCallback('saeed Image', on_mouse_click)

                while len(clicked_points) < 5:
                    cv2.waitKey(1)

                hsv_values = [image_hsv[y, x] for (x, y) in clicked_points]
                hsv_avg = np.mean(hsv_values, axis=0)

                # Calculate the HSV bounds for segmentation
                lower_bound = np.array([max(hsv_avg[0] - 10, 0), max(hsv_avg[1] - 50, 0), max(hsv_avg[2] - 50, 0)])
                upper_bound = np.array([min(hsv_avg[0] + 10, 179), min(hsv_avg[1] + 50, 255), min(hsv_avg[2] + 50, 255)])

                print(f'lower_bound = np.array({lower_bound.tolist()})')
                print(f'upper_bound = np.array({upper_bound.tolist()})')

                bounds_calculated = True


            # Segment the image using the calculated HSV bounds
            mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # Draw circles on the detected centers
            centers = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:num_objects]:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))

            for center in centers:
                cv2.circle(image, center, 10, (0, 0, 255), -1)

            output_path = os.path.join(output_dir, f'marked_image_{file_name}.jpg')
            cv2.imwrite(output_path, image)

            input_points = np.array(centers)
            print(f"input_points = np.array({input_points.tolist()})")

            # Load SAM model and perform object segmentation
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            device = "cpu"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)

            predictor.set_image(image)

            input_point = input_points
            input_label = np.array([1] * num_objects)
            print(f"input_label = np.array({input_label.tolist()})")

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            highest_score = -1
            best_mask = None

            for mask, score in zip(masks, scores):
                if score > highest_score:
                    highest_score = score
                    best_mask = mask


            # Apply the best mask to the image
            if best_mask is not None:
                binary_mask = (best_mask > 0.5).astype(np.uint8)
                shrunk_binary_mask = shrink_mask(binary_mask, percentage)

                segmented_image = (image * np.stack([shrunk_binary_mask] * 3, axis=-1))

            save_path = os.path.join("disk_images", (str(file_name[:-4]) + ".png"))
            cv.imwrite(save_path, segmented_image)


            # Calculate and save spectral data for the segmented region
            expanded_mask = shrunk_binary_mask[:, :, np.newaxis]
            data = Cal_data * expanded_mask

            averages = [average_slice_excluding_zeros(data[:, :, i]) for i in range(data.shape[2])]
            row = [file_name[:-4]] + averages

            hdr_path = os.path.join('.', hdr_file_path)
            csv_file_path = os.path.join(os.getcwd(), os.path.basename(os.getcwd()) + '.csv')
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    headers = ['Filename'] + extract_wavelengths(hdr_path)
                    writer.writerow(headers)

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)

            print(f"Successfully processed {file_name}")
        except Exception as process_error:
            print(f"Error processing {file_name}: {process_error}. Skipping file.")
    else:
        print(f"Skipping {file_name}: Processed before or HDR file not existed")

# Main function to process all files using multiprocessing
def main(white_ref, black_ref, num_objects, percentage, num_processes):
    disk_images = 'disk_images'
    if not os.path.exists(disk_images):
        os.makedirs(disk_images)
        print(f"Created directory: {disk_images}")

    current_directory = os.getcwd()
    file_names = [f for f in os.listdir('.') if f.endswith('.raw') and f not in [white_ref, black_ref]]

    with Pool(processes=num_processes) as pool:
        pool.starmap(process_file, [(file_name, white_ref, black_ref, num_objects, percentage, disk_images) for file_name in file_names])

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py [white_ref] [black_ref] [num_objects] [percentage] [num_processes]")
        sys.exit(1)

    white_ref = sys.argv[1]
    black_ref = sys.argv[2]
    num_objects = int(sys.argv[3]) 
    percentage = float(sys.argv[4]) if len(sys.argv) > 4 else 2
    num_processes = int(sys.argv[5]) if len(sys.argv) > 5 else os.cpu_count()

    if percentage < 1 or percentage > 20:
        print(f"Warning: Threshold value {threshold} is out of range. It should be between 1 and 20 which will be percentage.")
        sys.exit(1)

    main(white_ref, black_ref, num_objects, percentage, num_processes)
