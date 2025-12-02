#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolated Pixel Detection in Greyscale and Binary Images
-------------------------------------------------------

This script investigates methods for detecting isolated pixels—single-pixel
anomalies embedded within smooth greyscale regions or structured binary
patterns. Such anomalies occur in both synthetic and real-world images and are
difficult to detect reliably without strong assumptions about the surrounding
image structure.

While template matching (e.g., the hit-or-miss transform) performs well in
binary images, it becomes impractical for greyscale data due to the combinatorial
growth of possible neighbourhood configurations. Conversely, derivative-based
approaches such as Laplacian filtering suffer from noise sensitivity and
threshold dependence, making fully automated operation challenging.

To address these challenges, the script evaluates a range of classical and
a newly proposed perception-inspired method across both greyscale and binary images:

- **Hit-or-Miss Transform** – A robust template-matching method for isolated
  pixels in binary images.
- **Laplacian Derivative Filtering** – Using both manual and Otsu-based
  thresholds to identify high-frequency outliers.
- **Perception-Inspired Neuron Model ** – A parameter-free, biologically
  motivated detector based on anomaly detection and lateral inhibition 
  that identifies local intensity outliers without threshold tuning.

The methods are tested on a variety of synthetic and real images, including
smooth greyscale fields, natural photographs, and binary patterns containing
isolated noise pixels. Binary-image experiments include:
1. A synthetic black–white image with isolated pixels.
2. A real-world calculator image, where isolated white specks are to be removed.
3. An image combining one-pixel-wide lines and thicker segments to assess
   robustness in structured patterns.

Across the binary test images, all three methods—hit-and-miss, Laplacian filtering, 
and the neuron model—perform reliably. However, in the greyscale settings the 
Laplacian is threshold-sensitive, and difficult to tune to get good results, 
whereas the neuron model maintains strong performance without requiring any 
threshold tuning.
"""


# %% -------- Imports --------
import numpy as np
from PIL import Image
import cv2

from point_line_edge_detection.scripts.point_detection.functions import (
    show_plt_images,
    load_paths,
    save_if_enabled,
    run_neural_detection,
    run_laplacian_detection,
    hit_and_miss_transform,
)
from point_line_edge_detection.scripts.point_detection.process_image import (
    load_and_preprocess_image,
)

# %% -------- Parameters --------
KERNEL_SIZE = 3
BINARY_IMAGE_FLAG = True   
IMAGE_SAVE_SWITCH = False  # Set to True to save images

SMOOTH_IMAGE_FLAG = False  # Set to True to apply guassian smoothing
SMOOTHING_KERNEL_SIZE = 5  # Size of smoothing kernel

# Natural images (e.g., landscapes, portraits): 5x5 or 7x7 kernel size
# Medical images (e.g., MRI, CT scans): 3x3 or 5x5 kernel size (to preserve fine details)
# Textured images (e.g., fabrics, wood): 7x7 or 9x9 kernel size (to remove texture noise)
# Low-light images: 3x3 or 5x5 kernel size (to reduce noise amplification)

# Laplacian kernel
LAPLACIAN_KERNEL = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

# %% -------- Paths and Image Selection --------
root, data_path, image_save_path = load_paths()

# Greyscale image choices
IMAGE_OPTIONS = {
    0: "square_shades.png",            # synthetic
    1: "camera_man.png",               # real-world
    2: "turbine_blade_black_dot.png",  # real-world
    3: "mach_bands.png",               # synthetic

    # black and white images for comparison
    4: "circles_matlab.png",
    5: "calc.png",
    6: "crosses.png",

    # spatial and real-world images
    7: 'spatial_anomaly_nodes.png',
    8: 'turbine_blade_black_dot.png',
    9: 'ab1_cropped.png',
}

selected_image_index = 9
img_name = IMAGE_OPTIONS[selected_image_index]

if img_name is None:
    raise ValueError(f"Invalid image index: {selected_image_index}")

img, binary_image_flag, img_title = load_and_preprocess_image(img_name, data_path)

original_img = img.copy()

show_plt_images(img, img1_title=img_title)

if IMAGE_SAVE_SWITCH:
    Image.fromarray(img).convert('L').save(f"{image_save_path}/{img_name}")

# %% ============================================================
#                 Template Matching: Hit and Miss Transform
# ===============================================================
if np.unique(img).size > 2:
    print("Hit-and-miss transform is only applicable to binary images. Skipping this step.")
else:
    # Apply hit-and-miss transform
    h_and_miss_img = hit_and_miss_transform(img)
    numb_isolated = np.count_nonzero(h_and_miss_img)

    show_plt_images(img, "Original image", h_and_miss_img, "Hit and Miss Response")
    print(f"Number of isolated pixels located by Hit and Miss is: {numb_isolated}")

    save_if_enabled(IMAGE_SAVE_SWITCH, h_and_miss_img, image_save_path, img_name, prefix="h_&_m_")

# %% ============================================================
#                       Neural Model Detection
# ===============================================================
if SMOOTH_IMAGE_FLAG:
    img_for_neuron = cv2.GaussianBlur(img, (SMOOTHING_KERNEL_SIZE, SMOOTHING_KERNEL_SIZE), 0)
    show_plt_images(original_img, "Original image", img, "Smoothed image by Gaussian Filter: ")
else:
    img_for_neuron = img.copy()

filtered_image, filter_response, execution_time, num_isolated = run_neural_detection(
    img_for_neuron, kernel_size=KERNEL_SIZE
)

print(f"Execution time: {execution_time:.4f} seconds")
print(f"Number of isolated pixels detected by neural model: {num_isolated}")

show_plt_images(original_img, "Original image", filtered_image, "Filtered image")
show_plt_images(original_img, "Original image", filter_response, "Anomaly Response Pixels")

save_if_enabled(IMAGE_SAVE_SWITCH, filter_response, image_save_path, img_name, prefix="neuron_")

# %% ============================================================
#                 Laplacian Detection (Manual or Otsu)
# ===============================================================
laplacian_binary, lablacian_abs_response, num_laplacian, threshold_used = run_laplacian_detection(
    img, LAPLACIAN_KERNEL, use_manual_threshold=True, manual_ratio=0.7
)

print(f"Laplacian threshold used: {threshold_used}")
print(f"Number of Laplacian isolated pixels: {num_laplacian}")

show_plt_images(original_img, "Original image", lablacian_abs_response, "Laplacian Absolute Response")
show_plt_images(original_img, "Original image", laplacian_binary, "Laplacian Anomaly Response Pixels")

save_if_enabled(IMAGE_SAVE_SWITCH, laplacian_binary, image_save_path, img_name, prefix="laplacian_")

# %% ============================================================
#                 Otsu-only Laplacian
# ===============================================================

otsu_output, otsu_abs_response, num_otsu, threshold_used = run_laplacian_detection(
    img, LAPLACIAN_KERNEL, use_manual_threshold=False, manual_ratio=0.7
)

print(f"Otsu-Laplacian threshold used: {threshold_used}")
print(f"Number of Otsu-Laplacian isolated pixels: {num_otsu}")

show_plt_images(original_img, "Original image", otsu_abs_response, "Otsu-Laplacian Absolute Response")
show_plt_images(original_img, "Original image", otsu_output, "Otsu-Laplacian Anomaly Response Pixels")

save_if_enabled(IMAGE_SAVE_SWITCH, otsu_output, image_save_path, img_name, prefix="otsu-laplacian_")

# note:
# Laplacian output remains highly sensitive to threshold selection
# and still produces false positives even in simple scenarios. It also struggles to detect isolated pixels.
# otsu-method can fail in many images, e.g. the turbine blade image.

# %% ============================================================
#                       Median Filtering
# ===============================================================
# doesn't find isolated pixels, just smoothes, but gets rid of lines!

# Apply the median filter to the image and display the result
filtered_img = cv2.medianBlur(img, KERNEL_SIZE)
show_plt_images(original_img, "Original image", filtered_img, "Filtered image")

# %%
