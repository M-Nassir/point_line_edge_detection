#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolated Pixel Detection in Greyscale Images
--------------------------------------------

This script investigates the detection of isolated pixels in greyscale images
containing smooth intensity variations and inserted anomalies (pixel values 
ranging from [0, 255]).

Conventional template matching becomes impractical due to the combinatorial 
explosion of possible templates. Derivative-based approaches can detect such 
anomalies but are threshold-sensitive and difficult to automate reliably.

This script evaluates:
- Template matching limitations
- Laplacian-based derivative filtering (manual + Otsu thresholds)
- A perception-inspired neuron model for parameter-free anomaly detection
"""

# %% -------- Imports --------
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from point_line_edge_detection.scripts.point_detection.functions import (
    detect_isolated_points,
    show_plt_images,
    load_paths,
)
from point_line_edge_detection.scripts.point_detection.process_image import (
    process_image,
)

# %% -------- Parameters --------
KERNEL_SIZE = 3
BINARY_IMAGE_FLAG = True
IMAGE_SAVE_SWITCH = False

# %% -------- Paths and Image Selection --------
root, data_path, image_save_path = load_paths()

# Greyscale image choices
IMAGE_OPTIONS = {
    0: "square_shades.png",            # synthetic
    1: "camera_man.png",               # real-world
    2: "turbine_blade_black_dot.png",  # real-world
    3: "mach_bands.png",               # synthetic
}

selected_image_index = 3

if selected_image_index not in IMAGE_OPTIONS:
    raise ValueError(f"Invalid image index: {selected_image_index}")

img_name = IMAGE_OPTIONS[selected_image_index]
img, binary_image_flag, img_title = process_image(img_name, data_path)

show_plt_images(img, img1_title=img_title)

if IMAGE_SAVE_SWITCH:
    Image.fromarray(img).convert('L').save(f"{image_save_path}/{img_name}")

# %% ============================================================
#                       Template Matching
# ===============================================================

# NOTE:
# The hit-or-miss transform works ONLY for binary images.
# Not applicable for greyscale images used in this study.

# %% ============================================================
#                       Neural Model Detection
# ===============================================================
start_time = time.time()

filtered_image, filter_response = detect_isolated_points(
    img,
    excite_sum_num=1,
    inhib_sum_num=0,
    kernel_size=KERNEL_SIZE,
)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.4f} seconds")

show_plt_images(img, "Original image", filtered_image, "Filtered image")
show_plt_images(img, "Original image", filter_response, "Anomaly Response Pixels")

num_isolated = np.count_nonzero(filter_response)
print(f"Number of isolated pixels detected by neural model: {num_isolated}")

if IMAGE_SAVE_SWITCH:
    Image.fromarray((filter_response * 255).astype("uint8")).save(
        f"{image_save_path}/neuron_{img_name}"
    )

# %% ============================================================
#                      Image Derivatives: Laplacian
# ===============================================================

# Laplacian kernel (manual)
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

ddepth = cv2.CV_16S

# Apply Laplacian via filter2D
dst = cv2.filter2D(img, ddepth=ddepth, kernel=kernel)
abs_dst = np.abs(dst)

# Manual threshold (x% of max)
manual_threshold = int(0.7 * np.max(abs_dst))

# Otsu thresholding
_, otsu_threshold = cv2.threshold(
    np.uint8(abs_dst),
    0,
    abs_dst.max(),
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

use_simple = True
threshold_used = manual_threshold if use_simple else otsu_threshold

print(f"Using threshold: {'manual' if use_simple else 'Otsu'}")

# Binary anomaly map
laplacian_binary = (abs_dst > threshold_used).astype(int)

num_laplacian = np.sum(laplacian_binary)
print(f"Laplacian isolated pixels: {num_laplacian}")

show_plt_images(img, "Original image", laplacian_binary, "Laplacian Response")

if IMAGE_SAVE_SWITCH:
    Image.fromarray((laplacian_binary * 255).astype("uint8")).save(
        f"{image_save_path}/laplacian_{img_name}"
    )

# %% ============================================================
#                 Option 2: Otsu-only Laplacian
# ===============================================================
dst = cv2.filter2D(img, ddepth=ddepth, kernel=kernel)

dst_8u = np.uint8(255 * dst / np.max(dst))

_, otsu_threshold = cv2.threshold(
    dst_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

otsu_output = (dst_8u > otsu_threshold).astype(int)

print(f"Otsu-only Laplacian isolated pixels: {np.sum(otsu_output)}")

fig = plt.figure(figsize=(20, 8))
plt.gray()
plt.subplot(1, 2, 1).imshow(dst_8u)
plt.subplot(1, 2, 2).imshow(otsu_output)
plt.show()

# NOTE:
# Laplacian output remains highly sensitive to threshold selection
# and still produces false positives even in simple scenarios.

# %%
