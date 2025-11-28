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
    save_if_enabled,
)
from point_line_edge_detection.scripts.point_detection.process_image import (
    process_image,
)

# %% -------- Functions --------
def run_neural_detection(img, kernel_size=3):
    """Run the neural isolated-point detector and return results + timing."""
    start_time = time.time()

    filtered_image, filter_response = detect_isolated_points(
        img,
        excite_sum_num=1,
        inhib_sum_num=0,
        kernel_size=kernel_size,
    )

    execution_time = time.time() - start_time
    num_isolated = np.count_nonzero(filter_response)

    return filtered_image, filter_response, execution_time, num_isolated

def run_laplacian_detection(img, kernel, use_manual=True, manual_ratio=0.7):
    """Apply Laplacian filter and compute a binary anomaly map."""
    ddepth = cv2.CV_16S

    # Apply Laplacian
    dst = cv2.filter2D(img, ddepth=ddepth, kernel=kernel)
    abs_dst = np.abs(dst)

    # Manual threshold
    manual_threshold = int(manual_ratio * np.max(abs_dst))

    # Otsu threshold
    _, otsu_threshold = cv2.threshold(
        np.uint8(abs_dst),
        0,
        abs_dst.max(),
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    threshold_used = manual_threshold if use_manual else otsu_threshold

    laplacian_binary = (abs_dst > threshold_used).astype(int)
    num_detected = int(np.sum(laplacian_binary))

    return laplacian_binary, num_detected, threshold_used

def run_laplacian_otsu_only(img, kernel):
    """Laplacian filtered output using only Otsu threshold on normalized output."""
    ddepth = cv2.CV_16S

    dst = cv2.filter2D(img, ddepth=ddepth, kernel=kernel)
    dst_8u = np.uint8(255 * dst / np.max(dst))

    _, otsu_threshold = cv2.threshold(
        dst_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    otsu_binary = (dst_8u > otsu_threshold).astype(int)
    num_detected = int(np.sum(otsu_binary))

    return dst_8u, otsu_binary, num_detected

# %% -------- Parameters --------
KERNEL_SIZE = 3
BINARY_IMAGE_FLAG = True
IMAGE_SAVE_SWITCH = False

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
}

selected_image_index = 3
img_name = IMAGE_OPTIONS[selected_image_index]

if img_name is None:
    raise ValueError(f"Invalid image index: {selected_image_index}")

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
filtered_image, filter_response, execution_time, num_isolated = run_neural_detection(
    img, kernel_size=KERNEL_SIZE
)

print(f"Execution time: {execution_time:.4f} seconds")
print(f"Number of isolated pixels detected by neural model: {num_isolated}")

show_plt_images(img, "Original image", filtered_image, "Filtered image")
show_plt_images(img, "Original image", filter_response, "Anomaly Response Pixels")

save_if_enabled(IMAGE_SAVE_SWITCH, filter_response, image_save_path, img_name, prefix="neuron_")

# %% ============================================================
#                 Laplacian Detection (Manual or Otsu)
# ===============================================================
laplacian_binary, num_laplacian, threshold_used = run_laplacian_detection(
    img, LAPLACIAN_KERNEL, use_manual=True, manual_ratio=0.7
)

print(f"Using threshold: {threshold_used}")
print(f"Laplacian isolated pixels: {num_laplacian}")

show_plt_images(img, "Original image", laplacian_binary, "Laplacian Response")

save_if_enabled(
    IMAGE_SAVE_SWITCH, laplacian_binary, image_save_path, img_name, prefix="laplacian_"
)

# %% ============================================================
#                 Otsu-only Laplacian
# ===============================================================

dst_8u, otsu_output, num_otsu = run_laplacian_otsu_only(img, LAPLACIAN_KERNEL)

print(f"Otsu-only Laplacian isolated pixels: {num_otsu}")

fig = plt.figure(figsize=(20, 8))
plt.gray()
plt.subplot(1, 2, 1).imshow(dst_8u)
plt.subplot(1, 2, 2).imshow(otsu_output)
plt.show()

# %% NOTE:
# Laplacian output remains highly sensitive to threshold selection
# and still produces false positives even in simple scenarios.