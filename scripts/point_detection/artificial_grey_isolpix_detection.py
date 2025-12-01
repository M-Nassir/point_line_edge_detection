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
import numpy as np
from PIL import Image

from point_line_edge_detection.scripts.point_detection.functions import (
    show_plt_images,
    load_paths,
    save_if_enabled,
    run_neural_detection,
    run_laplacian_detection,
)
from point_line_edge_detection.scripts.point_detection.process_image import (
    load_and_preprocess_image,
)

# %% -------- Parameters --------
KERNEL_SIZE = 3
BINARY_IMAGE_FLAG = True   
IMAGE_SAVE_SWITCH = False  # Set to True to save images

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

img, binary_image_flag, img_title = load_and_preprocess_image(img_name, data_path)

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
    img, LAPLACIAN_KERNEL, use_manual_threshold=True, manual_ratio=0.7
)

print(f"Laplacian threshold used: {threshold_used}")
print(f"Number of Laplacian isolated pixels: {num_laplacian}")

show_plt_images(img, "Original image", laplacian_binary, "Laplacian Response")

save_if_enabled(IMAGE_SAVE_SWITCH, laplacian_binary, image_save_path, img_name, prefix="laplacian_")

# %% ============================================================
#                 Otsu-only Laplacian
# ===============================================================

otsu_output, num_otsu, threshold_used = run_laplacian_detection(
    img, LAPLACIAN_KERNEL, use_manual_threshold=False, manual_ratio=0.7
)

print(f"Otsu-Laplacian threshold used: {threshold_used}")
print(f"Number of Otsu-Laplacian isolated pixels: {num_otsu}")

show_plt_images(img, "Original image", otsu_output, "Otsu-Laplacian Response")

save_if_enabled(IMAGE_SAVE_SWITCH, otsu_output, image_save_path, img_name, prefix="otsu-laplacian_")

# %% NOTE:
# Laplacian output remains highly sensitive to threshold selection
# and still produces false positives even in simple scenarios. It also struggles to detect isolated pixels.