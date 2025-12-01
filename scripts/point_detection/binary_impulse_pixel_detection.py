#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:42:05 2025

Isolated Pixel Detection in Binary Images
Author: Nassir Mohammad

This script applies a series of methods for detecting isolated pixels in binary images, including:

- Hit-or-miss transform  
- Derivative-based filtering  
- Median filtering  
- The Perception Neuron (IsolPix) detection method using lateral inhibition

It operates on binary test images, including:

1. A synthetic image containing solid black and white regions with isolated pixels.  
2. A real-world image of a calculator, where the task is to detect and remove isolated white pixels.  
3. An image with both one-pixel-wide and thicker lines, used to demonstrate robustness in the presence of line segments.

The hit-or-miss transform reliably detects isolated pixels due to its template-matching approach. Similarly, the Perception Neuron (IsolPix) method shows strong performance, particularly in handling real-world noise.
"""

# %% -------- Imports --------
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
from pathlib import Path

from point_line_edge_detection.scripts.point_detection.functions import (
    detect_isolated_points,
    show_plt_images,
    load_paths,
    save_if_enabled,
    run_neural_detection,
    run_laplacian_detection
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

# Define available image options using a dictionary
IMAGE_OPTIONS = {
    0: "circles_matlab.png",
    1: "calc.png",
    2: "crosses.png"
}

selected_image_index = 2
img_name = IMAGE_OPTIONS[selected_image_index]

if img_name is None:
    raise ValueError(f"Invalid image index: {selected_image_index}")

img, binary_image_flag, img_title = load_and_preprocess_image(img_name, data_path)

show_plt_images(img, img1_title=img_title)

if IMAGE_SAVE_SWITCH:
    Image.fromarray(img).convert('L').save(f"{image_save_path}/{img_name}")


# %% Template Matching
############################

# %% test
# test_input = np.array([[0, 255, 0],
#                        [0, 0, 0],
#                        [0, 255, 0]], dtype=np.uint8)

# test_kernel = np.array([[0, 1, 0],
#                         [0,  0,  0],
#                         [0, 1, 0]], dtype=np.uint8)

# # test_input = np.array((
# #     [0, 0, 0, 0, 0, 0, 0, 0],
# #     [0, 255, 255, 255, 0, 0, 0, 255],
# #     [0, 255, 255, 255, 0, 0, 0, 0],
# #     [0, 255, 255, 255, 0, 255, 0, 0],
# #     [0, 0, 255, 0, 0, 0, 0, 0],
# #     [0, 0, 255, 0, 0, 255, 255, 0],
# #     [0,255, 0, 255, 0, 0, 255, 0],
# #     [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

# # test_kernel = np.array((
# #         [0, 1, -1],
# #         [1, -1, -1],
# #         [0, 1, 0]), dtype="int")

# # test_kernel = np.array((
# #         [-1, -1, -1],
# #         [-1, 1, -1],
# #         [-1, -1, -1]), dtype="int")

# cv2.morphologyEx(test_input,
#                  cv2.MORPH_HITMISS,
#                  np.asarray(test_kernel))


# %% hit and miss transform for central white pixel only

# # input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

# input_image = img_proc
# kernel = np.array([[-1, -1, -1],
#                    [-1,  1, -1],
#                    [-1, -1, -1]], dtype="int")

# single_pixels = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
# # single_pixels_inv = cv2.bitwise_not(single_pixels)
# # hm = cv2.bitwise_and(input_image, input_image, mask=single_pixels_inv)

# # show figure
# fig = plt.figure(figsize=(20, 8))
# ax1 = fig.add_subplot(111)
# ax1.imshow(single_pixels, cmap='gray')
# plt.show()

# %% hit and miss transform for central black pixel only

# # input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

# input_image = img
# kernel = np.array([[1,  1, 1],
#                    [1, -1, 1],
#                    [1,  1, 1]], dtype="int")

# single_pixels = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
# # single_pixels_inv = cv2.bitwise_not(single_pixels)
# # hm = cv2.bitwise_and(input_image, input_image, mask=single_pixels_inv)

# # show figure
# fig = plt.figure(figsize=(20, 8))
# ax1 = fig.add_subplot(111)
# ax1.imshow(single_pixels, cmap='gray')
# plt.show()

# %% hit or miss transform function
def hit_and_miss_transform(input_img: np.ndarray) -> np.ndarray:
    kernel_white = np.array([
        [-1, -1, -1],
        [-1,  1, -1],
        [-1, -1, -1]
    ], dtype=int)
    kernel_black = np.array([
        [1, 1, 1],
        [1, -1, 1],
        [1, 1, 1]
    ], dtype=int)
    white_pixels = cv2.morphologyEx(input_img, cv2.MORPH_HITMISS, kernel_white)
    black_pixels = cv2.morphologyEx(input_img, cv2.MORPH_HITMISS, kernel_black)
    return cv2.bitwise_or(white_pixels, black_pixels)

def plot_hit_and_miss_output(input_img, h_and_m_output):

    fig = plt.figure(figsize=(20, 8))
    plt.gray()
    # Original Image
    ax1 = fig.add_subplot(121)
    ax1.imshow(input_img)
    ax1.set_title('Original Image')

    ax2 = fig.add_subplot(122)
    ax2.imshow(h_and_m_output)
    ax2.set_title('Pixels Detected by Hit and Miss Transform')

    plt.tight_layout()
    plt.show()

h_and_miss_img = hit_and_miss_transform(img)
plot_hit_and_miss_output(img, h_and_miss_img)

print("Number of isolated pixels located by Hit and Miss is: {}"
      .format(np.count_nonzero(h_and_miss_img)))

if IMAGE_SAVE_SWITCH: 
    Image.fromarray(h_and_miss_img).convert('L').save(f"{image_save_path}/h_&_m_{img_name}")


# %% ============================================================
#                 Laplacian Detection (Manual or Otsu)
# ===============================================================
laplacian_binary, lablacian_abs_response, num_laplacian, threshold_used = run_laplacian_detection(
    img, LAPLACIAN_KERNEL, use_manual_threshold=True, manual_ratio=0.9
)

print(f"Laplacian threshold used: {threshold_used}")
print(f"Number of Laplacian isolated pixels: {num_laplacian}")

show_plt_images(img, "Original image", laplacian_binary, "Laplacian Response")
show_plt_images(img, "Original image", lablacian_abs_response, "Laplacian Absolute Response")

save_if_enabled(IMAGE_SAVE_SWITCH, laplacian_binary, image_save_path, img_name, prefix="laplacian_")


# %% test code for what kernel is being applied
# kernel =np.array([[0, 1, 0] , [1, -4, 1] , [0, 1, 0]])
# kernel =np.array([[-2, 0, -2] , [0, 8, 0] , [-2, 0, -2]])
# kernel =np.array([[-1, -1, -1] , [-1, 8, -1] , [-1, -1, -1]])

# v = np.array([
#     [0, 0, 0, 0, 0, 0, 0, ],
#     [0, 0, 0, 0, 0, 0, 0, ],
#     [0, 0, 0, 1, 0, 0, 0, ],
#     [0, 0, 0, 0, 0, 0, 0, ],
#     [0, 0, 0, 0, 0, 0, 0, ],

# ])
# v = v.astype(np.uint8)

# dst1 = cv2.filter2D(v, ddepth=cv2.CV_64F, kernel=kernel)
# dst1

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


# %% -------- Median Filtering --------
# doesn't find isolated pixels, just smoothes, but gets rid of lines!

# Apply the median filter to the image
filtered_img = cv2.medianBlur(img, KERNEL_SIZE)

show_plt_images(img, "Original image", filtered_img, "Filtered image")

# %%
