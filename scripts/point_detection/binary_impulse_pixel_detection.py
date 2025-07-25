#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:42:05 2025

Isolated Pixel Detection – A Highly Specialised Application  
Author: Nassir Mohammad

This script applies a series of methods for detecting isolated pixels in binary images, including:

- Hit-or-miss transform  
- Derivative-based filtering  
- Median filtering  
- The Perception Neuron (IsolPix) detection method

It operates on three binary test images:

1. A synthetic image containing solid black and white regions with isolated pixels.  
2. A real-world image of a calculator, where the task is to detect and remove isolated white pixels.  
3. An image with both one-pixel-wide and thicker lines, used to demonstrate robustness in the presence of line segments.

The hit-or-miss transform reliably detects isolated pixels due to its template-matching approach. Similarly, the Perception Neuron (IsolPix) method shows strong performance, particularly in handling real-world noise.
"""

# put imshow into function in utilties
# %%
############################
#
#           Setup
#
############################

# %% import image handling
from point_line_edge_detection.scripts.point_detection.functions import (
    detect_isolated_points, show_plt_images
)

import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
from pathlib import Path

# %% set parameters
kernel_size = 3
BINARY_IMAGE_FLAG = True
IMAGE_SAVE_SWITCH = False

# %% Paths to Images
############################
#     Define Image Paths
############################

# Base path to data directory and path file
base_path = Path("../../")
data_path = base_path / "data"
file_with_paths = base_path / "paths.txt"

# %% get path to save images
with open(file_with_paths) as f:
    image_save_path = f.readline()
    image_save_path = image_save_path[:-1]
    print(image_save_path)

# %% Read Selected Image
############################
# Load and preprocess 
# selected binary image
############################

# Define available image options using a dictionary
image_options = {
    0: "circles_matlab.png",
    1: "calc.png",
    2: "crosses.png"
}

# Select the desired image by its index (0-based)
selected_image_index = 0

# Retrieve image name safely
img_name = image_options.get(selected_image_index)
if img_name is None:
    raise ValueError(f"Invalid image index: {selected_image_index}")

img_path = data_path / img_name
img_proc = np.array(Image.open(img_path).convert('L'))  # dtype=uint8

# Initialise flag
binary_image_flag = False

if img_name == "circles_matlab.png":
    # Ensure the image is binary
    assert np.isin(img_proc, [0, 255]).all(), "Image must be binary (0 or 255)"

    # Crop region of interest
    img = img_proc[45:280, 100:300]

    # Manually insert isolated pixels
    img[25, 25] = 0
    img[55, 60] = 0
    img[30, 100] = 255
    img[70, 150] = 255
    img[180, 150] = 0

    binary_image_flag = True
    show_plt_images(img, "Original image with isolated pixels")

elif img_name == "calc.png":
    # Binarise with a threshold of 10
    _, img = cv2.threshold(img_proc, 10, 255, cv2.THRESH_BINARY)

    assert np.isin(img, [0, 255]).all(), "Image must be binary (0 or 255)"

    binary_image_flag = True
    show_plt_images(img, "Original image with isolated pixels")

elif img_name == "crosses.png":
    # Binarise with a threshold of 20
    _, img = cv2.threshold(img_proc, 20, 255, cv2.THRESH_BINARY)

    assert np.isin(img, [0, 255]).all(), "Image must be binary (0 or 255)"

    # Add isolated pixels
    img[200, 175] = 255
    img[75, 150] = 255
    img[75, 200] = 255

    binary_image_flag = True
    show_plt_images(img, "Original image with isolated pixels")

else:
    raise ValueError(f"Image not recognised: {img_name}")

if IMAGE_SAVE_SWITCH is True:

    image_to_write = Image.fromarray(img).convert('L')

    # save the path to where the paper figures are required
    save_path = image_save_path + '/' + img_name
    image_to_write.save(save_path)

# %% Template Matching
############################
#
#   Template Matching
#
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
def hit_and_miss_transform(input_img):

    # ***** don't use np.uint8: transforms numbers to 255 and 1 ******

    # Hit and miss transform for central white pixel only
    kernel_white = np.array([[-1, -1, -1],
                             [-1,  1, -1],
                             [-1, -1, -1]], dtype='int')
    single_pixels_white = cv2.morphologyEx(input_img, cv2.MORPH_HITMISS, kernel_white)

    # Hit and miss transform for central black pixel only
    kernel_black = np.array([[1,  1, 1],
                             [1, -1, 1],
                             [1,  1, 1]], dtype='int')

    single_pixels_black = cv2.morphologyEx(input_img, cv2.MORPH_HITMISS, kernel_black)

    # Combine white and black pixels into one image
    combined_pixels = cv2.bitwise_or(single_pixels_white, single_pixels_black)

    return combined_pixels

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

if IMAGE_SAVE_SWITCH is True:

    image_to_write = Image.fromarray(h_and_miss_img).convert('L')

    # save the path to where the paper figures are required
    save_path = image_save_path + '/' + 'h_&_m_' + img_name
    image_to_write.save(save_path)

# %%
##############################################
#   Laplacian Derivative Filtering for 
#   Detection of Isolated Pixel Anomalies
##############################################

# Define Laplacian kernel manually (cv2.Laplacian may use a different kernel internally)
laplacian_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Apply the kernel using 64-bit float output to preserve negative values
laplacian_response = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=laplacian_kernel)

# Convert to absolute values to focus on intensity magnitude (edge strength)
abs_response = np.abs(laplacian_response)

# Define a dynamic threshold as 90% of the maximum response
threshold = int(0.90 * np.max(abs_response))

# Create binary output map: 1 where response > threshold, else 0
isolated_pixel_map = np.where(abs_response > threshold, 1, 0)

# Visualise the Laplacian response and binary detection output
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.gray()
ax1.set_title("Laplacian Response (Abs)")
ax1.imshow(abs_response)
ax2.set_title("Detected Isolated Pixels")
ax2.imshow(isolated_pixel_map)
plt.show()

# Print the number of detected isolated pixels
num_isolated = np.sum(isolated_pixel_map)
print(f"Number of isolated pixels located by Laplacian: {num_isolated}")

# %%
############################
#
#   Median Filtering
#
############################

# doesn't find isolated pixels, just smoothes, but gets rid of lines!

# Define the kernel size for the median filter
kernel_size = 3

# Apply the median filter to the image
filtered_img = cv2.medianBlur(img, kernel_size)

# Threshold the filtered image to remove isolated points
# _, thresh_img = cv2.threshold(filtered_img, 128, 255, cv2.THRESH_BINARY)

# Plot the original image, filtered image, and thresholded image
fig, axs = plt.subplots(1, 2, figsize=(18, 10))

axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(filtered_img, cmap='gray')
axs[1].set_title('Filtered Image (Median Blur)')

# axs[2].imshow(thresh_img, cmap='gray')
# axs[2].set_title('Thresholded Image')

plt.show()

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

# %%
################################
#
#       Neuron Model
#
################################

# %% detect isolated pixels using neural network

start_time = time.time()
filtered_image, filter_response = detect_isolated_points(
    img, excite_sum_num=1, inhib_sum_num=0, kernel_size=kernel_size
)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

show_plt_images(img, 'Original image', filtered_image, "Filtered image")
show_plt_images(img, 'Original image', filter_response, "Anomaly Response Pixels")

# Print the number of isolated pixels
print("Number of isolated pixels located by net is: {}".
      format(np.count_nonzero(filter_response)))

# %%
