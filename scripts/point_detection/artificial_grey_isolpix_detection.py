#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 04:23:05 2022
Author: Nassir Mohammad

Isolated Pixel Detection in Greyscale Images
--------------------------------------------
This script investigates the detection of isolated pixels within greyscale images
containing smooth intensity variations and artificially inserted anomalies of varying 
pixel values [0, 255]. The task is challenging for standard template matching methods 
which are impractical due to the combinatorial explosion of possible templates. 
Derivative-based approaches can succeed in some cases but are sensitive to threshold 
values, reducing their robustness and limiting automation.

The script evaluates:
- Template matching limitations
- Laplacian-based derivative filtering (with manual and Otsu thresholding)
- A perception-inspired neural model for point anomaly detection that works parameter-free
"""
# %%
# --------- Setup --------     
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from point_line_edge_detection.scripts.point_detection.functions import (
    detect_isolated_points, show_plt_images
)

# Parameters
kernel_size = 3
BINARY_IMAGE_FLAG = True
IMAGE_SAVE_SWITCH = False

# %% 
# -------- Paths and Image Selection --------
path = Path("../../")
data_path = path / 'data/'
file_with_paths = path / 'paths.txt'

# get path to save images
with open(file_with_paths) as f:
    image_save_path = f.readline()
    image_save_path = image_save_path[:-1]
    print(image_save_path)

# Define available greyscale image options using a dictionary with numeric keys
image_options = {
    0: "square_shades.png",                   # synthetic image
    1: "camera_man.png",                      # real-world image
    2: "turbine_blade_black_dot.png",         # real-world image
    3: "mach_bands.png",                      # synthetic image
}

selected_image_index = 3
img_name = image_options[selected_image_index]
img_path = data_path / img_name
img = np.array(Image.open(img_path).convert('L'))

if img_name =='square_shades.png':
    
    # add the isolated pixels
    img[200][100] = 255
    img[75][150] = 255
    img[300][100] = 150

    img[300][350] = 20
    img[150][150] = 150
    img[100][120] = 90

    img[250][350] = 0

    # as the image is not natural and not noisy, use binary detection
    binary_image_flag = True
    img_title = 'Original image with 7 isolated pixels'

elif img_name =='mach_bands.png':

    # add the isolated pixels
    img[200][100] = 255
    img[75][150] = 200
    img[140][100] = 150

    img[140][80] = 20
    img[150][150] = 150
    img[100][120] = 90

    img[220][200] = 20
    img[240][240] = 50
    img[50][240] = 70

    # image is natural so do no use binary detection
    binary_image_flag = False
    img_title = 'Original image with 9 isolated pixels'

show_plt_images(img, img1_title=img_title)

if IMAGE_SAVE_SWITCH:
    Image.fromarray(img).convert('L').save(f"{image_save_path}/{img_name}")

# %%
##############################################
#   Template Matching: Hit-or-Miss Transform
##############################################

# Note: The hit-or-miss transform is a template matching method
# specifically designed for binary (black-and-white) images.
# It is not directly applicable to greyscale or colour images.

# %%
############################
#       Neuron model
############################

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

if IMAGE_SAVE_SWITCH:
    Image.fromarray((filter_response * 255).astype('uint8')).save(f"{image_save_path}/neuron_{img_name}")

# %%
############################
#
#   Image Derivatives
#
############################

# Apply Laplace function (cv2.Laplacian implementation appears to be using
# wrong kernel). Applies Laplacian operator then takes absolute value and uses
# threshold to decide if a pixel is an anomaly.

# %% Option 1: use filter2D and manual threshold or Otsu threshold

simple = True

# set laplacian kernel
kernel = np.array([[-1, -1, -1], 
                   [-1, 8, -1], 
                   [-1, -1, -1]
                   ])

# 16 bits
ddepth_param = cv2.CV_16S

# take a simple approach to filter and threshold
dst = cv2.filter2D(img, ddepth=ddepth_param, kernel=kernel)

# get absolute values of filtered results as values can be negative
abs_dst = np.abs(dst)

# find highest pixel value in image and take % of it
threshold_simple = int(0.7 * np.max(abs_dst))

# Use Otsu's thresholding method to determine if a pixel is an anomaly
_, threshold_otsu = cv2.threshold(np.uint8(dst), 0,
                                  abs_dst.max(),
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if simple is True:
    threshold = threshold_simple
    print('using threshold: simple')
else:
    threshold = threshold_otsu
    print('using threshold: otsu')

# threshold for x% of max value
boolean_deriv_im = np.where(abs_dst > threshold, 1, 0)

print("Number of isolated pixels located by Laplacian is: {}"
      .format(np.sum(boolean_deriv_im)))

show_plt_images(img, 'Original image',  boolean_deriv_im, 'Laplacian_0.9')

if IMAGE_SAVE_SWITCH:
    Image.fromarray((boolean_deriv_im * 255).astype('uint8')).save(f"{image_save_path}/laplacian_{img_name}")

# %% Option 2: use Otsu threshold (keep as separate code for development)

# filter the image
dst = cv2.filter2D(img, ddepth=ddepth_param, kernel=kernel)

# Convert dst to 8-bit unsigned integer type
dst_8u = np.uint8(255 * dst / np.max(dst))

# dst = cv2.Laplacian(img, ddepth, ksize=3)

# Use Otsu's thresholding method to determine if a pixel is an anomaly
_, threshold_otsu = cv2.threshold(dst_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

threshold_otsu

output_otsu = np.where(dst_8u > threshold_otsu, 1, 0)

print("Number of isolated pixels located by Laplacian is: {}"
      .format(np.sum(output_otsu)))

fig = plt.figure(figsize=(20, 8))
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(dst_8u)
ax2.imshow(output_otsu)
plt.show()

# **** Laplacian highly dependent upon the threshold value. Even then gives false positives.
# **** Reducing the threshold a lot still does not help much even in this simple example

# %%
