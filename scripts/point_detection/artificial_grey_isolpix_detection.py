#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 04:23:05 2022.
Isolated pixel detection - a highly specialised application.
@author: Nassir Mohammad

This script is written to experiment with isolated pixel detection in smooth
segments of grey level images with different shades and isolated pixels added
of varying intensities between [0,255]. Template matching fails here since
there are too many possiblities to check efficiently. Derivative filtering can
work to detect some isolated pixels, but fails elsewhere. Furthermore, it
requires the user to select a threshold parameter which restricts automation
and requires more manual human intervention. Otsu method is tested for
automatically finding the best threshold, but its results are unsatisfactory.
"""

# %%
############################
#
#           Setup
#
############################

# %% import image handling
import sys
sys.path.append("../")

import time

import numpy as np
import pandas as pd

from point_detection.functions import detect_isolated_points, display_image_plus_responses
import cv2

from matplotlib import pyplot as plt
from PIL import Image

# %% set parameters
kernel_size = 3
binary_image_flag = False

# %%
############################
#
#       Read Images
#
############################

# %% path to images
data_path = ("../../data/")
file_with_paths = '../../paths.txt'

# %% specify the greyscale images to input
image_options = [
    "square_shades.png",                # 0
    "camera_man.png",                   # 1 real world image
    "turbine_blade_black_dot.png"       # 2 real world image
]

# Select the desired image by its index (0-based)
selected_image_index = 2

# Get the selected image name
img_name = image_options[selected_image_index]

# %%

if img_name =='square_shades.png':
    img1 = data_path + img_name
    im = Image.open(img1).convert('L')
    img = np.array(im)

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

elif img_name =='turbine_blade_black_dot.png':
    img1 = data_path + img_name
    im = Image.open(img1).convert('L')
    img = np.array(im)

    binary_image_flag = False

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(img, cmap='gray')
plt.show()

# %%
############################
#
#   Template Matching
#
############################

# Direct application of hit or miss transform as template matching cannot
# be applied since it is designed for binary black/white pixel images.


# %%
############################
#
#       Neural Network
#
############################

# detect isolated pixels using neural network
if binary_image_flag is True:
    input_image = img
    print('using input image without blurring')
else:
    # blur the image, often said to be a process in vision before derivatives
    input_image = cv2.GaussianBlur(img, (5, 5), 0)
    print('filtered image with gaussian blur')

start_time = time.time()
filtered_image, filtered_response = detect_isolated_points(
    input_image, excite_num=1, inhib_sum_num=0, kernel_size=kernel_size
)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("Number of isolated pixels located by net is: {}"
      .format(np.sum(filtered_response)))

# Display anomaly response pixels
display_image_plus_responses(img, filtered_response, "Anomaly Response Pixels", kernel_size)

# Display filtered image
display_image_plus_responses(img, filtered_image, "Filtered Image", kernel_size)

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
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# 16 bits
ddepth_param = cv2.CV_16S

# take a simple approach to filter and threshold
dst = cv2.filter2D(img, ddepth=ddepth_param, kernel=kernel)

# get absolute values of filtered results as values can be negative
abs_dst = np.abs(dst)

# find highest pixel value in image and take % of it
threshold_simple = int(0.9 * np.max(abs_dst))

# Use Otsu's thresholding method to determine if a pixel is an anomaly
_, threshold_otsu = cv2.threshold(np.uint8(dst), 0, abs_dst.max(), cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if simple is True:
    threshold = threshold_simple
    print('using threshold: simple')
else:
    threshold = threshold_otsu
    print('using threshold: otsu')

# threshold for x% of max value
output_deriv_im = np.where(abs_dst > threshold, 1, 0)

print("Number of isolated pixels located by Laplacian is: {}"
      .format(np.sum(output_deriv_im)))

# Display the results
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(abs_dst, cmap='gray')
ax2.imshow(output_deriv_im, cmap='gray')
plt.show()

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
