#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 04:23:05 2022.
Isolated pixel detection - a highly specialised application.
@author: Nassir Mohammad
"""

# TODO: template matching/isolated pixel detection read Gonzalez book, compare methods online, DL approaches?
# TODO: finish the template matching method
# TODO: write paper on isolated pixel detection and first neural network method structure + diagram
# TODO: and setup for their detection. Aim is to do isolated pixel as an applied problem for the
# TODO: the new neural network and neuron model and to fully describe this method.
# try also implementing connected components? or maybe leave until last.
# we compare this new method against a couple of standard approaches in image processing.

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

from point_detection.functions import detect_isolated_points
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
    "camera_man.png",                   # 1
    "turbine_blade_black_dot.png"       # 2
]

# Select the desired image by its index (0-based)
selected_image_index = 0

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

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(img, cmap='gray')
plt.show()




# %%
############################
#
#       Neural Network
#
############################

# %% detect isolated pixels using neural network
if binary_image_flag is True:
    input_image = img
else:
    # blur the image, often said to be a process in vision before derivatives
    input_image = cv2.GaussianBlur(img, (3, 3), 0)

start_time = time.time()
filtered_image, filtered_response = detect_isolated_points(
    img, excite_num=1, inhib_sum_num=0, kernel_size=kernel_size
)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("Number of isolated pixels located by net is: {}"
      .format(np.sum(filtered_response)))

# Function to display image with original image
def display_image(image, title):
    n = img.shape[0]
    m = img.shape[1]
    new_image = np.array(image)
    new_image = new_image.reshape(n - kernel_size + 1, m - kernel_size + 1)

    if new_image.dtype != np.uint8:
        new_image = Image.fromarray((new_image * 255).astype(np.uint8))

    fig = plt.figure(figsize=(20, 8))
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img)
    ax2.imshow(new_image)
    ax2.set_title(title)
    plt.show()

# Display anomaly response pixels
display_image(filtered_response, "Anomaly Response Pixels")

# Display filtered image
display_image(filtered_image, "Filtered Image")

# %%
############################
#
#   Image Derivatives
#
############################

# %% Apply Laplace function (cv2.Laplacian implementation appears to be using
# wrong kernel). Applies Laplacian operator then takes absolute value and uses threshold to
# decide if a pixel is an anomaly.

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# ddepth = cv2.CV_16S
dst = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=kernel)

# dst = cv2.Laplacian(img, ddepth, ksize=3)

# converting back to uint8
abs_dst = np.abs(dst)  # cv2.convertScaleAbs(dst)

# find highest pixel value in image and take % of it
threshold = int(0.9 * np.max(abs_dst))

output = np.where(abs_dst > threshold, 1, 0)
# output = np.where(abs_dst == 2040, 1, 0)

print("Number of isolated pixels located by Laplacian is: {}"
      .format(np.sum(output)))

fig = plt.figure(figsize=(20, 8))
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(abs_dst)
ax2.imshow(output)
plt.show()

# **** Laplacian highly dependent upon the threshold value. Even then gives false positives.
# **** Reducing the threshold a lot still does not help much even in this simple example

# %%
############################
#
#   Template Matching
#
############################

# Direct application of hit or miss transform as template matching cannot
# be applied since it is designed for binary black/white pixel images.
