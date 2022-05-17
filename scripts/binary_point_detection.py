#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:42:05 2022.
Isolated pixel detection - a highly specialised application.
@author: Nassir Mohammad
"""

# %% import image handling
from scripts.utilities import show_detected_pixels, detect_isolated_points
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from perception import Perception

# %% set parameters
k = 3
binary_image_flag = True

# %% Read image
data_path = ("/Users/nassirmohammad/projects/computer_vision/"
             "percept_detection/point_line_edge_detection/data/")

# binary images
# img_name = "circles_matlab.png"
# img_name = "calc.png" (already has isolated pixels)
# img_name = "crosses.tif"

# %% write to disk for conversion
# v2.imwrite("/Users/nassirmohammad/projects/computer_vision/point_line_edge_detection/point_line_edge_detection/data/turbine_blade_black_dot.png", img)

# %% image: circles

# read the image
img_name = "circles_matlab.png"
img1 = data_path + img_name
im = Image.open(img1).convert('L')
img = np.array(im)

# add the isolated pixels
img[200][175] = 0
img[75][150] = 0
img[75][300] = 255

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(img, cmap='gray')
plt.show()

binary_image_flag = True

# %% image: calc

# read the image
img_name = "calc.png"
img1 = data_path + img_name
im = Image.open(img1).convert('L')
img = np.array(im)

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(img, cmap='gray')
plt.show()

binary_image_flag = True

# %% image: crosses

# read the image
img_name = "crosses.png"
img1 = data_path + img_name
im = Image.open(img1).convert('L')
img = np.array(im)

# add the isolated pixels
img[200][175] = 255
img[75][150] = 255
img[75][200] = 255

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(img, cmap='gray')
plt.show()
binary_image_flag = True

############################
#
# Template Matching
#
############################

# %% hit and miss transform for central white pixel only

input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

kernel = np.array([[-1, -1, -1],
                   [-1,  1, -1],
                   [-1, -1, -1]], dtype="int")

single_pixels = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
single_pixels_inv = cv2.bitwise_not(single_pixels)
hm = cv2.bitwise_and(input_image, input_image, mask=single_pixels_inv)

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(single_pixels_inv, cmap='gray')
plt.show()

# %% hit and miss transform for central black pixel only

input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

kernel = np.array([[1, 1, 1],
                   [1,  -1, 1],
                   [1, 1, 1]], dtype="int")

single_pixels = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
single_pixels_inv = cv2.bitwise_not(single_pixels)
hm = cv2.bitwise_and(input_image, input_image, mask=single_pixels_inv)

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(single_pixels_inv, cmap='gray')
plt.show()

# %% TODO: hit or miss tranform using erosion, and erosion of complement

# load image, ensure binary, remove bar on the left
# input_image = cv2.imread('calc.png', 0)
# input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

# # --- erode with kernel and complement
# input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]
# input_image_comp = cv2.bitwise_not(input_image)  # could just use 255-img

# kernel1 = np.array([[0, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 0]], np.uint8)

# # kernel = np.ones((3, 3), np.uint8)

# kernel2 = np.array([[1, 1, 1],
#                     [1, 0, 1],
#                     [1, 1, 1]], np.uint8)

# hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_ERODE, kernel1)
# hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
# hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)

# hm = cv2.erode(img, kernel1)


############################
#
# Image Derivatives
#
############################

# %% Apply Laplace function
ddepth = cv2.CV_16S
dst = cv2.Laplacian(img, ddepth, ksize=3)

# converting back to uint8
abs_dst = np.abs(dst)  # cv2.convertScaleAbs(dst)

# fig = plt.figure(figsize=(20, 8))
# ax1 = fig.add_subplot(111)
# ax1.imshow(abs_dst, cmap='gray')
# plt.show()

# find highest pixel value in image and take 90% of it
threshold = int(0.9 * np.max(abs_dst))

output = np.where(abs_dst > threshold, 1, 0)

# fig = plt.figure(figsize=(20, 8))
# ax1 = fig.add_subplot(111)
# ax1.imshow(output, cmap='gray')
# plt.show()

fig = plt.figure(figsize=(20, 8))
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(abs_dst)
ax2.imshow(output)
plt.show()


############################
#
# Neural Network
#
############################

# %% detect isolated pixels using neural network
if binary_image_flag is True:
    filtered_response, filtered_image = detect_isolated_points(
        img, excite_num=1, inhib_sum_num=0, kernel_size=k)
else:
    # blur the image, often said to be a process in vision before derivatives
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)

    # show blurred image
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(blurred_img, cmap='gray')
    plt.show()

    filtered_response, filtered_image = detect_isolated_points(
        blurred_img, excite_num=1, inhib_sum_num=0, kernel_size=k)

# show original image and detected isolated pixels
show_detected_pixels(img, filtered_response, kernel_size=3)

############################
#
# Neural Network Learning
#
############################

# can the neural network learn to detect isolated pixels, by these pixels
# being the ones that carry the meaning in the image data that is exposed to?
# An example of plasticity, or environmental conditioning together with
# critical periods of development?


############################

# SCRAP Code

############################

# %% show original and filtered image (without isolated pixels)
# fig = plt.figure(figsize=(20, 8))
# plt.gray()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)

# result1 = np.array(filtered_image).reshape(img.shape[0]-k+1, -1).astype(int)

# ax1.imshow(img, cmap='gray')
# ax2.imshow(result1, cmap='gray')
# plt.show()


# %%
a = np.array([10, 10, 10,
              10, 55, 55,
              10, 55, 55])

clf = Perception()
clf.fit_predict(a)
print(int(np.median(a)))
print(clf.anomalies_)

# %% compare with median filtered image
fig = plt.figure(figsize=(20, 8))
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ascent = np.array(filtered_image).reshape(img.shape[0]-k+1, -1)
# scipy.signal.medfilt2d(input, kernel_size=3)
# result = ndimage.median_filter(img, size=3)

result = cv2.medianBlur(img, 3)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()
