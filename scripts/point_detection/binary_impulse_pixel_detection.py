#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:42:05 2022.
Isolated pixel detection - a highly specialised application.
@author: Nassir Mohammad
"""

# put imshow into function in utilties
# %%
############################
#
#           Setup
#
############################

# %% import image handling
from point_detection.functions import detect_isolated_points
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# %% set parameters
kernel_size = 3
binary_image_flag = True

# %%
############################
#
#       Read Images
#
############################

# %% setup
image_save_switch = True

# %% path to images
data_path = ("../../data/")

# %%
with open('../../paths.txt') as f:
    image_save_path = f.readline()
    image_save_path = image_save_path[:-1]
    print(image_save_path)

# %% write to disk for conversion
# v2.imwrite("/Users/nassirmohammad/projects/computer_vision/point_line_edge_detection/point_line_edge_detection/data/turbine_blade_black_dot.png", img)

# %% image: circles

# read the image
img_name = "circles_matlab.png"
img1 = data_path + img_name
im = Image.open(img1).convert('L')
img = np.array(im)  # array([  0, 255], dtype=uint8)

# ensure image is binary of values {0, 255}
assert ((img == 0) | (img == 255)).all()

# add the isolated pixels
img[200][175] = 0
img[75][150] = 0
img[75][300] = 255

binary_image_flag = True

# show figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img, cmap='gray')
plt.show()

if image_save_switch is True:
    # save the path to where the paper figures are required
    save_path = image_save_path + '/' + img_name
    plt.savefig(save_path)

# %% image: calc

# read the image
img_name = "calc.png"  # (already has isolated pixels)
img1 = data_path + img_name
im = Image.open(img1).convert('L')
img_original = np.array(im)  # array([  0, 109, 128, 255], dtype=uint8)

# binarize the image
img = cv2.threshold(img_original, 10, 255, cv2.THRESH_BINARY)[1]

# ensure image is binary of values {0, 255}
assert ((img == 0) | (img == 255)).all()

binary_image_flag = True

# show figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img, cmap='gray')
plt.show()

if image_save_switch is True:
    # save the path to where the paper figures are required
    save_path = image_save_path + '/' + img_name
    plt.savefig(save_path)

# %% image: crosses

# read the image
img_name = "crosses.png"
img1 = data_path + img_name
im = Image.open(img1).convert('L')
img_original = np.array(im)  # array([ 20, 235], dtype=uint8)

# binarize the image
img = cv2.threshold(img_original, 20, 255, cv2.THRESH_BINARY)[1]

# ensure image is binary of values {0, 255}
assert ((img == 0) | (img == 255)).all()

# add the isolated pixels
img[200][175] = 255
img[75][150] = 255
img[75][200] = 255

binary_image_flag = True

# show figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img, cmap='gray')
plt.show()

if image_save_switch is True:
    # save the path to where the paper figures are required
    save_path = image_save_path + '/' + img_name
    plt.savefig(save_path)

# %%
############################
#
#   Template Matching
#
############################

# %% hit and miss transform for central white pixel only

# input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

input_image = img
kernel = np.array([[-1, -1, -1],
                   [-1,  1, -1],
                   [-1, -1, -1]], dtype="int")

single_pixels = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
# single_pixels_inv = cv2.bitwise_not(single_pixels)
# hm = cv2.bitwise_and(input_image, input_image, mask=single_pixels_inv)

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(single_pixels, cmap='gray')
plt.show()

# %% hit and miss transform for central black pixel only

# input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

input_image = img
kernel = np.array([[1,  1, 1],
                   [1, -1, 1],
                   [1,  1, 1]], dtype="int")

single_pixels = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
# single_pixels_inv = cv2.bitwise_not(single_pixels)
# hm = cv2.bitwise_and(input_image, input_image, mask=single_pixels_inv)

# show figure
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(single_pixels, cmap='gray')
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

# fig = plt.figure(figsize=(20, 8))
# ax1 = fig.add_subplot(111)
# ax1.imshow(output, cmap='gray')
# plt.show()

# %%
############################
#
#   Image Derivatives
#
############################

# %% Apply Laplace function
ddepth = cv2.CV_16S
dst = cv2.Laplacian(img, ddepth, ksize=3)

# converting back to uint8
abs_dst = np.abs(dst)  # cv2.convertScaleAbs(dst)

# find highest pixel value in image and take % of it
threshold = int(0.99 * np.max(abs_dst))

output = np.where(abs_dst > threshold, 1, 0)

print("Number of isolated pixels located by Laplacian is: {}"
      .format(np.sum(output)))

fig = plt.figure(figsize=(20, 8))
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(abs_dst)
ax2.imshow(output)
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

filtered_image, filtered_response = \
    detect_isolated_points(input_image,
                           excite_num=1,
                           inhib_sum_num=0,
                           kernel_size=kernel_size)

print("Number of isolated pixels located by net is: {}"
      .format(np.sum(filtered_response)))

# %% show only anomaly response pixels
n = img.shape[0]
m = img.shape[1]

new_image = np.array(filtered_response)
new_image = new_image.reshape(n-kernel_size+1, m-kernel_size+1)

# map the [0,1] image to [0,255]
new_image = Image.fromarray((new_image * 255).astype(np.uint8))

fig = plt.figure(figsize=(20, 8))
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img)
ax2.imshow(new_image)
plt.show()

# %% show filtered image
n = img.shape[0]
m = img.shape[1]

new_image = np.array(filtered_image)
new_image = \
    new_image.reshape(n-kernel_size+1, m-kernel_size+1).astype(np.uint8)

# image is already binary
# new_image = Image.fromarray((new_image * 255).astype(np.uint8))

fig = plt.figure(figsize=(20, 8))
plt.gray()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img)
ax2.imshow(new_image)
plt.show()
