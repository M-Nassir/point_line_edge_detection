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
image_save_switch = False

# %%
############################
#
#     Functions
#
############################
def show_plt_image(img):
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.imshow(img, cmap='gray')
    # plt.show()

    if img is not None:
        plt.imshow(img, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()

# %%
############################
#
#       Read Images
#
############################

# %% path to images
data_path = ("../../data/")
file_with_paths = '../../paths.txt'

# %% get path to save images
with open(file_with_paths) as f:
    image_save_path = f.readline()
    image_save_path = image_save_path[:-1]
    print(image_save_path)

# %% read selected image

image_options = [
    "circles_matlab.png",  # 0
    "calc.png",            # 1
    "crosses.png"          # 2
]

# Select the desired image by its index (0-based)
selected_image_index = 1

# Get the selected image name
img_name = image_options[selected_image_index]

img_raw = data_path + img_name
im_converted = Image.open(img_raw).convert('L')
img_proc = np.array(im_converted)  # array([  0, 255], dtype=uint8)

if img_name == 'circles_matlab.png':

    img = img_proc

    # ensure image is binary of values {0, 255}
    assert ((img == 0) | (img == 255)).all()

    # add the isolated pixels
    img[200][175] = 0
    img[75][150] = 0
    img[75][300] = 255

    binary_image_flag = True
    show_plt_image(img)

elif img_name == 'calc.png':

    # binarize the image
    img = cv2.threshold(img_proc, 10, 255, cv2.THRESH_BINARY)[1]

    # ensure image is binary of values {0, 255}
    assert ((img == 0) | (img == 255)).all()

    binary_image_flag = True
    show_plt_image(img)

elif img_name == 'crosses.png':

    # binarize the image
    img = cv2.threshold(img_proc, 20, 255, cv2.THRESH_BINARY)[1]

    # ensure image is binary of values {0, 255}
    assert ((img == 0) | (img == 255)).all()

    # add the isolated pixels
    img[200][175] = 255
    img[75][150] = 255
    img[75][200] = 255

    binary_image_flag = True

    show_plt_image(img)

else:
    raise Exception('Image name not defined, var: img_name')

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

# %% test
test_input = np.array([[0, 255, 0],
                       [0, 0, 0],
                       [0, 255, 0]], dtype=np.uint8)

test_kernel = np.array([[0, 1, 0],
                        [0,  0,  0],
                        [0, 1, 0]], dtype=np.uint8)

# test_input = np.array((
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 255, 255, 255, 0, 0, 0, 255],
#     [0, 255, 255, 255, 0, 0, 0, 0],
#     [0, 255, 255, 255, 0, 255, 0, 0],
#     [0, 0, 255, 0, 0, 0, 0, 0],
#     [0, 0, 255, 0, 0, 255, 255, 0],
#     [0,255, 0, 255, 0, 0, 255, 0],
#     [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

# test_kernel = np.array((
#         [0, 1, -1],
#         [1, -1, -1],
#         [0, 1, 0]), dtype="int")

# test_kernel = np.array((
#         [-1, -1, -1],
#         [-1, 1, -1],
#         [-1, -1, -1]), dtype="int")

cv2.morphologyEx(test_input,
                 cv2.MORPH_HITMISS,
                 np.asarray(test_kernel))


# %% hit and miss transform for central white pixel only

# input_image = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

input_image = img_proc
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

# %% hit or miss transform function
def hit_and_miss_transform(img_proc):
    fig = plt.figure(figsize=(20, 8))

    # Original Image
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_proc, cmap='gray')
    ax1.set_title('Original Image')

    # Hit and miss transform for central white pixel only
    kernel_white = np.array([[-1, -1, -1],
                             [-1,  1, -1],
                             [-1, -1, -1]], dtype="int")
    single_pixels_white = cv2.morphologyEx(img_proc, cv2.MORPH_HITMISS, kernel_white)

    # Hit and miss transform for central black pixel only
    kernel_black = np.array([[1,  1, 1],
                             [1, -1, 1],
                             [1,  1, 1]], dtype="int")
    single_pixels_black = cv2.morphologyEx(img_proc, cv2.MORPH_HITMISS, kernel_black)

    # Combine white and black pixels into one image
    combined_pixels = cv2.bitwise_or(single_pixels_white, single_pixels_black)

    ax2 = fig.add_subplot(122)
    ax2.imshow(combined_pixels, cmap='gray')
    ax2.set_title('Central White and Black Pixels')

    plt.tight_layout()
    plt.show()

hit_and_miss_transform(img_proc)


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

# %% Apply Laplace function (cv2.Laplacian implementation appears to be using
# wrong kernel)
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# ddepth = cv2.CV_16S
dst = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=kernel)

# dst = cv2.Laplacian(img, ddepth, ksize=3)

# converting back to uint8
abs_dst = np.abs(dst)  # cv2.convertScaleAbs(dst)

# find highest pixel value in image and take % of it
threshold = int(0.90 * np.max(abs_dst))

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
