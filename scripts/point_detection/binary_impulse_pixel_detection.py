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
from point_detection.functions import detect_isolated_points, display_image_plus_responses
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time

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
selected_image_index = 2

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
    ax2.set_title('Central White and Black Pixels')

    plt.tight_layout()
    plt.show()

h_and_miss_img = hit_and_miss_transform(img)
plot_hit_and_miss_output(img, h_and_miss_img)

print("Number of isolated pixels located by Laplacian is: {}"
      .format(np.count_nonzero(h_and_miss_img)))

# %%
############################
#
#   Image Derivatives
#
############################§

# Apply Laplace function (cv2.Laplacian implementation appears to be using
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
#       Neural Network Model
#
################################

# %% detect isolated pixels using neural network

start_time = time.time()
filtered_image, filtered_response = detect_isolated_points(
    img, excite_num=1, inhib_sum_num=0, kernel_size=kernel_size
)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# Print the number of isolated pixels
print("Number of isolated pixels located by net is: {}".format(np.count_nonzero(filtered_response)))

# Display anomaly response pixels
display_image_plus_responses(img, filtered_response, "Anomaly Response Pixels", kernel_size)

# Display filtered image
display_image_plus_responses(img, filtered_image, "Filtered Image", kernel_size)
