#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 04:23:05 2022.
Isolated pixel detection - a highly specialised application.
@author: Nassir Mohammad

This script will process some realworld isolated pixel examples. It will
also smooth an image if desired as some images can be quite noisy leading
to false positives. 3*3 or 5*5 kernels are good enough generally for such
imagery. An example is created to demonstrate the application of transforming a problem
where nodes might be spatially similar in terms of feature values, but there
maybe a single rogue node or malfunctioning node. This is represented as an
image from such network or spatial data, then the isolated pixels are detected.
Note that this application is different from that of filtering and smoothing an
image such as removing impulse noise because in that case we are not
necessarily looking to find the location of the actual points.
"""

# TODO: Generation of images
# TODO: this script should just do detection

# %% -------- Imports --------
import cv2
from PIL import Image, ImageFilter
from noise import snoise2
import numpy as np
from pathlib import Path
from IPython.display import display
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
new_param = True

# %% Load paths and image selection


def save_image(input_image, image_name):
    data_path = Path(__file__).resolve().parents[2] / "data"
    cv2.imwrite(str(data_path / image_name), input_image)

def show_image(input_image):
    output_image = Image.fromarray(input_image)
    display(output_image)    

def generate_perlin_noise(width, height, scale=300.0, octaves=6, persistence=0.5, lacunarity=3.0, base=42):
    xs = np.linspace(0, width / scale, width)
    ys = np.linspace(0, height / scale, height)
    noise_data = np.array([
        [snoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                 repeatx=1024, repeaty=1024, base=base) for x in xs]
        for y in ys
    ])
    norm = (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min())
    return (norm * 255).astype(np.uint8)

def add_isolated_pixels(img_array, n=10, delta_range=(20, 60)):
    for _ in range(n):
        y, x = np.random.randint(0, img_array.shape[0]), np.random.randint(0, img_array.shape[1])
        current = img_array[y, x]
        delta = np.random.uniform(*delta_range)
        extreme = current + delta if np.random.rand() > 0.5 else current - delta
        img_array[y, x] = np.clip(extreme, 0, 255)
    return img_array

def load_image(image_name):
    data_path = Path(__file__).resolve().parents[2] / "data"

    if image_name == 'ab1.png':
        img_path = data_path / image_name
        img_original = Image.open(img_path).convert('L')
        img = np.array(img_original)
        cropped_img = img[200:600, 400:800]
        return cropped_img

    elif image_name == 'spatial_anomaly_nodes.png':
        width, height = 128, 128
        noise_img = generate_perlin_noise(width, height)
        image = Image.fromarray(noise_img, mode='L').filter(ImageFilter.GaussianBlur(radius=2))
        img_array = np.array(image)
        img_array = add_isolated_pixels(img_array, n=10)
        return img_array

    else:
        raise ValueError(f"Unsupported image_name: {image_name}")

# # Load and write images to disk:
# orig_img = load_image('ab1.png')
# show_image(orig_img)
# save_image(orig_img, 'ab1_cropped.png')

# noise_img = load_image('spatial_anomaly_nodes.png')
# show_image(noise_img)
# save_image(noise_img, 'spatial_anomaly_nodes.png')


# %% -------- Paths and Image Selection --------
root, data_path, image_save_path = load_paths()

IMAGE_OPTIONS = {
    0: 'spatial_anomaly_nodes.png',
    1: 'turbine_blade_black_dot.png',
    # 2: 'ab1.png',
    3: 'ab1_cropped.png',
    # 4: 'xd3ke.png',
    # 6: 'test13.png',
}

selected_image_index = 1
img_name = IMAGE_OPTIONS[selected_image_index]

if img_name is None:
    raise ValueError(f"Invalid image index: {selected_image_index}")

img, binary_image_flag, img_title = load_and_preprocess_image(img_name, data_path)

show_plt_images(img, img1_title=img_title)

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

# Here are some specific guidelines for different types of images:

# Natural images (e.g., landscapes, portraits): 5x5 or 7x7 kernel size
# Medical images (e.g., MRI, CT scans): 3x3 or 5x5 kernel size (to preserve fine details)
# Textured images (e.g., fabrics, wood): 7x7 or 9x9 kernel size (to remove texture noise)
# Low-light images: 3x3 or 5x5 kernel size (to reduce noise amplification)

# %% detect isolated pixels using neural network

# if binary_image_flag is True:
#     proc_image = img_array.copy()
#     show_plt_images(img_array, 'Original image', proc_image, "proc_image")
# else:
#     # blur the image, often said to be a process in vision before derivatives
#     proc_image = cv2.GaussianBlur(img_array, (5, 5), 0)
#     show_plt_images(img_array, 'Original image', proc_image, "Gaussian blurred image: 5*5")

# # process image with neurons
# filtered_image, filter_response = \
#     detect_isolated_points(proc_image,
#                            excite_sum_num=1,
#                            inhib_sum_num=0,
#                            kernel_size=3)

# print("Number of isolated pixels located by net is: {}"
#       .format(np.sum(filter_response)))

# show_plt_images(img_array, 'Original image', filtered_image, "Filtered image")
# show_plt_images(img_array, 'Original image', filter_response, "Anomaly Response Pixels")

# neuron model works well in this example if the image is filtered with 5*5
# gaussin filter first. 3*3 filter works, but the neuron model detects
# lots of 'false positives' or noise that is meaningful.



# %% detect isolated pixels using neural network
# if binary_image_flag is True:
#     input_image = img_array
# else:
#     # blur the image, often said to be a process in vision before derivatives
#     input_image = cv2.GaussianBlur(img_array, (3, 3), 0)

# we don't blur the image in this example, even if we do we only find 3 additional
# isolated pixels.

# img = img_array.copy()

# %% ============================================================
#                 Laplacian Detection (Manual or Otsu)
# ===============================================================
laplacian_binary, lablacian_abs_response, num_laplacian, threshold_used = run_laplacian_detection(
    img, LAPLACIAN_KERNEL, use_manual_threshold=True, manual_ratio=0.9
)

print(f"Laplacian threshold used: {threshold_used}")
print(f"Number of Laplacian isolated pixels: {num_laplacian}")

show_plt_images(img, "Original image", lablacian_abs_response, "Laplacian Absolute Response")
show_plt_images(img, "Original image", laplacian_binary, "Laplacian Response")

save_if_enabled(IMAGE_SAVE_SWITCH, laplacian_binary, image_save_path, img_name, prefix="laplacian_")






# %% crop the suspicious_nodes image (isolated pixels are actually more than
# single pixels) - not required

# img_name = "suspicious_nodes.png"
# img_path = data_path + img_name

# # read and keep image as uint16, as conversion to uint8 drops details
# img = io.imread(img_path, as_gray=True)

# # Extract the pixel values from x=270 onward and y=150 onward
# cropped_img = img[150:, 270:]
# cropped_img[10][50] = 0

# # Save the pixel values as a new image (if needed)
# bottom_right_image = Image.fromarray(cropped_img)

# # Save the image as a new file
# bottom_right_image.save('bottom_right_corner.png')

# # If you want to display it
# # bottom_right_image.show()

# # Display the cropped image using matplotlib
# plt.imshow(cropped_img, cmap='gray')
# plt.title('Cropped Image')
# plt.show()
# %%
