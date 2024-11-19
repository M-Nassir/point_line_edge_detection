# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import numpy as np
from perception_nassir import Perception
from utils import get_rolling_windows
from matplotlib import pyplot as plt
from PIL import Image

def get_response_isolation(data,
                           idx_inhib, idx_excite,
                           inhib_sum_num=0, excite_sum_num=1,
                           kernel_size=3):

    data = data.astype(int)
    assert np.max(data) <= 65535  # 255
    assert np.min(data) >= 0

    trigger = None
    # c = None
    pixel_median = None

    clf = Perception()
    clf.fit_predict(data)
    labels = clf.labels_

    # ----------------------
    # isolated point detector
    # ----------------------
    #     [0,1,2
    #      3,4,5
    #      6,7,8]

    # these are natural constraints to the problem, we need more domain
    # expertise to see if this is a natural way to solve the problem.
    # template matching does not work as background could be anything, but
    # once we restrict pixel to be nearly black, the problem is solved.
    # perhaps we don't need darker

    # or (data[idx_excite] > 5):
    # if (data[idx_excite] > np.mean(data[idx_inhib])):
    #     c = np.zeros(kernel_size**2)
    #     trigger = 0

    #     return c, trigger, pixel_median
    # # dark code

    # could add on-center and off-center cells if they provide some good?

    # if inhibition area has anomaly excitations then neuron does not respond
    # if np.sum(labels[idx_inhib]) > inhib_sum_num:
    #     c = np.zeros(kernel_size**2)
    #     trigger = False

    # # if excite_pixel number not exceeded
    # elif (np.sum(labels[idx_excite]) < excite_sum_num) or :
    #     c = np.zeros(kernel_size**2)
    #     trigger = False
    # else:
    #     c = labels
    #     pixel_median = clf.training_median_
    #     trigger = True

    # return c, trigger, pixel_median

    # fire if no anomalies in inhibitory region and min number of anomalies in excite region, then fire
    if (np.sum(labels[idx_inhib]) <= inhib_sum_num) and (np.sum(labels[idx_excite]) <= excite_sum_num) \
        and (np.sum(labels[idx_excite]) >= 1):
        pixel_median = clf.training_median_
        trigger = True
    else:
        trigger = False

    return trigger, pixel_median

def show_plt_images(img1, img1_title, img2=None, img2_title=None):
    fig = plt.figure(figsize=(20, 8))
    plt.gray()

    if img2 is not None:
        # Display two images
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(img1)
        ax1.set_title(img1_title)
        ax1.axis('off')  # Turn off axis

        ax2.imshow(img2)
        ax2.set_title(img2_title if img2_title else 'Image 2')
        ax2.axis('off')  # Turn off axis
    else:
        # Display only one image
        ax1 = fig.add_subplot(111)

        ax1.imshow(img1)
        ax1.set_title(img1_title)
        ax1.axis('off')  # Turn off axis

    plt.show()

# Function to display image with original image
def display_image_plus_responses(img, img2, title):
    # n = img.shape[0]
    # m = img.shape[1]

    # # create the image from the filter response array
    # new_image = np.array(filter_resp_array)
    # new_image = new_image.reshape(n - kernel_size + 1, m - kernel_size + 1)

    # if new_image.dtype != np.uint8:
    #     new_image = Image.fromarray((new_image * 255).astype(np.uint8))

    fig = plt.figure(figsize=(20, 8))
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img)
    ax2.imshow(img2)
    ax2.set_title(title)
    plt.show()

def detect_isolated_points(img, excite_sum_num=1, inhib_sum_num=0, kernel_size=3):
    """
    Detects isolated points in an image using a neural network-inspired approach.

    Parameters:
    img (numpy array): The input image.
    excite_sum_num (int, optional): The minimum number of excitatory neurons required to fire. Defaults to 1.
    inhib_sum_num (int, optional): The sum of inhibitory neurons required to fire. Defaults to 0.
    kernel_size (int, optional): The size of the kernel used for the rolling window. Defaults to 3.

    Returns:
    tuple: A tuple containing the filtered image and the filter response.

    Notes:
    This function uses a rolling window approach to scan the image and detect isolated points.
    The detection is based on a neuron-inspired approach.
    The function returns two outputs: the filtered image, where isolated points are replaced with the median value,
    and the filter response, which is an image showing where the filter has fired.
    """

    # get image windows
    windows = get_rolling_windows(img,
                                  kernel_size=kernel_size,
                                  stride_length=1)

    # holder for filtered image: replaces pixel that fires with the median
    filtered_image_arr = np.zeros(len(windows), dtype=int)

    # boolean holder for filter_response; shows where filter fired
    filter_response_map_arr = np.zeros(len(windows), dtype=int)

    # convert list to NumPy array
    windows_array = np.array(windows)

    for i, w in enumerate(windows_array):

        # flatten the data to 1D array, ensure all values are ints in range
        data = w.flatten().astype(int)

        if kernel_size == 3:
            # -----------------------
            # isolated point detector
            # -----------------------
            #     [0,1,2
            #      3,4,5
            #      6,7,8]

            idx_inhib = [0, 1, 2, 3, 5, 6, 7, 8]
            idx_excite = [4]
            centre_pixel = 4

        elif kernel_size == 5:
            # -----------------------
            # isolated point detector
            # -----------------------
            #      0,  1,  2,  3,  4
            #      5,  6,  7,  8,  9
            #      10, 11, 12, 13, 14
            #      15, 16, 17, 18, 19
            #      20, 21, 22, 23, 24

            idx_inhib = [0, 1, 2, 3, 4,
                         5, 9,
                         10, 14,
                         15, 19,
                         20, 21, 22, 23, 24,
                         ]

            idx_excite = [6, 7, 8,
                          11, 12, 13,
                          16, 17, 18,
                          ]
            centre_pixel = 12
        elif kernel_size == 7:
            # -----------------------
            # isolated point detector
            # -----------------------
            #      0,  1,  2,  3,  4,  5,  6
            #      7,  8,  9,  10, 11, 12, 13
            #      14, 15, 16, 17, 18, 19, 20
            #      21, 22, 23, 24, 25, 26, 27
            #      28, 29, 30, 31, 32, 33, 34
            #      35, 36, 37, 38, 39, 40, 41
            #      42, 43, 44, 45, 46, 47, 48

            idx_inhib = [
                0, 1, 2, 3, 4, 5, 6,        # First row
                7, 8, 9, 10, 11, 12, 13,    # Second row
                14, 15, 19, 20,             # Third row (excluding the excitatory region)
                21, 22, 26, 27,             # Fourth row (excluding the excitatory region)
                28, 29, 33, 34,             # Fifth row (excluding the excitatory region)
                35, 36, 37, 38, 39, 40, 41, # Sixth row
                42, 43, 44, 45, 46, 47, 48  # Seventh row
            ]

            # Excitatory region (3x3 centered)
            idx_excite = [16, 17, 18,
                          23, 24, 25,
                          30, 31, 32]

            centre_pixel = 24

        # get the neuron response for the kernel_size area
        fired_correctly, pixel_median =\
            get_response_isolation(data, idx_inhib, idx_excite,
                                   inhib_sum_num=inhib_sum_num,
                                   excite_sum_num=excite_sum_num,
                                   kernel_size=kernel_size)

        # if the neuron fires then take this response only
        if fired_correctly is True:

            # replace the pixel value in original image with the median
            # (we make new filtered image)
            filtered_image_arr[i] = pixel_median

            # we get image of where the isolation detector has fired
            # (we get isolation pixels image)
            # filter_response.append(c.reshape(kernel_size, kernel_size))
            filter_response_map_arr[i] = 1

        else:

            # keep original pixel in new filtered image
            filtered_image_arr[i] = data[centre_pixel]

            # give filter_response zero kernel sized patch
            # default_response = np.zeros(kernel_size**2)
            # filter_response.append(
            #     default_response.reshape(kernel_size, kernel_size))

            filter_response_map_arr[i] = 0

        # format the arrays to be of the correct image size
        n = img.shape[0]
        m = img.shape[1]

        f_image = filtered_image_arr.reshape(n - kernel_size + 1, m - kernel_size + 1)
        f_respo = filter_response_map_arr.reshape(n - kernel_size + 1, m - kernel_size + 1)

    return f_image, f_respo
