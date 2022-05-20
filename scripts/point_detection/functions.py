# -*- coding: utf-8 -*-

import numpy as np
from perception import Perception
from scripts.utilities import get_rolling_windows


def get_response_isolation(data,
                           idx_inhib, idx_excite,
                           inhib_sum_num=0, excite_num=1,
                           kernel_size=3):

    data = data.astype(int)
    assert np.max(data) <= 255
    assert np.min(data) >= 0

    trigger = None
    c = None
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

    # if inhibition area has anomaly excitations then neuron does not respond
    if np.sum(labels[idx_inhib]) > inhib_sum_num:
        c = np.zeros(kernel_size**2)
        trigger = False

    # if excite_pixel not excited as there is only one
    elif np.sum(labels[idx_excite]) != excite_num:
        c = np.zeros(kernel_size**2)
        trigger = False
    else:
        c = labels
        pixel_median = clf.training_median_
        trigger = True

    return c, trigger, pixel_median


def detect_isolated_points(img, excite_num=1, inhib_sum_num=0, kernel_size=3):

    # holder for filter_response
    filter_response = []

    # holder for filtered image
    filtered_image = []

    # get image windows
    windows = get_rolling_windows(img,
                                  kernel_size=kernel_size, stride_length=1)

    for w in windows:

        # flatten the data to 1D array, ensure all values are ints in range
        data = w.flatten().astype(int)

        # ----------------------
        # isolated point detector
        # ----------------------
        #     [0,1,2
        #      3,4,5
        #      6,7,8]

        idx_inhib = [0, 1, 2, 3, 5, 6, 7, 8]
        idx_excite = [4]

        c, fired_correctly, pixel_median =\
            get_response_isolation(data, idx_inhib, idx_excite,
                                   inhib_sum_num=inhib_sum_num,
                                   excite_num=excite_num,
                                   kernel_size=kernel_size)

        # if the neuron fires then take this response only
        if fired_correctly is True:

            # replace the pixel value in original image with the median
            # (we make new filtered image)
            filtered_image.append(pixel_median)

            # we get image of where the isolation detector has fired
            # (we get isolation pixels image)
            # filter_response.append(c.reshape(kernel_size, kernel_size))
            filter_response.append(1)

        else:
            # keep original pixel in new filtered image
            filtered_image.append(data[4])

            # give filter_response zero kernel sized patch
            # default_response = np.zeros(kernel_size**2)
            # filter_response.append(
            #     default_response.reshape(kernel_size, kernel_size))

            filter_response.append(0)

    return filtered_image, filter_response
