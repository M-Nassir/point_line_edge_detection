# -*- coding: utf-8 -*-

from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from numpy import inf
import numpy as np
from perception import Perception
import random
# functions


def rolling_window(input_array, size_kernel, stride, print_dims=True):
    """Function to get rolling windows.

    Arguments:
        input_array {numpy.array}

        Input, by default it only works with depth equals to 1.
        It will be treated as a (height, width) image.
        If the input have (height, width, channel) dimensions,
        it will be rescaled to two-dimension (height, width)

        size_kernel {int}

        size of kernel to be applied.
        Usually 3,5,7.
        It means that a kernel of (size_kernel, size_kernel) will be applied
        to the image.

        stride {int or tuple}

        horizontal and vertical displacement

    Keyword Arguments:
        print_dims {bool} -- [description] (default: {True})

    Returns:
        [list] -- A list with the resulting numpy.arrays
    """
    # Check right input dimension
    assert(len(input_array.shape) in set(
        [1, 2])),\
        "input_array must have dimension 2 or 3. Yours have dimension {}"\
        .format(len(input_array))

    if input_array.shape == 3:
        input_array = input_array[:, :, 0]

    # Stride: horizontal and vertical displacement
    if isinstance(stride, int):
        sh, sw = stride, stride
    elif isinstance(stride, tuple):
        sh, sw = stride

    # Input dimension (height, width)
    n_ah, n_aw = input_array.shape

    # Filter dimension (or window)
    n_k = size_kernel

    dim_out_h = int(np.floor((n_ah - n_k) / sh + 1))
    dim_out_w = int(np.floor((n_aw - n_k) / sw + 1))

    # List to save output arrays
    list_tensor = []

    # Initialize row position
    start_row = 0
    for i in range(dim_out_h):
        start_col = 0
        for j in range(dim_out_w):

            # Get one window
            sub_array = input_array[start_row:(
                start_row+n_k), start_col:(start_col+n_k)]

            # Append sub_array
            list_tensor.append(sub_array)
            start_col += sw
        start_row += sh

    if print_dims:
        print("- Input tensor dimensions -- ", input_array.shape)
        print("- Kernel dimensions -- ", (n_k, n_k))
        print("- Stride (h,w) -- ", (sh, sw))
        print("- Total windows -- ", len(list_tensor))

    return list_tensor


def get_y_windows(image_array, kern=3, stride_length=1):

    # blur the image if required
    # blur_img = cv2.GaussianBlur(img[:,:], (5,5),0)
    # imshow(blur_img, cmap='gray');

    return rolling_window(image_array, size_kernel=kern, stride=stride_length, print_dims=True)


def get_response(data, labels, idx_inhib, idx_excite, inhib_sum_num=0, excite_num=3, kernel_size=3):

    trigger = 0

    # inhibition area has anomaly excitations i.e. light falls on it (off-centre) or darkness fall on it (on-centre)
    if np.sum(labels[idx_inhib]) > inhib_sum_num:
        c = np.zeros(kernel_size**2)

    # if not all excite region has anomaly excitations (don't really need this for complete inhibition)
    elif np.sum(labels[idx_excite]) > excite_num:
        c = np.zeros(kernel_size**2)

    # if average intensity value of inhibition area is > excitation area
#     elif np.mean((data[idx_inhib])) > np.mean((data[idx_excite])):
#         c = np.zeros(kernel_size**2)

    else:
        c = labels
        c[idx_inhib] = 0
        trigger = 1

    return c, trigger


def show_detected_pixels(img, filtered_response, kernel_size):

    # show the isolated pixels found image
    print(len(filtered_response))
    zero_img = np.zeros((img.shape[0], img.shape[1]))

    n = img.shape[0]
    m = img.shape[1]

    s = 0
    for i in range(n-kernel_size+1):
        for j in range(m-kernel_size+1):
            #         print(i)
            #         print(j)
            #         print(zero_img[i:kernel_size+i,j:kernel_size+j].shape)
            #         print(label_list[s].shape)
            #         print("--")
            temp_img = np.logical_or(
                zero_img[i:kernel_size+i, j:kernel_size+j], filtered_response[s])
            zero_img[i:kernel_size+i, j:kernel_size+j] = temp_img
            s = s+1

    new_im = Image.fromarray((zero_img * 255).astype(np.uint8))

    # new_im.resize((200,200))
    fig = plt.figure(figsize=(20, 8))
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    result1 = new_im
    ax1.imshow(img)
    ax2.imshow(result1)
    plt.show()

    return zero_img


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
    # experitse to see if this is a natural way to solve the problem.
    # template matching does not work as bakcround could be anything, but
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
        trigger = 0

    # if more or less than excite_num anomalies detected
    elif np.sum(labels[idx_excite]) != excite_num:
        c = np.zeros(kernel_size**2)
        trigger = 0
    else:
        c = labels
        pixel_median = clf.training_median_
        trigger = 1

    return c, trigger, pixel_median


def detect_isolated_points(img, excite_num=1, inhib_sum_num=0, kernel_size=3):

    # holder for filter_response
    filter_response = []

    # holder for filtered image
    filtered_image = []

    # get image windows
    y_windows = get_y_windows(img, kern=kernel_size, stride_length=1)

    for window in y_windows:

        # flatten the data to 1D array, ensure all values are ints in range
        data = window.flatten().astype(int)

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
        if fired_correctly == 1:

            # replace the pixel value in original image with the median (we make new filtered image)
            filtered_image.append(pixel_median)

            # we get image of where the isolation detector has fired (we get isolation pixels image)
            filter_response.append(c.reshape(kernel_size, kernel_size))

        else:
            # keep everything as is
            filtered_image.append(data[4])
            default_response = np.zeros(kernel_size**2)
            filter_response.append(
                default_response.reshape(kernel_size, kernel_size))

    return filter_response, filtered_image


def sp_noise(image, prob):
    '''
    Add salt pepper noise
         PROB: Noise ratio
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
