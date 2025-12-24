"""
Created on Fri Aug 18 20:41:58 2023

@author: nassirmohammad
"""

"""
The goal of this script is to filter an image with a 3*3 or 5*5 or 7*7 or 9*9
perception filter and to observe what structures of excitation and inhibition
are learnt over a simple artificial and natural image.

Subsequently, a second complex layer is to be used to learn higher level
features.

Then clustering to be carried out to group categories of images using this
higher level of representation over pixel based representation. Dimension
reduction can also be applied to this 'bag of words' model and compared to
that applied to the raw pixels and subsequent clustering.

Semi-supervised learning is leveraged to learn names for different groupings
and to differentiate or partition certain larger groups.

Learning primitive image features has been a focus of research for decades,
with sparse coding being one of the first successful approaches.
Other techniques, such as Independent Component Analysis (ICA) and
K-means clustering, have also been explored. However, these early methods
faced challenges in extending learning beyond the initial feature layer,
often due to limitations in their optimisation techniques or the absence
of frameworks to support hierarchical learning. Autoencoders,
while effective, emerged later as a key technique for learning data
representations in deeper layers; however, this method requires large amounts
of training data.

Aside from the deep learning methods, other methods are research areas
that were initially studied until deep learning took off.

An idea from this area appears to be to find the dictionary of sparse vectors
that describe natural images, compose new images into their sparse vectors,
and then apply a machine learning algorithm to discriminate them.

The goal of researchers was to carry out unsupervised
learning or semi-supervised or self-taught learning as is done by the human
brain, but coming across and realising that massive amounts of data could
train a model to learn the function is where the current AI focus is, primarily
perhaps due to its industrial applications.

There has also been much debate around optimal filters to detect what are
thought to be the first features of a neural network layer, i.e. edges. CNN's
are built so that they learn what the optimal features should be for training
data by modifying weights using backprogation in gradient descent.

However, I propose this to be an inefficient method to learn the first layer
filters and also not that which is carried out by the visual system. Rather, I
propose the visual system utilises some principles genetically encoded such
that it learns optimal filters and hence receptive fields for the environment
in which it is to function. Thus, learning these fields is a direct result
of the interaction with the environment and not a learning of weights from
a supervised classification task through backpropagation.

A result of the learning mechanism is that of all the possible image patterns
that can occur, binary or grey-scale, only a sparse fraction are learnt. Thus,
sparsity is not the goal of the network, rather it is an inherent property of
the learnt filters as a result of the sparse nature of meaningful information
found in the visual environment.

We should also show we can learn a new 'Calculus of meaningful information'
applied initially to 2D image data.

Resources:

    Morel Gestalt book + papers
    Perceptrons book
    Probability introduction book
    Bool + M&P
    Kofka book
    Piaget

    Marr: theory of edge detection, Vision introduction and chapter 1

    Hubel and Wiesel book papers
    Fukushima Main paper
    CNNs Lecun main paper
    Compare with with unsupervised learning classification

    Hebb book
    Monadology
    Turing, Shannon


process:

    1. read in the image
    2. design the filter (or have it learn by itself)
    3. apply the filter
    4. observe the results

Research outputs:

    1. New non-linear edge detection filter/neuron for calculating meaning. This
        method is the first that does not rely upon derivatives and then selection
        of a threshold. We also do not require labelled examples of edges.

    2. A method to learn sparse receptive fields by experiencing only the
        environment. No need to hand specify filters, learn through optimisation
        or learning through fiddling with weights from supervised backprop.

        The environment programs the network. (Perhaps through loss of filters
                                               or at layer layers)

    3. A neural network for learning and classifying simple 2D images.

    4. Comparison of V1 and V2 with our two layers of feature detection.

    5. Proposing the language or code or function of at least visual neurons.
       Proposing their fundamental function is the detection of distinctness
       or meaning.

Paper titles:
    A Calculus of meaningful information and
    The language of neurons in perception

"""

"""

Experiment 1, Simple Object recognition:

    Feature learning and representation:
        1. Get simple sets of images of triangles and squares
        2. Filter image using RGC over 3 * 3 (generic RF's'), reduce image by same factor
        2. learn V1 filters over 3 * 3
        3. form bag of words
        4. plot them geometrically in vector space according to frequency

    Grouping and classification:
        5. nearest neighbour classification at first until we get some labels?
            how to get initial seeds?
        6. cluster examples using unexpectedness to group classes together for
            stating classification.
        7. The name knowledge has to come from outside the system.
        8. Compare against clustering using pixel level information vs
           RGC level vs v1 level features where we use K-means and perception.
        9. Use a small sample of supervision to say which number is which
           and see if the rest of the numbers can be classified.

Experiment 2, MNIST digits:
    Build the bag of words model and operate over simple Mnist or shape images.
    Then try to build higher level gestalts from the partial models.

Experiment 3: Real world images:
    Apply filter over more images, especially real world small images.

"""
# TODO: double filter like in retina to LGN, or smooth before filtering
# TODO: either use mnist or create 100 examples of squares and triangles manually (do slight data augmentation and save files)
# TODO: apply k-means clustering of pixel level, RGC level and v1 level grouping and perception model
# TODO: do the baseline k-means clustering on two MNIST digits first using raw pixel values.
# RGV - fixed filters to get contrast information, v1 to get simple edges, v2 to get angles and junctions

# %% Setup
##############################################################################
#
#                                   Setup
#
##############################################################################

# -------------------------------------------
# import the required libraries
# -------------------------------------------
import os
from pathlib import Path
import sys
sys.path.append("../")

import cv2
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from IPython.display import display

from skimage.util import view_as_windows

import numpy as np
from joblib import Parallel, delayed

from utils import get_rolling_windows
from utils import plot_filter_image, plot_binary_image
from point_detection.functions import show_plt_images
from perception_nassir import Perception
from filtering.v1_line_edge_filters import *

# -------------------------------------------
# set the parameters for script
# -------------------------------------------

# set the kernel size of the filter:
# (likely will need several sizes in future)
kernel_size = 3  # 5, 7, 8, 9

# index range of pixels (initially for 3 * 3 kernel
filter_inds = list(range(kernel_size * kernel_size))

# exact number of RGC receptors that must be excited for feature detection in the 'excite_region'
# can this be learnt from viewing images (plasticity)?
rgc_excite_num = 0  # any number

# number of RGC receptors inhibiting the output of the neuron found in the 'inhibitory region'
rgc_inhib_sum_num = 0  # don't concern us now

# states whether the image read in is binary or grey scale
binary_image_flag = False

# whether to save the image to specified path
image_save_switch = False

# number of strides to take over an image using window size, reduces image size
stride_val = 2

# %% Functions
##############################################################################
#
#                             Functions
#
##############################################################################

# Create windows from image
def create_windows(img, kernel_size, stride_val):
    windows = view_as_windows(img, kernel_size, step=stride_val)
    num_windows = windows.shape[0] * windows.shape[1]
    flattened_windows = windows.reshape(num_windows, -1).astype(int)

    return flattened_windows, num_windows

# %% Read Images
##############################################################################
#
#                               Read Images
#
##############################################################################

cwd = Path(os.getcwd())

data_path = cwd.parent.parent / "data/"

# Read the first line from the paths.txt file
config_file_path = cwd.parent.parent / 'paths.txt'

image_save_path = config_file_path.read_text().splitlines()[0].strip()
print(image_save_path)

# %% read the images

image_options = [
    "lines_single_pixel_thick.gif", #0
    "circles_matlab.png",           #1
    "mach_bands.png",               #2
    "tri1.png",                     #3
    "sq1.png",                      #4
    "camera_man.png"
]

# Select the desired image by its index (0-based)
selected_image_index = 4

img_name = image_options[selected_image_index]
print(f'Selected image: {img_name}')

img_path = data_path / img_name
img_original = Image.open(data_path / img_name).convert('L')
img_array = np.array(img_original)

print(f'Size of image: {img_original.size}')
print(f'numpy array shape: {img_array.shape}')
display(img_original)

# %% read the image and convert to black and white image
def read_image_and_display(image_path, gauss_blur=0,
                           min_thresh=200, max_thresh=255,
                           binarize=False):

    im = Image.open(image_path).convert("L")
    print(np.unique(im, return_counts=True))

    if gauss_blur != 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=gauss_blur))

    im = np.array(im)

    print(np.unique(im, return_counts=True))

    if binarize is True:
        _, im = cv2.threshold(im, min_thresh, max_thresh, cv2.THRESH_BINARY)

    print(np.unique(im, return_counts=True))

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(im, cmap="gray")
    plt.show()

    return im

img = read_image_and_display(img_path, gauss_blur=0,
                             min_thresh=200,
                             max_thresh=255,
                             binarize=False)

# %% Canny edge detection
##############################################################################
#
#                       Apply Canny edge detection
#
##############################################################################
# def plot_side_by_side(original, processed, processed_title):
#     fig, axes = plt.subplots(1, 2, figsize=(20, 8))

#     axes[0].imshow(original, cmap="gray")
#     axes[0].set_title("Original Image")
#     axes[0].set_xticks([]), axes[0].set_yticks([])

#     axes[1].imshow(processed, cmap="gray")
#     axes[1].set_title(processed_title)
#     axes[1].set_xticks([]), axes[1].set_yticks([])

#     plt.tight_layout()
#     plt.show()


# Apply Canny edge detection
canny_edges = cv2.Canny(img, 10, 100)
show_plt_images(img1=img, img1_title='original image',
                img2=canny_edges, img2_title ='canny edges')

# %% Apply RGC Neuron Model to get generic meaningful inforamtion from image
##############################################################################
#
#     Learn image primitives using neurons and receptive field properties
#
##############################################################################
def rgc_filter(
    data,               # raw data
    labels,             # neuron anomaly responses
    idx_inhib,          # inhibition index locations
    idx_excite,         # excitation index locations
    r_type,             # type of neuron
    inhib_sum_num=0,    # can have anomaly in inhibition centre
    excite_sum_num=0,   # number of idx that need excitation i
    kernel_size=3,
):
    #     [0,1,2
    #      3,4,5
    #      6,7,8]

    # check if excite region is darker or brighter than central inhibition area
    excite_region_brighter = None
    if np.mean(data[idx_excite]) > np.mean(data[idx_inhib]):
        excite_region_brighter = True
    else:
        excite_region_brighter = False

    if r_type == "on":
        # on centre; if excite region is DARKER than centre
        # and if there at least one anomaly in receptive field, then fire
        if (
            excite_region_brighter is False

            # ensure there are anomalies
            and np.sum(labels) > 0

            # ensure anomalies in right area (assume anomalies can be anywhere)
            # and np.sum(labels[idx_excite]) > excite_sum_num
        ):
            output_labels = labels
            neuron_fired = True

            # print("on fired")

        else:
            neuron_fired = False
            output_labels = np.zeros(kernel_size**2)

        return output_labels, neuron_fired

    if r_type == "off":
        # off centre; if excite region is BRIGHTER than than centre,
        # and if there are at least one anomaly in receptive field, then fire
        if (
            excite_region_brighter is True

            # ensure some anomalies
            and np.sum(labels) > 0

            # ensure anomalies in correct area
            # and np.sum(labels[idx_excite]) > excite_sum_num

        ):
            output_labels = labels
            neuron_fired = True

            # print("off firing")

        else:
            neuron_fired = False
            output_labels = np.zeros(kernel_size**2)

        return output_labels, neuron_fired


# %% Application of Retinal Ganglion Cell Filters
# ---------------------------------------------------------------------------
#
#        for each window get the anomaly response on the retina
#        from the raw pixel intensities.
#
# ---------------------------------------------------------------------------
flattened_windows, num_windows = create_windows(img, kernel_size, stride_val)

# Preallocate response lists
l1_filter_response_on = np.zeros(num_windows, dtype=np.uint8)
l1_filter_response_off = np.zeros(num_windows, dtype=np.uint8)

clf = Perception()

def process_window(data):
    clf.fit_predict(data)
    labels = clf.labels_

    #     [0,1,2
    #      3,4,5
    #      6,7,8]

    # on cell response
    idx_inhib = [4]
    idx_excite = [0, 1, 2, 3, 5, 6, 7, 8]

    post_retina_img_on, on_center_fired = rgc_filter(
        data,
        labels,
        idx_inhib,
        idx_excite,
        inhib_sum_num=0,
        excite_sum_num=0,
        kernel_size=3,
        r_type="on",
    )

    # off cell response
    post_retina_img_off, off_center_fired = rgc_filter(
        data,
        labels,
        idx_inhib,
        idx_excite,
        inhib_sum_num=0,
        excite_sum_num=0,
        kernel_size=3,
        r_type="off",
    )

    # Return binary values based on whether the center fired for both ON and OFF filters
    return (1 if on_center_fired else 0, 1 if off_center_fired else 0)

# Apply in parallel to each window
responses = Parallel(n_jobs=-1)(delayed(process_window)(data) for data in flattened_windows)

# Unzip the responses into the preallocated response arrays
l1_filter_response_on[:], l1_filter_response_off[:] = zip(*responses)

# Convert to 255 or 0
l1_filter_response_on *= 255
l1_filter_response_off *= 255

# %% show the off centre and on centre RC response images
n, m = img.shape[:2]

response_off = np.array(l1_filter_response_off)
response_on = np.array(l1_filter_response_on)

combined_response = np.maximum(response_off, response_on)

# Calculate the number of windows based on the stride
height = (n - kernel_size) // stride_val + 1
width = (m - kernel_size) // stride_val + 1

# Ensure the dimensions are valid and positive
if height > 0 and width > 0:
    # Reshape the filter responses correctly based on stride
    response_off = response_off.reshape(height, width)
    response_on = response_on.reshape(height, width)
    combined_response = combined_response.reshape(height, width)
else:
    raise ValueError("Invalid dimensions for the response arrays with the given stride value.")


# Prepare titles and images for plotting
titles = ["Original", "Response Off", "Response On", "Canny", "combined response"]
images = [img, response_off, response_on, canny_edges, combined_response]

# Create a figure and axes
fig, axarr = plt.subplots(1, 5, figsize=(20, 8))

# Plot each image with its title
for ax, title, image in zip(axarr, titles, images):
    ax.set_title(title)
    ax.imshow(image, cmap='gray' if title != "Original" else None)
    ax.axis('off')  # Hide axes for cleaner presentation

# %% Learning V1 filters
# ---------------------------------------------------------------------------
#
#       Combine the off and on response images, then learn filters
#       through experience
#
# ---------------------------------------------------------------------------
# 8 straight filters could be learned, or could be innately processed. We
# choose to learn these innately, and then upon experience keep only those
# that are relevant to the environment.

# create windows from each response image
windows_v1, num_windows_v1 = create_windows(combined_response,
                                            kernel_size=3,
                                            stride_val=1)

# initialise classifier
clf = Perception()

def v1_simple_process(data, idx_excite, idx_inhib, line_angle):
    clf.fit_predict(data)
    labels = clf.labels_

    if np.sum(labels[idx_inhib]) > 0 or np.sum(labels[idx_excite]) != 3:
        return False, None  # No trigger
    else:
        return True, line_angle  # Trigger and return the angle

# Get excitation and inhibition indices for vertical line detection
idx_excite, idx_inhib, line_angle = get_left_vertical_line()

v1_filter_response = \
            Parallel(n_jobs=-1)(delayed(v1_simple_process)\
            (data, idx_excite, idx_inhib, line_angle=line_angle) for data in windows_v1)

line_angles = [x for trigger, x in v1_filter_response]
from collections import Counter
line_angles_value_counts = Counter(line_angles)

# Convert the response to 255 or 0 directly within the Parallel process
bool_values_v1_response = [255 if trigger else 0 for trigger, _ in v1_filter_response]

# reshape into an image
# Calculate the expected shape after applying a 3x3 kernel with stride 1
n, m = combined_response.shape[:2]
filtered_height = (n - kernel_size) // 1+1 # After applying 3x3 kernel with stride 1
filtered_width = (m - kernel_size) // 1+1
expected_size = filtered_height * filtered_width

# Ensure the total size matches expected size
if len(bool_values) != expected_size:
    raise ValueError(f"Data size mismatch. Expected {expected_size}, but got {len(bool_values)}.")

# Reshape the array
bool_values_v1_response_img = np.array(bool_values_v1_response,
                                       dtype=np.uint8).reshape((filtered_height,
                                                                filtered_width))

# Display the reshaped array as an image
plt.imshow(bool_values_v1_response_img, cmap='gray', interpolation='nearest')
plt.title("V1 simple cell response Image")
plt.axis('off')
plt.show()

# %%
# ---------------------------------------------------------------------------
#
#         each window get the anomaly response on the post_retina_img
#                           OFF-CENTRE
#
# ---------------------------------------------------------------------------

# OFF-CENTRE image (already in binary format)

# get  kernel size overlapping windows from the image
windows = get_rolling_windows(
    response_off, kernel_size=kernel_size, stride_length=stride_val
)

# Create a set to store unique arrays by binary pattern
unique_filters = set()

# List to store the final binary filters (without duplicates)
rf_filters = []

# the corresponding image patches that resulted in the learnt filter
image_patch_response = []

lf_filters = []

# for each window, apply the neuron filter
for w in windows:
    # flatten the data to 1D array, ensure all values are 'int' in range
    data = w.flatten().astype(int)

    # apply the perception fit_predict model
    clf = Perception()
    clf.fit_predict(data)
    labels = clf.labels_

    #     [0,1,2
    #      3,4,5
    #      6,7,8]

    # every time we have an anomaly response, save the binary filter image
    if np.sum(labels) > 0:
        idx_inhib = np.where(labels == 0)[0]
        idx_excite = np.where(labels == 1)[0]

        tp = tuple(labels)

        # Create an array filled with zeros
        # array_size = 9

        # # Adjust this to your desired array size
        # light_array = np.zeros(array_size).astype(int)

        # # check which area has more light intensity than the other
        # # a filter is a typle (rf of anomaly detection, light distribution)
        # if np.mean(data[idx_inhib]) < np.mean(data[idx_excite]):

        #     light_array[idx_excite] = 1

        # else:

        #     # make the inhibition area require higher
        #     light_array[idx_inhib] = 1

        # Convert the array to a tuple to make it hashable
        # arr_tuple = (tuple(labels), tuple(light_array))

        # now we have light_array, and labels

        # Check if the array is unique (maybe only need to add item to set?)
        if tp not in unique_filters:
            # Add the tuple to the set
            unique_filters.add(tp)

            # Append the array to the result list and the corresponding image patch
            rf_filters.append(labels.reshape(3, 3))
            # lf_filters.append(light_array.reshape(3, 3))
            image_patch_response.append(data.reshape(3, 3))

            # TODO: save all patches that fire the filter to a dictionary

print(len(rf_filters))
print(len(image_patch_response))

# %% plot some examples from the filters and image patches

number = 2

# plot the image
f, axarr = plt.subplots(1, 4, figsize=(20, 8))

axarr[0].set_title("Receptive field filter - OFF")
axarr[0].imshow(rf_filters[number])

axarr[1].set_title("Image patch response")
axarr[1].imshow(image_patch_response[number])

# axarr[2].set_title('Response On')
# axarr[2].imshow(response_on)

# axarr[3].set_title('Canny')
# axarr[3].imshow(edges_canny)

plt.show()

# plot_binary_image(rf_filters, number, colour_shade='Reds')
# # plot_binary_image(lf_filters, number, colour_shade='cividis')
# plot_filter_image(image_patch_response, number)

# %%
# ---------------------------------------------------------------------------
#
#               plot receptive field filter images
#
# ---------------------------------------------------------------------------

# List of random images for demonstration
image_list = rf_filters

# Determine the number of rows and columns for the plot
num_cols = int(np.minimum(4, len(rf_filters)))
num_rows = int(np.ceil(len(rf_filters) / num_cols))

# Create a figure and a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through the images and plot them
for i, ax in enumerate(axes):
    if i < len(image_list):
        # Use 'cmap' based on your image data
        ax.imshow(image_list[i], cmap="Reds")
        ax.set_title(f"Image {i+1}")
        ax.axis("off")  # Turn off axes

# Adjust layout to avoid overlapping titles and labels
plt.tight_layout()

# Show the plot
plt.show()


# %%
# ---------------------------------------------------------------------------
#
# Bag of words model over vector spaces (term matrix or consider tf-idf matrix)
#
# ---------------------------------------------------------------------------

# Let us build this model of bag of words; perform clustering/classification
# see how we can learn the next level of sentences from words! Words have a
# higher level of meaning than letters, sentences have higher level meaning
# than words, paragraphs have higher level of meaning than sentences.

# Each filter represents a word, the number of times it occurs is its frequency
# For an image we can thus compose a term-document frequency matrix, and
# make plots for each type.

# in theory we should be able to reliably classify each type of simple object.

# Compose dictionary set of all filters
# rf_filters

# map filter response to the dictionary table as an entry

# repeat upon more examples of squares and triangles of different sizes and orientations

# introduce more shapes, and grey scale values.

# demonstrate infeasibility of naive template matching and gradients + thresholding

# I want to learn one higher level though!


# %%
# ------------------------------------------------------------------------
#
#           Plot the full images filtered by a particular filter
#
# ---------------------------------------------------------------------------

# each binary filter specifies the excitation and inhibitory regions that
# are acceptable for firing/inhibiting when we have a filter response.
# thus, when the anomaly filter fires, its firing must also match the filter


# for each filter, apply the filter to the entire image and get a response


def filter_response(
    img, filter_pair, excite_num=0, inhib_sum_num=0, kernel_size=3
):
    # holder for filter_response
    filter_response = []

    # holder for filtered image
    filtered_image = []

    # get image windows
    windows = get_rolling_windows(
        img, kernel_size=kernel_size, stride_length=1
    )

    for w in windows:
        # flatten the data to 1D array, ensure all values are ints in range
        data = w.flatten().astype(int)

        # flatten the non-linear filter
        f_flattened = filter_pair[0].flatten().astype(int)

        # get inhibition region and excite region of filter
        #     [0,1,2
        #      3,4,5
        #      6,7,8]

        idx_inhib = np.where(f_flattened == 0)[0]
        idx_excite = np.where(f_flattened == 1)[0]

        clf = Perception()
        clf.fit_predict(data)
        labels = clf.labels_

        # if any inhibitory areas are exicted in the rf_filter don't fire
        if np.sum(labels[idx_inhib]) > inhib_sum_num:
            # exitation area must match that of the filter
            if np.sum(labels[idx_excite]) != np.sum(f_flattened):
                # create the light/dark binary light pattern
                # Create an array filled with zeros
                array_size = 9

                # Adjust this to your desired array size
                tmp_light_array = np.zeros(array_size).astype(int)

                # check which area has more light intensity than the other
                # a filter is a typle (rf of anomaly detection, light distribution)
                if np.mean(data[idx_inhib]) < np.mean(data[idx_excite]):
                    tmp_light_array[idx_excite] = 1

                else:
                    # make the inhibition area require higher
                    tmp_light_array[idx_inhib] = 1

                # check if light/dark distribution matches lf_filter

                # now check that the light distribution matches the filter_pair
                idx_dark = np.where(filter_pair[1] == 0)[0]
                idx_light = np.where(filter_pair[1] == 1)[0]

    return filtered_image, filter_response


# %%
filtered_images = []
filter_response_images = []


zipped_filters = zip(rf_filters, lf_filters)

for filter_pair in zipped_filters:
    # apply filter to input image
    # filtered_image, filter_response = filter_response(
    #     img, filter_pair, excite_num=0, inhib_sum_num=0, kernel_size=3)

    # holder for filter_response
    filter_response = []

    # holder for filtered image
    filtered_image = []

    # get image windows
    windows = get_rolling_windows(
        img, kernel_size=kernel_size, stride_length=1, print_dims=False
    )

    for w in windows:
        filter_fired = False

        # flatten the data to 1D array, ensure all values are ints in range
        data = w.flatten().astype(int)

        # flatten the non-linear filter
        f_flattened = filter_pair[0].flatten().astype(int)

        # get inhibition region and excite region of filter
        #     [0,1,2
        #      3,4,5
        #      6,7,8]

        idx_inhib = np.where(f_flattened == 0)[0]
        idx_excite = np.where(f_flattened == 1)[0]

        clf = Perception()
        clf.fit_predict(data)
        labels = clf.labels_

        # if any inhibitory areas are exicted in the rf_filter don't fire
        if np.sum(labels[idx_inhib]) <= inhib_sum_num:
            # exitation area must match that of the filter
            if np.sum(labels[idx_excite]) == np.sum(f_flattened):
                # create the light/dark binary light pattern
                # Create an array filled with zeros
                array_size = 9

                # Adjust this to your desired array size
                tmp_light_array = np.zeros(array_size).astype(int)

                # check which area has more light intensity than the other
                # a filter is a typle (rf of anomaly detection, light distribution)
                if np.mean(data[idx_inhib]) < np.mean(data[idx_excite]):
                    tmp_light_array[idx_excite] = 1

                else:
                    # make the inhibition area require higher
                    tmp_light_array[idx_inhib] = 1

                # check if light/dark distribution matches lf_filter

                # now check that the light distribution matches the filter_pair
                # idx_dark = np.where(filter_pair[1] == 0)[0]
                # idx_light = np.where(filter_pair[1] == 1)[0]

                if np.array_equiv(
                    filter_pair[1], tmp_light_array.reshape(3, 3)
                ):
                    # print(filter_pair[0])
                    # print(filter_pair[1])
                    # print(tmp_light_array.reshape(3, 3))

                    filter_fired = True

        if filter_fired is True:
            filter_response.append(1)
        else:
            filter_response.append(0)

    filter_response_images.append(filter_response)



# %%
