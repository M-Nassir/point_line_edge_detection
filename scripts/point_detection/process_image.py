import numpy as np
from PIL import Image
from pathlib import Path
import cv2

def load_and_preprocess_image(img_name, data_path):

    # form full image path
    img_path = data_path / img_name

    # load image as greyscale
    img = np.array(Image.open(img_path).convert('L'))

    if img_name =='square_shades.png':
        
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
        img_title = 'Original image with 7 isolated pixels'

    elif img_name =='mach_bands.png':

        # add the isolated pixels
        img[200][100] = 255
        img[75][150] = 200
        img[140][100] = 150

        img[140][80] = 20
        img[150][150] = 150
        img[100][120] = 90

        img[220][200] = 20
        img[240][240] = 50
        img[50][240] = 70

        # image is natural so do no use binary detection
        binary_image_flag = False
        img_title = 'Original image with 9 isolated pixels'

    elif img_name == "circles_matlab.png":
        
        # Ensure the image is binary
        assert np.isin(img, [0, 255]).all(), "Image must be binary (0 or 255)"

        # Crop region of interest
        img = img[45:280, 100:300]

        # Manually insert isolated pixels
        img[25, 25] = 0
        img[55, 60] = 0
        img[30, 100] = 255
        img[70, 150] = 255
        img[180, 150] = 0

        binary_image_flag = True
        img_title = "Original image with 5 isolated pixels"

    elif img_name == "calc.png":
        
        # Binarise with a threshold of 10
        _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

        assert np.isin(img, [0, 255]).all(), "Image must be binary (0 or 255)"

        binary_image_flag = True
        img_title = "Original image with original isolated pixels"

    elif img_name == "crosses.png":
        
        # Binarise with a threshold of 20
        _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        assert np.isin(img, [0, 255]).all(), "Image must be binary (0 or 255)"

        # Add isolated pixels
        img[200, 175] = 255
        img[75, 150] = 255
        img[75, 200] = 255

        binary_image_flag = True
        img_title = "Original image with 3 added isolated pixels"

    else:
        binary_image_flag = False
        img_title = 'Original image'

    return img, binary_image_flag, img_title