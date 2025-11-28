import numpy as np
from PIL import Image
from pathlib import Path

def process_image(img_name, data_path):

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

    else:
        binary_image_flag = False
        img_title = 'Original image'

    return img, binary_image_flag, img_title