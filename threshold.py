"""
Usage: ./threshold.py image.jpg
"""

import os
import sys

import numpy as np
import scipy.ndimage
from PIL import Image
import skimage.filter
import matplotlib.pyplot as plt


def show_image(image):
    plt.figure(figsize=(8, 4))
    plt.imshow(image)
    plt.show()


def threshold(image):
    thresh = thresh = skimage.filter.threshold_otsu(image)
    bw = image > thresh
    return bw


def main():
    file_name = sys.argv[1]

    # Open image and convert to numpy array
    image = Image.open(file_name).convert('L')
    im = np.array(image, dtype=int)

    bw = threshold(im)

    # Save image
    base, ext = os.path.splitext(file_name)
    scipy.misc.toimage(bw).save(base + '_threshold' + ext)


if __name__ == '__main__':
    main()
