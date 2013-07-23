import os
import sys
import math
import collections
import itertools
import heapq

import numpy as np
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
import skimage.color


# Percentage of length of the table. This is used to determine box around
# score blocks
SCORE_BLOCK_LENGTH = 0.06

# The HSV value range that is used to get blue color of the image
BLUE_RANGE_MIN = np.array([80, 70, 70], np.uint8)
BLUE_RANGE_MAX = np.array([130, 255, 255], np.uint8)



def coordinate_sort_key(p):
    """Takes tuple p which contains two points and returns the difference
    between them. p format: ((x1, y1), (x2, y2))
    """
    return math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][1])**2)


def find_table_ends(points):
    """Find two shortest lines between points. These two lines are the ends
    of the table."""
    combinations = itertools.combinations(points, 2)
    ends = heapq.nsmallest(2, combinations, key=coordinate_sort_key)
    return ends


def find_blue(hsv_image):
    """Takes image which is in HSV color space and returns new image which is
    black and white and all expect blue color is black.
    """
    return cv2.inRange(hsv_image, BLUE_RANGE_MIN, BLUE_RANGE_MAX)


def middle_point(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def table_end_middles(end):
    """Calculates table end's middle points which can be used to square score
    blocks
    """
    middle = middle_point(end[0], end[1])
    middle_a = middle_point(end[0], middle)
    middle_b = middle_point(end[1], middle)
    return middle, middle_a, middle_b


def calculate_coordinate_addition(middle1, middle2):
    """Calculates the needed coordinate delta that should be added to a middle
    point so the score blocks can be cropped.
    """
    x_diff = int((middle2[0] - middle1[0]) * SCORE_BLOCK_LENGTH)
    y_diff = int((middle2[1] - middle1[1]) * SCORE_BLOCK_LENGTH)
    return x_diff, y_diff


def calculate_score_box(middle_a, middle_b, addition):
    box_a = (middle_a[0] + addition[0], middle_a[1] + addition[1])
    box_b = (middle_b[0] + addition[0], middle_b[1] + addition[1])

    return (middle_b, middle_a, box_a, box_b)


def threshold(image):
    thresh = thresh = skimage.filter.threshold_otsu(image)
    bw = image > thresh
    return bw


def main():
    file_name = sys.argv[1]
    image = cv2.imread(file_name)
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    label_image = label(image)
    borders = np.logical_xor(bw, image)
    label_image[borders] = -1
    image_label_overlay = skimage.color.label2rgb(label_image, image=image)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image, ['Area', 'BoundingBox']):

        # skip small images
        if region['Area'] < 100:
            continue

        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region['BoundingBox']
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    plt.show()

if __name__ == '__main__':
    main()
