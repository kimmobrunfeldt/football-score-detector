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
from scipy import ndimage
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




def distance_between_points(p):
    """Takes tuple p which contains two points and returns the difference
    between them. p format: ((x1, y1), (x2, y2))
    """
    return math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][1])**2)


def find_table_ends(points):
    """Find two shortest lines between points. These two lines are the ends
    of the table."""
    combinations = itertools.combinations(points, 2)
    ends = heapq.nsmallest(2, combinations, key=distance_between_points)
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


def middle_distance_between_score_dots(points):

    distances = []
    for i, point in enumerate(points[:-1]):
        distances.append(distance_between_points((point, points[i + 1])))

    return float(sum(distances)) / len(distances)


def find_score(points):
    """Returns the score from points."""
    points.sort()
    middle_distance = middle_distance_between_score_dots(points)

    score = 0
    for i, point in enumerate(points[:-1]):
        if distance_between_points((point, points[i + 1])) > middle_distance:
            break

        score += 1
    return score


def threshold(image):
    thresh = skimage.filter.threshold_otsu(image)
    print thresh
    bw = image > thresh
    return bw


def find_score_from_image(image):
    # Threshold
    T = 25

    # find connected components
    labeled, nr_objects = ndimage.label(image > T)


    slices = ndimage.find_objects(labeled)

    # Center coordinates of objects
    objects = []
    for sl in slices:
        dy, dx = sl[0:2]
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2

        objects.append((x_center, y_center))

    return find_score(objects)


def main():
    file_name = sys.argv[1]
    dna = Image.open(file_name).convert('L')
    dna = np.array(dna, dtype=int)

    # Threshold
    T = 100

    t = dna > T

    scipy.misc.imsave('result.png', t)


    # find connected components
    labeled, nr_objects = ndimage.label(dna > T)
    slices = ndimage.find_objects(labeled)

    # Center coordinates of objects
    objects = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2

        objects.append((x_center, y_center))

    print 'score is', find_score(objects)

    plt.imshow(dna)
    plt.plot([x[0] for x in objects], [x[1] for x in objects], 'ro')
    plt.savefig('result2.png', bbox_inches = 'tight')


if __name__ == '__main__':
    main()
