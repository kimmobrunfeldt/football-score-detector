import os
import sys
import math
import collections
import itertools
import heapq

import numpy as np
import scipy.ndimage
from PIL import Image
import skimage.filter
import skimage
import matplotlib.pyplot as plt
from skimage.morphology import label, closing, square
import cv2


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


def main():
    file_name = sys.argv[1]
    image = cv2.imread(file_name)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bw_image = find_blue(hsv_image)

    cv2.imwrite('range.jpg', bw_image)

    non_zero_pixels = cv2.findNonZero(bw_image)

    rect = cv2.minAreaRect(non_zero_pixels)
    precise_corners = cv2.cv.BoxPoints(rect)
    corners = np.int0(np.around(precise_corners))
    ends = find_table_ends(corners)

    draw_points = []

    end1, end2 = ends
    middle1, middle1_a, middle1_b = table_end_middles(end1)
    middle2, middle2_a, middle2_b = table_end_middles(end2)

    addition1 = calculate_coordinate_addition(middle1, middle2)
    # This is basically opposite direction than addition1
    addition2 = calculate_coordinate_addition(middle2, middle1)

    # Calculate the bounding boxes for score blocks
    end1_box = calculate_score_box(middle1_a, middle1_b, addition1)
    end2_box = calculate_score_box(middle2_a, middle2_b, addition2)

    draw_points += [end1_box[0], end1_box[2]]
    draw_points += [end2_box[0], end2_box[2]]

    print end1_box[0], end1_box[2]

    score1_image = image[end1_box[0][1]:end1_box[2][1], end1_box[0][0]:end1_box[2][0]]
    cv2.imwrite('score.jpg', score1_image)

    for end in ends:
        cv2.line(image, tuple(end[0]), tuple(end[1]), (0, 0, 255), 5)

    for point in draw_points:
        print 'point:', point
        cv2.circle(image, tuple(point), 20, (255, 0, 0))

    cv2.imwrite('output2.jpg', image)


if __name__ == '__main__':
    main()
