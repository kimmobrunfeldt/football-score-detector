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

CV_FILLED = -1


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def show_image(image, **kwargs):
    plt.figure(figsize=(8, 4))
    plt.imshow(image, **kwargs)
    plt.show()


def threshold(image):
    thresh = thresh = skimage.filter.threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    return bw


def split_to_RGB(number, path_to_img):
    """ Split image into RGB channels """

    img = Image.open(path_to_img)
    img.convert('RGBA')
    r, g, b = img.split()
    return r


def find_corners(contours):
    print 'contours:', contours
    coordinates = [(i[0][0], i[0][1]) for i in contours]
    coordinates.sort()
    reverse_coordinates = [(y, x) for x, y in coordinates]
    reverse_coordinates.sort()

    point1 = coordinates[0]
    point2 = coordinates[-1]
    point3 = reverse_coordinates[0][1], reverse_coordinates[0][0]
    point4 = reverse_coordinates[-1][1], reverse_coordinates[-1][0]

    return point1, point2, point3, point4


def find_table_ends(points):
    """Find two shortest lines between points. These two lines are the ends
    of the table."""
    combinations = itertools.combinations(points, 2)
    ends = heapq.nsmallest(2, combinations, key=lambda p: math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][1])**2))
    return ends


def main():
    file_name = sys.argv[1]
    img = cv2.imread(file_name)
    im = np.array(img)
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    bw = cv2.inRange(hsv_im, np.array([80, 50, 50], np.uint8), np.array([130, 255, 255], np.uint8))
    cv2.imwrite('range.jpg', bw)

    non_zero_pixels = cv2.findNonZero(bw)


    corners = find_corners(non_zero_pixels)
    ends = find_table_ends(corners)

    for end in ends:
        cv2.line(img, end[0], end[1], (0, 0, 255), 5)

    for point in find_corners(non_zero_pixels):
        print 'corner:', point
        cv2.circle(img, tuple(point), 30, (255, 0, 0))

    cv2.imwrite('output2.jpg', img)


if __name__ == '__main__':
    main()
