"""
Dependencies:

- OpenCV >= 2.4.4
- Numpy
- Scipy
"""

import sys
import math
import itertools
import heapq
import urllib

import numpy as np
import scipy.ndimage as ndimage
import scipy.misc
from PIL import Image
import cv2


DEBUG = True

# Percentage of length of the table. This is used to determine box around
# score blocks
SCORE_BLOCK_LENGTH = 0.045

# These affect how tightly the scores are boxed
SCORE_INNER_MARGIN = 0.015
SCORE_TO_MIDDLE_MARGIN = 0.17

# http://stackoverflow.com/questions/10948589/choosing-correct-hsv-values-for-opencv-thresholding-with-inranges

# The HSV value range that is used to get blue color of the image
BLUE_RANGE_MIN = np.array([80, 70, 70], np.uint8)
BLUE_RANGE_MAX = np.array([130, 255, 255], np.uint8)

ORANGE_RANGE_MIN = np.array([9, 40, 40], np.uint8)
ORANGE_RANGE_MAX = np.array([24, 255, 255], np.uint8)


def distance_between_points(p):
    """Takes tuple p which contains two points and returns the difference
    between them. p format: ((x1, y1), (x2, y2))
    """
    return math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][1])**2)


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


def find_score_from_image(image):
    """Find score from black and white image."""
    # find connected components
    labeled, nr_objects = ndimage.label(image)
    slices = ndimage.find_objects(labeled)

    # Center coordinates of objects
    objects = []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2

        objects.append((x_center, y_center))

    return find_score(objects)


def find_lower_long_side(corners):
    """From corner points, finds the lowest side of the rectangle."""
    sorted_by_y = sorted([(y, x) for x, y in corners])
    lowest = sorted_by_y[-1]
    distance_a = distance_between_points((lowest, sorted_by_y[1]))
    distance_b = distance_between_points((lowest, sorted_by_y[2]))
    if distance_a > distance_b:
        return lowest, sorted_by_y[1]

    # Flip back to x, y format, because they were sorted based on y
    point_a = (lowest[1], lowest[0])
    point_b = (sorted_by_y[2][1], sorted_by_y[2][0])
    return point_a, point_b


def calculate_line_rotation(point_a, point_b):
    """Calculates rotation of a line from point_a point of view."""
    x_diff = float(point_b[0] - point_a[0])
    y_diff = point_b[1] - point_a[1]
    return math.atan(y_diff / x_diff)


def find_table_ends(points):
    """Finds two shortest lines between points. These two lines are the ends
    of the table."""
    combinations = itertools.combinations(points, 2)
    ends = heapq.nsmallest(2, combinations, key=distance_between_points)
    ends.sort()
    return ends


def find_blue(image):
    """Takes image which is in HSV color space and returns new image which is
    black and white and all expect blue color is black.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv_image, BLUE_RANGE_MIN, BLUE_RANGE_MAX)


def find_orange(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv_image, ORANGE_RANGE_MIN, ORANGE_RANGE_MAX)


def middle_point(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def table_end_middles(end):
    """Calculates table end's middle points which can be used to square score
    blocks
    """
    middle = middle_point(end[0], end[1])
    middle_a = middle_point(end[0], middle)
    middle_b = middle_point(end[1], middle)

    add1 = calculate_coordinate_addition(middle_a, middle, SCORE_TO_MIDDLE_MARGIN)
    add2 = calculate_coordinate_addition(middle_b, middle, SCORE_TO_MIDDLE_MARGIN)

    middle_a = (middle_a[0] + add1[0], middle_a[1] + add1[1])
    middle_b = (middle_b[0] + add2[0], middle_b[1] + add2[1])

    return middle, middle_a, middle_b


def calculate_coordinate_addition(middle1, middle2, percent=SCORE_BLOCK_LENGTH):
    """Calculates the needed coordinate delta that should be added to a middle
    point so the score blocks can be cropped.
    """
    x_diff = int((middle2[0] - middle1[0]) * percent)
    y_diff = int((middle2[1] - middle1[1]) * percent)
    return x_diff, y_diff


def calculate_score_box(middle_a, middle_b, addition):
    box_a = (middle_a[0] + addition[0], middle_a[1] + addition[1])
    box_b = (middle_b[0] + addition[0], middle_b[1] + addition[1])

    return (middle_b, middle_a, box_a, box_b)


def subimage(image, centre, theta, width, height):
    output_image = cv2.cv.CreateImage((width, height), 2, 3)
    mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                       [np.sin(theta), np.cos(theta), centre[1]]])
    map_matrix_cv = cv2.cv.fromarray(mapping)
    cv2.cv.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
    return output_image



def rotate_image(image, angle, rotation_point=(0, 0)):
    rot_mat = cv2.getRotationMatrix2D(rotation_point, angle, 1)

    shape = image.shape[1], image.shape[0]
    result = cv2.warpAffine(image, rot_mat, shape, flags=cv2.INTER_LINEAR)
    return result


def rad_to_deg(r):
    return 180.0 * r / math.pi


def straighten_table(image):
    # Find table corners
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bw_image = find_blue(hsv_image)

    np_nonzero = bw_image.nonzero()
    non_zero_pixels = np.array(zip(np_nonzero[0], np_nonzero[1]))

    rect = cv2.minAreaRect(non_zero_pixels)
    precise_corners = cv2.cv.BoxPoints(rect)
    corners = np.int0(np.around(precise_corners))

    # Find lowest long side of the table and straigthen based on it
    lower_a, lower_b = find_lower_long_side(corners)

    rotation = rad_to_deg(calculate_line_rotation(lower_a, lower_b))
    # Rotate based on the other end of the line
    rotated_image = rotate_image(image, rotation, rotation_point=lower_a)
    return rotated_image


def find_score_boxes(corners):
    """Finds the bounding boxes of table's ends."""
    ends = find_table_ends(corners)

    end1, end2 = ends
    middle1, middle1_a, middle1_b = table_end_middles(end1)
    middle2, middle2_a, middle2_b = table_end_middles(end2)

    add1 = calculate_coordinate_addition(middle1, middle2, SCORE_INNER_MARGIN)
    add2 = calculate_coordinate_addition(middle2, middle1, SCORE_INNER_MARGIN)

    middle1_a = (middle1_a[0] + add1[0], middle1_a[1] + add1[1])
    middle1_b = (middle1_b[0] + add1[0], middle1_b[1] + add1[1])

    middle2_a = (middle2_a[0] + add2[0], middle2_a[1] + add2[1])
    middle2_b = (middle2_b[0] + add2[0], middle2_b[1] + add2[1])

    addition1 = calculate_coordinate_addition(middle1, middle2)
    # This is basically opposite direction than addition1
    addition2 = calculate_coordinate_addition(middle2, middle1)

    # Calculate the bounding boxes for score blocks
    end1_box = calculate_score_box(middle1_a, middle1_b, addition1)
    end2_box = calculate_score_box(middle2_a, middle2_b, addition2)
    return end1_box, end2_box


def crop_boxes(image, boxes):
    """Crops given boxes from image, boxes contain all box corners, where
    first is top left and third is bottom right."""
    crops = []

    for box in boxes:
        tl, br = find_crop_corners(box)
        x1, y1 = tl[0], tl[1]
        x2, y2 = br[0], br[1]
        cropped = image[y1:y2, x1:x2]
        cropped = cv2.transpose(cropped)
        cropped = cv2.flip(cropped, 0)
        crops.append(cropped)

    return crops


def find_crop_corners(box):
    """Finds top left and bottom right from 4 coordinates."""
    sorted_by_x = sorted(box)
    left_coords = sorted_by_x[0:2]
    right_coords = sorted_by_x[2:]

    tl_y, tl_x = min([(y, x) for x, y in left_coords])
    br_y, br_x = max([(y, x) for x, y in right_coords])

    return (tl_x, tl_y), (br_x, br_y)


def draw_points(image, points):
    im = image.copy()

    for point in points:
        cv2.circle(im, tuple(point), 10, (0, 0, 255))

    return im


def get_score(image):
    new_size = (image.size[0] * 2, image.size[1] * 2)
    big = Image.new('RGB', new_size)

    # Place the actual image in the middle of the new empty canvas
    # The position was tested pretty much empirically
    big.paste(image, (int(image.size[0] / 1.5), image.size[1] / 2))

    array = np.array(big)
    # Convert RGB to BGR
    cv_image = array[:, :, ::-1].copy()

    rotated_image = straighten_table(cv_image)

    # Find table corners
    bw_image = find_blue(rotated_image)
    non_zero_pixels = cv2.findNonZero(bw_image)
    rect = cv2.minAreaRect(non_zero_pixels)
    precise_corners = cv2.cv.BoxPoints(rect)
    corners = np.int0(np.around(precise_corners))

    sorted_corners = [(x, y) for x, y in corners]
    tl, br = find_crop_corners(sorted_corners)
    sorted_corners.remove(tl)
    sorted_corners.remove(br)
    bl, tr = min(sorted_corners), max(sorted_corners)

    # Find bounding boxes for scores
    score_boxes = find_score_boxes([tl, bl, br, tr])
    if DEBUG:
        points = []
        for box in score_boxes:
            points += box

        # Add table corners
        points += [(x, y) for x, y in corners]

        im = draw_points(rotated_image, points)
        cv2.imwrite('debug/debug.jpg', im)

    score1_crop, score2_crop = crop_boxes(rotated_image, score_boxes)

    if DEBUG:
        cv2.imwrite('debug/crop1.jpg', score1_crop)
        cv2.imwrite('debug/crop2.jpg', score2_crop)


    # Find white block score
    bw_image = find_orange(score1_crop)

    if DEBUG:
        cv2.imwrite('debug/bw1.jpg', bw_image)

    white_score = 10 - find_score_from_image(bw_image)

    # Find blue block score
    image = Image.fromarray(score2_crop).convert('L')
    image = np.array(image, dtype=int)

    # Threshold
    T = 160
    bw_image = image > T
    if DEBUG:
        scipy.misc.imsave('debug/bw2.jpg', bw_image)

    blue_score = find_score_from_image(bw_image)

    return white_score, blue_score


def print_score(file_name):
    image = Image.open(file_name).convert('RGB')
    white, blue = get_score(image)
    print 'White   Blue'
    print '%s    -  %s' % (white, blue)


def main():
    file_name = sys.argv[1]

    if file_name == 'stream':
        while True:
            urllib.urlretrieve('http://10.3.100.96:8080/?action=snapshot', 'stream.jpg')
            print_score('stream.jpg')


    print_score(file_name)


if __name__ == '__main__':
    main()
