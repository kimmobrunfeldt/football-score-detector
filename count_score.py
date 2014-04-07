"""Usage: python count_score.py <image> [--debug]

Examples:
    python count_score.py testdata/real.jpg
    python count_score.py --debug testdata/real.jpg
"""

import heapq
import itertools
import json
import logging
import math
import traceback
import sys

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

# Area limits for found score blocks
# The score blocks' area must be in these bounds or it is dumped as a random
# 'trash'
MIN_SCORE_AREA = 5
MAX_SCORE_AREA = 120

# http://stackoverflow.com/questions/10948589/choosing-correct-hsv-values-for-opencv-thresholding-with-inranges

# The HSV value range that is used to get blue color of the image
BLUE_RANGE_MIN = np.array([80, 70, 70], np.uint8)
BLUE_RANGE_MAX = np.array([130, 255, 255], np.uint8)

ORANGE_RANGE_MIN = np.array([9, 40, 40], np.uint8)
ORANGE_RANGE_MAX = np.array([24, 255, 255], np.uint8)


def main():
    argv = sys.argv[:]
    if len(argv) < 2 or '-h' in argv or '--help' in argv:
        print __doc__
        sys.exit(1)

    level = logging.ERROR
    if '--debug' in argv:
        argv.remove('--debug')
        level = logging.DEBUG
    setup_logging(logging.getLogger(''), level=level)

    file_name = argv[1]
    print_score(file_name)


def setup_logging(root_logger, level=logging.DEBUG):
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    format = '%(message)s'

    formatter = logging.Formatter(format)
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)


def print_score(file_name):
    image = Image.open(file_name).convert('RGB')
    try:
        score = get_score(image)
    except Exception, e:
        out = {
            'success': False,
            'error': str(e),
            'stacktrace': traceback.format_exc()
        }
        print json.dumps(out)
        return

    score['success'] = True
    print json.dumps(score)


def get_score(image):
    """Finds score data from given RGB image which is Image object"""
    data = {}

    # Create new bigger canvas where the table image can be rotated.
    # This is needed because otherwise the table might be rotated outside
    # of image bounds.
    new_size = (image.size[0] * 2, image.size[1] * 2)
    big = Image.new('RGB', new_size)

    # Place the actual image in the middle of the new empty canvas
    # The position was tested pretty much empirically
    big.paste(image, (int(image.size[0] / 1.5), image.size[1] / 2))

    array = np.array(big)
    # Convert RGB to BGR, because OpenCV uses BGR
    cv_image = array[:, :, ::-1].copy()

    if DEBUG:
        cv2.imwrite('debug/large.jpg', cv_image)

    logging.debug('Straightening table..')
    rotated_image = straighten_table(cv_image)

    if DEBUG:
        cv2.imwrite('debug/large_straight.jpg', rotated_image)

    # Find table corners
    logging.debug('Finding table corners..')
    bw_image = find_blue(rotated_image)

    if DEBUG:
        cv2.imwrite('debug/found_blue_large.jpg', bw_image)

    non_zero_pixels = cv2.findNonZero(bw_image)
    rect = cv2.minAreaRect(non_zero_pixels)
    precise_corners = cv2.cv.BoxPoints(rect)
    corners = np.int0(np.around(precise_corners))

    sorted_corners = [(x, y) for x, y in corners]
    tl, br = find_crop_corners(sorted_corners)
    sorted_corners.remove(tl)
    sorted_corners.remove(br)
    bl, tr = min(sorted_corners), max(sorted_corners)

    if DEBUG:
        label_tl_im = draw_label(rotated_image, tl, 'A')
        label_tl_im = draw_points(label_tl_im, [tl])
        cv2.imwrite('debug/corner_a.jpg', label_tl_im)

        label_bl_im = draw_label(rotated_image, bl, 'B')
        label_bl_im = draw_points(label_bl_im, [bl])
        cv2.imwrite('debug/corner_b.jpg', label_bl_im)

        label_br_im = draw_label(rotated_image, br, 'C')
        label_br_im = draw_points(label_br_im, [br])
        cv2.imwrite('debug/corner_c.jpg', label_br_im)

        label_tr_im = draw_label(rotated_image, tr, 'D')
        label_tr_im = draw_points(label_tr_im, [tr])
        cv2.imwrite('debug/corner_d.jpg', label_tr_im)

        labels = draw_label(rotated_image, tl, 'A')
        labels = draw_label(labels, bl, 'B')
        labels = draw_label(labels, br, 'C')
        labels = draw_label(labels, tr, 'D')
        labels = draw_points(labels, [tl, bl, br, tr])
        cv2.imwrite('debug/corner_labels.jpg', labels)

    # Find bounding boxes for scores
    logging.debug('Finding and cropping score blocks..')
    score_boxes = find_score_boxes([tl, bl, br, tr], rotated_image)
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
        cv2.imwrite('debug/left_score_blocks.jpg', score1_crop)
        cv2.imwrite('debug/right_score_blocks.jpg', score2_crop)

    logging.debug('Counting left score..')
    bw_image = find_orange(score1_crop)

    if DEBUG:
        cv2.imwrite('debug/left_score_blocks_black_white.jpg', bw_image)

    objects = find_object_centers(bw_image)
    data['leftScore'] = 10 - find_score(objects)

    if DEBUG:
        centers_im = draw_points(score1_crop, objects, radius=2)
        cv2.imwrite('debug/centers_left.jpg', centers_im)

        create_score_image(data['leftScore'], 'left')

    logging.debug('Counting right score..')
    image = Image.fromarray(score2_crop).convert('L')
    image = np.array(image, dtype=int)

    # Threshold
    T = 160
    bw_image = image > T
    if DEBUG:
        scipy.misc.imsave('debug/right_score_blocks_black_white.jpg', bw_image)

    objects = find_object_centers(bw_image)
    data['rightScore'] = find_score(objects)

    if DEBUG:
        centers_im = draw_points(score2_crop, objects, radius=2)
        cv2.imwrite('debug/centers_right.jpg', centers_im)

        create_score_image(data['rightScore'], 'right')

    return data


def straighten_table(image):
    """Rotates a given image so that the football table is straight.

    In English:
        - Find the table and determine its corners
        - From corners, find lower long side of the table
        - Calculate rotation from the line and rotate the image
    """
    bw_image = find_blue(image)

    if DEBUG:
        cv2.imwrite('debug/found_blue.jpg', bw_image)

    non_zero_pixels = cv2.findNonZero(bw_image)

    rect = cv2.minAreaRect(non_zero_pixels)
    precise_corners = cv2.cv.BoxPoints(rect)
    corners = np.int0(np.around(precise_corners))

    if DEBUG:
        corners_im = draw_points(image, corners)
        cv2.imwrite('debug/found_corners.jpg', corners_im)

    # Find lowest long side of the table and straigthen based on it
    lower_a, lower_b = find_lower_long_side(corners)

    if DEBUG:
        lower_line_im = draw_lines(image, [[lower_a, lower_b]])
        cv2.imwrite('debug/lower_long_side.jpg', lower_line_im)

    rotation = rad_to_deg(calculate_line_rotation(lower_a, lower_b))
    # Rotate based on the other end of the line
    rotated_image = rotate_image(image, rotation, rotation_point=lower_a)
    return rotated_image


def find_lower_long_side(corners):
    """From corner points, finds the lowest long side of a rectangle.

    For example:

            d
           / \
          /   c          a-----d
         a   /     and   |     |
          \ /            b-----c
           b

    Both cases would return points b and c.
    """
    sorted_by_y = sorted([(y, x) for x, y in corners])
    lowest = sorted_by_y[-1]
    distance_a = distance_between_points((lowest, sorted_by_y[1]))
    distance_b = distance_between_points((lowest, sorted_by_y[2]))

    point_a = lowest
    point_b = sorted_by_y[1] if distance_a > distance_b else sorted_by_y[2]

    # Flip back to x, y format, because they were sorted based on y
    return flip(point_a), flip(point_b)


def find_blue(image):
    """Takes image which is in BGR color space and returns new image which is
    black and white and all blue regions are white.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv_image, BLUE_RANGE_MIN, BLUE_RANGE_MAX)


def find_orange(image):
    """Takes image which is in BGR color space and returns new image which is
    black and white and all orange regions are white.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv_image, ORANGE_RANGE_MIN, ORANGE_RANGE_MAX)


def find_crop_corners(box):
    """Finds top left and bottom right from 4 coordinates."""
    sorted_by_x = sorted(box)
    left_coords = sorted_by_x[0:2]
    right_coords = sorted_by_x[2:]

    tl_y, tl_x = min([(y, x) for x, y in left_coords])
    br_y, br_x = max([(y, x) for x, y in right_coords])

    return (tl_x, tl_y), (br_x, br_y)


def find_score_boxes(corners, image):
    """Finds the bounding boxes for scores based on table's corners.

    image is just passed for debugging purposes
    """
    ends = find_table_ends(corners)

    if DEBUG:
        ends_im = draw_lines(image, ends)
        cv2.imwrite('debug/table_ends.jpg', ends_im)

    end1, end2 = ends
    middle1, middle1_a, middle1_b = table_end_middles(end1)
    middle2, middle2_a, middle2_b = table_end_middles(end2)

    if DEBUG:
        middle_im = draw_points(image, [middle1, middle2])
        cv2.imwrite('debug/table_middle.jpg', middle_im)

        middle_im = draw_points(image, [middle1_a, middle1_b, middle2_a, middle2_b])
        cv2.imwrite('debug/table_middles_of_middle.jpg', middle_im)

    add1 = calculate_coordinate_addition(middle1, middle2, SCORE_INNER_MARGIN)
    add2 = calculate_coordinate_addition(middle2, middle1, SCORE_INNER_MARGIN)

    middle1_a = (middle1_a[0] + add1[0], middle1_a[1] + add1[1])
    middle1_b = (middle1_b[0] + add1[0], middle1_b[1] + add1[1])

    middle2_a = (middle2_a[0] + add2[0], middle2_a[1] + add2[1])
    middle2_b = (middle2_b[0] + add2[0], middle2_b[1] + add2[1])

    if DEBUG:
        middle_im = draw_points(image, [middle1_a, middle1_b, middle2_a, middle2_b])
        cv2.imwrite('debug/table_middles_of_middle_add.jpg', middle_im)

    addition1 = calculate_coordinate_addition(middle1, middle2)
    # This is basically opposite direction than addition1
    addition2 = calculate_coordinate_addition(middle2, middle1)

    # Calculate the bounding boxes for score blocks
    end1_box = calculate_score_box(middle1_a, middle1_b, addition1)
    end2_box = calculate_score_box(middle2_a, middle2_b, addition2)

    if DEBUG:
        score_box_im = draw_points(image, end1_box)
        score_box_im = draw_points(score_box_im, end2_box)
        cv2.imwrite('debug/table_score_box.jpg', score_box_im)

    return end1_box, end2_box


def find_table_ends(points):
    """Finds two shortest lines between points. These two lines are the ends
    of the table.
    """
    combinations = itertools.combinations(points, 2)
    ends = heapq.nsmallest(2, combinations, key=distance_between_points)
    ends.sort()
    return ends


def find_object_centers(image):
    """Find score from black and white image.

    Image should contain 12 white dots placed from left to right.
    Two outermost dots are not counted, they keep the score blocks in place.
    """
    # Find connected components
    labeled, nr_objects = ndimage.label(image)
    slices = ndimage.find_objects(labeled)

    # Center coordinates of objects
    objects = []
    for dy, dx in slices:
        # Skip too small or big regions
        area = abs((dx.stop - dx.start) * (dy.stop - dy.start))
        logging.debug('Found possible score block. Area: %s' % area)
        if area < MIN_SCORE_AREA or area > MAX_SCORE_AREA:
            logging.info('Skip object with area %s' % area)
            continue

        x_center = (dx.start + dx.stop - 1) / 2
        y_center = (dy.start + dy.stop - 1) / 2

        objects.append((x_center, y_center))

    logging.debug('Found %s objects which have correct area' % len(objects))

    if len(objects) != 12:
        err = 'Cannot find correct amount of score blocks. '
        err += 'Expected 12, but found %s' % len(objects)
        raise ValueError(err)

    return objects


def crop_boxes(image, boxes):
    """Crops given boxes from image, boxes contain all box corners, where
    first is top left and third is bottom right.
    """
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


def find_score(points):
    """Returns the score from points."""
    # Put points to left-to-right order(ordered by x coordinate)
    points.sort()
    middle_distance = average_distance_between_score_dots(points)

    score = 0
    for i, point in enumerate(points[:-1]):
        if distance_between_points((point, points[i + 1])) > middle_distance:
            break

        score += 1
    return score


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
    """Calculate bounding box for score blocks.

    middle_a: Upper 'middle of middle' point
    middle_b: Lower 'middle of middle' point
    addition: tuple (delta_x, delta_y)
              Indicates how much should be added to each middle of middle point
              to create the box.
    """
    box_a = (middle_a[0] + addition[0], middle_a[1] + addition[1])
    box_b = (middle_b[0] + addition[0], middle_b[1] + addition[1])

    return (middle_b, middle_a, box_a, box_b)

# Generic OpenCV functions

def rotate_image(image, angle, rotation_point=(0, 0)):
    """Rotates image.

    angle: Rotation angle in degrees
    rotation_point: Rotation origin coordinate.

    Returns rotated image.
    """
    rot_mat = cv2.getRotationMatrix2D(rotation_point, angle, 1)

    shape = image.shape[1], image.shape[0]
    result = cv2.warpAffine(image, rot_mat, shape, flags=cv2.INTER_LINEAR)
    return result


def draw_points(image, points, radius=3):
    """Draws points to a given image. Returns copy of image, original is not
    modified.
    """
    im = image.copy()

    for point in points:
        cv2.circle(im, tuple(point), radius, (0, 0, 255), radius)

    return im


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=3, thickness=2):
    """Draws label for point to a given image.
    Returns copy of image, original is not modified.
    """
    im = image.copy()

    # http://docs.opencv.org/modules/core/doc/drawing_functions.html#gettextsize
    # Returns bounding box and baseline -> ((width, height), baseline)
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    label_top_left = (x - size[0] / 2, y - size[1] / 2)
    cv2.putText(im, label, label_top_left, font, font_scale, (0, 0, 255), thickness)

    return im


def draw_lines(image, lines):
    """Draws lines to a given image. Lines are in format
    [[(x1, y1), (x2, y2)]]

    Returns copy of image, original is not modified.
    """
    im = image.copy()

    for p1, p2 in lines:
        im = draw_points(im, [p1, p2])
        cv2.line(im, p1, p2, (0, 0, 255), 3)

    return im


def create_score_image(score, side):
    blank_image = np.zeros((40, 200, 3), np.uint8)
    text_im = draw_label(blank_image, (100, 32), str(score), font_scale=0.85)
    cv2.imwrite('debug/%s_score.jpg' % side, text_im)

# Generic math functions

def flip(coord):
    """Flips two coordinates. (a, b) -> (b, a)"""
    a, b = coord
    return b, a


def distance_between_points(p):
    """Takes tuple p which contains two points and returns the difference
    between them. p format: ((x1, y1), (x2, y2))
    """
    return math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][1])**2)


def average_distance_between_score_dots(points):
    """Calculates the average distance between given points.

    For example with a, b and c points:

          d1    d2
        a----b------c

    Would return: (d1 + d2) / 2
    """
    distances = []
    for i, point in enumerate(points[:-1]):
        distances.append(distance_between_points((point, points[i + 1])))

    return float(sum(distances)) / len(distances)


def calculate_line_rotation(point_a, point_b):
    """Calculates rotation of a line from point_a point of view."""
    x_diff = float(point_b[0] - point_a[0])
    y_diff = point_b[1] - point_a[1]
    return math.atan(y_diff / x_diff)


def middle_point(p1, p2):
    """Returns point between p1 and p2"""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def rad_to_deg(r):
    """Converts radians to degrees"""
    return 180.0 * r / math.pi


if __name__ == '__main__':
    main()
