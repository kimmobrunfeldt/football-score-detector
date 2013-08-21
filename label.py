
import sys

import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops


def find_regions(image, min_area=0):
    """Finds regions from image which is numpy array.

    min_area: Minimum region area that is counted as region

    Return format:
    [
        {'area': area, 'box': (topLeft, bottomLeft, bottomRight, topRight)}
        ...
    ]
    """
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(6))

    regions = []

    for region in regionprops(bw, ['Area', 'BoundingBox']):

        # skip small regions
        if region['Area'] >= min_area:
            regions.append({'area': region['Area'], 'box': region['BoundingBox']})

    return regions


def main():
    file_name = sys.argv[1]
    image = Image.open(file_name).convert('L')  # Grayscale
    im = np.array(image, dtype=int)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(Image.open(file_name).convert('RGB'))

    regions = find_regions(im)
    regions.sort()

    minr, minc, maxr, maxc = regions[-1]['box']
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    main()


