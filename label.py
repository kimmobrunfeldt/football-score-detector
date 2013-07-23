
import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops


def find_regions(image, min_area=10):
    """Finds regions from image which is numpy array.

    min_area: Minimum region area that is counted as region

    Return format:
    [
        {'area': area, 'box': (topLeft, bottomLeft, bottomRight, topRight)}
        ...
    ]
    """
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    regions = []

    for region in regionprops(label_image, ['Area', 'BoundingBox']):

        # skip small regions
        if region['Area'] >= min_area:
            regions.append({'area': region['Area'], 'box': region['BoundingBox']})

    return regions


def main():
    image = Image.open('test2.jpg').convert('L')  # Grayscale
    im = np.array(image, dtype=int)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(Image.open('test2.jpg').convert('RGB'))

    regions = find_regions(im)
    regions.sort()

    minr, minc, maxr, maxc = regions[-1]['box']
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    main()


