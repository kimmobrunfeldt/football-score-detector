"""
===================
Canny edge detector
===================

The Canny filter is a multi-stage edge detector. It uses a filter based on the
derivative of a Gaussian in order to compute the intensity of the gradients.The
Gaussian reduces the effect of noise present in the image. Then, potential
edges are thinned down to 1-pixel curves by removing non-maximum pixels of the
gradient magnitude. Finally, edge pixels are kept or removed using hysteresis
thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the
noisier the image, the greater the width), and the low and high threshold for
the hysteresis thresholding.

"""

import numpy as np
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import skimage.data

from skimage import filter
from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing, square
from skimage.measure import regionprops

image = Image.open('test.jpg').convert('L') # Grayscale

im = np.array(image, dtype=int)
thresh = skimage.filter.threshold_otsu(im)
bw = im > thresh

scipy.misc.toimage(bw).save('new.jpg')
