from utils.constants import HOG_HISTOGRAM_BINS, HOG_HISTOGRAM_RANGE, HOG_KERNEL
from utils.utils import gradient, magnitude, orientation, weighted_histogram


def hog(img):
    magnitudes = magnitude(img, HOG_KERNEL)
    orientations = orientation(img, HOG_KERNEL)
    hist = weighted_histogram(magnitudes, HOG_HISTOGRAM_RANGE, HOG_HISTOGRAM_BINS, orientations)
    print(hist)
    return hist
