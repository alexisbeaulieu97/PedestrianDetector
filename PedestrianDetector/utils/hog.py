from utils.constants import (
    HOG_HIST_NORMALIZATION_SIZE,
    HOG_BLOCK_SIZE,
    HOG_HISTOGRAM_BINS,
    HOG_HISTOGRAM_RANGE,
    HOG_KERNEL,
)
from utils.utils import magnitude, orientation, split_blocks, weighted_histogram
import numpy as np


def hog(img):
    # calculate the magnitudes
    magnitudes = magnitude(img, HOG_KERNEL)

    # calculate the orientations
    orientations = orientation(img, HOG_KERNEL)

    # transform results into an array of 8x8 zones
    magnitude_zones = split_blocks(magnitudes, HOG_BLOCK_SIZE)
    orientation_zones = split_blocks(orientations, HOG_BLOCK_SIZE)

    # calculate the histogram of each zone
    hist = np.empty(
        magnitude_zones.shape,
        dtype=object,
    )
    for i in range(0, magnitude_zones.shape[0]):
        for j in range(0, magnitude_zones.shape[1]):
            hist[i, j] = weighted_histogram(
                magnitude_zones[i, j],
                orientation_zones[i, j],
                HOG_HISTOGRAM_BINS,
                HOG_HISTOGRAM_RANGE,
            )

    # normalize the histogram in 16x16 zones
    normalized_hist = split_blocks(hist, HOG_HIST_NORMALIZATION_SIZE)
    for i in range(0, normalized_hist.shape[0]):
        for j in range(0, normalized_hist.shape[1]):
            sq = normalized_hist[i, j] ** 2
            norm = np.sqrt(np.sum(np.sum(sq)))
            normalized_hist[i, j] = np.divide(normalized_hist[i, j], norm)

    # Concatenate results into feature array
    return np.concatenate(np.concatenate(np.concatenate(normalized_hist.flatten())))
