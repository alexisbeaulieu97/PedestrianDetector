from collections import namedtuple

import numpy as np
from PIL import Image
from scipy import signal

Gradient = namedtuple("Gradient", ["dx", "dy"])
Kernel = namedtuple("Kernel", ["x", "y"])


def gradient(img, kernel):
    return Gradient(
        signal.fftconvolve(img, kernel.x), signal.fftconvolve(img, kernel.y)
    )


def magnitude(img, kernel):
    grad = gradient(img, kernel)
    return np.sqrt(grad.dx ** 2 + grad.dy ** 2)


def orientation(img, kernel):
    grad = gradient(img, kernel)
    rad = np.arctan2(grad.dy, grad.dx)
    return np.degrees(rad) + 180


def open_image(img_path: str, dimensions: tuple):
    img = Image.open(img_path).convert("L")
    img = img.resize(dimensions, resample=Image.BICUBIC)
    return np.array(img, dtype=np.float)


def weighted_histogram(data, weights, n_bins, hist_range):
    data = np.array(data).flatten()
    weights = np.array(weights).flatten()
    hist = np.zeros(n_bins)
    step_size = (hist_range[1] - hist_range[0]) // n_bins
    for i, x in enumerate(weights):
        i = int(x / step_size) % n_bins
        hist[i] += data[i]
    return np.array(hist, dtype=np.float)


def split_blocks(arr, zone_size):
    arr = np.array(arr)
    split = np.empty(np.floor_divide(arr.shape, zone_size), dtype=object)
    for i in range(0, split.shape[0]):
        for j in range(0, split.shape[1]):
            split[i, j] = np.array(
                arr[
                    i * zone_size : i * zone_size + zone_size,
                    j * zone_size : j * zone_size + zone_size,
                ]
            )
    return split
