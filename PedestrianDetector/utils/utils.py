from collections import namedtuple

import numpy as np
from PIL import Image
from scipy import signal

Gradient = namedtuple('Gradient', ['dx', 'dy'])
Kernel = namedtuple('Kernel', ['x', 'y'])


def gradient(img, kernel):
    return Gradient(
        signal.fftconvolve(img, kernel.x),
        signal.fftconvolve(img, kernel.y)
    )


def magnitude(img, kernel):
    grad = gradient(img, kernel)
    return np.sqrt(grad.dx**2 + grad.dy**2)


def orientation(img, kernel):
    grad = gradient(img, kernel)
    rad = np.arctan2(grad.dy, grad.dx)
    return np.degrees(rad) + 180


def open_image(img_path: str):
    img = Image.open(img_path).convert('L')
    return np.asarray(img, dtype=float)


def weighted_histogram(data, hist_range, n_bins, weights):
    hist = np.zeros(n_bins)
    step_size = (hist_range[1] - hist_range[0]) // n_bins
    it = np.nditer(weights, flags=['multi_index'])
    for x in it:
        i = int(x/step_size) % n_bins
        hist[i] += data[it.multi_index]
    return np.array(hist, dtype=np.float)
