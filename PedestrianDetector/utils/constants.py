import os.path
import sys

from utils.utils import Kernel

HOG_KERNEL = Kernel(
    [[0, 0, 0], [-1, 0, 1], [0, 0, 0]], [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
)

HOG_HISTOGRAM_BINS = 9
HOG_BLOCK_SIZE = 8
HOG_HIST_NORMALIZATION_SIZE = 2
HOG_HISTOGRAM_RANGE = (0, 180)
IMG_SIZE = (64, 128)

TRAIN_BUFFER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(sys.modules["__main__"].__file__)), "train_buffer"
)

TRAIN_BUFFER_EXT = ".txt"
