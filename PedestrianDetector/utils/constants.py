from utils.utils import Kernel


HOG_KERNEL = Kernel(
    [
        [0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0]
    ],
    [
        [0, -1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ]
)

HOG_HISTOGRAM_BINS = 9
HOG_HISTOGRAM_RANGE = (0,180)
