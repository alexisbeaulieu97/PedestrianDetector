from matplotlib import pyplot as plt
from utils.constants import HOG_HISTOGRAM_BINS

from utils.hog import hog
from utils.utils import open_image
from utils.constants import IMG_SIZE


def main():
    img = open_image("./data/test.jpg", IMG_SIZE)
    histograms = hog(img)


if __name__ == "__main__":
    main()
