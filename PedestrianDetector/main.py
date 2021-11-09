from matplotlib import pyplot as plt
from utils.constants import HOG_HISTOGRAM_BINS

from utils.hog import hog
from utils.utils import open_image


def main():
    img = open_image('./data/test.jpg')
    result = hog(img)
    plt.bar(range(9), result)
    plt.show()


if __name__ == '__main__':
    main()
