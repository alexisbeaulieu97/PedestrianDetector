import os.path
from os import listdir
from shutil import rmtree

import click
import numpy as np
from sklearn import svm

from utils.constants import IMG_SIZE, TRAIN_BUFFER_EXT, TRAIN_BUFFER_PATH
from utils.hog import hog
from utils.utils import open_image


@click.group()
def cli():
    ...


@cli.command()
@click.argument("train_path", type=click.Path(exists=True, file_okay=False))
def train(train_path: str):
    # initialize the train buffer
    os.makedirs(TRAIN_BUFFER_PATH, exist_ok=True)

    # train using provided data
    for data_class in listdir(train_path):
        class_path = os.path.realpath(os.path.join(train_path, data_class))
        for filebase in listdir(class_path):
            filepath = os.path.join(class_path, filebase)
            train_file(filepath, data_class)


@cli.command()
@click.argument("testpath", type=click.Path(exists=True))
def test(testpath):
    X_train = []
    y_train = []

    # load all the trained data
    for data_class in listdir(TRAIN_BUFFER_PATH):
        data_class_dir = os.path.join(TRAIN_BUFFER_PATH, data_class)
        for i in listdir(data_class_dir):
            datafile = os.path.join(data_class_dir, i)
            data = np.loadtxt(datafile, float)
            X_train.append(data)
            y_train.append(data_class)

    # create svm from the trained data
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)

    # get test data features
    X_test = []
    if os.path.isdir(testpath):
        for filename in listdir(testpath):
            filepath = os.path.join(testpath, filename)
            X_test.append((filename, get_features(filepath)))
    else:
        X_test = [(os.path.basename(testpath), get_features(testpath))]

    # predict data class using svm
    for test_file, test_data in X_test:
        y_test = clf.predict([test_data])
        click.echo(f"{test_file} --> {y_test[0]}")


@cli.command()
def clear():
    rmtree(TRAIN_BUFFER_PATH, ignore_errors=True)


def train_file(filepath: str, data_class: str):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(os.path.join(TRAIN_BUFFER_PATH, data_class), exist_ok=True)
    feature_path = (
        os.path.join(TRAIN_BUFFER_PATH, data_class, filename) + TRAIN_BUFFER_EXT
    )

    if os.path.exists(feature_path):
        return

    features = get_features(filepath)
    np.savetxt(feature_path, features)


def get_features(filepath):
    img = open_image(filepath, IMG_SIZE)
    return hog(img)


if __name__ == "__main__":
    cli()
