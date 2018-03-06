from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
from tensorflow.contrib import predictor
from scipy import misc
import pandas as pd
import numpy as np

SAVED_MODEL_DIR = './model/1519607690'
TEST_IMAGES_DIR = './test_images'


def load_test_data():

    testImages = os.listdir(TEST_IMAGES_DIR)

    testData = []

    for testImage in testImages:
        image = misc.imread(os.path.join(TEST_IMAGES_DIR, testImage))
        image = misc.imresize(image, size=(64, 64))
        image = (image - (255 / 2.0)) / 255

        testData.append(np.array(image, dtype=float).flatten())

    testData = pd.DataFrame(testData)

    print(testData)

    return testData


def main(argv):
    with tf.device('/gpu:0'):
        fruits_predictor = predictor.from_saved_model(SAVED_MODEL_DIR)

        print('Model loaded...')
        testData = load_test_data()
        predictions = fruits_predictor(testData)

        print(predictions)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
