# !rm - r . / model / *

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import os
import numpy as np
import pandas as pd
# from giextractor import GoogleImageExtractor
from scipy import misc
from matplotlib import pyplot as plt


# def extract_images():
#     # Downlad images for the training and the test set
#     imageExtractor = GoogleImageExtractor(
#         imageQuery='apple fruit', imageCount=1000, destinationFolder=ImageClassifier.FRUIT_IMAGES_DIR)
#     imageExtractor.extract_images()

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


class ImageClassifier:

    _labels = {'apple': 0,
               'grapes': 1,
               'banana': 2,
               'mango': 3}

    FRUIT_IMAGES_DIR = './images/'
    BATCH_SIZE = 100

    def train_model(self):
        (trainX, trainY), (testX, testY) = self._load_data(0.8)

        featureColumns = self._get_feature_columns(trainX)

        estimatorConfig = tf.estimator.RunConfig(
            log_step_count_steps=100)

        classifier = tf.estimator.DNNClassifier(
            feature_columns=featureColumns,
            hidden_units=[50, 10, 10],
            n_classes=4,
            model_dir='./model/fruits_classifier',
            optimizer=tf.train.AdamOptimizer(0.001),
            config=estimatorConfig)

        classifier.train(
            input_fn=lambda: self._train_input_fn(
                trainX, trainY, self.BATCH_SIZE),
            steps=1000)

        eval_result = classifier.evaluate(
            input_fn=lambda: self._eval_input_fn(testX, testY, self.BATCH_SIZE))

        print('\nTest Set Accuracy: {accuracy: 0.3f}\n'.format(eval_result))

        saved_model_dir = self._save_model(classifier, featureColumns)

        print('Model saved at {}'.format(saved_model_dir))

    def _save_model(self, estimator, featureColumns):
        featureSpec = tf.feature_column.make_parse_example_spec(featureColumns)
        input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            featureSpec)

        return estimator.export_savedmodel(
            './model/fruits_classifier', input_receiver_fn, as_text=True)

    def _load_data(self, trainPercentage):

        trainData = []
        trainLabel = []
        testData = []
        testLabel = []

        for imageDir in os.listdir(self.FRUIT_IMAGES_DIR):
            imagesList = os.listdir(self.FRUIT_IMAGES_DIR + imageDir)

            # Separate the train and test set based on the train percentage specified
            trainEndIndex = int(len(imagesList) * trainPercentage)

            trainImageList = imagesList[0:trainEndIndex]
            testImageList = imagesList[trainEndIndex + 1:]

            print('{}: {}, training set: {}, test set: {}'.format(
                imageDir, len(imagesList), len(trainImageList), len(testImageList)))

            for trainImage in trainImageList:
                image = misc.imread(os.path.join(
                    self.FRUIT_IMAGES_DIR, imageDir, trainImage))
                image = misc.imresize(image, size=(64, 64))
                image = (image - (255 / 2.0)) / 255

                print(rebin(image, (10, 10)))
                trainData.append(np.array(image, dtype=float).flatten())
                trainLabel.append(self._get_label(imageDir))

            print('train data: {}, train label: {}'.format(
                len(trainData), len(trainLabel)))

            for testImage in testImageList:
                image = misc.imread(os.path.join(
                    self.FRUIT_IMAGES_DIR, imageDir, testImage))
                image = misc.imresize(image, size=(64, 64))

                image = (image - (255 / 2.0)) / 255
                testData.append(np.array(image, dtype=float).flatten())
                testLabel.append(self._get_label(imageDir))

            print('test data: {}, test label: {}'.format(
                len(testData), len(testLabel)))

        trainDataColumns = self._generate_column_headers(
            'X', 64 * 64 * 4)
        trainLabelColumns = self._generate_column_headers(
            'Y', 1)
        testDataColumns = self._generate_column_headers('X', 64 * 64 * 4)
        testLabelColumns = self._generate_column_headers('Y', 1)

        trainData, trainLabel, testData, testLabel = pd.DataFrame(
            trainData, columns=trainDataColumns).fillna(0), pd.DataFrame(
            trainLabel, columns=trainLabelColumns).fillna(0), pd.DataFrame(
            testData, columns=testDataColumns).fillna(0), pd.DataFrame(
            testLabel, columns=testLabelColumns).fillna(0)

        print(trainData.shape, testData.shape)
        print(trainLabel.shape, testLabel.shape)

        return (trainData, trainLabel), (testData, testLabel)

    def _generate_column_headers(self, prefix, columnsCount):
        columnHeaders = []

        for index in range(0, columnsCount):
            columnHeaders.append(prefix + '_' + str(index))

        return columnHeaders

    def _get_feature_columns(self, feature):
        featureColumns = []

        for key in feature.keys():
            featureColumns.append(tf.feature_column.numeric_column(key=key))

        return featureColumns

    def _get_label(self, imageDir):
        labelName = imageDir.split('_')[0]
        label = self._labels[labelName]

        return self._labels[labelName]

    def _train_input_fn(self, features, labels, batchSize):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        dataset = dataset.shuffle(1000).repeat().batch(batchSize)

        return dataset

    def _eval_input_fn(self, features, labels, batchSize):
        if labels is None:
            inputs = features
        else:
            inputs = (dict(features), labels)

        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(self.BATCH_SIZE)

        return dataset


def main(argv):
    imageClassifier = ImageClassifier()

    with tf.device('/gpu:0'):
        imageClassifier.train_model()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
