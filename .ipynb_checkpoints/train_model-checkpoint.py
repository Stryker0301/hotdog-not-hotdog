from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from giextractor import GoogleImageExtractor
from sklearn.preprocessing import MinMaxScaler
import os
from scipy import misc
import imageio

# Downlad images for the training and the test set
# imageExtractor = GoogleImageExtractor(
#     imageQuery='apple fruit', imageCount=1000)
# imageExtractor.extract_images()

trainingData = []
testingData = []
trainingLabels = [10]
testingLabels = [10]

imageList = os.listdir('./apple_fruit')

# Separate the training and the test images
trainingImages = imageList[0:800]
testingImages = imageList[801:len(imageList)]

# Read the image and resize every image to the size - 256 X 256
# The image array (type=ndarray) is a 3D array, convert it to a 2D array using np.reshape
for trainingImage in trainingImages:
    image = imageio.imread('./apple_fruit/' + trainingImage)
    image = misc.imresize(image, size=(256, 256))
    image = np.resize(image, new_shape=(256, 256))
    trainingData.append(image)

# Read the image and resize every image to the size - 256 X 256
# The image array (type=ndarray) is a 3D array, convert it to a 2D array using np.reshape
for testingImage in testingImages:
    image = imageio.imread('./apple_fruit/' + testingImage)
    image = misc.imresize(image, size=(256, 256))
    image = np.resize(image, new_shape=(256, 256))
    testingData.append(image)

# Converting the list of training and test images to an array creates a 3D array.
# Reshape the 3D array to a 2D array, of shape - (training_data_set_size, 256 X 256) to be able to pass as an input to the min-max scaler
# The image data is therefore flattened and the array is reshaped, without losing any data
X_training = np.array(trainingData).reshape((len(trainingData), 256 * 256))
Y_training = np.array(trainingLabels, dtype=float)
Y_training = np.resize(Y_training, new_shape=(len(trainingData), 1))

X_testing = np.array(testingData).reshape(len(testingData), 256 * 256)
Y_testing = np.array(testingLabels, dtype=float)
Y_testing = np.resize(Y_testing, new_shape=(len(testingData), 1))

# Scale the training and the testing image data set between 0 and 1
X_scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled_training = X_scaler.fit_transform(X_training)
X_scaled_testing = X_scaler.transform(X_testing)

# Define the model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# Define the number of inputs and outputs
number_of_inputs = 256 * 256
number_of_outputs = 1

# Define the count of neurons in every layer
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float64, shape=[None, number_of_inputs])

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name='weights1', dtype=tf.float64, shape=[
                              number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases1', dtype=tf.float64, shape=[
                             layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name='weights2', dtype=tf.float64, shape=[
                              layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases2', dtype=tf.float64, shape=[
                             layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name='weights3', dtype=tf.float64, shape=[
                              layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases3', dtype=tf.float64, shape=[
                             layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Ouput layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name='weights4', dtype=tf.float64, shape=[
                              layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases4', dtype=tf.float64, shape=[
                             layer_3_nodes], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# Cost Function
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float64, shape=[None, number_of_outputs])
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Train the model by optimizing the cost function
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.variable_scope('logging'):
    tf.summary.scalar('cost_summary', cost)
    summary = tf.summary.merge_all()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    training_writer = tf.summary.FileWriter('./logs/training', session.graph)
    testing_writer = tf.summary.FileWriter('./logs/testing', session.graph)

    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_training})

        if epoch % 5 == 0:
            training_cost, training_summary = session.run(
                [cost, summary], feed_dict={X: X_scaled_training, Y: Y_training})
            testing_cost, testing_summary = session.run(
                [cost, summary], feed_dict={X: X_scaled_testing, Y: Y_testing})

            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            print('Epoch: {}, Training Cost: {}, Testing Cost: {}'.format(
                epoch, training_cost, testing_cost))

    print('\nTraining is complete !')

    final_training_cost = session.run(
        cost, feed_dict={X: X_scaled_training, Y: Y_training})
    final_testing_cost = session.run(
        cost, feed_dict={X: X_scaled_testing, Y: Y_testing})

    print('Final Training Cost: {}'.format(final_training_cost))
    print('Final Testing Cost: {}'.format(final_testing_cost))

    final_prediction = session.run(prediction, feed_dict={X: X_scaled_testing})

    print('Final Prediction: {}'.format(final_prediction))
