

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # Input tensor --> [batch_size, 28, 28, 1]
    # Output tensor --> [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(input_layer, filters=32,
                             kernel_size=[5, 5], padding='same', activation=tf.nn.relu)

    # Input tensor --> [batch_size, 28, 28, 32]
    # Output tensor --> [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    # Input tensor --> [batch_size, 14, 14, 32]
    # Output tensor --> [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(pool1, filters=64,
                             kernel_size=[5, 5], padding='same', activation=tf.nn.relu)

    # Input tensor --> [batch_size, 14, 14, 64]
    # Output tensor --> [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    # Input tensor --> [batch_size, 7, 7, 64]
    # Output tensor --> [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Input tensor --> [batch_size, 7 * 7 * 64]
    # Output tensor --> [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Input tensor --> [batch_size, 1024]
    # Output tensor --> [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1, name='classes_tensor'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    print(labels.shape, logits.shape)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    print(train_data.shape)

    return

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='./model/mnist')

    tensors_to_log = {
        'predictions': 'softmax_tensor',
        'classes': 'classes_tensor'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(input_fn=train_input_fn,
                           steps=20000, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.device('/gpu:0'):
        tf.app.run()
