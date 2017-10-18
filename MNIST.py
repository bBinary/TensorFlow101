# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:35:27 2017

@author: Bart
"""
# https://www.tensorflow.org/get_started/mnist/beginners
# TODO: visualize exmeplary input
# get some info bout data
# comment code

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# MNIST data is split into three parts: 
# 55,000 data points of training data -> mnist.train.images.shape
# 10,000 points of test data
# 5,000 points of validation data

# labels: one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension
# mnist.train.labels.shape

# placeholder - a symbolic variable

x = tf.placeholder(tf.float32, [None, 784])
# Here None means that a dimension can be of any length.
# A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. It can be used and even modified by the computation.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax gives us a list of values between 0 and 1 that add up to 1:
# A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.
# weighted sum of the pixel intensities: The weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor.
# softmax = normalize(exp())
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

#  In some rough sense, the cross-entropy is measuring how inefficient our predictions are for describing the truth.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Each step of the loop, we get a "batch" of one hundred random data points from our training set. We run train_step feeding in the batches data to replace the placeholders.
# small batches of random data is called stochastic training

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy is ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}), "%")
