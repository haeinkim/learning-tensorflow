import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
form tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_float("learning_rate", 0.01, "initial learning rate")
flags.DEFINE_integer("max_steps", 10000, "number of iterations to train")

def main(_):

    #import data
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step=tf.train.GradientDescentOptimizer(flags.learning_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for batch_index in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(flags.batch_size)
        if batch_index % 10 == 0:
        sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images,
                                        y:mnist.test.labels}))


