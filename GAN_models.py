import tensorflow as tf
import tensorflow.contrib.layers as ly
from nn_functions import *
from HyperParam import *


def generator(input_tensor):
    # 3584
    net = ly.conv2d(input_tensor, num_outputs=64, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 1792
    net = ly.conv2d(net, num_outputs=128, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 896
    net = ly.conv2d(net, num_outputs=256, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 448
    net = ly.conv2d(net, num_outputs=512, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 224
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 112
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 56
    net = ly.conv2d(net, num_outputs=64, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm, activation_fn=leaky_relu,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 28
    net = ly.conv2d(net, num_outputs=1, kernel_size=3, stride=1, normalizer_fn=None, activation_fn=tf.nn.tanh,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 28

    return net


def critic(image, name=None):
    # 28
    net = ly.conv2d(image, num_outputs=64, kernel_size=3, stride=2, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 14
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 7
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 4
    net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=1, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 4
    net = tf.reshape(net, shape=[BATCH_SIZE, 4 * 4 * 512])
    # full connected, no batch_norm
    net = ly.fully_connected(net, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))

    return net