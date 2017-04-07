import tensorflow.contrib.layers as ly
from nn_functions import *
from HyperParam import *


def generator(input_tensor):
    # 448, 3
    net = ly.conv2d(input_tensor, num_outputs=64, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    net = ly.conv2d(net, num_outputs=64, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 224, 64
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 112, 128
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 56, 256
    net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 28, 512
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 28, 256
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 28, 128
    net = ly.conv2d(net, num_outputs=64, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 28, 64
    net = ly.conv2d(net, num_outputs=1, kernel_size=3, stride=1, normalizer_fn=None,
                    activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 28, 1

    return net


def critic(image):
    # 28, 1
    net = ly.conv2d(image, num_outputs=64, kernel_size=3, stride=2, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 14, 64
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 7, 128
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 4, 256
    net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=1, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 4, 512

    net = tf.reshape(net, shape=[BATCH_SIZE, 4 * 4 * 512])

    # 4*4*512 = 8196
    net = ly.fully_connected(net, 1024, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                             weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 1024
    score = ly.fully_connected(net, 1, activation_fn=None, normalizer_fn=ly.batch_norm,
                               weights_initializer=tf.random_normal_initializer(stddev=0.02))
    # 1

    return score
