from other_functions import *
from HyperParam import *


# def generator(input_tensor, name):
#     # 224, 1
#     net = ly.conv2d(input_tensor, num_outputs=64, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 112, 64
#     net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 56, 128
#     net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 56, 128
#     net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 256
#     net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 256
#     net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 512
#     net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 512
#     net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 256
#     net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 256
#     net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 128
#     net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 128
#     net = ly.conv2d(net, num_outputs=64, kernel_size=3, stride=1, normalizer_fn=ly.batch_norm,
#                     activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 64
#     net = ly.conv2d(net, num_outputs=1, kernel_size=3, stride=1, normalizer_fn=None,
#                     activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
#     # 28, 1
#
#     return net


# AutoEncoder structured
def generator(input_tensor):
    # 224, 1
    net = ly.conv2d(input_tensor, num_outputs=64, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 112, 64
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 56, 128
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 28, 256
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 14, 256
    net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 7, 512
    net = ly.conv2d(net, num_outputs=256, kernel_size=2, stride=1, normalizer_fn=ly.batch_norm,
                    activation_fn=leaky_relu, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 14, 256
    net = ly.conv2d_transpose(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu,
                              normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 14, 128
    net = ly.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2, activation_fn=leaky_relu,
                              normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 28, 64
    net = ly.conv2d_transpose(net, num_outputs=1, kernel_size=3, stride=1, activation_fn=tf.nn.tanh,
                              normalizer_fn=None, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 28, 1

    return net


def discriminator(image, isWGAN):
    # 28, 1
    net = ly.conv2d(image, num_outputs=64, kernel_size=3, stride=2, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 14, 64
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=1, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 14, 128
    net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 7, 128
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 7, 256
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 4, 256
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 4, 256
    net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, activation_fn=leaky_relu,
                    normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 4, 256

    net = tf.reshape(net, shape=[BATCH_SIZE, 4 * 4 * 256])

    # 4*4*256 = 4096
    net = ly.fully_connected(net, 1024, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                             weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 1024
    score = ly.fully_connected(net, 1,
                               activation_fn=None if isWGAN else tf.nn.sigmoid,
                               normalizer_fn=None,
                               weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
    # 1

    return score
