from other_functions import *


def construct_AB(input_tensor, name, reuse):
    with tf.variable_scope(name, reuse):
        # 224, 1
        net = ly.conv2d(input_tensor, num_outputs=64, kernel_size=4, stride=2, normalizer_fn=None,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 112, 64
        net = ly.conv2d(net, num_outputs=128, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 56, 128
        net = ly.conv2d(net, num_outputs=256, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 28, 256
        net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 256
        net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 7, 512
        net = ly.conv2d(net, num_outputs=256, kernel_size=2, stride=1, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 256
        net = ly.conv2d_transpose(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu,
                                  normalizer_fn=ly.batch_norm,
                                  weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 128
        net = ly.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2, activation_fn=leaky_relu,
                                  normalizer_fn=ly.batch_norm,
                                  weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 28, 64
        net = ly.conv2d_transpose(net, num_outputs=1, kernel_size=3, stride=1, activation_fn=tf.nn.sigmoid,
                                  normalizer_fn=None,
                                  weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 28, 1

    return net


def construct_BA(input_tensor, name, reuse):
    with tf.variable_scope(name, reuse):
        # 224, 1
        net = ly.conv2d(input_tensor, num_outputs=64, kernel_size=4, stride=2, normalizer_fn=None,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 112, 64
        net = ly.conv2d(net, num_outputs=128, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 56, 128
        net = ly.conv2d(net, num_outputs=256, kernel_size=4, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 28, 256
        net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 256
        net = ly.conv2d(net, num_outputs=512, kernel_size=3, stride=2, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 7, 512
        net = ly.conv2d(net, num_outputs=256, kernel_size=2, stride=1, normalizer_fn=ly.batch_norm,
                        activation_fn=leaky_relu,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 256
        net = ly.conv2d_transpose(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu,
                                  normalizer_fn=ly.batch_norm,
                                  weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 128
        net = ly.conv2d_transpose(net, num_outputs=64, kernel_size=3, stride=2, activation_fn=leaky_relu,
                                  normalizer_fn=ly.batch_norm,
                                  weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 28, 64
        net = ly.conv2d_transpose(net, num_outputs=1, kernel_size=3, stride=1, activation_fn=tf.nn.sigmoid,
                                  normalizer_fn=None,
                                  weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 28, 1

    return net


def construct_D_A(input_tensor, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        # 28, 1
        net = ly.conv2d(input_tensor, num_outputs=32, kernel_size=3, stride=2, activation_fn=leaky_relu,
                        normalizer_fn=None,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 32
        net = ly.conv2d(net, num_outputs=64, kernel_size=3, stride=1, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 64
        net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 7, 128
        net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 7, 256
        net = ly.conv2d(net, num_outputs=512, kernel_size=2, stride=2, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 4, 512
        net = ly.conv2d(net, num_outputs=512, kernel_size=2, stride=1, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 4, 512
        net = tf.reshape(net, shape=[BATCH_SIZE, 4 * 4 * 512])
        # 4*4*512 = 8192
        net = ly.fully_connected(net, 2048, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                                 weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 1024
        score = ly.fully_connected(net, 1,
                                   activation_fn=None if IS_WGAN else tf.nn.sigmoid,
                                   normalizer_fn=None,
                                   weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 1

    return score  # (batch_size,1)


def construct_D_B(input_tensor, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        # 28, 1
        net = ly.conv2d(input_tensor, num_outputs=32, kernel_size=3, stride=2, activation_fn=leaky_relu,
                        normalizer_fn=None,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 32
        net = ly.conv2d(net, num_outputs=64, kernel_size=3, stride=1, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 14, 64
        net = ly.conv2d(net, num_outputs=128, kernel_size=3, stride=2, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 7, 128
        net = ly.conv2d(net, num_outputs=256, kernel_size=3, stride=1, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 7, 256
        net = ly.conv2d(net, num_outputs=512, kernel_size=2, stride=2, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 4, 512
        net = ly.conv2d(net, num_outputs=512, kernel_size=2, stride=1, activation_fn=leaky_relu,
                        normalizer_fn=ly.batch_norm,
                        weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 4, 512
        net = tf.reshape(net, shape=[BATCH_SIZE, 4 * 4 * 512])
        # 4*4*512 = 8192
        net = ly.fully_connected(net, 2048, activation_fn=leaky_relu, normalizer_fn=ly.batch_norm,
                                 weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 1024
        score = ly.fully_connected(net, 1,
                                   activation_fn=None if IS_WGAN else tf.nn.sigmoid,
                                   normalizer_fn=None,
                                   weights_initializer=tf.random_normal_initializer(stddev=STDDEV))
        # 1

    return score  # (batch_size,1)