import tensorflow as tf
from HyperParam import *


def get_cervix_tensor(cer_real_ph):
    # 从0-255转化为0-1
    cer_real_tensor = cer_real_ph / MAX_PIXEL_VALUE

    # 对Cervix进行像素归一化
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    [ave_batch, var_batch] = tf.nn.moments(cer_real_tensor, axes=[0, 1, 2])
    with tf.control_dependencies([ema.apply([ave_batch, var_batch])]):
        channel_ave = ema.average(ave_batch)
        channel_std = tf.sqrt(ema.average(var_batch))
    cer_real_tensor = (cer_real_tensor - channel_ave) / channel_std
    return cer_real_tensor

def get_mnist_tensor(mnist_real_ph):
    mnist_real_tensor = mnist_real_ph / MAX_PIXEL_VALUE
    mnist_real_tensor = tf.expand_dims(mnist_real_tensor, axis=3)
    return mnist_real_tensor