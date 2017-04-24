import tensorflow as tf
from HyperParam import *
import tensorflow.contrib.layers as ly

STDDEV = 0.01

def leaky_relu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
]


def get_optimizers(fake, real, learning_rate, isWGAN):
    if isWGAN:
        return _get_optimizers_WGAN(fake, real, learning_rate)
    else:
        return _get_optimizers_GAN(fake, real, learning_rate)


def _get_optimizers_WGAN(fw_g_mnist, fw_mnist, learning_rate):
    loss_c = tf.reduce_mean(fw_g_mnist - fw_mnist)
    loss_g = tf.reduce_mean(-fw_g_mnist)
    # global_step
    global_step_c = tf.Variable(initial_value=0, trainable=False, name='global_step_c')
    global_step_g = tf.Variable(initial_value=0, trainable=False, name='global_step_g')
    # vars
    vars_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    # optms
    optm_c = ly.optimize_loss(loss_c, global_step_c, learning_rate, tf.train.RMSPropOptimizer,
                              variables=vars_c,
                              name='optimizer_c', summaries=OPTIMIZER_SUMMARIES)
    optm_g = ly.optimize_loss(loss_g, global_step_g, learning_rate, tf.train.RMSPropOptimizer,
                              variables=vars_g,
                              name='optimizer_g', summaries=OPTIMIZER_SUMMARIES)

    # clip
    clip_c_vars = [tf.assign(var, tf.clip_by_value(var, -CLIP, CLIP)) for var in vars_c]
    tuple_c_vars = tf.tuple(clip_c_vars, control_inputs=[optm_c])

    return tuple_c_vars, optm_g

def _get_optimizers_GAN(scores_g_z, scores_x, learning_rate):
    loss_dis = -(tf.reduce_mean(tf.log(1-scores_g_z) + tf.log(scores_x)))
    loss_gen = tf.reduce_mean(tf.log(1-scores_g_z))

    # global_step
    global_step_d = tf.Variable(initial_value=0, trainable=False)
    global_step_g = tf.Variable(initial_value=0, trainable=False)
    # vars
    vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    # optms
    optm_d = ly.optimize_loss(loss_dis,global_step_d,learning_rate,tf.train.RMSPropOptimizer,
                              variables=vars_d,
                              name='optm_d',summaries=OPTIMIZER_SUMMARIES)
    optm_g = ly.optimize_loss(loss_gen,global_step_g,learning_rate,tf.train.RMSPropOptimizer,
                              variables=vars_g,
                              name='optm_g',summaries=OPTIMIZER_SUMMARIES)
    return optm_d, optm_g