import tensorflow as tf
from HyperParam import *
import tensorflow.contrib.layers as ly

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
]


def get_optimizers(fw_g_mnist, fw_mnist, learning_rate):
    loss_c = tf.reduce_mean(fw_g_mnist - fw_mnist)
    loss_g = tf.reduce_mean(-fw_g_mnist)
    # global_step
    global_step_c = tf.Variable(initial_value=0, trainable=False)
    global_step_g = tf.Variable(initial_value=0, trainable=False)
    # vars
    vars_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    # optms
    optm_c = ly.optimize_loss(loss_c, global_step_c, learning_rate, tf.train.RMSPropOptimizer,
                              variables=vars_c,
                              name='optm_c', summaries=OPTIMIZER_SUMMARIES)
    optm_g = ly.optimize_loss(loss_g, global_step_g, learning_rate, tf.train.RMSPropOptimizer,
                              variables=vars_g,
                              name='optm_g', summaries=OPTIMIZER_SUMMARIES)

    # clip
    clip_c_vars = [tf.assign(var, tf.clip_by_value(var, -CLIP, CLIP)) for var in vars_c]
    tuple_c_vars = tf.tuple(clip_c_vars, control_inputs=[optm_c])

    return tuple_c_vars, optm_g
