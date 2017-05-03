import tensorflow as tf
import tensorflow.contrib.layers as ly
from HyperParam import *


OPTIMIZER_SUMMARIES = [
    # "learning_rate",
    "loss",
    "gradient_norm",
]

def getOptimizer(score_fake_0, score_real_0,
                 score_fake_1, score_real_1,
                 score_fake_2, score_real_2,
                 cross_entropy,
                 label,
                 learning_rate):
    # Loss_D = -W_distance
    loss_d_0 = tf.reduce_mean(score_fake_0 - score_real_0)
    loss_d_1 = tf.reduce_mean(score_fake_1 - score_real_1)
    loss_d_2 = tf.reduce_mean(score_fake_2 - score_real_2)

    # Loss_G = W_distance
    loss_g_0 = - tf.reduce_mean(score_fake_0)
    loss_g_1 = - tf.reduce_mean(score_fake_1)
    loss_g_2 = - tf.reduce_mean(score_fake_2)

    # global steps
    global_step_d_0 = tf.Variable(1, trainable=False, name='global_step_d_0')
    global_step_d_1 = tf.Variable(2, trainable=False, name='global_step_d_1')
    global_step_d_2 = tf.Variable(3, trainable=False, name='global_step_d_2')
    global_step_g = tf.Variable(4, trainable=False, name='global_step_g')

    # Variables
    vars_d_0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D_0')
    vars_d_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D_1')
    vars_d_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='D_2')
    vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='G')

    # returns:
    # d_0
    with tf.name_scope('D_0'):
        optm_d_0 = ly.optimize_loss(loss=loss_d_0, global_step=global_step_d_0, learning_rate=learning_rate,
                                    optimizer=tf.train.RMSPropOptimizer, variables=vars_d_0,
                                    summaries=OPTIMIZER_SUMMARIES,
                                    name='optmzr_d_0')
        clip_vars_d_0 = [tf.assign(var, tf.clip_by_value(var, -CLIP, CLIP)) for var in vars_d_0]
        tuple_vars_d_0 = tf.tuple(clip_vars_d_0, control_inputs=[optm_d_0])

    # d_1
    with tf.name_scope('D_1'):
        optm_d_1 = ly.optimize_loss(loss=loss_d_1, global_step=global_step_d_1, learning_rate=learning_rate,
                                    optimizer=tf.train.RMSPropOptimizer, variables=vars_d_1,
                                    summaries=OPTIMIZER_SUMMARIES,
                                    name='optmzr_d_1')
        clip_vars_d_1 = [tf.assign(var, tf.clip_by_value(var, -CLIP, CLIP)) for var in vars_d_1]
        tuple_vars_d_1 = tf.tuple(clip_vars_d_1, control_inputs=[optm_d_1])

    # d_2
    with tf.name_scope('D_2'):
        optm_d_2 = ly.optimize_loss(loss=loss_d_2, global_step=global_step_d_2, learning_rate=learning_rate,
                                    optimizer=tf.train.RMSPropOptimizer, variables=vars_d_2,
                                    summaries=OPTIMIZER_SUMMARIES,
                                    name='optmzr_d_2')
        clip_vars_d_2 = [tf.assign(var, tf.clip_by_value(var, -CLIP, CLIP)) for var in vars_d_2]
        tuple_vars_d_2 = tf.tuple(clip_vars_d_2, control_inputs=[optm_d_2])

    with tf.name_scope('G'):
        # g. matmul's oprand must be rank 2!
        loss_g = tf.matmul(tf.convert_to_tensor([[loss_g_0, loss_g_1, loss_g_2]]),  # [1,3]
                           2 * (label - 0.5), transpose_b=True)  # [1.3]
        loss_g = tf.reshape(loss_g, shape=()) + cross_entropy
        optm_g = ly.optimize_loss(loss=loss_g, global_step=global_step_g, learning_rate=learning_rate,
                                  optimizer=tf.train.RMSPropOptimizer, variables=vars_g, summaries=OPTIMIZER_SUMMARIES,
                                  name='optmzr_g')

    return tuple_vars_d_0,tuple_vars_d_1,tuple_vars_d_2,optm_g