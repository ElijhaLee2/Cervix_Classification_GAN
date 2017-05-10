from other_functions import *
from HyperParam import *


class Discriminator():
    def __init__(self, input, construct_fn, name, addional_name, reuse=False):
        full_name = name + '_' + addional_name
        with tf.name_scope(full_name):
            self.score = construct_fn(input, name=name, reuse=reuse, real_or_fake='score')  # (batch_size,1)
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

            tf.summary.scalar('score', tf.reduce_mean(self.score))
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=full_name)


class DiscriminatorOpimizer():
    def __init__(self, discriminator_real, discriminator_fake, learning_rate, name):
        with tf.name_scope(name):
            self.global_step = tf.Variable(1, trainable=False, name='global_step')
            self.variables = discriminator_fake.variables
            loss = tf.reduce_mean(discriminator_fake.score - discriminator_real.score)
            tf.summary.scalar('w_distance', -loss)
            self.optimizer = self._create_optimizer(loss, learning_rate, name=name)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=name)

    def _create_optimizer(self, loss, learning_rate, name):
        optimizer = ly.optimize_loss(loss=loss, global_step=self.global_step, learning_rate=learning_rate,
                                     optimizer=tf.train.RMSPropOptimizer, variables=self.variables,
                                     summaries=OPTIMIZER_SUMMARIES)
        clip_vars = [tf.assign(var, tf.clip_by_value(var, -CLIP, CLIP)) for var in self.variables]
        tuple_vars = tf.tuple(clip_vars, control_inputs=[optimizer])

        return tuple_vars
