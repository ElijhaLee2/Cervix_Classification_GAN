from other_functions import *
from HyperParam import *

def _construct(input_tensor, name, reuse, real_or_fake):
    with tf.name_scope(real_or_fake):
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


class Discriminator():
    def __init__(self, image_real, image_fake, name):
        with tf.name_scope(name):
            self.score_real = _construct(image_real, name=name, reuse=False,real_or_fake='real')     #(batch_size,1)
            self.score_fake = _construct(image_fake, name=name, reuse=True,real_or_fake='fake')    #(batch_size,1)
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

            tf.summary.scalar('/score_real', tf.reduce_mean(self.score_real))
            tf.summary.scalar('/score_fake', tf.reduce_mean(self.score_fake))
            tf.summary.image('Fake_imgs_image', image_fake, max_outputs=BATCH_SIZE)
            tf.summary.histogram('Fake_imgs_histogram', image_fake)
            tf.summary.histogram('Real_imgs_histogram', image_real)
            self.summary = tf.get_collection(tf.GraphKeys.SUMMARIES,scope=name)


class DiscriminatorOpimizer():
    def __init__(self, discriminator, learning_rate, name):
        with tf.name_scope(name):
            self.global_step = tf.Variable(1, trainable=False, name='global_step')
            self.variables = discriminator.variables
            loss = tf.reduce_mean(discriminator.score_fake - discriminator.score_real)
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
