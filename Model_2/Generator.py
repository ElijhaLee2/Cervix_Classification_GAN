from other_functions import *

class Generator():
    def __init__(self, input, construct_fn, name, additional_name, reuse=False):
        full_name = name+'_'+additional_name
        with tf.name_scope(full_name):
            self.generated_img = construct_fn(input, name, reuse)
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

            tf.summary.image('input', input, max_outputs=BATCH_SIZE)
            tf.summary.image(full_name + '_output', self.generated_img, max_outputs=BATCH_SIZE)
            tf.summary.histogram(full_name + '_output', self.generated_img)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,scope=full_name)   # 生成的图片也summary


class GeneratorOptimizer():
    def __init__(self, generator, discriminator, img_orig, img_reconst, learning_rate, name):
        with tf.name_scope(name):
            loss_reconst = tf.nn.l2_loss(img_orig-img_reconst)
            loss_gan = -tf.reduce_mean(discriminator.score)
            loss = loss_gan+loss_reconst

            self.global_step = tf.Variable(1, trainable=False, name='global_step')
            self.variables = generator.variables
            self.optimizer = self._create_optimizer(loss, learning_rate)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=name)  # 这里也包括了Generator里面的summary

    def _create_optimizer(self,loss, learning_rate):
        optimizer = ly.optimize_loss(loss=loss, global_step=self.global_step, learning_rate=learning_rate,
                                     optimizer=tf.train.RMSPropOptimizer, variables=self.variables,
                                     summaries=OPTIMIZER_SUMMARIES)
        return optimizer
