import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly

from Image_operation import *
from other_code.InceptionResnetV1_model import *

cer_batch_mngr = _ImageBatchManager(CERVIX_TRAIN_IMG_DIR, 'Type_1')
ph = tf.placeholder(tf.float32, shape=[batch_size, CERVIX_IMG_SIZE, CERVIX_IMG_SIZE, 3])
ph2 = tf.placeholder(tf.float32, shape=[batch_size, 1792])
res = inception_resnet_v1(ph)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=ph2, logits=res[0])
gs = tf.Variable(0)
optm = ly.optimize_loss(loss=tf.reduce_mean(loss),global_step=gs,learning_rate=0.02,optimizer=tf.train.RMSPropOptimizer)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(init)
    images = cer_batch_mngr.get_batch()
    labels = np.random.normal(size=[batch_size, 1792])
    res_res = sess.run(optm,feed_dict={ph:images,ph2:labels})
    print()