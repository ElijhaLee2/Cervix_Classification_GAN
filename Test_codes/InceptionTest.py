from InceptionResnetV1_model import *
from Image_Operate import *
import tensorflow.contrib.layers as ly
import numpy as np
import tensorflow as tf

cer_batch_mngr = ImageBatchManager(CERVIX_TRAIN_IMG_DIR,'Type_1')
ph = tf.placeholder(tf.float32,shape=[BATCH_SIZE,CERVIX_IMG_SIZE,CERVIX_IMG_SIZE,3])
ph2 = tf.placeholder(tf.float32,shape=[BATCH_SIZE,1792])
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
    labels = np.random.normal(size=[BATCH_SIZE,1792])
    res_res = sess.run(optm,feed_dict={ph:images,ph2:labels})
    print()