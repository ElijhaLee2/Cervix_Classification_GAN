import tensorflow as tf
import numpy as np
from HyperParam import *
import os as os


ph1 = tf.placeholder(tf.float32,shape=[2,2])
ph2 = tf.placeholder(tf.float32,shape=[2,2])
image1 = tf.expand_dims(ph1,axis=0)
image1 = tf.expand_dims(image1,axis=3)
image2 = tf.expand_dims(ph2,axis=0)
image2 = tf.expand_dims(image2,axis=3)

tf.summary.image('0-0.5',image1)
tf.summary.image('0-127',image2)

merge = tf.summary.merge_all()

list = os.listdir(EVENT_DIR)
for l in list:
    os.remove(os.path.join(EVENT_DIR,l))

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter(logdir=EVENT_DIR,graph=sess.graph)
    image1_res, image2_res, merge_res = sess.run([image1,image2,merge],feed_dict={ph1:np.array([[0,0],[0,0]]),ph2:np.array([[0,0.2],[0,0.2]])})
    file_writer.add_summary(merge_res)
    file_writer.flush()


"""
结论：tensorboard把[min,max]线性映射到[0,1]，然后将图片中的最大值作为白，最小值作为黑
"""
