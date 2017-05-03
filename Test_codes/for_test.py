# a = []
#
# for aa in a:
#     print('qwe')


# def a(b):
#     if b==1:
#         return 1
#     else:
#         return 1,2
#
# c = a(1)
# print(c.shape)


# import tensorflow as tf
#
# with tf.name_scope('0'):
#     with tf.name_scope('1'):
#         v1 = tf.Variable(1)
#
# with tf.name_scope('2'):
#     v2 = tf.Variable(2)
#
# with tf.name_scope('3'):
#     v3 = tf.Variable(3)
#
# v11 = tf.get_collection(tf.GraphKeys.VARIABLES,scope='1')
# print(v11)
# v22 = tf.get_collection(tf.GraphKeys.VARIABLES,scope='2')
# print(v22)
# v33 = tf.get_collection(tf.GraphKeys.VARIABLES,scope='3')
# print(v33)


# import numpy as np
# a = (np.random.uniform(0,3))
# b = int(a)
# print(a,b)


import tensorflow as tf
a = ph+1
ph = tf.placeholder(tf.float32,shape=())


with tf.Session() as sess:
    print(sess.run([a], feed_dict={ph:3.}))