import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import PIL.Image as Image
import numpy as np
# from os import *
import os.path as os_path
import os as os

IMAGE_OUT_PATH = 'MNIST'

for i in range(10):
    # 将int转为string要用str()
    num_path = IMAGE_OUT_PATH + '/' + str(i)
    if not os_path.isdir(num_path):
        os.mkdir(num_path)


mnist = input_data.read_data_sets('MNIST_data')
# images(看作是矩阵)是一个ndarray对象, 同时也是一个list(还有与这个list并列的其他ndarray的属性, 比如dtype, max, min, shape, size), list的每个元素是一行;
# ndarray的每行(看作是向量)是一个ndarray对象, 同时也是一个list(还有其他ndarray属性, 同前), list的每个元素是一个数(float32对象, 也有多种属性)
images = mnist.train.images
labels = mnist.train.labels

for i in range(images.shape[0]):
    img = images[i]
    lbl = labels[i]
    img = np.reshape(img, [28, 28]) * 255
    # 至今没明白L和P区别是啥
    image = Image.fromarray(img).convert(mode='L')
    # 需要确保文件夹存在, 而它不会自动创建
    image.save(IMAGE_OUT_PATH + '/' + str(lbl) + '/' + str(i) + '.bmp')

# 这个地方是最骚的 np.array()竟然能接受Image类的对象作为参数...
# asdasd = np.array(image1)
# dsadsa = np.array(image2)

# import module
# print(module.__version__)

# python --version