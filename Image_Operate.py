from HyperParam import *
import PIL.Image as Image
import os as os
import os.path as os_path
import numpy as np
import sys as sys


class CervixImageReader():
    def __init__(self, max_mem_size):
        self.max_mem_size = max_mem_size
        self.max_img_num = -1

        self.train_img_dir = TRAIN_IMG_DIR
        self.type_list = os.listdir(self.train_img_dir)

        self.type = None
        self.type_dir = None
        self.img_list = None
        self.img_num = -1
        self.current_img_index = -1
        self.epoch = -1

    def set_type(self, type):
        if self.type_list.count(type) == 0:
            print('No such type!')
            return

        self.type = type
        self.type_dir = os_path.join(self.train_img_dir, self.type)
        self.img_list = os.listdir(self.type_dir)
        self.img_num = len(self.img_list)
        self.current_img_index = 0
        self.epoch = 0

        # 设置 self.max_img_num
        image0 = Image.open(os_path.join(self.type_dir, self.img_list[0]))
        size_per_img = sys.getsizeof(np.array(image0))
        self.max_img_num = self.max_mem_size / size_per_img

    def read_images(self):
        if self.type == None:
            print('Set type first!')
            return None

        img_count = 0
        imgs = []
        while img_count < self.max_img_num:
            img = Image.open(os_path.join(self.type_dir, self.img_list[self.current_img_index]))
            imgs.append(np.array(img))
            self.current_img_index += 1
            if self.current_img_index == self.img_num:
                self.epoch += 1
                self.current_img_index = 0

        return imgs

class MnistImageReader():
    # TODO 全部没改
    def __init__(self, max_mem_size):
        self.max_mem_size = max_mem_size
        self.max_img_num = -1

        self.train_img_dir = TRAIN_IMG_DIR
        self.type_list = os.listdir(self.train_img_dir)

        self.type = None
        self.type_dir = None
        self.img_list = None
        self.img_num = -1
        self.current_img_index = -1
        self.epoch = -1

    def set_type(self, type):
        if self.type_list.count(type) == 0:
            print('No such type!')
            return

        self.type = type
        self.type_dir = os_path.join(self.train_img_dir, self.type)
        self.img_list = os.listdir(self.type_dir)
        self.img_num = len(self.img_list)
        self.current_img_index = 0
        self.epoch = 0

        # 设置 self.max_img_num
        image0 = Image.open(os_path.join(self.type_dir, self.img_list[0]))
        size_per_img = sys.getsizeof(np.array(image0))
        self.max_img_num = self.max_mem_size / size_per_img

    def read_images(self):
        if self.type == None:
            print('Set type first!')
            return None

        img_count = 0
        imgs = []
        while img_count < self.max_img_num:
            img = Image.open(os_path.join(self.type_dir, self.img_list[self.current_img_index]))
            imgs.append(np.array(img))
            self.current_img_index += 1
            if self.current_img_index == self.img_num:
                self.epoch += 1
                self.current_img_index = 0

        return imgs