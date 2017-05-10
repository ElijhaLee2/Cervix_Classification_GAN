import random
import PIL.Image as Image
import os as os
import os.path as os_path
import numpy as np
import sys as sys
import time as time


class ImageProvider:
    def __init__(self, img_dir, img_types, batch_size, buffer_size):
        """
        :param img_dir: 
        :param img_types: List, eg: ['Type_1','Type_2','Type_3'],['1','4','7']
        :param batch_size: 
        buffer_size: Total buff size in GB
        """
        type_num = len(img_types)
        type_nums = []
        self.bm = []
        for i in range(type_num):
            type_nums.append(len(os.listdir(os_path.join(img_dir, img_types[i]))))
        for i in range(type_num):
            self.bm.append(_BatchManager(img_dir=img_dir, type=img_types[i], batch_size=batch_size,
                                         buff_size_GB=buffer_size * type_nums[i] / sum(type_nums)))

    def get_image_batch(self, type):
        return self.bm[type].get_batch()


class _BatchManager:
    def __init__(self, img_dir, type, batch_size, buff_size_GB):
        self.cervixImageReader = _ImageReader(buff_size_GB * 1024 * 1024 * 1024, img_dir, type)

        self.imgs = self.cervixImageReader.read_images()
        self.img_num = self.cervixImageReader.max_img_num       # 一次能读进内存的最大图片数
        self.img_all_num = self.cervixImageReader.img_all_num   # 该type的图片总数

        self.current_img_index = 0
        self.img_read_count = 0
        self.epoch = 0

        self.batch_size = batch_size

    def get_batch(self):
        """
        Shuffle的原因是，如果所有图片全部被读入MEM，那么每次epoch完成时需要重新shuffle
        :return: 
        """
        img_count = 0
        batch = []
        while img_count < self.batch_size:
            batch.append(self.imgs[self.current_img_index])

            img_count += 1  # 用于和batch size比较
            self.current_img_index += 1
            self.img_read_count += 1

            # all images in the mem has been read
            if self.current_img_index == self.img_num:
                self.current_img_index = 0
                self.imgs = self.cervixImageReader.read_images()

            # all images of this type has been read
            if self.img_read_count == self.img_all_num:
                self.epoch += 1
                self.img_read_count = 0
                self.shuffle_imgs()


        return batch

    def shuffle_imgs(self):
        random.shuffle(self.imgs)


class _ImageReader:
    """
    每次读入时，先对img_list进行shuffle
    """
    def __init__(self, max_mem_size, img_dir, type):
        self.max_mem_size = max_mem_size
        self.all_imgs_read = False
        self.max_img_num = -1

        self.train_img_dir = img_dir
        self.type_list = os.listdir(self.train_img_dir)

        self.type = None
        self.type_dir = None
        self.img_list = None
        self.img_all_num = -1

        self.current_img_index = -1
        self.imgs = []

        self.set_type(type)

    def set_type(self, type):
        if self.type_list.count(type) == 0:
            print('No such type!')
            return
        self.type = type
        self.type_dir = os_path.join(self.train_img_dir, self.type)
        self.img_list = self.get_shuffled_img_list()
        self.img_all_num = len(self.img_list)
        self.current_img_index = 0

        # 设置 self.max_img_num
        image0 = Image.open(os_path.join(self.type_dir, self.img_list[0]))
        size_per_img = sys.getsizeof(np.array(image0))
        self.max_img_num = int(self.max_mem_size / size_per_img)
        if self.max_img_num > self.img_all_num:
            self.all_imgs_read = True
            self.max_img_num = self.img_all_num

    def read_images(self):
        if len(self.imgs) > 0 and self.all_imgs_read == True:
            return self.imgs

        img_count = 0

        print('(' + str(time.time()) + ') start reading image from:' + self.type_dir, end=' ...')
        while img_count < self.max_img_num:
            img = Image.open(os_path.join(self.type_dir, self.img_list[self.current_img_index]))
            self.imgs.append(np.array(img))

            # all images of THIS TYPE has been read
            self.current_img_index += 1
            if self.current_img_index == self.img_all_num:
                self.current_img_index = 0
                self.img_list = self.get_shuffled_img_list()

            img_count += 1
        print(str(self.max_img_num)+'/'+str(self.img_all_num)+' has been read!')

        return self.imgs

    def get_shuffled_img_list(self):
         img_list = os.listdir(self.type_dir)
         random.shuffle(img_list)
         return img_list
