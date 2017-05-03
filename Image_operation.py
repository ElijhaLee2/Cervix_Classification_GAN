from HyperParam import *
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
        type_nums = [0,0,0]
        for i in range(3):
            type_nums[i] = len(os.listdir(os_path.join(img_dir,img_types[i])))

        self.type_0_bm = _BatchManager(img_dir=img_dir, type=img_types[0], batch_size=batch_size,
                                       buff_size_GB=buffer_size * type_nums[0] / sum(type_nums))
        self.type_1_bm = _BatchManager(img_dir=img_dir, type=img_types[1], batch_size=batch_size,
                                       buff_size_GB=buffer_size * type_nums[1] / sum(type_nums))
        self.type_2_bm = _BatchManager(img_dir=img_dir, type=img_types[2], batch_size=batch_size,
                                       buff_size_GB=buffer_size * type_nums[2] / sum(type_nums))

    def get_image_batch(self,type):
        if type==0:
            return self.type_0_bm.get_batch()
        elif type==1:
            return self.type_1_bm.get_batch()
        elif type==2:
            return self.type_2_bm.get_batch()
        else:
            print('No such type: Type_%d' %(type))
            return None


class _BatchManager:
    def __init__(self, img_dir, type, batch_size, buff_size_GB):
        self.cervixImageReader = _ImageReader(buff_size_GB * 1024 * 1024 * 1024, img_dir, type)

        self.imgs = self.cervixImageReader.read_images()
        self.img_num = self.cervixImageReader.max_img_num
        self.img_all_num = self.cervixImageReader.img_all_num

        self.current_img_index = 0
        self.img_read_count = 0
        self.epoch = 0

        self.batch_size = batch_size

    def get_batch(self):
        img_count = 0
        batch = []
        while img_count < self.batch_size:
            batch.append(self.imgs[self.current_img_index])

            img_count += 1
            self.current_img_index += 1
            self.img_read_count += 1

            if self.current_img_index == self.img_num:
                self.current_img_index = 0
                self.imgs = self.cervixImageReader.read_images()

            if self.img_read_count == self.img_all_num:
                self.epoch += 1

            self.img_read_count = self.img_read_count % self.img_all_num


        return batch


class _ImageReader:
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
        self.img_list = os.listdir(self.type_dir)
        self.img_all_num = len(self.img_list)
        self.current_img_index = 0

        # 设置 self.max_img_num
        image0 = Image.open(os_path.join(self.type_dir, self.img_list[0]))
        size_per_img = sys.getsizeof(np.array(image0))
        self.max_img_num = int(self.max_mem_size/size_per_img)
        if self.max_img_num > self.img_all_num:
            self.all_imgs_read = True
            self.max_img_num = self.img_all_num

    def read_images(self):
        if len(self.imgs)>0 and self.all_imgs_read==True:
            return self.imgs

        img_count = 0

        print('(' + str(time.time()) + ') start reading image from:'+self.type_dir, end=' ...' )
        while img_count < self.max_img_num:
            img = Image.open(os_path.join(self.type_dir, self.img_list[self.current_img_index]))
            self.imgs.append(np.array(img))
            self.current_img_index = (self.current_img_index + 1) % self.img_all_num
            img_count += 1
        print('finished!')

        return self.imgs