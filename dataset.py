# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from PIL import Image

import torch.utils.data as data
from torchvision import transforms


class OCT_loader(data.Dataset):
    def __init__(self, data_dir, channel=1, scale_size=(512, 512), isTraining=True):
        super(OCT_loader, self).__init__()
        self.img_lst, self.gt_lst, self.hm_lst = self.get_dataPath(data_dir, isTraining)
        self.channel = channel
        self.scale_size = scale_size
        self.isTraining = isTraining
        self.name = ""
    
    def __getitem__(self, index):
        img_path = self.img_lst[index]
        self.name = img_path.split("/")[-1]
        gt_path = self.gt_lst[index]
        hm_path = self.hm_lst[index]
        simple_transform = transforms.ToTensor()
        
        img = Image.open(img_path)
        gt = Image.open(gt_path).convert("L")
        hm = Image.open(hm_path).convert("L")
        
        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        
        img = img.resize(self.scale_size)
        gt = gt.resize(self.scale_size)
        hm = hm.resize(self.scale_size)
        
        gt = np.array(gt)
        gt[gt>=128] = 255
        gt[gt<=127] = 0
        gt = Image.fromarray(gt)
        
        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)
            hm = hm.rotate(angel)
        
        img = simple_transform(img)
        gt = simple_transform(gt)
        hm = simple_transform(hm)
        
        return img, gt, hm
    
    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)
    
    def get_dataPath(self, data_dir, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param data_dir: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(data_dir, "train", "images")
            gt_dir = os.path.join(data_dir, "train", "labels")
            hm_dir = os.path.join(data_dir, "train", "heatmaps3")
        else:
            img_dir = os.path.join(data_dir, "test", "images")
            gt_dir = os.path.join(data_dir, "test", "labels")
            hm_dir = os.path.join(data_dir, "test", "heatmaps3")
        
        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        hm_lst = sorted(list(map(lambda x: os.path.join(hm_dir, x), os.listdir(hm_dir))))
        
        assert len(img_lst) == len(gt_lst) and len(gt_lst) == len(hm_lst)
        
        return img_lst, gt_lst, hm_lst
    
    def getFileName(self):
        return self.name
