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
        images = []
        masks = []
        heatmaps = []

        if isTraining:
            outer_path = os.path.join(data_dir, "train")
            folderlist = os.listdir(outer_path)
        else:
            outer_path = os.path.join(data_dir, "test")
            folderlist = os.listdir(outer_path)

        for folder in folderlist:
            middle_path = os.path.join(outer_path, folder)
            image_path = os.path.join(middle_path, "image_crop")
            label_path = os.path.join(middle_path, "choroid_label")
            heatmap_path = os.path.join(middle_path,"heatmaps2")
            images1 = os.listdir(image_path)

            for image in images1:
                imagePath = os.path.join(image_path, image)
                images.append(imagePath)

                maskPath = os.path.join(label_path, image)
                masks.append(maskPath)

                heatmapPath = os.path.join(heatmap_path, image)
                heatmaps.append(heatmapPath)


        return images, masks, heatmaps
    
    def getFileName(self):
        return self.name
