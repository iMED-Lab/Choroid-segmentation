# -*- coding: utf-8 -*-

import cv2
import math
import h5py
import numpy as np


# 灰度线性拉伸
def linear_stretch(img_arr):
    return 255.0 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))


def put_heatmap(heatmap, center, sigma=8):
    center_x, center_y = center
    height, width = heatmap.shape
    th = 4.6052
    delta = math.sqrt(th * 2)  # 3.0348640826238
    
    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))
    
    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))
    
    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[y, x] = max(heatmap[y, x], math.exp(-exp))
            heatmap[y, x] = min(heatmap[y, x], 1.0)
    
    return heatmap


def make_heatmap(mat_path, sigma=8):
    heatmap = np.zeros((320, 576))
    data = h5py.File(mat_path, "r")
    center_arr = data['point_PH2_6'].value
    print(center_arr)
    print(center_arr[1,2])
    for i in range(center_arr.shape[-1]):
        center_x = center_arr[0, i] - 1
        center_y = center_arr[1, i] - 1
        heatmap = put_heatmap(heatmap, (center_x, center_y), sigma)
    heatmap = linear_stretch(heatmap).astype(np.uint8)
    
    return heatmap


#heatmap = make_heatmap("point_PH2_6.mat")
#cv2.imwrite("heatmap.png", heatmap)
