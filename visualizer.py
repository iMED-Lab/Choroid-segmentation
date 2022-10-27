import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable as V
from torch import optim
from  PIL import Image
from skimage import filters,morphology
from torchvision import transforms 
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
import visdom
from visdom import Visdom 
import numpy as np 
import os
import logging
import scipy.misc as misc
import cv2

# visdom可视化
class Visualizer:
    def __init__(self,env="default",**kwargs):
        self.vis = Visdom(env=env,**kwargs)
        self.index = {}
        self.log_text = ""

    def reinit(self,env="default",**kwargs):
        self.vis = Visdom(env=env,**kwargs)

        return self

    def plot(self,name,y,**kwargs):
        x = self.index.get(name,0)
        # print(x,y)
        self.vis.line(Y=np.array([y]),X=np.array([x]),win=str(name),update=None if x == 0 else "append",**kwargs)
        self.index[name] = x + 1
    
    def plot_many(self,d):
        for k,v in d.iteritems():
            self.plot(k,v)

    def plot_line(self,name,x,y,**kwargs):
        self.vis.line(X=np.array([x]),Y=np.array([y]),win=str(name),update=None if x == 0 else "append",**kwargs)

    def img_many(self,d):
        for k,v in d.iteritems():
            self.img(k,v)

    def img(self,name,img_,**kwargs):
        self.vis.images((img_.detach().cpu().numpy()).astype(np.uint8),
                        win=str(name),
                        opts=dict(title=name),
                        **kwargs)

