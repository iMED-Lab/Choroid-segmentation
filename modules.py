# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def set_parameter_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class adapt_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(adapt_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        return x


class activation_block(nn.Module):
    def __init__(self):
        super(activation_block, self).__init__()
    
    def forward(self, x):
        return torch.exp(-(x-0.5)**2) + 1.0 - math.exp(-0.25)  # torch.square##


class FEB(nn.Module):
    def __init__(self,ch_in, r_lst=[1, 2, 4, 6, 8]):
        super(FEB, self).__init__()
        self.branch = nn.Conv2d(ch_in, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.branch1 = nn.Conv2d(ch_in, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.branch2 = nn.Conv2d(ch_in, 1, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.branch4 = nn.Conv2d(ch_in, 1, kernel_size=3, stride=1, padding=4, dilation=4, bias=True)
        self.branch6 = nn.Conv2d(ch_in, 1, kernel_size=3, stride=1, padding=6, dilation=6, bias=True)
        self.branch8 = nn.Conv2d(ch_in, 1, kernel_size=3, stride=1, padding=8, dilation=8, bias=True)
        self.merge_layer = nn.Conv2d(len(r_lst)+1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_lst = [self.branch(x),self.branch1(x),self.branch2(x),self.branch4(x),self.branch6(x),self.branch8(x)]
        x = torch.cat(x_lst, dim=1)
        x = self.merge_layer(x)
        x = self.sigmoid(x)
        
        return x


class BEM_block(nn.Module):
    def __init__(self, ch_in, r_lst=[1, 2, 4, 6, 8], ks=3):
        super(BEM_block, self).__init__()
        self.feb = FEB(ch_in, r_lst)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # ##
        self.sigmoid = nn.Sigmoid()  # ##
        self.conv = nn.Conv1d(1, 1, kernel_size=ks, padding=(ks-1)//2, bias=True)  # ##
        
        self.extract_layer = adapt_layer(ch_in, 1)
        self.activation = activation_block()
    
    def forward(self, x):
        z = self.avg_pool(x)  # ##
        z = self.conv(z.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # ##
        z = self.sigmoid(z)  # ##
        
        w = self.extract_layer(x)  # ##
        w = self.activation(w)  # ##
        
        y = self.feb(x)
        x = x * w.expand_as(x) * y.expand_as(x) * z.expand_as(x) + x  # Original BPB: x = x * y.expand_as(x) + x
        
        return y, x


class VGG(nn.Module):
    def __init__(self, model="vgg19"):
        super(VGG, self).__init__()
        assert model == "vgg16" or model == "vgg19"
        if model == "vgg16":
            net = models.vgg16(pretrained=True)
            set_parameter_requires_grad(net.features, requires_grad=False)
            self.layer1 = net.features[:4]
            self.layer2 = net.features[5:9]
            self.layer3 = net.features[10:16]
            self.layer4 = net.features[17:23]
            self.layer5 = net.features[24:30]
        else:
            net = models.vgg19(pretrained=True)
            net.eval()
            net.features[0]=nn.Conv2d(3, 64, kernel_size=3, padding=1)
            set_parameter_requires_grad(net.features, requires_grad=False)
            self.layer1 = net.features[:4]
            self.layer2 = net.features[5:9]
            self.layer3 = net.features[10:18]
            self.layer4 = net.features[19:27]
            self.layer5 = net.features[28:36]
    
    def forward(self, x):
        relu1 = self.layer1(x)
        x = F.max_pool2d(relu1, kernel_size=2, stride=2)
        
        relu2 = self.layer2(x)
        x = F.max_pool2d(relu2, kernel_size=2, stride=2)
        
        relu3 = self.layer3(x)
        x = F.max_pool2d(relu3, kernel_size=2, stride=2)
        
        relu4 = self.layer4(x)
        x = F.max_pool2d(relu4, kernel_size=2, stride=2)
        
        relu5 = self.layer5(x)
        x = F.max_pool2d(relu5, kernel_size=2, stride=2)
        
        return relu1, relu2, relu3, relu4, relu5, x


class perception_loss(nn.Module):
    def __init__(self, model="vgg19", loss=nn.L1Loss(), weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(perception_loss, self).__init__()
        assert model == "vgg16" or model == "vgg19"
        self.model = VGG(model).cuda()
        self.loss = loss
        self.weights = weights
    
    def forward(self, x, y):
        rx1, rx2, rx3, rx4, rx5, _ = self.model(x)
        ry1, ry2, ry3, ry4, ry5, _ = self.model(y)
        loss = self.weights[0] * self.loss(rx1, ry1) + self.weights[1] * self.loss(rx2, ry2) + self.weights[2] * self.loss(rx3, ry3) + self.weights[3] * self.loss(rx4, ry4) + self.weights[4] * self.loss(rx5, ry5)
        
        return loss
