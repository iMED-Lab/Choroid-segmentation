import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import BEM_block

device = torch.device("cuda")

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class mUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(mUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)

        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = self.relu(x)
        return x


class UNetBEM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetBEM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down22 = BEM_block(256)
        self.down3 = down(256, 512)
        self.down33 = BEM_block(512)
        self.down4 = down(512, 512)
        self.down44 = BEM_block(512)
        self.up1 = up(1024, 256)
        self.up11 = BEM_block(256)
        self.up2 = up(512, 128)
        self.up22 = BEM_block(128)
        self.up3 = up(256, 64)
        self.up33 = BEM_block(64)
        self.up4 = up(128, 32)
        self.outc = outconv(32, n_classes)
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        y1,x3 = self.down22(x3)
        x4 = self.down3(x3)
        y2,x4 = self.down33(x4)
        x5 = self.down4(x4)
        y3,x5 = self.down44(x5)
        x = self.up1(x5, x4)
        y4,x = self.up11(x)
        x = self.up2(x, x3)
        y5,x = self.up22(x)
        x = self.up3(x, x2)
        y6,x = self.up33(x)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.sigmoid(x)
        # x = self.relu(x)
        return x,y1,y2,y3,y4,y5,y6
