import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from modules import BEM_block,VGG,perception_loss
from UNetBEM import UNetBEM
from torch.utils.data import DataLoader
from dataset import OCT_loader
from SBE import SBE
import cv2
import os
from visualizer import Visualizer
from torchvision import transforms
from metrics import all_score
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np

args = {
    "root":"",
    "img_save_path":"./save_img",
    "epoches":400,
    "lr":1e-4,
    "snapshot":100,
    "test_step":1,
    "ckpt_path":"",
    "batch_size":8,
    "name":"",
}


def save_ckpt(net, discriminator, epoch):
    if not os.path.exists(args["ckpt_path"]):
        os.makedirs(args["ckpt_path"])
    state = {"model": net, "discriminator": discriminator}
    torch.save(state, args["ckpt_path"] + args["name"] + "_epoch_" + str(epoch) + ".pkl")
    print("---> save model:{} <---".format(args["ckpt_path"]))

def test(model,device):
    root_path = args["root"]
    dataset = OCT_loader(root_path , isTraining=False)
    test_loader = DataLoader(dataset,batch_size=1,shuffle = True)
    model.eval()
    iou, dice = [], []
    acc, sen, pre, f1 = [], [], [], []
    with torch.no_grad():

        for img_lst, gt_lst, hm_lst in test_loader:

            image = img_lst.float().to(device)
            label = gt_lst.float().to(device)
            heatmap = hm_lst.float().to(device)
            pred,y1,y2,y3,y4,y5,y6 = model(image)
            pred = pred * 255
            pred_trans = (pred.detach().cpu().numpy()).astype(np.uint8)
            heatmap = (heatmap.detach().cpu().numpy()).astype(np.uint8)
            pred_trans = pred_trans.squeeze()
            _, pred_trans = cv2.threshold(pred_trans, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            label_trans = (label.detach().cpu().numpy()).astype(np.uint8)
            label_trans = label_trans.squeeze()*255
            np.set_printoptions(threshold=np.inf)
            #print(pred_trans)

            indicator = all_score(pred_trans,label_trans)
            print("iou:", indicator[0], "dice:", indicator[1], "acc:", indicator[2], "sen:", indicator[3], "pre:",
                  indicator[4], "f1:", indicator[5])
            dice.append(indicator[1])
            iou.append(indicator[0])
            acc.append(indicator[2])
            sen.append(indicator[3])
            pre.append(indicator[4])
            f1.append(indicator[5])

            pred_trans = torch.Tensor(pred_trans)

            vis.img(name="images", img_=image[0, :, :, :] * 255)
            vis.img(name="labels", img_=label[0, :, :, :]*255)
            vis.img(name="perdiction", img_=pred_trans[:, :])

    return np.mean(iou), np.mean(dice), np.mean(acc), np.mean(sen), np.mean(pre), np.mean(f1)


def train_net(model,device, root_path = args["root"], epochs = args["epoches"], batch_size = args["batch_size"], lr = args["lr"]):


    dataset = OCT_loader(root_path, isTraining = True)
    discriminator = VGG(model="vgg19").to(device)
    #discriminator = discriminator(2).to(device)

    dataloader = DataLoader(dataset, batch_size = batch_size,shuffle = True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(),lr=lr)
    Map_loss = nn.CrossEntropyLoss().to(device)
    Seg_loss = nn.BCELoss().to(device)
    BA_loss = nn.BCELoss().to(device)

    iters = 1
    iou = 0.
    dice = 0.
    acc = 0.
    pre = 0.
    sen = 0.
    f1 = 0.

    for epoch in range(epochs):
        model.train()
        discriminator.train()

        for img_lst, gt_lst, hm_lst in dataloader:

            image = img_lst.float().to(device)
            label = gt_lst.float().to(device)
            heatmap = hm_lst.float().to(device)
          
            optimizer.zero_grad()
            pred, y1, y2, y3, y4, y5, y6 = model(image)
            loss_seg = Seg_loss(pred,label)
            #loss_map.backward()
            loss_seg.backward()
            optimizer.step()

            optimizer.zero_grad()
            pred, y1, y2, y3, y4, y5, y6 = model(image)
            heatmap_transformer = transforms.Resize((128,128))
            heatmap_transformer2 = transforms.Resize((64, 64))
            heatmap_transformer3 = transforms.Resize((32, 32))
            heatmap_transformer4 = transforms.Resize((256, 256))
            groundt1 = heatmap_transformer(heatmap)
            groundt2 = heatmap_transformer2(heatmap)
            groundt3 = heatmap_transformer3(heatmap)
            groundt4 = heatmap_transformer4(heatmap)
        
            
            groundt1 = groundt1.long().squeeze()
            groundt2 = groundt2.long().squeeze()
            groundt3 = groundt3.long().squeeze()
            groundt4 = groundt4.long().squeeze()
            
            
            loss_map = ((Map_loss(y1,groundt1) +  Map_loss(y2,groundt2)\
                        + Map_loss(y3,groundt3)+ Map_loss(y4,groundt2)\
                        + Map_loss(y5,groundt1) + Map_loss(y6,groundt4)))/6
            #print(y1.shape,y2.shape,y3.shape,y4.shape,y5.shape,y6.shape)
            loss_map.backward()
            optimizer.step()

            optimizer.zero_grad()
            pred, y1, y2, y3, y4, y5, y6 = model(image)
            x1 = torch.cat([label,heatmap,heatmap],dim=1)
            y = torch.cat([pred,heatmap,heatmap],dim=1)
            loss_per = perception_loss(model="vgg19")
            loss_per = loss_per(x1,y)
            loss_per.backward()
            optimizer.step()

            print("[{0:d}:{1:d}] --- loss_seg:{2:.10f},loss_per:{3:.10f},loss_map:{4:.10f}".format(epoch + 1,iters,loss_seg.item(),loss_per.item(),loss_map.item()))

            iters += 1
            vis.plot(name="seg_loss", y=loss_seg.item(), opts=dict(title="seg_loss", xlabel="batch", ylabel="loss"))
            vis.plot(name="map_loss", y=loss_map.item(), opts=dict(title="map_loss", xlabel="batch", ylabel="loss"))
            vis.plot(name="per_loss", y=loss_per.item(), opts=dict(title="per_loss", xlabel="batch", ylabel="loss"))
            #vis.plot(name="map_loss", y=loss_map.item(), opts=dict(title="map_loss", xlabel="batch", ylabel="loss"))

        if (epoch + 1) % 2 == 0:
            test_iou, test_dice, acc, sen, pre, f1 = test(model,device)
            print(test_iou, test_dice)
            vis.plot(name="iou", y=test_iou, opts=dict(title="iou", xlabel="epoch", ylabel="iou"))
            vis.plot(name="dice", y=test_dice, opts=dict(title="dice", xlabel="epoch", ylabel="dice"))
            vis.plot(name="acc", y=acc, opts=dict(title="acc", xlabel="epoch", ylabel="acc"))
            vis.plot(name="sen", y=sen, opts=dict(title="sen", xlabel="epoch", ylabel="sen"))
            vis.plot(name="pre", y=pre, opts=dict(title="pre", xlabel="epoch", ylabel="pre"))
            vis.plot(name="f1", y=f1, opts=dict(title="f1", xlabel="epoch", ylabel="f1"))
            if (test_iou > iou) & (test_dice > dice):
                save_ckpt(model, discriminator, epoch)
                iou = test_iou
                dice = test_dice

if __name__ == "__main__":


    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNetBEM(1, 1).to(device)
    model = nn.DataParallel(model) # .to(device)

    vis = Visualizer(env="vis_name")
    train_net(model,device)

















