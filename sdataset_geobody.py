""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
import skimage.morphology as sm
# from utils import random_click
import random

import sys


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def show_box(box,ax):
    x0,y0 = box[0],box[1]
    w,h = box[2]-box[0],box[3]-box[1]
    ax.add_patch(plt.Rectangle((x0,y0),w,h,edgecolor='green',facecolor=(0,0,0,0),lw=2))


def random_click(mask, point_labels = 1, inout = 1):
    maskEroison = sm.binary_erosion(mask, sm.disk(7))
    indices = np.argwhere(maskEroison == inout)
    return indices[np.random.randint(len(indices))]

def random_click_global(mask):
    bw,bh = 12,12
    bbmask = np.zeros_like(mask)
    bbmask[bw:-bw,bh:-bh] = 1
    indices = np.argwhere(bbmask == 1)
    pos = indices[np.random.randint(len(indices))]
    posX,posY = pos
    catInd = mask[posX,posY]
    rmask = (mask==catInd).astype(np.int16)
    # print(pos.shape)
    return pos,rmask


def generateBbox(indices,hh,ww,dila_size=40):
    bbox1 = indices[:,1].min()-np.random.randint(dila_size)
    bbox2 = indices[:,0].min()-np.random.randint(dila_size)
    bbox3 = indices[:,1].max()+np.random.randint(dila_size)
    bbox4 = indices[:,0].max()+np.random.randint(dila_size)
    bbox1 = bbox1 if bbox1 > 0 else 0
    bbox2 = bbox2 if bbox2 > 0 else 0
    bbox3 = bbox3 if bbox3 < hh else hh-1
    bbox4 = bbox4 if bbox4 < ww else ww-1
    
    bbox  = np.array([bbox1,bbox2,bbox3,bbox4])
    return bbox

def random_click_specified(mask, val_sp=1):
    hh,ww = np.squeeze(mask).shape
    # pt_num = np.random.randint(1,10)
    pt_num = 20
    indices = np.argwhere(mask == val_sp)

    # print(indices[:,1].min(),indices[:,1].max(),indices[:,0].min(),indices[:,0].max())
    rmask = (mask==val_sp).astype(np.int16)
    hh,ww = rmask.shape
    
    pos = indices[np.random.choice(len(indices),pt_num,replace=False)]

    # negPtIndices = np.argwhere(rmask==0)

    newpos = np.zeros_like(pos)
    newpos[:,0] = pos[:,1]
    newpos[:,1] = pos[:,0]
    pos_lab = np.ones((pt_num,))

    bbox = generateBbox(indices,hh,ww)
    return newpos,pos_lab,rmask,bbox
    # return np.squeeze(newpos),1,rmask,bbox
    
def createWellLogMask(low_mask):
    lmk = low_mask[...,0]
    hl,wl = np.where(lmk==1)
    wlmin,wlmax = wl.min(),wl.max()
    centerLine = np.random.randint(wlmin+5,wlmax-5)
    # print("centerLine: ")
    # print(centerLine)
    wellLogMask = np.zeros_like(lmk)
    wellLogMask[:,centerLine-5:centerLine+5] = 1
    lmk *= wellLogMask
    return lmk[None,...]



class geoDataset(Dataset):
    def __init__(self,dataPath,metaFile,img_size=1024):
        self.dataPath = dataPath
        self.metaFile = metaFile
        self.img_size = img_size
        self.pixel_mean=[123.675, 116.28, 103.53]
        self.pixel_std=[58.395, 57.12, 57.375]

        self.loadDataPara()
        

    def loadDataPara(self):
        with open(self.metaFile,'r') as f:
            self.alfArr = f.readlines()
        

    def __getitem__(self,ind):
        cv2.setNumThreads(0)
        
        tmpPara = self.alfArr[ind]
        tmpParaArr = tmpPara.split()
        
        tmpCategory = tmpParaArr[1]
        tmpSxPath = tmpParaArr[2]
        tmpLxPath = tmpParaArr[3]
        tmpValue  = int(tmpParaArr[4])
        tmpSrcFile= tmpParaArr[5]
        tmpLoca= tmpParaArr[6]
        
        # print(tmpSxPath)
        sx = cv2.imread(os.path.join(self.dataPath,tmpSxPath))
        lx = cv2.imread(os.path.join(self.dataPath,tmpLxPath))
        
        newsize = (self.img_size, self.img_size)
        sx = cv2.resize(sx,newsize)
        lx = cv2.resize(lx,newsize)
        tlx = lx[...,0]

        sx = (sx - self.pixel_mean) / self.pixel_std
        sx = sx.transpose((2,0,1))

        
        # tx = (tlx==tmpValue).astype(np.single)
        tx = ((tlx>=tmpValue-1)&(tlx<=tmpValue+1)).astype(np.single)
        
        # tx = sm.binary_opening(tx, sm.disk(3))

        # tx = sm.binary_opening(tx, sm.disk(7))
        # tx = sm.binary_closing(tx, sm.disk(4)) 
        # tx = tx.astype(np.int8)
        # print(tx.max(),tx.min())

        

        pt,pt_lab,rmask,bbox = random_click_specified(np.array(tx),1)
        low_mask = cv2.resize(lx,(256,256))
        low_mask = (low_mask==tmpValue).astype(np.single)
        low_mask = createWellLogMask(low_mask)
        tx = tx[np.newaxis,...]
        image_meta_dict = {
            "category":tmpCategory,
            "imgFile":tmpSxPath,
            "labFile":tmpLxPath,
            "filename_or_obj": tmpSrcFile,
            "location": tmpLoca,
        }

        return {
            "image":sx,
            "label":tx,
            'p_label':pt_lab, # point_label
            'pt':pt, # point
            'bbox': bbox, # box
            'low_mask': low_mask,
            'rawInfo':tmpParaArr,
            'image_meta_dict':image_meta_dict,
        }
        # return {
        #     'image':img, # inputs
        #     'label': mask, # mask
        #     'p_label':pt_lab, # point_label
        #     'pt':pt, # point
        #     'bbox': bbox, # box
        #     'low_mask': low_mask,
        #     'image_meta_dict':image_meta_dict,
        # }


    def __len__(self):
        return len(self.alfArr)


if __name__ == '__main__':

    prefix = "/Your_Path/samDataset/"
    gd = geoDataset(prefix,prefix+"tinyTrain.txt")
    sz = gd[int(sys.argv[1])]

    # for ind in range(len(gd)):
    #     print(ind)
    #     tmpsz = gd[ind]

    
    print(sz["image"].shape)
    print(sz["label"].shape)
    print(sz['pt'].shape)
    print(sz['p_label'])
    print(sz['bbox'].shape)
    print(sz['low_mask'].shape)

    # np.save("tmptx.npy",sz['label'])
    # print(sz["imgFile"])
    # print(sz["labFile"])

    plt.figure()
    plt.subplot(131)
    plt.imshow(sz["image"].transpose(1,2,0))
    plt.subplot(132)
    plt.imshow(sz["image"].transpose(1,2,0))
    plt.imshow(sz["label"][0,...],cmap='jet',alpha=.5)
    show_box(sz['bbox'],plt.gca())
    plt.subplot(133)
    plt.imshow(sz["label"][0,...])
    # show_points(sz['pt'],sz['p_label'],plt.gca())
    
    plt.savefig("test.jpg")

    
