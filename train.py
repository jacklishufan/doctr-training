from builtins import breakpoint
import enum
from seg import U2NETP
from GeoTr import GeoTr
from IllTr import IllTr
from inference_ill import rec_ill

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
import cv2
import glob
import os
from PIL import Image
import argparse
import warnings
from data import Doc3DDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
warnings.filterwarnings('ignore')


class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk.detach() * x

        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        return bm,msk
        

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        

def rec(opt):
    # print(torch.__version__) # 1.5.1
    if not os.path.exists(opt.gsave_path):  # create save path
        os.mkdir(opt.gsave_path)
    if not os.path.exists(opt.isave_path):  # create save path
        os.mkdir(opt.isave_path)
    
    GeoTr_Seg_model = GeoTr_Seg().cuda()
    # reload segmentation model
    reload_segmodel(GeoTr_Seg_model.msk, opt.Seg_path)
    # reload geometric unwarping model
    reload_model(GeoTr_Seg_model.GeoTr, opt.GeoTr_path)
    
    IllTr_model = IllTr().cuda()
    # reload illumination rectification model
    reload_model(IllTr_model, opt.IllTr_path)
    
    dataset = Doc3DDataset()
    loader = DataLoader(dataset,shuffle=True,batch_size=1)
    # To eval mode
    GeoTr_Seg_model.train()
    IllTr_model.train()

    num_epochs = 30
    optimizer = AdamW(GeoTr_Seg_model.parameters(),lr=1e-4, weight_decay=1e-4)
    for epoch in range(num_epochs):
        #losses = []
        for idx,(img,grid,mask) in enumerate(loader):
             optimizer.zero_grad()
             bm,msk = GeoTr_Seg_model(img.cuda())
             grid = grid.cuda()
             bm_transformed =F.grid_sample(F.interpolate(bm,(448,448)), grid, align_corners=True)
             grid_normalized =  (grid + 1)/2
             # CE Loss
             mask = ((mask+1)/2 ).cuda()
             ce_loss = F.binary_cross_entropy(msk,F.interpolate(mask.unsqueeze(0),(288,288)).float())
             tr_loss = torch.abs(grid_normalized.permute(0,3,1,2)-bm_transformed) 
             b,h,w = mask.shape
             mask = mask.view(b,1,h,w).repeat(1,2,1,1)
             tr_loss *= mask
             tr_loss = tr_loss.mean()
             loss = tr_loss  + ce_loss # disable for now, seems unstable
             #losses.append(loss.detach().cpu().item())
             optimizer.step()
             if idx % 5 == 0:
                print(f'Epoch {epoch+1} ITER {idx+1} Loss  {loss.item()} Loss REG  {tr_loss.item()} Loss CE  {ce_loss.item()}: ')
        #print(f'Epoch {epoch+1} Loss AVG {np.mean(losses)}: ')
    torch.save(GeoTr_Seg_model.GeoTr.state_dict(),'geo_tr_final.pth')
    torch.save(GeoTr_Seg_model.msk.state_dict(),'geo_seg_final.pth')


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--distorrted_path',  default='./distorted/')
    parser.add_argument('--gsave_path',  default='./geo_rec/')
    parser.add_argument('--isave_path',  default='./ill_rec/')
    parser.add_argument('--Seg_path',  default='./model_pretrained/seg.pth')
    parser.add_argument('--GeoTr_path',  default='./model_pretrained/geotr.pth')
    parser.add_argument('--IllTr_path',  default='./model_pretrained/illtr.pth')
    parser.add_argument('--ill_rec',  default=False)
    
    opt = parser.parse_args()

    rec(opt)


if __name__ == '__main__':
    main()
