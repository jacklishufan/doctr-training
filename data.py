import os
from torch.utils.data.dataset import Dataset
import os,cv2
import OpenEXR as exr
import numpy as np
import Imath
import torch
from PIL import Image

class Doc3DDataset(Dataset):
    def __init__(self,root='data') -> None:
        super().__init__()
        self.image_files = os.listdir(os.path.join(root,'imgs'))
        self.root = root

    def __len__(self):
        return len(self.image_files)

    def get_grid(self,f):
        exrfile = exr.InputFile(f)
        raw_bytesx = exrfile.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
        raw_bytesy = exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
        raw_bytes_mask = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
        depth_vectorx = np.frombuffer(raw_bytesx, dtype=np.float32)
        depth_vectory = np.frombuffer(raw_bytesy, dtype=np.float32)
        depth_vectorz = np.frombuffer(raw_bytes_mask, dtype=np.float32)
        height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
        width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
        depth_mapx = np.reshape(depth_vectorx, (height, width)) * 2 - 1
        depth_mapy = np.reshape(depth_vectory, (height, width)) * 2 - 1
        depth_vectorz = np.reshape(depth_vectorz, (height, width)) * 2 - 1
        grid = np.stack([depth_mapy,-depth_mapx],axis=2)
        return torch.from_numpy(grid),depth_vectorz

    def get_img(self,f):
        im_ori = np.array(Image.open(f))[:, :, :3] / 255. 
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im)
        return im
    def __getitem__(self, index) :
        file_name = self.image_files[index]
        img_path = os.path.join(self.root,'imgs',file_name)
        uv_path = os.path.join(self.root,'uvs',file_name.replace('.png','.EXR'))
        grid,mask = self.get_grid(uv_path)
        img = self.get_img(img_path)
        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255. 
        return img,grid,mask
        

