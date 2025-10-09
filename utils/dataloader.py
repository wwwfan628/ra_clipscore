from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import natsort
import numpy as np
import pandas as pd



class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


class AttrDataSet(Dataset):
    def __init__(self, img_dir, anno_file, attrs, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.attrs = attrs
        all_imgs = os.listdir(img_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        anno_file = pd.read_csv(anno_file, header=[0], delimiter=",")
        self.anno_file = anno_file.loc[self.total_imgs]
        self.labels = self.anno_file[self.attrs].values

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        labels = self.labels[idx]
        return tensor_image, labels


# class CustomDataSet2(Dataset):
#     def __init__(self, main_dir, transform):
#         self.main_dir = main_dir
#         self.transform = transform
#         all_imgs = os.listdir(main_dir)
#         self.total_imgs = natsort.natsorted(all_imgs)
#
#     def __len__(self):
#         return len(self.total_imgs)
#
#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
#         image = Image.open(img_loc).convert("RGB")
#         array_img = np.array(image)
#         ori_height, ori_width = array_img.shape[:2]
#         patch_size = 16
#         preprocess = self.transform(int(np.ceil(int(ori_height) / patch_size) * patch_size),
#                                     int(np.ceil(int(ori_width) / patch_size) * patch_size))
#         tensor_image = preprocess(image)
#         return tensor_image