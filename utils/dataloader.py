from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import natsort
import numpy as np



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