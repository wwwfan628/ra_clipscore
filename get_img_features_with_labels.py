import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import natsort
from tqdm import tqdm
import re
import clip
import eval_clip
import pickle


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, anno_file_path='/Users/yifanlu/Desktop/Datasets/CelebA/Anno/list_attr_celeba.txt'):
        r = re.compile('[0-9]{6}.jpg')
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.anno_file = pd.read_csv(anno_file_path, header=[0], delimiter=r"\s+")
        # self.labels = [0 if r.match(f) else 1 for f in self.total_imgs]  # 0: face 1: furniture
        self.labels = [
            0 if r.match(f) and self.anno_file.loc[f]['Smiling'] == -1 else
            1 if r.match(f) and self.anno_file.loc[f]['Smiling'] == 1 else
            2
            for f in self.total_imgs] # 0: not black hair 1: black hair 2: furniture

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = self.labels[idx]
        return tensor_image, label


def get_img_features(img_dir, batch_size=1000):
    # calculate image features
    print("calculating image features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)
    model, preprocess = eval_clip.load("RN101", device=device)
    model.eval()
    dataset = CustomDataSet(main_dir=img_dir, transform=preprocess)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                             num_workers=2)

    img_features_list = []
    labels_list = []
    with torch.no_grad():
        for index, (datas) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # image_features = model.encode_image(datas[0].to(device))
            image_features, _ = model.encode_image(datas[0].to(device))
            img_features_list.append(image_features)
            labels_list.append(datas[1])

    img_features = torch.cat(img_features_list)
    labels = torch.cat(labels_list)
    return img_features, labels
    # return labels

if __name__ == '__main__':
    img_features, labels = get_img_features('/Users/yifanlu/Desktop/Datasets/CelebA5k_furniture15k')
    # labels = get_img_features('/Users/yifanlu/Desktop/Datasets/CelebA5k_furniture15k')
    with open(f'img_features_local.pkl', 'wb') as f:
        pickle.dump(img_features, f)

    with open(f'img_labels_blackhair.pkl', 'wb') as f:
        pickle.dump(labels, f)
