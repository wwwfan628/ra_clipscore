import torch
from tqdm import tqdm
import numpy as np
import os
import torchvision
import eval_clip
from utils import *

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_text_features_eval_clip(text_list):
    print("calculating text features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = eval_clip.load("RN101", device=device)
    model.eval()
    with torch.no_grad():
        text = eval_clip.tokenize(text_list).to(device) # [44, 77]
        text_features = model.encode_text(text) # [44, 512]
    return text_features


def get_text_features_eval_clip_local_vit(text_list):
    print("calculating text features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = eval_clip.load("ViT-B/16", device=device)
    model.eval()
    with torch.no_grad():
        text = eval_clip.tokenize(text_list).to(device) # [44, 77]
        text_features = model.encode_text(text) # [44, 512]
    return text_features


def get_img_features_eval_clip(img_dir, batch_size=1000):
    # calculate image features
    print("calculating image features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = eval_clip.load("RN101", device=device)
    model.eval()
    dataset = CustomDataSet(main_dir=img_dir, transform=preprocess)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                             num_workers=2)

    img_features_list = []

    with torch.no_grad():
        for index, (datas) in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_features, _ = model.encode_image(datas.to(device))   # datas [1000, 3, 224, 224]  image features [1000, 512, 50]
            # image_features = model.encode_image(datas.to(device))
            img_features_list.append(image_features)

    img_features = torch.cat(img_features_list) # [5000, 512, 50]
    return img_features


def get_img_features_eval_clip_local_vit(img_dir, batch_size=1000):
    # calculate image features
    print("calculating image features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = eval_clip.load("ViT-B/16", device=device)
    model.eval()
    dataset = CustomDataSet(main_dir=img_dir, transform=preprocess)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                             num_workers=2)

    img_features_list = []

    with torch.no_grad():
        for index, (datas) in tqdm(enumerate(dataloader), total=len(dataloader)):   # [1000, 3, 224, 224]
            h, w = model.visual.input_resolution, model.visual.input_resolution
            image_features, attn_weight_list = model.encode_image(datas.to(device), h, w, attn_mask=1)   # datas [1000, 3, 224, 224]  image features [1000, 512, 50]
            # image_features = model.encode_image(datas.to(device))
            img_features_list.append(image_features)

    img_features = torch.cat(img_features_list) # [5000, 512, 50]
    return img_features, attn_weight_list


def get_evalcs(img_features, text_features):
    assert text_features.shape[0] % 2 == 0
    print(f"calculating EvalCS")

    # normalize features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [44, 512]
    img_features = img_features.transpose(1,2)
    image_features_norm = img_features / img_features.norm(dim=1, keepdim=True) # [5000, 512, 50]

    # Class-Specific Region Feature Aggregation
    output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])  # input [5000, 512, 50] weight [44, 512, 1] out [5000, 44, 50]
    b, c, _ = output.shape  # b 5000 c 44
    output_half = output[:, c // 2:]  # [5000, 22, 50]
    w_half = F.softmax(output_half, dim=-1)  # [5000, 22, 50]
    w = torch.cat([w_half, w_half], dim=1)  # [5000, 44, 50]
    output = 5 * (output * w).sum(-1)  # [5000, 44]

    b, c = output.shape

    # convert the shape of logits to [b, 2, num_class]
    logits = output.resize(b, 2, c // 2)  # [5000, 2, 22]
    Softmax = torch.nn.Softmax(dim=1)
    scores = Softmax(logits).cpu()[:, 1]    # [5000, 22]

    # # Class-Specific Region Feature Aggregation
    # output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])  # input [5000, 512, 50] weight [44, 512, 1] out [5000, 44, 50]
    # b, a, p = output.shape  # b 5000 a 44 p 50
    # logits_per_patch = output.reshape(b, 2, a // 2, p)  # [5000, 2, 22, 50]
    # scores_per_patch = F.softmax(logits_per_patch, dim=1)[:, 1]   # [5000, 22, 50]
    # w = F.softmax(output[:, a // 2:], dim=-1)  # [5000, 22, 50]
    # scores = 5 * (scores_per_patch * w).sum(-1)  # [5000, 22]
    return scores


def get_evalcs_local_vit(img_features, text_features, attn_weight_list):
    print(f"calculating EvalCS with local ViT")

    # normalize features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [n_attr, 512]
    image_features_norm = img_features / img_features.norm(dim=1, keepdim=True) # [n_batch, 197, 512]

    # cosine similarity as logits
    logit_scale = 100.000
    logits_per_patch = logit_scale * image_features_norm @ text_features.t()    # [n_batch, 197, n_attr]
    # logits_per_patch = logits_per_patch.softmax(dim=-1)
    attn_weight = torch.stack(attn_weight_list[:-1], dim=0).mean(dim=0)
    logits_per_patch = torch.matmul(attn_weight, logits_per_patch)  # [n_batch, 197, 197] [n_batch, 197, n_attr] [n_batch, 197, n_attr]
    # logits_per_patch = logits_per_patch.softmax(dim=-1)
    # class_per_patch = logits_per_patch.squeeze().argmax(dim=1)[1:]

    # # Class-Specific Region Feature Aggregation
    # output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])  # input [5000, 512, 50] weight [44, 512, 1] out [5000, 44, 50]
    # b, c, _ = output.shape  # b 5000 c 44
    # output_half = output[:, c // 2:]  # [5000, 22, 50]
    # w_half = F.softmax(output_half, dim=-1)  # [5000, 22, 50]
    # w = torch.cat([w_half, w_half], dim=1)  # [5000, 44, 50]
    # output = 5 * (output * w).sum(-1)  # [5000, 44]
    #
    # b, c = output.shape
    #
    # # convert the shape of logits to [b, 2, num_class]
    # logits = output.resize(b, 2, c // 2)  # [5000, 2, 22]
    # Softmax = torch.nn.Softmax(dim=1)
    # scores = Softmax(logits).cpu()[:, 1]    # [5000, 22]

    # # Class-Specific Region Feature Aggregation
    # output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])  # input [5000, 512, 50] weight [44, 512, 1] out [5000, 44, 50]
    # b, a, p = output.shape  # b 5000 a 44 p 50
    # logits_per_patch = output.reshape(b, 2, a // 2, p)  # [5000, 2, 22, 50]
    # scores_per_patch = F.softmax(logits_per_patch, dim=1)[:, 1]   # [5000, 22, 50]
    # w = F.softmax(output[:, a // 2:], dim=-1)  # [5000, 22, 50]
    # scores = 5 * (scores_per_patch * w).sum(-1)  # [5000, 22]
    return logits_per_patch
