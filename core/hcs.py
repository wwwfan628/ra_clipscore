import torch
import clip
from tqdm import tqdm 
import numpy as np
import os
import torchvision
from utils import *

CLIP_BACKBONE = "ViT-B/32"   # "RN101"    # "ViT-B/32"

def get_text_mean(text_list, clip_backbone=CLIP_BACKBONE):
    print("calculating text mean")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_backbone, device=device)
    
    with torch.no_grad():
        text = clip.tokenize(text_list).to(device)
        for index, single_txt in tqdm(enumerate(text), total = text.shape[0]):
            text_feature = model.encode_text(single_txt.unsqueeze(0))
            text_features = text_feature if index==0 else torch.vstack((text_features,text_feature))
        text_mean = torch.mean(text_features, axis=0)
                            
    return text_mean 


def get_img_mean(img_dir, batch_size, clip_backbone=CLIP_BACKBONE):
    print("calculating image mean")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_backbone, device=device)
    model.eval()
    #real_dataset = torchvision.datasets.DatasetFolder(root=f"{img_dir}", loader=custom_loader, transform=preprocess, extensions=) 
    real_dataset = CustomDataSet(main_dir=f"{img_dir}", transform=preprocess)
    
    real_dataloader = torch.utils.data.DataLoader(
                real_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)

    img_list_box = torch.zeros((len(real_dataloader.dataset),512),dtype=torch.float).to(device)
    
    with torch.no_grad():   
        for index, (datas) in tqdm(enumerate(real_dataloader), total = len(real_dataloader)):
            image_features = model.encode_image(datas.to(device))
            img_list_box[index*batch_size:(index+1)*batch_size]=image_features
            
    img_mean = img_list_box.mean(dim=0)

    return img_mean


def get_hcs(img_dir, img_mean, text_mean, text_list, batch_size=1000, clip_backbone=CLIP_BACKBONE):
    print(f"calculating HCS for {img_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_backbone, device=device)
    real_dataset = CustomDataSet(main_dir=f"{img_dir}", transform=preprocess)
    real_dataloader = torch.utils.data.DataLoader(
                real_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
                
    img_mean = img_mean.to(torch.float)
    img_mean = img_mean.unsqueeze(0).to(device)

    text_list_prompt = [f"This is a photo of {a}." for a in text_list]
    text = clip.tokenize(text_list_prompt).to(device)
    attribute_num = len(text_list_prompt)
    
    with torch.no_grad():
        text_features = model.encode_text(text)        
        similarity_per_attribute = torch.zeros((len(real_dataloader.dataset),attribute_num),dtype=torch.float).to(device)
        
        for index, (datas) in tqdm(enumerate(real_dataloader), total = len(real_dataloader)):
            image_features = model.encode_image(datas.to(device))                                
            img_feat = image_features - img_mean
            text_feat = text_features - text_mean
            img_feat /= img_feat.norm(dim=-1, keepdim=True)   
             
            for i in range(attribute_num): 
                text_feat_temp = text_feat[i]
                text_feat_temp /= text_feat_temp.norm(dim=-1, keepdim=True)
                sim = (100.0 * img_feat @ text_feat_temp.T)
                similarity_per_attribute[index*batch_size:(index+1)*batch_size,i]=sim

    return similarity_per_attribute


#########################################################################################################################

def get_hcs_for_one_img(img, img_mean, text_mean, text_list, clip_backbone=CLIP_BACKBONE):
    print(f"calculating HCS for {img}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_backbone, device=device)
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)

    img_mean = img_mean.to(torch.float16)
    img_mean = img_mean.unsqueeze(0).to(device)

    text_list_prompt = ["a photo of " + a for a in text_list]
    text = clip.tokenize(text_list_prompt).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        image_features = model.encode_image(image)
        img_feat = image_features - img_mean
        text_feat = text_features - text_mean
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        scores = (100.0 * img_feat @ text_feat.T)
    return scores.squeeze()



def get_dualcs_for_one_img(img, text_list, clip_backbone=CLIP_BACKBONE):
    print(f"calculating DualCS for {img}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_backbone, device=device)
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)

    text_list_prompt = [f"This is a photo of {item}." if i == 0 else f"This is a photo without {item}." for item in text_list for i in range(2)]
    text = clip.tokenize(text_list_prompt).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        cs = image_features @ text_features.T
        cs_softmax = torch.softmax(cs.reshape([-1, 2]), dim=1)
        scores = 100 * ((cs_softmax * 2)-1)
        scores = scores.reshape([1, -1])
    return scores.squeeze()


def get_text_features(text_list, clip_backbone=CLIP_BACKBONE):
    print("calculating text features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_backbone, device=device)

    with torch.no_grad():
        text = clip.tokenize(text_list).to(device)
        text_features = model.encode_text(text)
    return text_features

def get_img_features(img_dir, batch_size=1000, clip_backbone=CLIP_BACKBONE):
    # calculate image features
    print("calculating image features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_backbone, device=device)
    model.eval()
    dataset = CustomDataSet(main_dir=img_dir, transform=preprocess)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                             num_workers=2)

    img_features_list = []

    with torch.no_grad():
        for index, (datas) in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_features = model.encode_image(datas.to(device))
            img_features_list.append(image_features)

    img_features = torch.cat(img_features_list)
    return img_features

def get_hcs_new(img_features_real, img_features_gen, text_features):
    print(f"calculating HCS")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_mean = torch.mean(img_features_real, axis=0)
    img_mean = img_mean.unsqueeze(0).to(device)
    text_mean = torch.mean(text_features, axis=0)

    img_features = img_features_gen - img_mean
    text_features = text_features - text_mean
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100 * img_features @ text_features.T)

    return similarity

def get_dualcs(img_features, text_features):
    assert text_features.shape[0] % 2 == 0
    num_attributs = text_features.shape[0] // 2
    print(f"calculating DualCS")
    with torch.no_grad():
        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        cs = img_features @ text_features.T
        cs_softmax = torch.softmax(cs.reshape([-1, num_attributs, 2]), dim=2)
        # scores = 100 * ((cs_softmax * 2)-1)
        scores = 100 * cs_softmax
        # scores = 100 * (cs_softmax>=0.5).float()
        # scores = scores.reshape([-1, 40])
        scores = scores[:, :, 0]
    return scores


def get_hcs_scaled(img_features, text_features):
    print(f"calculating Scaled HCS")
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    cs = img_features @ text_features.T
    cs_softmax = torch.softmax(cs, dim=1)
    scores = 100 * cs_softmax
    return scores


def get_original_cs(img_features, text_features):
    print(f"calculating Original CS")
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    scores = 100 * (img_features @ text_features.T)
    return scores


def get_hcs_without_minus_mean_text_features(img_features_real, img_features_gen, text_features):
    print(f"calculating HCS without minus mean text features")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_mean = torch.mean(img_features_real, axis=0)
    img_mean = img_mean.unsqueeze(0).to(device)

    img_features = img_features_gen - img_mean
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100 * img_features @ text_features.T)

    return similarity

#########################################********************************************************

def get_hcs_try(img_features_real, img_features_gen, text_features):
    print(f"calculating HCS")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_mean = torch.mean(img_features_real, axis=0)
    img_mean = img_mean.unsqueeze(0).to(device)
    text_mean = torch.mean(text_features, axis=0)

    img_features = img_features_gen - img_mean
    text_features = text_features - text_mean
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # lambda_reg = 1e-3
    XXT = text_features @ text_features.mT  # (22, 22)
    # inv_XXT = torch.inverse(XXT + lambda_reg * torch.eye(XXT.shape[0]))
    inv_XXT = torch.inverse(XXT)
    weights = 100 * img_features @ text_features.T @ inv_XXT  # (1000, 22)
    return weights

