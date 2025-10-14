import argparse
import os.path
import pandas as pd
import core
import utils
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def main(args):
    start_time = time.time()

    anno_file = pd.read_csv(args.anno_file, header=[0], delimiter=",")
    # How many images are positive for each attributes
    # attrs = list(anno_file.columns)
    attrs = [
        'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Male',
        'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Sideburns', 'Smiling',
        'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
        'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]

    count_positive = (anno_file[attrs] == 1).sum().values

    # text_list = [a.lower().replace('_', ' ') for a in attrs]
    text_list = [
        'arched eyebrows', 'bags under eyes', 'bald', 'bangs', 'big nose',
        'black hair', 'blond hair', 'brown hair', 'chubby', 'double chin',
        'eyeglasses', 'goatee', 'gray hair', 'heavy makeup', 'male',
        'mouth slightly open', 'mustache', 'beard', 'sideburns', 'smiling',
        'straight hair', 'wavy hair', 'wearing earrings', 'wearing hat',
        'wearing lipstick', 'wearing necklace', 'wearing necktie', 'young'
    ]

    if 'vit' in args.score_type or 'rn' in args.score_type:
        text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    elif args.score_type == 'clip_score' or args.score_type == 'hcs':
        text_list_prompt = [f"This is a photo of {item}." for item in text_list]

    # calculate scores
    if 'vit' in args.score_type:
        scores, scores_per_patch, labels = core.get_evalcs_local_vit_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size) # [n_img, n_attr]  [n_img, 196, n_attr]
    elif 'rn' in args.score_type:
        scores, labels = core.get_evalcs_local_rn_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'clip_score':
        scores, labels = core.get_cs_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'hcs':
        scores, labels = core.get_hcs_with_labels(img_dir=args.img_dir, ref_img_dir=args.ref_img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)

    y_pred = np.zeros_like(scores, dtype=int)
    for j in range(len(count_positive)):
        topk_idx = np.argsort(scores[:, j])[::-1][:count_positive[j]]  # indices of top k scores
        y_pred[topk_idx, j] = 1

    labels[:, attrs.index('No_Beard')] = labels[:, attrs.index('No_Beard')] * (-1)
    labels[labels==-1] = 0

    # Per-attribute accuracy
    attr_acc = (y_pred == labels).mean(axis=0)
    mean_attr_acc = attr_acc.mean()

    # F1 scores
    f1_macro = f1_score(labels, y_pred, average='macro')
    f1_micro = f1_score(labels, y_pred, average='micro')

    print("Per-attribute Accuracy:", mean_attr_acc)
    print("Macro F1:", f1_macro)
    print("Micro F1:", f1_micro)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='celeba_classification', help='experiment name, to find the corresponding attribute list')
    parser.add_argument('--img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img', help='celeba50k images')  # /home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img  /Users/yifanlu/Downloads/CelebA50k/Img
    parser.add_argument('--ref_img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/ffhq/DiffuseIT_ffhq_make_up', help='celeba 50k')  # /home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img  /Users/yifanlu/Downloads/CelebA50k/Img
    parser.add_argument('--anno_file', type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Anno/list_attr_celeba50k.txt', help='celeba50k_furniture15k annotation file path')  # /home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Anno/list_attr_celeba50k.txt  /Users/yifanlu/Downloads/CelebA50k/Anno/list_attr_celeba50k.txt
    parser.add_argument('--score_type', type=str, default='EvalCS_local_vit', help='score type, e.g. hcs, clip_score, EvalCS_local_vit, EvalCS_local_rn')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to extract image features')

    args = parser.parse_args()
    main(args)