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
    count_positive = (anno_file == 1).sum().values
    attrs = (anno_file == 1).sum().index

    text_list = [a.lower().replace('_', ' ') for a in attrs]

    if 'vit' in args.score_type or 'rn' in args.score_type:
        text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    elif args.score_type == 'clip_score' or args.score_type == 'hcs':
        text_list_prompt = [f"This is a photo of {item}." for item in text_list]

    # calculate scores
    if 'vit' in args.score_type:
        scores, scores_per_patch, labels = core.get_evalcs_local_vit_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size) # [n_img, n_attr]  [n_img, 196, n_attr]
    elif 'rn' in args.score_type:
        scores, scores_per_patch, labels = core.get_evalcs_local_rn_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'clip_score':
        scores, labels = core.get_cs_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'hcs':
        scores, labels = core.get_hcs_with_labels(img_dir=args.img_dir, ref_img_dir=args.ref_img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)

    y_pred = np.zeros_like(scores, dtype=int)
    for j in range(len(count_positive)):
        topk_idx = np.argsort(scores[:, j])[::-1][:count_positive[j]]  # indices of top k scores
        y_pred[topk_idx, j] = 1

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
    parser.add_argument('--img_dir', type=str, default='/Users/yifanlu/Downloads/CelebA50k/Img', help='celeba50k_furniture15k images')  # /home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Img  /Users/yifanlu/Downloads/face50k_furniture15k/Img
    parser.add_argument('--ref_img_dir', type=str, default='/Users/yifanlu/Downloads/CelebA50k/Img', help='celeba 50k')  # /home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img  /Users/yifanlu/Downloads/face50k_furniture15k/Img
    parser.add_argument('--anno_file', type=str, default='/Users/yifanlu/Downloads/CelebA50k/Anno/list_attr_celeba50k.txt', help='celeba50k_furniture15k annotation file path')  # /home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Anno/list_attr_celeba50k_furniture15k.txt  /Users/yifanlu/Downloads/face50k_furniture15k/Anno/list_attr_celeba50k_furniture15k.txt
    parser.add_argument('--n_attr', type=int, default=40, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--score_type', type=str, default='EvalCS_local_vit', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to extract image features')

    args = parser.parse_args()
    main(args)