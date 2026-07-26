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
from sklearn.metrics import f1_score, roc_auc_score



def get_num_images(action_file, target_actions):
    num_images_list = []

    for target_action in target_actions:
        found = False

        with open(action_file, "r") as f:
            next(f)  # Skip header

            for line in f:
                parts = line.split()
                action = parts[0]
                num_images = int(parts[-1])

                if action == target_action:
                    num_images_list.append(num_images)
                    found = True
                    break

        if not found:
            raise ValueError(f"{target_action} not found.")

    return num_images_list



def main(args):
    start_time = time.time()

    # How many images are positive for each attributes
    attrs = [ 'running', 'jumping']

    count_positive = get_num_images(args.action_file, attrs)

    # text_list = [a.lower().replace('_', ' ') for a in attrs]
    text_list = ['running', 'jumping']

    if 'vit' in args.score_type or 'rn' in args.score_type:
        text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    elif args.score_type == 'clip_score' or args.score_type == 'hcs':
        text_list_prompt = [f"This is a photo of {item}." for item in text_list]

    # calculate scores
    if 'vit' in args.score_type:
        scores, scores_per_patch, labels = core.get_evalcs_local_vit_with_labels_action(img_dir=args.img_dir, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size) # [n_img, n_attr]  [n_img, 196, n_attr]
    elif 'rn' in args.score_type:
        scores, labels = core.get_evalcs_local_rn_with_labels_action(img_dir=args.img_dir, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'clip_score':
        scores, labels = core.get_cs_with_labels_action(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'hcs':
        scores, labels = core.get_hcs_with_labels_action(img_dir=args.img_dir, ref_img_dir=args.ref_img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)

    y_pred = np.zeros_like(scores, dtype=int)
    for j in range(len(count_positive)):
        topk_idx = np.argsort(scores[:, j])[::-1][:count_positive[j]]  # indices of top k scores
        y_pred[topk_idx, j] = 1

    # Per-attribute accuracy
    attr_acc = (y_pred == labels).mean(axis=0)
    mean_attr_acc = attr_acc.mean()

    # Average Accuracy (mean balanced accuracy across attributes)
    average_acc_per_attr = []
    auroc_per_attr = []
    for j, attr in enumerate(attrs):
        positive_mask = labels[:, j] == 1
        negative_mask = labels[:, j] == 0

        if positive_mask.any() and negative_mask.any():
            true_positive_rate = y_pred[positive_mask, j].mean()
            true_negative_rate = (y_pred[negative_mask, j] == 0).mean()
            average_acc_per_attr.append(
                (true_positive_rate + true_negative_rate) / 2
            )
            auroc_per_attr.append(roc_auc_score(labels[:, j], scores[:, j]))
        else:
            print(
                f"Warning: skipping {attr!r} for Average Accuracy and AUROC "
                "because only one label class is present."
            )

    average_acc = (
        float(np.mean(average_acc_per_attr))
        if average_acc_per_attr
        else float("nan")
    )
    auroc = float(np.mean(auroc_per_attr)) if auroc_per_attr else float("nan")

    # F1 scores
    f1_macro = f1_score(labels, y_pred, average='macro')
    f1_micro = f1_score(labels, y_pred, average='micro')

    print("Per-attribute Accuracy:", mean_attr_acc)
    print("Average Accuracy:", average_acc)
    print("AUROC:", auroc)
    print("Macro F1:", f1_macro)
    print("Micro F1:", f1_micro)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='celeba_classification', help='experiment name, to find the corresponding attribute list')
    parser.add_argument('--img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img', help='celeba50k images')  # /home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img  /Users/yifanlu/Downloads/CelebA50k/Img
    parser.add_argument('--action_file', type=str, default='/home/x_yiflu/dataset/Datasets/Stanford40/ImageSplits/action.txt', help='action.txt file')
    parser.add_argument('--ref_img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/furniture_15k', help='celeba 50k')  # /home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img  /Users/yifanlu/Downloads/CelebA50k/Img
    parser.add_argument('--score_type', type=str, default='EvalCS_local_vit', help='score type, e.g. hcs, clip_score, EvalCS_local_vit, EvalCS_local_rn')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to extract image features')

    args = parser.parse_args()
    main(args)
