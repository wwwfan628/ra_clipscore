import pandas as pd
import numpy as np
import argparse
import random
import os

from tqdm import tqdm

# Download furniture dataset from: https://www.kaggle.com/datasets/udaysankarmukherjee/furniture-image-dataset?resource=download


def main(args):
    os.makedirs(args.target_img_path, exist_ok=True)
    os.makedirs(args.target_anno_path, exist_ok=True)

    # move celeba faces
    annotations = pd.read_csv(args.celeba_anno_path, header=[0], delimiter=r"\s+")
    celeba_figs = annotations.index.tolist()
    random.shuffle(celeba_figs)
    annotations_celeba5k = annotations.loc[celeba_figs[:args.num_figs]]
    figs_list_furniture15k = os.listdir(args.furniture_path)
    labels_furniture15k = np.zeros([len(figs_list_furniture15k), len(annotations.columns)])
    annotations_furniture15k = pd.DataFrame(labels_furniture15k, index=figs_list_furniture15k, columns=annotations.columns)
    annotations_merged = pd.concat([annotations_celeba5k, annotations_furniture15k], ignore_index=False, sort=False)

    for fig_file in tqdm(celeba_figs[:args.num_figs]):
        fig_path = os.path.join(args.celeba_img_path, fig_file)
        target_path = os.path.join(args.target_img_path, fig_file)
        scp_command = f'scp {fig_path} {target_path}'
        os.system(scp_command)

    for fig_file in tqdm(figs_list_furniture15k):
        fig_path = os.path.join(args.furniture_path, fig_file)
        target_path = os.path.join(args.target_img_path, fig_file)
        scp_command = f'scp {fig_path} {target_path}'
        os.system(scp_command)

    annotations_merged.to_csv(os.path.join(args.target_anno_path, 'list_attr_celeba50k_furniture15k.txt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--furniture_path', type=str, default='/home/x_yiflu/dataset/Datasets/furniture_15k')  # furniture dataset folder
    parser.add_argument('--num_figs', type=int, default=50000)
    parser.add_argument('--celeba_anno_path', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Anno/list_attr_celeba.txt')
    parser.add_argument('--celeba_img_path', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img')  # celeba dataset folder
    parser.add_argument('--target_img_path', type=str, default='/home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Img')  # save under this path
    parser.add_argument('--target_anno_path', type=str, default='/home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Anno')  # save under this path

    args = parser.parse_args()

    main(args)