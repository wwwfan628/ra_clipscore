import pandas as pd
import numpy as np
import argparse
import random
import os

from tqdm import tqdm


def main(args):
    os.makedirs(args.target_img_path, exist_ok=True)
    os.makedirs(args.target_anno_path, exist_ok=True)

    # move celeba faces
    annotations = pd.read_csv(args.celeba_anno_path, header=[0], delimiter=r"\s+")
    celeba_figs = annotations.index.tolist()
    random.shuffle(celeba_figs)
    annotations_celeba50k = annotations.loc[celeba_figs[:args.num_figs]]

    for fig_file in tqdm(celeba_figs[:args.num_figs]):
        fig_path = os.path.join(args.celeba_img_path, fig_file)
        target_path = os.path.join(args.target_img_path, fig_file)
        scp_command = f'scp {fig_path} {target_path}'
        os.system(scp_command)

    annotations_celeba50k.to_csv(os.path.join(args.target_anno_path, 'list_attr_celeba50k.txt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_figs', type=int, default=50000)
    parser.add_argument('--celeba_anno_path', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Anno/list_attr_celeba.txt')
    parser.add_argument('--celeba_img_path', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img')  # celeba dataset folder
    parser.add_argument('--target_img_path', type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img')  # save under this path
    parser.add_argument('--target_anno_path', type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Anno')  # save under this path

    args = parser.parse_args()

    main(args)