import pandas as pd
import numpy as np
import argparse
import random
import os
from tqdm import tqdm

def main(args):
    fig_list = os.listdir(args.img_folder_path)
    random.shuffle(fig_list)
    os.makedirs(args.target_folder_path, exist_ok=True)
    for i in tqdm(range(args.num_figs)):
        fig_path = os.path.join(args.img_folder_path, fig_list[i])
        target_path = os.path.join(args.target_folder_path, fig_list[i])
        scp_command_man = f'scp {fig_path} {target_path}'
        commands = scp_command_man + ';'
        os.system(commands)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder_path', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA/Img')  # CelebA images folder
    parser.add_argument('--ann_file_path', type=str, default='/Users/yifanlu/Desktop/Datasets/Anno/CelebA/list_attr_celeba.txt')  # CelebA attributes annotation file path
    parser.add_argument('--num_figs', type=int, default=5000)
    parser.add_argument('--target_folder_path', type=str, default='/Users/yifanlu/Desktop/CelebA_5K')  # save under this path

    args = parser.parse_args()

    main(args)