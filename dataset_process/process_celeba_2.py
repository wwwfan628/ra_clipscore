import pandas as pd
import numpy as np
import argparse
import random
import os
from tqdm import tqdm

def main(args):
    base_folder_path = '/Users/yifanlu/Desktop/Datasets/CelebA_process/CelebA_5k'
    fig_list = os.listdir(args.img_folder_path)
    random.shuffle(fig_list)
    for i in tqdm(range(len(fig_list)//args.num_figs)):
        target_folder_path = '/Users/yifanlu/Desktop/Datasets/CelebA_process/CelebA_5k_Eyeglasses_Injection_' + str((i+1) * args.num_figs)
        os.makedirs(target_folder_path, exist_ok=True)
        # copy all base figures
        scp_command_base = f'scp -r {base_folder_path}/. {target_folder_path}/.'
        os.system(scp_command_base)
        # copy all to be injected figures
        for j in tqdm(range((i+1)*args.num_figs)):
            fig_path = os.path.join(args.img_folder_path, fig_list[j])
            target_path = os.path.join(target_folder_path, fig_list[j])
            scp_command_fig = f'scp {fig_path} {target_path}'
            os.system(scp_command_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder_path', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/eyeglasses/train/with_eyeglasses')  # CelebA images folder
    parser.add_argument('--ann_file_path', type=str, default='/Users/yifanlu/Desktop/Datasets/Anno/CelebA/list_attr_celeba.txt')  # CelebA attributes annotation file path
    parser.add_argument('--num_figs', type=int, default=100)
    parser.add_argument('--target_folder_path', type=str, default='/Users/yifanlu/Desktop/CelebA_5K')  # save under this path

    args = parser.parse_args()

    main(args)