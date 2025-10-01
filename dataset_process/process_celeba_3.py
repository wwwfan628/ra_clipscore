import pandas as pd
import numpy as np
import argparse
import random
import os
from tqdm import tqdm

def main(args):
    annotations = pd.read_csv(args.ann_file_path, header=[0], delimiter=r"\s+")
    male_figs = list(annotations[annotations['Male'] == 1].index)
    random.shuffle(male_figs)
    female_figs = list(annotations[annotations['Male'] == -1].index)
    random.shuffle(female_figs)

    for i in tqdm(range(11)):
        target_folder_path = os.path.join(args.output_path, f'man1k_woman{i}00')
        os.makedirs(target_folder_path, exist_ok=True)
        # copy 1k male faces
        for male_fig in male_figs[:1000]:
            fig_path = os.path.join(args.img_folder_path, male_fig)
            target_path = os.path.join(target_folder_path, male_fig)
            scp_command = f'scp {fig_path} {target_path}'
            os.system(scp_command)
        # copy i*100 female faces
        for female_fig in female_figs[:i*100]:
            fig_path = os.path.join(args.img_folder_path, female_fig)
            target_path = os.path.join(target_folder_path, female_fig)
            scp_command = f'scp {fig_path} {target_path}'
            os.system(scp_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder_path', type=str, default='/Users/yifanlu/Downloads/Datasets/CelebA/Img')  # CelebA images folder
    parser.add_argument('--ann_file_path', type=str, default='/Users/yifanlu/Downloads/list_attr_celeba.txt')  # CelebA attributes annotation file path
    parser.add_argument('--num_figs', type=int, default=5000)
    parser.add_argument('--output_path', type=str, default='/Users/yifanlu/Desktop/CelebA_gender')  # save under this path

    args = parser.parse_args()

    main(args)