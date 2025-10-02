import pandas as pd
import numpy as np
import argparse
import random
import os
from tqdm import tqdm

# Download furniture dataset from: https://www.kaggle.com/datasets/udaysankarmukherjee/furniture-image-dataset?resource=download


def main(args):
    os.makedirs(args.target_path, exist_ok=True)
    classes = [f for f in os.listdir(args.dataset_path) if not f.startswith('.')]
    for class_name in classes:
        class_folder_path = os.path.join(args.dataset_path, class_name)
        fig_list = os.listdir(class_folder_path)
        random.shuffle(fig_list)
        for i in tqdm(range(args.num_figs//len(classes))):
            fig_path = os.path.join(class_folder_path, fig_list[i])
            target_fig_name = class_name + '_' + fig_list[i]
            target_path = os.path.join(args.target_path, target_fig_name)
            scp_command_man = f'scp {fig_path} {target_path}'
            commands = scp_command_man + ';'
            os.system(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/Users/yifanlu/Desktop/Datasets/furniture')  # furniture dataset folder
    parser.add_argument('--num_figs', type=int, default=15000)
    parser.add_argument('--target_path', type=str, default='/Users/yifanlu/Desktop/Datasets/furniture_15k')  # save under this path

    args = parser.parse_args()

    main(args)