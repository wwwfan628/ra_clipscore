import pandas as pd
import numpy as np
import argparse
import random
import os
from tqdm import tqdm

# Download furniture dataset from: https://www.kaggle.com/datasets/udaysankarmukherjee/furniture-image-dataset?resource=download


def main(args):
    os.makedirs(args.target_path, exist_ok=True)

    # move celeba faces
    annotations = pd.read_csv(args.celeba_ann_path, header=[0], delimiter=r"\s+")
    figs_list_face5k = os.listdir('/Users/yifanlu/Desktop/Datasets/CelebA_process/CelebA_5k')
    annotations_face_exclude5k = (annotations.loc[~ annotations.index.isin(figs_list_face5k)])
    male_figs = annotations_face_exclude5k[annotations_face_exclude5k['Male'] == 1].index.tolist()
    random.shuffle(male_figs)
    smile_figs = annotations_face_exclude5k[annotations_face_exclude5k['Smiling'] == 1].index.tolist()
    random.shuffle(smile_figs)
    eyeglasses_figs = annotations_face_exclude5k[annotations_face_exclude5k['Eyeglasses'] == 1].index.tolist()
    random.shuffle(eyeglasses_figs)
    to_move_face1k = list(set(male_figs[:args.num_figs//3] + smile_figs[:args.num_figs//3] + eyeglasses_figs[:args.num_figs//3]))

    while len(to_move_face1k)< args.num_figs:
        to_move_face1k += random.sample(annotations_face_exclude5k.index.tolist(), args.num_figs-len(to_move_face1k))

    for fig_file in tqdm(to_move_face1k):
        fig_path = os.path.join(args.celeba_path, fig_file)
        target_path = os.path.join(args.target_path, fig_file)
        scp_command = f'scp {fig_path} {target_path}'
        os.system(scp_command)

    classes = [f for f in os.listdir(args.furniture_path) if not f.startswith('.')]
    for class_name in classes:
        class_folder_path = os.path.join(args.furniture_path, class_name)
        fig_list = os.listdir(class_folder_path)
        random.shuffle(fig_list)
        for i in tqdm(range(args.num_figs // len(classes))):
            fig_path = os.path.join(class_folder_path, fig_list[i])
            target_fig_name = class_name + '_' + fig_list[i]
            target_path = os.path.join(args.target_path, target_fig_name)
            scp_command_man = f'scp {fig_path} {target_path}'
            commands = scp_command_man + ';'
            os.system(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--furniture_path', type=str, default='/Users/yifanlu/Desktop/Datasets/furniture')  # furniture dataset folder
    parser.add_argument('--num_figs', type=int, default=1000)
    parser.add_argument('--celeba_ann_path', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA/Anno/list_attr_celeba.txt')
    parser.add_argument('--celeba_path', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA/Img')  # celeba dataset folder
    parser.add_argument('--target_path', type=str, default='/Users/yifanlu/Desktop/Datasets/face1k_furniture1k/Img')  # save under this path

    args = parser.parse_args()

    main(args)