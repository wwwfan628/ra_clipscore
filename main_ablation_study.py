import argparse
import os.path

import core
import utils
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from scipy.stats import gaussian_kde


def main(args):
    start_time = time.time()

    text_list = utils.load_candidate(experiment_name=f"{args.exp}_{args.n_attr}", n_attr=args.n_attr)

    text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]

    # path definition, creat directory if not exists
    save_dir = f"results_dir/ablation_study/"
    os.makedirs(save_dir, exist_ok=True)

    attrs = ["Black_Hair", "Eyeglasses", "Male", "Smiling"]
    # calculate scores
    if 'vit' in args.score_type:
        scores, scores_per_patch, labels = core.get_evalcs_local_vit_ablation_study(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size) # [n_img, n_attr]  [n_img, 196, n_attr]
    elif 'rn' in args.score_type:
        scores, scores_per_patch, labels = core.get_evalcs_local_rn(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'clip_score':
        scores, labels = core.get_cs_ablation_study(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'hcs':
        scores, labels = core.get_hcs_ablation_study(img_dir=args.img_dir, ref_img_dir=args.ref_img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)

    for i, attr in enumerate(attrs):
        scores_positive = scores[labels[i]==1]
        scores_negative = scores[labels[i]==-1]
        scores_furniture = scores[labels[i] == 0]

        # plot and save figures
        fig1, ax1 = plt.subplots()

        for x, label, color in zip(
                [scores_negative, scores_positive, scores_furniture],
                [f'w/o {text_list[i]}', f'w/ {text_list[i]}', 'furniture'],
                ['tab:red', 'tab:green', 'tab:blue']
        ):
            kde = gaussian_kde(x)
            x_range = np.linspace(min(x) - 0.1, max(x) + 0.1, 500)
            ax1.plot(x_range, kde(x_range), label=label, color=color)

        ax1.set_title(f'Distribution of {args.score_type}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True)

        result_save_path = f"{save_dir}/{attr}_{args.score_type}.png"
        fig1.savefig(result_save_path)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='ablation_study', help='experiment name, to find the corresponding attribute list')
    parser.add_argument('--img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Img', help='celeba50k_furniture15k images')
    parser.add_argument('--ref_img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img', help='celeba 50k')  #/Users/yifanlu/Downloads/Imagenet256-BigGAN/530  # /Users/yifanlu/Downloads/Imagenet256-LDM/530  # /Users/yifanlu/Downloads/imagenet256-StyleGAN-XL/530
    parser.add_argument('--anno_file', type=str, default='/home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Anno/list_attr_celeba50k_furniture15k.txt', help='celeba50k_furniture15k annotation file path')
    parser.add_argument('--n_attr', type=int, default=4, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--score_type', type=str, default='EvalCS_local_vit', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to extract image features')

    args = parser.parse_args()
    main(args)