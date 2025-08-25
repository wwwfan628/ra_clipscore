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

def main(args):
    start_time = time.time()

    experiment_name = f"{args.exp}_{args.n_attr}_{args.attr_type}_{args.score_type}"

    text_list, stats = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr, attr_type=args.attr_type)

    # text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    text_list_prompt = [f"This is a photo of {item}." for item in text_list]

    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)

    # path definition, creat directory if not exists
    save_dir = f"results_dir/{experiment_name}/{dataset_real}_vs_{dataset_gen}/"
    os.makedirs(save_dir, exist_ok=True)
    results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    # calculate text, image mean
    if os.path.exists(f'text_features_{experiment_name}.pkl'):
        with open(f'text_features_{experiment_name}.pkl', 'rb') as f:
            text_features = pickle.load(f)
    else:
        text_features = core.get_text_features_eval_clip_local_vit(text_list_prompt)  # [44, 512]
        with open(f'text_features_{experiment_name}.pkl', 'wb') as f:
            pickle.dump(text_features, f)

    if os.path.exists(f'img_features_{dataset_real}_{args.score_type}.pkl'):
        with open(f'img_features_{dataset_real}_{args.score_type}.pkl', 'rb') as f:
            img_features_real = pickle.load(f)
    else:
        img_features_real, attn_weight_list_real = core.get_img_features_eval_clip_local_vit(img_dir=args.real_dir, batch_size=1000)
        with open(f'img_features_{dataset_real}_{args.score_type}.pkl', 'wb') as f:
            pickle.dump(img_features_real, f)

    if os.path.exists(f'img_features_{dataset_gen}_{args.score_type}.pkl'):
        with open(f'img_features_{dataset_gen}_{args.score_type}.pkl', 'rb') as f:
            img_features_gen = pickle.load(f)
    else:
        img_features_gen, attn_weight_list_gen = core.get_img_features_eval_clip_local_vit(img_dir=args.gen_dir, batch_size=1000)   # [5782, 512, 50]
        with open(f'img_features_{dataset_gen}_{args.score_type}.pkl', 'wb') as f:
            pickle.dump(img_features_gen, f)

    # calculate scores
    scores_real = core.get_evalcs_local_vit(img_features_real, text_features, attn_weight_list_real).detach().cpu().numpy() # [n_batch, 197, n_attr]
    scores_gen = core.get_evalcs_local_vit(img_features_gen, text_features, attn_weight_list_gen).detach().cpu().numpy()    # [n_batch, 197, n_attr]

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)  # [197, n_attr]
    scores_mean_gen = scores_gen.mean(axis=0)   # [197, n_attr]

    scores_diff = scores_mean_real[1:] - scores_mean_gen[1:]    # [197, n_attr]
    scores_diff = (scores_diff - scores_diff.min()) / (scores_diff.max() - scores_diff.min() + 1e-8)

    # Visualization
    heatmap = (scores_diff.reshape(14, 14))
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar()
    plt.title("Heatmap of mean difference")
    plt.show()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='coco', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--real_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/airplane_100_1')
    parser.add_argument('--gen_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/airplane_100_2')
    parser.add_argument('--n_attr', type=int, default=1, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')
    parser.add_argument('--score_type', type=str, default='EvalCS', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')

    args = parser.parse_args()
    main(args)