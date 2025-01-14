import argparse
import os.path

import core
import utils
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def main(args):

    if args.score_type == 'dualCS':
        args.n_attr *= 2

    experiment_name= f"{args.exp}_{args.n_attr}_{args.attr_type}_{args.score_type}"

    text_list, stats = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr, attr_type=args.attr_type)

    real_dataset = os.path.basename(args.real_dir)
    gen_dataset = os.path.basename(args.gen_dir)

    # calculate text, image mean
    if os.path.exists(f'text_features_{experiment_name}.pkl'):
        with open(f'text_features_{experiment_name}.pkl', 'rb') as f:
            text_features = pickle.load(f)
            text_mean = text_features.mean(dim=0)
    else:
        text_features = core.get_text_features(text_list)
        text_mean = text_features.mean(dim=0)
        with open(f'text_features_{experiment_name}.pkl', 'wb') as f:
            pickle.dump(text_features, f)

    if os.path.exists(f'img_features_{real_dataset}.pkl'):
        with open(f'img_features_{real_dataset}.pkl', 'rb') as f:
            img_features_real = pickle.load(f)
            img_mean_real = img_features_real.mean(dim=0)
    else:
        img_features_real = core.get_img_features(img_dir=args.real_dir, batch_size=1000)
        img_mean_real = img_features_real.mean(dim=0)
        with open(f'img_features_{real_dataset}.pkl', 'wb') as f:
            pickle.dump(img_features_real, f)

    if os.path.exists(f'img_features_{gen_dataset}.pkl'):
        with open(f'img_features_{gen_dataset}.pkl', 'rb') as f:
            img_features_gen = pickle.load(f)
            img_mean_gen = img_features_gen.mean(dim=0)
    else:
        img_features_gen = core.get_img_features(img_dir=args.gen_dir, batch_size=1000)
        img_mean_gen = img_features_gen.mean(dim=0)
        with open(f'img_features_{gen_dataset}.pkl', 'wb') as f:
            pickle.dump(img_features_gen, f)

    # calculate scores
    if args.score_type == 'HCS':
        scores_real = core.get_hcs(args.real_dir, img_mean_real, text_mean, text_list, batch_size=1000)
        scores_gen = core.get_hcs(args.gen_dir, img_mean_real, text_mean, text_list, batch_size=1000)
        # score_real = core.get_hcs_new(img_features_real, img_features_real, text_features)
        # score_gen = core.get_hcs_new(img_features_real, img_features_gen, text_features)
    elif args.score_type == 'dualCS':
        scores_real = core.get_dualcs(img_features_real, text_features)
        scores_gen = core.get_dualcs(img_features_gen, text_features)

    # calculate SaD and PaD
    SaD = core.get_SaD(scores_real, scores_gen, text_list, args.n_point, args.score_type)
    PaD = core.get_PaD(scores_real, scores_gen, text_list, args.n_point, args.score_type)

    # get results
    mean_SaD, mean_diff = utils.plot_SaD(SaD, text_list, experiment_name, args.real_dir, args.gen_dir, args.n_point, args.score_type)
    mean_PaD = utils.plot_PaD(PaD, text_list, experiment_name, args.real_dir, args.gen_dir, args.n_point, args.score_type)

    print(f"mean difference = {mean_diff}, SaD = {mean_SaD}")
    print(f"PaD = {mean_PaD}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='celebA', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--real_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/non_smiling_man_smiling_woman_2')
    parser.add_argument('--gen_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/smiling_man_non_smiling_woman_1')
    parser.add_argument('--n_attr', type=int, default=20, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')
    parser.add_argument('--score_type', type=str, default='dualCS', help='score type, e.g. HCS or dualCS')

    args = parser.parse_args()
    main(args)