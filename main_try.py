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

    text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]

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
        text_features = core.get_text_features_eval_clip(text_list_prompt)  # [44, 512]
        with open(f'text_features_{experiment_name}.pkl', 'wb') as f:
            pickle.dump(text_features, f)

    if os.path.exists(f'img_features_{dataset_real}.pkl'):
        with open(f'img_features_{dataset_real}.pkl', 'rb') as f:
            img_features_real = pickle.load(f)
    else:
        img_features_real = core.get_img_features_eval_clip(img_dir=args.real_dir, batch_size=1000)
        with open(f'img_features_{dataset_real}.pkl', 'wb') as f:
            pickle.dump(img_features_real, f)

    if os.path.exists(f'img_features_{dataset_gen}.pkl'):
        with open(f'img_features_{dataset_gen}.pkl', 'rb') as f:
            img_features_gen = pickle.load(f)
    else:
        img_features_gen = core.get_img_features_eval_clip(img_dir=args.gen_dir, batch_size=1000)   # [5782, 512, 50]
        with open(f'img_features_{dataset_gen}.pkl', 'wb') as f:
            pickle.dump(img_features_gen, f)

    # calculate scores
    scores_real = core.get_evalcs(img_features_real, text_features).detach().cpu().numpy()
    scores_gen = core.get_evalcs(img_features_gen, text_features).detach().cpu().numpy()

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)  # shape: n_attr
    scores_mean_gen = scores_gen.mean(axis=0)
    scores_cov_real = np.cov(scores_real.T)  # phi^2 = EX² - x_mean², shape: n_attr*n_attr
    scores_cov_gen = np.cov(scores_gen.T)

    # calculate different distribution distances
    single_attribute_results = core.compute_single_reduced(scores_real, scores_gen, scores_mean_real, scores_mean_gen, scores_cov_real, scores_cov_gen, text_list)
    pair_attributes_results = core.compute_pair_reduced(scores_real, scores_gen, scores_mean_real, scores_mean_gen, scores_cov_real, scores_cov_gen, text_list)
    all_attributes_results = core.compute_all(scores_real, scores_gen, scores_mean_real, scores_mean_gen, scores_cov_real, scores_cov_gen, text_list)

    results_dict = {'attr_list': text_list,
                    'scores_real': scores_real, 'scores_gen': scores_gen,
                    'scores_mean_real': scores_mean_real, 'scores_mean_gen': scores_mean_gen,
                    'scores_cov_real': scores_cov_real, 'scores_cov_gen': scores_cov_gen,
                    'forward_KL_single': single_attribute_results[0], 'reverse_KL_single': single_attribute_results[1],
                    'mean_difference_single': single_attribute_results[2],
                    'f_distance_single': single_attribute_results[3],
                    'kernel_distance_single': single_attribute_results[4],
                    'forward_KL_pair': pair_attributes_results[0], 'reverse_KL_pair': pair_attributes_results[1],
                    'mean_difference_pair': pair_attributes_results[2], 'f_distance_pair': pair_attributes_results[3],
                    'kernel_distance_pair': pair_attributes_results[4],
                    'forward_KL_all': all_attributes_results[0], 'reverse_KL_all': all_attributes_results[1],
                    'mean_difference_all': all_attributes_results[2], 'f_distance_all': all_attributes_results[3],
                    'kernel_distance_all': all_attributes_results[4]}

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)

    # plot bar diagram and save figures
    core.plot_single_reduced(single_attribute_results, text_list, save_dir)
    core.plot_pair_reduced(pair_attributes_results, text_list, save_dir)

    # print out all attributes results
    print('##### Printing results of all attributes #####')
    print('Forward KL divergence: {:.3f}\nReverse KL divergence: {:.3f}\nMean difference: {:.3f}\n'
          'F-distance: {:.3f}\nKernel distance: {:.3f}\n'.format(
        all_attributes_results[0], all_attributes_results[1], all_attributes_results[2], all_attributes_results[3],
        all_attributes_results[4]))

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='celebA', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--real_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/CelebA_5K')
    parser.add_argument('--gen_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/CelebA_5k_Eyeglasses_Injection_800')
    parser.add_argument('--n_attr', type=int, default=22, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')
    parser.add_argument('--score_type', type=str, default='EvalCS', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')

    args = parser.parse_args()
    main(args)