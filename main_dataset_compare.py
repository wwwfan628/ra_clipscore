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

    experiment_name = f"{args.exp}_{args.n_attr}_{args.score_type}"

    text_list = utils.load_candidate(experiment_name=f"{args.exp}_{args.n_attr}", n_attr=args.n_attr)

    text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]

    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)

    # path definition, creat directory if not exists
    save_dir = f"results_dir/{experiment_name}/{dataset_real}_vs_{dataset_gen}/"
    os.makedirs(save_dir, exist_ok=True)
    results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    # calculate scores
    if 'vit' in args.score_type:
        scores_real, scores_per_patch_real = core.get_evalcs_local_vit(img_dir=args.real_dir, text_list=text_list_prompt, batch_size=args.batch_size) # [n_img, n_attr]  [n_img, 196, n_attr]
        scores_gen, scores_per_patch_gen = core.get_evalcs_local_vit(img_dir=args.gen_dir, text_list=text_list_prompt, batch_size=args.batch_size)   # [n_img, n_attr]  [n_img, 196, n_attr]
    elif 'rn' in args.score_type:
        scores_real = core.get_evalcs_local_rn(img_dir=args.real_dir, text_list=text_list_prompt, batch_size=args.batch_size)
        scores_gen = core.get_evalcs_local_rn(img_dir=args.gen_dir, text_list=text_list_prompt, batch_size=args.batch_size)

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)  # shape: n_attr
    scores_mean_gen = scores_gen.mean(axis=0)
    scores_cov_real = np.cov(scores_real.T)  # phi^2 = EX² - x_mean², shape: n_attr*n_attr
    scores_cov_gen = np.cov(scores_gen.T)

    # calculate different distribution distances
    single_attribute_results = core.compute_single(scores_real, scores_gen, scores_mean_real, scores_mean_gen,
                                                   scores_cov_real, scores_cov_gen, text_list)
    pair_attributes_results = core.compute_pair(scores_real, scores_gen, scores_mean_real, scores_mean_gen,
                                                scores_cov_real, scores_cov_gen, text_list)
    all_attributes_results = core.compute_all(scores_real, scores_gen, scores_mean_real, scores_mean_gen,
                                              scores_cov_real, scores_cov_gen, text_list)

    results_dict = {'attr_list': text_list,
                    'scores_real': scores_real, 'scores_gen': scores_gen,
                    'scores_mean_real': scores_mean_real, 'scores_mean_gen': scores_mean_gen,
                    'scores_cov_real': scores_cov_real, 'scores_cov_gen': scores_cov_gen,
                    'forward_KL_single': single_attribute_results[0], 'reverse_KL_single': single_attribute_results[1],
                    'mean_difference_single': single_attribute_results[2],
                    'f_distance_single': single_attribute_results[3],
                    'mmd_polynomial_single': single_attribute_results[4],
                    'mmd_rbf_single': single_attribute_results[5],
                    'forward_KL_pair': pair_attributes_results[0], 'reverse_KL_pair': pair_attributes_results[1],
                    'mean_difference_pair': pair_attributes_results[2], 'f_distance_pair': pair_attributes_results[3],
                    'mmd_polynomial_pair': pair_attributes_results[4],
                    'mmd_rbf_pair': pair_attributes_results[5],
                    'forward_KL_all': all_attributes_results[0], 'reverse_KL_all': all_attributes_results[1],
                    'mean_difference_all': all_attributes_results[2], 'f_distance_all': all_attributes_results[3],
                    'mmd_polynomial_all': all_attributes_results[4],
                    'mmd_rbf_all': all_attributes_results[5]}

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)

    # plot bar diagram and save figures
    core.plot_single(single_attribute_results, text_list, save_dir)
    core.plot_pair(pair_attributes_results, text_list, save_dir)

    # print out all attributes results
    print('##### Printing results of all attributes #####')
    print('Forward KL divergence: {:.3f}\nReverse KL divergence: {:.3f}\nMean difference: {:.3f}\n'
          'F-distance: {:.3f}\nKernel distance: {:.3f}\n'.format(
        all_attributes_results[0], all_attributes_results[1], all_attributes_results[2], all_attributes_results[3],
        all_attributes_results[4]))

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='compare', help='experiment name, to find the corresponding attribute list')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Dataset_Compare/A', help='the directory of real images')
    parser.add_argument('--gen_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Dataset_Compare/B', help='the directory of generated images')  #/Users/yifanlu/Downloads/Imagenet256-BigGAN/530  # /Users/yifanlu/Downloads/Imagenet256-LDM/530  # /Users/yifanlu/Downloads/imagenet256-StyleGAN-XL/530
    parser.add_argument('--n_attr', type=int, default=20, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--score_type', type=str, default='EvalCS_local_vit', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to extract image features')

    args = parser.parse_args()
    main(args)