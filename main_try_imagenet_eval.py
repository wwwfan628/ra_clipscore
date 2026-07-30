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
import json

def compute_single_attr(class_id, class_label, real_dir, gen_dir, save_dir):
    text_list_prompt = [f"This is a photo without {class_label}."] + [f"This is a photo of {class_label}."]
    dataset_real = os.path.basename(real_dir)
    dataset_gen = os.path.basename(gen_dir)
    dataset_real_path = os.path.join(real_dir, 'train', str(class_id))
    dataset_gen_path = os.path.join(gen_dir, str(class_id))


    # calculate scores
    scores_real, scores_per_patch_real = core.get_evalcs_local_vit(dataset_real_path, text_list_prompt) # [n_batch, n_attr], [n_batch, 196, n_attr]
    scores_gen, scores_per_patch_gen = core.get_evalcs_local_vit(dataset_gen_path, text_list_prompt)  # [n_batch, n_attr], [n_batch, 196, n_attr]

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)  # [n_attr]
    scores_mean_gen = scores_gen.mean(axis=0)  # [n_attr]
    scores_var_real = scores_real.var(axis=0)
    scores_var_gen = scores_gen.var(axis=0)

    # mean_diff = (scores_mean_real - scores_mean_gen)[:, 0]  # [196, n_attr]
    # f_distance = (np.square(scores_mean_real - scores_mean_gen) + scores_var_real + scores_var_gen - 2 * np.sqrt(scores_var_real * scores_var_gen))[:, 0]
    # kl_divergence = (np.log(np.sqrt(scores_var_gen) / np.sqrt(scores_var_real)) + ((scores_var_real + np.square((scores_mean_real - scores_mean_gen))) / (2 * scores_var_gen)) - 0.5)[:, 0]
    # calculate different distribution distances
    forward_KL_single, reverse_KL_single, mean_difference_single, f_distance_single, mmd_polynomial_single, mmd_rbf_single = core.compute_single(scores_real, scores_gen, scores_mean_real, scores_mean_gen,
                                                   scores_var_real, scores_var_gen, [class_label])

    return forward_KL_single, reverse_KL_single, mean_difference_single, f_distance_single, mmd_polynomial_single, mmd_rbf_single


def main(args):
    start_time = time.time()
    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)

    # path definition, creat directory if not exists
    forward_KL_single_results = []
    reverse_KL_single_results = []
    mean_difference_single_results = []
    f_distance_single_results = []
    mmd_polynomial_single_results = []
    mmd_rbf_single_results = []

    with open('imagenet-simple-labels.json') as f:
        labels = json.load(f)

    for class_id in range(1000):
        start_time_i = time.time()
        class_label = labels[class_id]
        forward_KL_single, reverse_KL_single, mean_difference_single, f_distance_single, mmd_polynomial_single, mmd_rbf_single = compute_single_attr(class_id, class_label, args.real_dir, args.gen_dir, f'results_dir/{args.exp}')

        forward_KL_single_results.append(forward_KL_single)
        reverse_KL_single_results.append(reverse_KL_single)
        mean_difference_single_results.append(mean_difference_single)
        f_distance_single_results.append(f_distance_single)
        mmd_polynomial_single_results.append(mmd_polynomial_single)
        mmd_rbf_single_results.append(mmd_rbf_single)

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_i))
        print(f'Excution Time for Class {class_id} ({class_label}): {elapsed}!')
        print(f'forward_KL_single: {forward_KL_single}')

    print(
        f'{dataset_real}_vs_{dataset_gen}: '
        f'forward_KL_single_results: mean={np.mean(forward_KL_single_results):.6f}, '
        f'std={np.std(forward_KL_single_results):.6f}'
    )

    os.makedirs(f'results_dir/{args.exp}/{dataset_real}_vs_{dataset_gen}', exist_ok=True)
    results_dict = {'forward_KL_single_results': forward_KL_single_results,
                    'reverse_KL_single_results': reverse_KL_single_results,
                    'mean_difference_single_results': mean_difference_single_results,
                    'f_distance_single_results': f_distance_single_results,
                    'mmd_polynomial_single_results': mmd_polynomial_single_results,
                    'mmd_rbf_single_results': mmd_rbf_single_results}

    with open(f'results_dir/{args.exp}/{dataset_real}_vs_{dataset_gen}/results_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

    max_10_labels_forward_KL_single = sorted(range(1000), key=lambda i: forward_KL_single_results[i], reverse=True)[:10]
    print("According to forward KL single (max):")
    for i in max_10_labels_forward_KL_single:
        print(f"  index={i}, label={labels[i]}")

    max_10_labels_reverse_KL_single = sorted(range(1000), key=lambda i: reverse_KL_single_results[i], reverse=True)[:10]
    print("According to reverse KL single (max):")
    for i in max_10_labels_reverse_KL_single:
        print(f"  index={i}, label={labels[i]}")

    max_10_labels_f_distance_single = sorted(range(1000), key=lambda i: f_distance_single_results[i], reverse=True)[:10]
    print("According to f-distance (max):")
    for i in max_10_labels_f_distance_single:
        print(f"  index={i}, label={labels[i]}")

    max_10_labels_mean_difference_single = sorted(range(1000), key=lambda i: np.abs(mean_difference_single_results)[i], reverse=True)[:10]
    print("According to mean difference (max):")
    for i in max_10_labels_mean_difference_single:
        print(f"  index={i}, label={labels[i]}")

    max_10_labels_mmd_polynomial_single = sorted(range(1000), key=lambda i: mmd_polynomial_single_results[i], reverse=True)[:10]
    print("According to MMD polynomial single (max):")
    for i in max_10_labels_mmd_polynomial_single:
        print(f"  index={i}, label={labels[i]}")

    max_10_labels_mmd_rbf_single = sorted(range(1000), key=lambda i: mmd_rbf_single_results[i], reverse=True)[:10]
    print("According to MMD rbf single (max):")
    for i in max_10_labels_mmd_rbf_single:
        print(f"  index={i}, label={labels[i]}")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='imagenet_eval_exp', help='experiment name')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/imagenet256/train',)
    parser.add_argument('--gen_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Imagenet256-LDM')  #/Users/yifanlu/Downloads/Imagenet256-BigGAN/530  # /Users/yifanlu/Downloads/Imagenet256-LDM/530  # /Users/yifanlu/Downloads/imagenet256-StyleGAN-XL/530

    args = parser.parse_args()
    main(args)