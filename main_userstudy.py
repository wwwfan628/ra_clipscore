import argparse
import core
import utils
import os
import time
import pickle
import numpy as np

def main(args):
    start_time = time.time()

    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)
    real_dir = f"{args.real_dir}/{args.attr_index}"
    gen_dir = f"{args.gen_dir}/{args.attr_index}"

    text_list = [args.attr]

    # calculate scores
    if args.score == 'hcs':
        text_mean = core.get_text_mean(text_list)
        img_mean = core.get_img_mean(img_dir=real_dir, batch_size=1000)
        scores_real = core.get_hcs(real_dir, img_mean, text_mean, text_list, batch_size=1000)
        scores_gen = core.get_hcs(gen_dir, img_mean, text_mean, text_list, batch_size=1000)
    elif args.score == 'clip_score':
        scores_real = core.get_cs(real_dir, text_list, batch_size=1000)
        scores_gen = core.get_hcs(gen_dir, text_list, batch_size=1000)
    elif args.score == 'evalcs_local_rn':
        scores_real = core.get_evalcs_local_rn(real_dir, text_list, batch_size=1000)
        scores_gen = core.get_evalcs_local_rn(gen_dir, text_list, batch_size=1000)

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)  # shape: n_attr
    scores_mean_gen = scores_gen.mean(axis=0)
    scores_cov_real = np.cov(scores_real.T)  # phi^2 = EX² - x_mean², shape: n_attr*n_attr
    scores_cov_gen = np.cov(scores_gen.T)

    # calculate different distribution distances
    single_attribute_results = core.compute_single(scores_real, scores_gen, scores_mean_real, scores_mean_gen, scores_cov_real, scores_cov_gen, text_list)

    # save results
    save_dir = f"results_dir/{args.exp}/{args.attr_index}_{args.attr}/{dataset_real}_vs_{dataset_gen}/"
    os.makedirs(save_dir, exist_ok=True)
    results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    results_dict = {'attr_list': text_list,
                    'scores_real': scores_real, 'scores_gen': scores_gen,
                    'scores_mean_real': scores_mean_real, 'scores_mean_gen': scores_mean_gen,
                    'scores_cov_real': scores_cov_real, 'scores_cov_gen': scores_cov_gen,
                    'forward_KL_single': single_attribute_results[0], 'reverse_KL_single': single_attribute_results[1],
                    'mean_difference_single': single_attribute_results[2],
                    'f_distance_single': single_attribute_results[3],
                    'mmd_polynomial_single': single_attribute_results[4],
                    'mmd_rbf_single': single_attribute_results[5]}

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"forward_KL: {single_attribute_results[0]}, reverse_KL_single: {single_attribute_results[1]},"
          f"mean_difference_single: {single_attribute_results[2]}, f_distance_single: {single_attribute_results[3]},"
          f"mmd_polynomial_single: {single_attribute_results[4]}, mmd_rbf_single: {single_attribute_results[5]}")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='imagenet_userstudy', help='experiment name')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/imagenet256/val')
    parser.add_argument('--gen_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Imagenet256-LDM')
    parser.add_argument('--attr_index', type=int, default=965, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--attr', type=str, default='burrito', help='attribute to evaluate')
    parser.add_argument('--score', type=str, default='evalcs_local_rn', help='score type, e.g. hcs, clip_score, evalcs_local_rn')

    args = parser.parse_args()

    main(args)