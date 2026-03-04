import argparse
import core
import utils
import os
import time
import pickle
import numpy as np
import scipy



def compute_KL_divergence_1d_gaussian_pdf(p_mean, q_mean, p_var, q_var):
    # KL(p,q) = log(phi_q/phi_p) + ((phi_p² + (mean_p - mean_q)²)/2phi_q²) - 0.5   p represents original, q generated
    KL_div = np.log(np.sqrt(q_var) / np.sqrt(p_var)) + (
            (p_var + np.square((p_mean - q_mean))) / (2 * q_var)) - 0.5
    return float(KL_div)


def compute_frechet_distance(real_mean, gen_mean, real_cov, gen_cov):
    m = np.square(gen_mean - real_mean).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(gen_cov, real_cov), disp=False)  # pylint: disable=no-member
    frechet_distance = np.real(m + np.trace(gen_cov + real_cov - s * 2))
    return float(frechet_distance)



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
        scores_real = core.get_hcs(real_dir, img_mean, text_mean, text_list, batch_size=1000).detach().cpu().numpy()
        scores_gen = core.get_hcs(gen_dir, img_mean, text_mean, text_list, batch_size=1000).detach().cpu().numpy()
    elif args.score == 'clip_score':
        scores_real = core.get_cs(real_dir, text_list, batch_size=1000).detach().cpu().numpy()
        scores_gen = core.get_hcs(gen_dir, text_list, batch_size=1000).detach().cpu().numpy()
    elif args.score == 'evalcs_local_rn':
        text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
        scores_real = core.get_evalcs_local_rn(real_dir, text_list_prompt, batch_size=1000)
        scores_gen = core.get_evalcs_local_rn(gen_dir, text_list_prompt, batch_size=1000)

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)  # shape: n_attr
    scores_mean_gen = scores_gen.mean(axis=0)
    scores_cov_real = np.asarray(np.cov(scores_real.T))  # phi^2 = EX² - x_mean², shape: n_attr*n_attr
    scores_cov_gen = np.asarray(np.cov(scores_gen.T))
    if scores_cov_real.ndim == 0:
       scores_cov_real = scores_cov_real.reshape(1, 1)
    if scores_cov_gen.ndim == 0:
       scores_cov_gen = scores_cov_gen.reshape(1, 1)

    # calculate different distribution distances
    forward_KL = compute_KL_divergence_1d_gaussian_pdf(scores_mean_real, scores_mean_gen, scores_cov_real, scores_cov_gen)
    reverse_KL = compute_KL_divergence_1d_gaussian_pdf(scores_mean_gen, scores_mean_real, scores_cov_gen, scores_cov_real)
    mean_diff = np.abs(scores_mean_real - scores_mean_gen)
    f_dist = compute_frechet_distance(scores_mean_real, scores_mean_gen, scores_cov_real, scores_cov_gen)

    # save results
    save_dir = f"results_dir/{args.exp}/{args.attr_index}_{args.attr}/{dataset_real}_vs_{dataset_gen}/"
    os.makedirs(save_dir, exist_ok=True)
    results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    results_dict = {'attr_list': text_list,
                    'scores_real': scores_real, 'scores_gen': scores_gen,
                    'scores_mean_real': scores_mean_real, 'scores_mean_gen': scores_mean_gen,
                    'scores_cov_real': scores_cov_real, 'scores_cov_gen': scores_cov_gen,
                    'forward_KL': forward_KL, 'reverse_KL': reverse_KL,
                    'mean_diff': mean_diff,
                    'f_dist': f_dist
                   }

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"forward_KL: {forward_KL}, reverse_KL: {reverse_KL}, mean_diff: {mean_diff}, f_dist: {f_dist}.")

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