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
    # text_list_prompt = [f"This is a photo of {item}." for item in text_list]

    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)

    # path definition, creat directory if not exists
    save_dir = f"results_dir/{experiment_name}/{dataset_real}_vs_{dataset_gen}/"
    os.makedirs(save_dir, exist_ok=True)
    results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    # # calculate text, image mean
    # if os.path.exists(f'text_features_{experiment_name}.pkl'):
    #     with open(f'text_features_{experiment_name}.pkl', 'rb') as f:
    #         text_features = pickle.load(f)
    # else:
    #     text_features = core.get_text_features_eval_clip_local_vit(text_list_prompt)  # [44, 512]
    #     with open(f'text_features_{experiment_name}.pkl', 'wb') as f:
    #         pickle.dump(text_features, f)
    #
    # if os.path.exists(f'img_features_{dataset_real}_{args.score_type}.pkl') and os.path.exists(f'attn_weight_{dataset_real}_{args.score_type}.pkl'):
    #     with open(f'img_features_{dataset_real}_{args.score_type}.pkl', 'rb') as f:
    #         img_features_real = pickle.load(f)
    #     with open(f'attn_weight_{dataset_real}_{args.score_type}.pkl', 'rb') as f:
    #         attn_weight_list_real = pickle.load(f)
    # else:
    #     img_features_real, attn_weight_list_real = core.get_img_features_eval_clip_local_vit(img_dir=args.real_dir, batch_size=1000)
    #     with open(f'img_features_{dataset_real}_{args.score_type}.pkl', 'wb') as f:
    #         pickle.dump(img_features_real, f)
    #     with open(f'attn_weight_{dataset_real}_{args.score_type}.pkl', 'wb') as f:
    #         pickle.dump(attn_weight_list_real, f)
    #
    # if os.path.exists(f'img_features_{dataset_gen}_{args.score_type}.pkl') and os.path.exists(f'attn_weight_{dataset_gen}_{args.score_type}.pkl'):
    #     with open(f'img_features_{dataset_gen}_{args.score_type}.pkl', 'rb') as f:
    #         img_features_gen = pickle.load(f)
    #     with open(f'attn_weight_{dataset_gen}_{args.score_type}.pkl', 'rb') as f:
    #         attn_weight_list_gen = pickle.load(f)
    # else:
    #     img_features_gen, attn_weight_list_gen = core.get_img_features_eval_clip_local_vit(img_dir=args.gen_dir, batch_size=1000)   # [5782, 512, 50]
    #     with open(f'img_features_{dataset_gen}_{args.score_type}.pkl', 'wb') as f:
    #         pickle.dump(img_features_gen, f)
    #     with open(f'attn_weight_{dataset_gen}_{args.score_type}.pkl', 'wb') as f:
    #         pickle.dump(attn_weight_list_gen, f)

    text_features = core.get_text_features_eval_clip_local_vit(text_list_prompt)  # [44, 512]
    img_features_real, attn_weight_list_real = core.get_img_features_eval_clip_local_vit(img_dir=args.real_dir, batch_size=1000)
    img_features_gen, attn_weight_list_gen = core.get_img_features_eval_clip_local_vit(img_dir=args.gen_dir, batch_size=1000)

    # calculate scores
    scores_real = core.get_evalcs_local_vit(img_features_real, text_features, attn_weight_list_real).detach().cpu().numpy() # [n_batch, 197, n_attr]
    scores_gen = core.get_evalcs_local_vit(img_features_gen, text_features, attn_weight_list_gen).detach().cpu().numpy()    # [n_batch, 197, n_attr]

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)[1:]  # [197, n_attr]
    scores_mean_gen = scores_gen.mean(axis=0)[1:]   # [197, n_attr]
    scores_var_real = scores_real.var(axis=0)[1:]
    scores_var_gen = scores_gen.var(axis=0)[1:]

    mean_diff = (scores_mean_real - scores_mean_gen)[:, 0]    # [196, n_attr]
    mean_diff = (mean_diff - mean_diff.min()) / (mean_diff.max() - mean_diff.min() + 1e-8)
    f_distance = (np.square(scores_mean_real - scores_mean_gen) + scores_var_real + scores_var_gen - 2 * np.sqrt(scores_var_real*scores_var_gen))[:, 0]
    f_distance = (f_distance - f_distance.min()) / (f_distance.max() - f_distance.min())
    kl_divergence = compute_KL_divergence_1d_gaussian_pdf(scores_mean_real, scores_mean_gen, scores_var_real, scores_var_gen)[:, 0]
    kl_divergence = (kl_divergence - kl_divergence.min()) / (kl_divergence.max() - kl_divergence.min())

    # Visualization
    score_name = 'dual prompts: first refine with attention, then softmax'

    heatmap = (mean_diff.reshape(14, 14))
    plt.imshow(heatmap, cmap="rainbow")
    plt.colorbar()
    plt.title(f"Heatmap of Mean Difference \n {score_name}")
    plt.show()

    heatmap = (f_distance.reshape(14, 14))
    plt.imshow(heatmap, cmap="rainbow")
    plt.colorbar()
    plt.title(f"Heatmap of F-Distance \n {score_name}")
    plt.show()

    heatmap = (kl_divergence.reshape(14, 14))
    plt.imshow(heatmap, cmap="rainbow")
    plt.colorbar()
    plt.title(f"Heatmap of KL-Divergence \n {score_name}")
    plt.show()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
    print(f'Excution Time: {elapsed}!')


def compute_KL_divergence_1d_gaussian_pdf(p_mean, q_mean, p_var, q_var):
    # KL(p,q) = log(phi_q/phi_p) + ((phi_p² + (mean_p - mean_q)²)/2phi_q²) - 0.5   p represents original, q generated
    KL_div = np.log(np.sqrt(q_var) / np.sqrt(p_var)) + ((p_var + np.square((p_mean - q_mean))) / (2 * q_var)) - 0.5
    return KL_div


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='imagenet', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--real_dir', type=str, default='/Users/yifanlu/Downloads/imagenet256/val/430',)
    parser.add_argument('--gen_dir', type=str, default='/Users/yifanlu/Downloads/imagenet256-StyleGAN-XL/430')  #/Users/yifanlu/Downloads/Imagenet256-BigGAN/530  # /Users/yifanlu/Downloads/Imagenet256-LDM/530  # /Users/yifanlu/Downloads/imagenet256-StyleGAN-XL/530
    parser.add_argument('--n_attr', type=int, default=1, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')
    parser.add_argument('--score_type', type=str, default='EvalCS', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')

    args = parser.parse_args()
    main(args)