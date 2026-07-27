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

    text_features = core.get_text_features_eval_clip_local_vit(text_list_prompt)  # [44, 512]

    # if os.path.exists(f'{save_dir}/img_features/{dataset_real}/{class_id}.pkl') and os.path.exists(f'{save_dir}/attn_weight/{dataset_real}/{class_id}.pkl'):
    #     with open(f'{save_dir}/img_features/{dataset_real}/{class_id}.pkl', 'rb') as f:
    #         img_features_real = pickle.load(f)
    #     with open(f'{save_dir}/attn_weight/{dataset_real}/{class_id}.pkl', 'rb') as f:
    #         attn_weight_list_real = pickle.load(f)
    # else:
    #     img_features_real, attn_weight_list_real = core.get_img_features_eval_clip_local_vit(img_dir=dataset_real_path, batch_size=1000)
    #     with open(f'{save_dir}/img_features/{dataset_real}/{class_id}.pkl', 'wb') as f:
    #         pickle.dump(img_features_real, f)
    #     with open(f'{save_dir}/attn_weight/{dataset_real}/{class_id}.pkl', 'wb') as f:
    #         pickle.dump(attn_weight_list_real, f)
    #
    # if os.path.exists(f'{save_dir}/img_features/{dataset_gen}/{class_id}.pkl') and os.path.exists(f'{save_dir}/attn_weight/{dataset_gen}/{class_id}.pkl'):
    #     with open(f'{save_dir}/img_features/{dataset_gen}/{class_id}.pkl', 'rb') as f:
    #         img_features_gen = pickle.load(f)
    #     with open(f'{save_dir}/attn_weight/{dataset_gen}/{class_id}.pkl', 'rb') as f:
    #         attn_weight_list_gen = pickle.load(f)
    # else:
    #     img_features_gen, attn_weight_list_gen = core.get_img_features_eval_clip_local_vit(img_dir=dataset_gen_path, batch_size=1000)
    #     with open(f'{save_dir}/img_features/{dataset_gen}/{class_id}.pkl', 'wb') as f:
    #         pickle.dump(img_features_gen, f)
    #     with open(f'{save_dir}/attn_weight/{dataset_gen}/{class_id}.pkl', 'wb') as f:
    #         pickle.dump(attn_weight_list_gen, f)

    # calculate scores
    _, scores_real = core.get_evalcs_local_vit(dataset_real_path, text_list_prompt) # [n_batch, 196, n_attr]
    _, scores_gen = core.get_evalcs_local_vit(dataset_gen_path, text_list_prompt)  # [n_batch, 196, n_attr]

    # compute empirical mean and empirical covariance matrix, then convert GPU tensor to CPU numpy array
    scores_mean_real = scores_real.mean(axis=0)  # [196, n_attr]
    scores_mean_gen = scores_gen.mean(axis=0)  # [196, n_attr]
    scores_var_real = scores_real.var(axis=0)
    scores_var_gen = scores_gen.var(axis=0)

    mean_diff = (scores_mean_real - scores_mean_gen)[:, 0]  # [196, n_attr]
    f_distance = (np.square(scores_mean_real - scores_mean_gen) + scores_var_real + scores_var_gen - 2 * np.sqrt(scores_var_real * scores_var_gen))[:, 0]
    kl_divergence = (np.log(np.sqrt(scores_var_gen) / np.sqrt(scores_var_real)) + ((scores_var_real + np.square((scores_mean_real - scores_mean_gen))) / (2 * scores_var_gen)) - 0.5)[:, 0]

    return mean_diff, f_distance, kl_divergence


def main(args):
    start_time = time.time()
    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)
    # os.makedirs(f'results_dir/{args.exp}/img_features/{dataset_real}', exist_ok=True)
    # os.makedirs(f'results_dir/{args.exp}/attn_weight/{dataset_real}', exist_ok=True)
    # os.makedirs(f'results_dir/{args.exp}/img_features/{dataset_gen}', exist_ok=True)
    # os.makedirs(f'results_dir/{args.exp}/attn_weight/{dataset_gen}', exist_ok=True)

    # path definition, creat directory if not exists
    mean_diff_results = []
    f_distance_results = []
    kl_divergence_results = []

    mean_diff_max_results = []
    f_distance_max_results = []
    kl_divergence_max_results = []

    mean_diff_mean_results = []
    f_distance_mean_results = []
    kl_divergence_mean_results = []

    mean_diff_results_norm = []
    f_distance_results_norm = []
    kl_divergence_results_norm = []

    with open('imagenet-simple-labels.json') as f:
        labels = json.load(f)

    for class_id in range(1000):
        start_time_i = time.time()
        class_label = labels[class_id]
        mean_diff, f_distance, kl_divergence = compute_single_attr(class_id, class_label, args.real_dir, args.gen_dir, f'results_dir/{args.exp}')

        mean_diff_results.append(mean_diff)
        mean_diff_max_results.append(mean_diff.max())
        mean_diff_mean_results.append(mean_diff.mean())
        mean_diff_norm = (mean_diff - mean_diff.min()) / (mean_diff.max() - mean_diff.min() + 1e-8)
        mean_diff_results_norm.append(mean_diff_norm)

        f_distance_results.append(f_distance)
        f_distance_max_results.append(f_distance.max())
        f_distance_mean_results.append(f_distance.mean())
        f_distance_norm = (f_distance - f_distance.min()) / (f_distance.max() - f_distance.min())
        f_distance_results_norm.append(f_distance_norm)

        kl_divergence_results.append(kl_divergence)
        kl_divergence_max_results.append(kl_divergence.max())
        kl_divergence_mean_results.append(kl_divergence.mean())
        kl_divergence_norm = (kl_divergence - kl_divergence.min()) / (kl_divergence.max() - kl_divergence.min())
        kl_divergence_results_norm.append(kl_divergence_norm)

        if args.visualize == True:
            # Visualization
            os.makedirs(f'results_dir/{args.exp}/{dataset_real}_vs_{dataset_gen}/{class_id}', exist_ok=True)
            score_name = 'dual prompts: first refine with attention, then softmax'

            heatmap = (mean_diff_norm.reshape(14, 14))
            plt.imshow(heatmap, cmap="rainbow")
            plt.colorbar()
            plt.title(f"Heatmap of Mean Difference \n {score_name}")
            plt.savefig(f'results_dir/{args.exp}/{dataset_real}_vs_{dataset_gen}/{class_id}/mean_diff.pdf')
            plt.close()

            heatmap = (f_distance_norm.reshape(14, 14))
            plt.imshow(heatmap, cmap="rainbow")
            plt.colorbar()
            plt.title(f"Heatmap of F-Distance \n {score_name}")
            plt.savefig(f'results_dir/{args.exp}/{dataset_real}_vs_{dataset_gen}/{class_id}/f-distance.pdf')
            plt.close()

            heatmap = (kl_divergence_norm.reshape(14, 14))
            plt.imshow(heatmap, cmap="rainbow")
            plt.colorbar()
            plt.title(f"Heatmap of KL-Divergence \n {score_name}")
            plt.savefig(f'results_dir/{args.exp}/{dataset_real}_vs_{dataset_gen}/{class_id}/kl-divergence.pdf')
            plt.close()

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_i))
        print(f'Excution Time for Class {class_id} ({class_label}): {elapsed}!')

    print(
        f'mean_diff_mean_results: mean={np.mean(mean_diff_mean_results):.6f}, '
        f'std={np.std(mean_diff_mean_results):.6f}'
    )

    results_dict = {'mean_diff_results': mean_diff_results, 'f_distance_results': f_distance_results,
                 'kl_divergence_results': kl_divergence_results, 'mean_diff_max_results': mean_diff_max_results,
                 'f_distance_max_results': f_distance_max_results, 'kl_divergence_max_results': kl_divergence_max_results,
                 'mean_diff_mean_results': mean_diff_mean_results, 'f_distance_mean_results': f_distance_mean_results,
                 'kl_divergence_mean_results': kl_divergence_mean_results, 'mean_diff_results_norm': mean_diff_results_norm,
                 'f_distance_results_norm': f_distance_results_norm, 'kl_divergence_results_norm': kl_divergence_results_norm}

    with open(f'results_dir/{args.exp}/{dataset_real}_vs_{dataset_gen}/results_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)

    max_10_labels_mean_diff = sorted(range(1000), key=lambda i: mean_diff_max_results[i], reverse=True)[:10]
    print("According to mean_diff (max):")
    for i in max_10_labels_mean_diff:
        print(f"  index={i}, label={labels[i]}")

    mean_10_labels_mean_diff = sorted(range(1000), key=lambda i: mean_diff_mean_results[i], reverse=True)[:10]
    print("According to mean_diff (mean):")
    for i in mean_10_labels_mean_diff:
        print(f"  index={i}, label={labels[i]}")

    max_10_labels_f_distance = sorted(range(1000), key=lambda i: f_distance_max_results[i], reverse=True)[:10]
    print("According to f-distance (max):")
    for i in max_10_labels_f_distance:
        print(f"  index={i}, label={labels[i]}")

    mean_10_labels_f_distance = sorted(range(1000), key=lambda i: f_distance_mean_results[i], reverse=True)[:10]
    print("According to f-distance (mean):")
    for i in mean_10_labels_f_distance:
        print(f"  index={i}, label={labels[i]}")

    max_10_labels_kl_divergence = sorted(range(1000), key=lambda i: kl_divergence_max_results[i], reverse=True)[:10]
    print("According to kl-divergence (max):")
    for i in max_10_labels_kl_divergence:
        print(f"  index={i}, label={labels[i]}")

    mean_10_labels_kl_divergence = sorted(range(1000), key=lambda i: kl_divergence_mean_results[i], reverse=True)[:10]
    print("According to kl-divergence (mean):")
    for i in mean_10_labels_kl_divergence:
        print(f"  index={i}, label={labels[i]}")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='imagenet_exp', help='experiment name')
    parser.add_argument('--real_dir', type=str, default='/Users/yifanlu/Downloads/imagenet256',)
    parser.add_argument('--gen_dir', type=str, default='/Users/yifanlu/Downloads/Imagenet256-LDM')  #/Users/yifanlu/Downloads/Imagenet256-BigGAN/530  # /Users/yifanlu/Downloads/Imagenet256-LDM/530  # /Users/yifanlu/Downloads/imagenet256-StyleGAN-XL/530
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()
    main(args)
