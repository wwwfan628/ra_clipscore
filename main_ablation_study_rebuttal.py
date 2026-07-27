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
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import wasserstein_distance, gaussian_kde


def _kde_pdf(scores_positive, scores_negative, bins=512):
    kde1 = gaussian_kde(scores_positive)
    kde2 = gaussian_kde(scores_negative)
    lo = min(scores_positive.min(), scores_negative.min())
    hi = max(scores_positive.max(), scores_negative.max())
    xs = np.linspace(lo, hi, bins)
    p = kde1(xs)
    q = kde2(xs)

    p /= np.trapz(p, xs)
    q /= np.trapz(q, xs)
    return xs, p, q


def separation_scores(scores_positive, scores_negative, bins=512):
    # Cohen's d
    mu1, mu2 = np.mean(scores_positive), np.mean(scores_negative)
    s1, s2 = np.std(scores_positive, ddof=1), np.std(scores_negative, ddof=1)
    cohen_d = abs(mu1 - mu2) / np.sqrt((s1 ** 2 + s2 ** 2) / 2)

    # Wasserstein
    w_dist = wasserstein_distance(scores_positive, scores_negative)

    # KDE + Bhattacharyya & Hellinger
    xs, p, q = _kde_pdf(scores_positive, scores_negative, bins=bins)
    bc = np.trapz(np.sqrt(p * q), xs)              # Bhattacharyya coefficient in [0,1]
    bhattacharyya = -np.log(max(bc, 1e-15))        # distance
    hellinger = np.sqrt(max(0.0, 1.0 - bc))        # Hellinger distance in [0,1]

    # Overlap
    overlap = np.trapz(np.minimum(p, q), xs)       # Overlap area OVL in [0,1]
    separation_1_minus_ovl = 1.0 - overlap

    return dict(
        wasserstein=w_dist,
        cohen_d=cohen_d,
        bhattacharyya=bhattacharyya,
        hellinger=hellinger,
        separation_1_minus_ovl=separation_1_minus_ovl
    )



def main(args):

    start_time = time.time()
    if args.attr_type=='all':
        attrs = [
            "Arched_Eyebrows",
            "Bags_Under_Eyes",
            "Bangs",
            "Black_Hair",
            "Blond_Hair",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Gray_Hair",
            "Male",
            "Narrow_Eyes",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Sideburns",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Necklace",
            "Young",
        ]

        text_list = [
            "arched eyebrows",
            "eye bags",
            "hair bangs",
            "black hair",
            "blond hair",
            "brown hair",
            "thick eyebrows",
            "chubby face",
            "double chin",
            "gray hair",
            "male",
            "narrow eyes",
            "pale skin",
            "pointed nose",
            "receding hairline",
            "sideburns",
            "straight hair",
            "wavy hair",
            "wearing a necklace",
            "young person"
        ]
    elif args.attr_type=='global':
        attrs = [
            "Blond_Hair",
            "Brown_Hair",
            "Chubby",
            "Double_Chin",
            "Gray_Hair",
            "Male",
            "Narrow_Eyes",
            "Receding_Hairline",
            "Straight_Hair",
            "Wavy_Hair",
            "Young"
        ]

        text_list = [
            "blond hair",
            "brown hair",
            "chubby face",
            "double chin",
            "gray hair",
            "male",
            "narrow eyes",
            "receding hairline",
            "straight hair",
            "wavy hair",
            "young person",
        ]
    elif args.attr_type=='local':
        attrs = [
            "Arched_Eyebrows",
            "Bags_Under_Eyes",
            "Bushy_Eyebrows",
            "Pointy_Nose",
            "Sideburns",
            "Wearing_Necklace"
        ]

        text_list = [
            "arched eyebrows",
            "eye bags",
            "thick eyebrows",
            "pointed nose",
            "sideburns",
            "wearing a necklace",
        ]

    if (('vit' in args.score_type) and (args.mode != 'without_dual_prompts')) or (('rn' in args.score_type) and (args.mode != 'without_dual_prompts')):
        text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    elif args.score_type == 'clip_score' or args.score_type == 'hcs' or args.mode == 'without_dual_prompts':
        text_list_prompt = [f"This is a photo of {item}." for item in text_list]

    # path definition, creat directory if not exists
    save_dir = f"results_dir/ablation_study/"
    os.makedirs(save_dir, exist_ok=True)

    # calculate scores
    if 'vit' in args.score_type:
        # mode: full, without_dual_prompts, without_local_features, without_global_embeddings, without_attn_refine
        scores, scores_per_patch, labels = core.get_evalcs_local_vit_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size, mode=args.mode) # [n_img, n_attr]  [n_img, 196, n_attr]
    elif 'rn' in args.score_type:
        # mode: full, without_dual_prompts, without_local_features, without_global_embeddings
        scores, labels = core.get_evalcs_local_rn_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size, mode=args.mode)
    elif args.score_type == 'clip_score':
        scores, labels = core.get_cs_with_labels(img_dir=args.img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)
    elif args.score_type == 'hcs':
        scores, labels = core.get_hcs_with_labels(img_dir=args.img_dir, ref_img_dir=args.ref_img_dir, anno_file=args.anno_file, attrs=attrs, text_list_prompt=text_list_prompt, batch_size=args.batch_size)

    scores = scores.astype(np.float64)
    separation_by_metric = {}

    for i, attr in enumerate(attrs):
        scores_positive = scores[labels[:, i]==1][:, i]
        scores_negative = scores[labels[:, i]==-1][:, i]
        scores_furniture = scores[labels[:, i] == 0][:, i]

        # compute separation:
        separation = separation_scores(scores_positive, scores_negative)
        print(f"########## {args.score_type}--{args.mode}--{attr}  ##########")
        for k, v in separation.items():
            print(f"{k}: {v}\n")
            separation_by_metric.setdefault(k, []).append(v)

        if args.visualize:
            # =====================
            #  0.25linewidth Figure Defaults
            # =====================
            plt.rcParams.update({
                "pdf.fonttype": 42,
                "ps.fonttype": 42,

                # readable when scaled to 0.25 linewidth
                "font.size": 20,
                "axes.labelsize": 20,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 24,

                # cleaner axes
                "axes.linewidth": 1.0,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.major.size": 3.5,
                "ytick.major.size": 3.5,
            })

            # plot and save figures
            fig = plt.figure(figsize=(6,4))  # CVPR small ratio

            for x, label, color in zip(
                    [scores_negative, scores_positive],
                    [f'w/o {text_list[i]}', f'w/ {text_list[i]}'],
                    ['tab:red', 'tab:green', 'tab:blue']
            ):
                kde = gaussian_kde(x)
                x_range = np.linspace(min(x) - 0.1, max(x) + 0.1, 500)

                # =====================
                #    Plot
                # =====================
                plt.plot(x_range, kde(x_range), lw=2.2, label=label)

            # plt.xlabel(f"EvalCS_{attr}")
            # plt.ylabel("Density")

            # Light thin grid
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.35)

            # Legend outside top-left
            plt.legend(
                loc="upper left",
                frameon=True,
                facecolor="white",
                framealpha=0.35,
                edgecolor="none",
                handlelength=1.5
            )

            plt.tight_layout()

            result_save_path = f"{save_dir}/{attr}_{args.score_type}_{args.mode}.png"
            fig.savefig(result_save_path, dpi=600, bbox_inches="tight")

    print(f"########## {args.score_type}--{args.mode}--all_attributes  ##########")
    for metric, values in separation_by_metric.items():
        values = np.asarray(values, dtype=np.float64)
        print(f"{metric}: mean={np.mean(values)}, std={np.std(values)}\n")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='ablation_study', help='experiment name, to find the corresponding attribute list')
    parser.add_argument('--img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Img', help='celeba50k_furniture15k images')  # /home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Img  /Users/yifanlu/Downloads/face50k_furniture15k/Img
    parser.add_argument('--ref_img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/furniture_15k', help='celeba 50k')  # /home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img  /Users/yifanlu/Downloads/face50k_furniture15k/Img
    parser.add_argument('--anno_file', type=str, default='/home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Anno/list_attr_celeba50k_furniture15k.txt', help='celeba50k_furniture15k annotation file path')  # /home/x_yiflu/dataset/Datasets/Ablation_Study/face50k_furniture15k/Anno/list_attr_celeba50k_furniture15k.txt  /Users/yifanlu/Downloads/face50k_furniture15k/Anno/list_attr_celeba50k_furniture15k.txt
    parser.add_argument('--n_attr', type=int, default=4, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--score_type', type=str, default='EvalCS_local_vit', help='score type, e.g. hcs, clip_score, EvalCS_local_vit, EvalCS_local_rn')
    parser.add_argument('--mode', type=str, default='full', help='choose from: full, without_dual_prompts, without_local_features, without_global_embeddings, without_attn_refine')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size to extract image features')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--attr_type', type=str, default='all', help='choose from: all, global, local')

    args = parser.parse_args()
    main(args)
