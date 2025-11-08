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
from matplotlib.patches import Patch


def plot_scores(hcs, racs, text_list, save_path):
    print("########## plotting scores ##########")
    x = np.arange(len(text_list))
    bar_w = 0.35  # compact bar width for small figure

    # --- Figure setup: small square ---
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    ax2 = ax.twinx()  # second y-axis for RA-CLIPScore

    # Convert to numpy if torch tensors
    hcs = hcs.cpu().numpy()
    racs = racs.cpu().numpy().squeeze()

    # --- Compute limits ---
    ymin_hcs, ymax_hcs = np.min(hcs), np.max(hcs)
    ymin_racs, ymax_racs = np.min(racs), np.max(racs)

    # --- Elegant, color-blind-safe colors ---
    color_hcs = "#2878B5"   # deep teal–blue
    color_racs = "#E76F51"  # warm orange–red

    # --- Plot bars (side-by-side) ---
    ax.bar(x - bar_w/2, hcs, width=bar_w, color=color_hcs,
           edgecolor="none", label="HCS", zorder=2)
    ax2.bar(x + bar_w/2, racs, width=bar_w, color=color_racs,
            edgecolor="none", label="RA-CLIPScore", zorder=2)

    # --- Center both y-axes around 0 ---
    abs_max_hcs = max(abs(ymin_hcs), abs(ymax_hcs))
    abs_max_hcs *= 1.05
    abs_max_racs = max(abs(ymin_racs), abs(ymax_racs))
    abs_max_racs *= 1.05
    ax.set_ylim(-abs_max_hcs, abs_max_hcs)
    ax2.set_ylim(-abs_max_racs, abs_max_racs)

    # --- Axis labels ---
    ax.set_ylabel("HCS", labelpad=2)
    ax2.set_ylabel("RA-CLIPScore", labelpad=2)
    ax2.yaxis.set_label_position("right")

    # --- X labels (attributes) ---
    ax.set_xticks(x)
    ax.set_xticklabels(text_list, rotation=45, ha="right")

    # --- Grid and 0 line ---
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)

    # --- Minimal CVPR-like styling ---
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax2.spines["right"].set_linewidth(0.7)
    ax2.spines["top"].set_visible(False)

    # --- Font & layout for compact figure ---
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    # --- Legend ---
    legend_handles = [
        Patch(facecolor=color_hcs, label="HCS"),
        Patch(facecolor=color_racs, label="RA-CLIPScore"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    # --- Add border frame around entire chart area ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Make sure twin axis (ax2) doesn’t hide the border
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def main(args):
    start_time = time.time()

    text_list = utils.load_candidate(experiment_name=f"{args.exp}_{args.n_attr}", n_attr=args.n_attr)

    text_list_prompt_dual = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    text_list_prompt_single = [f"This is a photo of {item}." for item in text_list]

    # calculate scores
    racs = core.get_evalcs_local_rn_single_image(img_path=args.img_path, text_list=text_list_prompt_dual)
    img_mean_real = core.get_img_mean(img_dir=args.real_dir, batch_size=1000)
    text_mean = core.get_text_mean(text_list=text_list_prompt_dual)
    hcs = core.get_hcs_for_one_img(img=args.img_path, img_mean=img_mean_real, text_mean=text_mean, text_list=text_list_prompt_single)


    results_dict = {'attr_list': text_list, 'racs': racs, 'hcs': hcs}

    with open(args.save_path, 'wb') as f:
        pickle.dump(results_dict, f)

    # plot bar diagram and save figures
    plot_scores(hcs, racs, text_list, args.save_path)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='single', help='experiment name')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img', help='directory for real images for HCS') # /home/x_yiflu/dataset/Datasets/CelebA/Img  /Users/yifanlu/Downloads/train_34
    parser.add_argument('--img_path', default='./man.png', type=str)
    parser.add_argument('--save_path', default='./man_eval_results_2.pdf', type=str)
    parser.add_argument('--n_attr', type=int, default=5, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')

    args = parser.parse_args()
    main(args)