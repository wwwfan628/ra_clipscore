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
    y = np.arange(len(text_list))
    bar_h = 0.28  # compact spacing for small figure

    # --- Figure setup: small and square for paper plots ---
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    ax2 = ax.twiny()

    # Convert to numpy if torch tensors
    hcs = hcs.cpu().numpy()
    racs = racs.cpu().numpy().squeeze()

    # --- Compute limits ---
    xmin, xmax = np.min(hcs), np.max(hcs)
    rmin, rmax = np.min(racs), np.max(racs)

    # --- Find RA-CLIPScore value corresponding to HCS=0 ---
    racs_at_hcs0 = rmin + (0 - xmin) / (xmax - xmin) * (rmax - rmin)

    # --- Align axes so HCS=0 corresponds to racs_at_hcs0 ---
    ax.set_xlim(xmin, xmax)
    ax2.set_xlim(rmin - racs_at_hcs0, rmax - racs_at_hcs0)

    # --- Elegant two-color scheme (color-blind safe) ---
    color_hcs = "#2878B5"  # teal–blue
    color_racs = "#E76F51"  # orange–red

    # --- Plot bars ---
    ax.barh(y - bar_h / 2, hcs, height=bar_h,
            color=color_hcs, label="HCS", edgecolor="none")
    ax2.barh(y + bar_h / 2, racs - racs_at_hcs0, height=bar_h,
             color=color_racs, label="RA-CLIPScore", edgecolor="none")

    # --- X ticks ---
    ax.set_xticks([xmin, 0, xmax])
    ax2.set_xticks([rmin - racs_at_hcs0, 0, rmax - racs_at_hcs0])
    ax2.set_xticklabels([f"{rmin:.2f}", f"{racs_at_hcs0:.2f}", f"{rmax:.2f}"])

    # --- Labels ---
    ax.set_xlabel("HCS", labelpad=2)
    ax2.set_xlabel("RA-CLIPScore", labelpad=2)
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()

    # --- Y labels ---
    ax.set_yticks(y)
    ax.set_yticklabels(text_list)

    # --- CVPR minimal aesthetic tweaks ---
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        if spine == "right":
            ax2.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax2.spines["top"].set_linewidth(0.7)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    # --- Font & layout for compact figure ---
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    # --- Legend (optional, clean) ---
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_hcs, label="HCS"),
        Patch(facecolor=color_racs, label="RA-CLIPScore"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")

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
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img', help='directory for real images for HCS') # /home/x_yiflu/dataset/Datasets/CelebA/Img
    parser.add_argument('--img_path', default='./man.png', type=str)
    parser.add_argument('--save_path', default='./man_eval_results_0.pdf', type=str)
    parser.add_argument('--n_attr', type=int, default=5, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')

    args = parser.parse_args()
    main(args)