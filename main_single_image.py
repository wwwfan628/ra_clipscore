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


# def plot_scores(hcs, racs, text_list, save_path):
#     print("########## plotting scores ##########")
#     x = np.arange(len(text_list))
#     bar_w = 0.35  # compact bar width for small figure
#
#     # --- Figure setup: small square ---
#     fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
#     ax2 = ax.twinx()  # secondary y-axis for RA-CLIPScore
#
#     # Convert to numpy if torch tensors
#     hcs = hcs.cpu().numpy()
#     racs = racs.cpu().numpy().squeeze()
#
#     # --- Colors (color-blind safe) ---
#     color_hcs = "#2878B5"   # deep teal–blue
#     color_racs = "#E76F51"  # warm orange–red
#
#     # --- Plot bars (side-by-side) ---
#     ax.bar(x - bar_w/2, hcs, width=bar_w, color=color_hcs,
#            edgecolor="none", label="HCS", zorder=2)
#     ax2.bar(x + bar_w/2, racs, width=bar_w, color=color_racs,
#             edgecolor="none", label="RA-CLIPScore", zorder=2)
#
#     # --- Fixed y-axis limits ---
#     ax.set_ylim(-5, 14)
#     ax2.set_ylim(-0.5, 1.4)
#
#     # --- Axis labels ---
#     ax.set_ylabel("HCS", labelpad=2)
#     ax2.set_ylabel("RA-CLIPScore", labelpad=2)
#     ax2.yaxis.set_label_position("right")
#
#     # --- X labels (attributes) ---
#     ax.set_xticks(x)
#     ax.set_xticklabels(text_list, rotation=45, ha="right")
#
#     # --- Grid and 0-line ---
#     ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
#     ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
#
#     # --- Minimal CVPR-like styling ---
#     for spine in ["top", "right"]:
#         ax.spines[spine].set_visible(False)
#     ax.spines["left"].set_linewidth(0.7)
#     ax.spines["bottom"].set_linewidth(0.7)
#     ax2.spines["right"].set_linewidth(0.7)
#     ax2.spines["top"].set_visible(False)
#
#     # --- Fonts & layout for compact figure ---
#     plt.rcParams.update({
#         "font.size": 8,
#         "axes.labelsize": 8,
#         "xtick.labelsize": 7,
#         "ytick.labelsize": 7,
#         "legend.fontsize": 7,
#     })
#
#     # --- Legend ---
#     legend_handles = [
#         Patch(facecolor=color_hcs, label="HCS"),
#         Patch(facecolor=color_racs, label="RA-CLIPScore"),
#     ]
#     ax.legend(handles=legend_handles, frameon=False, loc="lower right")
#
#     # --- Add border frame around entire chart area ---
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_linewidth(0.8)
#         spine.set_color("black")
#
#     # Ensure twin axis doesn’t hide the frame
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#
#     plt.tight_layout(pad=0.1)
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()



def plot_scores(hcs_data, racs_data, text_lists, save_path):
    experiments = list(text_lists.keys())

    # --- Setup figure (2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5.0), dpi=300, sharex=True)

    # Colors
    color_hcs = ["#1B6CA8", "#2878B5", "#6AA2D8"]
    color_racs = ["#C75336", "#E76F51", "#F29C7E"]

    bar_w = 0.35
    spacing = 1.0  # spacing between experiment blocks

    # Compute x positions & labels
    x_positions, x_labels = [], []
    pos = 0
    exp_ranges = {}
    for exp in experiments:
        attrs = text_lists[exp]
        n = len(attrs)
        x_positions.extend(pos + np.arange(n))
        x_labels.extend(attrs)
        exp_ranges[exp] = (pos, pos + n - 1)
        pos += n + spacing
    x = np.array(x_positions)

    # -------- Plot HCS (top) --------
    pos = 0
    for i, exp in enumerate(experiments):
        vals = hcs_data[exp]
        n = len(vals)
        indices = np.arange(pos, pos + n)
        ax1.bar(indices, vals, width=bar_w, color=color_hcs[i], label=exp)
        pos += n + spacing

    ax1.set_ylabel("HCS")
    ax1.set_ylim(-5, 14)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, frameon=False, loc="upper left", ncol=3, fontsize=7)

    # -------- Plot RA-CLIPScore (bottom) --------
    pos = 0
    for i, exp in enumerate(experiments):
        vals = racs_data[exp]
        n = len(vals)
        indices = np.arange(pos, pos + n)
        ax2.bar(indices, vals, width=bar_w, color=color_racs[i], label=exp)
        pos += n + spacing

    ax2.set_ylabel("RA-CLIPScore")
    ax2.set_ylim(-0.5, 1.4)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, frameon=False, loc="upper left", ncol=3, fontsize=7)

    # ----- X-axis -----
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")

    # Vertical separators between experiments
    for exp_i in range(1, len(experiments)):
        sep = exp_i * (len(text_lists[experiments[exp_i - 1]]) + spacing) - spacing / 2
        ax1.axvline(sep, color="black", linestyle=":", linewidth=0.6, alpha=0.7)
        ax2.axvline(sep, color="black", linestyle=":", linewidth=0.6, alpha=0.7)

    # Experiment labels above groups
    for exp, (start, end) in exp_ranges.items():
        mid = (start + end) / 2
        ax1.text(mid, 13.5, exp, ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Styling
    for ax in [ax1, ax2]:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def main(args):
    start_time = time.time()

    text_lists = {
    "Exp1": ["long hair", "short hair", "young", "elder", "makeup", "no makeup"],
    "Exp2": ["short hair", "young", "elder", "makeup", "no makeup"],
    "Exp3": ["long hair", "short hair", "young", "makeup", "no makeup"]
    }

    hcs_results = {}
    racs_results = {}
    results = {}

    img_mean_real = core.get_img_mean(img_dir=args.real_dir, batch_size=1024)
    for exp_name, text_list in text_lists.items():
        text_list_prompt_dual = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
        text_list_prompt_single = [f"This is a photo of {item}." for item in text_list]

        # calculate scores
        racs = core.get_evalcs_local_rn_single_image(img_path=args.img_path, text_list=text_list_prompt_dual)
        text_mean = core.get_text_mean(text_list=text_list_prompt_dual)
        hcs = core.get_hcs_for_one_img(img=args.img_path, img_mean=img_mean_real, text_mean=text_mean, text_list=text_list_prompt_single)

        hcs_results[exp_name] = hcs.cpu().numpy().squeeze()
        racs_results[exp_name] = racs.cpu().numpy().squeeze()
        results[exp_name] = {"text_list": text_list, "hcs": hcs, "racs": racs}

    # plot bar diagram and save figures
    plot_scores(hcs_results, racs_results, text_lists, args.save_path)

    with open(args.save_path, 'wb') as f:
        pickle.dump(results, f)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='single', help='experiment name')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img', help='directory for real images for HCS') # /home/x_yiflu/dataset/Datasets/CelebA/Img  /Users/yifanlu/Downloads/train_34
    parser.add_argument('--img_path', default='./man.png', type=str)
    parser.add_argument('--save_path', default='./man_eval_results.pdf', type=str)
    parser.add_argument('--n_attr', type=int, default=5, help='number of attributes to calculate SaD and PaD')

    args = parser.parse_args()
    main(args)