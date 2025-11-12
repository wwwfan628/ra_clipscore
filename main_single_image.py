import argparse
import os.path

import core
import utils
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import time
from matplotlib.patches import Patch


mpl.rcParams.update({
    "pdf.fonttype": 42,       # Embed TrueType fonts
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

# --- Align each experiment’s values to master attribute list ---
def pad_to_master(master_attrs, exp_attrs, values):
    idx = {a: i for i, a in enumerate(exp_attrs)}
    out = np.zeros(len(master_attrs), dtype=float)
    for j, a in enumerate(master_attrs):
        if a in idx:
            out[j] = float(values[idx[a]])
    return out


def plot_two_histograms_over_three_sets(
    hcs_vals, racs_vals, master_attrs, per_exp_attrs, save_path
):
    """
    Two subplots:
        - Top: HCS (blue)
        - Bottom: RA-CLIPScore (orange)
    Each shows three experiments aligned by master attribute list.
    Missing attributes are filled with 0.
    """

    # --- Align data (fill missing with zeros) ---
    H = [pad_to_master(master_attrs, per_exp_attrs[i], hcs_vals[i]) for i in range(3)]
    R = [pad_to_master(master_attrs, per_exp_attrs[i], racs_vals[i]) for i in range(3)]

    x = np.arange(len(master_attrs))
    bar_w = 0.22
    offsets = [-bar_w, 0.0, bar_w]

    # --- Colors ---
    blue_shades   = ["#1B6CA8", "#2878B5", "#6AA2D8"]  # HCS
    orange_shades = ["#C75336", "#E76F51", "#F29C7E"]  # RA-CLIPScore

    # --- Experiment names ---
    exp_labels = ["All Attr.", "w/o long hair", "w/o elder"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.4), dpi=300, sharex=True)

    # --- HCS (blue, top) ---
    for i in range(3):
        ax1.bar(x + offsets[i], H[i], width=bar_w, color=blue_shades[i],
                edgecolor="none", label=exp_labels[i], zorder=2)
    ax1.set_ylabel("HCS")
    ax1.set_ylim(-5, 14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(master_attrs, rotation=35, ha="right")  # ticks also on top
    ax1.axhline(0, color="gray", lw=0.6, ls="--", zorder=1)
    ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.6, zorder=0)
    ax1.legend(frameon=False, loc="upper left", ncol=3,
               handlelength=1.2, columnspacing=0.8)

    # --- RA-CLIPScore (orange, bottom) ---
    for i in range(3):
        ax2.bar(x + offsets[i], R[i], width=bar_w, color=orange_shades[i],
                edgecolor="none", label=exp_labels[i], zorder=2)
    ax2.set_ylabel("RA-CLIPScore")
    ax2.set_ylim(-0.5, 1.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(master_attrs, rotation=35, ha="right")
    ax2.axhline(0, color="gray", lw=0.6, ls="--", zorder=1)
    ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.6, zorder=0)
    ax2.legend(frameon=False, loc="upper left", ncol=3,
               handlelength=1.2, columnspacing=0.8)

    # --- Styling ---
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    # --- Save combined figure ---
    plt.tight_layout(pad=0.25)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")

    # HCS only
    fig_h, ax_h = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    for i in range(3):
        ax_h.bar(x + offsets[i], H[i], width=bar_w, color=blue_shades[i],
                 edgecolor="none", label=exp_labels[i], zorder=2)
    ax_h.set_ylabel("HCS")
    ax_h.set_ylim(-5, 14)
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(master_attrs, rotation=35, ha="right")
    ax_h.axhline(0, color="gray", lw=0.6, ls="--", zorder=1)
    ax_h.grid(axis="y", ls="--", lw=0.4, alpha=0.6, zorder=0)
    ax_h.legend(frameon=False, loc="upper left", ncol=3)
    for s in ["top", "right"]:
        ax_h.spines[s].set_visible(False)
    fig_h.tight_layout(pad=0.2)
    fig_h.savefig(save_path.replace(".pdf", "_HCS.pdf"), dpi=300, bbox_inches="tight")
    fig_h.savefig(save_path.replace(".pdf", "_HCS.png"), dpi=300, bbox_inches="tight")

    # RA-CLIPScore only
    fig_r, ax_r = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    for i in range(3):
        ax_r.bar(x + offsets[i], R[i], width=bar_w, color=orange_shades[i],
                 edgecolor="none", label=exp_labels[i], zorder=2)
    ax_r.set_ylabel("RA-CLIPScore")
    ax_r.set_ylim(-0.5, 1.4)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(master_attrs, rotation=35, ha="right")
    ax_r.axhline(0, color="gray", lw=0.6, ls="--", zorder=1)
    ax_r.grid(axis="y", ls="--", lw=0.4, alpha=0.6, zorder=0)
    ax_r.legend(frameon=False, loc="upper left", ncol=3)
    for s in ["top", "right"]:
        ax_r.spines[s].set_visible(False)
    fig_r.tight_layout(pad=0.2)
    fig_r.savefig(save_path.replace(".pdf", "_RACS.pdf"), dpi=300, bbox_inches="tight")
    fig_r.savefig(save_path.replace(".pdf", "_RACS.png"), dpi=300, bbox_inches="tight")

    plt.show()


def main(args):
    start_time = time.time()

    master_attrs = ["long hair", "short hair", "young", "elder", "makeup", "no makeup"]
    text_lists = [
        ["long hair", "short hair", "young", "elder", "makeup", "no makeup"],  # Exp1
        ["short hair", "young", "elder", "makeup", "no makeup"],  # Exp2
        ["long hair", "short hair", "young", "makeup", "no makeup"],  # Exp3
    ]

    hcs_results = []
    racs_results = []
    results = []

    img_mean_real = core.get_img_mean(img_dir=args.real_dir, batch_size=1024)
    for text_list in text_lists:
        text_list_prompt_dual = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
        text_list_prompt_single = [f"This is a photo of {item}." for item in text_list]

        # calculate scores
        racs = core.get_evalcs_local_rn_single_image(img_path=args.img_path, text_list=text_list_prompt_dual)
        text_mean = core.get_text_mean(text_list=text_list_prompt_dual)
        hcs = core.get_hcs_for_one_img(img=args.img_path, img_mean=img_mean_real, text_mean=text_mean, text_list=text_list_prompt_single)

        hcs_results.append(hcs.cpu().numpy().squeeze())
        racs_results.append(racs.cpu().numpy().squeeze())
        results.append({"text_list": text_list, "hcs": hcs, "racs": racs})

    # plot bar diagram and save figures
    plot_two_histograms_over_three_sets(
        hcs_results, racs_results, master_attrs, text_lists, args.save_path
    )

    with open(args.save_path, 'wb') as f:
        pickle.dump(results, f)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='single', help='experiment name')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img', help='directory for real images for HCS') # /home/x_yiflu/dataset/Datasets/CelebA/Img  /Users/yifanlu/Downloads/train_34
    parser.add_argument('--img_path', default='./man.png', type=str)
    parser.add_argument('--save_path', default='./triple_experiment.pdf', type=str)
    parser.add_argument('--n_attr', type=int, default=5, help='number of attributes to calculate SaD and PaD')

    args = parser.parse_args()
    main(args)