import argparse
import os.path

import core
import utils
import pickle
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    experiment_name= f"{args.exp}_{args.n_attr}_{args.attr_type}"

    text_list, stats = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr, attr_type=args.attr_type)

    # calculate text, image mean
    text_features = core.get_text_features(text_list)
    if os.path.exists(args.text_feat_mean_file):
        with open(args.text_feat_mean_file, 'rb') as f:
            text_mean = pickle.load(f)
    else:
        text_mean = core.get_text_mean(text_list)
        with open(args.text_feat_mean_file, 'wb') as f:
            pickle.dump(text_features, f)

    if os.path.exists(args.img_feat_mean_file):
        with open(args.img_feat_mean_file, 'rb') as f:
            img_mean = pickle.load(f)
    else:
        img_mean = core.get_img_mean(img_dir=args.real_dir, batch_size=1000)
        with open(args.img_feat_mean_file, 'wb') as f:
            pickle.dump(img_mean, f)

    # calculate HCS for one image
    HCS = core.get_hcs_for_one_img(args.test_img, img_mean, text_mean, text_list)

    # plot_bar
    # check if results has the same length as labels
    assert HCS.shape[0] == len(text_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(len(text_list)))
    ax.set_xticklabels(text_list, rotation=90, fontsize=6)
    ax.bar(text_list, HCS, width=0.8)

    plt.title(f"Heterogeneous CLIPScore\n")
    plt.ylabel('Heterogeneous CLIPScore for each attribute', fontsize=5)
    plt.tight_layout()
    plt.savefig(f"hcs_dolphin_bar35.png", dpi=2000)
    plt.show()


    # calculate DualCS for one image
    DualCS = core.get_dualcs_for_one_img(args.test_img, text_list)

    # plot_bar
    # check if results has the same length as labels
    assert DualCS.shape[0] == len(text_list)

    DualCS = DualCS[::2]
    text_list = text_list[::2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(len(text_list)))
    ax.set_xticklabels(text_list, rotation=90, fontsize=6)
    ax.bar(text_list, DualCS, width=0.8)

    plt.title(f"Dual CLIPScore\n")
    plt.ylabel('Dual CLIPScore for each attribute', fontsize=5)
    plt.tight_layout()
    plt.savefig(f"dcs_dolphin_bar35.png", dpi=2000)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='celebA', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--test_img', type=str, default='/Users/yifanlu/Desktop/dolphin.png')
    parser.add_argument('--img_feat_mean_file', type=str, default='./img_feat_mean.pkl')
    parser.add_argument('--text_feat_mean_file', type=str, default='./text_feat_mean.pkl')
    parser.add_argument('--real_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/non_smiling_man_smiling_woman_1')
    parser.add_argument('--n_attr', type=int, default=20, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')

    args = parser.parse_args()
    main(args)