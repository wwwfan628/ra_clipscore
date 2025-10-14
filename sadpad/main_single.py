import argparse
import core
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='ffhq', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img', help='directory for real images')
    parser.add_argument('--img_path', default='./man.png', type=str)
    parser.add_argument('--save_path', default='./hcs.png', type=str)
    parser.add_argument('--n_attr', type=int, default=5, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')

    args = parser.parse_args()
    experiment_name= f"{args.exp}_{args.n_attr}_{args.attr_type}"

    text_list, stats = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr, attr_type=args.attr_type)

    #calculate text, image mean
    text_mean = core.get_text_mean(text_list)
    img_mean = core.get_img_mean(img_dir=args.real_dir, batch_size=1000)

    #calculate HCS
    HCS = core.get_hcs_single(args.img_path, img_mean, text_mean, text_list)

    #plot
    assert len(HCS) == len(text_list)

    # 设置 seaborn 风格
    sns.set_theme(style="whitegrid")  # 白色背景+网格

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=text_list, y=HCS.cpu().numpy(), color="skyblue")

    # 设置标题和标签
    plt.title(f"Heterogeneous CLIP Score (HCS)\nmean={np.mean(HCS.cpu().numpy()):.4f}, var={np.var(HCS.cpu().numpy(), dtype=np.float32):.4f}")
    plt.ylabel('HCS for each attribute', fontsize=8)
    ymin, ymax = -7, 11
    plt.ylim((ymin, ymax))
    plt.xticks(rotation=90)
    plt.tight_layout()
    if args.save_path is not None:
        plt.savefig(args.save_path, dpi=300)
    else:
        plt.show()