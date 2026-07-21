import argparse
import core
import utils
import os
import time
from tqdm import tqdm
import pickle


def main(args):
    start_time = time.time()
    experiment_name = f"{args.exp}_{args.attr_index}_{args.attr}"
    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)

    text_list = [args.attr]

    # calculate text, image mean
    text_mean = core.get_text_mean(text_list)
    img_mean = core.get_img_mean(img_dir=args.real_dir, batch_size=1000)

    # calculate HCS
    HCS_real = core.get_hcs(args.real_dir, img_mean, text_mean, text_list, batch_size=1000)
    HCS_gen = core.get_hcs(args.gen_dir, img_mean, text_mean, text_list, batch_size=1000)

    # calculate SaD
    SaD = core.get_SaD(HCS_real, HCS_gen, text_list, args.n_point)

    # save results
    save_dir = f"results_dir/{experiment_name}/{dataset_real}_vs_{dataset_gen}/"
    os.makedirs(save_dir, exist_ok=True)
    results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    results_dict = {'attr_list': text_list, 'SaD': SaD}

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)

    print(f"SaD = {SaD}")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='imagenet', help='experiment name')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/imagenet256/val')
    parser.add_argument('--gen_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Imagenet256-LDM')
    parser.add_argument('--attr_index', type=int, default=965, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')
    parser.add_argument('--attr', type=str, default='burrito', help='attribute to evaluate')

    args = parser.parse_args()

    main(args)
