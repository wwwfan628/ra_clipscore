import argparse 
import core
import sadpad.utils as utils
import os
import time
from tqdm import tqdm
import pickle


def main(args):
    start_time = time.time()
    experiment_name = f"{args.exp}_{args.n_attr}_{args.attr_type}"

    text_list, stats = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr,
                                            attr_type=args.attr_type)

    # calculate text, image mean
    text_mean = core.get_text_mean(text_list)
    img_mean = core.get_img_mean(img_dir=args.real_dir, batch_size=1000)

    # calculate HCS
    HCS_real = core.get_hcs(args.real_dir, img_mean, text_mean, text_list, batch_size=1000)
    HCS_gen = core.get_hcs(args.gen_dir, img_mean, text_mean, text_list, batch_size=1000)

    # calculate SaD and PaD
    SaD = core.get_SaD(HCS_real, HCS_gen, text_list, args.n_point)
    PaD = core.get_PaD(HCS_real, HCS_gen, text_list, args.n_point)

    # save results
    dataset_real = os.path.basename(args.real_dir)
    dataset_gen = os.path.basename(args.gen_dir)
    experiment_name = f"{args.exp}_{args.n_attr}_{args.attr_type}_datasetcompare"
    save_dir = f"results_dir/{experiment_name}/{dataset_real}_vs_{dataset_gen}/"
    os.makedirs(save_dir, exist_ok=True)
    results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    results_dict = {'attr_list': text_list, 'SaD': SaD, 'PaD': PaD}

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)

    # get results
    mean_SaD, mean_diff = utils.plot_SaD(SaD, text_list, experiment_name, args.real_dir, args.gen_dir, args.n_point)
    mean_PaD = utils.plot_PaD(PaD, text_list, experiment_name, args.real_dir, args.gen_dir, args.n_point)

    print(f"SaD = {mean_SaD}")
    print(f"SaD = {mean_SaD}, PaD = {mean_PaD}")

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f'Excution Time: {elapsed}!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='ffhq', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--real_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Dataset_Compare/A')
    parser.add_argument('--gen_dir', type=str, default='/home/x_yiflu/dataset/Datasets/Dataset_Compare/B')
    parser.add_argument('--n_attr', type=int, default=20, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')

    args = parser.parse_args()

    main(args)
