import argparse 
import core
import utils

from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='celebA', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--real_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/CelebA_5K')
    parser.add_argument('--gen_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/CelebA_process/CelebA_5k_Eyeglasses_Injection_800')
    parser.add_argument('--n_attr', type=int, default=2, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--n_point', type=int, default=10000, help='number of points to calculate KLD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')
    parser.add_argument('--score_type', type=str, default='HCS', help='score type, e.g. HCS for this script')

    args = parser.parse_args()

    experiment_name = f"{args.exp}_{args.n_attr}_{args.attr_type}_{args.score_type}"

    text_list, stats = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr, attr_type=args.attr_type)

    #calculate text, image mean
    text_mean = core.get_text_mean(text_list)
    img_mean = core.get_img_mean(img_dir=args.real_dir, batch_size=100)

    #calculate HCS
    HCS_real = core.get_hcs(args.real_dir, img_mean, text_mean, text_list, batch_size=1000)
    HCS_gen = core.get_hcs(args.gen_dir, img_mean, text_mean, text_list, batch_size=1000)

    #calculate SaD and PaD
    SaD = core.get_SaD(HCS_real,HCS_gen, text_list, args.n_point)
    PaD = core.get_PaD(HCS_real,HCS_gen, text_list, args.n_point)

    #get results
    mean_SaD, mean_diff = utils.plot_SaD(SaD, text_list, experiment_name, args.real_dir, args.gen_dir, args.n_point)
    mean_PaD = utils.plot_PaD(PaD, text_list, experiment_name, args.real_dir, args.gen_dir, args.n_point)

    print(f"SaD = {mean_SaD}")
    print(f"SaD = {mean_SaD}, PaD = {mean_PaD}")