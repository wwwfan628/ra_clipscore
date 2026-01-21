import pandas as pd
import argparse
import random
import os
import shutil

#----------------------------------------- Original Setting ----------------------------------------------------
#########################################################################################
# CelebA Statistics：
# Male 84434 (w/ Eyeglasses 10478)   Female 118165 (w/ Eyeglasses 2715)  Total 202599 (w/ Eyeglasses 13193)
# p_Male = 0.4167   p_Glasses = 0.0651
# If we want to create set of 50000 images, bottleneck is the number of woman with eyeglasses: 50000 * 0.0651 = 3255 > 2715, male 50000 * 0.4167 = 20835 < 84434 OK
#
# Therefore we create set with eyeglasses 2715.
#
# Set A/B: Male w/ Eyeglasses 2715    Male w/o Eyeglasses 50000x0.4167-2715=18120   Female w/o Eyeglasses 50000x(1-0.4167)=29165
#
# Set C: Female w/ Eyeglasses 2715    Female w/o Eyeglasses 50000x(1-0.4167)-2715=26450   Male w/o Eyeglasses 50000x0.4167=20835
#
# Marginal distribution 41.7% Male   5.43% Eyeglasses for Set A/B/C
########################################################################################
#
# N = 50000
# M_target = 20835
# G_target = 2715
#-----------------------------------------------------------------------------------------------------------------


#----------------------------------------- New Setting ----------------------------------------------------------
#########################################################################################
# CelebA Statistics：
# Male 84434 (w/ Eyeglasses 10478)  Female 118165 (w/ Eyeglasses 2715)  Total 202599 (w/ Eyeglasses 13193)
# 10478/84434=0.1241                2715/118165=0.0229                  13193/202599=0.0651
#
# A/B/C: 50000 images
# Set A/B: Male 25000 Female 25000   Male w/ Eyeglasses 25000*0.1=2500    Male w/o Eyeglasses 25000-2500=22500   Female w/ Eyeglasses 0   Female w/o Eyeglasses 25000
# Set C: Male 25000 Female 25000    Male w/ Eyeglasses 0    Male w/o Eyeglasses 25000   Female w/ Eyeglasses 25000*0.1=2500    Female w/o Eyeglasses 25000-2500=22500
########################################################################################

N = 50000
M_target = 25000
G_target = 2500
#-----------------------------------------------------------------------------------------------------------------

def main(args):
    annotations = pd.read_csv(args.celeba_ann_file_path, header=[0], delimiter=r"\s+")
    MG = list(annotations[(annotations['Male'] == 1) & (annotations['Eyeglasses'] == 1)].index)
    MnG = list(annotations[(annotations['Male'] == 1) & (annotations['Eyeglasses'] == -1)].index)
    FG = list(annotations[(annotations['Male'] == -1) & (annotations['Eyeglasses'] == 1)].index)
    FnG = list(annotations[(annotations['Male'] == -1) & (annotations['Eyeglasses'] == -1)].index)

    sets = {
        "A": {"MG": G_target, "MnG": M_target - G_target, "FnG": N - M_target},
        "B": {"MG": G_target, "MnG": M_target - G_target, "FnG": N - M_target},
        "C": {"FG": G_target, "MnG": M_target, "FnG": N - M_target - G_target},
    }

    pools = {"MG": MG, "MnG": MnG, "FG": FG, "FnG": FnG}

    for setname, quotas in sets.items():
        out_dir = os.path.join(args.dst_root, setname)
        os.makedirs(out_dir, exist_ok=True)
        chosen = []
        for bucket, k in quotas.items():
            if k > 0:
                chosen += random.sample(pools[bucket], k)
        for fn in chosen:
            src = os.path.join(args.celeba_img_root, fn)
            dst = os.path.join(out_dir, fn)
            shutil.copy2(src, dst)
        print(f"Set {setname}: {len(chosen)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeba_img_root', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Img')  # CelebA images folder
    parser.add_argument('--celeba_ann_file_path', type=str, default='/home/x_yiflu/dataset/Datasets/CelebA/Anno/list_attr_celeba.txt')  # CelebA attributes annotation file path
    parser.add_argument('--dst_root', type=str, default='/home/x_yiflu/dataset/Datasets/Dataset_Compare_2')  # save under this path

    args = parser.parse_args()

    main(args)