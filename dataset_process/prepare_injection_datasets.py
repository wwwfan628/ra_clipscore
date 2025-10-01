import os, shutil, argparse

def main(args):
    for i in range(11):
        translation_prompt = os.path.basename(args.trans_root)
        dst_path = os.path.join(args.dst_root, translation_prompt, f'{i*100}')
        os.makedirs(dst_path, exist_ok=True)

        # copy ffhq_50k folder and rename in one step
        shutil.copytree(args.ffhq50k_root, dst_path, dirs_exist_ok=True)
        print(f"Copied FFHQ50k {args.ffhq50k_root} → {dst_path}")
        translated_images = [img for img in os.listdir(args.trans_root) if img.endswith('.png')]
        subset = translated_images[:i*100]

        # copy to destination
        for j, img in enumerate(subset):
            src = os.path.join(args.trans_root, img)
            dst = os.path.join(dst_path, img)
            shutil.copy(src, dst)
        print(f"Done! {i*100} images saved in:", dst_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffhq50k_root', type=str, default='/home/x_yiflu/dataset/Datasets/ffhq/ffhq_50k')
    parser.add_argument('--trans_root', type=str, default='/home/x_yiflu/dataset/Datasets/ffhq/DiffuseIT_ffhq_make_up')  # CelebA attributes annotation file path
    parser.add_argument('--dst_root', type=str, default='/home/x_yiflu/dataset/Datasets/Injection_Experiment')  # save under this path
    args = parser.parse_args()

    main(args)