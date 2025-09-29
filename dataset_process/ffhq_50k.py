import os, shutil, argparse, random


def main(args):
    os.makedirs(args.dst_root, exist_ok=True)

    # collect all image paths
    all_images = []
    for folder in sorted(os.listdir(args.src_root)):
        folder_path = os.path.join(args.src_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith((".png", ".jpg")):
                all_images.append(os.path.join(folder_path, file))

    print(f"Total images found: {len(all_images)}")

    # randomly sample 50k
    subset = random.sample(all_images, 50000)

    # copy to destination
    for i, src in enumerate(subset):
        filename = os.path.basename(src)  # e.g. "01234.png"
        dst = os.path.join(args.dst_root, filename)
        shutil.copy(src, dst)

    print("Done! 50k images saved in:", args.dst_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str, default='/proj/berzelius-2024-331/users/x_yiflu/ffhq-dataset/images1024x1024', help='source root path')
    parser.add_argument('--dst_root', type=str, default='/home/x_yiflu/dataset/Datasets/ffhq/ffhq_50k', help='destination root path')

    args = parser.parse_args()
    main(args)