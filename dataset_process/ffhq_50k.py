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

    if args.random:
        # randomly sample images
        subset = random.sample(all_images, args.num_img)
    else:
        subset = all_images[args.start_index : args.start_index + args.num_img]

    # copy to destination
    for i, src in enumerate(subset):
        filename = os.path.basename(src)  # e.g. "01234.png"
        dst = os.path.join(args.dst_root, filename)
        shutil.copy(src, dst)

    print(f"Done! {args.num_img} images saved in:", args.dst_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str, default='/home/x_yiflu/dataset/Datasets/ffhq/ffhq_full', help='source root path')
    parser.add_argument('--dst_root', type=str, default='/home/x_yiflu/dataset/Datasets/ffhq/ffhq_50k', help='destination root path')
    parser.add_argument('--num_img', type=int, default='50000', help='number of images')
    parser.add_argument('--start_index', type=int, default='0', help='sample from this index')
    parser.add_argument('--random', action='store_true', help='whether to sample the images randomly')

    args = parser.parse_args()
    main(args)