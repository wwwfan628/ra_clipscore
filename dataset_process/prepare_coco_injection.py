from pycocotools.coco import COCO
import pandas as pd
import argparse
import os
import json
import random
from pathlib import Path
from typing import Optional
import shutil

##########################################################################################
# Print out the statistics of coco dataset, based on the result we choose chair class for injection
# Path to COCO annotation file
# annFile = "./instances_train2014.json"
#
# coco = COCO(annFile)
#
# # Get all category names
# cats = coco.loadCats(coco.getCatIds())
# cat_names = [c["name"] for c in cats]
#
# results = []
# for cat in cats:
#     cat_id = cat["id"]
#     img_ids = coco.getImgIds(catIds=[cat_id])
#     results.append((cat["name"], len(img_ids)))
#
# # Save to DataFrame
# df = pd.DataFrame(results, columns=["class","images_with_class"])
# df = df.sort_values(by="images_with_class", ascending=False)
#
# print(df)
#########################################################################################


def main(
    ann_file: str,
    images_dir: str,
    out_dir_root: str = "/home/x_yiflu/dataset/Datasets/Injection_Experiment_coco",
    seed: int = 0,
    base_size: int = 50000,
    inject_sizes: list[int] = list(range(100,1100,100)),
    target_classes: list[Optional[str]] = [None, "chair"],  # None: randomly sampled from coco (comparison group)
):
    random.seed(seed)

    coco = COCO(ann_file)

    # check all imgs
    all_img_ids = coco.getImgIds()

    if base_size > len(all_img_ids):
        raise ValueError("base_size larger than dataset")

    # Base dataset: random 50k
    base_ids = set(random.sample(all_img_ids, base_size))
    base_ids_list = sorted(base_ids)

    # create base dataset coco50k
    out_dir = Path(out_dir_root)
    base_img_dir = out_dir / f"coco_{base_size}"
    base_img_dir.mkdir(parents=True, exist_ok=True)

    base_imgs = coco.loadImgs(base_ids_list)
    for info in base_imgs:
        src = Path(images_dir) / info["file_name"]
        dst = base_img_dir / info["file_name"]
        shutil.copy2(src, dst)
    print(f"BASE DATASET: {base_size} base images → {base_img_dir}!")


    for target_class in target_classes:
        # Injection datasets
        if target_class is None:
            inj_img_dir = out_dir / "original"
            inj_img_dir.mkdir(parents=True, exist_ok=True)
            # Random injection, no class constraint
            candidates = sorted(set(all_img_ids) - base_ids)
            if inject_sizes[-1] > len(candidates):
                raise ValueError("Not enough images to sample injection set")
            candidates = sorted(random.sample(candidates, inject_sizes[-1]))
            inject_ids_list = []
            inject_name_list = []
            for num_inject in inject_sizes:
                inject_ids = candidates[:num_inject]
                inject_name = f"coco_{num_inject}"
                inject_ids_list.append(inject_ids)
                inject_name_list.append(inject_name)
        else:
            # Class-based injection
            inj_img_dir = out_dir / f"{target_class}"
            inj_img_dir.mkdir(parents=True, exist_ok=True)
            cat_ids = coco.getCatIds(catNms=[target_class])
            if not cat_ids:
                raise ValueError(f"Category '{target_class}' not found")

            cat_id = cat_ids[0]
            class_img_ids = set(coco.getImgIds(catIds=[cat_id]))
            candidates = sorted(class_img_ids - base_ids)
            if inject_sizes[-1] > len(candidates):
                raise ValueError(
                    f"Not enough '{target_class}' images after excluding base"
                )
            candidates = sorted(random.sample(candidates, inject_sizes[-1]))
            inject_ids_list = []
            inject_name_list = []
            for num_inject in inject_sizes:
                inject_ids = candidates[:num_inject]
                inject_name = f"{target_class}_{num_inject}"
                inject_ids_list.append(inject_ids)
                inject_name_list.append(inject_name)

        for inject_ids, inject_name in zip(inject_ids_list, inject_name_list):
            inject_dir = inj_img_dir / inject_name
            inject_dir.mkdir(parents=True, exist_ok=True)
            base_imgs = coco.loadImgs(base_ids_list)
            for info in base_imgs:
                src = Path(images_dir) / info["file_name"]
                dst = inject_dir / info["file_name"]
                shutil.copy2(src, dst)
            print(f"INJECTION DATASET {inject_name}: {base_size} base images → {inject_dir}!")
            inj_imgs = coco.loadImgs(inject_ids)
            for info in inj_imgs:
                src = Path(images_dir) / info["file_name"]
                dst = inject_dir / info["file_name"]
                shutil.copy2(src, dst)
            print(f"INJECTION DATASET {inject_name}: {len(inject_ids)} {target_class} images → {inject_dir}!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, default='/home/x_yiflu/dataset/Datasets/coco/annotations/instances_train2014.json')
    parser.add_argument('--img_dir', type=str, default='/home/x_yiflu/dataset/Datasets/coco/train2014')  # CelebA attributes annotation file path
    parser.add_argument('--dst_root', type=str, default='/home/x_yiflu/dataset/Datasets/Injection_Experiment_coco')  # save under this path
    args = parser.parse_args()

    main(
        ann_file=args.ann_file,
        images_dir=args.img_dir,
        out_dir_root=args.dst_root,
        seed=0,
        base_size=50000,
        inject_sizes=list(range(100, 1100, 100)),
        target_classes=[None, "chair"]
    )
