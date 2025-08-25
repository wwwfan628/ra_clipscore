import json, os, random, shutil
from pathlib import Path

def find_coco_airplanes(
    coco_root: str,
    n_images: int = 100,
    seed: int = 42,
    copy_to: str | None = None,
    save_list_to: str | None = "airplane_100.txt",
):
    """
    coco_root layout (standard):
      coco_root/
        annotations/
          instances_train2014.json
          instances_val2014.json
        train2014/
        val2014/
    """
    coco_root = Path(coco_root)
    ann_dir = coco_root / "annotations"

    ann_files = [
        ann_dir / "instances_train2014.json",
        ann_dir / "instances_val2014.json",
    ]
    anns = []
    for p in ann_files:
        if p.exists():
            with open(p, "r") as f:
                anns.append((p.stem, json.load(f)))  # (split_name, data)

    if not anns:
        raise FileNotFoundError("No COCO 2014 annotation files found in 'annotations/'.")

    # Find category_id for 'airplane'
    cat_id = None
    for _, data in anns:
        for c in data["categories"]:
            if c["name"].lower() == "airplane":
                cat_id = c["id"]
                break
        if cat_id is not None:
            break
    if cat_id is None:
        raise RuntimeError("Category 'airplane' not found in annotations.")

    # Helper: build lookup of images with airplane per split
    def collect(split_name, data):
        # image_id -> file_name
        img_by_id = {img["id"]: img["file_name"] for img in data["images"]}
        # any annotation with cat_id marks the image as containing airplane
        has_plane = {ann["image_id"] for ann in data["annotations"] if ann["category_id"] == cat_id}
        # Map to paths
        split_dir = coco_root / ("train2014" if "train" in split_name else "val2014")
        return [str(split_dir / img_by_id[iid]) for iid in has_plane if iid in img_by_id]

    all_plane_paths = []
    for split_name, data in anns:
        all_plane_paths.extend(collect(split_name, data))

    # Deduplicate; only keep files that actually exist on disk
    all_plane_paths = list({p for p in all_plane_paths if os.path.isfile(p)})

    if not all_plane_paths:
        raise RuntimeError("No airplane images found (or image files missing on disk).")

    random.Random(seed).shuffle(all_plane_paths)
    selected = all_plane_paths[: min(n_images, len(all_plane_paths))]

    # Optionally copy the images to a destination folder
    if copy_to:
        dst = Path(copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p in selected:
            shutil.copy2(p, dst / Path(p).name)

    # Optionally save a text list of paths
    if save_list_to:
        with open(save_list_to, "w") as f:
            for p in selected:
                f.write(p + "\n")

    return selected

if __name__ == "__main__":
    # Example usage:
    paths = find_coco_airplanes(
        coco_root="/Users/yifanlu/Desktop/Datasets/coco2014",
        n_images=200,
        seed=123,
        copy_to="/Users/yifanlu/Desktop/Datasets/airplane_100_1",    # or None to skip copying
        save_list_to="airplane_100.txt",           # or None to skip saving list
    )
    print(f"Collected {len(paths)} airplane images.")