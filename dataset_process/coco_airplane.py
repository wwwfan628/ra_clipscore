import json, os, random, shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

def find_coco_airplanes_by_side(
    coco_root: str,
    n_images: int = 100,
    side: str = "left",                 # "left" or "right"
    split: float = 0.5,                 # vertical split as fraction of width
    min_bbox_area_ratio: float = 0.0,   # filter very small airplanes (area/img_area)
    seed: int = 42,
    copy_to: str | None = None,
    save_list_to: str | None = "airplane_by_side.txt",
) -> List[str]:
    """
    Select COCO 2014 images that contain 'airplane' with bbox centers on the requested side.
    NOW: an image is kept only if **ALL** airplane bboxes satisfy the side rule (+ area filter).

    COCO layout:
      coco_root/
        annotations/
          instances_train2014.json
          instances_val2014.json
        train2014/
        val2014/
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if not (0.0 < split < 1.0):
        raise ValueError("split must be in (0,1)")

    coco_root = Path(coco_root)
    ann_dir = coco_root / "annotations"
    ann_files = [ann_dir / "instances_train2014.json", ann_dir / "instances_val2014.json"]

    # Load annotations
    datasets: List[Tuple[str, Dict[str, Any]]] = []
    for p in ann_files:
        if p.exists():
            with open(p, "r") as f:
                datasets.append((p.stem, json.load(f)))
    if not datasets:
        raise FileNotFoundError("No COCO 2014 annotation files found under 'annotations/'.")

    # Find category_id for 'airplane'
    cat_id = None
    for _, data in datasets:
        for c in data["categories"]:
            if c["name"].lower() == "airplane":
                cat_id = c["id"]
                break
        if cat_id is not None:
            break
    if cat_id is None:
        raise RuntimeError("Category 'airplane' not found.")

    def collect(split_name: str, data: Dict[str, Any]) -> List[str]:
        img_meta = {img["id"]: img for img in data["images"]}

        # group airplane anns per image
        plane_anns_per_img: Dict[int, list] = {}
        for ann in data["annotations"]:
            if ann.get("category_id") == cat_id and "bbox" in ann and ann.get("iscrowd", 0) == 0:
                plane_anns_per_img.setdefault(ann["image_id"], []).append(ann)

        split_dir = coco_root / ("train2014" if "train" in split_name else "val2014")

        selected_paths = []
        for iid, anns in plane_anns_per_img.items():
            img = img_meta.get(iid)
            if not img:
                continue
            W, H = img.get("width"), img.get("height")
            if not (isinstance(W, int) and isinstance(H, int) and W > 0 and H > 0):
                continue
            split_x = split * W
            img_area = float(W * H)

            # ✅ ALL bboxes must satisfy the side rule (and area filter)
            keep = True
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue  # ignore degenerate bbox; doesn't auto-fail the image
                center_x = x + 0.5 * w
                area_ok = (w * h) / img_area >= min_bbox_area_ratio
                if side == "left":
                    cond = center_x < split_x
                else:  # "right"
                    cond = center_x >= split_x
                if not (cond and area_ok):
                    keep = False
                    break

            if keep:
                f = img.get("file_name")
                if not f:
                    continue
                img_path = split_dir / f
                if img_path.is_file():
                    selected_paths.append(str(img_path))
        return selected_paths

    # Collect train+val and dedupe
    all_paths = []
    for split_name, data in datasets:
        all_paths.extend(collect(split_name, data))
    all_paths = list(dict.fromkeys(all_paths))  # stable dedupe

    if not all_paths:
        raise RuntimeError("No matching airplane images found (try adjusting split/area threshold).")

    random.Random(seed).shuffle(all_paths)
    selected = all_paths[: min(n_images, len(all_paths))]

    if copy_to:
        dst = Path(copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p in selected:
            shutil.copy2(p, dst / Path(p).name)

    if save_list_to:
        with open(save_list_to, "w") as f:
            for p in selected:
                f.write(p + "\n")

    return selected


if __name__ == "__main__":
    paths = find_coco_airplanes_by_side(
        coco_root="/Users/yifanlu/Desktop/Datasets/coco2014",
        n_images=500,
        side="right",               # or "right"
        split=0.6,                 # move toward 0.5 for more permissive; e.g., 0.55 for 'left'
        min_bbox_area_ratio=0.0, # ignore tiny airplanes; set 0.0 to keep all
        seed=123,
        copy_to="/Users/yifanlu/Desktop/Datasets/airplanes_right_all_bboxes_500",
        save_list_to="airplanes_left_all_bboxes_100.txt",
    )
    print(f"Collected {len(paths)} images.")