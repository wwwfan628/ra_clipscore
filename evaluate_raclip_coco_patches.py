"""Evaluate RA-CLIPScore patch localization on 10 COCO classes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, PILToTensor, Resize
from tqdm import tqdm


CLASSES = [
    "airplane", "bus", "car", "train", "bicycle",
    "motorcycle", "cat", "dog", "horse", "cow",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default='/proj/common-datasets/COCO/annotations/instances_val2014.json')
    parser.add_argument("--images", type=Path, default='/proj/common-datasets/COCO/val2014')
    parser.add_argument("--output", type=Path, default=Path("raclip_coco_patch_metrics.json"))
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def patch_ground_truth(coco, annotations, resolution, patch_size, coverage):
    mask = np.zeros((annotations[0]["height"], annotations[0]["width"]), dtype=np.uint8)
    for annotation in annotations:
        mask |= coco.annToMask(annotation).astype(np.uint8)

    transform = Compose([
        Resize(resolution, interpolation=InterpolationMode.NEAREST),
        CenterCrop(resolution),
        PILToTensor(),
    ])
    mask = transform(Image.fromarray(mask)).float().unsqueeze(0)
    return (F.avg_pool2d(mask, patch_size).flatten() >= coverage).numpy()


def average_precision(gt, scores):
    order = np.argsort(-scores, kind="stable")
    ranked_gt = gt[order]
    precision = np.cumsum(ranked_gt) / np.arange(1, len(gt) + 1)
    return float(precision[ranked_gt].mean())


def main():
    args = arguments()
    try:
        import eval_clip
    except ModuleNotFoundError as error:
        raise SystemExit(f"Missing dependency: pip install {error.name}") from error

    coco = COCO(str(args.annotations))
    category_ids = coco.getCatIds(catNms=CLASSES)
    category_names = {c["id"]: c["name"] for c in coco.loadCats(category_ids)}
    if set(category_names.values()) != set(CLASSES):
        raise RuntimeError("The annotation file does not contain all 10 selected classes")

    image_ids = sorted({
        image_id
        for category_id in category_ids
        for image_id in coco.getImgIds(catIds=[category_id])
    })
    if args.max_images is not None:
        image_ids = image_ids[:args.max_images]

    model, preprocess = eval_clip.load("ViT-B/16", device=args.device, jit=False)
    model.eval()
    resolution = int(model.visual.input_resolution)
    patch_size = int(model.visual.conv1.kernel_size[0])

    negative = [f"This is a photo without {name}." for name in CLASSES]
    positive = [f"This is a photo of {name}." for name in CLASSES]
    text = eval_clip.tokenize(negative + positive).to(args.device)
    class_column = {name: i for i, name in enumerate(CLASSES)}

    metrics = defaultdict(lambda: {"ap": [], "iou": []})
    skipped = 0
    with torch.inference_mode():
        for image_id in tqdm(image_ids):
            info = coco.loadImgs([image_id])[0]
            image_path = args.images / info["file_name"]
            if not image_path.exists():
                skipped += 1
                continue

            with Image.open(image_path) as image:
                image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(args.device)
            _, patch_scores = model(image_tensor, text)
            patch_scores = patch_scores[0].float().cpu().numpy()

            annotation_ids = coco.getAnnIds(
                imgIds=[image_id], catIds=category_ids, iscrowd=False
            )
            by_category = defaultdict(list)
            for annotation in coco.loadAnns(annotation_ids):
                annotation["height"], annotation["width"] = info["height"], info["width"]
                by_category[annotation["category_id"]].append(annotation)

            for category_id, annotations in by_category.items():
                gt = patch_ground_truth(
                    coco, annotations, resolution, patch_size, args.coverage
                )
                n = int(gt.sum())
                if n == 0:
                    skipped += 1
                    continue

                scores = patch_scores[:, class_column[category_names[category_id]]]
                prediction = np.zeros_like(gt)
                prediction[np.argpartition(scores, -n)[-n:]] = True
                intersection = int(np.logical_and(prediction, gt).sum())
                union = int(np.logical_or(prediction, gt).sum())

                metrics[category_names[category_id]]["ap"].append(
                    average_precision(gt, scores)
                )
                metrics[category_names[category_id]]["iou"].append(intersection / union)

    per_class = {
        name: {
            "queries": len(metrics[name]["ap"]),
            "patch_ap": float(np.mean(metrics[name]["ap"])),
            "top_n_patch_iou": float(np.mean(metrics[name]["iou"])),
        }
        for name in CLASSES if metrics[name]["ap"]
    }
    if not per_class:
        raise RuntimeError("No valid (image, class) queries were evaluated")
    result = {
        "configuration": {
            "classes": CLASSES,
            "coverage_threshold": args.coverage,
            "patch_grid": [resolution // patch_size, resolution // patch_size],
            "skipped_queries_or_images": skipped,
        },
        "macro_average": {
            "patch_ap": float(np.mean([v["patch_ap"] for v in per_class.values()])),
            "top_n_patch_iou": float(
                np.mean([v["top_n_patch_iou"] for v in per_class.values()])
            ),
        },
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["macro_average"], indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
