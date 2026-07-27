"""Evaluate vanilla CLIP patch-token localization on 10 COCO classes."""

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

import clip
from clip.model import VisionTransformer


CLASSES = [
    "airplane", "bus", "car", "train", "bicycle",
    "motorcycle", "cat", "dog", "horse", "cow",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/proj/common-datasets/COCO/annotations/instances_val2014.json"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("/proj/common-datasets/COCO/val2014"),
    )
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--output", type=Path, default=Path("clip_coco_patch_metrics.json"))
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("clip_coco_patch_scores.jsonl"),
        help="JSONL file containing the score of every patch for each evaluated query",
    )
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Mapped positive-prompt cosine-score threshold used for patch accuracy",
    )
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


def encode_patch_tokens(model, images):
    """Return projected final-layer ViT patch tokens, excluding the CLS token."""
    visual = model.visual
    if not isinstance(visual, VisionTransformer):
        raise TypeError("Patch-token evaluation requires a ViT CLIP model")

    x = visual.conv1(images.type(model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x[:, 1:, :])
    if visual.proj is not None:
        x = x @ visual.proj
    return x


def clip_patch_scores(model, images, text_features):
    """Compute positive-prompt cosine scores, mapped from [-1, 1] to [0, 1]."""
    patch_features = F.normalize(encode_patch_tokens(model, images).float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    cosine_similarity = patch_features @ text_features.T
    return (cosine_similarity + 1.0) / 2.0


def average_precision(gt, scores):
    order = np.argsort(-scores, kind="stable")
    ranked_gt = gt[order]
    precision = np.cumsum(ranked_gt) / np.arange(1, len(gt) + 1)
    return float(precision[ranked_gt].mean())


def main():
    args = arguments()
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

    model, preprocess = clip.load(args.model, device=args.device, jit=False)
    model.eval()
    if not isinstance(model.visual, VisionTransformer):
        raise ValueError(f"{args.model} is not a ViT model; choose e.g. ViT-B/16")
    resolution = int(model.visual.input_resolution)
    patch_size = int(model.visual.conv1.kernel_size[0])
    grid_size = resolution // patch_size

    positive = [f"This is a photo of {name}." for name in CLASSES]
    text = clip.tokenize(positive).to(args.device)
    class_column = {name: i for i, name in enumerate(CLASSES)}

    metrics = defaultdict(lambda: {"ap": [], "iou": [], "accuracy": []})
    skipped = 0
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_output.open("w") as scores_file, torch.inference_mode():
        text_features = model.encode_text(text)
        for image_id in tqdm(image_ids):
            info = coco.loadImgs([image_id])[0]
            image_path = args.images / info["file_name"]
            if not image_path.exists():
                skipped += 1
                continue

            with Image.open(image_path) as image:
                image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(args.device)
            patch_scores = clip_patch_scores(
                model, image_tensor, text_features
            )[0].cpu().numpy()

            annotation_ids = coco.getAnnIds(
                imgIds=[image_id], catIds=category_ids, iscrowd=False
            )
            by_category = defaultdict(list)
            for annotation in coco.loadAnns(annotation_ids):
                annotation["height"], annotation["width"] = info["height"], info["width"]
                by_category[annotation["category_id"]].append(annotation)

            for category_id, annotations in by_category.items():
                class_name = category_names[category_id]
                gt = patch_ground_truth(
                    coco, annotations, resolution, patch_size, args.coverage
                )
                n = int(gt.sum())
                if n == 0:
                    skipped += 1
                    continue

                scores = patch_scores[:, class_column[class_name]]
                prediction = np.zeros_like(gt, dtype=bool)
                prediction[np.argpartition(scores, -n)[-n:]] = True
                intersection = int(np.logical_and(prediction, gt).sum())
                union = int(np.logical_or(prediction, gt).sum())
                threshold_prediction = scores >= args.score_threshold

                metrics[class_name]["ap"].append(average_precision(gt, scores))
                metrics[class_name]["iou"].append(intersection / union)
                metrics[class_name]["accuracy"].append(
                    float(np.mean(threshold_prediction == gt))
                )
                scores_file.write(json.dumps({
                    "image_id": image_id,
                    "file_name": info["file_name"],
                    "class": class_name,
                    "grid": [grid_size, grid_size],
                    "patch_scores": scores.tolist(),
                    "patch_labels": gt.astype(np.uint8).tolist(),
                }) + "\n")

    per_class = {
        name: {
            "queries": len(metrics[name]["ap"]),
            "patch_accuracy": float(np.mean(metrics[name]["accuracy"])),
            "patch_ap": float(np.mean(metrics[name]["ap"])),
            "top_n_patch_iou": float(np.mean(metrics[name]["iou"])),
        }
        for name in CLASSES if metrics[name]["ap"]
    }
    if not per_class:
        raise RuntimeError("No valid (image, class) queries were evaluated")

    result = {
        "configuration": {
            "model": args.model,
            "classes": CLASSES,
            "prompts": positive,
            "patch_score": "(cosine_similarity + 1) / 2",
            "coverage_threshold": args.coverage,
            "score_threshold": args.score_threshold,
            "patch_grid": [grid_size, grid_size],
            "skipped_queries_or_images": skipped,
            "scores_output": str(args.scores_output),
        },
        "macro_average": {
            "patch_accuracy": float(
                np.mean([v["patch_accuracy"] for v in per_class.values()])
            ),
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
    print(f"Patch scores saved to {args.scores_output}")
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
