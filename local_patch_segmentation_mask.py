import argparse
import os.path

import core
import utils
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time

import eval_clip

import os
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import colorsys


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform_without_normalize(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor()
    ])


def _make_distinct_colors(n, seed=0, s=0.65, v=0.95):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    rng.shuffle(hues)
    return [colorsys.hsv_to_rgb(h, s, v) for h in hues]


def _pick_topk_classes(up_logits, keep_k=5, background_id=0, small_obj_tau=0.7, select_mode="max"):
    """
    Select which classes to keep (including background).
    up_logits: [1, C, H, W] logits upsampled to pixel space.
    select_mode in {"mean","sum","max","argmax_freq"}:
      - "sum": total prob mass over image (recommended)
      - "mean": average prob (area-normalized)
      - "max": highest prob anywhere (small objects favored, can be noisy)
      - "argmax_freq": frequency after argmax
    Returns: kept_ids (list[int]) with background_id included.
    """
    C = up_logits.shape[1]
    probs = up_logits.softmax(dim=1)          # [1,C,H,W]
    non_bg = [i for i in range(C) if i != background_id]

    if select_mode == "mean":
        scores = probs.mean(dim=(0,2,3))      # [C]
    elif select_mode == "sum":
        scores = probs.sum(dim=(0,2,3))       # [C]
    elif select_mode == "max":
        scores = probs.amax(dim=(0,2,3))      # [C]
    elif select_mode == "argmax_freq":
        raw = up_logits.argmax(dim=1)[0]      # [H,W]
        scores = torch.bincount(raw.view(-1), minlength=C).float()
    else:
        raise ValueError("select_mode must be one of: mean, sum, max, argmax_freq")

    # base top-k among non-background
    base_scores = scores[non_bg]
    k = min(keep_k, len(non_bg))
    topk_vals, topk_idx = torch.topk(base_scores, k)
    kept = {non_bg[i.item()] for i in topk_idx}

    # small-object safeguard: also keep any class with high max prob
    max_probs = probs.amax(dim=(0,2,3))       # [C]
    high_conf = {i for i in non_bg if max_probs[i] >= small_obj_tau}

    # merge + sort by score, include background first
    merged = [background_id] + sorted((kept | high_conf), key=lambda i: scores[i], reverse=True)
    return merged[:min(len(merged), keep_k + 1)]  # cap to K + background



# ---------- main: logits -> mask with top-k filtering ----------
def logits_to_mask_keep_topk(
    patch_logits, out_h, out_w, grid=(14,14),
    keep_k=5, background_id=0,
    select_mode="sum", small_obj_tau=0.7,
    reassign_to_kept=False
):
    """
    patch_logits: (Gx, Gy, C) or (Gx*Gy, C) logits per patch per class
    select_mode: how to pick top-k classes ("sum" recommended)
    reassign_to_kept:
        - False: argmax over all classes, then non-kept -> background
        - True:  zero out non-kept logits first, then argmax (fallback to kept)
    Returns:
        mask: [H,W] int tensor (background black later in plot)
        kept_ids: list[int] kept classes incl. background
        up_logits: [1,C,H,W] upsampled logits (for debugging/metrics)
    """
    Gx, Gy = grid
    if patch_logits.ndim == 2:
        patch_logits = patch_logits.view(Gx, Gy, -1)  # (Gx, Gy, C)
    C = patch_logits.shape[-1]

    # Upsample logits to pixel space
    logits = patch_logits.permute(2,0,1)[None]  # [1,C,Gx,Gy]
    up_logits = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)  # [1,C,H,W]

    # Pick kept classes
    kept_ids = _pick_topk_classes(
        up_logits, keep_k=keep_k, background_id=background_id,
        small_obj_tau=small_obj_tau, select_mode=select_mode
    )
    kept_set = set(kept_ids)

    if reassign_to_kept:
        # Fallback method: restrict logits to kept classes, then argmax
        mask_logits = up_logits.clone()
        not_kept = [i for i in range(C) if i not in kept_set]
        if len(not_kept) > 0:
            mask_logits[:, not_kept] = float('-inf')
        mask = mask_logits.argmax(dim=1)[0]
    else:
        # Strict method: argmax first, then force non-kept to background
        raw_mask = up_logits.argmax(dim=1)[0]
        mask = raw_mask.clone()
        # send non-kept to background
        allowed = torch.tensor(list(kept_set), device=mask.device)
        keep_bool = torch.isin(mask, allowed)
        mask[~keep_bool] = background_id

    return mask, kept_ids, up_logits


# ---------- plotting ----------
def plot_mask_overlay(
    image_tensor, mask, class_names, kept_ids, seed=0, alpha=0.5, background_id=0
):
    """
    image_tensor: [3,H,W]
    mask: [H,W] int
    class_names: list[str], index = class id
    kept_ids: list[int] (includes background id)
    """
    H, W = mask.shape
    img_pil = to_pil_image(image_tensor.detach().cpu())

    num_classes = len(class_names)

    # Colors: background black, others distinct
    colors = _make_distinct_colors(num_classes - 1, seed=seed)
    colors = [(0,0,0)] + colors  # index 0 -> black
    cmap = ListedColormap(colors, name="distinct_classes")
    norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1), cmap.N)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img_pil); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(img_pil)
    axs[1].imshow(mask.cpu().numpy(), cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest")
    axs[1].set_title(f"Segmentation (top-{len(kept_ids)-1} + background)")
    axs[1].axis("off")

    # Legend only for present kept classes
    present = torch.unique(mask).cpu().tolist()
    present_kept = [i for i in present if i in kept_ids]
    patches = [mpatches.Patch(color=cmap.colors[i], label=class_names[i]) for i in present_kept]
    axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()



def main(args):
    start_time = time.time()

    experiment_name = f"{args.exp}_{args.n_attr}_{args.attr_type}_{args.score_type}"

    text_list, stats = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr, attr_type=args.attr_type)

    # dual prompts
    # text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    # single prompt
    text_list_prompt = [f"This is a photo of {item}." for item in text_list]

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = eval_clip.load(args.backbone, device=device)
    model.eval()

    # get image from path
    img = Image.open(args.img_pth).convert("RGB")
    image = preprocess(img).unsqueeze(0).to(device)   # [1, 3, 224, 224]
    preprocess2 = _transform_without_normalize(model.visual.input_resolution)
    original_image_to_plot = preprocess2(img)

    # path definition, creat directory if not exists
    img_name = os.path.basename(args.img_pth)
    save_dir = f"results_dir/{experiment_name}/local_patch_segmentation/{img_name}/"
    os.makedirs(save_dir, exist_ok=True)
    # results_save_path = f"{save_dir}/experiment_results_dict.pkl"

    model.eval()
    with torch.no_grad():
        text = eval_clip.tokenize(text_list_prompt).to(device)  # single prompt [80, 77] dual prompts [160, 77]
        text_features = model.encode_text(text)  # [80, 512/1024]

    with torch.no_grad():
        h, w = model.visual.input_resolution, model.visual.input_resolution
        if args.backbone == 'ViT-B/16':
            image_features, attn_weight_list = model.encode_image(image, h, w, attn_mask=1)   # image [1, 3, 224, 224] image features [1, 197, 512], attn_weight [1, 197, 197]
        elif 'RN' in args.backbone:
            image_features, attn_weight_list = model.encode_image(image)    # image [1, 3, 224, 224] image features [1, 1024, 50], attn_weight [1, 50, 50]
            image_features = image_features.transpose(1, 2)

    # No fusion: directly using local patch features to obtain class of local patch
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_patch = logit_scale * image_features @ text_features.t()
    # class_per_patch = logits_per_patch.squeeze().argmax(dim=1)[1:]
    # instead of using class label per patch, resize the logits to original image size and polt per pixel mask
    if args.backbone == 'ViT-B/16':
        grid = (14, 14)
    elif 'RN' in args.backbone:
        grid = (7, 7)
    mask, kept_ids, _ = logits_to_mask_keep_topk(
        logits_per_patch.squeeze()[1:], h, w, grid=grid,
        keep_k=5, background_id=0,
        select_mode="sum", small_obj_tau=0.7,
        reassign_to_kept=False  # set True to fallback to kept classes
    )

    # plot original and patch segmentation
    plot_mask_overlay(original_image_to_plot, mask, text_list, kept_ids, seed=0, alpha=0.5, background_id=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='coco', help='experiment type, e.g. celebA, FFHQ or AFHQ')
    parser.add_argument('--img_pth', type=str, default='/Users/yifanlu/Desktop/Datasets/coco/val2014/COCO_val2014_000000145512.jpg')
    parser.add_argument('--n_attr', type=int, default=80, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')
    parser.add_argument('--score_type', type=str, default='EvalCS', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')
    parser.add_argument('--backbone', type=str, default='ViT-B/16', help='backbone for CLIP, e.g. ViT-B/16 or RN50, RN101')

    args = parser.parse_args()
    main(args)