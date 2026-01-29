import argparse
import os.path

import core
import utils
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
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
import torch.nn.functional as F
import cv2
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform_without_normalize(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor()
    ])


def mask_attn(logits_coarse, logits, h, w, attn_weight):  # P_coarse, P^(hat)_refined, M_attn*A_l
    patch_size = 16
    candidate_cls_list = []
    logits_refined = logits.clone()  # [196, 80]

    logits_max = torch.max(logits, dim=0)[0]  # [80]

    for tempid, tempv in enumerate(logits_max):
        if tempv > 0:
            candidate_cls_list.append(tempid)  # P_local
    for ccls in candidate_cls_list:
        temp_logits = logits[:, ccls]  # [196]
        temp_logits = temp_logits - temp_logits.min()
        temp_logits = temp_logits / temp_logits.max()
        mask = temp_logits
        mask = mask.reshape(h // patch_size, w // patch_size)  # [14, 14]

        box, cnt = scoremap2bbox(mask.detach().cpu().numpy(), threshold=0.5, multi_contour_eval=True)
        aff_mask = torch.zeros((mask.shape[0], mask.shape[1])).to(logits_coarse.device)  # [14, 14]
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, mask.shape[0] * mask.shape[1])  # [1, 196]
        trans_mat = attn_weight * aff_mask  # [196, 196] * [1, 196] = [196, 196]    M_attn*A_l*M_cls
        logits_refined_ccls = torch.matmul(trans_mat, logits_coarse[:, ccls:ccls + 1])  # [196, 196] * [196, 1] = [196, 1]
        logits_refined[:, ccls] = logits_refined_ccls.squeeze()
    return logits_refined  # M_attn*A_l*M_cls*P_coarse


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)   # (14, 14, 1)
    _, thr_gray_heatmap = cv2.threshold(    # (14, 14)
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)

# ----------------- utilities -----------------
def _make_distinct_colors(n, seed=0, s=0.65, v=0.95):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    rng.shuffle(hues)
    return [colorsys.hsv_to_rgb(h, s, v) for h in hues]


def _pick_topk_classes_patch(
    patch_logits, patch_ids, grid, keep_k=5, background_id=0,
    select_mode="sum", small_obj_tau=0.7
):
    """
    Select top-k (non-background) classes at the PATCH level.
    One of patch_logits (Gx,Gy,C) or patch_ids (Gx,Gy) must be provided.

    select_mode ∈ {"mean","sum","max","argmax_freq"}:
      - with logits: compute softmax per patch → aggregate across patches
      - with ids only: uses "argmax_freq" (counts per class)
    Returns kept_ids (including background_id first).
    """
    Gx, Gy = grid

    if patch_logits is not None:
        # probs: [Gx,Gy,C]
        C = patch_logits.shape[-1]
        probs = torch.softmax(patch_logits, dim=-1)

        non_bg = [i for i in range(C) if i != background_id]

        if select_mode == "mean":
            scores = probs.mean(dim=(0,1))        # [C]
        elif select_mode == "sum":
            scores = probs.sum(dim=(0,1))         # [C]
        elif select_mode == "max":
            scores = probs.amax(dim=(0,1))        # [C]
        elif select_mode == "argmax_freq":
            ids = probs.argmax(dim=-1)            # [Gx,Gy]
            scores = torch.bincount(ids.view(-1), minlength=C).float()
        else:
            raise ValueError("select_mode must be one of: mean, sum, max, argmax_freq")

        base_scores = scores[non_bg]
        k = min(keep_k, len(non_bg))
        topk_vals, topk_idx = torch.topk(base_scores, k)
        kept = {non_bg[i.item()] for i in topk_idx}

        # small-object safeguard (based on per-patch max prob)
        max_probs = probs.amax(dim=(0,1))         # [C]
        high_conf = {i for i in non_bg if max_probs[i] >= small_obj_tau}

        merged = [background_id] + sorted((kept | high_conf), key=lambda i: scores[i], reverse=True)
        return merged[:min(len(merged), keep_k + 1)]

    else:
        # ids only
        assert patch_ids is not None, "Need either patch_logits or patch_ids."
        C = int(patch_ids.max().item()) + 1
        ids = patch_ids
        counts = torch.bincount(ids.view(-1), minlength=C).float()
        non_bg = [i for i in range(C) if i != background_id]
        base_scores = counts[non_bg]
        k = min(keep_k, len(non_bg))
        topk_vals, topk_idx = torch.topk(base_scores, k)
        kept = [background_id] + [non_bg[i.item()] for i in topk_idx]
        return kept


# ----------------- main: patch classification -> resized mask -----------------
def patch_classification_to_mask_keep_topk(
    image_tensor,
    patch_data,                 # logits: [Gx*Gy,C] or [Gx,Gy,C]  OR  ids: [Gx*Gy] or [Gx,Gy]
    class_names,
    grid=(14,14),
    background_id=0,
    keep_k=5,
    select_mode="sum",          # "sum","mean","max","argmax_freq"
    small_obj_tau=0.7,
    reassign_to_kept=False      # if logits given: fallback to kept classes before argmax
):
    """
    Returns:
      up_mask: [H,W] int tensor
      kept_ids: list[int] including background_id
    """
    Gx, Gy = grid
    H, W = image_tensor.shape[1:]

    # Normalize shapes & detect type
    if patch_data.dim() == 2 and patch_data.size(1) > 1:
        # [G*G, C] logits
        patch_logits = patch_data.view(Gx, Gy, -1).contiguous()
        patch_ids = None
    elif patch_data.dim() == 3 and patch_data.size(-1) > 1:
        # [Gx, Gy, C] logits
        patch_logits = patch_data.contiguous()
        patch_ids = None
    else:
        # ids: [G*G] or [Gx,Gy]
        patch_logits = None
        patch_ids = patch_data.view(Gx, Gy).contiguous()

    # Pick kept classes at PATCH scale
    kept_ids = _pick_topk_classes_patch(
        patch_logits, patch_ids, (Gx,Gy),
        keep_k=keep_k, background_id=background_id,
        select_mode=select_mode, small_obj_tau=small_obj_tau
    )
    kept_set = set(kept_ids)

    # Build patch-level mask (Gx,Gy)
    if patch_logits is not None:
        if reassign_to_kept:
            # Zero out logits of non-kept classes per patch, then argmax
            C = patch_logits.shape[-1]
            mask_logits = patch_logits.clone()
            not_kept = [i for i in range(C) if i not in kept_set]
            if len(not_kept) > 0:
                mask_logits[..., not_kept] = float('-inf')
            patch_mask = mask_logits.argmax(dim=-1)  # [Gx,Gy]
        else:
            # Argmax first, then force non-kept to background
            raw = patch_logits.argmax(dim=-1)        # [Gx,Gy]
            patch_mask = raw.clone()
            # force to background if not kept
            allowed = torch.tensor(list(kept_set), device=patch_mask.device)
            keep_bool = torch.isin(patch_mask, allowed)
            patch_mask[~keep_bool] = background_id
    else:
        # ids only
        patch_mask = patch_ids.clone()
        allowed = torch.tensor(list(kept_set), device=patch_mask.device)
        keep_bool = torch.isin(patch_mask, allowed)
        patch_mask[~keep_bool] = background_id

    # Upsample to image size with nearest (keeps discrete IDs)
    up_mask = F.interpolate(
        patch_mask[None,None].float(), size=(H,W), mode="nearest"
    )[0,0].long()

    return up_mask, kept_ids


# ----------------- plotting -----------------
def plot_patch_mask_overlay(
    image_tensor, up_mask, class_names, kept_ids, seed=0, alpha=0.5, background_id=0, save_dir=None
):
    """
    Black background, distinct colors for others, legend only for present kept classes.
    """
    img_pil = to_pil_image(image_tensor.detach().cpu())
    num_classes = len(class_names)

    colors = _make_distinct_colors(num_classes - 1, seed=seed)
    colors = [(0,0,0)] + colors  # class 0 -> black
    cmap = ListedColormap(colors, name="distinct_classes")
    norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1), cmap.N)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img_pil); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(img_pil)
    axs[1].imshow(up_mask.cpu().numpy(), cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest")
    axs[1].set_title(f"top-{len(kept_ids)-1} + background")
    axs[1].axis("off")

    present = torch.unique(up_mask).cpu().tolist()
    present_kept = [i for i in present if i in kept_ids]
    patches = [mpatches.Patch(color=cmap.colors[i], label=class_names[i]) for i in present_kept]
    axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/patch_scores.pdf")
    plt.show()


def main(args):
    start_time = time.time()

    experiment_name = f"{args.exp}_{args.n_attr}_{args.attr_type}"

    text_list = utils.load_candidate(experiment_name=experiment_name, n_attr=args.n_attr)

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
        text = eval_clip.tokenize(text_list_prompt).to(device)  # single prompt [22, 77] dual prompts [44, 77]
        text_features = model.encode_text(text)  # [44, 512]

    with torch.no_grad():
        h, w = model.visual.input_resolution, model.visual.input_resolution
        if args.backbone == 'ViT-B/16':
            image_features, attn_weight_list = model.encode_image(image, attn_mask=1)  # image [1, 3, 224, 224] image features [1, 197, 512], attn_weight [1, 197, 197]
            attn_weight = torch.stack(attn_weight_list[:-1], dim=0).mean(dim=0)
        elif 'RN' in args.backbone:
            image_features, attn_weight = model.encode_image(image)  # image [1, 3, 224, 224] image features [1, 1024, 50], attn_weight [1, 50, 50]
            image_features = image_features.transpose(1, 2)

    # No fusion: directly using local patch features to obtain class of local patch
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)  # [1, 197, 512]
    text_features = text_features / text_features.norm(dim=1, keepdim=True)   # [80, 512]

    logit_scale = model.logit_scale.exp()  # 100
    logits = logit_scale * image_features @ text_features.t()  # torch.Size([1, 197, 80])
    logits_per_patch = logits[:, 1:, :]  # [1, 196, 80]
    logits_per_patch = logits_per_patch.softmax(dim=-1)  # [1, 196, 80]

    if args.attn_corr:
        attn_weight_list_updated = [aw[:, 1:, 1:] for aw in attn_weight_list]  # len(attn_weight)=12 elements # shape [1, 196, 196]
        attn_weight = torch.stack(attn_weight_list_updated[:-1], dim=0)[:]  # [12, 1, 196, 196] pick the last three layers: [3, 1, 196, 196]
        attn_weight = torch.mean(attn_weight, dim=0)[0]  # [196, 196]
        if args.global_corr:
            global_attn_weight_list = [aw[:, :1, 1:] for aw in attn_weight_list]
            global_attn_weight = torch.stack(global_attn_weight_list, dim=0).mean(dim=0)[0] # [1, 196]
            global_attn_weight = global_attn_weight.repeat(attn_weight_list_updated[0].shape[1], 1)
            attn_weight = 0.8 * attn_weight + 0.2 * global_attn_weight

        if args.attn_mask:
            attn_vote = torch.stack(attn_weight_list_updated[:-1], dim=0).squeeze()  # [12, 196, 196]

            thres0 = attn_vote.reshape(attn_vote.shape[0], -1)  # [12, 38416=196*196]
            thres0 = torch.mean(thres0, dim=-1).reshape(attn_vote.shape[0], 1, 1)  # [12, 1, 1]
            thres0 = thres0.repeat(1, attn_vote.shape[1], attn_vote.shape[2])  # [12, 196, 196]

            attn_cnt = attn_vote > thres0  # [12, 196, 196]
            attn_cnt = attn_cnt.float()  # [12, 196, 196]
            attn_cnt = torch.sum(attn_cnt, dim=0)  # [196, 196]
            attn_cnt = attn_cnt >= 11  # [196, 196]  M_attn
            attn_weight = attn_weight * attn_cnt.float()  # [196, 196]  M_attn * A_l

        logits_coarse = logits_per_patch.squeeze()  # [196, 80]   P_coarse
        logits_per_patch = torch.matmul(attn_weight, logits_per_patch)  # [1, 196, 80]    M_attn * A_l * P_coarse
        logits_per_patch = logits_per_patch.squeeze()  # [196, 80]
        if args.cls_mask:
            logits_per_patch = mask_attn(logits_coarse, logits_per_patch, h, w, attn_weight)  # [196, 80] P_refined = M_attn * A_l * M_cls * P_coarse

    # # cosine similarity as logits
    # logit_scale = model.logit_scale.exp()
    # logits_per_patch = logit_scale * image_features @ text_features.t()
    # if args.attn_corr:
    #     logits_per_patch = logits_per_patch.softmax(dim=-1)
    #     logits_per_patch = torch.matmul(attn_weight, logits_per_patch)
    # class_per_patch = logits_per_patch.squeeze().argmax(dim=1)[1:]

    if args.backbone == 'ViT-B/16':
        grid = (14, 14)
    elif 'RN' in args.backbone:
        grid = (7, 7)
    # plot original and patch segmentation
    # up_mask, kept_ids = patch_classification_to_mask_keep_topk(
    #     original_image_to_plot,
    #     logits_per_patch.squeeze()[1:],                # or patch_ids
    #     text_list,
    #     grid=grid,
    #     background_id=0,
    #     keep_k=5,
    #     select_mode="sum",
    #     small_obj_tau=0.7,
    #     reassign_to_kept=False    # if logits available, fallback among kept classes
    # )

    up_mask, kept_ids = patch_classification_to_mask_keep_topk(
        original_image_to_plot,
        logits_per_patch.squeeze(),  # or patch_ids
        text_list,
        grid=grid,
        background_id=0,
        keep_k=4,
        select_mode="mean",
        small_obj_tau=0.7,
        reassign_to_kept=False  # if logits available, fallback among kept classes
    )

    plot_patch_mask_overlay(original_image_to_plot, up_mask, text_list, kept_ids, seed=0, alpha=0.5, background_id=0, save_dir=save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='coco', help='experiment type, e.g. celebA, FFHQ or AFHQ')
    parser.add_argument('--img_pth', type=str, default='/Users/yifanlu/Desktop/val2014/COCO_val2014_000000020485.jpg') # COCO_val2014_000000145512.jpg
    parser.add_argument('--n_attr', type=int, default=4, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')
    parser.add_argument('--score_type', type=str, default='EvalCS', help='score type, e.g. CS, HCS, dualCS, scaledHCS or simplifiedHCS')
    parser.add_argument('--backbone', type=str, default='ViT-B/16',help='backbone for CLIP, e.g. ViT-B/16 or RN50, RN101')
    parser.add_argument('--attn_corr', action='store_true', help='whether or not using attention weights to correct logits')
    parser.add_argument('--global_corr', action='store_true', help='whether or not using attention weights to correct logits')
    parser.add_argument('--attn_mask', action='store_true', help='whether or not using attention weights to correct logits')
    parser.add_argument('--cls_mask', action='store_true', help='whether or not using attention weights to correct logits')

    args = parser.parse_args()
    main(args)