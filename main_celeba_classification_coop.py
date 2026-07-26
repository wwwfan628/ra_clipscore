import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_clip import evalcoop
from train_eval_coop import _transform
from utils import AttrDataSet


ATTRS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Male',
        'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Sideburns', 'Smiling',
        'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
        'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
COOP_TEXT = ['arched eyebrows', 'bags under eyes', 'bald', 'bangs', 'big nose',
        'black hair', 'blond hair', 'brown hair', 'chubby', 'double chin',
        'eyeglasses', 'goatee', 'gray hair', 'heavy makeup', 'male',
        'mouth slightly open', 'mustache', 'no beard', 'sideburns', 'smiling',
        'straight hair', 'wavy hair', 'wearing earrings', 'wearing hat',
        'wearing lipstick', 'wearing necklace', 'wearing necktie', 'young']  # same order/text as train_eval_coop.py


def load_coop_scores(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = evalcoop(COOP_TEXT)
    state_dict = torch.load(args.checkpoint, map_location=device)

    # train_eval_coop.py may save from either a normal model or DataParallel.
    is_parallel = isinstance(model, torch.nn.DataParallel)
    has_module_prefix = next(iter(state_dict)).startswith("module.")
    if has_module_prefix and not is_parallel:
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    elif is_parallel and not has_module_prefix:
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    image_encoder = model.module.image_encoder if is_parallel else model.image_encoder
    dataset = AttrDataSet(
        args.img_dir,
        args.anno_file,
        ATTRS,
        _transform(image_encoder.input_resolution),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    scores, labels = [], []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="EvalCoOp"):
            scores.append(model(images.to(device)).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def evaluate(scores, labels):
    labels = (labels == 1).astype(int)
    count_positive = labels.sum(axis=0)
    predictions = np.zeros_like(labels)
    for j, n_positive in enumerate(count_positive):
        topk = np.argsort(scores[:, j])[::-1][:n_positive]
        predictions[topk, j] = 1

    attr_acc = (predictions == labels).mean(axis=0)
    attr_f1 = np.array([
        f1_score(labels[:, j], predictions[:, j], zero_division=0)
        for j in range(len(ATTRS))
    ])
    attr_auroc = np.array([
        roc_auc_score(labels[:, j], scores[:, j])
        if np.unique(labels[:, j]).size == 2 else np.nan
        for j in range(len(ATTRS))
    ])

    results = []
    for j, attr in enumerate(ATTRS):
        results.append({
            "attribute": attr,
            "n_positive": count_positive[j],
            "acc": attr_acc[j],
            "f1": attr_f1[j],
            "auroc": attr_auroc[j],
        })
    results.append({
        "attribute": "Mean",
        "n_positive": '',
        "acc": attr_acc.mean(),
        "f1": attr_f1.mean(),
        "auroc": np.nanmean(attr_auroc),
    })
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    return results, macro_f1, micro_f1


def main(args):
    start = time.time()
    scores, labels = load_coop_scores(args)
    results, macro_f1, micro_f1 = evaluate(scores, labels)
    table = pd.DataFrame(results).set_index("attribute")
    print("\n" + table.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"\nMicro F1: {micro_f1:.4f}")
    print("Execution Time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Img')
    parser.add_argument("--anno_file", type=str, default='/home/x_yiflu/dataset/Datasets/Multiclasses_Classification/CelebA50k/Anno/list_attr_celeba50k.txt')
    parser.add_argument("--checkpoint", default="checkpoint.pth")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    main(parser.parse_args())
