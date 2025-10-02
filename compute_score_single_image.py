import argparse
import os.path

import utils
import core
import torch
import eval_clip
from PIL import Image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./bedroom.png', help='path to image')
    parser.add_argument('--score_type', type=str, default='EvalCS_local_vit', help='score type, e.g. HCS for this script')

    args = parser.parse_args()

    text_list = ['male', 'heavy makeup', 'wearing earrings', 'black hair', 'blond hair', 'wearing eyeglasses',
                 'wearing hat', 'young', 'with beard', 'smiling', 'dog', 'cat', 'tiger', 'lion', 'bear',
                 'elephant', 'giraffe', 'zebra', 'wolf', 'fox', 'rabbit', 'squirrel', 'eagle', 'penguin', 'dolphin',
                 'whale', 'snake', 'turtle', 'frog', 'butterfly', 'sofa', 'chair', 'bed', 'table', 'closet',
                 'bookshelf', 'mirror', 'nightstand', 'carpet', 'television', 'refrigerator', 'washing machine',
                 'microwave oven', 'bowl', 'plate', 'cup', 'curtain', 'pillow', 'clock', 'computer']

    img_name = os.path.basename(args.img_path).split('.')[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'vit' in args.score_type:
        print(f"********** calculating EvalCS with local ViT **********")
        model, preprocess = eval_clip.load("ViT-B/16", device=device)
    elif 'rn' in args.score_type:
        print(f"********** calculating EvalCS with local RN **********")
        model, preprocess = eval_clip.load("RN101", device=device)
    model.eval()

    image = preprocess(Image.open(args.img_path)).unsqueeze(0).to(device)

    text_list_prompt = [f"This is a photo without {item}." for item in text_list] + [f"This is a photo of {item}." for item in text_list]
    text = eval_clip.tokenize(text_list_prompt).to(device)

    with torch.no_grad():
        if 'vit' in args.score_type:
            scores, scores_per_patch = model(image, text)  # images [1, 3, 224, 224]  text [2 * n_attr, 77]  scores [1, n_attr]  scores_per_patch [1, 196, n_attr]
        elif 'rn' in args.score_type:
            scores = model(image, text)

    scores = scores.cpu().numpy().squeeze()

    text_list_sort_by_scores = [text_list[j] for j in scores.argsort()[::-1]]
    scores[::-1].sort()
    core.plot_bar(scores, text_list_sort_by_scores, f'{img_name}_scores_bar.png', './')

