import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import os
import re
import natsort
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torch.utils.data import random_split, DataLoader
import argparse
from eval_clip import evalcoop
from utils.attribute_list import *
import numpy as np


class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_file='list_attr_celeba.txt', img_dir='Img', transform=None, selected_attrs=None):
        """
        root_dir: path to CelebA root directory
        attr_file: file with attributes
        img_dir: subdirectory with images
        transform: torchvision transforms
        selected_attrs: list of attribute names to use (None = all 40)
        """
        self.root_dir = root_dir
        self.attr_path = os.path.join(root_dir, attr_file)
        self.img_dir = os.path.join(root_dir, img_dir)
        self.transform = transform

        r = re.compile('[0-9]{6}.jpg')

        # Load attributes
        df = pd.read_csv(self.attr_path, header=[0], delimiter=r"\s+")
        df = df.replace(-1, 0)  # Convert -1 to 0 for binary classification

        all_imgs = os.listdir(self.img_dir)
        self.filenames = natsort.natsorted(all_imgs)
        if selected_attrs is not None:
            self.attr_names = selected_attrs
        else:
            self.attr_names = df.columns.tolist()
        self.labels = [df[self.attr_names].loc[f].values.astype('float32') if r.match(f) else
                       np.zeros(len(self.attr_names)).astype('float32')
                       for f in self.filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, save_path='checkpoint.pth',num_epochs=10):
    model.to(device)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # Multi-label requires float targets

            outputs = model(inputs)
            print(outputs.device)
            print(labels.device)
            print(inputs.device)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        val_loss, metrics = validate_model(model, val_loader, loss_fn, device)

        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {metrics['accuracy']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1']:.4f} | mAP: {metrics['mAP']:.4f}")
        print("-" * 80)


def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            probs = model(inputs)
            loss = loss_fn(probs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = (probs > 0.5).int()
            targets = labels.cpu().int()

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='samples', zero_division=0)
    mAP = average_precision_score(all_targets, all_probs, average='samples')

    avg_loss = running_loss / len(val_loader.dataset)

    return avg_loss, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP': mAP
    }


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



def main(args):

    experiment_name = f"{args.exp}_{args.n_attr}_{args.attr_type}"

    text_list = attr_dict[experiment_name][:args.n_attr]

    model = evalcoop(text_list)

    selected_attrs = ['Male', 'Smiling', 'Eyeglasses']  # choose what you need

    transform = _transform(model.image_encoder.input_resolution)

    # Create full dataset
    dataset = CelebADataset(
        root_dir=args.dataset_dir,
        selected_attrs=selected_attrs,
        transform=transform
    )

    # Define split sizes (e.g., 80% train, 20% val)
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    # Random split
    generator = torch.Generator().manual_seed(32)  # fix seed
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.BCELoss()
    try:
        optimizer = torch.optim.Adam(model.prompt_params(), lr=1e-2)
    except:
        optimizer = torch.optim.SGD(model.module.prompt_params(), lr=1e-2)

    train_model(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs=50)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Train EvalCoop', description='Script to train EvalCoop')

    parser.add_argument('--exp', type=str, default='celebA', help='experiment type, e.g. FFHQ or AFHQ')
    parser.add_argument('--dataset_dir', type=str, default='/Users/yifanlu/Desktop/Datasets/face1k_furniture1k')
    parser.add_argument('--n_attr', type=int, default=3, help='number of attributes to calculate SaD and PaD')
    parser.add_argument('--attr_type', type=str, default='USER', help='attribute type, e.g. BLIP or USER')

    args = parser.parse_args()
    main(args)