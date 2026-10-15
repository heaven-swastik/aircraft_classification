"""Predict the manufacturer of one or more aircraft photos with the trained ResNet-18.

    python predict.py --weights best_model.pt photo1.jpg photo2.jpg

Photos that come straight from FGVC-Aircraft have a 20 px copyright strip at the bottom,
which I removed during training. For those images add --crop-bottom 20.
"""
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, meta["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    idx_to_class = {i: name for name, i in meta["class_to_idx"].items()}
    return model, preprocess, idx_to_class


@torch.inference_mode()
def predict(path, model, preprocess, idx_to_class, device, topk=3, crop_bottom=0):
    img = Image.open(path).convert("RGB")
    if crop_bottom:
        w, h = img.size
        img = img.crop((0, 0, w, h - crop_bottom))
    probs = torch.softmax(model(preprocess(img).unsqueeze(0).to(device)), dim=1)[0].cpu()
    top_p, top_i = probs.topk(topk)
    return [(idx_to_class[int(i)], float(p)) for p, i in zip(top_p, top_i)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to best_model.pt")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--crop-bottom", type=int, default=0, help="pixels to cut off the bottom (20 for FGVC-Aircraft originals)")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, idx_to_class = load_model(args.weights, device)
    for path in args.images:
        print(f"\n{path}")
        for name, p in predict(path, model, preprocess, idx_to_class, device, args.topk, args.crop_bottom):
            print(f"  {name:25s} {p:6.1%}")


if __name__ == "__main__":
    main()
