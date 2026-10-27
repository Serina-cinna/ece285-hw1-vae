"""Evaluate unconditional samples against the untouched original test split."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

from model import ConvVAE


class ChestXrayDataset(Dataset):
    def __init__(self, root, image_size):
        self.paths = sorted(p for p in Path(root).rglob("*")
                            if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
        if not self.paths:
            raise FileNotFoundError(f"No test images found under {root}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("L"))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root containing the untouched test folder")
    parser.add_argument("--output", default="outputs/rigorous_v1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output = Path(args.output)
    checkpoint = torch.load(output / "vae.pth", map_location="cpu", weights_only=True)
    model = ConvVAE(checkpoint["latent_dim"], checkpoint["image_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_set = ChestXrayDataset(Path(args.data) / "test", checkpoint["image_size"])
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    fid = FID(device=device)
    inception = InceptionScore(device=device)
    samples = []
    seen = 0

    for real in loader:
        count = real.size(0)
        real = real.to(device)
        latent = torch.randn(count, checkpoint["latent_dim"], device=device)
        fake = model.decode(latent).sigmoid()
        real_rgb = F.interpolate(real.repeat(1, 3, 1, 1), size=(299, 299), mode="bilinear", align_corners=False)
        fake_rgb = F.interpolate(fake.repeat(1, 3, 1, 1), size=(299, 299), mode="bilinear", align_corners=False)
        fid.update((fake_rgb, real_rgb))
        inception.update(fake_rgb)
        if len(samples) < 64:
            samples.extend(fake[:min(count, 64 - len(samples))].cpu())
        seen += count

    results = {"test_images": len(test_set), "images_per_distribution": seen,
               "inception_score": float(inception.compute()), "fid": float(fid.compute()),
               "metric": "PyTorch-Ignite Inception-v3", "seed": args.seed,
               "real_distribution": "original test split", "fid_uncertainty": "not estimated"}
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    grid = make_grid(torch.stack(samples), nrow=8, normalize=True, value_range=(0, 1), padding=2)
    plt.figure(figsize=(9, 9))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output / "generated_samples.png", dpi=180, bbox_inches="tight", pad_inches=0)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
