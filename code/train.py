"""Train a chest X-ray VAE with a fixed train/validation split."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from model import ConvVAE, vae_loss


class ChestXrayDataset(Dataset):
    def __init__(self, root, image_size=128):
        root = Path(root)
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("L"))


def evaluate_loss(model, loader, device):
    model.eval()
    sums = np.zeros(3, dtype=np.float64)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            logits, mean, log_variance = model(batch)
            loss, reconstruction, kl = vae_loss(logits, batch, mean, log_variance)
            sums += [loss.item(), reconstruction.item(), kl.item()]
    return (sums / len(loader.dataset)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root containing the original train folder")
    parser.add_argument("--output", default="outputs/rigorous_v1")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_root = Path(args.data) / "train"
    dataset = ChestXrayDataset(train_root, args.image_size)
    validation_size = max(1, round(0.1 * len(dataset)))
    training_size = len(dataset) - validation_size
    split_generator = torch.Generator().manual_seed(args.split_seed)
    training_set, validation_set = random_split(
        dataset, [training_size, validation_size], generator=split_generator
    )
    workers = min(6, os.cpu_count() or 6) if device.type == "cuda" else 0
    loader_args = {"batch_size": args.batch_size, "num_workers": workers,
                   "pin_memory": device.type == "cuda", "persistent_workers": workers > 0}
    train_loader = DataLoader(training_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    model = ConvVAE(args.latent_dim, args.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5
    )
    history = {"train_total": [], "train_reconstruction": [], "train_kl": [],
               "validation_total": [], "validation_reconstruction": [], "validation_kl": [],
               "learning_rate": []}
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 0
    last_checkpoint = output / "last_checkpoint.pth"
    if args.resume and last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = saved["history"]
        start_epoch = saved["epoch"]
        best_epoch = int(np.argmin(history["validation_total"])) + 1
        best_validation = min(history["validation_total"])
        stale_epochs = start_epoch - best_epoch
        if saved.get("torch_rng") is not None:
            torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        if saved.get("numpy_rng") is not None:
            np.random.set_state(saved["numpy_rng"])
        if saved.get("python_rng") is not None:
            random.setstate(saved["python_rng"])
        print(f"Resuming from epoch {start_epoch}; best validation epoch={best_epoch}")
    start = time.time()
    print(f"Device={device}; GPU={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")
    print(f"Images: train={training_size}, validation={validation_size}, original_test=624")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        sums = np.zeros(3, dtype=np.float64)
        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits, mean, log_variance = model(batch)
            loss, reconstruction, kl = vae_loss(logits, batch, mean, log_variance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            sums += [loss.item(), reconstruction.item(), kl.item()]
            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch:03d}/{args.epochs} batch {step:03d}/{len(train_loader)}", flush=True)

        train_values = (sums / training_size).tolist()
        validation_values = evaluate_loss(model, validation_loader, device)
        for key, value in zip(("train_total", "train_reconstruction", "train_kl"), train_values):
            history[key].append(value)
        for key, value in zip(("validation_total", "validation_reconstruction", "validation_kl"), validation_values):
            history[key].append(value)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        scheduler.step(validation_values[0])
        print(f"Epoch {epoch:03d}: train={train_values[0]:.2f} val={validation_values[0]:.2f} "
              f"(recon={validation_values[1]:.2f}, kl={validation_values[2]:.2f}) "
              f"lr={optimizer.param_groups[0]['lr']:.2g}", flush=True)

        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "history": history,
                      "epoch": epoch, "seed": args.seed,
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                      "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        torch.save(checkpoint, last_checkpoint)
        if validation_values[0] < best_validation:
            best_validation = validation_values[0]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "latent_dim": args.latent_dim,
                        "image_size": args.image_size, "best_epoch": epoch,
                        "validation_loss": best_validation}, output / "best_model.pth")
        else:
            stale_epochs += 1
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation epoch was {best_epoch}.", flush=True)
            break

    best = torch.load(output / "best_model.pth", map_location="cpu", weights_only=True)
    torch.save({"model": best["model"], "latent_dim": args.latent_dim,
                "image_size": args.image_size, "best_epoch": best_epoch}, output / "vae.pth")
    history.update({"epochs_run": len(history["train_total"]), "best_epoch": best_epoch,
                    "best_validation_loss": best_validation, "train_images": training_size,
                    "validation_images": validation_size, "seconds": round(time.time() - start, 1),
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "seed": args.seed, "split_seed": args.split_seed,
                    "image_size": args.image_size, "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate, "latent_dim": args.latent_dim,
                    "split": "90/10 random split of original train folder; test folder held out"})
    (output / "history.json").write_text(json.dumps(history, indent=2))

    plt.figure(figsize=(7, 4))
    for key, label in (("train_total", "Train ELBO"), ("validation_total", "Validation ELBO")):
        plt.plot(range(1, len(history[key]) + 1), history[key], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO per image")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "loss_curve.png", dpi=180)
    print(f"Saved best checkpoint from epoch {best_epoch} to {output}", flush=True)


if __name__ == "__main__":
    main()
