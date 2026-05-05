"""
Training script for the MNIST diffusion model.

Usage
-----
python train.py --schedule linear --T 1000 --epochs 30 --run_name linear_T1000
"""

import argparse
import os
import platform
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import UNet
from diffusion import DiffusionModel

# On Windows, multiprocessing uses "spawn" which can conflict with DataLoader
# workers; use 0 workers to avoid the issue.
_NUM_WORKERS = 0 if platform.system() == "Windows" else 2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_mnist_loader(batch_size: int = 128, train: bool = True, data_root: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # map to [-1, 1]
    ])
    dataset = datasets.MNIST(data_root, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=_NUM_WORKERS, pin_memory=True, drop_last=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    schedule: str = "linear",
    T: int = 1000,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-4,
    run_name: str = "run",
    device: str = "auto",
    ckpt_dir: str = "./checkpoints",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train] device={device}  schedule={schedule}  T={T}  epochs={epochs}")

    os.makedirs(ckpt_dir, exist_ok=True)

    loader = get_mnist_loader(batch_size=batch_size)

    unet = UNet(in_ch=1, base_ch=64, ch_mult=(1, 2, 4), time_emb_dim=256).to(device)
    diffusion = DiffusionModel(unet, T=T, schedule=schedule, device=device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        unet.train()
        total_loss = 0.0
        for x, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = x.to(device)
            loss = diffusion.p_losses(x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": unet.state_dict(),
                    "schedule": schedule,
                    "T": T,
                    "loss": best_loss,
                },
                ckpt_path,
            )

    # also save final checkpoint
    final_path = os.path.join(ckpt_dir, f"{run_name}_final.pt")
    torch.save(
        {
            "epoch": epochs,
            "model_state": unet.state_dict(),
            "schedule": schedule,
            "T": T,
            "loss": avg_loss,
        },
        final_path,
    )
    print(f"[train] saved final checkpoint → {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default="linear",
                        choices=["linear", "cosine", "quadratic"])
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    args = parser.parse_args()

    train(
        schedule=args.schedule,
        T=args.T,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.run_name,
        device=args.device,
        ckpt_dir=args.ckpt_dir,
    )
