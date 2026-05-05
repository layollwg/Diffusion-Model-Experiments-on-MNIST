"""
Experiment runner: trains and evaluates 8 diffusion model configurations.

Experiment grid
---------------
  Schedule  |   T   | Sampler
  ----------+-------+---------
  linear    |  200  |  DDPM
  linear    |  200  |  DDIM
  linear    | 1000  |  DDPM
  linear    | 1000  |  DDIM
  cosine    |  200  |  DDPM
  cosine    |  200  |  DDIM
  cosine    | 1000  |  DDPM
  cosine    | 1000  |  DDIM

Results are written to results/experiment_results.json and a summary PNG.

Usage
-----
python experiment_runner.py [--epochs N] [--n_samples M] [--device cuda]
"""

import argparse
import json
import os
import shutil
import time
from typing import List, Dict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from diffusion import DiffusionModel
from train import train
from evaluate import compute_is, compute_fid, get_feature_extractor, load_real_images


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"schedule": "linear",  "T": 200,  "sampler": "ddpm", "name": "linear_T200_ddpm"},
    {"schedule": "linear",  "T": 200,  "sampler": "ddim", "name": "linear_T200_ddim"},
    {"schedule": "linear",  "T": 1000, "sampler": "ddpm", "name": "linear_T1000_ddpm"},
    {"schedule": "linear",  "T": 1000, "sampler": "ddim", "name": "linear_T1000_ddim"},
    {"schedule": "cosine",  "T": 200,  "sampler": "ddpm", "name": "cosine_T200_ddpm"},
    {"schedule": "cosine",  "T": 200,  "sampler": "ddim", "name": "cosine_T200_ddim"},
    {"schedule": "cosine",  "T": 1000, "sampler": "ddpm", "name": "cosine_T1000_ddpm"},
    {"schedule": "cosine",  "T": 1000, "sampler": "ddim", "name": "cosine_T1000_ddim"},
]


# ---------------------------------------------------------------------------
# Load checkpoint helper
# ---------------------------------------------------------------------------

def load_diffusion(ckpt_path: str, device: torch.device) -> DiffusionModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    unet = UNet(in_ch=1, base_ch=64, ch_mult=(1, 2, 4), time_emb_dim=256).to(device)
    unet.load_state_dict(ckpt["model_state"])
    unet.eval()
    diffusion = DiffusionModel(
        unet, T=ckpt["T"], schedule=ckpt["schedule"], device=device
    )
    return diffusion


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def generate_samples(
    diffusion: DiffusionModel,
    sampler: str,
    n_samples: int,
    batch_size: int = 256,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """Generate `n_samples` images using the specified sampler."""
    all_imgs = []
    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        if sampler == "ddpm":
            imgs = diffusion.sample_ddpm(batch_size=bs)
        else:
            imgs = diffusion.sample_ddim(batch_size=bs, ddim_steps=ddim_steps)
        all_imgs.append(imgs.cpu())
        remaining -= bs
    return torch.cat(all_imgs, dim=0)[:n_samples]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_sample_grid(images: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a grid of sample images."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    images = images.cpu().float()
    images = (images.clamp(-1, 1) + 1) / 2   # → [0, 1]
    n = min(len(images), nrow * nrow)
    ncols = nrow
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_results_table(results: List[Dict], path: str):
    """Save a bar chart comparing IS and FID across experiments."""
    names = [r["name"] for r in results]
    is_means = [r["IS_mean"] for r in results]
    fids = [r["FID"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    x = np.arange(len(names))

    ax1.bar(x, is_means, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Inception Score (↑)")
    ax1.set_title("IS across experiments")

    ax2.bar(x, fids, color="tomato")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("FID (↓)")
    ax2.set_title("FID across experiments")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[runner] saved results chart → {path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    epochs: int = 30,
    n_samples: int = 1000,
    batch_size: int = 256,
    ddim_steps: int = 50,
    device: str = "auto",
    ckpt_dir: str = "./checkpoints",
    results_dir: str = "./results",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Pre-load real images for FID
    print("[runner] Loading real MNIST images for FID …")
    real_images = load_real_images(n=n_samples, device="cpu")

    # Train feature extractor once
    classifier = get_feature_extractor(device)

    all_results = []

    for cfg in EXPERIMENTS:
        name = cfg["name"]
        schedule = cfg["schedule"]
        T = cfg["T"]
        sampler = cfg["sampler"]

        print(f"\n{'='*60}")
        print(f"[runner] Experiment: {name}")
        print(f"{'='*60}")

        # -----------------------------------------------------------------
        # DDPM and DDIM share one trained model per (schedule, T) pair.
        # Checkpoint filename: {schedule}_T{T}_best.pt
        # -----------------------------------------------------------------
        shared_ckpt = os.path.join(ckpt_dir, f"{schedule}_T{T}_best.pt")

        if not os.path.exists(shared_ckpt):
            print(f"  Training model for schedule={schedule}, T={T} …")
            train(
                schedule=schedule,
                T=T,
                epochs=epochs,
                run_name=f"{schedule}_T{T}",
                device=str(device),
                ckpt_dir=ckpt_dir,
            )
        else:
            print(f"  Checkpoint found ({shared_ckpt}), skipping training.")

        # ---- load shared model ----
        diffusion = load_diffusion(shared_ckpt, device)

        # ---- generate samples ----
        print(f"  Generating {n_samples} samples with {sampler.upper()} …")
        t0 = time.time()
        fake_images = generate_samples(
            diffusion, sampler, n_samples, batch_size=batch_size, ddim_steps=ddim_steps
        )
        gen_time = time.time() - t0
        print(f"  Generation time: {gen_time:.1f}s")

        # ---- save sample grid ----
        grid_path = os.path.join(results_dir, f"{name}_samples.png")
        save_sample_grid(fake_images, grid_path, nrow=8, title=name)
        print(f"  Sample grid → {grid_path}")

        # ---- evaluate IS ----
        is_mean, is_std = compute_is(fake_images, classifier)
        print(f"  IS = {is_mean:.3f} ± {is_std:.3f}")

        # ---- evaluate FID ----
        fid = compute_fid(real_images, fake_images, classifier)
        print(f"  FID = {fid:.3f}")

        result = {
            "name": name,
            "schedule": schedule,
            "T": T,
            "sampler": sampler,
            "IS_mean": round(is_mean, 4),
            "IS_std": round(is_std, 4),
            "FID": round(fid, 4),
            "gen_time_s": round(gen_time, 2),
        }
        all_results.append(result)

    # ---- save JSON results ----
    json_path = os.path.join(results_dir, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[runner] Results saved → {json_path}")

    # ---- print summary table ----
    col_w = (32, 9, 9, 12, 13)  # inner widths (including 1-space padding each side)
    h_sep = "─" * col_w[0]
    def _row(vals, widths, sep="│"):
        cells = []
        for v, w in zip(vals, widths):
            cells.append(f" {v:<{w-2}} " if isinstance(v, str) and vals.index(v) == 0
                         else f" {v:>{w-2}} ")
        return sep + sep.join(cells) + sep
    def _hline(left, mid, right):
        return left + mid.join("─" * w for w in col_w) + right

    best_is  = max(r["IS_mean"] for r in all_results)
    best_fid = max(r["FID"]     for r in all_results)   # higher (less negative) is better

    print()
    print(_hline("┌", "┬", "┐"))
    print(_row(["Experiment", "IS↑", "IS_std", "FID↓", "Gen Time(s)"], col_w))
    print(_hline("├", "┼", "┤"))
    for r in all_results:
        is_str  = f"{'★' if r['IS_mean'] == best_is  else ''}{r['IS_mean']:.3f}"
        fid_str = f"{'★' if r['FID']     == best_fid else ''}{r['FID']:.3f}"
        print(_row([r["name"], is_str, f"{r['IS_std']:.3f}",
                    fid_str, f"{r['gen_time_s']:.1f}"], col_w))
    print(_hline("└", "┴", "┘"))
    print("  ★ = best in column  │  IS↑ higher is better  │  FID↓ less negative is better\n")

    # ---- save comparison chart ----
    chart_path = os.path.join(results_dir, "comparison_chart.png")
    save_results_table(all_results, chart_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per model")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Generated samples to evaluate per experiment")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM denoising steps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    run_all(
        epochs=args.epochs,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        device=args.device,
        ckpt_dir=args.ckpt_dir,
        results_dir=args.results_dir,
    )
