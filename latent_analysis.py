"""
Latent-space analysis: demonstrate that starting from noise far from the origin
of the standard Gaussian leads to degraded image quality.

For a well-trained DDPM/DDIM, the reverse process starts from x_T ~ N(0,I).
When we start from scaled noise x_T = scale * N(0,I), the model is being
evaluated out-of-distribution, producing blurry / artifact-ridden samples.

This script:
1. Loads the best checkpoint (defaults to cosine_T1000_ddim).
2. Generates samples starting from noise with various L2 norms (scales).
3. Computes IS and FID for each scale.
4. Saves image grids and a summary plot.

Usage
-----
python latent_analysis.py [--ckpt checkpoints/cosine_T1000_ddim_best.pt]
                          [--n_samples 500] [--device auto]
"""

import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from diffusion import DiffusionModel
from evaluate import compute_is, compute_fid, get_feature_extractor, load_real_images
from experiment_runner import load_diffusion, save_sample_grid


# ---------------------------------------------------------------------------
# Latent scale analysis
# ---------------------------------------------------------------------------

SCALES = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]   # multipliers for the initial noise


def run_latent_analysis(
    ckpt_path: str,
    n_samples: int = 500,
    batch_size: int = 128,
    ddim_steps: int = 50,
    device: str = "auto",
    results_dir: str = "./results",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[latent] Loading model from {ckpt_path}")
    diffusion = load_diffusion(ckpt_path, device)
    classifier = get_feature_extractor(device)
    real_images = load_real_images(n=n_samples, device="cpu")

    scale_results = []

    for scale in SCALES:
        print(f"\n[latent] Scale = {scale:.1f}")

        # Generate fixed base noise then scale it
        base_noise = torch.randn(n_samples, 1, 28, 28)
        # Normalise to unit norm per sample, then scale to desired norm
        norms = base_noise.view(n_samples, -1).norm(dim=1, keepdim=True)
        unit_noise = base_noise.view(n_samples, -1) / (norms + 1e-8)
        # Scale: ||x|| ~ scale * sqrt(28*28) to match the expected norm
        expected_norm = scale * (28 * 28) ** 0.5
        scaled_noise = (unit_noise * expected_norm).view(n_samples, 1, 28, 28)

        # Generate images in batches
        all_imgs = []
        remaining = n_samples
        offset = 0
        while remaining > 0:
            bs = min(batch_size, remaining)
            noise_batch = scaled_noise[offset: offset + bs].to(device)
            imgs = diffusion.sample_ddim(
                batch_size=bs, ddim_steps=ddim_steps, noise=noise_batch
            )
            all_imgs.append(imgs.cpu())
            offset += bs
            remaining -= bs
        fake_images = torch.cat(all_imgs, dim=0)

        # Save sample grid
        grid_path = os.path.join(results_dir, f"latent_scale_{scale:.0f}_samples.png")
        save_sample_grid(
            fake_images, grid_path, nrow=8,
            title=f"Noise scale = {scale:.1f} (expected ‖x_T‖ ≈ {expected_norm:.1f})"
        )

        # Compute metrics
        is_mean, is_std = compute_is(fake_images, classifier)
        fid = compute_fid(real_images, fake_images, classifier)
        print(f"  IS = {is_mean:.3f} ± {is_std:.3f}   FID = {fid:.3f}")

        scale_results.append({
            "scale": scale,
            "IS_mean": round(is_mean, 4),
            "IS_std": round(is_std, 4),
            "FID": round(fid, 4),
        })

    # ---- summary plot ----
    scales = [r["scale"] for r in scale_results]
    is_vals = [r["IS_mean"] for r in scale_results]
    fid_vals = [r["FID"] for r in scale_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(scales, is_vals, "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("Noise scale")
    ax1.set_ylabel("Inception Score (↑)")
    ax1.set_title("IS vs Noise Scale")
    ax1.axvline(x=1.0, color="grey", linestyle="--", label="Normal scale")
    ax1.legend()

    ax2.plot(scales, fid_vals, "o-", color="tomato", linewidth=2)
    ax2.set_xlabel("Noise scale")
    ax2.set_ylabel("FID (↓)")
    ax2.set_title("FID vs Noise Scale")
    ax2.axvline(x=1.0, color="grey", linestyle="--", label="Normal scale")
    ax2.legend()

    plt.suptitle("Effect of latent noise magnitude on generation quality", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "latent_scale_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[latent] Analysis plot saved → {plot_path}")

    return scale_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="./checkpoints/cosine_T1000_best.pt",
        help="Path to the best model checkpoint (e.g. cosine_T1000_best.pt)",
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    run_latent_analysis(
        ckpt_path=args.ckpt,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        device=args.device,
        results_dir=args.results_dir,
    )
