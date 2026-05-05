# Diffusion Model Experiments on MNIST

> **HA3 — Deep Learning | HKUST(GZ)**
> Systematic experiments with DDPM / DDIM on MNIST across different noise schedules and time-step counts, evaluated with Inception Score (IS) and Fréchet Inception Distance (FID), plus a latent-space out-of-distribution analysis.

---

## Repository structure

```
model.py              U-Net denoising backbone with sinusoidal time embeddings
diffusion.py          Noise schedules (linear / cosine / quadratic) + DDPM & DDIM samplers
train.py              Single-model training script (CLI)
evaluate.py           IS and FID evaluation utilities (LeNet feature extractor)
experiment_runner.py  Runs all 8 experiments end-to-end and writes results/
latent_analysis.py    Latent-space noise-scale analysis (OOD demonstration)
requirements.txt      Python dependencies

checkpoints/          Saved model weights (created at runtime)
results/              Generated images + metric charts (created at runtime)
data/                 MNIST dataset (downloaded automatically on first run)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.8 and PyTorch ≥ 2.0 required. A CUDA GPU is strongly recommended.

---

## Quick start — run all 8 experiments

```bash
python experiment_runner.py --epochs 30 --n_samples 1000 --device cuda
```

After completion, `results/` contains:
- `<name>_samples.png`   — 8×8 image grids for every experiment
- `comparison_chart.png` — IS and FID bar charts across all 8 experiments
- `experiment_results.json` — full numeric results

### Experiment grid

| # | Schedule | T    | Sampler | Shared checkpoint |
|---|----------|------|---------|-------------------|
| 1 | linear   |  200 | DDPM    | `linear_T200_best.pt` |
| 2 | linear   |  200 | DDIM    | `linear_T200_best.pt` |
| 3 | linear   | 1000 | DDPM    | `linear_T1000_best.pt` |
| 4 | linear   | 1000 | DDIM    | `linear_T1000_best.pt` |
| 5 | cosine   |  200 | DDPM    | `cosine_T200_best.pt` |
| 6 | cosine   |  200 | DDIM    | `cosine_T200_best.pt` |
| 7 | cosine   | 1000 | DDPM    | `cosine_T1000_best.pt` |
| 8 | cosine   | 1000 | DDIM    | `cosine_T1000_best.pt` |

DDPM and DDIM share the same trained weights; only the reverse-sampling algorithm differs. This means **4 training runs** cover all 8 experiments.

---

## Latent-space analysis

Demonstrates that starting from noise far outside the unit Gaussian degrades quality:

```bash
python latent_analysis.py \
    --ckpt  checkpoints/cosine_T1000_best.pt \
    --n_samples 500 \
    --device cuda
```

Outputs in `results/`:
- `latent_scale_<N>_samples.png` — grids at each noise scale
- `latent_scale_analysis.png`    — IS and FID vs noise scale

---

## Train a single model

```bash
python train.py \
    --schedule cosine \
    --T 1000 \
    --epochs 30 \
    --run_name cosine_T1000 \
    --device cuda
```

Available schedules: `linear`, `cosine`, `quadratic`

---

## Evaluation metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| **IS** (Inception Score) | ↑ higher | Sharpness and diversity of generated images, measured via a LeNet classifier trained on MNIST |
| **FID** (Fréchet Inception Distance) | ↓ lower | Distributional distance between real and generated image features |

> Because MNIST (28×28 greyscale) is too small for standard InceptionV3, both metrics use a lightweight CNN feature extractor trained on MNIST itself — a standard practice for small-scale benchmarks.

---

## Model architecture

The denoising backbone is a compact U-Net:

- **Input:** 28×28×1 noisy image + scalar time step
- **Time conditioning:** sinusoidal positional embedding → 2-layer MLP → injected into every residual block
- **Encoder:** 3 × (ResBlock × 2 + stride-2 conv)
- **Bottleneck:** 2 × ResBlock
- **Decoder:** 3 × (ConvTranspose + skip concat + ResBlock × 2)
- **Output:** predicted noise ε of same shape as input

Channel widths: 64 → 128 → 256 (base 64, multipliers ×1 ×2 ×4).

---

## Noise schedules

| Name | Formula | Reference |
|------|---------|-----------|
| **Linear** | β_t = β_min + (β_max − β_min)·t/T | Ho et al., 2020 |
| **Cosine** | ᾱ_t = cos²(((t/T + s)/(1+s))·π/2) | Nichol & Dhariwal, 2021 |
| **Quadratic** | β_t = (√β_min + (√β_max − √β_min)·t/T)² | Common variant |

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
2. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR*.
3. Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.
