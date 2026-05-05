"""
Evaluation utilities: Inception Score (IS) and Fréchet Inception Distance (FID).

Both metrics are computed using a LeNet-style classifier trained on MNIST
(or the real InceptionV3 when images are upscaled to ≥75×75).

For MNIST we use a lightweight CNN trained on the dataset itself as the
"inception" feature extractor, following common practice for small-scale
benchmarks.

Usage
-----
from evaluate import compute_is, compute_fid, get_feature_extractor
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature extractor (LeNet-style classifier trained on MNIST)
# ---------------------------------------------------------------------------

class MNISTClassifier(nn.Module):
    """Small CNN that outputs a 10-class logit vector and intermediate features."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                           # 14×14
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                           # 7×7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
        )
        self.head = nn.Linear(256, 10)

    def forward(self, x, return_features: bool = False):
        feat = self.fc(self.features(x))
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits


_CLASSIFIER_CACHE: MNISTClassifier | None = None
_CLASSIFIER_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "mnist_cls.pt")


def get_feature_extractor(device: torch.device) -> MNISTClassifier:
    """Return a trained MNIST classifier (trains once and caches the weights)."""
    global _CLASSIFIER_CACHE
    if _CLASSIFIER_CACHE is not None:
        return _CLASSIFIER_CACHE.to(device)

    model = MNISTClassifier().to(device)
    os.makedirs(os.path.dirname(_CLASSIFIER_CKPT), exist_ok=True)

    if os.path.exists(_CLASSIFIER_CKPT):
        model.load_state_dict(torch.load(_CLASSIFIER_CKPT, map_location=device))
        model.eval()
        _CLASSIFIER_CACHE = model
        return model

    print("[evaluate] Training MNIST classifier for FID/IS …")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=256, shuffle=True, num_workers=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  cls epoch {epoch+1}/5")
    torch.save(model.state_dict(), _CLASSIFIER_CKPT)
    model.eval()
    _CLASSIFIER_CACHE = model
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_probs_and_feats(images: torch.Tensor, classifier: MNISTClassifier):
    """
    Parameters
    ----------
    images : (N, 1, 28, 28) tensor in [-1, 1]

    Returns
    -------
    probs  : (N, 10)  softmax probabilities
    feats  : (N, 256) penultimate features
    """
    logits, feats = classifier(images, return_features=True)
    return logits.softmax(dim=-1), feats


def _collect_probs_feats(
    images: torch.Tensor,
    classifier: MNISTClassifier,
    batch_size: int = 256,
):
    all_probs, all_feats = [], []
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size].to(next(classifier.parameters()).device)
        p, f = _get_probs_and_feats(batch, classifier)
        all_probs.append(p.cpu())
        all_feats.append(f.cpu())
    return torch.cat(all_probs), torch.cat(all_feats)


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------

def compute_is(
    images: torch.Tensor,
    classifier: MNISTClassifier,
    splits: int = 10,
    batch_size: int = 256,
) -> tuple[float, float]:
    """
    Compute Inception Score (mean ± std over `splits` splits).

    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] )

    Parameters
    ----------
    images : (N, 1, 28, 28) generated images in [-1, 1]

    Returns
    -------
    (mean_IS, std_IS)
    """
    classifier.eval()
    probs, _ = _collect_probs_feats(images, classifier, batch_size)

    scores = []
    chunk = len(probs) // splits
    for k in range(splits):
        p_yx = probs[k * chunk: (k + 1) * chunk]           # (chunk, C)
        p_y = p_yx.mean(dim=0, keepdim=True)                # (1, C)
        kl = (p_yx * (p_yx.log() - p_y.log())).sum(dim=1)  # (chunk,)
        scores.append(kl.mean().exp().item())

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Fréchet Inception Distance
# ---------------------------------------------------------------------------

def _matrix_sqrt_np(A: np.ndarray) -> np.ndarray:
    """Numerically stable matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Fréchet distance between two Gaussians N(mu1,Σ1) and N(mu2,Σ2)."""
    diff = mu1 - mu2
    # product of covariance matrices
    covmean = _matrix_sqrt_np(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return (
        float(diff @ diff)
        + float(np.trace(sigma1 + sigma2 - 2 * covmean))
    )


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    classifier: MNISTClassifier,
    batch_size: int = 256,
) -> float:
    """
    Compute FID between real and generated images.

    Parameters
    ----------
    real_images : (N, 1, 28, 28) in [-1, 1]
    fake_images : (M, 1, 28, 28) in [-1, 1]

    Returns
    -------
    FID score (lower is better)
    """
    classifier.eval()
    _, real_feats = _collect_probs_feats(real_images, classifier, batch_size)
    _, fake_feats = _collect_probs_feats(fake_images, classifier, batch_size)

    mu_r = real_feats.numpy().mean(axis=0)
    mu_f = fake_feats.numpy().mean(axis=0)
    sigma_r = np.cov(real_feats.numpy(), rowvar=False)
    sigma_f = np.cov(fake_feats.numpy(), rowvar=False)

    return _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)


# ---------------------------------------------------------------------------
# Convenience: load real MNIST images
# ---------------------------------------------------------------------------

def load_real_images(n: int = 5000, device: str = "cpu") -> torch.Tensor:
    """Return the first `n` MNIST test images as a (N, 1, 28, 28) tensor in [-1,1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    images, _ = next(iter(loader))
    return images.to(device)
