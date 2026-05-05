"""
Diffusion core: noise schedules, DDPM forward/reverse, DDIM reverse.

Noise schedules
---------------
* linear  – Β linearly spaced between β_start and β_end (Ho et al. 2020)
* cosine  – α̅ cosine schedule              (Nichol & Dhariwal 2021)
* quadratic – Β quadratically spaced        (common variant)

Samplers
--------
* DDPM – stochastic reverse diffusion (Ho et al. 2020)
* DDIM – deterministic reverse diffusion (Song et al. 2021)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Literal


# ---------------------------------------------------------------------------
# Noise schedule factory
# ---------------------------------------------------------------------------

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008):
    """Cosine schedule as in Nichol & Dhariwal 2021."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def quadratic_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2


SCHEDULE_REGISTRY = {
    "linear": linear_beta_schedule,
    "cosine": cosine_beta_schedule,
    "quadratic": quadratic_beta_schedule,
}


# ---------------------------------------------------------------------------
# Diffusion model wrapper
# ---------------------------------------------------------------------------

class DiffusionModel(nn.Module):
    """
    Wraps a denoising U-Net with diffusion utilities.

    Parameters
    ----------
    model        : U-Net (or any ε-predictor) with signature (x, t) -> ε
    T            : total diffusion time steps
    schedule     : noise schedule name ("linear", "cosine", "quadratic")
    device       : torch device
    """

    def __init__(
        self,
        model: nn.Module,
        T: int = 1000,
        schedule: Literal["linear", "cosine", "quadratic"] = "linear",
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.T = T
        self.device = torch.device(device)

        # ----- precompute schedule tensors -----
        betas = SCHEDULE_REGISTRY[schedule](T).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]]
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        # posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             posterior_variance.clamp(min=1e-20).log())

    # ------------------------------------------------------------------
    # Forward process: q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Sample x_t ~ q(x_t | x_0) using the reparameterisation trick."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    # ------------------------------------------------------------------
    # Training loss  (simple noise-prediction loss)
    # ------------------------------------------------------------------

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor = None):
        """
        Returns the simple L2 denoising loss.

        Parameters
        ----------
        x0 : (B, 1, H, W) clean images in [-1, 1]
        t  : (B,) optional time steps; sampled uniformly if None
        """
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (B,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_noisy, _ = self.q_sample(x0, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return ((noise - predicted_noise) ** 2).mean()

    # ------------------------------------------------------------------
    # Reverse process: DDPM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_ddpm(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """One DDPM reverse step: x_{t-1} ~ p(x_{t-1} | x_t)."""
        t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        eps = self.model(x, t_batch)

        # compute mean of p(x_{t-1} | x_t)
        betas_t = self._extract(self.betas, t_batch, x.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
        sqrt_recip_alpha = (1.0 / self.alphas.sqrt())
        sqrt_recip_alpha_t = self._extract(sqrt_recip_alpha, t_batch, x.shape)

        model_mean = sqrt_recip_alpha_t * (x - betas_t / sqrt_one_minus_t * eps)

        if t == 0:
            return model_mean
        else:
            posterior_var = self._extract(self.posterior_variance, t_batch, x.shape)
            z = torch.randn_like(x)
            return model_mean + posterior_var.sqrt() * z

    @torch.no_grad()
    def sample_ddpm(self, batch_size: int = 16, img_shape=(1, 28, 28),
                    noise: torch.Tensor = None) -> torch.Tensor:
        """Full DDPM sampling loop (T → 0)."""
        if noise is None:
            x = torch.randn(batch_size, *img_shape, device=self.device)
        else:
            x = noise.to(self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample_ddpm(x, t)
        return x

    # ------------------------------------------------------------------
    # Reverse process: DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int = 16,
        img_shape=(1, 28, 28),
        ddim_steps: int = 50,
        eta: float = 0.0,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al. 2021).

        Parameters
        ----------
        ddim_steps : number of denoising steps (≤ T)
        eta        : stochasticity (0 = fully deterministic DDIM)
        noise      : optional starting noise; sampled if None
        """
        device = self.device
        # evenly spaced subset of time steps
        c = self.T // ddim_steps
        timesteps = list(reversed(range(0, self.T, c)))[:ddim_steps]

        if noise is None:
            x = torch.randn(batch_size, *img_shape, device=device)
        else:
            x = noise.to(device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps = self.model(x, t_batch)

            alpha_bar_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
                alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev_batch, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            # predicted x0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            # direction pointing to x_t
            sigma = (
                eta
                * ((1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt()
                * (1 - alpha_bar_t / alpha_bar_prev).sqrt()
            )
            dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            noise_t = sigma * torch.randn_like(x)

            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt + noise_t

        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Gather values from `a` at indices `t` and broadcast to `shape`."""
        out = a.gather(0, t)
        while out.dim() < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)
