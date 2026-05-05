"""
Small U-Net denoising model for 28×28 greyscale MNIST images.

Architecture
------------
* Encoder: three down-sampling blocks (Conv + GroupNorm + SiLU)
* Bottleneck: two residual blocks
* Decoder: three up-sampling blocks with skip connections
* Time-step conditioning via sinusoidal embeddings projected into every block
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time-step embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Continuous sinusoidal embedding for scalar time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) float tensor of time steps in [0, T-1]

        Returns
        -------
        (B, dim) embedding tensor
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]           # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with time-step conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Lightweight U-Net for MNIST (1×28×28).

    Parameters
    ----------
    base_ch    : base channel count (default 64)
    ch_mult    : channel multipliers for each scale
    time_emb_dim : dimension of sinusoidal time embedding
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        ch_mult: tuple = (1, 2, 4),
        time_emb_dim: int = 256,
    ):
        super().__init__()

        # ---------- time embedding ----------
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        channels = [base_ch * m for m in ch_mult]   # e.g. [64, 128, 256]

        # ---------- encoder ----------
        self.enc_in = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = channels[0]
        for ch in channels:
            self.enc_blocks.append(
                nn.ModuleList([
                    ResBlock(prev_ch, ch, time_emb_dim),
                    ResBlock(ch, ch, time_emb_dim),
                ])
            )
            self.downsamples.append(Downsample(ch))
            prev_ch = ch

        # ---------- bottleneck ----------
        self.mid1 = ResBlock(prev_ch, prev_ch, time_emb_dim)
        self.mid2 = ResBlock(prev_ch, prev_ch, time_emb_dim)

        # ---------- decoder ----------
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for ch in reversed(channels):
            self.upsamples.append(Upsample(prev_ch))
            # skip connection doubles the channel count
            self.dec_blocks.append(
                nn.ModuleList([
                    ResBlock(prev_ch + ch, ch, time_emb_dim),
                    ResBlock(ch, ch, time_emb_dim),
                ])
            )
            prev_ch = ch

        # ---------- output ----------
        self.out = nn.Sequential(
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, in_ch, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 28, 28) noisy image
        t : (B,)            integer time steps

        Returns
        -------
        (B, 1, 28, 28) predicted noise
        """
        t_emb = self.time_emb(t)   # (B, time_emb_dim)

        # encoder
        x = self.enc_in(x)
        skips = []
        for (r1, r2), down in zip(self.enc_blocks, self.downsamples):
            x = r1(x, t_emb)
            x = r2(x, t_emb)
            skips.append(x)
            x = down(x)

        # bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # decoder
        for (r1, r2), up, skip in zip(self.dec_blocks, self.upsamples, reversed(skips)):
            x = up(x)
            # handle any size mismatch from odd spatial dims
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = r1(x, t_emb)
            x = r2(x, t_emb)

        return self.out(x)
