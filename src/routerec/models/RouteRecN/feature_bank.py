"""Shared lightweight feature bank for RouteRecN."""

from __future__ import annotations

import math
from fnmatch import fnmatch
from typing import Sequence

import torch
import torch.nn as nn


_DEFAULT_SINUSOIDAL_PATTERNS = [
    "*time*",
    "*gap*",
    "*int*",
    "*pop*",
    "*valid_r*",
]


def _to_ratio(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values

    x = values.float()
    finite = torch.isfinite(x)
    if not finite.any():
        return torch.zeros_like(x)

    xf = x[finite]
    lo = xf.min()
    hi = xf.max()
    if lo >= -1e-6 and hi <= 1.0 + 1e-6:
        ratio = x.clamp(0.0, 1.0)
        ratio[~finite] = 0.5
        return ratio

    ratio = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    ratio = ratio.clamp(0.0, 1.0)
    ratio[~finite] = 0.5
    return ratio


def _matches_pattern(name: str, patterns: Sequence[str]) -> bool:
    lname = str(name).lower().strip()
    for pattern in patterns:
        p = str(pattern).lower().strip()
        if not p:
            continue
        if any(ch in p for ch in "*?[]"):
            if fnmatch(lname, p):
                return True
        elif p in lname:
            return True
    return False


class SharedFeatureBank(nn.Module):
    """Transform raw scalar features once and share the bank across router/experts."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        mode: str = "linear",
        sinusoidal_patterns: Sequence[str] | None = None,
        sinusoidal_n_freqs: int = 4,
    ):
        super().__init__()
        self.feature_names = [str(name) for name in feature_names]
        self.mode = str(mode).lower().strip()
        if self.mode not in {"linear", "sinusoidal_selected"}:
            raise ValueError(
                "feature_encoder_mode must be one of ['linear','sinusoidal_selected'], "
                f"got {mode}"
            )

        self.sinusoidal_patterns = [
            str(p) for p in (sinusoidal_patterns or _DEFAULT_SINUSOIDAL_PATTERNS)
        ]
        self.sinusoidal_n_freqs = max(int(sinusoidal_n_freqs), 0)

        if self.mode == "linear":
            self.bank_dim = 1
        else:
            self.bank_dim = 1 + 2 * self.sinusoidal_n_freqs

        mask = [
            1.0 if _matches_pattern(name, self.sinusoidal_patterns) else 0.0
            for name in self.feature_names
        ]
        self.register_buffer(
            "sinusoidal_mask",
            torch.tensor(mask, dtype=torch.float32),
            persistent=False,
        )
        if self.sinusoidal_n_freqs > 0:
            freqs = torch.tensor(
                [float(2 ** i) for i in range(self.sinusoidal_n_freqs)],
                dtype=torch.float32,
            )
        else:
            freqs = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("frequencies", freqs, persistent=False)

    def config_snapshot(self) -> dict:
        return {
            "feature_encoder_mode": self.mode,
            "feature_encoder_bank_dim": int(self.bank_dim),
            "feature_encoder_sinusoidal_features": list(self.sinusoidal_patterns),
            "feature_encoder_sinusoidal_n_freqs": int(self.sinusoidal_n_freqs),
            "feature_encoder_sinusoidal_feature_count": int(self.sinusoidal_mask.sum().item()),
        }

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        ratio = _to_ratio(feat)
        if self.mode == "linear":
            return ratio.unsqueeze(-1)

        out = ratio.new_zeros(*ratio.shape, self.bank_dim)
        out[..., 0] = ratio
        if self.sinusoidal_n_freqs <= 0:
            return out

        freq_shape = [1] * ratio.dim() + [self.sinusoidal_n_freqs]
        angles = ratio.unsqueeze(-1) * self.frequencies.view(*freq_shape) * (2.0 * math.pi)
        mask_shape = [1] * (ratio.dim() - 1) + [ratio.size(-1), 1]
        mask = self.sinusoidal_mask.view(*mask_shape).to(dtype=ratio.dtype)

        sin = torch.sin(angles) * mask
        cos = torch.cos(angles) * mask
        out[..., 1 : 1 + self.sinusoidal_n_freqs] = sin
        out[..., 1 + self.sinusoidal_n_freqs :] = cos
        return out
