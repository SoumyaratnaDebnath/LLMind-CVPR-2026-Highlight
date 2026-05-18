from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_budget_size(h: int, w: int, p: float) -> Tuple[int, int]:
    p = float(max(min(p, 1.0), 1e-6))
    target = h * w * p
    scale = math.sqrt(target / float(h * w))
    hs = max(1, int(round(h * scale)))
    ws = max(1, int(round(w * scale)))
    return hs, ws


def uniform_sample(
    frame: torch.Tensor,
    p: float,
    mode: str = "grid",
    antialias: bool = True,
    up_method: str = "bilinear",
) -> torch.Tensor:
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    if not frame.is_floating_point():
        frame = frame.float()

    b, c, h, w = frame.shape
    hs, ws = compute_budget_size(h, w, p)

    if hs == h and ws == w:
        out = frame.clone()
        return out.squeeze(0) if squeeze_back else out

    if mode not in ("grid", "random", "sparse"):
        raise ValueError("mode must be 'grid', 'random', or 'sparse'")

    if mode == "sparse":
        budget = max(1, int(round(p * h * w)))
        idx = torch.linspace(0, h * w - 1, steps=budget, device=frame.device)
        idx = torch.round(idx).long()
        idx = torch.unique(idx, sorted=True)
        if idx.numel() < budget:
            all_idx = torch.arange(h * w, device=frame.device)
            mask = torch.ones(h * w, device=frame.device, dtype=torch.bool)
            mask[idx] = False
            fill = all_idx[mask][: (budget - idx.numel())]
            idx = torch.cat([idx, fill], dim=0)
        y = idx // w
        x = idx % w
        mask = torch.zeros((b, 1, h, w), device=frame.device, dtype=frame.dtype)
        mask[:, 0, y, x] = 1.0
        out = frame * mask
        return out.squeeze(0) if squeeze_back else out

    if mode == "grid":
        down = F.interpolate(
            frame, size=(hs, ws), mode="bilinear", align_corners=False, antialias=antialias
        )
    else:
        ys = torch.linspace(-1.0, 1.0, hs, device=frame.device, dtype=frame.dtype)
        xs = torch.linspace(-1.0, 1.0, ws, device=frame.device, dtype=frame.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        jitter_y = (torch.rand_like(yy) - 0.5) * (2.0 / max(h, 1))
        jitter_x = (torch.rand_like(xx) - 0.5) * (2.0 / max(w, 1))
        grid = torch.stack([xx + jitter_x, yy + jitter_y], dim=-1)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        down = F.grid_sample(frame, grid, mode="bilinear", align_corners=False)

    if up_method == "nearest":
        up = F.interpolate(down, size=(h, w), mode="nearest")
    else:
        up = F.interpolate(down, size=(h, w), mode=up_method, align_corners=False)

    return up.squeeze(0) if squeeze_back else up
