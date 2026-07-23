import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = VARIANT_ROOT.parents[1]
VAE_ROOT = VARIANT_ROOT / "CSUA_LDM_VAE_NoUnc"
VAE_CODE_ROOT = VAE_ROOT / "CSUA_LDM_VAE_NoUnc_Code"

for search_path in (CURRENT_DIR, PROJECT_ROOT, VARIANT_ROOT, VAE_ROOT, VAE_CODE_ROOT):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import random

import torch

from CSUA_LDM_NoUnc_Config import *
from CSUA_LDM_Utils import ensure_batched_4d


def _dual_branch_lr_up_probability(epoch_idx: int) -> float:
    if not bool(CSUA_LDM_DIFFUSION_NOUNC_DUAL_BRANCH_ENABLED):
        return 1.0
    epoch_idx = int(max(epoch_idx, 0))
    if epoch_idx < int(CSUA_LDM_DIFFUSION_NOUNC_DUAL_BRANCH_WARMUP_EPOCHS):
        return 0.0
    ramp_epochs = max(int(CSUA_LDM_DIFFUSION_NOUNC_DUAL_BRANCH_RAMP_EPOCHS), 1)
    ramp_progress = min(
        max(
            (epoch_idx + 1 - int(CSUA_LDM_DIFFUSION_NOUNC_DUAL_BRANCH_WARMUP_EPOCHS)) / float(ramp_epochs),
            0.0,
        ),
        1.0,
    )
    return float(max(0.0, min(1.0, float(CSUA_LDM_DIFFUSION_NOUNC_DUAL_BRANCH_MAX_LR_UP_PROB)))) * ramp_progress


def select_dual_branch_source(epoch_idx: int | None) -> str:
    if epoch_idx is None or (not bool(CSUA_LDM_DIFFUSION_NOUNC_DUAL_BRANCH_ENABLED)):
        return "lr_up"
    return "lr_up" if random.random() < _dual_branch_lr_up_probability(epoch_idx) else "hr_down_up"


def cosine_beta_scheduler(
    timesteps: int = 1000,
    min_beta: float = 0.0001,
    max_beta: float = 0.999,
    s: float = 0.008,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    timesteps = int(timesteps)
    if timesteps <= 0:
        raise ValueError(f"`timesteps` must be > 0, got {timesteps}.")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps=steps, device=device, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + float(s)) / (1.0 + float(s)) * torch.pi * 0.5).pow(2)
    alphas_cumprod = alphas_cumprod / torch.clamp(alphas_cumprod[0], min=1e-20)
    betas = 1.0 - (alphas_cumprod[1:] / torch.clamp(alphas_cumprod[:-1], min=1e-20))
    return torch.clamp(betas, min=float(min_beta), max=float(max_beta)).to(dtype=dtype)


def extract(v: torch.Tensor, i: torch.Tensor, shape) -> torch.Tensor:
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)
    return out.view([i.shape[0]] + [1] * (len(shape) - 1))
