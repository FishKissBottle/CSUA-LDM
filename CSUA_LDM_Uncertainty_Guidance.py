"""Utilities for uncertainty-guided sampling maps used by DDPM samplers."""

import torch


def _sanitize_uncertainty_tensor(uncertainty: torch.Tensor) -> torch.Tensor:
    uncertainty = torch.nan_to_num(uncertainty, nan=0.0, posinf=1e6, neginf=0.0)
    return torch.clamp(uncertainty, min=0.0, max=1e6)


def normalize_uncertainty_map(
    uncertainty: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    uncertainty = _sanitize_uncertainty_tensor(uncertainty)
    flat = uncertainty.float().reshape(uncertainty.shape[0], -1)
    unc_min = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    unc_max = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    normalized = (uncertainty - unc_min) / torch.clamp(unc_max - unc_min, min=eps)

    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(normalized, min=0.0, max=1.0).to(dtype=uncertainty.dtype)


def map_uncertainty_to_reliability(
    normalized_uncertainty: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    alpha = float(max(alpha, 1e-6))
    reliability = 1.0 / (1.0 + alpha * normalized_uncertainty)

    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(reliability, min=0.0, max=1.0)


def build_guidance_maps(
    uncertainty_source_map: torch.Tensor,
    reliability_alpha: float,
    noise_gate_min: float,
    noise_gate_max: float,
) -> dict[str, torch.Tensor]:
    noise_gate_min = float(max(0.0, min(noise_gate_min, noise_gate_max)))
    noise_gate_max = float(max(noise_gate_min, noise_gate_max))

    uncertainty_norm = normalize_uncertainty_map(uncertainty_source_map)
    reliability = map_uncertainty_to_reliability(
        uncertainty_norm,
        alpha=reliability_alpha,
    )

    noise_gate = noise_gate_max - reliability * (noise_gate_max - noise_gate_min)

    return {
        "uncertainty_source": uncertainty_source_map,
        "uncertainty_norm": uncertainty_norm,
        "reliability": reliability,
        "noise_gate": torch.clamp(noise_gate, min=noise_gate_min, max=noise_gate_max),
    }
