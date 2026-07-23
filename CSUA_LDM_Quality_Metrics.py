import math
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from piqa import GMSD, FSIM, SSIM

try:
    import lpips as _lpips_backend
except ImportError as exc:
    _lpips_backend = None
    _LPIPS_IMPORT_ERROR = exc
else:
    _LPIPS_IMPORT_ERROR = None

# Module-level cache for LPIPS metric instances to avoid repeated construction.
_LPIPS_CACHE = {}
_LPIPS_WARNING_EMITTED = False


def _warn_lpips_unavailable_once():
    global _LPIPS_WARNING_EMITTED
    if _LPIPS_WARNING_EMITTED:
        return
    warning_text = (
        "LPIPS backend `lpips` is not installed. LPIPS metrics will be reported as N/A."
    )
    if _LPIPS_IMPORT_ERROR is not None:
        warning_text += f" Original import error: {_LPIPS_IMPORT_ERROR}"
    warnings.warn(warning_text, RuntimeWarning, stacklevel=2)
    _LPIPS_WARNING_EMITTED = True


def _get_lpips_metric(device: torch.device, net: str = "alex") -> torch.nn.Module | None:
    """Return a cached LPIPS metric instance on the requested device."""
    if _lpips_backend is None:
        _warn_lpips_unavailable_once()
        return None

    key = (str(device), net)
    if key not in _LPIPS_CACHE:
        _LPIPS_CACHE[key] = _lpips_backend.LPIPS(net=net).to(device).eval()
    return _LPIPS_CACHE[key]


def format_metric_value(value, digits: int = 6, na_text: str = "N/A") -> str:
    """Format a scalar metric value, mapping None/NaN/Inf to a display placeholder."""
    if value is None:
        return na_text
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            value = value.mean()
        value = float(value.detach().float().cpu().item())
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return na_text

    if not math.isfinite(value):
        return na_text
    return f"{value:.{int(digits)}f}"


def _validate_pair_basic(recon_imgs: torch.Tensor, true_imgs: torch.Tensor):
    """Validate common shape constraints for metric inputs."""
    if recon_imgs.dim() != 4 or true_imgs.dim() != 4:
        raise ValueError(
            f"Expected 4D tensors (B, C, H, W), got {tuple(recon_imgs.shape)} and {tuple(true_imgs.shape)}."
        )
    if recon_imgs.shape != true_imgs.shape:
        raise ValueError(
            f"Input shape mismatch: recon={tuple(recon_imgs.shape)}, true={tuple(true_imgs.shape)}."
        )


def _validate_3_or_4_channels(recon_imgs: torch.Tensor):
    """Validate channel count for metrics implemented as RGB or RGB+NIR."""
    c = recon_imgs.shape[1]
    if c not in (3, 4):
        raise ValueError(f"Only 3-channel or 4-channel inputs are supported, got C={c}.")


def _resolve_rgb_indices(
    recon_imgs: torch.Tensor,
    rgb_indices=None,
    is_BGR: bool | None = None,
) -> list[int]:
    """Resolve an explicitly supplied RGB evaluation order."""
    if rgb_indices is not None:
        resolved = [int(idx) for idx in rgb_indices]
    elif is_BGR is not None:
        resolved = [2, 1, 0] if bool(is_BGR) else [0, 1, 2]
    else:
        raise ValueError("Provide `rgb_indices` or explicitly set `is_BGR`.")

    if len(resolved) != 3:
        raise ValueError(f"`rgb_indices` must contain exactly 3 channels, got {resolved}.")

    c = int(recon_imgs.shape[1])
    if min(resolved) < 0 or max(resolved) >= c:
        raise ValueError(f"`rgb_indices`={resolved} is invalid for an input with {c} channels.")
    return resolved


def _prepare_metric_pair(recon_imgs: torch.Tensor, true_imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate metric inputs and compute metrics in float32.

    AMP inference can leave reconstructions in float16, while piqa metric
    kernels are float32 by default. Normalizing the dtype here keeps all
    metrics numerically stable and avoids dtype mismatches inside conv2d.
    """
    _validate_pair_basic(recon_imgs, true_imgs)
    return recon_imgs.to(dtype=torch.float32), true_imgs.to(device=recon_imgs.device, dtype=torch.float32)


def _mean_metric_per_sample(metric_output: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reduce a metric output to one scalar per sample.

    Many third-party metric implementations return tensors shaped like
    ``(B,)`` or ``(B, 1, 1, 1)`` when ``reduction='none'`` is used. This
    helper normalizes those outputs to a stable per-sample vector so the final
    dataset average is invariant to microbatch partitioning.
    """
    if not isinstance(metric_output, torch.Tensor):
        metric_output = torch.as_tensor(metric_output)

    metric_output = metric_output.to(dtype=torch.float32)
    if metric_output.ndim == 0:
        if int(batch_size) != 1:
            raise ValueError(
                f"Expected a batch-aware metric output for batch_size={batch_size}, got a scalar tensor."
            )
        return metric_output.reshape(1)

    if int(metric_output.shape[0]) != int(batch_size):
        if int(metric_output.numel()) == int(batch_size):
            return metric_output.reshape(batch_size).to(dtype=torch.float32)
        raise ValueError(
            f"Metric output shape {tuple(metric_output.shape)} is incompatible with batch_size={batch_size}."
        )

    return metric_output.reshape(batch_size, -1).mean(dim=1)


def compute_PSNR(recon_imgs: torch.Tensor, true_imgs: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8):
    """Compute sample-wise PSNR and return the batch mean.

    Args:
        recon_imgs: Reconstructed image tensor in [0, 1], shape (B, C, H, W).
        true_imgs: Reference image tensor in [0, 1], shape (B, C, H, W).
        max_val: Maximum signal value.
        eps: Numerical lower bound for MSE.

    Returns:
        Scalar tensor of mean PSNR (dB) over the batch.

    Notes:
        Computing PSNR per sample and then averaging keeps the reported value
        invariant to evaluation microbatch partitioning, which is more suitable
        for dataset-level metric aggregation than computing a single PSNR from
        the whole batch MSE.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)

    mse_per_sample = torch.mean((recon_imgs - true_imgs) ** 2, dim=(1, 2, 3))
    mse_per_sample = torch.clamp(mse_per_sample, min=eps)
    psnr_per_sample = 10 * torch.log10((max_val ** 2) / mse_per_sample)
    return psnr_per_sample.mean()


def compute_SSIM(recon_imgs: torch.Tensor, true_imgs: torch.Tensor):
    """Compute sample-wise SSIM and return the batch mean.

    Args:
        recon_imgs: Reconstructed image tensor in [0, 1], shape (B, C, H, W).
        true_imgs: Reference image tensor in [0, 1], shape (B, C, H, W).

    Returns:
        Scalar tensor of mean SSIM over the batch.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)
    _validate_3_or_4_channels(recon_imgs)

    n_channels = recon_imgs.shape[1]
    ssim_metric = SSIM(n_channels=n_channels, reduction="none").to(recon_imgs.device)
    ssim_per_sample = _mean_metric_per_sample(
        ssim_metric(recon_imgs, true_imgs),
        batch_size=int(recon_imgs.shape[0]),
    )
    return ssim_per_sample.mean()


def compute_GMSD(
    recon_imgs: torch.Tensor,
    true_imgs: torch.Tensor,
    rgb_indices=None,
    *,
    is_BGR: bool | None = None,
):
    """Compute GMSD for RGB branch and optional NIR branch.

    For 4-channel inputs, NIR is evaluated by repeating channel-3 to 3 channels.

    Args:
        recon_imgs: Reconstructed image tensor in [0, 1], shape (B, C, H, W).
        true_imgs: Reference image tensor in [0, 1], shape (B, C, H, W).
        rgb_indices: Explicit RGB channel order, e.g. [2, 1, 0]. Required unless `is_BGR` is set.
        is_BGR: Optional BGR-order flag. Ignored if `rgb_indices` is set.

    Returns:
        Tuple:
            rgb_gmsd_val: Scalar tensor of mean RGB-branch GMSD over the batch.
            nir_gmsd_val: Scalar tensor of mean NIR-branch GMSD for C=4, otherwise None.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)
    _validate_3_or_4_channels(recon_imgs)

    gmsd_metric = GMSD(reduction="none").to(recon_imgs.device)
    rgb_indices = _resolve_rgb_indices(recon_imgs, rgb_indices=rgb_indices, is_BGR=is_BGR)
    rgb_gmsd_val = _mean_metric_per_sample(
        gmsd_metric(recon_imgs[:, rgb_indices, :, :], true_imgs[:, rgb_indices, :, :]),
        batch_size=int(recon_imgs.shape[0]),
    ).mean()

    if recon_imgs.shape[1] == 3:
        nir_gmsd_val = None
    else:
        nir_gmsd_val = _mean_metric_per_sample(
            gmsd_metric(recon_imgs[:, [3, 3, 3], :, :], true_imgs[:, [3, 3, 3], :, :]),
            batch_size=int(recon_imgs.shape[0]),
        ).mean()

    return rgb_gmsd_val, nir_gmsd_val


def compute_FSIM(
    recon_imgs: torch.Tensor,
    true_imgs: torch.Tensor,
    rgb_indices=None,
    *,
    is_BGR: bool | None = None,
):
    """Compute FSIM for RGB branch and optional NIR branch.

    For 4-channel inputs, NIR is evaluated by repeating channel-3 to 3 channels.
    The RGB branch uses chromatic FSIM, and the NIR branch uses grayscale FSIM.

    Args:
        recon_imgs: Reconstructed image tensor in [0, 1], shape (B, C, H, W).
        true_imgs: Reference image tensor in [0, 1], shape (B, C, H, W).
        rgb_indices: Explicit RGB channel order, e.g. [2, 1, 0]. Required unless `is_BGR` is set.
        is_BGR: Optional BGR-order flag. Ignored if `rgb_indices` is set.

    Returns:
        Tuple:
            rgb_fsim_val: Scalar tensor of mean RGB-branch FSIM over the batch.
            nir_fsim_val: Scalar tensor of mean NIR-branch FSIM for C=4, otherwise None.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)
    _validate_3_or_4_channels(recon_imgs)

    rgb_indices = _resolve_rgb_indices(recon_imgs, rgb_indices=rgb_indices, is_BGR=is_BGR)
    fsim_rgb_metric = FSIM(chromatic=True, reduction="none").to(recon_imgs.device)
    rgb_fsim_val = _mean_metric_per_sample(
        fsim_rgb_metric(recon_imgs[:, rgb_indices, :, :], true_imgs[:, rgb_indices, :, :]),
        batch_size=int(recon_imgs.shape[0]),
    ).mean()

    if recon_imgs.shape[1] == 3:
        nir_fsim_val = None
    else:
        fsim_nir_metric = FSIM(chromatic=False, reduction="none").to(recon_imgs.device)
        nir_fsim_val = _mean_metric_per_sample(
            fsim_nir_metric(recon_imgs[:, [3, 3, 3], :, :], true_imgs[:, [3, 3, 3], :, :]),
            batch_size=int(recon_imgs.shape[0]),
        ).mean()

    return rgb_fsim_val, nir_fsim_val


def compute_LPIPS(
    recon_imgs: torch.Tensor,
    true_imgs: torch.Tensor,
    rgb_indices=None,
    *,
    is_BGR: bool | None = None,
    net: str = "alex",
):
    """Compute LPIPS for RGB branch and optional NIR branch.

    For 4-channel inputs, NIR is evaluated by repeating channel-3 to 3 channels.
    The LPIPS implementation expects inputs in [-1, 1], so the function performs
    the [0, 1] -> [-1, 1] scaling internally.

    Args:
        recon_imgs: Reconstructed image tensor in [0, 1], shape (B, C, H, W).
        true_imgs: Reference image tensor in [0, 1], shape (B, C, H, W).
        rgb_indices: Explicit RGB channel order, e.g. [2, 1, 0]. Required unless `is_BGR` is set.
        is_BGR: Optional BGR-order flag. Ignored if `rgb_indices` is set.
        net: LPIPS backbone network ('alex', 'vgg', or 'squeeze').

    Returns:
        Tuple:
            rgb_lpips_val: Scalar tensor of mean RGB-branch LPIPS over the batch.
            nir_lpips_val: Scalar tensor of mean NIR-branch LPIPS for C=4, otherwise None.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)
    _validate_3_or_4_channels(recon_imgs)

    rgb_indices = _resolve_rgb_indices(recon_imgs, rgb_indices=rgb_indices, is_BGR=is_BGR)
    lpips_metric = _get_lpips_metric(recon_imgs.device, net=net)
    if lpips_metric is None:
        return None, None

    # LPIPS expects inputs in [-1, 1].
    recon_rgb = recon_imgs[:, rgb_indices, :, :] * 2.0 - 1.0
    true_rgb = true_imgs[:, rgb_indices, :, :] * 2.0 - 1.0
    rgb_lpips_val = _mean_metric_per_sample(
        lpips_metric(recon_rgb, true_rgb),
        batch_size=int(recon_imgs.shape[0]),
    ).mean()

    if recon_imgs.shape[1] == 3:
        nir_lpips_val = None
    else:
        recon_nir = recon_imgs[:, [3, 3, 3], :, :] * 2.0 - 1.0
        true_nir = true_imgs[:, [3, 3, 3], :, :] * 2.0 - 1.0
        nir_lpips_val = _mean_metric_per_sample(
            lpips_metric(recon_nir, true_nir),
            batch_size=int(recon_imgs.shape[0]),
        ).mean()

    return rgb_lpips_val, nir_lpips_val


def compute_SAM(recon_imgs: torch.Tensor, true_imgs: torch.Tensor, eps: float = 1e-8):
    """Compute sample-wise Spectral Angle Mapper (SAM) in degrees.

    Args:
        recon_imgs: Reconstructed image tensor, shape (B, C, H, W).
        true_imgs: Reference image tensor, shape (B, C, H, W).
        eps: Threshold for valid spectral norm product.

    Returns:
        Scalar tensor of mean SAM over the batch, where each sample is reduced
        over its own valid pixels first.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)

    pred = recon_imgs.permute(0, 2, 3, 1).reshape(recon_imgs.shape[0], -1, recon_imgs.shape[1])
    gt = true_imgs.permute(0, 2, 3, 1).reshape(true_imgs.shape[0], -1, true_imgs.shape[1])

    dot = (pred * gt).sum(dim=2)
    pred_norm = torch.linalg.norm(pred, dim=2)
    gt_norm = torch.linalg.norm(gt, dim=2)
    den = pred_norm * gt_norm

    valid = den > eps
    if not torch.any(valid):
        return torch.tensor(0.0, device=recon_imgs.device, dtype=torch.float32)

    cos_theta = torch.zeros_like(dot)
    cos_theta[valid] = dot[valid] / den[valid]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    sam_deg = torch.rad2deg(torch.arccos(cos_theta))
    sam_deg = torch.where(valid, sam_deg, torch.zeros_like(sam_deg))

    valid_counts = valid.sum(dim=1)
    sam_sum_per_sample = sam_deg.sum(dim=1)
    valid_counts_f = valid_counts.to(dtype=torch.float32).clamp_min(1.0)
    sam_mean_per_sample = sam_sum_per_sample / valid_counts_f
    sam_mean_per_sample = torch.where(
        valid_counts > 0,
        sam_mean_per_sample,
        torch.zeros_like(sam_mean_per_sample),
    )
    return sam_mean_per_sample.mean()


def compute_ERGAS(
    recon_imgs: torch.Tensor,
    true_imgs: torch.Tensor,
    scale_ratio: float,
    eps: float = 1e-8,
):
    """Compute sample-wise ERGAS and return the batch mean.

    Args:
        recon_imgs: Reconstructed image tensor in [0, 1], shape (B, C, H, W).
        true_imgs: Reference image tensor in [0, 1], shape (B, C, H, W).
        scale_ratio: Super-resolution scale factor used in ERGAS.
        eps: Numerical lower bound for safe division.

    Returns:
        Scalar tensor of mean ERGAS over the batch.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)
    scale_ratio = float(scale_ratio)
    if scale_ratio <= 0.0:
        raise ValueError(f"`scale_ratio` must be positive, got {scale_ratio}.")

    rmse_per_band = torch.sqrt(
        torch.mean((recon_imgs - true_imgs) ** 2, dim=(2, 3)).clamp_min(0.0)
    )
    mean_ref_per_band = torch.mean(true_imgs, dim=(2, 3)).abs().clamp_min(eps)
    rel_rmse_sq = (rmse_per_band / mean_ref_per_band).square()
    ergas_per_sample = (100.0 / scale_ratio) * torch.sqrt(rel_rmse_sq.mean(dim=1).clamp_min(0.0))
    return ergas_per_sample.mean()


def compute_SCC(
    recon_imgs: torch.Tensor,
    true_imgs: torch.Tensor,
    eps: float = 1e-8,
):
    """Compute sample-wise spectral correlation coefficient (SCC) and return the batch mean.

    SCC is computed as the average Pearson correlation between reconstructed and
    reference spectral vectors over valid pixels in each sample.
    """
    recon_imgs, true_imgs = _prepare_metric_pair(recon_imgs, true_imgs)
    pred = recon_imgs.permute(0, 2, 3, 1).reshape(recon_imgs.shape[0], -1, recon_imgs.shape[1])
    gt = true_imgs.permute(0, 2, 3, 1).reshape(true_imgs.shape[0], -1, true_imgs.shape[1])

    pred_centered = pred - pred.mean(dim=2, keepdim=True)
    gt_centered = gt - gt.mean(dim=2, keepdim=True)

    numerator = (pred_centered * gt_centered).sum(dim=2)
    pred_norm = torch.linalg.norm(pred_centered, dim=2)
    gt_norm = torch.linalg.norm(gt_centered, dim=2)
    denominator = pred_norm * gt_norm

    valid = denominator > eps
    if not torch.any(valid):
        return torch.tensor(0.0, device=recon_imgs.device, dtype=torch.float32)

    corr = torch.zeros_like(numerator)
    corr[valid] = numerator[valid] / denominator[valid]
    corr = torch.clamp(corr, -1.0, 1.0)

    corr = torch.where(valid, corr, torch.zeros_like(corr))
    valid_counts = valid.sum(dim=1)
    corr_sum_per_sample = corr.sum(dim=1)
    valid_counts_f = valid_counts.to(dtype=torch.float32).clamp_min(1.0)
    corr_mean_per_sample = corr_sum_per_sample / valid_counts_f
    corr_mean_per_sample = torch.where(
        valid_counts > 0,
        corr_mean_per_sample,
        torch.zeros_like(corr_mean_per_sample),
    )
    return corr_mean_per_sample.mean()
