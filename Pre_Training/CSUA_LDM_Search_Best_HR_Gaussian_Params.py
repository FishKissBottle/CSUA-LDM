"""Search the best HR Gaussian sigma for matching LR-up and HR-derived high-frequency content.

This script uses the currently active dataset/runtime configuration from
``Main_Model/CSUA_LDM_Config.py`` and searches only ``hr_gaussian_sigma``.

Search pipeline:
1. Read the validation split defined by the active config.
2. Extract high-frequency residuals from ``hr`` and ``lr_up`` in the FFT domain
   using separately configurable normalized high-pass radii.
3. Optionally blur the LR-up high-frequency residual with the configured
   ``lr_gaussian_kernel`` / ``lr_gaussian_sigma`` when
   ``lr_gaussian_blur_enabled=True``.
4. For each candidate ``hr_gaussian_sigma``:
   - derive kernel size via ``2 * ceil(3 * sigma) + 1``
   - blur the HR high-frequency residual
   - downsample to LR size and upsample back to HR size
   - compare against LR-up high-frequency residual using PSNR or LPIPS
5. Report and save the best sigma.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is expected, but keep script usable.
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Main_Model.CSUA_LDM_Config import (
    CSUA_LDM_DATASET_DIRS,
    CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    CSUA_LDM_HR_CROP_SIZE,
    CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
    CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
    CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
    CSUA_LDM_HR_DEGRADATION_UP_MODE,
    CSUA_LDM_INPUT_CHANNELS,
    CSUA_LDM_LR_CROP_SIZE,
    CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
    CSUA_LDM_LR_GAUSSIAN_KERNEL,
    CSUA_LDM_LR_GAUSSIAN_SIGMA,
    CSUA_LDM_LR_UP_MODE,
    CSUA_LDM_SUPERRES_SCALE,
    CSUA_LDM_VALID_BATCH_SIZE,
    CSUA_LDM_VIS_BAND_ORDER,
    DEVICE,
    NUM_WORKERS,
    RANDOM_SEED,
)
from CSUA_LDM_Dataset import SuperResolutionDataset, _gaussian_kernel_2d
from CSUA_LDM_Quality_Metrics import compute_LPIPS, compute_PSNR, format_metric_value
from CSUA_LDM_Utils import set_random_seed


DEFAULT_SIGMA_SEARCH_OFFSETS = [
    -2.00,
    -1.75,
    -1.50,
    -1.25,
    -1.00,
    -0.75,
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00
]
# DEFAULT_HR_FFT_RADIUS = 0.035
# DEFAULT_LR_FFT_RADIUS = 0.020
DEFAULT_HR_FFT_RADIUS = 0.0110
DEFAULT_LR_FFT_RADIUS = 0.0050
DEFAULT_METRIC = "lpips"
# Master switch for plt batch preview:
# True  -> call matplotlib to show the first-sample RGB high-frequency preview
# False -> skip plt display (preview PNG saving can still be enabled separately)
DEFAULT_PLT_BATCH_PREVIEW_ENABLED = True
DEFAULT_PREVIEW_PAUSE_SECONDS = 0.8
DEFAULT_SAVE_PREVIEW_IMAGES = True


def _kernel_from_sigma(sigma: float) -> int:
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}.")
    return int(2 * math.ceil(sigma * 3.0) + 1)


def _build_sigma_candidates_from_yaml_center(center_sigma: float) -> list[float]:
    center_sigma = float(center_sigma)
    candidates = []
    for offset in DEFAULT_SIGMA_SEARCH_OFFSETS:
        sigma = center_sigma + float(offset)
        if sigma <= 0.0:
            continue
        candidates.append(round(sigma, 6))

    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError(
            "No valid hr_gaussian_sigma candidates were generated. "
            f"Center sigma={center_sigma}, offsets={DEFAULT_SIGMA_SEARCH_OFFSETS}"
        )
    return candidates


def _make_output_dirs() -> tuple[Path, Path]:
    output_dir = Path("debug_hr_gaussian_sigma_search")
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, preview_dir


def _resolve_valid_dataset_root_and_split() -> tuple[str, str]:
    root_dir = str(CSUA_LDM_DATASET_DIRS["rootdir_list_forValid"])
    split_name = str(CSUA_LDM_DATASET_DIRS["split_name_forValid"])
    return root_dir, split_name


def _build_valid_dataset() -> SuperResolutionDataset:
    root_dir, split_name = _resolve_valid_dataset_root_and_split()
    return SuperResolutionDataset(
        root_dir=root_dir,
        is_train=False,
        split_name=split_name,
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
        superres_scale=CSUA_LDM_SUPERRES_SCALE,
        lr_crop_size=CSUA_LDM_LR_CROP_SIZE,
        hr_crop_size=CSUA_LDM_HR_CROP_SIZE,
        input_channels=CSUA_LDM_INPUT_CHANNELS,
        hr_degradation_gaussian_kernel=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        hr_degradation_gaussian_sigma=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        hr_degradation_down_mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        hr_degradation_up_mode=CSUA_LDM_HR_DEGRADATION_UP_MODE,
        soft_hr_gaussian_blur_enabled=False,
        soft_hr_gaussian_kernel=1,
        soft_hr_gaussian_sigma=0.0,
        soft_hr_down_mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        soft_hr_up_mode=CSUA_LDM_HR_DEGRADATION_UP_MODE,
        lr_up_mode=CSUA_LDM_LR_UP_MODE,
        lr_gaussian_blur_enabled=False,
        lr_gaussian_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
        **CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )


def _build_valid_loader(batch_size: int) -> DataLoader:
    dataset = _build_valid_dataset()
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=bool(NUM_WORKERS > 0),
    )


def _fft_highpass_mask(height: int, width: int, radius: float, device, dtype) -> torch.Tensor:
    if not (0.0 <= float(radius) <= 1.0):
        raise ValueError(f"Normalized FFT radius must be in [0, 1], got {radius}.")

    yy = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radial = torch.sqrt(grid_x.square() + grid_y.square()) / math.sqrt(1.0)
    mask = (radial >= float(radius)).to(dtype=dtype)
    return mask.unsqueeze(0).unsqueeze(0)


def _extract_high_frequency(imgs: torch.Tensor, radius: float) -> torch.Tensor:
    if imgs.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(imgs.shape)}.")
    _, _, height, width = imgs.shape
    mask = _fft_highpass_mask(height, width, radius, imgs.device, imgs.dtype)
    spectrum = torch.fft.fftshift(torch.fft.fft2(imgs, dim=(-2, -1)), dim=(-2, -1))
    filtered = spectrum * mask
    recon = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)).real
    return recon


def _gaussian_blur_batch(imgs: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1 or sigma <= 0.0:
        return imgs
    if kernel_size % 2 == 0:
        kernel_size += 1

    batch, channels, _, _ = imgs.shape
    padding = kernel_size // 2
    kernel = _gaussian_kernel_2d(kernel_size, sigma).to(device=imgs.device, dtype=imgs.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    imgs_padded = F.pad(imgs, (padding, padding, padding, padding), mode="replicate")
    return F.conv2d(imgs_padded, kernel, groups=channels)


def _resize_batch(imgs: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    safe_mode = mode if mode in {"nearest", "linear", "bilinear", "bicubic", "trilinear", "area"} else "bicubic"
    if safe_mode in {"nearest", "area"}:
        return F.interpolate(imgs, size=size, mode=safe_mode)
    return F.interpolate(imgs, size=size, mode=safe_mode, align_corners=False)


def _shared_residual_to_unit_interval(
    pred_imgs: torch.Tensor,
    ref_imgs: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_abs = pred_imgs.abs().amax(dim=(1, 2, 3), keepdim=True)
    ref_abs = ref_imgs.abs().amax(dim=(1, 2, 3), keepdim=True)
    shared_scale = torch.maximum(pred_abs, ref_abs).clamp_min(eps)
    pred_unit = ((pred_imgs / shared_scale) + 1.0) * 0.5
    ref_unit = ((ref_imgs / shared_scale) + 1.0) * 0.5
    return pred_unit.clamp(0.0, 1.0), ref_unit.clamp(0.0, 1.0)


def _tensor_stats_str(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float()
    finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    total_count = int(tensor.numel())
    if finite_count <= 0:
        return f"shape={tuple(tensor.shape)}, finite=0/{total_count}"

    finite_vals = tensor[finite_mask]
    return (
        f"shape={tuple(tensor.shape)}, "
        f"finite={finite_count}/{total_count}, "
        f"min={float(finite_vals.min().item()):.6g}, "
        f"max={float(finite_vals.max().item()):.6g}, "
        f"mean={float(finite_vals.mean().item()):.6g}"
    )


def _assert_finite_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    batch_index: int,
    sigma: float | None = None,
    kernel_size: int | None = None,
) -> None:
    if bool(torch.isfinite(tensor).all().item()):
        return

    context_parts = [f"batch_index={int(batch_index)}", f"tensor={name}"]
    if sigma is not None:
        context_parts.append(f"sigma={float(sigma):.6f}")
    if kernel_size is not None:
        context_parts.append(f"kernel={int(kernel_size)}")
    context = ", ".join(context_parts)
    raise RuntimeError(
        "Non-finite tensor detected during hr_gaussian_sigma search. "
        f"{context}. Stats: {_tensor_stats_str(tensor)}"
    )


def _assert_finite_metric_value(
    metric_name: str,
    metric_value: float,
    *,
    batch_index: int,
    sigma: float,
    kernel_size: int,
    pred_unit: torch.Tensor,
    ref_unit: torch.Tensor,
) -> None:
    if math.isfinite(float(metric_value)):
        return

    raise RuntimeError(
        f"Non-finite {metric_name} detected during hr_gaussian_sigma search. "
        f"batch_index={int(batch_index)}, sigma={float(sigma):.6f}, kernel={int(kernel_size)}. "
        f"pred_unit stats: {_tensor_stats_str(pred_unit)}; "
        f"ref_unit stats: {_tensor_stats_str(ref_unit)}"
    )


def _metric_value(metric_name: str, pred_imgs: torch.Tensor, ref_imgs: torch.Tensor) -> float:
    metric_name = str(metric_name).strip().lower()
    if metric_name == "psnr":
        return float(compute_PSNR(pred_imgs, ref_imgs).detach().cpu().item())
    if metric_name == "lpips":
        rgb_lpips, nir_lpips = compute_LPIPS(
            pred_imgs,
            ref_imgs,
            rgb_indices=CSUA_LDM_VIS_BAND_ORDER,
        )
        if rgb_lpips is None:
            raise RuntimeError("LPIPS backend is unavailable. Please install `lpips` or switch metric to PSNR.")
        value = float(rgb_lpips.detach().cpu().item())
        if nir_lpips is not None:
            value = 0.5 * (value + float(nir_lpips.detach().cpu().item()))
        return value
    raise ValueError(f"Unsupported metric '{metric_name}'. Use 'psnr' or 'lpips'.")


def _metric_higher_is_better(metric_name: str) -> bool:
    return str(metric_name).strip().lower() == "psnr"


def _save_or_show_preview(
    *,
    batch_index: int,
    hr_sigma: float,
    hr_kernel: int,
    lr_kernel: int | None,
    lr_sigma: float | None,
    hr_preview: torch.Tensor,
    lr_ref_preview: torch.Tensor,
    preview_dir: Path,
    show_preview: bool,
    preview_pause_seconds: float,
    save_preview: bool,
) -> None:
    preview_tensors = [lr_ref_preview, hr_preview]
    shared_scale = torch.stack([tensor.abs().amax() for tensor in preview_tensors]).max().clamp_min(1e-6)

    def _to_rgb_unit(img_chw: torch.Tensor):
        img_unit = (((img_chw / shared_scale) + 1.0) * 0.5).clamp(0.0, 1.0)
        return img_unit[CSUA_LDM_VIS_BAND_ORDER, :, :].detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    lr_rgb = _to_rgb_unit(lr_ref_preview)
    axes[0].imshow(lr_rgb)
    if lr_kernel is not None and lr_sigma is not None:
        axes[0].set_title(f"LR-Up HF + Gaussian(k={lr_kernel}, s={lr_sigma:.3f})")
    else:
        axes[0].set_title("LR-Up High-Frequency")
    axes[0].axis("off")

    hr_rgb = _to_rgb_unit(hr_preview)
    axes[1].imshow(hr_rgb)
    axes[1].set_title(
        f"HR HF -> Gaussian(k={hr_kernel}, s={hr_sigma:.3f}) -> DownUp"
    )
    axes[1].axis("off")

    fig.suptitle(
        f"Batch {batch_index:04d} First Sample High-Frequency Preview",
        fontsize=13,
    )
    fig.tight_layout()

    if save_preview:
        fig.savefig(
            preview_dir / f"batch_{batch_index:04d}_hr_sigma_{hr_sigma:.3f}_k_{hr_kernel}_highfreq_preview.png",
            dpi=180,
            bbox_inches="tight",
        )

    if show_preview:
        plt.show(block=False)
        plt.pause(max(0.001, float(preview_pause_seconds)))
    plt.close(fig)


def search_best_hr_sigma(
    *,
    sigma_candidates: list[float],
    preview_sigma: float,
    metric_name: str,
    hr_fft_radius: float,
    lr_fft_radius: float,
    batch_size: int,
    show_batch_preview: bool,
    preview_pause_seconds: float,
    save_preview_images: bool,
) -> tuple[list[dict], int, float]:
    loader = _build_valid_loader(batch_size=batch_size)
    output_dir, preview_dir = _make_output_dirs()

    start_time = time.time()
    metric_sums = {float(sigma): 0.0 for sigma in sigma_candidates}
    processed_samples = 0
    target_total = len(loader.dataset)

    progress = tqdm(total=target_total, desc="Search hr_gaussian_sigma", unit="img") if tqdm is not None else None

    higher_is_better = _metric_higher_is_better(metric_name)

    for batch_index, batch in enumerate(loader):
        lr_up_imgs = batch[1].to(device=DEVICE, dtype=torch.float32)
        hr_imgs = batch[4].to(device=DEVICE, dtype=torch.float32)
        batch_size_now = int(lr_up_imgs.shape[0])

        hr_highfreq = _extract_high_frequency(hr_imgs, hr_fft_radius)
        lr_highfreq = _extract_high_frequency(lr_up_imgs, lr_fft_radius)
        _assert_finite_tensor("hr_imgs", hr_imgs, batch_index=batch_index)
        _assert_finite_tensor("lr_up_imgs", lr_up_imgs, batch_index=batch_index)
        _assert_finite_tensor("hr_highfreq", hr_highfreq, batch_index=batch_index)
        _assert_finite_tensor("lr_highfreq", lr_highfreq, batch_index=batch_index)

        if CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED and CSUA_LDM_LR_GAUSSIAN_KERNEL > 1 and CSUA_LDM_LR_GAUSSIAN_SIGMA > 0.0:
            lr_highfreq_ref = _gaussian_blur_batch(
                lr_highfreq,
                kernel_size=CSUA_LDM_LR_GAUSSIAN_KERNEL,
                sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
            )
        else:
            lr_highfreq_ref = lr_highfreq
        _assert_finite_tensor("lr_highfreq_ref", lr_highfreq_ref, batch_index=batch_index)

        preview_hr_item: tuple[float, int, torch.Tensor] | None = None
        preview_lr_ref = lr_highfreq_ref[0].detach().cpu()

        for sigma in sigma_candidates:
            sigma = float(sigma)
            kernel_size = _kernel_from_sigma(sigma)

            hr_highfreq_blurred = _gaussian_blur_batch(
                hr_highfreq,
                kernel_size=kernel_size,
                sigma=sigma,
            )
            _assert_finite_tensor(
                "hr_highfreq_blurred",
                hr_highfreq_blurred,
                batch_index=batch_index,
                sigma=sigma,
                kernel_size=kernel_size,
            )
            hr_highfreq_down = _resize_batch(
                hr_highfreq_blurred,
                size=(CSUA_LDM_LR_CROP_SIZE, CSUA_LDM_LR_CROP_SIZE),
                mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
            )
            _assert_finite_tensor(
                "hr_highfreq_down",
                hr_highfreq_down,
                batch_index=batch_index,
                sigma=sigma,
                kernel_size=kernel_size,
            )
            hr_highfreq_down_up = _resize_batch(
                hr_highfreq_down,
                size=(CSUA_LDM_HR_CROP_SIZE, CSUA_LDM_HR_CROP_SIZE),
                mode=CSUA_LDM_HR_DEGRADATION_UP_MODE,
            )
            _assert_finite_tensor(
                "hr_highfreq_down_up",
                hr_highfreq_down_up,
                batch_index=batch_index,
                sigma=sigma,
                kernel_size=kernel_size,
            )

            pred_unit, ref_unit = _shared_residual_to_unit_interval(
                hr_highfreq_down_up,
                lr_highfreq_ref,
            )
            _assert_finite_tensor(
                "pred_unit",
                pred_unit,
                batch_index=batch_index,
                sigma=sigma,
                kernel_size=kernel_size,
            )
            _assert_finite_tensor(
                "ref_unit",
                ref_unit,
                batch_index=batch_index,
                sigma=sigma,
                kernel_size=kernel_size,
            )

            if (
                (show_batch_preview or save_preview_images)
                and preview_hr_item is None
                and abs(sigma - float(preview_sigma)) < 1e-9
            ):
                preview_hr_item = (sigma, kernel_size, hr_highfreq_down_up[0].detach().cpu())

            metric_value = _metric_value(metric_name, pred_unit, ref_unit)
            _assert_finite_metric_value(
                metric_name,
                metric_value,
                batch_index=batch_index,
                sigma=sigma,
                kernel_size=kernel_size,
                pred_unit=pred_unit,
                ref_unit=ref_unit,
            )
            metric_sums[sigma] += metric_value * batch_size_now

        if (show_batch_preview or save_preview_images) and preview_hr_item is not None:
            _save_or_show_preview(
                batch_index=batch_index,
                hr_sigma=preview_hr_item[0],
                hr_kernel=preview_hr_item[1],
                lr_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL if CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED else None,
                lr_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA if CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED else None,
                hr_preview=preview_hr_item[2],
                lr_ref_preview=preview_lr_ref,
                preview_dir=preview_dir,
                show_preview=show_batch_preview,
                preview_pause_seconds=preview_pause_seconds,
                save_preview=save_preview_images,
            )

        processed_samples += batch_size_now
        if progress is not None:
            progress.update(batch_size_now)

    if progress is not None:
        progress.close()

    elapsed = time.time() - start_time
    if processed_samples == 0:
        raise RuntimeError("No samples were processed during hr_gaussian_sigma search.")

    results = []
    for sigma in sigma_candidates:
        sigma = float(sigma)
        results.append(
            {
                "hr_gaussian_sigma": sigma,
                "hr_gaussian_kernel": _kernel_from_sigma(sigma),
                "metric_name": metric_name,
                "metric_value": metric_sums[sigma] / processed_samples,
            }
        )

    results.sort(key=lambda item: item["metric_value"], reverse=higher_is_better)

    csv_path = output_dir / "hr_gaussian_sigma_search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["hr_gaussian_sigma", "hr_gaussian_kernel", "metric_name", "metric_value"],
        )
        writer.writeheader()
        writer.writerows(results)

    report_path = output_dir / "hr_gaussian_sigma_search_report.txt"
    best = results[0]
    lines = [
        "HR Gaussian Sigma Search Report",
        "=" * 40,
        f"Dataset root: {CSUA_LDM_DATASET_DIRS['rootdir_list_forValid']}",
        f"Split: {CSUA_LDM_DATASET_DIRS['split_name_forValid']}",
        f"Samples processed: {processed_samples}",
        f"Metric: {metric_name}",
        f"HR FFT radius: {hr_fft_radius}",
        f"LR FFT radius: {lr_fft_radius}",
        f"LR blur enabled: {CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED}",
        f"LR blur kernel: {CSUA_LDM_LR_GAUSSIAN_KERNEL}",
        f"LR blur sigma: {CSUA_LDM_LR_GAUSSIAN_SIGMA}",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "Best result",
        "-" * 20,
        f"hr_gaussian_sigma: {best['hr_gaussian_sigma']}",
        f"hr_gaussian_kernel: {best['hr_gaussian_kernel']}",
        f"{metric_name}: {format_metric_value(best['metric_value'])}",
        "",
        "All candidates",
        "-" * 20,
    ]
    for item in results:
        lines.append(
            f"sigma={item['hr_gaussian_sigma']:<6g} "
            f"kernel={item['hr_gaussian_kernel']:<3d} "
            f"{metric_name}={format_metric_value(item['metric_value'])}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return results, processed_samples, elapsed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the best hr_gaussian_sigma using current CSUA_LDM dataset configuration."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=["psnr", "lpips"],
        help="Comparison metric used to rank sigma candidates.",
    )
    parser.add_argument(
        "--hr-fft-radius",
        type=float,
        default=DEFAULT_HR_FFT_RADIUS,
        help="Normalized high-pass FFT radius for HR images in [0, 1].",
    )
    parser.add_argument(
        "--lr-fft-radius",
        type=float,
        default=DEFAULT_LR_FFT_RADIUS,
        help="Normalized high-pass FFT radius for LR-up images in [0, 1].",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(CSUA_LDM_VALID_BATCH_SIZE),
        help="Validation batch size used during the search.",
    )
    parser.add_argument(
        "--show-batch-preview",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PLT_BATCH_PREVIEW_ENABLED,
        help="Whether to use plt to display the batch preview of extracted HR/LR high-frequency RGB images.",
    )
    parser.add_argument(
        "--preview-pause-seconds",
        type=float,
        default=DEFAULT_PREVIEW_PAUSE_SECONDS,
        help="Non-blocking plt preview dwell time in seconds for each batch preview.",
    )
    parser.add_argument(
        "--save-preview-images",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_PREVIEW_IMAGES,
        help="Whether to save preview PNGs under debug_hr_gaussian_sigma_search/previews.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    set_random_seed(RANDOM_SEED)

    sigma_center = float(CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA)
    sigma_candidates = _build_sigma_candidates_from_yaml_center(sigma_center)

    print("=" * 72)
    print("Search best hr_gaussian_sigma")
    print("=" * 72)
    print(f"Valid root           : {CSUA_LDM_DATASET_DIRS['rootdir_list_forValid']}")
    print(f"Valid split          : {CSUA_LDM_DATASET_DIRS['split_name_forValid']}")
    print(f"Metric               : {args.metric}")
    print(f"YAML hr sigma center : {sigma_center}")
    print(f"Sigma offsets        : {DEFAULT_SIGMA_SEARCH_OFFSETS}")
    print(f"Candidate hr sigma   : {sigma_candidates}")
    preview_sigma = min(sigma_candidates, key=lambda value: abs(float(value) - sigma_center))
    print(f"Preview hr sigma     : {preview_sigma}")
    print(f"Derived kernels      : {[_kernel_from_sigma(v) for v in sigma_candidates]}")
    print(f"HR FFT radius        : {args.hr_fft_radius}")
    print(f"LR FFT radius        : {args.lr_fft_radius}")
    print(f"LR blur enabled      : {CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED}")
    print(f"LR blur kernel       : {CSUA_LDM_LR_GAUSSIAN_KERNEL}")
    print(f"LR blur sigma        : {CSUA_LDM_LR_GAUSSIAN_SIGMA}")
    print(f"PLT batch preview    : {args.show_batch_preview}")
    print(f"Preview pause (s)    : {args.preview_pause_seconds}")
    print(f"Save preview images  : {args.save_preview_images}")
    print(f"Batch size           : {args.batch_size}")
    print("=" * 72)

    results, processed_samples, elapsed = search_best_hr_sigma(
        sigma_candidates=sigma_candidates,
        preview_sigma=float(preview_sigma),
        metric_name=args.metric,
        hr_fft_radius=float(args.hr_fft_radius),
        lr_fft_radius=float(args.lr_fft_radius),
        batch_size=int(args.batch_size),
        show_batch_preview=bool(args.show_batch_preview),
        preview_pause_seconds=float(args.preview_pause_seconds),
        save_preview_images=bool(args.save_preview_images),
    )

    best = results[0]
    print("\nBest result")
    print("-" * 32)
    print(f"processed_samples    : {processed_samples}")
    print(f"elapsed_seconds      : {elapsed:.2f}")
    print(f"best_hr_sigma        : {best['hr_gaussian_sigma']}")
    print(f"best_hr_kernel       : {best['hr_gaussian_kernel']}")
    print(f"best_{args.metric}   : {format_metric_value(best['metric_value'])}")

    print("\nAll candidates")
    print("-" * 32)
    for item in results:
        print(
            f"sigma={item['hr_gaussian_sigma']:<6g} "
            f"kernel={item['hr_gaussian_kernel']:<3d} "
            f"{args.metric}={format_metric_value(item['metric_value'])}"
        )


if __name__ == "__main__":
    main()
