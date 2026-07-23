import os
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from CSUA_LDM_Dataset import SuperResolutionDataset, _gaussian_blur_2d
from CSUA_LDM_Quality_Metrics import (
    compute_ERGAS,
    compute_FSIM,
    compute_GMSD,
    compute_LPIPS,
    compute_PSNR,
    compute_SAM,
    compute_SCC,
    compute_SSIM,
    format_metric_value,
)
from CSUA_LDM_Utils import geo_to_numpy_array, load_model, save_rgb_datas, save_tif_datas


CSUA_LDM_STANDARD_RESULTS_ROOT = PROJECT_ROOT / "CSUA_LDM_Evaluation_Results"


def build_standard_results_root(mode: str, tag: str) -> Path:
    mode_key = str(mode).strip().lower()
    if mode_key == "evaluate":
        mode_dir = None
    elif mode_key == "spatial_evaluate":
        mode_dir = "SpatialEvaluate"
    elif mode_key == "inference":
        mode_dir = "Inference"
    elif mode_key == "spectral_evaluate":
        mode_dir = "SpectralEvaluate"
    else:
        raise ValueError(
            "`mode` must be one of 'evaluate', 'spatial_evaluate', 'inference', or 'spectral_evaluate'."
        )

    safe_tag = str(tag).strip().replace("\\", "_").replace("/", "_").replace(" ", "_")
    if mode_dir is None:
        return CSUA_LDM_STANDARD_RESULTS_ROOT / safe_tag
    return CSUA_LDM_STANDARD_RESULTS_ROOT / mode_dir / safe_tag


def count_model_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _to_stats_device(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return tensor.to(device=ref.device, dtype=ref.dtype)


def normalize_imgs(imgs: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    mean = _to_stats_device(mean_t, imgs)
    std = _to_stats_device(std_t, imgs)
    return (imgs - mean) / std


def denormalize_imgs(imgs: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    mean = _to_stats_device(mean_t, imgs)
    std = _to_stats_device(std_t, imgs)
    return torch.clamp(imgs * std + mean, min=0.0, max=1.0)


def _metric_to_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            value = value.mean()
        return float(value.detach().float().cpu().item())
    return float(value)


@torch.no_grad()
def compute_quality_metrics(
    sr_norm: torch.Tensor,
    hr_norm: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    vis_band_order: list[int],
) -> dict[str, float]:
    sr_denorm = denormalize_imgs(sr_norm.float(), mean_t, std_t)
    hr_denorm = denormalize_imgs(hr_norm.float(), mean_t, std_t)

    gmsd_rgb, gmsd_nir = compute_GMSD(sr_denorm, hr_denorm, rgb_indices=vis_band_order)
    fsim_rgb, fsim_nir = compute_FSIM(sr_denorm, hr_denorm, rgb_indices=vis_band_order)
    lpips_rgb, lpips_nir = compute_LPIPS(sr_denorm, hr_denorm, rgb_indices=vis_band_order)
    return {
        "psnr": _metric_to_float(compute_PSNR(sr_denorm, hr_denorm)),
        "ssim": _metric_to_float(compute_SSIM(sr_denorm, hr_denorm)),
        "gmsd_rgb": _metric_to_float(gmsd_rgb),
        "gmsd_nir": _metric_to_float(gmsd_nir),
        "fsim_rgb": _metric_to_float(fsim_rgb),
        "fsim_nir": _metric_to_float(fsim_nir),
        "lpips_rgb": _metric_to_float(lpips_rgb),
        "lpips_nir": _metric_to_float(lpips_nir),
    }


def format_metric_summary(metrics: dict[str, float]) -> str:
    return (
        f"PSNR: {format_metric_value(metrics['psnr'])}, "
        f"SSIM: {format_metric_value(metrics['ssim'])}, "
        f"GMSD_RGB: {format_metric_value(metrics['gmsd_rgb'])}, "
        f"GMSD_NIR: {format_metric_value(metrics['gmsd_nir'])}, "
        f"FSIM_RGB: {format_metric_value(metrics['fsim_rgb'])}, "
        f"FSIM_NIR: {format_metric_value(metrics['fsim_nir'])}, "
        f"LPIPS_RGB: {format_metric_value(metrics['lpips_rgb'])}, "
        f"LPIPS_NIR: {format_metric_value(metrics['lpips_nir'])}"
    )


def format_spectral_metric_summary(metrics: dict[str, float]) -> str:
    return (
        f"SAM: {format_metric_value(metrics['sam'])}, "
        f"ERGAS: {format_metric_value(metrics['ergas'])}, "
        f"SCC: {format_metric_value(metrics['scc'])}"
    )


def collated_geos_to_batch(draw_geos):
    return np.array([geo_to_numpy_array(draw_geo) for draw_geo in draw_geos]).T


def ensure_output_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_root / "RGB_Comparisons"
    tif_dir = output_root / "TIF_Outputs"
    report_dir = output_root / "Reports"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    tif_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return rgb_dir, tif_dir, report_dir


def ensure_explicit_output_dirs(
    *,
    rgb_dir: str | Path,
    tif_dir: str | Path,
    report_path: str | Path | None = None,
) -> None:
    Path(rgb_dir).mkdir(parents=True, exist_ok=True)
    Path(tif_dir).mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)


def write_text_report(report_path: str | Path, lines: list[str]) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


def build_unit_interval_stats(
    num_channels: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_t = torch.zeros((1, int(num_channels), 1, 1), dtype=dtype)
    std_t = torch.ones((1, int(num_channels), 1, 1), dtype=dtype)
    return mean_t, std_t


def read_checkpoint_epoch(checkpoint_path: str | Path) -> int:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        return int(checkpoint["epoch"])
    return -1


def load_model_with_ema_fallback(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> tuple[torch.nn.Module, bool, int]:
    effective_load_ema = True
    loaded_epoch = -1
    try:
        model, _, _, loaded_epoch, _, _, _ = load_model(
            checkpoint_path,
            model,
            None,
            None,
            resume_training=False,
            load_ema=effective_load_ema,
            strict=strict,
        )
    except KeyError:
        effective_load_ema = False
        model, _, _, loaded_epoch, _, _, _ = load_model(
            checkpoint_path,
            model,
            None,
            None,
            resume_training=False,
            load_ema=effective_load_ema,
            strict=strict,
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, effective_load_ema, int(loaded_epoch)


def build_dataset(
    *,
    root_dir: str,
    split_name: str,
    vis_band_order: list[int],
    superres_scale: int,
    lr_crop_size: int,
    hr_crop_size: int,
    input_channels: int,
    hr_degradation_gaussian_kernel: int,
    hr_degradation_gaussian_sigma: float,
    hr_degradation_down_mode: str,
    hr_degradation_up_mode: str,
    soft_hr_gaussian_blur_enabled: bool,
    soft_hr_gaussian_kernel: int,
    soft_hr_gaussian_sigma: float,
    soft_hr_down_mode: str,
    soft_hr_up_mode: str,
    lr_up_mode: str,
    lr_gaussian_blur_enabled: bool,
    lr_gaussian_kernel: int,
    lr_gaussian_sigma: float,
    raw_scaling_kwargs: dict,
) -> SuperResolutionDataset:
    return SuperResolutionDataset(
        root_dir=root_dir,
        is_train=False,
        split_name=split_name,
        vis_band_order=vis_band_order,
        superres_scale=superres_scale,
        lr_crop_size=lr_crop_size,
        hr_crop_size=hr_crop_size,
        input_channels=input_channels,
        hr_degradation_gaussian_kernel=hr_degradation_gaussian_kernel,
        hr_degradation_gaussian_sigma=hr_degradation_gaussian_sigma,
        hr_degradation_down_mode=hr_degradation_down_mode,
        hr_degradation_up_mode=hr_degradation_up_mode,
        soft_hr_gaussian_blur_enabled=soft_hr_gaussian_blur_enabled,
        soft_hr_gaussian_kernel=soft_hr_gaussian_kernel,
        soft_hr_gaussian_sigma=soft_hr_gaussian_sigma,
        soft_hr_down_mode=soft_hr_down_mode,
        soft_hr_up_mode=soft_hr_up_mode,
        lr_up_mode=lr_up_mode,
        lr_gaussian_blur_enabled=lr_gaussian_blur_enabled,
        lr_gaussian_kernel=lr_gaussian_kernel,
        lr_gaussian_sigma=lr_gaussian_sigma,
        **raw_scaling_kwargs,
    )


def build_data_loader(
    dataset: SuperResolutionDataset,
    *,
    batch_size: int,
    num_workers: int,
    persistent_workers_enabled: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(persistent_workers_enabled and int(num_workers) > 0),
        drop_last=False,
    )


def iter_eval_loader(
    data_loader: DataLoader,
    *,
    desc: str = "Evaluating",
):
    total = len(data_loader) if hasattr(data_loader, "__len__") else None
    return tqdm(
        data_loader,
        total=total,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )


def _resolve_downsample_kwargs(mode: str) -> dict:
    resolved_mode = str(mode)
    if resolved_mode not in ("nearest", "linear", "bilinear", "bicubic", "trilinear", "area"):
        resolved_mode = "area"
    return {
        "mode": resolved_mode,
        "align_corners": None if resolved_mode == "area" else False,
    }


@torch.no_grad()
def degrade_hr_like_to_lr(
    hr_like_imgs: torch.Tensor,
    *,
    superres_scale: int,
    gaussian_kernel: int,
    gaussian_sigma: float,
    down_mode: str,
) -> torch.Tensor:
    if hr_like_imgs.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {tuple(hr_like_imgs.shape)}.")

    superres_scale = int(superres_scale)
    if superres_scale <= 0:
        raise ValueError(f"`superres_scale` must be positive, got {superres_scale}.")

    b, _, h_hr, w_hr = hr_like_imgs.shape
    if h_hr % superres_scale != 0 or w_hr % superres_scale != 0:
        raise ValueError(
            f"HR-like tensor shape {(h_hr, w_hr)} is not divisible by scale={superres_scale}."
        )

    if int(gaussian_kernel) > 0 and float(gaussian_sigma) > 0.0:
        blurred = torch.stack(
            [
                _gaussian_blur_2d(
                    hr_like_imgs[idx],
                    int(gaussian_kernel),
                    float(gaussian_sigma),
                )
                for idx in range(int(b))
            ],
            dim=0,
        )
    else:
        blurred = hr_like_imgs

    down_kwargs = _resolve_downsample_kwargs(down_mode)
    return F.interpolate(
        blurred,
        size=(h_hr // superres_scale, w_hr // superres_scale),
        **down_kwargs,
    )


@torch.no_grad()
def compute_spectral_metrics(
    sr_norm: torch.Tensor,
    lr_ref: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    *,
    superres_scale: int,
    gaussian_kernel: int,
    gaussian_sigma: float,
    down_mode: str,
) -> dict[str, float]:
    sr_denorm = denormalize_imgs(sr_norm.float(), mean_t, std_t)
    lr_ref = torch.clamp(lr_ref.to(device=sr_denorm.device, dtype=torch.float32), min=0.0, max=1.0)
    sr_degraded = degrade_hr_like_to_lr(
        sr_denorm,
        superres_scale=superres_scale,
        gaussian_kernel=gaussian_kernel,
        gaussian_sigma=gaussian_sigma,
        down_mode=down_mode,
    )

    if tuple(sr_degraded.shape) != tuple(lr_ref.shape):
        raise ValueError(
            f"LR-domain shape mismatch: degraded_sr={tuple(sr_degraded.shape)}, lr_ref={tuple(lr_ref.shape)}."
        )

    return {
        "sam": _metric_to_float(compute_SAM(sr_degraded, lr_ref)),
        "ergas": _metric_to_float(compute_ERGAS(sr_degraded, lr_ref, scale_ratio=float(superres_scale))),
        "scc": _metric_to_float(compute_SCC(sr_degraded, lr_ref)),
    }


@torch.no_grad()
def evaluate_dataset(
    data_loader: DataLoader,
    *,
    predict_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    predict_batch_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    vis_band_order: list[int],
) -> tuple[float, dict[str, float]]:
    if predict_fn is None and predict_batch_fn is None:
        raise ValueError("Either `predict_fn` or `predict_batch_fn` must be provided.")

    total_loss_sum = 0.0
    metric_sums = {
        "psnr": 0.0,
        "ssim": 0.0,
        "gmsd_rgb": 0.0,
        "gmsd_nir": 0.0,
        "fsim_rgb": 0.0,
        "fsim_nir": 0.0,
        "lpips_rgb": 0.0,
        "lpips_nir": 0.0,
    }
    num_samples = 0

    for _, lr_up_imgs, _, hr_down_up_imgs, hr_imgs, _, _ in iter_eval_loader(
        data_loader,
        desc="Evaluating",
    ):
        batch_size = int(lr_up_imgs.shape[0])
        if batch_size <= 0:
            continue

        hr_norm = normalize_imgs(hr_imgs.float(), mean_t, std_t)
        if predict_batch_fn is not None:
            sr_norm = predict_batch_fn(lr_up_imgs, hr_down_up_imgs, hr_imgs, num_samples).float()
        else:
            sr_norm = predict_fn(lr_up_imgs, num_samples).float()

        if tuple(sr_norm.shape) != tuple(hr_norm.shape):
            raise ValueError(
                f"SR/HR shape mismatch: sr={tuple(sr_norm.shape)}, hr={tuple(hr_norm.shape)}."
            )

        loss_chunk = F.l1_loss(sr_norm, hr_norm.to(device=sr_norm.device), reduction="mean")
        metrics = compute_quality_metrics(
            sr_norm,
            hr_norm.to(device=sr_norm.device),
            mean_t,
            std_t,
            vis_band_order,
        )

        total_loss_sum += float(loss_chunk.detach().float().cpu().item()) * batch_size
        for metric_name, metric_value in metrics.items():
            metric_sums[metric_name] += float(metric_value) * batch_size
        num_samples += batch_size

    if num_samples <= 0:
        raise RuntimeError("Evaluation dataset produced no samples.")

    denom = float(num_samples)
    avg_metrics = {name: value / denom for name, value in metric_sums.items()}
    return total_loss_sum / denom, avg_metrics


@torch.no_grad()
def evaluate_spectral_dataset(
    data_loader: DataLoader,
    *,
    reference_data_loader: DataLoader | None = None,
    predict_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    predict_batch_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    superres_scale: int,
    gaussian_kernel: int,
    gaussian_sigma: float,
    down_mode: str,
) -> tuple[float, dict[str, float]]:
    if predict_fn is None and predict_batch_fn is None:
        raise ValueError("Either `predict_fn` or `predict_batch_fn` must be provided.")

    if reference_data_loader is not None:
        primary_total = len(data_loader) if hasattr(data_loader, "__len__") else None
        reference_total = len(reference_data_loader) if hasattr(reference_data_loader, "__len__") else None
        if (
            primary_total is not None
            and reference_total is not None
            and int(primary_total) != int(reference_total)
        ):
            raise ValueError(
                "Primary spectral loader and reference spectral loader must have the same "
                f"number of batches, got {primary_total} vs {reference_total}."
            )
        reference_iter = iter(reference_data_loader)
    else:
        reference_iter = None

    total_loss_sum = 0.0
    metric_sums = {
        "sam": 0.0,
        "ergas": 0.0,
        "scc": 0.0,
    }
    num_samples = 0

    for lr_imgs, lr_up_imgs, _, hr_down_up_imgs, hr_imgs, _, _ in iter_eval_loader(
        data_loader,
        desc="Spectral Evaluating",
    ):
        batch_size = int(lr_imgs.shape[0])
        if batch_size <= 0:
            continue

        if reference_iter is not None:
            try:
                reference_batch = next(reference_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Reference spectral loader ended earlier than the primary spectral loader."
                ) from exc
            lr_ref_imgs = reference_batch[0]
            if int(lr_ref_imgs.shape[0]) != batch_size:
                raise ValueError(
                    "Primary spectral batch size and reference spectral batch size must match, "
                    f"got {batch_size} vs {int(lr_ref_imgs.shape[0])}."
                )
        else:
            lr_ref_imgs = lr_imgs

        if predict_batch_fn is not None:
            sr_norm = predict_batch_fn(lr_up_imgs, hr_down_up_imgs, hr_imgs, num_samples).float()
        else:
            sr_norm = predict_fn(lr_up_imgs, num_samples).float()

        metrics = compute_spectral_metrics(
            sr_norm.float(),
            lr_ref_imgs.to(device=sr_norm.device, dtype=torch.float32),
            mean_t,
            std_t,
            superres_scale=superres_scale,
            gaussian_kernel=gaussian_kernel,
            gaussian_sigma=gaussian_sigma,
            down_mode=down_mode,
        )
        degraded_sr = degrade_hr_like_to_lr(
            denormalize_imgs(sr_norm.float(), mean_t, std_t),
            superres_scale=superres_scale,
            gaussian_kernel=gaussian_kernel,
            gaussian_sigma=gaussian_sigma,
            down_mode=down_mode,
        )
        lr_ref = torch.clamp(
            lr_ref_imgs.to(device=degraded_sr.device, dtype=torch.float32),
            min=0.0,
            max=1.0,
        )
        if tuple(degraded_sr.shape) != tuple(lr_ref.shape):
            raise ValueError(
                f"LR-domain shape mismatch: degraded_sr={tuple(degraded_sr.shape)}, lr_ref={tuple(lr_ref.shape)}."
            )

        loss_chunk = F.l1_loss(degraded_sr, lr_ref, reduction="mean")

        total_loss_sum += float(loss_chunk.detach().float().cpu().item()) * batch_size
        for metric_name, metric_value in metrics.items():
            metric_sums[metric_name] += float(metric_value) * batch_size
        num_samples += batch_size

    if num_samples <= 0:
        raise RuntimeError("Spectral evaluation dataset produced no samples.")

    denom = float(num_samples)
    avg_metrics = {name: value / denom for name, value in metric_sums.items()}
    return total_loss_sum / denom, avg_metrics


@torch.no_grad()
def save_draw_outputs(
    *,
    draw_lr_up_imgs: torch.Tensor,
    draw_hr_down_up_imgs: torch.Tensor | None = None,
    draw_hr_imgs: torch.Tensor,
    draw_projs,
    draw_geos,
    predict_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    predict_batch_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    vis_band_order: list[int],
    rgb_save_path: str,
    tif_save_path: str,
    save_tif_imgs: bool,
) -> None:
    if predict_fn is None and predict_batch_fn is None:
        raise ValueError("Either `predict_fn` or `predict_batch_fn` must be provided.")

    lr_up_denorm = denormalize_imgs(
        normalize_imgs(draw_lr_up_imgs.float(), mean_t, std_t),
        mean_t,
        std_t,
    ).cpu()
    hr_denorm = denormalize_imgs(
        normalize_imgs(draw_hr_imgs.float(), mean_t, std_t),
        mean_t,
        std_t,
    ).cpu()
    if predict_batch_fn is not None:
        if draw_hr_down_up_imgs is None:
            raise ValueError("`draw_hr_down_up_imgs` is required when `predict_batch_fn` is used.")
        sr_norm = predict_batch_fn(
            draw_lr_up_imgs,
            draw_hr_down_up_imgs,
            draw_hr_imgs,
            0,
        ).detach().float().cpu()
    else:
        sr_norm = predict_fn(draw_lr_up_imgs, 0).detach().float().cpu()
    sr_denorm = denormalize_imgs(sr_norm.float(), mean_t, std_t).cpu()

    preview_count = min(3, int(sr_denorm.shape[0]))
    result_rgb_imgs = torch.cat(
        [
            lr_up_denorm[:preview_count],
            sr_denorm[:preview_count],
            hr_denorm[:preview_count],
        ],
        dim=0,
    )
    save_rgb_datas(
        result_rgb_imgs[:, vis_band_order, :, :].cpu(),
        nrow=preview_count,
        savepath=rgb_save_path,
        is_showminmax=True,
        use_sqrt=True,
    )

    if save_tif_imgs and preview_count > 0:
        save_tif_datas(
            sr_denorm[:preview_count].cpu(),
            savepath=tif_save_path,
            img_sources_num=1,
            geotransforms=collated_geos_to_batch(draw_geos)[:preview_count],
            projections=list(draw_projs)[:preview_count],
            is_showminmax=True,
        )


def write_evaluation_info(
    *,
    report_path: str | Path,
    tag: str,
    dataset_dirs: dict,
    test_dataset_size: int,
    draw_dataset_size: int,
    test_loss: float,
    test_metrics: dict[str, float],
    diffusion_checkpoint_path: str,
    diffusion_checkpoint_epoch: int,
    diffusion_loaded_ema: bool,
    diffusion_model: torch.nn.Module,
    vae_checkpoint_path: str,
    vae_checkpoint_epoch: int,
    vae_loaded_ema: bool,
    vae_model: torch.nn.Module,
    save_tif_imgs: bool,
    rgb_dir: str,
    tif_dir: str,
) -> None:
    diffusion_epoch_idx = int(diffusion_checkpoint_epoch)
    diffusion_epoch_num = diffusion_epoch_idx + 1 if diffusion_epoch_idx >= 0 else -1
    vae_epoch_idx = int(vae_checkpoint_epoch)
    vae_epoch_num = vae_epoch_idx + 1 if vae_epoch_idx >= 0 else -1

    lines = [
        f"=== {tag} Spatial Evaluation Report ===",
        f"Diffusion checkpoint: {diffusion_checkpoint_path}",
        f"Diffusion loaded EMA: {bool(diffusion_loaded_ema)}",
        f"Diffusion checkpoint epoch index: {diffusion_epoch_idx}",
        f"Diffusion checkpoint epoch number: {diffusion_epoch_num if diffusion_epoch_num > 0 else 'N/A'}",
        f"Diffusion parameter count: {count_model_parameters(diffusion_model)}",
        f"VAE checkpoint: {vae_checkpoint_path}",
        f"VAE loaded EMA: {bool(vae_loaded_ema)}",
        f"VAE checkpoint epoch index: {vae_epoch_idx}",
        f"VAE checkpoint epoch number: {vae_epoch_num if vae_epoch_num > 0 else 'N/A'}",
        f"VAE parameter count: {count_model_parameters(vae_model)}",
        f"Test root: {dataset_dirs['rootdir_list_forTest']}",
        f"Test split: {dataset_dirs.get('split_name_forTest', None)}",
        f"Test dataset size: {int(test_dataset_size)}",
        f"Draw root: {dataset_dirs['rootdir_list_forDraw']}",
        f"Draw split: {dataset_dirs.get('split_name_forDraw', None)}",
        f"Draw dataset size: {int(draw_dataset_size)}",
        f"Save TIF images: {bool(save_tif_imgs)}",
        f"RGB output dir: {rgb_dir}",
        f"TIF output dir: {tif_dir}",
        f"Spatial evaluation report path: {report_path}",
        f"Test L1: {format_metric_value(test_loss)}",
        "Quality Metrics:",
        f"  PSNR: {format_metric_value(test_metrics['psnr'])}",
        f"  SSIM: {format_metric_value(test_metrics['ssim'])}",
        f"  GMSD_RGB: {format_metric_value(test_metrics['gmsd_rgb'])}",
        f"  GMSD_NIR: {format_metric_value(test_metrics['gmsd_nir'])}",
        f"  FSIM_RGB: {format_metric_value(test_metrics['fsim_rgb'])}",
        f"  FSIM_NIR: {format_metric_value(test_metrics['fsim_nir'])}",
        f"  LPIPS_RGB: {format_metric_value(test_metrics['lpips_rgb'])}",
        f"  LPIPS_NIR: {format_metric_value(test_metrics['lpips_nir'])}",
        f"Quality Metrics Summary => {format_metric_summary(test_metrics)}",
    ]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


def write_spectral_evaluation_info(
    *,
    report_path: str | Path,
    tag: str,
    dataset_dirs: dict,
    test_dataset_size: int,
    draw_dataset_size: int,
    test_loss: float,
    test_metrics: dict[str, float],
    diffusion_checkpoint_path: str,
    diffusion_checkpoint_epoch: int,
    diffusion_loaded_ema: bool,
    diffusion_model: torch.nn.Module,
    vae_checkpoint_path: str,
    vae_checkpoint_epoch: int,
    vae_loaded_ema: bool,
    vae_model: torch.nn.Module,
    save_tif_imgs: bool,
    rgb_dir: str,
    tif_dir: str,
    superres_scale: int,
    gaussian_kernel: int,
    gaussian_sigma: float,
    down_mode: str,
    spectral_reference_mode: str = "D_hr(SR) vs LR_raw",
) -> None:
    diffusion_epoch_idx = int(diffusion_checkpoint_epoch)
    diffusion_epoch_num = diffusion_epoch_idx + 1 if diffusion_epoch_idx >= 0 else -1
    vae_epoch_idx = int(vae_checkpoint_epoch)
    vae_epoch_num = vae_epoch_idx + 1 if vae_epoch_idx >= 0 else -1

    lines = [
        f"=== {tag} Spectral Evaluation Report ===",
        f"Diffusion checkpoint: {diffusion_checkpoint_path}",
        f"Diffusion loaded EMA: {bool(diffusion_loaded_ema)}",
        f"Diffusion checkpoint epoch index: {diffusion_epoch_idx}",
        f"Diffusion checkpoint epoch number: {diffusion_epoch_num if diffusion_epoch_num > 0 else 'N/A'}",
        f"Diffusion parameter count: {count_model_parameters(diffusion_model)}",
        f"VAE checkpoint: {vae_checkpoint_path}",
        f"VAE loaded EMA: {bool(vae_loaded_ema)}",
        f"VAE checkpoint epoch index: {vae_epoch_idx}",
        f"VAE checkpoint epoch number: {vae_epoch_num if vae_epoch_num > 0 else 'N/A'}",
        f"VAE parameter count: {count_model_parameters(vae_model)}",
        f"Test root: {dataset_dirs['rootdir_list_forTest']}",
        f"Test split: {dataset_dirs.get('split_name_forTest', None)}",
        f"Test dataset size: {int(test_dataset_size)}",
        f"Draw root: {dataset_dirs['rootdir_list_forDraw']}",
        f"Draw split: {dataset_dirs.get('split_name_forDraw', None)}",
        f"Draw dataset size: {int(draw_dataset_size)}",
        f"Spectral degradation scale: {int(superres_scale)}",
        f"Spectral degradation gaussian kernel: {int(gaussian_kernel)}",
        f"Spectral degradation gaussian sigma: {float(gaussian_sigma)}",
        f"Spectral degradation down mode: {str(down_mode)}",
        f"Spectral reference mode: {str(spectral_reference_mode)}",
        f"Save TIF images: {bool(save_tif_imgs)}",
        f"RGB output dir: {rgb_dir}",
        f"TIF output dir: {tif_dir}",
        f"Spectral evaluation report path: {report_path}",
        f"LR-domain L1: {format_metric_value(test_loss)}",
        "Spectral Metrics:",
        f"  SAM: {format_metric_value(test_metrics['sam'])}",
        f"  ERGAS: {format_metric_value(test_metrics['ergas'])}",
        f"  SCC: {format_metric_value(test_metrics['scc'])}",
        f"Spectral Metrics Summary => {format_spectral_metric_summary(test_metrics)}",
    ]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
