"""Generate paper-ready spatial and spectral qualitative comparisons.

The parent process runs one isolated child process per dataset so every model
imports the matching YAML configuration and checkpoint. Representative samples
are selected from the test split by RGB edge/contrast score.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "mathtext.rm": "Times New Roman",
        "axes.unicode_minus": False,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
    }
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "论文章节" / "与其他模型对比的定性展示"
RENDER_ROOT = Path(os.environ.get("CSUA_LDM_PAPER_RENDER_ROOT", str(OUTPUT_ROOT)))
DATASET_YAMLS = {
    "oli2msi": ROOT / "configs" / "datasets" / "oli2msi.yaml",
    "sen2naip": ROOT / "configs" / "datasets" / "sen2naip.yaml",
    "fy3fs2": ROOT / "configs" / "datasets" / "fy3fs2.yaml",
}
SPECTRAL_BAND_LAYOUTS = {
    # Every spectral figure uses the same semantic B, G, R, (NIR) order.
    "oli2msi": ((2, 1, 0), ("B", "G", "R")),
    "sen2naip": ((2, 1, 0, 3), ("B", "G", "R", "NIR")),
    "fy3fs2": ((0, 1, 2, 3), ("B", "G", "R", "NIR")),
}
SELECTED_TEST_INDICES = {
    "oli2msi": [62, 25],
    "sen2naip": [50, 32],
    "fy3fs2": [140, 141],
}
CSUA_SAMPLING_MODE = "uncguided"
METHOD_DISPLAY_NAMES = {
    "CSUA_LDM (style-lr_up)": "CSUA-LDM (style-lr_up)",
    "CSUA_LDM (style-hr)": "CSUA-LDM (style-hr)",
}
MODEL_SCRIPTS = {
    "DSen2": ROOT / "Comparison_Models" / "DSen2" / "CSUA_LDM_DSen2_Code" / "CSUA_LDM_DSen2_SpatialEvaluate.py",
    "EDSR": ROOT / "Comparison_Models" / "EDSR" / "CSUA_LDM_EDSR_Code" / "CSUA_LDM_EDSR_SpatialEvaluate.py",
    "ESRGAN": ROOT / "Comparison_Models" / "ESRGAN" / "CSUA_LDM_ESRGAN_Code" / "CSUA_LDM_ESRGAN_SpatialEvaluate.py",
    "RCAN": ROOT / "Comparison_Models" / "RCAN" / "CSUA_LDM_RCAN_Code" / "CSUA_LDM_RCAN_SpatialEvaluate.py",
    "ResShift": ROOT / "Comparison_Models" / "ResShift" / "CSUA_LDM_ResShift_Code" / "CSUA_LDM_ResShift_SpatialEvaluate.py",
    "SwinIR": ROOT / "Comparison_Models" / "SwinIR" / "CSUA_LDM_SwinIR_Code" / "CSUA_LDM_SwinIR_SpatialEvaluate.py",
}


def _load_module(name: str, path: Path):
    candidates = [path.parent, path.parent.parent]
    rcan_code = path.parent.parent / "RCAN_TrainCode" / "code"
    if rcan_code.is_dir():
        candidates.append(rcan_code)
        for module_name in tuple(sys.modules):
            if module_name == "model" or module_name.startswith("model."):
                del sys.modules[module_name]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_model(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_ema(model: torch.nn.Module, checkpoint: str | Path):
    from CSUA_LDM_Evaluate_Common import load_model_with_ema_fallback

    loaded, _, _ = load_model_with_ema_fallback(model, checkpoint)
    loaded.eval()
    return loaded


def _denorm(x: torch.Tensor) -> torch.Tensor:
    from Main_Model.CSUA_LDM_Config import hr_ch_mean, hr_ch_std
    from CSUA_LDM_Evaluate_Common import denormalize_imgs

    return denormalize_imgs(x.float(), hr_ch_mean, hr_ch_std).clamp(0.0, 1.0)


def _rgb(x: torch.Tensor, bands: list[int]) -> np.ndarray:
    arr = x.detach().float().cpu()[bands].permute(1, 2, 0).numpy()
    return np.sqrt(np.clip(arr, 0.0, 1.0))


def _select_indices(dataset, bands: list[int], count: int) -> list[int]:
    candidate_count = min(len(dataset), 36)
    candidates = np.unique(np.linspace(0, len(dataset) - 1, candidate_count).round().astype(int))
    scored: list[tuple[float, int]] = []
    for index in candidates:
        hr = dataset[int(index)][4].float().clamp(0.0, 1.0)
        gray = hr[bands].mean(dim=0)
        dx = gray[:, 1:] - gray[:, :-1]
        dy = gray[1:, :] - gray[:-1, :]
        score = float(dx.abs().mean() + dy.abs().mean() + 0.25 * gray.std())
        scored.append((score, int(index)))
    scored.sort(reverse=True)
    selected: list[int] = []
    min_gap = max(len(dataset) // 12, 1)
    for _, index in scored:
        if all(abs(index - previous) >= min_gap for previous in selected):
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def _best_roi(hr: torch.Tensor, bands: list[int], fraction: float = 0.63) -> tuple[int, int, int]:
    gray = hr[bands].mean(dim=0, keepdim=True).unsqueeze(0)
    grad = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    grad += F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    size = max(int(min(hr.shape[-2:]) * fraction), 32)
    pooled = F.avg_pool2d(grad, kernel_size=size, stride=1)
    flat_index = int(pooled.flatten().argmax())
    out_w = pooled.shape[-1]
    return flat_index // out_w, flat_index % out_w, size


def _resize_roi_around_center(
    roi: tuple[int, int, int],
    image_height: int,
    image_width: int,
    fraction: float,
) -> tuple[int, int, int]:
    y, x, original_size = roi
    target_size = max(int(min(image_height, image_width) * fraction), 32)
    # Matching parity preserves the original ROI center exactly on the pixel grid.
    if (original_size - target_size) % 2 != 0:
        target_size -= 1
    target_size = max(min(target_size, image_height, image_width), 1)
    offset = (original_size - target_size) // 2
    resized_y = min(max(y + offset, 0), image_height - target_size)
    resized_x = min(max(x + offset, 0), image_width - target_size)
    return resized_y, resized_x, target_size


def _save_candidate_sheet(dataset_name: str, dataset, indices: list[int], bands: list[int]) -> None:
    columns = 4
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.7 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        hr = dataset[index][4]
        ax.imshow(_rgb(hr, bands))
        ax.set_title(f"test index {index}\n{Path(dataset.hr_img_paths[index]).name}", fontsize=8)
        ax.axis("off")
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        RENDER_ROOT / f"{dataset_name}_Test_Candidate_ContactSheet.jpg",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


@torch.inference_mode()
def _predict_standard_models(lr: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    from Main_Model.CSUA_LDM_Config import CSUA_LDM_SUPERRES_SCALE, hr_ch_mean, hr_ch_std
    from CSUA_LDM_Utils import norm_img

    outputs: dict[str, torch.Tensor] = {
        "Bilinear": F.interpolate(
            lr,
            scale_factor=float(CSUA_LDM_SUPERRES_SCALE),
            mode="bilinear",
            align_corners=False,
        )
    }
    for label in ("DSen2", "EDSR", "ESRGAN", "RCAN", "SwinIR"):
        module = _load_module(f"qualitative_{label.lower()}", MODEL_SCRIPTS[label])
        if label == "DSen2":
            model = module.build_dsen2_model().to(device)
            checkpoint = module.CSUA_LDM_DSEN2_MODEL_SAVEPATH
        elif label == "EDSR":
            model = module.build_edsr_model().to(device)
            checkpoint = module.CSUA_LDM_EDSR_MODEL_SAVEPATH
        elif label == "ESRGAN":
            model = module.build_rrdb_generator(
                in_nc=int(module.CSUA_LDM_ESRGAN_INPUT_CHANNELS),
                out_nc=int(module.CSUA_LDM_ESRGAN_INPUT_CHANNELS),
                nf=int(module.CSUA_LDM_ESRGAN_N_FEATS),
                nb=int(module.CSUA_LDM_ESRGAN_N_RRDB),
                gc=int(module.CSUA_LDM_ESRGAN_GC),
                scale=int(module.CSUA_LDM_ESRGAN_SCALE),
            ).to(device)
            checkpoint = module.CSUA_LDM_ESRGAN_FINETUNE_MODEL_SAVEPATH
        elif label == "RCAN":
            model = module.build_rcan_model().to(device)
            checkpoint = module.CSUA_LDM_RCAN_MODEL_SAVEPATH
        else:
            model = module.build_swinir_model().to(device)
            checkpoint = module.CSUA_LDM_SWINIR_MODEL_SAVEPATH
        model = _load_ema(model, checkpoint)
        outputs[label] = _denorm(model(norm_img(lr.to(device), hr_ch_mean, hr_ch_std)).cpu())
        _free_model(model)
    return outputs


@torch.inference_mode()
def _predict_resshift(lr: torch.Tensor, hr: torch.Tensor, device: torch.device) -> torch.Tensor:
    module = _load_module("qualitative_resshift", MODEL_SCRIPTS["ResShift"])
    model = module.build_resshift_model().to(device)
    diffusion = module.build_resshift_diffusion()
    vae = module.ResShiftFirstStageAdapter(device=device, load_ema=True)
    module.load_resshift_checkpoint(
        module.CSUA_LDM_RESSHIFT_MODEL_SAVEPATH,
        model=model,
        load_ema=True,
        strict=True,
        map_location=device,
    )
    prepared = module.prepare_latent_batch(lr, hr, vae=vae, device=device)
    latent = module.sample_resshift_latents(model, diffusion, prepared["z_y"], use_amp=False)
    result = module.denormalize_hr_imgs(module.decode_latent_to_hr_norm(latent, vae)).cpu().clamp(0.0, 1.0)
    _free_model(model)
    del diffusion, vae, prepared, latent
    _free_model()
    return result


@torch.inference_mode()
def _predict_csua(
    lr_up: torch.Tensor,
    hr_down_up: torch.Tensor,
    hr: torch.Tensor,
    device: torch.device,
    style_source: str,
) -> torch.Tensor:
    spatial = _load_module(
        "qualitative_csua",
        ROOT / "Main_Model" / "CSUA_LDM_SpatialEvaluate.py",
    )
    from Main_Model.CSUA_LDM_Config import CSUA_LDM_VAE_SAVEPATH, RANDOM_SEED
    from CSUA_LDM_VAE_helpers import build_vae_model

    vae, _, _ = build_vae_model(Path(CSUA_LDM_VAE_SAVEPATH), device, load_ema=True)
    diffusion, _, _ = spatial.build_diffusion_model(device)
    spatial.configure_sampling_uncertainty_guidance("on")
    result_norm = spatial.run_diffusion_chain(
        lr_up.to(device),
        vae_model=vae,
        diffusion_model=diffusion,
        device=device,
        hr_down_up_img=hr_down_up.to(device),
        hr_img=hr.to(device),
        style_source=style_source,
        show_pbar=True,
        return_uncertainty=False,
        init_seed=int(RANDOM_SEED),
    )["images"]["hr"].detach().float().cpu()
    result = _denorm(result_norm).cpu()
    _free_model(vae)
    _free_model(diffusion)
    return result


def _roi_psnr_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    rgb_indices: list[int],
) -> tuple[float, float]:
    from CSUA_LDM_Quality_Metrics import compute_LPIPS, compute_PSNR

    metric_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_batch = pred.float().clamp(0.0, 1.0).unsqueeze(0).to(metric_device)
    target_batch = target.float().clamp(0.0, 1.0).unsqueeze(0).to(metric_device)
    psnr = float(compute_PSNR(pred_batch, target_batch).detach().cpu().item())
    lpips_rgb, _ = compute_LPIPS(pred_batch, target_batch, rgb_indices=rgb_indices[:3])
    if lpips_rgb is None:
        raise RuntimeError("LPIPS is unavailable; install or repair the `lpips` backend.")
    return psnr, float(lpips_rgb.detach().cpu().item())


def _save_spatial_figure(dataset_name: str, samples: list[dict], bands: list[int], out_dir: Path) -> None:
    methods = [
        "HR",
        "Bilinear",
        "DSen2",
        "EDSR",
        "RCAN",
        "ESRGAN",
        "SwinIR",
        "ResShift",
        "CSUA_LDM (style-lr_up)",
        "CSUA_LDM (style-hr)",
    ]
    fig = plt.figure(figsize=(11.9, 3.92 * len(samples)))
    grid = fig.add_gridspec(
        len(samples),
        6,
        # Match the large image height to the two crop rows, including their gap.
        width_ratios=[2.21, 1, 1, 1, 1, 1],
        hspace=0.13,
        wspace=0.0,
    )
    alignment_axes: list[tuple[plt.Axes, list[plt.Axes]]] = []
    for row, sample in enumerate(samples):
        y, x, size = _resize_roi_around_center(
            sample["roi"],
            image_height=int(sample["HR"].shape[-2]),
            image_width=int(sample["HR"].shape[-1]),
            fraction=0.40,
        )
        if dataset_name == "sen2naip":
            image_height = int(sample["HR"].shape[-2])
            image_width = int(sample["HR"].shape[-1])
            shift = max(int(round(min(image_height, image_width) * 0.05)), 1)
            y = min(y + shift, image_height - size)
            if row == 1:
                x = max(x - 2 * shift, 0)
        elif dataset_name == "fy3fs2" and row == 0:
            image_height = int(sample["HR"].shape[-2])
            image_width = int(sample["HR"].shape[-1])
            shift = max(int(round(min(image_height, image_width) * 0.05)), 1)
            y = max(y - 2 * shift, 0)
            x = min(x + 2 * shift, image_width - size)
        ax = fig.add_subplot(grid[row, 0])
        ax.set_anchor("N")
        ax.imshow(_rgb(sample["HR"], bands))
        # Align the frame with pixel edges so ROIs touching an image boundary remain visible.
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                size,
                size,
                fill=False,
                edgecolor="#ff3b30",
                linewidth=2.2,
            )
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(
            f"test_img_{sample['index']} from {dataset_name.upper()}",
            fontsize=12.0,
            labelpad=5,
        )
        sub = grid[row, 1:].subgridspec(
            4,
            5,
            width_ratios=[1, 1, 1, 1, 1],
            height_ratios=[1, 0.235, 1, 0.235],
            wspace=-0.03,
            hspace=0.0,
        )
        hr_crop = sample["HR"][:, y:y + size, x:x + size]
        crop_axes: list[plt.Axes] = []
        for pos, method in enumerate(methods):
            crop = sample[method][:, y:y + size, x:x + size]
            image_row = (pos // 5) * 2
            column = pos % 5
            crop_ax = fig.add_subplot(sub[image_row, column])
            crop_axes.append(crop_ax)
            crop_ax.set_anchor("N")
            crop_ax.imshow(_rgb(crop, bands), interpolation="nearest")
            crop_ax.set_xticks([])
            crop_ax.set_yticks([])
            for spine in crop_ax.spines.values():
                spine.set_visible(False)
            if method == "HR":
                caption = "HR\nPSNR/LPIPS"
            else:
                psnr, lpips = _roi_psnr_lpips(crop, hr_crop, bands)
                caption = f"{METHOD_DISPLAY_NAMES.get(method, method)}\n{psnr:.2f}/{lpips:.4f}"
            caption_ax = fig.add_subplot(sub[image_row + 1, column])
            caption_ax.axis("off")
            caption_ax.text(
                0.5,
                0.48,
                caption,
                fontsize=10.0,
                linespacing=1.15,
                ha="center",
                va="center",
                transform=caption_ax.transAxes,
            )
        alignment_axes.append((ax, crop_axes))
    path = out_dir / f"{dataset_name}_Spatial_Qualitative_Comparison.jpg"
    fig.subplots_adjust(left=0.012, right=0.992, top=0.985, bottom=0.035)
    fig.canvas.draw()
    figure_width, figure_height = fig.get_size_inches()
    large_to_crop_gap = 0.020
    for large_ax, crop_axes in alignment_axes:
        top = crop_axes[0].get_position().y1
        bottom = crop_axes[-1].get_position().y0
        height = top - bottom
        width = height * figure_height / figure_width
        x_position = crop_axes[0].get_position().x0 - width - large_to_crop_gap
        large_ax.set_position(
            [x_position, bottom, width, height],
            which="both",
        )
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def _scale_roi_to_lr(roi: tuple[int, int, int], scale: int) -> tuple[int, int, int, int]:
    y, x, size = roi
    y0 = y // scale
    x0 = x // scale
    y1 = (y + size + scale - 1) // scale
    x1 = (x + size + scale - 1) // scale
    return y0, x0, y1 - y0, x1 - x0


def _spectral_vector(
    image: torch.Tensor,
    roi: tuple[int, int, int, int],
    channel_order: tuple[int, ...],
) -> np.ndarray:
    y, x, height, width = roi
    vector = (
        image[list(channel_order), y:y + height, x:x + width]
        .mean(dim=(1, 2))
        .float()
        .cpu()
        .numpy()
    )
    return vector


def _save_spectral_figure(dataset_name: str, samples: list[dict], bands: list[int], out_dir: Path) -> None:
    from Main_Model.CSUA_LDM_Config import (
        CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        CSUA_LDM_SUPERRES_SCALE,
    )
    from CSUA_LDM_Evaluate_Common import degrade_hr_like_to_lr

    if dataset_name not in SPECTRAL_BAND_LAYOUTS:
        raise KeyError(f"Missing spectral band layout for dataset: {dataset_name}")
    channel_order, band_labels = SPECTRAL_BAND_LAYOUTS[dataset_name]
    band_positions = np.arange(len(band_labels))
    shown = [
        "Bilinear",
        "DSen2",
        "EDSR",
        "RCAN",
        "ESRGAN",
        "SwinIR",
        "ResShift",
        "CSUA_LDM (style-lr_up)",
        "HR",
    ]
    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(8.5, 3.06 * len(samples)),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.70, 1.30]},
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(shown)))
    for row, sample in enumerate(samples):
        superres_scale = int(CSUA_LDM_SUPERRES_SCALE)
        lr_height = int(sample["LR"].shape[-2])
        lr_width = int(sample["LR"].shape[-1])
        lr_roi = (0, 0, lr_height, lr_width)
        axes[row, 0].imshow(_rgb(sample["LR"], bands), interpolation="nearest")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)
        axes[row, 0].set_xlabel(
            f"test_img_{sample['index']} from {dataset_name.upper()}",
            fontsize=12.0,
            labelpad=5,
        )
        reference = _spectral_vector(sample["LR"], lr_roi, channel_order)
        reference_scale = max(float(np.max(np.abs(reference))), 1e-8)
        reference = reference / reference_scale
        axes[row, 1].plot(
            band_positions, reference,
            marker="s", linewidth=2.5, color="black", label="LR reference",
        )
        for color, method in zip(colors, shown):
            degraded = degrade_hr_like_to_lr(
                sample[method].unsqueeze(0),
                superres_scale=superres_scale,
                gaussian_kernel=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
                gaussian_sigma=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
                down_mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
            )[0]
            vector = _spectral_vector(degraded, lr_roi, channel_order) / reference_scale
            axes[row, 1].plot(
                band_positions,
                vector,
                marker="o",
                linewidth=1.8,
                color=color,
                label=METHOD_DISPLAY_NAMES.get(method, method),
            )
        axes[row, 1].set_xticks(band_positions, band_labels)
        axes[row, 1].set_xlabel("Spectral band", fontsize=12.0, labelpad=2)
        axes[row, 1].set_ylabel("LR-normalized reflectance", fontsize=12.0)
        axes[row, 1].tick_params(axis="both", labelsize=10.0)
        axes[row, 1].grid(alpha=0.25)
        if row == 0:
            y_min, y_max = axes[row, 1].get_ylim()
            headroom = {
                "oli2msi": 0.65,
                "sen2naip": 0.40,
                "fy3fs2": 0.25,
            }[dataset_name]
            axes[row, 1].set_ylim(y_min, y_max + headroom * (y_max - y_min))
        elif dataset_name == "sen2naip" and row == 1:
            y_min, y_max = axes[row, 1].get_ylim()
            axes[row, 1].set_ylim(y_min, y_max + 0.45 * (y_max - y_min))
        legend_location = "upper right" if dataset_name == "oli2msi" else "upper left"
        axes[row, 1].legend(fontsize=10.0, ncol=2, loc=legend_location)
    path = out_dir / f"{dataset_name}_Spectral_Qualitative_Comparison.jpg"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _run_child(
    dataset_name: str,
    sample_count: int,
    manual_indices: list[int] | None = None,
    preview_only: bool = False,
    figure_type: str = "both",
) -> None:
    from Main_Model.CSUA_LDM_Config import (
        CSUA_LDM_DATASET_DIRS,
        CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
        CSUA_LDM_HR_CROP_SIZE,
        CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        CSUA_LDM_HR_DEGRADATION_UP_MODE,
        CSUA_LDM_HR_DOWN_MODE,
        CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
        CSUA_LDM_HR_GAUSSIAN_KERNEL,
        CSUA_LDM_HR_GAUSSIAN_SIGMA,
        CSUA_LDM_HR_UP_MODE,
        CSUA_LDM_INPUT_CHANNELS,
        CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        CSUA_LDM_LR_GAUSSIAN_KERNEL,
        CSUA_LDM_LR_GAUSSIAN_SIGMA,
        CSUA_LDM_LR_UP_MODE,
        CSUA_LDM_LR_CROP_SIZE,
        CSUA_LDM_SUPERRES_SCALE,
        CSUA_LDM_VIS_BAND_ORDER,
        DEVICE,
    )
    from CSUA_LDM_Dataset import SuperResolutionDataset
    from CSUA_LDM_Utils import set_random_seed

    set_random_seed(999, deterministic=False)
    device = torch.device(DEVICE)
    dataset = SuperResolutionDataset(
        root_dir=CSUA_LDM_DATASET_DIRS["rootdir_list_forTest"],
        is_train=False,
        split_name=CSUA_LDM_DATASET_DIRS["split_name_forTest"],
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
        superres_scale=CSUA_LDM_SUPERRES_SCALE,
        lr_crop_size=CSUA_LDM_LR_CROP_SIZE,
        hr_crop_size=CSUA_LDM_HR_CROP_SIZE,
        input_channels=CSUA_LDM_INPUT_CHANNELS,
        hr_degradation_gaussian_kernel=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        hr_degradation_gaussian_sigma=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        hr_degradation_down_mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        hr_degradation_up_mode=CSUA_LDM_HR_DEGRADATION_UP_MODE,
        soft_hr_gaussian_blur_enabled=CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
        soft_hr_gaussian_kernel=CSUA_LDM_HR_GAUSSIAN_KERNEL,
        soft_hr_gaussian_sigma=CSUA_LDM_HR_GAUSSIAN_SIGMA,
        soft_hr_down_mode=CSUA_LDM_HR_DOWN_MODE,
        soft_hr_up_mode=CSUA_LDM_HR_UP_MODE,
        lr_up_mode=CSUA_LDM_LR_UP_MODE,
        lr_gaussian_blur_enabled=CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        lr_gaussian_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
        **CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )
    raw_lr_dataset = SuperResolutionDataset(
        root_dir=CSUA_LDM_DATASET_DIRS["rootdir_list_forTest"],
        is_train=False,
        split_name=CSUA_LDM_DATASET_DIRS["split_name_forTest"],
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
        superres_scale=CSUA_LDM_SUPERRES_SCALE,
        lr_crop_size=CSUA_LDM_LR_CROP_SIZE,
        hr_crop_size=CSUA_LDM_HR_CROP_SIZE,
        input_channels=CSUA_LDM_INPUT_CHANNELS,
        lr_up_mode=CSUA_LDM_LR_UP_MODE,
        lr_gaussian_blur_enabled=False,
        **CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )
    if manual_indices:
        indices = [int(index) for index in manual_indices]
        invalid = [index for index in indices if index < 0 or index >= len(dataset)]
        if invalid:
            raise IndexError(f"Test indices out of range for {dataset_name}: {invalid}")
    else:
        indices = _select_indices(dataset, list(CSUA_LDM_VIS_BAND_ORDER), sample_count)
    if preview_only:
        _save_candidate_sheet(dataset_name, dataset, indices, list(CSUA_LDM_VIS_BAND_ORDER))
        return
    batch = [dataset[index] for index in indices]
    raw_batch = [raw_lr_dataset[index] for index in indices]
    lr = torch.stack([item[0] for item in raw_batch]).float()
    lr_up = torch.stack([item[1] for item in batch]).float()
    hr_down_up = torch.stack([item[3] for item in batch]).float()
    hr = torch.stack([item[4] for item in batch]).float()

    cache_dir = OUTPUT_ROOT / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{dataset_name}_indices-{'-'.join(str(index) for index in indices)}"
        f"_csua-{CSUA_SAMPLING_MODE}.pt"
    )
    if cache_path.is_file():
        outputs = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        outputs = _predict_standard_models(lr, device)
        outputs["ResShift"] = _predict_resshift(lr, hr, device)
        outputs["CSUA_LDM (style-lr_up)"] = _predict_csua(lr_up, hr_down_up, hr, device, "lr_up")
        outputs["CSUA_LDM (style-hr)"] = _predict_csua(lr_up, hr_down_up, hr, device, "hr")
        outputs["HR"] = hr
        outputs["LR"] = lr
        torch.save({name: value.detach().float().cpu() for name, value in outputs.items()}, cache_path)

    samples: list[dict] = []
    for position, index in enumerate(indices):
        sample = {name: value[position] for name, value in outputs.items()}
        sample["index"] = int(index)
        roi = _best_roi(hr[position], list(CSUA_LDM_VIS_BAND_ORDER))
        if dataset_name == "fy3fs2" and position == 0:
            y, x, size = roi
            roi = (max(y - 2 * int(CSUA_LDM_SUPERRES_SCALE), 0), x, size)
        elif dataset_name == "fy3fs2" and position == 1:
            y, x, size = roi
            max_y = int(hr[position].shape[-2]) - size
            roi = (min(y + int(CSUA_LDM_SUPERRES_SCALE), max_y), x, size)
        sample["roi"] = roi
        samples.append(sample)

    if figure_type in {"spatial", "both"}:
        spatial_dir = RENDER_ROOT / "Spatial"
        spatial_dir.mkdir(parents=True, exist_ok=True)
        _save_spatial_figure(dataset_name, samples, list(CSUA_LDM_VIS_BAND_ORDER), spatial_dir)
    if figure_type in {"spectral", "both"}:
        spectral_dir = RENDER_ROOT / "Spectral"
        spectral_dir.mkdir(parents=True, exist_ok=True)
        _save_spectral_figure(dataset_name, samples, list(CSUA_LDM_VIS_BAND_ORDER), spectral_dir)
    (RENDER_ROOT / f"{dataset_name}_selected_test_indices.txt").write_text(
        "\n".join(
            f"index={index}\tfile={Path(dataset.hr_img_paths[index]).name}"
            for index in indices
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_YAMLS), default=None)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated test indices. Overrides automatic representative selection.",
    )
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--figure-type",
        choices=("spatial", "spectral", "both"),
        default="both",
        help="Select which qualitative comparison figure type to render.",
    )
    args = parser.parse_args()
    manual_indices = [int(value.strip()) for value in args.indices.split(",") if value.strip()]
    if args.dataset:
        os.environ["CSUA_LDM_DATASET_YAML_OVERRIDE"] = str(DATASET_YAMLS[args.dataset])
        selected_indices = manual_indices or (
            None if args.preview_only else SELECTED_TEST_INDICES.get(args.dataset)
        )
        _run_child(
            args.dataset,
            max(int(args.samples_per_dataset), 1),
            manual_indices=selected_indices,
            preview_only=bool(args.preview_only),
            figure_type=args.figure_type,
        )
        return
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_name, yaml_path in DATASET_YAMLS.items():
        env = os.environ.copy()
        env["CSUA_LDM_DATASET_YAML_OVERRIDE"] = str(yaml_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset",
            dataset_name,
            "--samples-per-dataset",
            str(args.samples_per_dataset),
            "--figure-type",
            args.figure_type,
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
