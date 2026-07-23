"""Generate spatial and spectral qualitative figures for the ablation study.

The script uses the same test indices and ROI rule as the model-comparison
figures. Main reuses the existing uncertainty-guided comparison cache, while
the remaining EMA ablation variants are inferred once and cached locally.
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
from PIL import Image

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
OUTPUT_ROOT = ROOT / "论文章节" / "消融试验的定性展示"
RENDER_ROOT = Path(os.environ.get("CSUA_LDM_PAPER_RENDER_ROOT", str(OUTPUT_ROOT)))
COMPARISON_CACHE_ROOT = ROOT / "论文章节" / "与其他模型对比的定性展示" / "_cache"
DATASET_YAMLS = {
    "oli2msi": ROOT / "configs" / "datasets" / "oli2msi.yaml",
    "sen2naip": ROOT / "configs" / "datasets" / "sen2naip.yaml",
    "fy3fs2": ROOT / "configs" / "datasets" / "fy3fs2.yaml",
}
SELECTED_TEST_INDICES = {
    "oli2msi": [62, 25],
    "sen2naip": [50, 32],
    "fy3fs2": [140, 141],
}
SPECTRAL_BAND_LAYOUTS = {
    "oli2msi": ((2, 1, 0), ("B", "G", "R")),
    "sen2naip": ((2, 1, 0, 3), ("B", "G", "R", "NIR")),
    "fy3fs2": ((0, 1, 2, 3), ("B", "G", "R", "NIR")),
}

VARIANT_SCRIPTS = {
    "Main-DDPM": ROOT / "Main_Model" / "CSUA_LDM_SpatialEvaluate.py",
    "NoDisent": ROOT / "Ablation_Models" / "NoDisent" / "CSUA_LDM_NoDisent_SpatialEvaluate.py",
    "NoDisent-DDPM": ROOT / "Ablation_Models" / "NoDisent" / "CSUA_LDM_NoDisent_SpatialEvaluate.py",
    "NoUnc": ROOT / "Ablation_Models" / "NoUnc" / "CSUA_LDM_NoUnc_SpatialEvaluate.py",
    "Vanilla": ROOT / "Ablation_Models" / "Vanilla" / "CSUA_LDM_Vanilla_SpatialEvaluate.py",
}
SPATIAL_METHODS = (
    "HR",
    "LR_up",
    "Vanilla",
    "NoUnc",
    "NoDisent-DDPM",
    "NoDisent",
    "Main-DDPM",
    "Main",
)
SPECTRAL_METHODS = (
    "Vanilla",
    "NoUnc",
    "NoDisent-DDPM",
    "NoDisent",
    "Main-DDPM",
    "Main",
)
COMBINED_DATASET_ORDER = ("oli2msi", "sen2naip", "fy3fs2")


def _rgb(image: torch.Tensor, bands: list[int]) -> np.ndarray:
    array = image.detach().float().cpu()[bands].permute(1, 2, 0).numpy()
    return np.sqrt(np.clip(array, 0.0, 1.0))


def _best_roi(
    hr: torch.Tensor,
    bands: list[int],
    fraction: float = 0.63,
) -> tuple[int, int, int]:
    gray = hr[bands].mean(dim=0, keepdim=True).unsqueeze(0)
    gradient = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    gradient += F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    size = max(int(min(hr.shape[-2:]) * float(fraction)), 32)
    pooled = F.avg_pool2d(gradient, kernel_size=size, stride=1)
    flat_index = int(pooled.flatten().argmax())
    return flat_index // pooled.shape[-1], flat_index % pooled.shape[-1], size


def _resize_roi_around_center(
    roi: tuple[int, int, int],
    image_height: int,
    image_width: int,
    fraction: float,
) -> tuple[int, int, int]:
    y, x, original_size = roi
    target_size = max(int(min(image_height, image_width) * fraction), 32)
    if (original_size - target_size) % 2 != 0:
        target_size -= 1
    target_size = max(min(target_size, image_height, image_width), 1)
    offset = (original_size - target_size) // 2
    resized_y = min(max(y + offset, 0), image_height - target_size)
    resized_x = min(max(x + offset, 0), image_width - target_size)
    return resized_y, resized_x, target_size


def _resolve_spatial_roi(
    dataset_name: str,
    row: int,
    roi: tuple[int, int, int],
    image_height: int,
    image_width: int,
) -> tuple[int, int, int]:
    y, x, size = _resize_roi_around_center(
        roi,
        image_height=image_height,
        image_width=image_width,
        fraction=0.40,
    )
    shift = max(int(round(min(image_height, image_width) * 0.05)), 1)
    if dataset_name == "sen2naip":
        y = min(y + shift, image_height - size)
        if row == 1:
            x = max(x - 2 * shift, 0)
    elif dataset_name == "fy3fs2" and row == 0:
        y = max(y - 2 * shift, 0)
        x = min(x + 2 * shift, image_width - size)
    return y, x, size


def _load_module(name: str, path: Path):
    for candidate in (path.parent, path.parent.parent):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import evaluation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_models(*models) -> None:
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _denormalize_prediction(prediction: torch.Tensor, module) -> torch.Tensor:
    from CSUA_LDM_Evaluate_Common import denormalize_imgs

    return denormalize_imgs(
        prediction.detach().float().cpu(),
        module.hr_ch_mean,
        module.hr_ch_std,
    ).clamp(0.0, 1.0)


@torch.inference_mode()
def _predict_variant(
    variant: str,
    lr_up: torch.Tensor,
    hr_down_up: torch.Tensor,
    hr: torch.Tensor,
    device: torch.device,
    style_source: str = "lr_up",
) -> torch.Tensor:
    module = _load_module(
        f"ablation_qualitative_{variant.lower().replace('-', '_')}",
        VARIANT_SCRIPTS[variant],
    )
    diffusion_model, _, _ = module.build_diffusion_model(device)

    if variant in ("Main-DDPM", "NoDisent", "NoDisent-DDPM"):
        guidance_mode = "on" if variant == "NoDisent" else "off"
        module.configure_sampling_uncertainty_guidance(guidance_mode)

    if variant == "Main-DDPM":
        vae_model, _, _ = module.build_vae_model(
            Path(module.CSUA_LDM_VAE_SAVEPATH), device, load_ema=True
        )
        output = module.run_diffusion_chain(
            lr_up.to(device),
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            hr_down_up_img=hr_down_up.to(device),
            hr_img=hr.to(device),
            style_source=style_source,
            show_pbar=True,
            return_uncertainty=False,
            init_seed=int(module.RANDOM_SEED),
        )["images"]["hr"]
    elif variant in ("NoDisent", "NoDisent-DDPM"):
        vae_model, _, _ = module.build_vae_model(
            Path(module.CSUA_LDM_VAE_NODISENT_SAVEPATH), device, load_ema=True
        )
        output = module.run_diffusion_chain(
            lr_up.to(device),
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            show_pbar=True,
            return_uncertainty=False,
            init_seed=int(module.RANDOM_SEED),
        )["images"]["hr"]
    elif variant == "NoUnc":
        vae_model, _, _ = module.build_vae_model(
            Path(module.CSUA_LDM_VAE_NOUNC_SAVEPATH), device, load_ema=True
        )
        output = module.run_diffusion_chain(
            lr_up.to(device),
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            hr_down_up_img=hr_down_up.to(device),
            hr_img=hr.to(device),
            style_source=style_source,
            show_pbar=True,
            init_seed=int(module.RANDOM_SEED),
        )["images"]["hr"]
    elif variant == "Vanilla":
        vae_model, _, _ = module.build_vae_model(
            Path(module.CSUA_LDM_VAE_VANILLA_SAVEPATH), device, load_ema=True
        )
        output, _ = module.run_diffusion_chain(
            lr_up.to(device),
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            show_pbar=True,
            init_seed=int(module.RANDOM_SEED),
        )
    else:
        raise KeyError(f"Unsupported ablation variant: {variant}")

    result = _denormalize_prediction(output, module)
    del output, vae_model, diffusion_model, module
    _free_models()
    return result


def _build_test_datasets():
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
        CSUA_LDM_LR_CROP_SIZE,
        CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        CSUA_LDM_LR_GAUSSIAN_KERNEL,
        CSUA_LDM_LR_GAUSSIAN_SIGMA,
        CSUA_LDM_LR_UP_MODE,
        CSUA_LDM_SUPERRES_SCALE,
        CSUA_LDM_VIS_BAND_ORDER,
    )
    from CSUA_LDM_Evaluate_Common import build_dataset

    common_kwargs = dict(
        root_dir=CSUA_LDM_DATASET_DIRS["rootdir_list_forTest"],
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
        lr_gaussian_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
        raw_scaling_kwargs=CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )
    test_dataset = build_dataset(
        **common_kwargs,
        lr_gaussian_blur_enabled=CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
    )
    raw_lr_dataset = build_dataset(
        **common_kwargs,
        lr_gaussian_blur_enabled=False,
    )
    return test_dataset, raw_lr_dataset, list(CSUA_LDM_VIS_BAND_ORDER)


def _roi_psnr_lpips(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rgb_indices: list[int],
) -> tuple[float, float]:
    from CSUA_LDM_Quality_Metrics import compute_LPIPS, compute_PSNR

    metric_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction = prediction.float().clamp(0.0, 1.0).unsqueeze(0).to(metric_device)
    target = target.float().clamp(0.0, 1.0).unsqueeze(0).to(metric_device)
    psnr = float(compute_PSNR(prediction, target).detach().cpu().item())
    lpips_rgb, _ = compute_LPIPS(prediction, target, rgb_indices=rgb_indices[:3])
    if lpips_rgb is None:
        raise RuntimeError("LPIPS is unavailable; install or repair the `lpips` backend.")
    return psnr, float(lpips_rgb.detach().cpu().item())


def _spectral_vector(
    image: torch.Tensor,
    roi: tuple[int, int, int, int],
    channel_order: tuple[int, ...],
) -> np.ndarray:
    y, x, height, width = roi
    vector = image[list(channel_order), y:y + height, x:x + width].mean(dim=(1, 2))
    return vector.float().cpu().numpy()


def _save_spatial_figure(
    dataset_name: str,
    samples: list[dict],
    bands: list[int],
    output_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(11.8, 3.84 * len(samples)))
    grid = fig.add_gridspec(
        len(samples),
        5,
        width_ratios=[2.18, 1, 1, 1, 1],
        hspace=0.12,
        wspace=0.0,
    )
    alignment_axes: list[tuple[plt.Axes, list[plt.Axes]]] = []
    for row, sample in enumerate(samples):
        y, x, size = _resolve_spatial_roi(
            dataset_name,
            row,
            sample["roi"],
            image_height=int(sample["HR"].shape[-2]),
            image_width=int(sample["HR"].shape[-1]),
        )
        large_ax = fig.add_subplot(grid[row, 0])
        large_ax.set_anchor("N")
        large_ax.imshow(_rgb(sample["HR"], bands))
        large_ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                size,
                size,
                fill=False,
                edgecolor="#ff3b30",
                linewidth=2.2,
            )
        )
        large_ax.set_xticks([])
        large_ax.set_yticks([])
        for spine in large_ax.spines.values():
            spine.set_visible(False)
        large_ax.set_xlabel(
            f"test_img_{sample['index']} from {dataset_name.upper()}",
            fontsize=12.0,
            labelpad=5,
        )

        crop_grid = grid[row, 1:].subgridspec(
            4,
            4,
            height_ratios=[1, 0.225, 1, 0.225],
            wspace=-0.4,
            hspace=0.0,
        )
        hr_crop = sample["HR"][:, y:y + size, x:x + size]
        crop_axes: list[plt.Axes] = []
        for position, method in enumerate(SPATIAL_METHODS):
            crop = sample[method][:, y:y + size, x:x + size]
            image_row = (position // 4) * 2
            column = position % 4
            ax = fig.add_subplot(crop_grid[image_row, column])
            crop_axes.append(ax)
            ax.set_anchor("N")
            ax.imshow(_rgb(crop, bands), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if method == "HR":
                caption = "HR\nPSNR/LPIPS"
            else:
                psnr, lpips = _roi_psnr_lpips(crop, hr_crop, bands)
                caption = f"{method}\n{psnr:.2f}/{lpips:.4f}"
            caption_ax = fig.add_subplot(crop_grid[image_row + 1, column])
            caption_ax.axis("off")
            caption_ax.text(
                0.5,
                0.50,
                caption,
                fontsize=10.0,
                linespacing=1.12,
                ha="center",
                va="center",
                transform=caption_ax.transAxes,
            )
        alignment_axes.append((large_ax, crop_axes))

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
        large_ax.set_position([x_position, bottom, width, height], which="both")

    output_path = output_dir / f"{dataset_name}_Ablation_Spatial_Qualitative_Comparison.jpg"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return output_path


def _save_spectral_figure(
    dataset_name: str,
    samples: list[dict],
    bands: list[int],
    output_dir: Path,
) -> Path:
    from Main_Model.CSUA_LDM_Config import (
        CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        CSUA_LDM_SUPERRES_SCALE,
    )
    from CSUA_LDM_Evaluate_Common import degrade_hr_like_to_lr

    channel_order, band_labels = SPECTRAL_BAND_LAYOUTS[dataset_name]
    band_positions = np.arange(len(band_labels))
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(SPECTRAL_METHODS)))
    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(8.5, 3.06 * len(samples)),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.70, 1.30]},
    )
    for row, sample in enumerate(samples):
        scale = int(CSUA_LDM_SUPERRES_SCALE)
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
            band_positions,
            reference,
            marker="s",
            linewidth=2.7,
            color="black",
            label="LR reference",
        )
        for color, method in zip(colors, SPECTRAL_METHODS):
            degraded = degrade_hr_like_to_lr(
                sample[f"Spectral::{method}"].unsqueeze(0),
                superres_scale=scale,
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
                label=method,
            )
        axes[row, 1].set_xticks(band_positions, band_labels)
        axes[row, 1].set_xlabel("Spectral band", fontsize=12.0, labelpad=2)
        axes[row, 1].set_ylabel("LR-normalized reflectance", fontsize=12.0)
        axes[row, 1].tick_params(axis="both", labelsize=10.0)
        axes[row, 1].grid(alpha=0.25)
        if row == 0:
            y_limits = {
                "oli2msi": (0.6997395694255829, 1.2673643428087236),
                "sen2naip": (-0.005806757137179375, 2.89102719835937),
                "fy3fs2": (0.5135374069213867, 1.1603208184242249),
            }[dataset_name]
            axes[row, 1].set_ylim(*y_limits)
        elif dataset_name == "sen2naip" and row == 1:
            y_min, y_max = axes[row, 1].get_ylim()
            axes[row, 1].set_ylim(y_min, y_max + 0.45 * (y_max - y_min))
        legend_location = "upper right" if dataset_name == "oli2msi" else "upper left"
        axes[row, 1].legend(fontsize=10.0, ncol=2, loc=legend_location)

    output_path = output_dir / f"{dataset_name}_Ablation_Spectral_Qualitative_Comparison.jpg"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _first_sample_row(
    image_path: Path,
    trailing_white_space: int = 25,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image)
    white_fraction = (array.min(axis=2) >= 248).mean(axis=1)
    search_start = int(image.height * 0.38)
    search_end = int(image.height * 0.62)
    blank_rows = white_fraction[search_start:search_end] >= 0.985

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for offset, is_blank in enumerate(blank_rows):
        if is_blank and run_start is None:
            run_start = offset
        elif not is_blank and run_start is not None:
            runs.append((run_start, offset))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(blank_rows)))

    if runs:
        longest_start, longest_end = max(runs, key=lambda run: run[1] - run[0])
        boundary = search_start + (longest_start + longest_end) // 2
    else:
        boundary = image.height // 2
    row = image.crop((0, 0, image.width, boundary))
    row_array = np.asarray(row)
    row_white_fraction = (row_array.min(axis=2) >= 248).mean(axis=1)
    content_rows = np.flatnonzero(row_white_fraction < 0.985)
    if content_rows.size == 0:
        return row
    bottom = min(int(content_rows[-1]) + trailing_white_space, row.height)
    return row.crop((0, 0, row.width, bottom))


def _combine_first_sample_figures(kind: str) -> Path:
    if kind not in {"Spatial", "Spectral"}:
        raise ValueError(f"Unsupported qualitative figure kind: {kind}")
    rows: list[Image.Image] = []
    separator = 15 if kind == "Spatial" else 18
    trailing_white_space = 20 if kind == "Spatial" else 25
    for dataset_name in COMBINED_DATASET_ORDER:
        source = (
            RENDER_ROOT
            / kind
            / f"{dataset_name}_Ablation_{kind}_Qualitative_Comparison.jpg"
        )
        if not source.is_file():
            raise FileNotFoundError(f"Missing source figure for composition: {source}")
        rows.append(
            _first_sample_row(
                source,
                trailing_white_space=trailing_white_space,
            )
        )

    canvas_width = max(row.width for row in rows)
    canvas_height = sum(row.height for row in rows) + separator * (len(rows) - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    y = 0
    for row in rows:
        x = (canvas_width - row.width) // 2
        canvas.paste(row, (x, y))
        y += row.height + separator

    output_canvas = canvas
    if kind == "Spatial":
        canvas_array = np.asarray(canvas)
        content_columns = np.flatnonzero(
            (canvas_array.min(axis=2) < 245).any(axis=0)
        )
        if content_columns.size:
            right = min(int(content_columns[-1]) + 13, canvas.width)
            output_canvas = canvas.crop((0, 0, right, canvas.height))

    output_path = RENDER_ROOT / kind / f"Ablation_{kind}_Qualitative_Comparison.jpg"
    output_canvas.save(output_path, quality=95, subsampling=0, dpi=(300, 300))
    for row in rows:
        row.close()
    if output_canvas is not canvas:
        output_canvas.close()
    canvas.close()
    return output_path


def _combine_first_samples() -> tuple[Path, Path]:
    spatial_path = _combine_first_sample_figures("Spatial")
    spectral_path = _combine_first_sample_figures("Spectral")
    print(f"Saved: {spatial_path}")
    print(f"Saved: {spectral_path}")
    return spatial_path, spectral_path


def _run_dataset(dataset_name: str, sample_count: int) -> None:
    from Main_Model.CSUA_LDM_Config import DEVICE
    from CSUA_LDM_Utils import set_random_seed

    set_random_seed(999, deterministic=False)
    test_dataset, raw_lr_dataset, bands = _build_test_datasets()
    selected_indices = SELECTED_TEST_INDICES[dataset_name]
    sample_count = min(max(int(sample_count), 1), len(selected_indices))
    indices = selected_indices[:sample_count]

    batch = [test_dataset[index] for index in indices]
    raw_batch = [raw_lr_dataset[index] for index in indices]
    lr = torch.stack([item[0] for item in raw_batch]).float().clamp(0.0, 1.0)
    lr_up = torch.stack([item[1] for item in batch]).float().clamp(0.0, 1.0)
    hr_down_up = torch.stack([item[3] for item in batch]).float().clamp(0.0, 1.0)
    hr = torch.stack([item[4] for item in batch]).float().clamp(0.0, 1.0)

    index_token = "-".join(str(index) for index in indices)
    comparison_cache_path = (
        COMPARISON_CACHE_ROOT
        / f"{dataset_name}_indices-{index_token}_csua-uncguided.pt"
    )
    if not comparison_cache_path.is_file():
        raise FileNotFoundError(
            "The matching comparison cache is required so Main uses exactly the same "
            f"prediction as the model-comparison figure: {comparison_cache_path}"
        )
    comparison_cache = torch.load(
        comparison_cache_path, map_location="cpu", weights_only=False
    )
    outputs: dict[str, torch.Tensor] = {
        "HR": comparison_cache["HR"].float().cpu(),
        "LR": comparison_cache["LR"].float().cpu(),
        "LR_up": lr_up.float().cpu(),
        "Main": comparison_cache["CSUA_LDM (style-lr_up)"].float().cpu(),
        "Spatial::Main": comparison_cache["CSUA_LDM (style-hr)"].float().cpu(),
    }

    cache_dir = OUTPUT_ROOT / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ablation_cache_path = cache_dir / f"{dataset_name}_indices-{index_token}.pt"
    if ablation_cache_path.is_file():
        cached_variants = torch.load(
            ablation_cache_path, map_location="cpu", weights_only=False
        )
        outputs.update({name: value.float().cpu() for name, value in cached_variants.items()})

    device = torch.device(DEVICE)
    for variant in VARIANT_SCRIPTS:
        if variant in outputs:
            continue
        print(f"[{dataset_name}] Generating {variant} for test indices {indices}...")
        outputs[variant] = _predict_variant(
            variant,
            lr_up=lr_up,
            hr_down_up=hr_down_up,
            hr=hr,
            device=device,
        ).float().cpu()
        torch.save(
            {
                name: outputs[name].detach().float().cpu()
                for name in (*VARIANT_SCRIPTS, "Spatial::Main-DDPM", "Spatial::NoUnc")
                if name in outputs
            },
            ablation_cache_path,
        )

    for variant in ("Main-DDPM", "NoUnc"):
        spatial_key = f"Spatial::{variant}"
        if spatial_key in outputs:
            continue
        print(
            f"[{dataset_name}] Generating {variant} with style-hr for "
            f"test indices {indices}..."
        )
        outputs[spatial_key] = _predict_variant(
            variant,
            lr_up=lr_up,
            hr_down_up=hr_down_up,
            hr=hr,
            device=device,
            style_source="hr",
        ).float().cpu()
        torch.save(
            {
                name: outputs[name].detach().float().cpu()
                for name in (*VARIANT_SCRIPTS, "Spatial::Main-DDPM", "Spatial::NoUnc")
                if name in outputs
            },
            ablation_cache_path,
        )

    samples: list[dict] = []
    manifest_lines = [
        f"dataset={dataset_name}",
        f"test_indices={indices}",
        "Main/NoDisent sampling=uncertainty-guided",
        "Main-DDPM/NoDisent-DDPM/NoUnc/Vanilla sampling=standard DDPM",
        "Spatial Main/Main-DDPM/NoUnc style_source=hr",
        "Spectral Main/Main-DDPM/NoUnc style_source=lr_up",
        f"main_source_cache={comparison_cache_path}",
    ]
    for position, index in enumerate(indices):
        sample: dict[str, object] = {
            "index": int(index),
            "LR": outputs["LR"][position],
            "LR_up": outputs["LR_up"][position],
            "HR": outputs["HR"][position],
            "roi": _best_roi(outputs["HR"][position], bands),
        }
        for variant in ("Main", *VARIANT_SCRIPTS):
            spatial_key = f"Spatial::{variant}"
            sample[variant] = outputs.get(spatial_key, outputs[variant])[position]
            sample[f"Spectral::{variant}"] = outputs[variant][position]
        samples.append(sample)
        manifest_lines.append(
            f"index={index}\thr={Path(test_dataset.hr_img_paths[index]).name}"
            f"\tlr={Path(raw_lr_dataset.lr_img_paths[index]).name}"
            f"\tspatial_roi={_resolve_spatial_roi(dataset_name, position, sample['roi'], int(sample['HR'].shape[-2]), int(sample['HR'].shape[-1]))}"
            "\tspectral_scope=full_image"
        )

    spatial_dir = RENDER_ROOT / "Spatial"
    spectral_dir = RENDER_ROOT / "Spectral"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    spectral_dir.mkdir(parents=True, exist_ok=True)
    spatial_path = _save_spatial_figure(dataset_name, samples, bands, spatial_dir)
    spectral_path = _save_spectral_figure(dataset_name, samples, bands, spectral_dir)
    (RENDER_ROOT / f"{dataset_name}_ablation_manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {spatial_path}")
    print(f"Saved: {spectral_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_YAMLS), default=None)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    if args.combine_only:
        _combine_first_samples()
        return
    if args.dataset:
        # Bind the requested dataset before importing any config facade.
        os.environ["CSUA_LDM_DATASET_YAML_OVERRIDE"] = str(DATASET_YAMLS[args.dataset])
        _run_dataset(args.dataset, args.samples_per_dataset)
        return

    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_name, yaml_path in DATASET_YAMLS.items():
        environment = os.environ.copy()
        environment["CSUA_LDM_DATASET_YAML_OVERRIDE"] = str(yaml_path)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset",
                dataset_name,
                "--samples-per-dataset",
                str(max(int(args.samples_per_dataset), 1)),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    _combine_first_samples()


if __name__ == "__main__":
    main()
