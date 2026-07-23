"""Analyze stepwise and full-chain uncertainty against reconstruction errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "mathtext.rm": "Times New Roman",
        "axes.unicode_minus": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
VAE_CODE_ROOT = ROOT / "Main_Model" / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_Code"
DIFFUSION_CODE_ROOT = ROOT / "Main_Model" / "CSUA_LDM_Diffusion" / "CSUA_LDM_Diffusion_Code"
for search_path in (ROOT, VAE_CODE_ROOT, DIFFUSION_CODE_ROOT):
    path_text = str(search_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


DATASET_YAMLS = {
    "oli2msi": ROOT / "configs" / "datasets" / "oli2msi.yaml",
    "sen2naip": ROOT / "configs" / "datasets" / "sen2naip.yaml",
    "fy3fs2": ROOT / "configs" / "datasets" / "fy3fs2.yaml",
}
REPRESENTATIVE_TEST_INDICES = {
    "oli2msi": 62,
    "sen2naip": 50,
    "fy3fs2": 140,
}
DEFAULT_TIMESTEPS = "50,250,500,750,950"
SUMMARY_FIELDS = (
    "dataset",
    "num_samples",
    "latent_observations",
    "latent_spearman",
    "latent_spearman_ci95",
    "final_content_spearman",
    "final_content_spearman_ci95",
    "final_reverse_content_spearman",
    "final_reverse_content_spearman_ci95",
    "pixel_spearman",
    "pixel_spearman_ci95",
    "edge_spearman",
    "edge_spearman_ci95",
    "latent_top10_coverage",
    "final_content_top10_coverage",
    "final_reverse_content_top10_coverage",
    "pixel_top10_coverage",
    "edge_top10_coverage",
)


def _parse_timesteps(value: str, total_steps: int) -> list[int]:
    timesteps: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        timestep = int(item)
        if timestep < 0 or timestep >= total_steps:
            raise ValueError(f"Timestep {timestep} is outside [0, {total_steps - 1}].")
        if timestep not in timesteps:
            timesteps.append(timestep)
    if not timesteps:
        raise ValueError("At least one timestep is required.")
    return timesteps


def _subset_indices(length: int, max_samples: int) -> list[int]:
    if length <= 0:
        raise RuntimeError("Test dataset is empty.")
    if max_samples <= 0 or max_samples >= length:
        return list(range(length))
    return np.unique(np.linspace(0, length - 1, max_samples).round().astype(int)).tolist()


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten().float()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")
    x_rank = _rankdata(x[finite])
    y_rank = _rankdata(y[finite])
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = torch.sqrt(x_rank.square().sum() * y_rank.square().sum())
    if float(denom) <= 1e-12:
        return float("nan")
    return float((x_rank * y_rank).sum().div(denom).item())


def _top_fraction_coverage(x: torch.Tensor, y: torch.Tensor, fraction: float = 0.1) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(x.numel()) * float(fraction))))
    x_idx = torch.topk(x, k=k, largest=True, sorted=False).indices
    y_idx = torch.topk(y, k=k, largest=True, sorted=False).indices
    mask = torch.zeros(x.numel(), device=x.device, dtype=torch.bool)
    mask[x_idx] = True
    return float(mask[y_idx].float().mean().item())


def _decile_errors(uncertainty: torch.Tensor, error: torch.Tensor) -> list[float]:
    uncertainty = uncertainty.flatten().float()
    error = error.flatten().float()
    finite = torch.isfinite(uncertainty) & torch.isfinite(error)
    uncertainty = uncertainty[finite]
    error = error[finite]
    if uncertainty.numel() < 10:
        return [float("nan")] * 10
    error = error / torch.clamp(error.mean(), min=1e-12)
    sorted_indices = torch.argsort(uncertainty, stable=True)
    return [
        float(error[index_chunk].mean().item()) if index_chunk.numel() else float("nan")
        for index_chunk in torch.tensor_split(sorted_indices, 10)
    ]


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(array.mean())
    if array.size < 2:
        return mean, float("nan")
    ci95 = float(1.96 * array.std(ddof=1) / math.sqrt(array.size))
    return mean, ci95


def _sobel_magnitude(images: torch.Tensor) -> torch.Tensor:
    channels = int(images.shape[1])
    kernel_x = images.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(images, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(images, kernel_y, padding=1, groups=channels)
    return torch.sqrt(torch.clamp(grad_x.square() + grad_y.square(), min=1e-12))


def _build_test_dataset(common, cfg):
    return common.build_dataset(
        root_dir=cfg.CSUA_LDM_DATASET_DIRS["rootdir_list_forTest"],
        split_name=cfg.CSUA_LDM_DATASET_DIRS["split_name_forTest"],
        vis_band_order=cfg.CSUA_LDM_VIS_BAND_ORDER,
        superres_scale=cfg.CSUA_LDM_SUPERRES_SCALE,
        lr_crop_size=cfg.CSUA_LDM_LR_CROP_SIZE,
        hr_crop_size=cfg.CSUA_LDM_HR_CROP_SIZE,
        input_channels=cfg.CSUA_LDM_INPUT_CHANNELS,
        hr_degradation_gaussian_kernel=cfg.CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        hr_degradation_gaussian_sigma=cfg.CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        hr_degradation_down_mode=cfg.CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        hr_degradation_up_mode=cfg.CSUA_LDM_HR_DEGRADATION_UP_MODE,
        soft_hr_gaussian_blur_enabled=cfg.CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
        soft_hr_gaussian_kernel=cfg.CSUA_LDM_HR_GAUSSIAN_KERNEL,
        soft_hr_gaussian_sigma=cfg.CSUA_LDM_HR_GAUSSIAN_SIGMA,
        soft_hr_down_mode=cfg.CSUA_LDM_HR_DOWN_MODE,
        soft_hr_up_mode=cfg.CSUA_LDM_HR_UP_MODE,
        lr_up_mode=cfg.CSUA_LDM_LR_UP_MODE,
        lr_gaussian_blur_enabled=cfg.CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        lr_gaussian_kernel=cfg.CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=cfg.CSUA_LDM_LR_GAUSSIAN_SIGMA,
        raw_scaling_kwargs=cfg.CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )


@torch.inference_mode()
def _encode_hr(vae_model, hr: torch.Tensor, norm_img, hr_ch_mean, hr_ch_std):
    vae_dtype = next(vae_model.parameters()).dtype
    hr_norm = norm_img(hr, hr_ch_mean, hr_ch_std).to(dtype=vae_dtype)
    params = vae_model.encode(hr_norm)
    _, _, content, style, logvar_content, logvar_style = vae_model.reparameterize(params)
    return content, style, logvar_content, logvar_style


def _representative_data(
    hr: torch.Tensor,
    prediction: torch.Tensor,
    uncertainty: torch.Tensor,
    pixel_error: torch.Tensor,
    edge_error: torch.Tensor,
    vis_band_order: list[int],
    dataset_name: str,
    sample_index: int,
    style_source: str,
) -> dict[str, np.ndarray]:
    def rgb(tensor: torch.Tensor) -> np.ndarray:
        indices = list(vis_band_order[:3])
        image = torch.clamp(tensor[0, indices], 0.0, 1.0)
        return torch.sqrt(image).permute(1, 2, 0).cpu().numpy()

    return {
        "hr_rgb": rgb(hr),
        "prediction_rgb": rgb(prediction),
        "uncertainty": uncertainty[0].detach().float().cpu().numpy(),
        "pixel_error": pixel_error[0].detach().float().cpu().numpy(),
        "edge_error": edge_error[0].detach().float().cpu().numpy(),
        "dataset_name": np.asarray(dataset_name),
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "style_source": np.asarray(style_source),
    }


def _render_representative_figure(
    output_path: Path,
    data: dict[str, np.ndarray],
    heatmap_ranges: dict[str, tuple[float, float]] | None = None,
    display_transform: str = "sqrt",
) -> None:
    dataset_name = str(np.asarray(data["dataset_name"]).item())
    sample_index = int(np.asarray(data["sample_index"]).item())
    style_source = str(np.asarray(data.get("style_source", "hr")).item())
    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.15))
    axes[0].imshow(data["hr_rgb"], interpolation="nearest")
    axes[0].set_xlabel(
        f"HR\ntest_img_{sample_index} from {dataset_name.upper()}",
        fontsize=13.0,
        fontfamily="Times New Roman",
        linespacing=1.15,
        labelpad=3,
    )
    axes[1].imshow(data["prediction_rgb"], interpolation="nearest")
    axes[1].set_xlabel(
        f"SR result\n(style-{style_source})",
        fontsize=13.0,
        fontfamily="Times New Roman",
        labelpad=3,
    )

    map_panels = (
        ("uncertainty", "Pixel uncertainty"),
        ("pixel_error", "Pixel error"),
        ("edge_error", "Edge error"),
    )
    for axis, (key, label) in zip(axes[2:], map_panels):
        value = np.asarray(data[key], dtype=np.float32)
        if heatmap_ranges is not None:
            vmin, vmax = heatmap_ranges[key]
        else:
            vmin = float(np.nanmin(value))
            vmax = float(np.nanmax(value))
        scale = max(float(vmax) - float(vmin), 1e-8)
        normalized = np.clip((value - float(vmin)) / scale, 0.0, 1.0)
        if display_transform == "sqrt":
            display_value = np.sqrt(normalized)
        elif display_transform == "linear":
            display_value = normalized
        else:
            raise ValueError(f"Unsupported heatmap display transform: {display_transform}")
        axis.imshow(display_value, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_xlabel(label, fontsize=13.0, fontfamily="Times New Roman", labelpad=3)

    for axis in axes:
        axis.set_anchor("N")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.008, right=0.992, top=0.985, bottom=0.13, wspace=0.035)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def _save_representative_figure(
    output_path: Path,
    hr: torch.Tensor,
    prediction: torch.Tensor,
    uncertainty: torch.Tensor,
    pixel_error: torch.Tensor,
    edge_error: torch.Tensor,
    vis_band_order: list[int],
    dataset_name: str,
    sample_index: int,
    style_source: str,
    diagnostic_maps: dict[str, torch.Tensor] | None = None,
) -> None:
    data = _representative_data(
        hr=hr,
        prediction=prediction,
        uncertainty=uncertainty,
        pixel_error=pixel_error,
        edge_error=edge_error,
        vis_band_order=vis_band_order,
        dataset_name=dataset_name,
        sample_index=sample_index,
        style_source=style_source,
    )
    if diagnostic_maps is not None:
        for key, value in diagnostic_maps.items():
            data[key] = value[0].detach().float().cpu().numpy()
    np.savez_compressed(output_path.with_name("uncertainty_error_example_data.npz"), **data)
    _render_representative_figure(output_path, data)


def _save_summary_plot(output_path: Path, summary: dict, deciles: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = (
        "Stepwise latent v",
        "Accumulated content",
        "Final-step content",
        "Final pixel",
        "Final edge",
    )
    values = (
        summary["latent_spearman"],
        summary["final_content_spearman"],
        summary["final_reverse_content_spearman"],
        summary["pixel_spearman"],
        summary["edge_spearman"],
    )
    errors = (
        summary["latent_spearman_ci95"],
        summary["final_content_spearman_ci95"],
        summary["final_reverse_content_spearman_ci95"],
        summary["pixel_spearman_ci95"],
        summary["edge_spearman_ci95"],
    )
    colors = ("#2f6573", "#5f8f62", "#7a9f89", "#d58b47", "#9c4f52")
    axes[0].bar(labels, values, yerr=errors, color=colors, capsize=4)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean Spearman correlation")
    axes[0].set_title("Uncertainty-error correlation")

    x = np.arange(1, 11)
    for key, label, color in zip(
        ("latent", "final_content", "final_reverse_content", "pixel", "edge"),
        labels,
        colors,
    ):
        axes[1].plot(x, deciles[key], marker="o", linewidth=1.7, label=label, color=color)
    axes[1].set_xlabel("Uncertainty decile (low to high)")
    axes[1].set_ylabel("Relative mean error")
    axes[1].set_title("Error across uncertainty deciles")
    axes[1].set_xticks(x)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _aggregate_results(output_root: Path, display_transform: str = "sqrt") -> None:
    output_root = Path(output_root)
    summaries: list[dict] = []
    for dataset_name in DATASET_YAMLS:
        result_dir = output_root / dataset_name / "samples-all"
        summary_path = result_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing complete correlation results for {dataset_name}: {result_dir}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)

    aggregate_csv = output_root / "summary_all_datasets.csv"
    with aggregate_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in SUMMARY_FIELDS})

    output_suffix = "" if display_transform == "sqrt" else f"_{display_transform}"
    example_paths = []
    example_data: list[dict[str, np.ndarray]] = []
    for dataset_name in DATASET_YAMLS:
        result_dir = output_root / dataset_name / "samples-all"
        example_path = result_dir / f"uncertainty_error_example{output_suffix}.jpg"
        data_path = result_dir / "uncertainty_error_example_data.npz"
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing representative heatmap data: {data_path}")
        with np.load(data_path, allow_pickle=False) as loaded:
            example_data.append({key: loaded[key] for key in loaded.files})
        example_paths.append(example_path)

    heatmap_keys = ("uncertainty", "pixel_error", "edge_error")
    per_heatmap_ranges = {}
    for dataset_name, data in zip(DATASET_YAMLS, example_data):
        per_heatmap_ranges[dataset_name] = {}
        for key in heatmap_keys:
            values = np.asarray(data[key]).ravel()
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                raise RuntimeError(
                    f"Representative heatmap data for {dataset_name}/{key} contains no finite values."
                )
            per_heatmap_ranges[dataset_name][key] = (
                float(np.min(finite_values)),
                float(np.max(finite_values)),
            )
    scale_metadata = {
        "normalization": f"per_heatmap_minmax_{display_transform}_display",
        "display_transform": display_transform,
        "datasets": list(DATASET_YAMLS),
        "ranges": {
            dataset_name: {
                key: {
                    "min": value_range[0],
                    "max": value_range[1],
                }
                for key, value_range in dataset_ranges.items()
            }
            for dataset_name, dataset_ranges in per_heatmap_ranges.items()
        },
    }
    (output_root / f"representative_heatmap_scales{output_suffix}.json").write_text(
        json.dumps(scale_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for dataset_name, output_path, data in zip(DATASET_YAMLS, example_paths, example_data):
        _render_representative_figure(
            output_path,
            data,
            heatmap_ranges=per_heatmap_ranges[dataset_name],
            display_transform=display_transform,
        )

    example_images = [plt.imread(path) for path in example_paths]
    example_fig, example_axes = plt.subplots(3, 1, figsize=(18.6, 10.8))
    for axis, image in zip(example_axes, example_images):
        axis.imshow(image)
        axis.set_anchor("E")
        axis.axis("off")
    example_fig.subplots_adjust(left=0.005, right=0.945, top=0.995, bottom=0.005, hspace=0.0)
    colorbar_axis = example_fig.add_axes([0.968, 0.18, 0.012, 0.64])
    colorbar = example_fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
            cmap="magma",
        ),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_ticks([0.0, 1.0], labels=["Low", "High"])
    colorbar_label = "Sqrt-normalized value" if display_transform == "sqrt" else "Normalized value"
    colorbar.set_label(
        colorbar_label,
        fontsize=13.0,
        fontfamily="Times New Roman",
        labelpad=-8,
    )
    colorbar.ax.tick_params(labelsize=12.0, length=2, pad=2)
    aggregate_examples = output_root / f"uncertainty_error_examples_all_datasets{output_suffix}.jpg"
    example_fig.savefig(aggregate_examples, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(example_fig)
    print(f"Saved: {aggregate_csv}")
    print(f"Saved: {aggregate_examples}")


def _run_analysis(args) -> None:
    import Main_Model.CSUA_LDM_Config as cfg
    import CSUA_LDM_Evaluate_Common as common
    import Main_Model.CSUA_LDM_SpatialEvaluate as spatial_eval
    from CSUA_LDM_Diffusion_helpers import cosine_beta_scheduler
    from CSUA_LDM_Utils import (
        denorm_mean,
        denorm_std,
        norm_img,
        set_random_seed,
        sigma_from_logvar,
    )
    from CSUA_LDM_VAE_helpers import build_vae_model, reproject_images_to_content_style_mu

    run_diffusion_chain = spatial_eval.run_diffusion_chain
    set_random_seed(cfg.RANDOM_SEED, deterministic=False)
    device = torch.device(cfg.DEVICE)
    timesteps = _parse_timesteps(args.timesteps, cfg.CSUA_LDM_DIFFUSION_STEPS)
    examples_timestep = int(args.examples_timestep)

    dataset = _build_test_dataset(common, cfg)
    dataset_name = str(cfg.CSUA_LDM_DATASET_NAME).lower()
    representative_index = REPRESENTATIVE_TEST_INDICES[dataset_name]
    if representative_index >= len(dataset):
        raise IndexError(
            f"Representative test index {representative_index} is outside {dataset_name} "
            f"test dataset length {len(dataset)}."
        )
    if args.examples_only:
        indices = [representative_index]
        timesteps = [examples_timestep]
    else:
        indices = _subset_indices(len(dataset), int(args.max_samples))
        if representative_index not in indices:
            indices[-1] = representative_index
            indices.sort()
    subset = Subset(dataset, indices)
    loader = common.build_data_loader(
        subset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        persistent_workers_enabled=bool(args.num_workers > 0),
    )

    vae_model, vae_loaded_ema, _ = build_vae_model(
        Path(cfg.CSUA_LDM_VAE_SAVEPATH), device, load_ema=True
    )
    diffusion_model, diffusion_loaded_ema, _ = spatial_eval.build_diffusion_model(device)
    guidance_enabled = spatial_eval.configure_sampling_uncertainty_guidance("on")
    betas = cosine_beta_scheduler(
        timesteps=cfg.CSUA_LDM_DIFFUSION_STEPS,
        device=device,
        dtype=torch.float32,
    )
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    scale = torch.tensor(float(cfg.LATENT_SCALING_FACTOR), device=device, dtype=torch.float32)

    sample_tag = "all" if args.examples_only or len(indices) == len(dataset) else str(len(indices))
    output_dir = Path(args.output_root) / dataset_name / f"samples-{sample_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_observation: list[dict[str, float | int | str]] = []
    latent_deciles: list[list[float]] = []
    final_content_deciles: list[list[float]] = []
    final_reverse_content_deciles: list[list[float]] = []
    pixel_deciles: list[list[float]] = []
    edge_deciles: list[list[float]] = []
    latent_rhos: list[float] = []
    latent_coverages: list[float] = []
    final_content_rhos: list[float] = []
    final_content_coverages: list[float] = []
    final_reverse_content_rhos: list[float] = []
    final_reverse_content_coverages: list[float] = []
    pixel_rhos: list[float] = []
    pixel_coverages: list[float] = []
    edge_rhos: list[float] = []
    edge_coverages: list[float] = []
    final_per_sample: list[dict[str, float | int | str]] = []
    representative_saved = False
    sample_offset = 0

    print(
        f"Dataset={dataset_name}, split=test, samples={len(indices)}, timesteps={timesteps}, "
        f"VAE_EMA={vae_loaded_ema}, Diffusion_EMA={diffusion_loaded_ema}, "
        f"Guidance={guidance_enabled}, final_style={args.style_source}"
    )
    for batch in tqdm(loader, desc=f"Correlation {dataset_name}", unit="batch", dynamic_ncols=True):
        _, lr_up, _, _, hr, _, _ = batch
        lr_up = lr_up.to(device=device, dtype=torch.float32)
        hr = hr.to(device=device, dtype=torch.float32)
        batch_size = int(hr.shape[0])

        (
            _,
            _,
            cond_content,
            _,
            cond_logvar_content,
            _,
        ) = reproject_images_to_content_style_mu(lr_up, vae_model, device=device)
        target_content, _, _, _ = _encode_hr(
            vae_model,
            hr,
            norm_img,
            cfg.hr_ch_mean,
            cfg.hr_ch_std,
        )

        y0 = cond_content.float() * scale
        x0 = target_content.float() * scale
        y0_sigma = sigma_from_logvar(
            cond_logvar_content.float(),
            soften_lambda=cfg.CSUA_LDM_DIFFUSION_UNC_COND_LOGVAR_SOFTEN_LAMBDA,
            scale=scale,
        )

        for timestep in timesteps:
            t = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
            alpha_bar = alphas_cumprod[t].view(batch_size, 1, 1, 1)
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-20))
            generator = torch.Generator(device=device)
            generator.manual_seed(int(cfg.RANDOM_SEED + sample_offset * 1009 + timestep))
            noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
            v_target = sqrt_alpha_bar * noise - sqrt_one_minus * x0

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                model_out = diffusion_model(x_t, y0, y0_sigma, t, return_dict=True)
            pred_v = model_out["pred_v_mean"].float()
            pred_v_var = torch.clamp(model_out["pred_v_var"].float(), min=1e-8, max=1e4)
            v_error = (pred_v - v_target).square()
            uncertainty_map = pred_v_var.mean(dim=1)
            latent_error_map = v_error.mean(dim=1)

            for local_index in range(batch_size):
                latent_rho = _spearman(uncertainty_map[local_index], latent_error_map[local_index])
                latent_coverage = _top_fraction_coverage(
                    uncertainty_map[local_index], latent_error_map[local_index]
                )
                latent_rhos.append(latent_rho)
                latent_coverages.append(latent_coverage)
                latent_deciles.append(
                    _decile_errors(uncertainty_map[local_index], latent_error_map[local_index])
                )
                per_observation.append(
                    {
                        "dataset": dataset_name,
                        "sample_index": int(indices[sample_offset + local_index]),
                        "timestep": int(timestep),
                        "latent_spearman": latent_rho,
                        "latent_top10_coverage": latent_coverage,
                    }
                )


        inference_seed = (
            int(args.inference_seed)
            if args.inference_seed is not None
            else int(cfg.RANDOM_SEED + sample_offset)
        )
        chain_out = run_diffusion_chain(
            lr_up_img=lr_up,
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            hr_img=hr,
            style_source=args.style_source,
            show_pbar=False,
            return_uncertainty=True,
            init_seed=inference_seed,
        )
        prediction = torch.clamp(
            denorm_mean(chain_out["images"]["hr"].float(), cfg.hr_ch_mean, cfg.hr_ch_std),
            0.0,
            1.0,
        )
        pixel_uncertainty_std = denorm_std(
            chain_out["pixel_uncertainty"]["hr"]["std"].float(),
            cfg.hr_ch_std,
        )
        uncertainty_image = pixel_uncertainty_std.square().mean(dim=1, keepdim=True)
        content_ch = int(cfg.CSUA_LDM_VAE_CONTENT_CHANNELS)
        final_content = chain_out["latents"]["hr"][:, :content_ch].float()
        final_content_var = chain_out["latent_uncertainty"]["hr"]["var"][
            :, :content_ch
        ].float()
        final_reverse_content_var = chain_out["reverse_uncertainty"]["hr"]["var"][
            :, :content_ch
        ].float()
        final_content_uncertainty = final_content_var.mean(dim=1, keepdim=True)
        final_reverse_content_uncertainty = final_reverse_content_var.mean(
            dim=1, keepdim=True
        )
        final_content_error = (final_content - target_content.float()).square().mean(
            dim=1, keepdim=True
        )
        pixel_error = (prediction - hr).abs().mean(dim=1, keepdim=True)
        edge_error = (_sobel_magnitude(prediction) - _sobel_magnitude(hr)).abs().mean(
            dim=1, keepdim=True
        )

        for local_index in range(batch_size):
            current_index = int(indices[sample_offset + local_index])
            final_content_rho = _spearman(
                final_content_uncertainty[local_index], final_content_error[local_index]
            )
            final_content_coverage = _top_fraction_coverage(
                final_content_uncertainty[local_index], final_content_error[local_index]
            )
            final_reverse_content_rho = _spearman(
                final_reverse_content_uncertainty[local_index],
                final_content_error[local_index],
            )
            final_reverse_content_coverage = _top_fraction_coverage(
                final_reverse_content_uncertainty[local_index],
                final_content_error[local_index],
            )
            pixel_rho = _spearman(
                uncertainty_image[local_index], pixel_error[local_index]
            )
            edge_rho = _spearman(
                uncertainty_image[local_index], edge_error[local_index]
            )
            pixel_coverage = _top_fraction_coverage(
                uncertainty_image[local_index], pixel_error[local_index]
            )
            edge_coverage = _top_fraction_coverage(
                uncertainty_image[local_index], edge_error[local_index]
            )
            final_content_rhos.append(final_content_rho)
            final_content_coverages.append(final_content_coverage)
            final_reverse_content_rhos.append(final_reverse_content_rho)
            final_reverse_content_coverages.append(final_reverse_content_coverage)
            pixel_rhos.append(pixel_rho)
            edge_rhos.append(edge_rho)
            pixel_coverages.append(pixel_coverage)
            edge_coverages.append(edge_coverage)
            final_content_deciles.append(
                _decile_errors(
                    final_content_uncertainty[local_index], final_content_error[local_index]
                )
            )
            final_reverse_content_deciles.append(
                _decile_errors(
                    final_reverse_content_uncertainty[local_index],
                    final_content_error[local_index],
                )
            )
            pixel_deciles.append(
                _decile_errors(uncertainty_image[local_index], pixel_error[local_index])
            )
            edge_deciles.append(
                _decile_errors(uncertainty_image[local_index], edge_error[local_index])
            )
            final_per_sample.append(
                {
                    "dataset": dataset_name,
                    "sample_index": current_index,
                    "final_content_spearman": final_content_rho,
                    "final_content_top10_coverage": final_content_coverage,
                    "final_reverse_content_spearman": final_reverse_content_rho,
                    "final_reverse_content_top10_coverage": final_reverse_content_coverage,
                    "pixel_spearman": pixel_rho,
                    "pixel_top10_coverage": pixel_coverage,
                    "edge_spearman": edge_rho,
                    "edge_top10_coverage": edge_coverage,
                }
            )

            if not representative_saved and current_index == representative_index:
                sample_slice = slice(local_index, local_index + 1)
                _save_representative_figure(
                    output_dir / "uncertainty_error_example.jpg",
                    hr[sample_slice],
                    prediction[sample_slice],
                    uncertainty_image[sample_slice, 0],
                    pixel_error[sample_slice, 0],
                    edge_error[sample_slice, 0],
                    cfg.CSUA_LDM_VIS_BAND_ORDER,
                    dataset_name,
                    current_index,
                    args.style_source,
                )
                representative_saved = True

        sample_offset += batch_size

    if not representative_saved:
        raise RuntimeError(
            f"Representative test index {representative_index} was not included in the evaluated subset."
        )
    if args.examples_only:
        print(
            f"Saved representative example: {output_dir / 'uncertainty_error_example.jpg'} "
            f"(inference_seed={inference_seed})"
        )
        return

    latent_mean, latent_ci = _mean_ci95(latent_rhos)
    final_content_mean, final_content_ci = _mean_ci95(final_content_rhos)
    final_reverse_content_mean, final_reverse_content_ci = _mean_ci95(
        final_reverse_content_rhos
    )
    pixel_mean, pixel_ci = _mean_ci95(pixel_rhos)
    edge_mean, edge_ci = _mean_ci95(edge_rhos)
    summary = {
        "dataset": dataset_name,
        "num_samples": len(indices),
        "latent_observations": len(latent_rhos),
        "latent_spearman": latent_mean,
        "latent_spearman_ci95": latent_ci,
        "final_content_spearman": final_content_mean,
        "final_content_spearman_ci95": final_content_ci,
        "final_reverse_content_spearman": final_reverse_content_mean,
        "final_reverse_content_spearman_ci95": final_reverse_content_ci,
        "pixel_spearman": pixel_mean,
        "pixel_spearman_ci95": pixel_ci,
        "edge_spearman": edge_mean,
        "edge_spearman_ci95": edge_ci,
        "latent_top10_coverage": _mean_ci95(latent_coverages)[0],
        "final_content_top10_coverage": _mean_ci95(final_content_coverages)[0],
        "final_reverse_content_top10_coverage": _mean_ci95(
            final_reverse_content_coverages
        )[0],
        "pixel_top10_coverage": _mean_ci95(pixel_coverages)[0],
        "edge_top10_coverage": _mean_ci95(edge_coverages)[0],
        "timesteps": timesteps,
        "image_error_source": "complete_reverse_sampling",
        "reverse_sampling_steps": int(cfg.CSUA_LDM_DIFFUSION_STEPS),
        "sampling_uncertainty_guidance": bool(guidance_enabled),
        "style_source": args.style_source,
        "split": "test",
        "vae_ema": bool(vae_loaded_ema),
        "diffusion_ema": bool(diffusion_loaded_ema),
    }
    deciles = {
        "latent": np.nanmean(np.asarray(latent_deciles, dtype=np.float64), axis=0).tolist(),
        "final_content": np.nanmean(
            np.asarray(final_content_deciles, dtype=np.float64), axis=0
        ).tolist(),
        "final_reverse_content": np.nanmean(
            np.asarray(final_reverse_content_deciles, dtype=np.float64), axis=0
        ).tolist(),
        "pixel": np.nanmean(np.asarray(pixel_deciles, dtype=np.float64), axis=0).tolist(),
        "edge": np.nanmean(np.asarray(edge_deciles, dtype=np.float64), axis=0).tolist(),
    }

    per_fields = sorted({key for row in per_observation for key in row})
    with (output_dir / "per_observation.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=per_fields)
        writer.writeheader()
        writer.writerows(per_observation)
    with (output_dir / "final_per_sample.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=tuple(final_per_sample[0]))
        writer.writeheader()
        writer.writerows(final_per_sample)
    with (output_dir / "decile_errors.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "decile",
                "latent_relative_error",
                "final_content_relative_error",
                "final_reverse_content_relative_error",
                "pixel_relative_error",
                "edge_relative_error",
            )
        )
        for index in range(10):
            writer.writerow(
                (
                    index + 1,
                    deciles["latent"][index],
                    deciles["final_content"][index],
                    deciles["final_reverse_content"][index],
                    deciles["pixel"][index],
                    deciles["edge"][index],
                )
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({field: summary[field] for field in SUMMARY_FIELDS})
    _save_summary_plot(output_dir / "uncertainty_error_statistics.jpg", summary, deciles)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_dir}")


def _child_command(args, dataset_name: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        dataset_name,
        "--child",
        "--timesteps",
        args.timesteps,
        "--examples-timestep",
        str(args.examples_timestep),
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--output-root",
        str(args.output_root),
        "--style-source",
        args.style_source,
    ]
    if args.examples_only:
        command.append("--examples-only")
    if args.inference_seed is not None:
        command.extend(("--inference-seed", str(args.inference_seed)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CSUA_LDM v-uncertainty correlation with actual prediction errors."
    )
    parser.add_argument("--dataset", choices=("current", "all", *DATASET_YAMLS), default="current")
    parser.add_argument("--timesteps", default=DEFAULT_TIMESTEPS)
    parser.add_argument(
        "--examples-timestep",
        "--image-timestep",
        dest="examples_timestep",
        type=int,
        default=500,
        help="Single latent timestep used when --examples-only is selected.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 uses the complete test split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Override the reverse-sampling seed, primarily for reproducibility diagnostics.",
    )
    parser.add_argument(
        "--style-source",
        choices=("lr_up", "hr"),
        default="hr",
        help="Choose the style latent used for final SR decoding and pixel-uncertainty propagation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "CSUA_LDM_Uncertainty_Error_Correlation_Results",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--heatmap-display",
        choices=("sqrt", "linear"),
        default="sqrt",
        help="Display transform used when rendering aggregate representative heatmaps.",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Redraw representative examples without overwriting complete-test statistics.",
    )
    args = parser.parse_args()

    if args.aggregate_only:
        _aggregate_results(args.output_root, display_transform=args.heatmap_display)
        return
    if not args.child and args.dataset != "current":
        dataset_names = tuple(DATASET_YAMLS) if args.dataset == "all" else (args.dataset,)
        for dataset_name in dataset_names:
            environment = os.environ.copy()
            environment["CSUA_LDM_DATASET_YAML_OVERRIDE"] = str(DATASET_YAMLS[dataset_name])
            subprocess.run(
                _child_command(args, dataset_name),
                cwd=ROOT,
                env=environment,
                check=True,
            )
        if args.dataset == "all":
            _aggregate_results(args.output_root, display_transform=args.heatmap_display)
        return
    _run_analysis(args)


if __name__ == "__main__":
    main()
