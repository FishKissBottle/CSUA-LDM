"""Search uncertainty-guided sampling parameters on the validation split.

The search uses the same generated SR image to compute spatial and spectral
metrics, so every parameter combination requires only one diffusion pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATASET_YAMLS = {
    "oli2msi": ROOT / "configs" / "datasets" / "oli2msi.yaml",
    "sen2naip": ROOT / "configs" / "datasets" / "sen2naip.yaml",
    "fy3fs2": ROOT / "configs" / "datasets" / "fy3fs2.yaml",
}
DEFAULT_ALPHAS = "0.5,1.0,1.5,2.0"
DEFAULT_NOISE_GATE_MINS = "0.8,0.85,0.9,0.95"
SPATIAL_METRICS = ("psnr", "ssim", "gmsd_rgb", "fsim_rgb", "lpips_rgb")
SPECTRAL_METRICS = ("sam", "ergas", "scc")
HIGHER_IS_BETTER = {"psnr", "ssim", "fsim_rgb", "scc"}
CSV_FIELDS = (
    "reliability_alpha",
    "noise_gate_min",
    "noise_gate_max",
    "num_samples",
    *SPATIAL_METRICS,
    *SPECTRAL_METRICS,
    "spatial_score",
    "spectral_score",
    "balanced_score",
    "selection_score",
)


def _parse_float_list(text: str, *, name: str) -> list[float]:
    values: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value: {item}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def _parse_parameter_pairs(text: str, gate_max: float) -> list[tuple[float, float, float]]:
    pairs: list[tuple[float, float, float]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid parameter pair '{item}'. Expected reliability_alpha:noise_gate_min."
            )
        pair = (float(parts[0]), float(parts[1]), float(gate_max))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _subset_indices(length: int, max_samples: int) -> list[int]:
    if length <= 0:
        raise RuntimeError("Validation dataset is empty.")
    if max_samples <= 0 or max_samples >= length:
        return list(range(length))
    return np.unique(np.linspace(0, length - 1, max_samples).round().astype(int)).tolist()


def _build_validation_dataset(common, cfg, *, raw_lr: bool):
    return common.build_dataset(
        root_dir=cfg.CSUA_LDM_DATASET_DIRS["rootdir_list_forValid"],
        split_name=cfg.CSUA_LDM_DATASET_DIRS["split_name_forValid"],
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
        lr_gaussian_blur_enabled=False if raw_lr else cfg.CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        lr_gaussian_kernel=cfg.CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=cfg.CSUA_LDM_LR_GAUSSIAN_SIGMA,
        raw_scaling_kwargs=cfg.CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )


def _configure_guidance(sampler_mod, alpha: float, gate_min: float, gate_max: float) -> None:
    sampler_mod.CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_ENABLED = True
    sampler_mod.CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_RELIABILITY_ALPHA = float(alpha)
    sampler_mod.CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MIN = float(gate_min)
    sampler_mod.CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MAX = float(gate_max)


def _average_metrics(metric_sums: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        raise RuntimeError("No validation samples were evaluated.")
    return {name: value / float(count) for name, value in metric_sums.items()}


def _decode_with_hr_style(
    *,
    chain_out: dict,
    hr: torch.Tensor,
    vae_model,
    device: torch.device,
    content_channels: int,
    reproject_images_to_content_style_mu,
) -> torch.Tensor:
    _, _, _, hr_style, _, hr_style_logvar = reproject_images_to_content_style_mu(
        hr,
        vae_model,
        device=device,
    )
    latent_mean = chain_out["latents"]["hr"].float()
    latent_var = chain_out["latent_uncertainty"]["hr"]["var"].float()
    content = latent_mean[:, :content_channels]
    content_logvar = torch.log(torch.clamp(latent_var[:, :content_channels], min=1e-8, max=1e4))
    vae_dtype = next(vae_model.parameters()).dtype
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
        decoded = vae_model.decode(
            content.to(dtype=vae_dtype),
            hr_style.to(dtype=vae_dtype),
            content_logvar.to(dtype=vae_dtype),
            hr_style_logvar.to(dtype=vae_dtype),
        )
    decoded_mean = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
    return decoded_mean.float()


@torch.inference_mode()
def _evaluate_combination(
    *,
    primary_loader,
    reference_loader,
    vae_model,
    diffusion_model,
    device: torch.device,
    cfg,
    common,
    run_diffusion_chain,
    alpha: float,
    gate_min: float,
    gate_max: float,
    sampler_mod,
    reproject_images_to_content_style_mu,
) -> dict[str, float]:
    _configure_guidance(sampler_mod, alpha, gate_min, gate_max)
    metric_sums = {name: 0.0 for name in (*SPATIAL_METRICS, *SPECTRAL_METRICS)}
    num_samples = 0
    reference_iter = iter(reference_loader)
    desc = f"alpha={alpha:g}, gate=[{gate_min:g},{gate_max:g}]"

    for batch in tqdm(primary_loader, desc=desc, unit="batch", dynamic_ncols=True):
        try:
            reference_batch = next(reference_iter)
        except StopIteration as exc:
            raise RuntimeError("Raw-LR reference loader ended before the primary loader.") from exc

        _, lr_up, _, hr_down_up, hr, _, _ = batch
        lr_raw = reference_batch[0]
        batch_size = int(lr_up.shape[0])
        if batch_size <= 0:
            continue

        chain_out = run_diffusion_chain(
            lr_up,
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            hr_down_up_img=hr_down_up,
            hr_img=hr,
            style_source="lr_up",
            show_pbar=False,
            return_uncertainty=True,
            init_seed=int(cfg.RANDOM_SEED + num_samples),
        )
        sr_spectral_norm = chain_out["images"]["hr"].detach().float()
        sr_spatial_norm = _decode_with_hr_style(
            chain_out=chain_out,
            hr=hr,
            vae_model=vae_model,
            device=device,
            content_channels=int(cfg.CSUA_LDM_VAE_CONTENT_CHANNELS),
            reproject_images_to_content_style_mu=reproject_images_to_content_style_mu,
        )
        hr_norm = common.normalize_imgs(hr.float(), cfg.hr_ch_mean, cfg.hr_ch_std).to(
            sr_spatial_norm.device
        )

        spatial = common.compute_quality_metrics(
            sr_spatial_norm,
            hr_norm,
            cfg.hr_ch_mean,
            cfg.hr_ch_std,
            cfg.CSUA_LDM_VIS_BAND_ORDER,
        )
        spectral = common.compute_spectral_metrics(
            sr_spectral_norm,
            lr_raw.to(device=sr_spectral_norm.device, dtype=torch.float32),
            cfg.hr_ch_mean,
            cfg.hr_ch_std,
            superres_scale=cfg.CSUA_LDM_SUPERRES_SCALE,
            gaussian_kernel=cfg.CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
            gaussian_sigma=cfg.CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
            down_mode=cfg.CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        )
        for name in SPATIAL_METRICS:
            metric_sums[name] += float(spatial[name]) * batch_size
        for name in SPECTRAL_METRICS:
            metric_sums[name] += float(spectral[name]) * batch_size
        num_samples += batch_size

    try:
        next(reference_iter)
    except StopIteration:
        pass
    else:
        raise RuntimeError("Raw-LR reference loader contains more batches than the primary loader.")

    result = {
        "reliability_alpha": float(alpha),
        "noise_gate_min": float(gate_min),
        "noise_gate_max": float(gate_max),
        "num_samples": int(num_samples),
    }
    result.update(_average_metrics(metric_sums, num_samples))
    return result


def _utility_scores(rows: list[dict[str, float]], metric_names: tuple[str, ...]) -> list[float]:
    per_row: list[list[float]] = [[] for _ in rows]
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low = float(values[finite].min())
        high = float(values[finite].max())
        if high - low <= 1e-12:
            utilities = np.ones_like(values)
        elif metric_name in HIGHER_IS_BETTER:
            utilities = (values - low) / (high - low)
        else:
            utilities = (high - values) / (high - low)
        for index, utility in enumerate(utilities):
            if math.isfinite(float(utility)):
                per_row[index].append(float(utility))
    return [float(np.mean(items)) if items else float("nan") for items in per_row]


def _score_rows(rows: list[dict[str, float]], objective: str) -> None:
    spatial_scores = _utility_scores(rows, SPATIAL_METRICS)
    spectral_scores = _utility_scores(rows, SPECTRAL_METRICS)
    for row, spatial_score, spectral_score in zip(rows, spatial_scores, spectral_scores):
        balanced_score = 0.5 * spatial_score + 0.5 * spectral_score
        row["spatial_score"] = spatial_score
        row["spectral_score"] = spectral_score
        row["balanced_score"] = balanced_score
        row["selection_score"] = row[f"{objective}_score"]


def _read_existing_rows(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            converted = {key: float(value) for key, value in row.items() if value not in (None, "")}
            converted["num_samples"] = int(converted["num_samples"])
            rows.append(converted)
    return rows


def _write_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CSV_FIELDS})


def _write_summary(output_dir: Path, dataset_name: str, objective: str, rows: list[dict[str, float]]) -> None:
    ranked = sorted(rows, key=lambda row: float(row["selection_score"]), reverse=True)
    best = ranked[0]
    payload = {
        "dataset": dataset_name,
        "split": "valid",
        "style_source": "lr_up",
        "objective": objective,
        "best_parameters": {
            "reliability_alpha": best["reliability_alpha"],
            "noise_gate_min": best["noise_gate_min"],
            "noise_gate_max": best["noise_gate_max"],
        },
        "metrics": {name: best[name] for name in (*SPATIAL_METRICS, *SPECTRAL_METRICS)},
        "scores": {
            "spatial": best["spatial_score"],
            "spectral": best["spectral_score"],
            "balanced": best["balanced_score"],
            "selection": best["selection_score"],
        },
        "num_samples": best["num_samples"],
    }
    (output_dir / "best_params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {dataset_name.upper()} uncertainty-guidance search",
        "",
        f"- Split: `valid`",
        f"- Style source: `lr_up`",
        f"- Objective: `{objective}`",
        f"- Samples: `{best['num_samples']}`",
        f"- Best `reliability_alpha`: `{best['reliability_alpha']}`",
        f"- Best `noise_gate_min`: `{best['noise_gate_min']}`",
        f"- Fixed `noise_gate_max`: `{best['noise_gate_max']}`",
        "",
        "| Rank | Alpha | Gate min | Gate max | Spatial | Spectral | Balanced | Selected |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['reliability_alpha']:.6g} | {row['noise_gate_min']:.6g} | "
            f"{row['noise_gate_max']:.6g} | {row['spatial_score']:.6f} | "
            f"{row['spectral_score']:.6f} | {row['balanced_score']:.6f} | "
            f"{row['selection_score']:.6f} |"
        )
    (output_dir / "best_params.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_search(args) -> None:
    vae_code = ROOT / "Main_Model" / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_Code"
    diffusion_code = ROOT / "Main_Model" / "CSUA_LDM_Diffusion" / "CSUA_LDM_Diffusion_Code"
    for path in (ROOT, vae_code, diffusion_code):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    import Main_Model.CSUA_LDM_Config as cfg
    import CSUA_LDM_Diffusion_sampler as sampler_mod
    import CSUA_LDM_Evaluate_Common as common
    import Main_Model.CSUA_LDM_SpatialEvaluate as spatial_eval
    from CSUA_LDM_Utils import set_random_seed
    from CSUA_LDM_VAE_helpers import build_vae_model, reproject_images_to_content_style_mu

    run_diffusion_chain = spatial_eval.run_diffusion_chain
    alphas = _parse_float_list(args.alphas, name="alphas")
    gate_mins = _parse_float_list(args.noise_gate_mins, name="noise_gate_mins")
    gate_max = float(args.noise_gate_max)
    explicit_pairs = _parse_parameter_pairs(args.pairs, gate_max)
    if any(alpha <= 0.0 for alpha in alphas):
        raise ValueError("All reliability_alpha candidates must be positive.")
    if gate_max <= 0.0 or any(gate_min < 0.0 or gate_min > gate_max for gate_min in gate_mins):
        raise ValueError("Expected 0 <= noise_gate_min <= noise_gate_max.")

    set_random_seed(cfg.RANDOM_SEED, deterministic=False)
    device = torch.device(cfg.DEVICE)
    primary_dataset = _build_validation_dataset(common, cfg, raw_lr=False)
    reference_dataset = _build_validation_dataset(common, cfg, raw_lr=True)
    if len(primary_dataset) != len(reference_dataset):
        raise RuntimeError(
            f"Primary/raw validation size mismatch: {len(primary_dataset)} vs {len(reference_dataset)}."
        )
    indices = _subset_indices(len(primary_dataset), int(args.max_samples))
    primary_subset = Subset(primary_dataset, indices)
    reference_subset = Subset(reference_dataset, indices)
    batch_size = int(args.batch_size or cfg.CSUA_LDM_VALID_BATCH_SIZE)
    primary_loader = common.build_data_loader(
        primary_subset,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        persistent_workers_enabled=bool(args.num_workers > 0),
    )
    reference_loader = common.build_data_loader(
        reference_subset,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        persistent_workers_enabled=bool(args.num_workers > 0),
    )

    dataset_name = str(cfg.CSUA_LDM_DATASET_NAME).lower()
    sample_tag = "all" if len(indices) == len(primary_dataset) else str(len(indices))
    output_dir = Path(args.output_root) / dataset_name / f"samples-{sample_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "grid_results.csv"
    rows = [] if args.overwrite else _read_existing_rows(csv_path)
    completed = {
        (
            float(row["reliability_alpha"]),
            float(row["noise_gate_min"]),
            float(row["noise_gate_max"]),
        )
        for row in rows
        if int(row["num_samples"]) == len(indices)
    }

    vae_model, vae_loaded_ema, _ = build_vae_model(
        Path(cfg.CSUA_LDM_VAE_SAVEPATH),
        device,
        load_ema=True,
    )
    diffusion_model, diffusion_loaded_ema, _ = spatial_eval.build_diffusion_model(device)
    print(
        f"Dataset={dataset_name}, split=valid, samples={len(indices)}, batch_size={batch_size}, "
        f"VAE_EMA={vae_loaded_ema}, Diffusion_EMA={diffusion_loaded_ema}"
    )

    combinations = explicit_pairs or [
        (alpha, gate_min, gate_max) for alpha in alphas for gate_min in gate_mins
    ]
    for combo_index, (alpha, gate_min, current_gate_max) in enumerate(combinations, start=1):
        key = (float(alpha), float(gate_min), float(current_gate_max))
        if key in completed:
            print(f"[{combo_index}/{len(combinations)}] Skip completed: {key}")
            continue
        print(f"[{combo_index}/{len(combinations)}] Evaluate: {key}")
        row = _evaluate_combination(
            primary_loader=primary_loader,
            reference_loader=reference_loader,
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            cfg=cfg,
            common=common,
            run_diffusion_chain=run_diffusion_chain,
            alpha=alpha,
            gate_min=gate_min,
            gate_max=current_gate_max,
            sampler_mod=sampler_mod,
            reproject_images_to_content_style_mu=reproject_images_to_content_style_mu,
        )
        rows.append(row)
        _score_rows(rows, args.objective)
        _write_csv(csv_path, rows)
        _write_summary(output_dir, dataset_name, args.objective, rows)

    _score_rows(rows, args.objective)
    _write_csv(csv_path, rows)
    _write_summary(output_dir, dataset_name, args.objective, rows)
    best = max(rows, key=lambda row: float(row["selection_score"]))
    print(
        "Best parameters: "
        f"reliability_alpha={best['reliability_alpha']}, "
        f"noise_gate_min={best['noise_gate_min']}, "
        f"noise_gate_max={best['noise_gate_max']}, "
        f"{args.objective}_score={best['selection_score']:.6f}"
    )
    print(f"Results: {output_dir}")


def _build_child_command(args, dataset_name: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        dataset_name,
        "--child",
        "--alphas",
        args.alphas,
        "--noise-gate-mins",
        args.noise_gate_mins,
        "--noise-gate-max",
        str(args.noise_gate_max),
        "--pairs",
        args.pairs,
        "--objective",
        args.objective,
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--output-root",
        str(args.output_root),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search CSUA_LDM uncertainty-guided sampling parameters on validation data."
    )
    parser.add_argument("--dataset", choices=("current", "all", *DATASET_YAMLS), default="current")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--noise-gate-mins", default=DEFAULT_NOISE_GATE_MINS)
    parser.add_argument("--noise-gate-max", type=float, default=1.0)
    parser.add_argument(
        "--pairs",
        default="",
        help="Exact alpha:gate_min pairs, e.g. 0.5:0.8,1.0:0.9; overrides the grid.",
    )
    parser.add_argument("--objective", choices=("balanced", "spatial", "spectral"), default="balanced")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Validation samples to use; 0 evaluates the complete validation split.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses the configured validation batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "CSUA_LDM_Uncertainty_Guidance_Search_Results",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.child and args.dataset != "current":
        names = tuple(DATASET_YAMLS) if args.dataset == "all" else (args.dataset,)
        for dataset_name in names:
            env = os.environ.copy()
            env["CSUA_LDM_DATASET_YAML_OVERRIDE"] = str(DATASET_YAMLS[dataset_name])
            subprocess.run(_build_child_command(args, dataset_name), cwd=ROOT, env=env, check=True)
        return
    _run_search(args)


if __name__ == "__main__":
    main()
