import argparse
import math
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


VARIANT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = VARIANT_ROOT.parents[1]
MODULE_DIR = Path(__file__).resolve().parent
for search_path in (PROJECT_ROOT, VARIANT_ROOT, MODULE_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import CSUA_LDM_NoDisent_Config as CSUA_LDM_NoDisent_Config_module
from CSUA_LDM_NoDisent_Config import (
    DEVICE as DEFAULT_DEVICE,
    NUM_WORKERS,
    PERSISTENT_WORKERS_ENABLED,
    CSUA_LDM_DATASET_DIRS,
    CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    CSUA_LDM_VAE_NODISENT_SAVEPATH,
    CSUA_LDM_VAE_NODISENT_TAG,
    RANDOM_SEED,
    CSUA_LDM_SUPERRES_SCALE,
    CSUA_LDM_LR_CROP_SIZE,
    CSUA_LDM_HR_CROP_SIZE,
    CSUA_LDM_INPUT_CHANNELS,
    CSUA_LDM_VIS_BAND_ORDER,
    CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
    CSUA_LDM_HR_GAUSSIAN_KERNEL,
    CSUA_LDM_HR_GAUSSIAN_SIGMA,
    CSUA_LDM_HR_DOWN_MODE,
    CSUA_LDM_HR_UP_MODE,
    hr_ch_mean,
    hr_ch_std,
)
from CSUA_LDM_Utils import ensure_batched_4d, resolve_device, seed_dataloader_worker, set_random_seed
from CSUA_LDM_Dataset import SuperResolutionDataset
from CSUA_LDM_VAE_NoDisent_helpers import (
    build_vae_model,
)


DEFAULT_VAE_CKPT = Path(CSUA_LDM_VAE_NODISENT_SAVEPATH)
DEFAULT_ROOT_DIR = Path(CSUA_LDM_DATASET_DIRS["rootdir_list_forTrain"])
VARIANT_YAML_PATH = PROJECT_ROOT / "configs" / "variants" / "nodisent.yaml"
HR_STAGE_NAME = "hr_content"


def normalize_latent_scaling_factor(latent_scaling_factor):
    if latent_scaling_factor is None:
        return None

    if isinstance(latent_scaling_factor, torch.Tensor):
        if latent_scaling_factor.numel() != 1:
            raise ValueError(
                f"`LATENT_SCALING_FACTOR` must be a scalar, got tensor with shape {tuple(latent_scaling_factor.shape)}."
            )
        value = float(latent_scaling_factor.detach().view(()).item())
    else:
        value = float(latent_scaling_factor)

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Invalid `LATENT_SCALING_FACTOR`: {value}. It must be finite and > 0."
        )
    return value


def write_latent_scaling_factor_to_config(latent_scaling_factor, yaml_path: Path | str = VARIANT_YAML_PATH):
    latent_scaling_factor = normalize_latent_scaling_factor(latent_scaling_factor)
    rounded_value = round(float(latent_scaling_factor), 6)
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Read dataset name from config
    dataset_name = str(getattr(CSUA_LDM_NoDisent_Config_module._cfg.dataset, "name", "unknown")).lower()

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "latent" not in data:
        data["latent"] = {}
    if "scaling_factor_by_dataset" not in data["latent"]:
        data["latent"]["scaling_factor_by_dataset"] = {}

    data["latent"]["scaling_factor_by_dataset"][dataset_name] = rounded_value

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Verify write
    with open(yaml_path, "r", encoding="utf-8") as f:
        verify = yaml.safe_load(f)
    written = verify.get("latent", {}).get("scaling_factor_by_dataset", {}).get(dataset_name)
    if written != rounded_value:
        raise RuntimeError(f"Verification failed: expected {rounded_value}, got {written}")

    print(f"[ScalingFactor] raw={latent_scaling_factor}")
    print(f"[ScalingFactor] rounded={rounded_value:.6f}")
    print(f"[ScalingFactor] dataset={dataset_name}")
    print(f"[ScalingFactor] wrote to {yaml_path}")
    return rounded_value


class RunningScalarStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.skipped_nonfinite = 0

    def update(self, x: torch.Tensor):
        x64 = x.detach().to(dtype=torch.float64)
        finite_mask = torch.isfinite(x64)
        finite_count = int(finite_mask.sum().item())
        skipped_count = int(x64.numel()) - finite_count
        self.skipped_nonfinite += skipped_count
        if finite_count <= 0:
            return

        x64 = x64[finite_mask]
        self.count += finite_count
        self.sum += float(x64.sum().item())
        self.sum_sq += float(x64.square().sum().item())

    def finalize(self, eps: float = 1e-12) -> dict[str, float | int]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize stats with zero elements.")

        mean = self.sum / float(self.count)
        var = self.sum_sq / float(self.count) - mean * mean
        var = max(var, float(eps))
        std = var ** 0.5
        scaling_factor = 1.0 / std

        finite_values = torch.tensor([mean, std, scaling_factor], dtype=torch.float64)
        if not all(torch.isfinite(finite_values).tolist()):
            raise RuntimeError(
                f"Non-finite stats detected: mean={mean}, std={std}, scaling_factor={scaling_factor}."
            )

        return {
            "numel": int(self.count),
            "skipped_nonfinite": int(self.skipped_nonfinite),
            "mean": float(mean),
            "std": float(std),
            "scaling_factor": float(scaling_factor),
        }


def finite_sample_mask(x: torch.Tensor) -> torch.Tensor:
    """Return per-sample finite mask for a tensor with batch dimension first."""
    if x.dim() < 2:
        return torch.isfinite(x).view(-1)
    return torch.isfinite(x).flatten(start_dim=1).all(dim=1)


def standardize_hr_imgs(hr_imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    hr_imgs = ensure_batched_4d(hr_imgs).detach().to(device=device, dtype=torch.float32)
    mean_t = hr_ch_mean.to(device=device, dtype=hr_imgs.dtype)
    std_t = hr_ch_std.to(device=device, dtype=hr_imgs.dtype)
    return (hr_imgs - mean_t) / std_t.clamp_min(1e-12)


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Compute HR content-only latent scaling factor for {CSUA_LDM_VAE_NODISENT_TAG}."
    )
    parser.add_argument("--root-dir", type=str, default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--vae-ckpt", type=str, default=str(DEFAULT_VAE_CKPT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")
    if args.num_workers < 0:
        raise ValueError("`num_workers` must be >= 0.")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("`max_batches` must be > 0 when provided.")


def format_hr_stats(row: dict[str, float | int | str]):
    header = (
        f"{'stage':<8}  {'numel':>12}  {'skip_in':>8}  "
        f"{'skip_lat':>8}  {'skip_val':>10}  {'mean':>14}  {'std':>14}  "
        f"{'scaling_factor':>18}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{str(row['stage']):<8}  {int(row['numel']):>12}  "
        f"{int(row['skipped_input_samples']):>8}  "
        f"{int(row['skipped_latent_samples']):>8}  "
        f"{int(row['skipped_nonfinite']):>10}  "
        f"{float(row['mean']):>14.8f}  {float(row['std']):>14.8f}  "
        f"{float(row['scaling_factor']):>18.8f}"
    )


def compute_latent_scaling_factor(
    root_dir: Path,
    vae_ckpt: Path,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    device: torch.device,
    seed: int,
):
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    set_random_seed(seed)
    data_generator = torch.Generator()
    data_generator.manual_seed(int(seed))

    dataset = SuperResolutionDataset(
        root_dir=str(root_dir),
        is_train=True,
        split_name=CSUA_LDM_DATASET_DIRS["split_name_forTrain"],
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
        superres_scale=CSUA_LDM_SUPERRES_SCALE,
        lr_crop_size=CSUA_LDM_LR_CROP_SIZE,
        hr_crop_size=CSUA_LDM_HR_CROP_SIZE,
        input_channels=CSUA_LDM_INPUT_CHANNELS,
        **CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
        soft_hr_gaussian_blur_enabled=CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
        soft_hr_gaussian_kernel=CSUA_LDM_HR_GAUSSIAN_KERNEL,
        soft_hr_gaussian_sigma=CSUA_LDM_HR_GAUSSIAN_SIGMA,
        soft_hr_down_mode=CSUA_LDM_HR_DOWN_MODE,
        soft_hr_up_mode=CSUA_LDM_HR_UP_MODE,
    )
    persistent_workers_flag = bool(num_workers > 0 and PERSISTENT_WORKERS_ENABLED)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_dataloader_worker,
        generator=data_generator,
        persistent_workers=persistent_workers_flag,
    )

    model, effective_load_ema, _ = build_vae_model(vae_ckpt=vae_ckpt, device=device, load_ema=True)

    hr_stats = RunningScalarStats()
    skipped_input_samples = 0
    skipped_latent_samples = 0
    verified_hr_eval_mu = False
    batches_processed = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    with torch.no_grad():
        with tqdm(loader, total=total_batches, ncols=120, desc=f"Compute {CSUA_LDM_VAE_NODISENT_TAG} scaling factor") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                hr_imgs = batch[4]
                hr_tensor = standardize_hr_imgs(hr_imgs, device=device)
                input_mask = finite_sample_mask(hr_tensor)
                input_skipped = int((~input_mask).sum().item())
                if input_skipped > 0:
                    skipped_input_samples += input_skipped
                    if int(input_mask.sum().item()) <= 0:
                        continue
                    hr_tensor = hr_tensor[input_mask]

                z_params = model.encode(hr_tensor)
                rep_out = model.reparameterize(z_params)

                if not (isinstance(rep_out, (list, tuple)) and len(rep_out) >= 3):
                    raise TypeError(
                        "NoDisent VAE `reparameterize()` is expected to return (z, mu, logvar)."
                    )
                z_content = rep_out[0]
                mu_content = rep_out[1]

                latent_mask = finite_sample_mask(z_content) & finite_sample_mask(mu_content)
                latent_skipped = int((~latent_mask).sum().item())
                if latent_skipped > 0:
                    skipped_latent_samples += latent_skipped
                    if int(latent_mask.sum().item()) <= 0:
                        continue
                    z_content = z_content[latent_mask]
                    mu_content = mu_content[latent_mask]

                if not torch.equal(z_content, mu_content):
                    diff = float((z_content - mu_content).abs().max().item())
                    raise RuntimeError(
                        f"Eval latent mismatch detected for HR content: max|z_content-mu_content|={diff:.8e}."
                    )
                verified_hr_eval_mu = True

                z = mu_content
                latent_finite_mask = finite_sample_mask(z)
                latent_finite_skipped = int((~latent_finite_mask).sum().item())
                if latent_finite_skipped > 0:
                    skipped_latent_samples += latent_finite_skipped
                    if int(latent_finite_mask.sum().item()) <= 0:
                        continue
                    z = z[latent_finite_mask]

                hr_stats.update(z)

                batches_processed += 1
                skipped_samples_total = skipped_input_samples + skipped_latent_samples
                pbar.set_postfix({"Batches": str(batches_processed), "Skipped": str(skipped_samples_total)})

    if batches_processed <= 0:
        raise RuntimeError("No batches were processed. Check dataset path and --max-batches.")

    if not verified_hr_eval_mu:
        raise RuntimeError("HR latent was not verified for eval z==mu.")

    global_stats_dict = hr_stats.finalize()
    latent_scaling_factor = float(global_stats_dict["scaling_factor"])
    hr_row = {
        "stage": HR_STAGE_NAME,
        "skipped_input_samples": int(skipped_input_samples),
        "skipped_latent_samples": int(skipped_latent_samples),
        **global_stats_dict,
    }
    skipped_summary = {
        "input_samples": int(skipped_input_samples),
        "latent_samples": int(skipped_latent_samples),
        "nonfinite_values_after_filter": int(global_stats_dict["skipped_nonfinite"]),
    }

    payload = {
        "vae_checkpoint": str(vae_ckpt.resolve()),
        "data_root": str(root_dir.resolve()),
        "load_ema": bool(effective_load_ema),
        "device": str(device),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_batches": None if max_batches is None else int(max_batches),
        "seed": int(seed),
        "eval_mode": True,
        "vae_training_domain": "hr_content_only",
        "latent_source": "encode(...) -> reparameterize(...)[1] (posterior mean)",
        "eval_z_equals_mu_verified": True,
        "latent_scaling_factor": latent_scaling_factor,
        "global_stats": global_stats_dict,
        "hr_stats": hr_row,
        "stages": {HR_STAGE_NAME: hr_row},
        "skipped_nonfinite": skipped_summary,
    }

    print(f"\n{CSUA_LDM_VAE_NODISENT_TAG} Content-only latent scaling diagnostics")
    format_hr_stats(hr_row)
    print(
        "\nGlobal latent stats: "
        f"numel={int(global_stats_dict['numel'])}, "
        f"mean={float(global_stats_dict['mean']):.8f}, "
        f"std={float(global_stats_dict['std']):.8f}"
    )
    skipped_total = (
        int(skipped_input_samples)
        + int(skipped_latent_samples)
        + int(global_stats_dict["skipped_nonfinite"])
    )
    if skipped_total > 0:
        print(
            "Skipped non-finite data: "
            f"input_samples={int(skipped_input_samples)}, "
            f"latent_samples={int(skipped_latent_samples)}, "
            f"latent_values={int(global_stats_dict['skipped_nonfinite'])}"
        )
    print(f"\nLATENT_SCALING_FACTOR = {latent_scaling_factor}")
    print("Recommended usage:")
    print("  latent_scaled = latent * scaling_factor")
    print("  latent = latent_scaled / scaling_factor")
    return latent_scaling_factor, payload



def main():
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)
    root_dir = Path(args.root_dir)
    vae_ckpt = Path(args.vae_ckpt)

    latent_scaling_factor, _ = compute_latent_scaling_factor(
        root_dir=root_dir,
        vae_ckpt=vae_ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        device=device,
        seed=args.seed,
    )

    print('latent_scaling_factor: ', latent_scaling_factor)
    write_latent_scaling_factor_to_config(latent_scaling_factor)

    return latent_scaling_factor


if __name__ == "__main__":
    main()
