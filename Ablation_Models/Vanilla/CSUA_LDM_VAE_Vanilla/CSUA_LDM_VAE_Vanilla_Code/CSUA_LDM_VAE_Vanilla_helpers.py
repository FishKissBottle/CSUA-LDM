import sys
from pathlib import Path


VARIANT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = VARIANT_ROOT.parents[1]
MODULE_DIR = Path(__file__).resolve().parent
for search_path in (REPO_ROOT, VARIANT_ROOT, MODULE_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import torch

from CSUA_LDM_Vanilla_Config import (
    DEVICE as CONFIG_DEVICE,
    CSUA_LDM_INPUT_CHANNELS,
    CSUA_LDM_LATENT_HIDDENCHANNEL,
    CSUA_LDM_VIS_BAND_ORDER,
    CSUA_LDM_VAE_VANILLA_DRAW_MICROBATCH_SIZE,
    hr_ch_mean,
    hr_ch_std,
)
from CSUA_LDM_Quality_Metrics import (
    compute_FSIM,
    compute_GMSD,
    compute_LPIPS,
    compute_PSNR,
    compute_SAM,
    compute_SSIM,
    format_metric_value,
)
from CSUA_LDM_Utils import (
    denorm_mean,
    denorm_std,
    ensure_batched_4d,
    load_model,
    norm_img,
)
from CSUA_LDM_VAE_Vanilla_model import LatentAutoEncoderVanilla


def _to_float_scalar(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            value = value.mean()
        return float(value.detach().item())
    return float(value)


def evaluate_quality_metrics(rebuild_imgs: torch.Tensor, real_imgs: torch.Tensor) -> dict[str, float]:
    psnr_val = _to_float_scalar(compute_PSNR(rebuild_imgs, real_imgs))
    ssim_val = _to_float_scalar(compute_SSIM(rebuild_imgs, real_imgs))
    sam_val = _to_float_scalar(compute_SAM(rebuild_imgs, real_imgs))

    gmsd_rgb_val, gmsd_nir_val = compute_GMSD(
        rebuild_imgs,
        real_imgs,
        rgb_indices=CSUA_LDM_VIS_BAND_ORDER,
    )
    fsim_rgb_val, fsim_nir_val = compute_FSIM(
        rebuild_imgs,
        real_imgs,
        rgb_indices=CSUA_LDM_VIS_BAND_ORDER,
    )
    lpips_rgb_val, lpips_nir_val = compute_LPIPS(
        rebuild_imgs,
        real_imgs,
        rgb_indices=CSUA_LDM_VIS_BAND_ORDER,
    )

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "sam": sam_val,
        "gmsd_rgb": _to_float_scalar(gmsd_rgb_val),
        "gmsd_nir": _to_float_scalar(gmsd_nir_val),
        "fsim_rgb": _to_float_scalar(fsim_rgb_val),
        "fsim_nir": _to_float_scalar(fsim_nir_val),
        "lpips_rgb": _to_float_scalar(lpips_rgb_val),
        "lpips_nir": _to_float_scalar(lpips_nir_val),
    }


def format_metric_summary(tag: str, metrics: dict[str, float]) -> str:
    return (
        f"{tag}: PSNR={format_metric_value(metrics['psnr'])}, "
        f"SSIM={format_metric_value(metrics['ssim'])}, "
        f"SAM={format_metric_value(metrics['sam'])}, "
        f"GMSD_RGB={format_metric_value(metrics['gmsd_rgb'])}, "
        f"GMSD_NIR={format_metric_value(metrics['gmsd_nir'])}, "
        f"FSIM_RGB={format_metric_value(metrics['fsim_rgb'])}, "
        f"FSIM_NIR={format_metric_value(metrics['fsim_nir'])}, "
        f"LPIPS_RGB={format_metric_value(metrics['lpips_rgb'])}, "
        f"LPIPS_NIR={format_metric_value(metrics['lpips_nir'])}"
    )


def build_stage_cache_from_batch(
    batch,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"`batch` must be a tuple/list, got {type(batch)!r}.")
    if len(batch) != 7:
        raise ValueError(
            "VAE stage cache expects the super-resolution 7-tuple "
            "(lr_img, lr_up_img, hr_down_img, hr_down_up_img, hr_img, proj, geo), "
            f"got length {len(batch)}."
        )

    cache_device = torch.device(device)
    _, lr_up_img, _, hr_down_up_img, hr_img, _, _ = batch
    raw_lr_up = ensure_batched_4d(lr_up_img).detach().to(device=cache_device, dtype=torch.float32)
    raw_hr_down_up = ensure_batched_4d(hr_down_up_img).detach().to(device=cache_device, dtype=torch.float32)
    raw_hr = ensure_batched_4d(hr_img).detach().to(device=cache_device, dtype=torch.float32)
    stage_cache: dict[str, torch.Tensor] = {
        "lr_up": raw_lr_up.detach(),
        "hr_down_up": raw_hr_down_up.detach(),
        "hr": raw_hr.detach(),
    }
    return stage_cache


def get_stage_img(stage_cache: dict[str, torch.Tensor], stage_name: str) -> torch.Tensor:
    stage_name = str(stage_name).lower()
    if stage_name in stage_cache:
        out = stage_cache[stage_name]
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Cached stage `{stage_name}` is not a tensor.")
        return out
    raise KeyError(f"Stage `{stage_name}` is missing from the VAE stage cache.")


def iter_microbatch_indices(sample_indices: torch.Tensor, microbatch_size: int):
    idx_view = sample_indices.detach().to(dtype=torch.long).view(-1)
    chunk_size = max(int(microbatch_size), 1)
    for start in range(0, int(idx_view.numel()), chunk_size):
        yield idx_view[start:start + chunk_size]


def build_vae_model(
    vae_ckpt: Path,
    device: torch.device,
    load_ema: bool = True,
) -> tuple[LatentAutoEncoderVanilla, bool, int]:
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")

    model = LatentAutoEncoderVanilla(
        img_channels=CSUA_LDM_INPUT_CHANNELS,
        z_channels=CSUA_LDM_LATENT_HIDDENCHANNEL,
    ).to(device)

    effective_load_ema = bool(load_ema)
    loaded_epoch = -1
    try:
        model, _, _, loaded_epoch, _, _, _ = load_model(
            vae_ckpt,
            model,
            None,
            None,
            resume_training=False,
            load_ema=effective_load_ema,
        )
    except KeyError:
        if not effective_load_ema:
            raise
        effective_load_ema = False
        model, _, _, loaded_epoch, _, _, _ = load_model(
            vae_ckpt,
            model,
            None,
            None,
            resume_training=False,
            load_ema=effective_load_ema,
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, effective_load_ema, int(loaded_epoch)


def _get_module_param_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    return torch.float32


@torch.no_grad()
def encode_images_to_mu(
    imgs: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_VANILLA_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    draw_microbatch_size = int(draw_microbatch_size)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(CONFIG_DEVICE)
    device = torch.device(device)
    imgs_in = ensure_batched_4d(imgs).detach().to(device=device)
    total_samples = int(imgs_in.shape[0])
    if total_samples <= 0:
        raise ValueError("`imgs` must contain at least one sample.")

    vae_dtype = _get_module_param_dtype(model)

    chunks = []
    all_indices = torch.arange(total_samples, device=device, dtype=torch.long)
    for idx_chunk in iter_microbatch_indices(all_indices, draw_microbatch_size):
        imgs_chunk = imgs_in.index_select(0, idx_chunk).contiguous().to(dtype=vae_dtype)
        z_params = model.encode(imgs_chunk)
        rep_out = model.reparameterize(z_params)
        if isinstance(rep_out, (list, tuple)) and len(rep_out) >= 2:
            mu = rep_out[1]
        else:
            mu = rep_out[0] if isinstance(rep_out, (list, tuple)) else rep_out
        chunks.append(mu)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def decode_latents_deterministic(
    latents: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_VANILLA_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    draw_microbatch_size = int(draw_microbatch_size)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(CONFIG_DEVICE)
    device = torch.device(device)
    latents_in = ensure_batched_4d(latents).detach().to(device=device)
    total_samples = int(latents_in.shape[0])
    if total_samples <= 0:
        raise ValueError("`latents` must contain at least one sample.")

    vae_dtype = _get_module_param_dtype(model)

    pred_mean_chunks = []
    pred_var_chunks = []
    pred_std_chunks = []
    all_indices = torch.arange(total_samples, device=device, dtype=torch.long)

    for idx_chunk in iter_microbatch_indices(all_indices, draw_microbatch_size):
        latents_chunk = latents_in.index_select(0, idx_chunk).contiguous().to(dtype=vae_dtype)
        pred_mean = model.decode(latents_chunk)
        pred_var = torch.zeros_like(pred_mean)
        pred_std = torch.zeros_like(pred_mean)
        pred_mean_chunks.append(pred_mean)
        pred_var_chunks.append(pred_var)
        pred_std_chunks.append(pred_std)

    return (
        torch.cat(pred_mean_chunks, dim=0),
        torch.cat(pred_var_chunks, dim=0),
        torch.cat(pred_std_chunks, dim=0),
    )


@torch.no_grad()
def reproject_images_to_latent_mu(
    imgs: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_VANILLA_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the stable reprojection condition chain for raw image inputs."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(CONFIG_DEVICE)
    device = torch.device(device)
    vae_dtype = _get_module_param_dtype(model)

    # Outer train/eval microbatching already controls memory, so reprojection
    # runs as one direct batch here instead of nesting another draw-microbatch split.
    imgs_norm = norm_img(ensure_batched_4d(imgs), hr_ch_mean, hr_ch_std).detach().to(device=device, dtype=vae_dtype)
    z_params = model.encode(imgs_norm)
    _, mu_latent, _ = model.reparameterize(z_params)
    reproj_mean = model.decode(mu_latent)

    reproj_params = model.encode(reproj_mean)
    _, reproj_latent, _ = model.reparameterize(reproj_params)
    return reproj_mean, reproj_latent


@torch.no_grad()
def generate_samples_deterministic(
    imgs: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_VANILLA_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = encode_images_to_mu(
        imgs,
        model,
        draw_microbatch_size=draw_microbatch_size,
        device=device,
    )
    return decode_latents_deterministic(
        latents,
        model,
        draw_microbatch_size=draw_microbatch_size,
        device=device,
    )


@torch.no_grad()
def generate_stage_reconstruction_outputs(
    stage_cache: dict[str, torch.Tensor],
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_VANILLA_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    if not isinstance(stage_cache, dict):
        raise TypeError("`generate_stage_reconstruction_outputs` expects a stage cache dict.")

    stage_outputs: dict[str, dict[str, torch.Tensor]] = {}
    for stage_name in stage_cache:
        if str(stage_name).startswith("_"):
            continue
        stage_norm = get_stage_img(stage_cache, stage_name)
        stage_latent = encode_images_to_mu(
            stage_norm,
            model,
            draw_microbatch_size=draw_microbatch_size,
            device=device,
        )
        pred_mean, pred_var, pred_std = decode_latents_deterministic(
            stage_latent,
            model,
            draw_microbatch_size=draw_microbatch_size,
            device=device,
        )
        stage_outputs[stage_name] = {
            "latent_mu": stage_latent,
            "pred_mean": pred_mean,
            "pred_var": pred_var,
            "pred_std": pred_std,
        }
    return stage_outputs
