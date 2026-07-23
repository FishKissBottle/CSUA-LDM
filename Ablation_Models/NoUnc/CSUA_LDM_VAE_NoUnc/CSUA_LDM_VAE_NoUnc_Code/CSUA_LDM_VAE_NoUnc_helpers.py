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

from CSUA_LDM_NoUnc_Config import (
    DEVICE as CONFIG_DEVICE,
    CSUA_LDM_INPUT_CHANNELS,
    CSUA_LDM_LATENT_HIDDENCHANNEL,
    CSUA_LDM_VIS_BAND_ORDER,
    CSUA_LDM_VAE_NOUNC_CONTENT_CHANNELS,
    CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
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
from CSUA_LDM_VAE_NoUnc_model import LatentAutoEncoder


VAR_EPS = 1e-12


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
) -> tuple[LatentAutoEncoder, bool, int]:
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")

    model = LatentAutoEncoder(
        img_channels=CSUA_LDM_INPUT_CHANNELS,
        z_channels=CSUA_LDM_LATENT_HIDDENCHANNEL,
        content_channels=CSUA_LDM_VAE_NOUNC_CONTENT_CHANNELS,
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
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
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
        if isinstance(rep_out, (list, tuple)) and len(rep_out) >= 6:
            mu = torch.cat([rep_out[2], rep_out[3]], dim=1)
        elif isinstance(rep_out, (list, tuple)) and len(rep_out) >= 2:
            mu = rep_out[0]
        else:
            mu = rep_out[0] if isinstance(rep_out, (list, tuple)) else rep_out
        chunks.append(mu)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_images_to_content_style(
    imgs: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode images and return separate content and style mu."""
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

    content_chunks = []
    style_chunks = []
    all_indices = torch.arange(total_samples, device=device, dtype=torch.long)
    for idx_chunk in iter_microbatch_indices(all_indices, draw_microbatch_size):
        imgs_chunk = imgs_in.index_select(0, idx_chunk).contiguous().to(dtype=vae_dtype)
        z_params = model.encode(imgs_chunk)
        _, _, mu_content, mu_style, _, _ = model.reparameterize(z_params)
        content_chunks.append(mu_content)
        style_chunks.append(mu_style)
    return (
        torch.cat(content_chunks, dim=0),
        torch.cat(style_chunks, dim=0),
    )


@torch.no_grad()
def decode_latents(
    latents: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> torch.Tensor:
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
    all_indices = torch.arange(total_samples, device=device, dtype=torch.long)

    for idx_chunk in iter_microbatch_indices(all_indices, draw_microbatch_size):
        latents_chunk = latents_in.index_select(0, idx_chunk).contiguous().to(dtype=vae_dtype)
        content = latents_chunk[:, :model.content_channels]
        style = latents_chunk[:, model.content_channels:]
        pred_mean = model.decode(content, style)
        pred_mean_chunks.append(pred_mean)

    return torch.cat(pred_mean_chunks, dim=0)


@torch.no_grad()
def decode_content_with_style(
    content: torch.Tensor,
    style: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Decode by concatenating content and style latents."""
    z_full = torch.cat([ensure_batched_4d(content), ensure_batched_4d(style)], dim=1)
    return decode_latents(
        z_full,
        model,
        draw_microbatch_size=draw_microbatch_size,
        device=device,
    )


@torch.no_grad()
def reproject_images_to_content_style_mu(
    imgs: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    _, _, mu_content, mu_style, _, _ = model.reparameterize(z_params)
    reproj_mean = model.decode(mu_content, mu_style)

    reproj_params = model.encode(reproj_mean)
    _, _, reproj_content, reproj_style, _, _ = model.reparameterize(reproj_params)
    return reproj_mean, reproj_content, reproj_style


@torch.no_grad()
def generate_samples(
    imgs: torch.Tensor,
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    latents = encode_images_to_mu(
        imgs,
        model,
        draw_microbatch_size=draw_microbatch_size,
        device=device,
    )
    return decode_latents(
        latents,
        model,
        draw_microbatch_size=draw_microbatch_size,
        device=device,
    )


@torch.no_grad()
def generate_stage_reconstruction_outputs(
    stage_cache: dict[str, torch.Tensor],
    model,
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    if not isinstance(stage_cache, dict):
        raise TypeError("`generate_stage_reconstruction_outputs` expects a stage cache dict.")

    stage_outputs: dict[str, dict[str, torch.Tensor]] = {}
    for stage_name in stage_cache:
        if str(stage_name).startswith("_"):
            continue
        stage_norm = get_stage_img(stage_cache, stage_name)
        content, style = encode_images_to_content_style(
            stage_norm,
            model,
            draw_microbatch_size=draw_microbatch_size,
            device=device,
        )
        stage_latent = torch.cat([content, style], dim=1)
        pred_mean = decode_latents(
            stage_latent,
            model,
            draw_microbatch_size=draw_microbatch_size,
            device=device,
        )
        stage_outputs[stage_name] = {
            "latent_mu": stage_latent,
            "pred_mean": pred_mean,
        }
    return stage_outputs
