import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VAE_ROOT = PROJECT_ROOT / "Main_Model" / "CSUA_LDM_VAE"
VAE_CODE_ROOT = VAE_ROOT / "CSUA_LDM_VAE_Code"

for candidate in (CURRENT_DIR, PROJECT_ROOT, VAE_ROOT, VAE_CODE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from Main_Model.CSUA_LDM_Config import *
from CSUA_LDM_Diffusion_helpers import (
    cosine_beta_scheduler,
)
from CSUA_LDM_Utils import (
    VAEDecoderPropagator,
    _clamp_var,
    load_model,
    set_random_seed,
)
from CSUA_LDM_Diffusion_model import CSUA_LDM_Diffusion_UNet
from CSUA_LDM_Diffusion_sampler import DDPMSampler
from CSUA_LDM_VAE_model import LatentAutoEncoder


def _load_or_reuse_vae_model(vae_model, latent_ch: int):
    owns_vae = vae_model is None
    prev_vae_train = None

    if owns_vae:
        if not os.path.exists(CSUA_LDM_VAE_SAVEPATH):
            raise FileNotFoundError(f"{CSUA_LDM_VAE_SAVEPATH} does not exist")
        model = LatentAutoEncoder(
            img_channels=CSUA_LDM_INPUT_CHANNELS,
            z_channels=CSUA_LDM_LATENT_HIDDENCHANNEL,
        ).to(DEVICE)
        model, _, _, _, _, _, _ = load_model(
            CSUA_LDM_VAE_SAVEPATH,
            model,
            None,
            None,
            False,
            load_ema=True,
        )
        for p_ in model.parameters():
            p_.requires_grad = False
    else:
        model = vae_model
        prev_vae_train = model.training

    model.eval()
    if int(getattr(model, "z_channels", latent_ch)) != int(latent_ch):
        raise ValueError(
            f"VAE_GNLL_F latent mismatch: expected {latent_ch}, got {getattr(model, 'z_channels', None)}"
        )
    return model, owns_vae, prev_vae_train


def _load_or_reuse_diffusion_model(diffusion_model, content_ch: int):
    owns_diffusion = diffusion_model is None
    prev_diff_train = None

    if owns_diffusion:
        if not os.path.exists(CSUA_LDM_DIFFUSION_MODEL_SAVEPATH):
            raise FileNotFoundError(
                f"{CSUA_LDM_DIFFUSION_MODEL_SAVEPATH} does not exist"
            )

        model = CSUA_LDM_Diffusion_UNet(
            ch=CSUA_LDM_DIFFUSION_UNET_CH,
            out_ch=content_ch,
            ch_mult=CSUA_LDM_DIFFUSION_UNET_CH_MULT,
            attn_resolutions=CSUA_LDM_DIFFUSION_UNET_ATTN_RESOLUTIONS,
            dropout=CSUA_LDM_DIFFUSION_UNET_DROPOUT,
            resamp_with_conv=CSUA_LDM_DIFFUSION_UNET_RESAMP_WITH_CONV,
            in_channels=content_ch,
            resolution=CSUA_LDM_DIFFUSION_UNET_RESOLUTION,
            cond_channels=content_ch,
            cond_scale=CSUA_LDM_DIFFUSION_COND_SCALE,
            window_attn_min_resolution=CSUA_LDM_DIFFUSION_WINDOW_ATTN_MIN_RESOLUTION,
        attn_base_head_dim=CSUA_LDM_DIFFUSION_ATTN_BASE_HEAD_DIM,
        attn_heads_cap=CSUA_LDM_DIFFUSION_ATTN_HEADS_CAP,
        ).to(DEVICE)

        model, _, _, _, _, _, _ = load_model(
            CSUA_LDM_DIFFUSION_MODEL_SAVEPATH,
            model,
            None,
            None,
            resume_training=False,
            load_ema=True,
        )
        for p_ in model.parameters():
            p_.requires_grad = False
    else:
        model = diffusion_model
        prev_diff_train = model.training

    model.eval()
    return model, owns_diffusion, prev_diff_train


def _restore_modes(diffusion_model, vae_model, owns_diffusion, owns_vae, prev_diff_train, prev_vae_train):
    if not owns_diffusion and prev_diff_train is not None:
        diffusion_model.train(prev_diff_train)
    if not owns_vae and prev_vae_train is not None:
        vae_model.train(prev_vae_train)


def _accumulate_history_uncertainty(
    sampler: DDPMSampler,
    sample_latent_var_scaled: torch.Tensor,
    input_uncert_var: torch.Tensor,
    scale: torch.Tensor,
    start_step: int,
) -> torch.Tensor:
    """Use the sampler-propagated uncertainty directly in latent space.

    The sampler output already contains the naturally attenuated historical
    uncertainty plus the uncertainty introduced by the current denoising step.
    Do not explicitly add the full incoming branch uncertainty back here.
    """
    scale_sq = scale.pow(2).clamp(min=1e-12)
    sample_latent_var = _clamp_var(sample_latent_var_scaled / scale_sq, min_val=1e-8, max_val=1e6)
    return sample_latent_var


def _extract_guidance_diagnostics(sample_info: dict) -> dict:
    return {k: v for k, v in sample_info.items() if str(k).startswith("guidance_")}


@torch.no_grad()
def CSUA_LDM_Diffusion_Denoise(
    y0_e_content=None,
    y0_e_style=None,
    y0_sigma_sq_e_content=None,
    y0_sigma_sq_e_style=None,
    x0_gt_e_content=None,
    x0_gt_mu_content=None,
    steps=CSUA_LDM_DIFFUSION_STEPS,
    vae_model=None,
    diffusion_model=None,
    show_pbar=False,
    x_init=None,
    init_seed=RANDOM_SEED,
    print_recon_loss: bool = True,
    return_uncertainty: bool = False,
    decode_style_override: torch.Tensor | None = None,
    decode_logvar_style_override: torch.Tensor | None = None,
):
    if y0_e_content is None or y0_e_style is None:
        raise ValueError("`y0_e_content` and `y0_e_style` must not be None.")
    if x0_gt_e_content is None:
        raise ValueError("`x0_gt_e_content` must not be None.")
    if y0_sigma_sq_e_content is None or y0_sigma_sq_e_style is None:
        raise ValueError("`y0_sigma_sq_e_content` and `y0_sigma_sq_e_style` must not be None.")

    if y0_e_content.shape != x0_gt_e_content.shape:
        raise ValueError(
            f"Latent shape mismatch: y0_e_content={tuple(y0_e_content.shape)} "
            f"vs x0_gt_e_content={tuple(x0_gt_e_content.shape)}"
        )
    if x0_gt_mu_content is not None and y0_e_content.shape != x0_gt_mu_content.shape:
        raise ValueError(
            f"Latent shape mismatch: y0_e_content={tuple(y0_e_content.shape)} "
            f"vs x0_gt_mu_content={tuple(x0_gt_mu_content.shape)}"
        )
    if y0_sigma_sq_e_content.shape != y0_e_content.shape or y0_sigma_sq_e_style.shape != y0_e_style.shape:
        raise ValueError(
            f"y0_sigma_sq shape mismatch: expected ({tuple(y0_e_content.shape)}, {tuple(y0_e_style.shape)}), "
            f"got ({tuple(y0_sigma_sq_e_content.shape)}, {tuple(y0_sigma_sq_e_style.shape)})"
        )

    content_ch = CSUA_LDM_VAE_CONTENT_CHANNELS
    style_ch = y0_e_style.shape[1]
    if int(y0_e_content.shape[1]) != content_ch:
        raise ValueError(
            f"y0_e_content channels ({y0_e_content.shape[1]}) must equal content_ch ({content_ch})."
        )
    if content_ch + style_ch != CSUA_LDM_LATENT_HIDDENCHANNEL:
        raise ValueError(
            f"content_ch ({content_ch}) + style_ch ({style_ch}) must equal "
            f"CSUA_LDM_LATENT_HIDDENCHANNEL ({CSUA_LDM_LATENT_HIDDENCHANNEL})."
        )

    y0_e_content = y0_e_content.to(DEVICE)
    y0_e_style = y0_e_style.to(DEVICE)
    y0_sigma_sq_e_content = y0_sigma_sq_e_content.to(DEVICE)
    y0_sigma_sq_e_style = y0_sigma_sq_e_style.to(DEVICE)
    x0_gt_e_content = x0_gt_e_content.to(DEVICE)
    recon_target_content = x0_gt_e_content
    if x0_gt_mu_content is not None:
        x0_gt_mu_content = x0_gt_mu_content.to(DEVICE)
        recon_target_content = x0_gt_mu_content

    if decode_style_override is not None:
        decode_style_override = decode_style_override.to(DEVICE)
        if decode_logvar_style_override is not None:
            decode_logvar_style_override = decode_logvar_style_override.to(DEVICE)
    scale = torch.tensor(float(LATENT_SCALING_FACTOR), device=y0_e_content.device, dtype=y0_e_content.dtype).view(1, 1, 1, 1)

    y0_e_content_scaled = y0_e_content * scale
    y0_sigma_sq_scaled = y0_sigma_sq_e_content * scale.pow(2)

    vae, owns_vae, prev_vae_train = _load_or_reuse_vae_model(
        vae_model,
        latent_ch=CSUA_LDM_LATENT_HIDDENCHANNEL
    )
    propagator = VAEDecoderPropagator(vae)
    diffusion, owns_diffusion, prev_diff_train = _load_or_reuse_diffusion_model(
        diffusion_model,
        content_ch=content_ch,
    )

    betas = cosine_beta_scheduler(
        timesteps=steps,
        device=DEVICE,
        dtype=torch.float32,
    )
    sampler = DDPMSampler(
        diffusion,
        betas=betas,
    ).to(DEVICE)

    set_random_seed(init_seed)

    x_init_raw = None if x_init is None else x_init.to(device=y0_e_content.device, dtype=y0_e_content.dtype)
    x_init_content = None if x_init_raw is None else x_init_raw[:, :content_ch, ...]
    x_init_content_scaled = None if x_init_content is None else x_init_content * scale
    if x_init_content_scaled is None:
        x_init_raw_use = torch.randn_like(y0_e_content_scaled)
    else:
        x_init_raw_use = x_init_content_scaled
    if bool((x_init_raw_use.abs().sum() == 0).item()):
        x_init_raw_use = 0.1 * torch.randn_like(y0_e_content_scaled)
    x_init_scaled = x_init_raw_use
    if int(x_init_scaled.shape[1]) != content_ch:
        raise ValueError(
            f"x_init channels ({x_init_scaled.shape[1]}) must equal content channels ({content_ch})."
        )

    x0_hat_e_content_scaled, sample_info = sampler(
        y0=y0_e_content_scaled,
        x_init=x_init_scaled,
        show_pbar=show_pbar,
        y0_sigma_sq=y0_sigma_sq_scaled,
    )
    x0_hat_e_content_scaled = torch.nan_to_num(x0_hat_e_content_scaled, nan=0.0, posinf=1e4, neginf=-1e4)

    x0_hat_e_content = x0_hat_e_content_scaled / scale.clamp(min=1e-12)

    # Select style for final decoding. By default use condition-side style;
    # override allows draw visualization to use target (hr) style instead.
    style_for_decode = y0_e_style if decode_style_override is None else decode_style_override
    style_logvar_for_decode = y0_sigma_sq_e_style if decode_logvar_style_override is None else decode_logvar_style_override
    full_latent = torch.cat([x0_hat_e_content, style_for_decode], dim=1)

    if bool(print_recon_loss):
        recon = F.mse_loss(x0_hat_e_content, recon_target_content).item()
        full_ch = float(full_latent.shape[1])
        print(f"[Latent Stats] FullCh: {full_ch:.1f}, Recon={recon:.6f} (content only)")

    reverse_var_content = _clamp_var(sample_info["reverse_var"] / scale.pow(2).clamp(min=1e-12), min_val=1e-8, max_val=1e6)
    latent_uncert_var_content = _accumulate_history_uncertainty(
        sampler=sampler,
        sample_latent_var_scaled=sample_info["latent_var"],
        input_uncert_var=y0_sigma_sq_e_content,
        scale=scale,
        start_step=sample_info.get("start_step", torch.tensor(sampler.T - 1, device=x0_hat_e_content_scaled.device, dtype=torch.long)),
    )
    style_var = _clamp_var(style_logvar_for_decode, min_val=1e-8, max_val=1e6)
    # Style is fixed during reverse diffusion, so it has no reverse-step variance.
    reverse_var = torch.cat([reverse_var_content, torch.zeros_like(style_var)], dim=1)
    latent_uncert_var = torch.cat([latent_uncert_var_content, style_var], dim=1)
    reverse_logvar = torch.log(torch.clamp(reverse_var, min=1e-8))
    latent_logvar = torch.log(torch.clamp(latent_uncert_var, min=1e-8))

    if return_uncertainty:
        ret = propagator.build_uncertainty_pack(
            latent_mean=full_latent,
            reverse_var=reverse_var,
            reverse_logvar=reverse_logvar,
            latent_uncert_var=latent_uncert_var,
            logvar_content=latent_logvar[:, :content_ch, ...],
            logvar_style=latent_logvar[:, content_ch:, ...],
        )
        ret.update(_extract_guidance_diagnostics(sample_info))
        _restore_modes(diffusion, vae, owns_diffusion, owns_vae, prev_diff_train, prev_vae_train)
        return ret

    x_d, _ = propagator.decode(
        full_latent[:, :content_ch, ...], style_for_decode,
        latent_logvar[:, :content_ch, ...], latent_logvar[:, content_ch:, ...],
    )
    _restore_modes(diffusion, vae, owns_diffusion, owns_vae, prev_diff_train, prev_vae_train)
    return x_d, full_latent, latent_logvar

