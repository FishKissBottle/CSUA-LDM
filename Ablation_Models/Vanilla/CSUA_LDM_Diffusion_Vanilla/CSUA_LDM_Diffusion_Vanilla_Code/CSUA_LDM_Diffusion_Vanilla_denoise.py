import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = VARIANT_ROOT.parents[1]
VAE_ROOT = VARIANT_ROOT / "CSUA_LDM_VAE_Vanilla"
VAE_CODE_ROOT = VAE_ROOT / "CSUA_LDM_VAE_Vanilla_Code"

for candidate in (CURRENT_DIR, PROJECT_ROOT, VARIANT_ROOT, VAE_ROOT, VAE_CODE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


import os

import torch
import torch.nn.functional as F

from CSUA_LDM_Vanilla_Config import *
from CSUA_LDM_Diffusion_Vanilla_helpers import (
    cosine_beta_scheduler,
)
from CSUA_LDM_Utils import load_model, set_random_seed
from CSUA_LDM_Diffusion_Vanilla_model import CSUA_LDM_Diffusion_Vanilla_UNet
from CSUA_LDM_Diffusion_Vanilla_sampler import DDPMSampler
from CSUA_LDM_VAE_Vanilla_model import LatentAutoEncoderVanilla


def _load_or_reuse_vae_model(vae_model, latent_ch: int):
    owns_vae = vae_model is None
    prev_vae_train = None

    if owns_vae:
        if not os.path.exists(CSUA_LDM_VAE_VANILLA_SAVEPATH):
            raise FileNotFoundError(f"{CSUA_LDM_VAE_VANILLA_SAVEPATH} does not exist")
        model = LatentAutoEncoderVanilla(
            img_channels=CSUA_LDM_INPUT_CHANNELS,
            z_channels=CSUA_LDM_LATENT_HIDDENCHANNEL,
        ).to(DEVICE)
        model, _, _, _, _, _, _ = load_model(
            CSUA_LDM_VAE_VANILLA_SAVEPATH,
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
            f"VAE latent mismatch: expected {latent_ch}, got {getattr(model, 'z_channels', None)}"
        )
    return model, owns_vae, prev_vae_train


def _load_or_reuse_diffusion_model(diffusion_model, latent_ch: int):
    owns_diffusion = diffusion_model is None
    prev_diff_train = None

    if owns_diffusion:
        if not os.path.exists(CSUA_LDM_DIFFUSION_VANILLA_MODEL_SAVEPATH):
            raise FileNotFoundError(
                f"{CSUA_LDM_DIFFUSION_VANILLA_MODEL_SAVEPATH} does not exist"
            )

        model = CSUA_LDM_Diffusion_Vanilla_UNet(
            ch=CSUA_LDM_DIFFUSION_VANILLA_UNET_CH,
            out_ch=latent_ch,
            ch_mult=CSUA_LDM_DIFFUSION_VANILLA_UNET_CH_MULT,
            attn_resolutions=CSUA_LDM_DIFFUSION_VANILLA_UNET_ATTN_RESOLUTIONS,
            dropout=CSUA_LDM_DIFFUSION_VANILLA_UNET_DROPOUT,
            resamp_with_conv=CSUA_LDM_DIFFUSION_VANILLA_UNET_RESAMP_WITH_CONV,
            in_channels=latent_ch,
            resolution=CSUA_LDM_DIFFUSION_VANILLA_UNET_RESOLUTION,
            cond_channels=latent_ch,
            cond_scale=CSUA_LDM_DIFFUSION_VANILLA_COND_SCALE,
            window_attn_min_resolution=CSUA_LDM_DIFFUSION_VANILLA_WINDOW_ATTN_MIN_RESOLUTION,
            attn_base_head_dim=CSUA_LDM_DIFFUSION_VANILLA_ATTN_BASE_HEAD_DIM,
            attn_heads_cap=CSUA_LDM_DIFFUSION_VANILLA_ATTN_HEADS_CAP,
        ).to(DEVICE)

        model, _, _, _, _, _, _ = load_model(
            CSUA_LDM_DIFFUSION_VANILLA_MODEL_SAVEPATH,
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


@torch.no_grad()
def CSUA_LDM_Diffusion_Vanilla_Denoise(
    y0_e=None,
    x0_gt_e=None,
    x0_gt_mu_e=None,
    steps=CSUA_LDM_DIFFUSION_VANILLA_STEPS,
    vae_model=None,
    diffusion_model=None,
    show_pbar=False,
    x_init=None,
    init_seed=RANDOM_SEED,
    print_recon_loss: bool = True,
):
    if y0_e is None:
        raise ValueError("`y0_e` must not be None.")
    if x0_gt_e is None:
        raise ValueError("`x0_gt_e` must not be None.")

    if y0_e.shape != x0_gt_e.shape:
        raise ValueError(
            f"Latent shape mismatch: y0_e={tuple(y0_e.shape)} "
            f"vs x0_gt_e={tuple(x0_gt_e.shape)}"
        )
    if x0_gt_mu_e is not None and y0_e.shape != x0_gt_mu_e.shape:
        raise ValueError(
            f"Latent shape mismatch: y0_e={tuple(y0_e.shape)} "
            f"vs x0_gt_mu_e={tuple(x0_gt_mu_e.shape)}"
        )

    latent_ch = CSUA_LDM_LATENT_HIDDENCHANNEL
    if int(y0_e.shape[1]) != latent_ch:
        raise ValueError(
            f"y0_e channels ({y0_e.shape[1]}) must equal latent_ch ({latent_ch})."
        )

    y0_e = y0_e.to(DEVICE)
    x0_gt_e = x0_gt_e.to(DEVICE)
    recon_target = x0_gt_e
    if x0_gt_mu_e is not None:
        x0_gt_mu_e = x0_gt_mu_e.to(DEVICE)
        recon_target = x0_gt_mu_e
    scale = torch.tensor(float(LATENT_SCALING_FACTOR), device=y0_e.device, dtype=y0_e.dtype).view(1, 1, 1, 1)

    y0_e_scaled = y0_e * scale

    vae, owns_vae, prev_vae_train = _load_or_reuse_vae_model(
        vae_model,
        latent_ch=CSUA_LDM_LATENT_HIDDENCHANNEL
    )
    diffusion, owns_diffusion, prev_diff_train = _load_or_reuse_diffusion_model(
        diffusion_model,
        latent_ch=latent_ch,
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
    start_step = sampler.resolve_start_step()

    set_random_seed(init_seed)

    x_init_raw = None if x_init is None else x_init.to(device=y0_e.device, dtype=y0_e.dtype)
    x_init_scaled = None if x_init_raw is None else x_init_raw * scale
    if x_init_scaled is None:
        x_init_scaled = torch.randn_like(y0_e_scaled)
    else:
        eps = torch.randn_like(y0_e_scaled)
        alpha_bar_start = torch.clamp(sampler.alphas_cumprod[int(start_step)], min=1e-12)
        sqrt_alpha_bar_start = torch.sqrt(alpha_bar_start)
        sqrt_one_minus_alpha_bar_start = torch.sqrt(1.0 - alpha_bar_start)
        x_init_scaled = sqrt_alpha_bar_start * x_init_scaled + sqrt_one_minus_alpha_bar_start * eps

    if bool((x_init_scaled.abs().sum() == 0).item()):
        x_init_scaled = 0.1 * torch.randn_like(y0_e_scaled)

    if int(x_init_scaled.shape[1]) != latent_ch:
        raise ValueError(
            f"x_init channels ({x_init_scaled.shape[1]}) must equal latent channels ({latent_ch})."
        )

    x0_hat_e_scaled = sampler(
        y0=y0_e_scaled,
        x_init=x_init_scaled,
        start_step=start_step,
        show_pbar=show_pbar,
    )
    x0_hat_e_scaled = torch.nan_to_num(x0_hat_e_scaled, nan=0.0, posinf=1e4, neginf=-1e4)

    x0_hat_e = x0_hat_e_scaled / scale.clamp(min=1e-12)
    full_latent = x0_hat_e

    if bool(print_recon_loss):
        recon = F.mse_loss(x0_hat_e, recon_target).item()
        full_ch = float(full_latent.shape[1])
        print(f"[Latent Stats] FullCh: {full_ch:.1f}, Recon={recon:.6f}")

    pred_mean = vae.decode(full_latent)

    _restore_modes(diffusion, vae, owns_diffusion, owns_vae, prev_diff_train, prev_vae_train)
    return pred_mean, full_latent
