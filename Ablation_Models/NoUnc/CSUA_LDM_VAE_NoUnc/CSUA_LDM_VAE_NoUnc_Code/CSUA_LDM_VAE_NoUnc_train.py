import sys
from pathlib import Path


VARIANT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = VARIANT_ROOT.parents[1]
MODULE_DIR = Path(__file__).resolve().parent
for _search_path in (REPO_ROOT, VARIANT_ROOT, MODULE_DIR):
    search_path_str = str(_search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import gc
import math
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from osgeo import gdal
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from CSUA_LDM_NoUnc_Config import *
from CSUA_LDM_Dataset import (
    SuperResolutionDataset,
    validate_hr_image_sizes_multiple_of_8,
)
from CSUA_LDM_VAE_NoUnc_helpers import (
    build_stage_cache_from_batch,
    denorm_mean,
    denorm_std,
    norm_img,
    encode_images_to_content_style,
    evaluate_quality_metrics,
    format_metric_summary,
    generate_stage_reconstruction_outputs,
    get_stage_img,
)
from CSUA_LDM_VAE_NoUnc_model import LatentAutoEncoder

from CSUA_LDM_Utils import (
    geo_to_numpy_array,
    build_ema_model,
    get_training_monitor_state,
    iter_microbatch_slices,
    load_model,
    log_info,
    restore_rng_state,
    save_model,
    save_rgb_datas,
    save_tif_datas,
    set_random_seed,
    stash_rng_state,
    training_monitor,
    update_ema_model,
)

gdal.UseExceptions()


def ensure_output_dirs():
    os.makedirs(os.path.dirname(CSUA_LDM_VAE_NOUNC_SAVEPATH), exist_ok=True)
    os.makedirs(CSUA_LDM_VAE_NOUNC_LOG_DIR, exist_ok=True)
    os.makedirs(CSUA_LDM_VAE_NOUNC_RGB_DIR, exist_ok=True)
    os.makedirs(CSUA_LDM_VAE_NOUNC_TIF_DIR, exist_ok=True)


def kl_loss_gaussian(mu: torch.Tensor, logvar: torch.Tensor):
    """Compute per-element KL divergence (mean reduction) with numeric stability."""
    logvar_clipped = torch.clamp(logvar, min=-20.0, max=20.0)
    kl_map = -0.5 * (1 + logvar_clipped - mu.pow(2) - logvar_clipped.exp())
    return kl_map.mean()

def content_variance_loss(mu_c: torch.Tensor, target_std: float = 0.5, eps: float = 1e-4):
    """Penalty for collapsed content: encourage per-element std >= target."""
    z = mu_c.flatten(1)  # [B, C*H*W]
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()

def create_scheduler(optimizer: torch.optim.Optimizer, min_lr: float, patience: int):
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        min_lr=min_lr,
        factor=0.5,
        threshold=0.0,
        patience=patience
    )
    return scheduler


def _encode_extract_content_style(model, imgs, use_amp):
    """Encode images and extract content/style z/mu/logvar splits."""
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        z_params = model.encode(imgs)
        z_content, z_style, mu_content, mu_style, logvar_content, logvar_style = model.reparameterize(z_params)
    return z_content, z_style, mu_content, mu_style, logvar_content, logvar_style


def _get_cached_encoding(cache, key, model, imgs, use_amp):
    """Fetch encoded z_params from microbatch cache, computing on first miss."""
    if cache is None:
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            return model.encode(imgs)
    if key not in cache:
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            cache[key] = model.encode(imgs)
    return cache[key]


def _decode_compute_losses(model, z_content, z_style, target_imgs, use_amp):
    """Decode with given content/style and compute reconstruction losses (mean reduction)."""
    if CSUA_LDM_VAE_NOUNC_STYLE_DROPOUT_RATE > 0.0 and model.training:
        drop_mask = (
            torch.rand((z_style.shape[0], 1, 1, 1), device=z_style.device)
            < CSUA_LDM_VAE_NOUNC_STYLE_DROPOUT_RATE
        )
        if bool(drop_mask.any().item()):
            z_style = torch.where(drop_mask, torch.zeros_like(z_style), z_style)
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        pred_mean = model.decode(z_content, z_style)
        l1 = F.l1_loss(pred_mean, target_imgs, reduction="mean")
        l2 = F.mse_loss(pred_mean, target_imgs, reduction="mean")
    return l1, l2


def train(
    dataloader,
    model,
    optimizer,
    content_kl_weight: float = CSUA_LDM_VAE_NOUNC_CONTENT_KL_WEIGHT,
    style_kl_weight: float = CSUA_LDM_VAE_NOUNC_STYLE_KL_WEIGHT,
    l1_weight: float = CSUA_LDM_VAE_NOUNC_L1_WEIGHT,
    l2_weight: float = CSUA_LDM_VAE_NOUNC_L2_WEIGHT,
    grad_clip_max_norm: float | None = CSUA_LDM_GRAD_CLIP_MAX_NORM,
    content_align_weight: float = CSUA_LDM_VAE_NOUNC_CONTENT_ALIGN_WEIGHT,
    style_align_weight: float = CSUA_LDM_VAE_NOUNC_STYLE_ALIGN_WEIGHT,
    content_variance_weight: float = CSUA_LDM_VAE_NOUNC_CONTENT_VARIANCE_WEIGHT,
    content_variance_target_std: float = CSUA_LDM_VAE_NOUNC_CONTENT_VARIANCE_TARGET_STD,
    ema_model=None,
    ema_decay: float = CSUA_LDM_EMA_DECAY,
    train_microbatch_size: int = CSUA_LDM_VAE_NOUNC_TRAIN_MICROBATCH_SIZE,
    start_global_iteration: int = 0,
    max_iterations: int = 0,
    enable_cache: bool = CSUA_LDM_VAE_NOUNC_ENABLE_MICROBATCH_ENCODING_CACHE,
):
    """Run one training epoch with stochastic self/cross reconstruction sampling.

    Memory-optimized: microbatch selective encoding cache, split backward for self/cross/kl/align,
    and explicit del on intermediate tensors.
    """
    device = DEVICE
    model.train()
    use_amp = bool(torch.cuda.is_available())
    train_microbatch_size = max(int(train_microbatch_size), 1)

    total_loss_sum = 0.0
    l1_loss_sum = 0.0
    l2_loss_sum = 0.0
    kl_loss_sum = 0.0
    content_align_loss_sum = 0.0
    style_align_loss_sum = 0.0
    content_var_loss_sum = 0.0
    num_samples = 0
    skipped_nonfinite_loss = 0
    skipped_nonfinite_grad = 0
    updated_iterations = 0
    reached_max_iterations = False

    with tqdm(dataloader, leave=False, colour="#ff924a", ncols=130) as pbar:
        for batch in pbar:
            if max_iterations > 0 and start_global_iteration + updated_iterations >= max_iterations:
                reached_max_iterations = True
                break
            _, lr_up_img, _, hr_down_up_img, hr_img, _, _ = batch
            lr_up_img = norm_img(lr_up_img, hr_ch_mean, hr_ch_std).to(device=device, dtype=torch.float32, non_blocking=True)
            hr_down_up_img = norm_img(hr_down_up_img, hr_ch_mean, hr_ch_std).to(device=device, dtype=torch.float32, non_blocking=True)
            hr_img = norm_img(hr_img, hr_ch_mean, hr_ch_std).to(device=device, dtype=torch.float32, non_blocking=True)

            batch_size = int(lr_up_img.shape[0])
            if batch_size <= 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            # Sample branches (Self-KL bound, Cross-Align bound)
            self_branch_name = ("g1", "g2", "g3")[int(torch.randint(3, (1,)).item())]
            cross_branch_name = ("g4", "g5")[int(torch.randint(2, (1,)).item())]
            batch_total_loss_sum = 0.0
            batch_l1_loss_sum = 0.0
            batch_l2_loss_sum = 0.0
            batch_kl_loss_sum = 0.0
            batch_ca_loss_sum = 0.0
            batch_sa_loss_sum = 0.0
            batch_cv_loss_sum = 0.0
            skip_batch = False

            for mb_start, mb_end in iter_microbatch_slices(batch_size, train_microbatch_size):
                microbatch_size = int(mb_end - mb_start)
                microbatch_weight = float(microbatch_size) / float(batch_size)
                lr_up = lr_up_img[mb_start:mb_end]
                hr_down_up = hr_down_up_img[mb_start:mb_end]
                hr = hr_img[mb_start:mb_end]

                enc_cache = {} if enable_cache else None

                # ---------- Self-reconstruction + KL (bound) ----------
                if self_branch_name == "g1":
                    z_params = _get_cached_encoding(enc_cache, "lr_up", model, lr_up, use_amp)
                    zc, zs, mc, ms, lvc, lvs = model.reparameterize(z_params)
                    target_img = lr_up
                elif self_branch_name == "g2":
                    z_params = _get_cached_encoding(enc_cache, "hr_down_up", model, hr_down_up, use_amp)
                    zc, zs, mc, ms, lvc, lvs = model.reparameterize(z_params)
                    target_img = hr_down_up
                else:
                    z_params = _get_cached_encoding(enc_cache, "hr", model, hr, use_amp)
                    zc, zs, mc, ms, lvc, lvs = model.reparameterize(z_params)
                    target_img = hr

                l1, l2 = _decode_compute_losses(model, zc, zs, target_img, use_amp)
                kl_c = kl_loss_gaussian(mc, lvc)
                kl_s = kl_loss_gaussian(ms, lvs)
                kl_tensor = float(content_kl_weight) * kl_c + float(style_kl_weight) * kl_s

                self_kl_loss = (
                    float(l1_weight) * l1
                    + float(l2_weight) * l2
                    + kl_tensor
                )
                log_self_l1 = float(l1.detach().cpu().item())
                log_self_l2 = float(l2.detach().cpu().item())
                kl_val = float(kl_tensor.detach().cpu().item())

                if not bool(torch.isfinite(self_kl_loss).all().item()):
                    skipped_nonfinite_loss += 1
                    print(f"[train] Skip non-finite self_kl_loss batch, loss={float(self_kl_loss.mean().cpu().item())}")
                    if enable_cache:
                        del enc_cache
                    optimizer.zero_grad(set_to_none=True)
                    skip_batch = True
                    break

                self_kl_backward = self_kl_loss * microbatch_weight
                self_kl_backward.backward(retain_graph=enable_cache)
                if enable_cache:
                    del self_kl_backward, self_kl_loss, zc, zs, mc, ms, lvc, lvs, l1, l2, kl_c, kl_s, kl_tensor
                else:
                    del self_kl_backward, self_kl_loss, zc, zs, mc, ms, lvc, lvs, l1, l2, kl_c, kl_s, kl_tensor

                # ---------- Cross-reconstruction + Alignment (bound) ----------
                if cross_branch_name == "g4":
                    z_params1 = _get_cached_encoding(enc_cache, "lr_up", model, lr_up, use_amp)
                    zc, _, mc1, _, _, _ = model.reparameterize(z_params1)
                    z_params2 = _get_cached_encoding(enc_cache, "hr_down_up", model, hr_down_up, use_amp)
                    _, zs, mc2, _, _, _ = model.reparameterize(z_params2)
                    l1, l2 = _decode_compute_losses(model, zc, zs, hr_down_up, use_amp)
                    align_loss = F.mse_loss(mc1, mc2)
                    align_tensor = float(content_align_weight) * align_loss
                    ca_val = float(align_loss.detach().cpu().item())
                    sa_val = 0.0
                    del mc1, mc2, align_loss
                else:
                    z_params1 = _get_cached_encoding(enc_cache, "hr", model, hr, use_amp)
                    zc, _, _, ms1, _, _ = model.reparameterize(z_params1)
                    z_params2 = _get_cached_encoding(enc_cache, "hr_down_up", model, hr_down_up, use_amp)
                    _, zs, _, ms2, _, _ = model.reparameterize(z_params2)
                    l1, l2 = _decode_compute_losses(model, zc, zs, hr, use_amp)
                    align_loss = F.mse_loss(ms2, ms1)
                    align_tensor = float(style_align_weight) * align_loss
                    sa_val = float(align_loss.detach().cpu().item())
                    ca_val = 0.0
                    del ms1, ms2, align_loss

                cross_align_loss = (
                    float(l1_weight) * l1
                    + float(l2_weight) * l2
                    + align_tensor
                )
                log_cross_l1 = float(l1.detach().cpu().item())
                log_cross_l2 = float(l2.detach().cpu().item())

                if not bool(torch.isfinite(cross_align_loss).all().item()):
                    skipped_nonfinite_loss += 1
                    print(f"[train] Skip non-finite cross_align_loss batch, loss={float(cross_align_loss.mean().cpu().item())}")
                    if enable_cache:
                        del enc_cache
                    optimizer.zero_grad(set_to_none=True)
                    skip_batch = True
                    break

                cross_align_backward = cross_align_loss * microbatch_weight
                cross_align_backward.backward(retain_graph=enable_cache)
                del cross_align_backward, cross_align_loss, zc, zs, l1, l2, align_tensor

                # ---------- Content variance (anti-collapse) ----------
                cv_val = 0.0
                cv_tensor = None
                z_params_lr = _get_cached_encoding(enc_cache, "lr_up", model, lr_up, use_amp)
                lr_mu_c = model.reparameterize(z_params_lr)[2]
                z_params_du = _get_cached_encoding(enc_cache, "hr_down_up", model, hr_down_up, use_amp)
                du_mu_c = model.reparameterize(z_params_du)[2]
                cv_loss = (
                    content_variance_loss(lr_mu_c, target_std=content_variance_target_std)
                    + content_variance_loss(du_mu_c, target_std=content_variance_target_std)
                ) * 0.5
                cv_tensor = float(content_variance_weight) * cv_loss
                cv_val = float(cv_loss.detach().cpu().item())
                del lr_mu_c, du_mu_c, cv_loss

                if cv_tensor is not None:
                    if not bool(torch.isfinite(cv_tensor).all().item()):
                        skipped_nonfinite_loss += 1
                        print("[train] Skip non-finite content_variance batch")
                        if enable_cache:
                            del enc_cache
                        optimizer.zero_grad(set_to_none=True)
                        skip_batch = True
                        break
                    cv_backward_loss = cv_tensor * microbatch_weight
                    cv_backward_loss.backward()
                    del cv_backward_loss, cv_tensor

                microbatch_l1 = log_self_l1 + log_cross_l1
                microbatch_l2 = log_self_l2 + log_cross_l2
                microbatch_kl = kl_val

                if enable_cache:
                    del enc_cache

                microbatch_total_loss_val = (
                    float(l1_weight) * microbatch_l1
                    + float(l2_weight) * microbatch_l2
                    + microbatch_kl
                    + float(content_align_weight) * ca_val
                    + float(style_align_weight) * sa_val
                    + float(content_variance_weight) * cv_val
                )

                batch_total_loss_sum += microbatch_total_loss_val * microbatch_size
                batch_l1_loss_sum += microbatch_l1 * microbatch_size
                batch_l2_loss_sum += microbatch_l2 * microbatch_size
                batch_kl_loss_sum += microbatch_kl * microbatch_size
                batch_ca_loss_sum += ca_val * microbatch_size
                batch_sa_loss_sum += sa_val * microbatch_size
                batch_cv_loss_sum += cv_val * microbatch_size

            if skip_batch:
                continue

            # ---------- Gradient clipping & Optimizer step ----------
            if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(grad_clip_max_norm)
                )
                grad_norm_value = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                if not np.isfinite(grad_norm_value):
                    skipped_nonfinite_grad += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

            optimizer.step()
            updated_iterations += 1
            hit_max_iterations = (
                max_iterations > 0 and start_global_iteration + updated_iterations >= max_iterations
            )

            if ema_model is not None:
                update_ema_model(ema_model, model, decay=ema_decay)

            # ---------- Accumulate logging ----------
            batch_loss = batch_total_loss_sum / float(batch_size)
            batch_l1 = batch_l1_loss_sum / float(batch_size)
            batch_l2 = batch_l2_loss_sum / float(batch_size)
            batch_kl = batch_kl_loss_sum / float(batch_size)
            ca_val = batch_ca_loss_sum / float(batch_size)
            sa_val = batch_sa_loss_sum / float(batch_size)
            cv_val = batch_cv_loss_sum / float(batch_size)

            total_loss_sum += batch_total_loss_sum
            l1_loss_sum += batch_l1_loss_sum
            l2_loss_sum += batch_l2_loss_sum
            kl_loss_sum += batch_kl_loss_sum
            content_align_loss_sum += batch_ca_loss_sum
            style_align_loss_sum += batch_sa_loss_sum
            content_var_loss_sum += batch_cv_loss_sum
            num_samples += batch_size

            pbar.set_postfix(
                ordered_dict={
                    "Loss": f"{batch_loss:>4.5f}",
                    "L1": f"{batch_l1:>4.3f}",
                    "L2": f"{batch_l2:>4.3f}",
                    "KL": f"{batch_kl:>4.3f}",
                    "CA": f"{ca_val:>4.3f}",
                    "SA": f"{sa_val:>4.3f}",
                    "CV": f"{cv_val:>4.3f}",
                }
            )

            if hit_max_iterations:
                reached_max_iterations = True
                break

    if skipped_nonfinite_loss + skipped_nonfinite_grad > 0:
        print(
            f"[train] Skipped batches summary: "
            f"non_finite_loss={skipped_nonfinite_loss}, non_finite_grad_norm={skipped_nonfinite_grad}"
        )

    denom = max(num_samples, 1)
    return (
        total_loss_sum / denom,
        l1_loss_sum / denom,
        l2_loss_sum / denom,
        kl_loss_sum / denom,
        content_align_loss_sum / denom,
        style_align_loss_sum / denom,
        content_var_loss_sum / denom,
        updated_iterations,
        reached_max_iterations,
    )


@torch.no_grad()
def valid(
    dataloader,
    model,
    content_kl_weight: float = CSUA_LDM_VAE_NOUNC_CONTENT_KL_WEIGHT,
    style_kl_weight: float = CSUA_LDM_VAE_NOUNC_STYLE_KL_WEIGHT,
    l1_weight: float = CSUA_LDM_VAE_NOUNC_L1_WEIGHT,
    l2_weight: float = CSUA_LDM_VAE_NOUNC_L2_WEIGHT,
    reproducible_eval: bool = CSUA_LDM_REPRODUCIBLE_EVAL,
    eval_seed: int | None = None,
    content_align_weight: float = CSUA_LDM_VAE_NOUNC_CONTENT_ALIGN_WEIGHT,
    style_align_weight: float = CSUA_LDM_VAE_NOUNC_STYLE_ALIGN_WEIGHT,
    content_variance_weight: float = CSUA_LDM_VAE_NOUNC_CONTENT_VARIANCE_WEIGHT,
    content_variance_target_std: float = CSUA_LDM_VAE_NOUNC_CONTENT_VARIANCE_TARGET_STD,
    eval_microbatch_size: int = CSUA_LDM_VAE_NOUNC_VALID_MICROBATCH_SIZE,
):
    """Run one evaluation epoch with 3-stage content-style disentangled reconstruction."""
    device = DEVICE
    model.eval()
    use_amp = bool(torch.cuda.is_available())
    eval_microbatch_size = max(int(eval_microbatch_size), 1)

    total_loss_sum = 0.0
    l1_loss_sum = 0.0
    l2_loss_sum = 0.0
    kl_loss_sum = 0.0
    content_align_loss_sum = 0.0
    style_align_loss_sum = 0.0
    content_var_loss_sum = 0.0
    num_samples = 0
    skipped_nonfinite = 0

    rng_state = stash_rng_state() if reproducible_eval else None
    if reproducible_eval and eval_seed is not None:
        set_random_seed(eval_seed)

    try:
        for batch in dataloader:
            _, lr_up_img, _, hr_down_up_img, hr_img, _, _ = batch
            lr_up_img = norm_img(lr_up_img, hr_ch_mean, hr_ch_std).to(device=device, dtype=torch.float32, non_blocking=True)
            hr_down_up_img = norm_img(hr_down_up_img, hr_ch_mean, hr_ch_std).to(device=device, dtype=torch.float32, non_blocking=True)
            hr_img = norm_img(hr_img, hr_ch_mean, hr_ch_std).to(device=device, dtype=torch.float32, non_blocking=True)

            batch_size = int(lr_up_img.shape[0])
            if batch_size <= 0:
                continue

            for mb_start, mb_end in iter_microbatch_slices(batch_size, eval_microbatch_size):
                lr_up = lr_up_img[mb_start:mb_end]
                hr_down_up = hr_down_up_img[mb_start:mb_end]
                hr = hr_img[mb_start:mb_end]
                microbatch_size = int(lr_up.shape[0])
                if microbatch_size <= 0:
                    continue

                lr_up_z_c, lr_up_z_s, lr_up_mu_c, lr_up_mu_s, lr_up_logvar_c, lr_up_logvar_s = _encode_extract_content_style(model, lr_up, use_amp)
                hr_down_up_z_c, hr_down_up_z_s, hr_down_up_mu_c, hr_down_up_mu_s, hr_down_up_logvar_c, hr_down_up_logvar_s = _encode_extract_content_style(model, hr_down_up, use_amp)
                hr_z_c, hr_z_s, hr_mu_c, hr_mu_s, hr_logvar_c, hr_logvar_s = _encode_extract_content_style(model, hr, use_amp)

                g1 = _decode_compute_losses(model, lr_up_z_c, lr_up_z_s, lr_up, use_amp)
                g2 = _decode_compute_losses(model, hr_down_up_z_c, hr_down_up_z_s, hr_down_up, use_amp)
                g3 = _decode_compute_losses(model, hr_z_c, hr_z_s, hr, use_amp)
                g4 = _decode_compute_losses(model, lr_up_z_c, hr_down_up_z_s, hr_down_up, use_amp)
                g5 = _decode_compute_losses(model, hr_z_c, hr_down_up_z_s, hr, use_amp)

                batch_l1 = (g1[0] + g2[0] + g3[0]) / 3.0 + (g4[0] + g5[0]) / 2.0
                batch_l2 = (g1[1] + g2[1] + g3[1]) / 3.0 + (g4[1] + g5[1]) / 2.0

                kl_lr_up_c = kl_loss_gaussian(lr_up_mu_c, lr_up_logvar_c)
                kl_lr_up_s = kl_loss_gaussian(lr_up_mu_s, lr_up_logvar_s)
                kl_lr_up = float(content_kl_weight) * kl_lr_up_c + float(style_kl_weight) * kl_lr_up_s

                kl_hr_down_up_c = kl_loss_gaussian(hr_down_up_mu_c, hr_down_up_logvar_c)
                kl_hr_down_up_s = kl_loss_gaussian(hr_down_up_mu_s, hr_down_up_logvar_s)
                kl_hr_down_up = float(content_kl_weight) * kl_hr_down_up_c + float(style_kl_weight) * kl_hr_down_up_s

                kl_hr_c = kl_loss_gaussian(hr_mu_c, hr_logvar_c)
                kl_hr_s = kl_loss_gaussian(hr_mu_s, hr_logvar_s)
                kl_hr = float(content_kl_weight) * kl_hr_c + float(style_kl_weight) * kl_hr_s

                batch_kl = (kl_lr_up + kl_hr_down_up + kl_hr) / 3.0

                content_align_loss = F.mse_loss(lr_up_mu_c, hr_down_up_mu_c)
                style_align_loss = F.mse_loss(hr_down_up_mu_s, hr_mu_s)

                cv_loss = (
                    content_variance_loss(lr_up_mu_c, target_std=content_variance_target_std)
                    + content_variance_loss(hr_down_up_mu_c, target_std=content_variance_target_std)
                ) * 0.5

                total_loss = (
                    float(l1_weight) * batch_l1
                    + float(l2_weight) * batch_l2
                    + batch_kl
                    + float(content_align_weight) * content_align_loss
                    + float(style_align_weight) * style_align_loss
                    + float(content_variance_weight) * cv_loss
                )

                if not bool(torch.isfinite(total_loss).all().item()):
                    skipped_nonfinite += 1
                    print(f"[valid] Skip non-finite loss batch.")
                    continue

                total_loss_sum += float(total_loss.detach().float().cpu().item()) * microbatch_size
                l1_loss_sum += float(batch_l1.detach().float().cpu().item()) * microbatch_size
                l2_loss_sum += float(batch_l2.detach().float().cpu().item()) * microbatch_size
                kl_loss_sum += float(batch_kl.detach().float().cpu().item()) * microbatch_size
                content_align_loss_sum += float(content_align_loss.detach().float().cpu().item()) * microbatch_size
                style_align_loss_sum += float(style_align_loss.detach().float().cpu().item()) * microbatch_size
                content_var_loss_sum += float(cv_loss.detach().float().cpu().item()) * microbatch_size
                num_samples += microbatch_size

        if skipped_nonfinite > 0:
            print(f"[valid] Skipped batches summary: non_finite_loss={skipped_nonfinite}")

        denom = max(num_samples, 1)
        return (
            total_loss_sum / denom,
            l1_loss_sum / denom,
            l2_loss_sum / denom,
            kl_loss_sum / denom,
            content_align_loss_sum / denom,
            style_align_loss_sum / denom,
            content_var_loss_sum / denom
        )
    finally:
        restore_rng_state(rng_state)


def main(
    dataset_dict: dict = CSUA_LDM_DATASET_DIRS,
    resume_train: bool = False,
    content_kl_weight: float = CSUA_LDM_VAE_NOUNC_CONTENT_KL_WEIGHT,
    style_kl_weight: float = CSUA_LDM_VAE_NOUNC_STYLE_KL_WEIGHT,
    l1_weight: float = CSUA_LDM_VAE_NOUNC_L1_WEIGHT,
    l2_weight: float = CSUA_LDM_VAE_NOUNC_L2_WEIGHT,
    ema_decay: float = CSUA_LDM_EMA_DECAY,
    grad_clip_max_norm: float | None = CSUA_LDM_GRAD_CLIP_MAX_NORM,
    reproducible_eval: bool = CSUA_LDM_REPRODUCIBLE_EVAL,
    save_tif_imgs: bool = CSUA_LDM_SAVE_TIF_IMGS,
    train_microbatch_size: int = CSUA_LDM_VAE_NOUNC_TRAIN_MICROBATCH_SIZE,
    eval_microbatch_size: int = CSUA_LDM_VAE_NOUNC_VALID_MICROBATCH_SIZE,
    draw_microbatch_size: int = CSUA_LDM_VAE_NOUNC_DRAW_MICROBATCH_SIZE
):
    """Entry point for VAE_NoUnc."""
    set_random_seed(RANDOM_SEED)
    ensure_output_dirs()
    writer = SummaryWriter(CSUA_LDM_VAE_NOUNC_LOG_DIR)
    if CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED:
        print(
            f"[Soft-HR Target] ENABLED | kernel={CSUA_LDM_HR_GAUSSIAN_KERNEL}, "
            f"sigma={CSUA_LDM_HR_GAUSSIAN_SIGMA}, down={CSUA_LDM_HR_DOWN_MODE}, up={CSUA_LDM_HR_UP_MODE}"
        )
    else:
        print("[Soft-HR Target] DISABLED")

    # 1) Check checkpoint existence
    VAE_NOUNC_CHECKPOINT_PATH = CSUA_LDM_VAE_NOUNC_SAVEPATH
    vae_model_is_exist = os.path.exists(VAE_NOUNC_CHECKPOINT_PATH)
    if vae_model_is_exist and not resume_train:
        raise RuntimeError("CSUA_LDM_VAE_NOUNC_MODEL already exists, but resume_train=False.")

    z_channels = CSUA_LDM_LATENT_HIDDENCHANNEL

    # 2) Build vae_model / optimizer / scheduler
    vae_model = LatentAutoEncoder(
        img_channels=CSUA_LDM_INPUT_CHANNELS,
        z_channels=z_channels,
        
        content_channels=CSUA_LDM_VAE_NOUNC_CONTENT_CHANNELS,
        style_channels=CSUA_LDM_VAE_NOUNC_STYLE_CHANNELS,
    ).to(DEVICE)
    ema_model = build_ema_model(vae_model)

    optimizer = torch.optim.Adam(
        vae_model.parameters(),
        lr=CSUA_LDM_LEARNING_RATE,
        betas=(0.9, 0.999)
    )
    scheduler = create_scheduler(optimizer, MIN_LEARNING_RATE, SCHEDULER_PATIENCE)

    if resume_train:
        vae_model, optimizer, scheduler, last_epoch, last_loss_dict, ckpt_extra, ema_decay = load_model(
            VAE_NOUNC_CHECKPOINT_PATH, 
            vae_model, 
            optimizer, 
            scheduler, 
            ema_model=ema_model, 
            ema_decay=ema_decay
        )
        if not isinstance(last_loss_dict, dict):
            last_loss_dict = {}
        if not isinstance(ckpt_extra, dict):
            ckpt_extra = {}
        start_epoch = max(int(last_epoch), 0)
        global_iteration = int(ckpt_extra.get("global_iteration", 0))
    else:
        last_epoch = None
        last_loss_dict = None
        ckpt_extra = {}
        global_iteration = 0
        start_epoch = 0
        ema_model.load_state_dict(vae_model.state_dict(), strict=True)


    # 3) Build datasets / dataloaders
    train_split_name = dataset_dict.get("split_name_forTrain", None)
    valid_split_name = dataset_dict.get("split_name_forValid", None)
    draw_split_name = dataset_dict.get("split_name_forDraw", None)

    # Preflight check: HR image sizes must be multiples of 8
    for _split_key, _root_key in (
        ("split_name_forTrain", "rootdir_list_forTrain"),
        ("split_name_forValid", "rootdir_list_forValid"),
        ("split_name_forDraw", "rootdir_list_forDraw"),
    ):
        validate_hr_image_sizes_multiple_of_8(
            root_dir=dataset_dict[_root_key],
            split_name=dataset_dict.get(_split_key, None),
        )

    dataset_common_kwargs = dict(
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
        train_transforms=all_train_transforms,
        test_transforms=all_valid_transforms,
        superres_scale=CSUA_LDM_SUPERRES_SCALE,
        lr_crop_size=CSUA_LDM_LR_CROP_SIZE,
        hr_crop_size=CSUA_LDM_HR_CROP_SIZE,
        input_channels=CSUA_LDM_INPUT_CHANNELS,
        **CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
        hr_degradation_gaussian_kernel=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        hr_degradation_gaussian_sigma=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        hr_degradation_down_mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        hr_degradation_up_mode=CSUA_LDM_HR_DEGRADATION_UP_MODE,
        lr_up_mode=CSUA_LDM_LR_UP_MODE,
        lr_gaussian_blur_enabled=CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        lr_gaussian_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
        soft_hr_gaussian_blur_enabled=CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
        soft_hr_gaussian_kernel=CSUA_LDM_HR_GAUSSIAN_KERNEL,
        soft_hr_gaussian_sigma=CSUA_LDM_HR_GAUSSIAN_SIGMA,
        soft_hr_down_mode=CSUA_LDM_HR_DOWN_MODE,
        soft_hr_up_mode=CSUA_LDM_HR_UP_MODE,
    )

    train_dataset = SuperResolutionDataset(
        root_dir=dataset_dict["rootdir_list_forTrain"],
        is_train=True,
        split_name=train_split_name,
        **dataset_common_kwargs,
    )
    valid_dataset = SuperResolutionDataset(
        root_dir=dataset_dict["rootdir_list_forValid"],
        is_train=False,
        split_name=valid_split_name,
        **dataset_common_kwargs,
    )
    draw_dataset = SuperResolutionDataset(
        root_dir=dataset_dict["rootdir_list_forDraw"],
        is_train=False,
        split_name=draw_split_name,
        **dataset_common_kwargs,
    )
    train_split_label = str(train_split_name) if train_split_name else "train"
    valid_split_label = str(valid_split_name) if valid_split_name else "valid"
    draw_split_label = str(draw_split_name) if draw_split_name else "draw"
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(RANDOM_SEED)
    persistent_workers_flag = bool(NUM_WORKERS > 0 and PERSISTENT_WORKERS_ENABLED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CSUA_LDM_TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=train_loader_generator,
        persistent_workers=persistent_workers_flag,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CSUA_LDM_VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers_flag,
    )
    draw_loader = DataLoader(
        draw_dataset,
        batch_size=CSUA_LDM_DRAW_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers_flag,
    )

    print(
        f"[CSUA_LDM_VAE_NOUNC] Batch sizes => "
        f"train={CSUA_LDM_TRAIN_BATCH_SIZE}, valid={CSUA_LDM_VALID_BATCH_SIZE}, draw={CSUA_LDM_DRAW_BATCH_SIZE}"
    )
    print(
        f"[CSUA_LDM_VAE_NOUNC] Dataset splits => "
        f"train={train_split_label}, valid={valid_split_label}, draw={draw_split_label}"
    )
    print(
        f"[CSUA_LDM_VAE_NOUNC] Microbatch sizes => "
        f"train={train_microbatch_size}, valid={eval_microbatch_size}, draw={draw_microbatch_size}"
    )

    # 4) Prepare draw batch (one fixed batch for visualization)
    draw_batch = next(iter(draw_loader))
    lr_img_db, lr_up_img_db, hr_down_img_db, hr_down_up_img_db, hr_img_db, draw_projs, draw_geos = draw_batch
    lr_up_img_db = norm_img(lr_up_img_db, hr_ch_mean, hr_ch_std)
    hr_down_up_img_db = norm_img(hr_down_up_img_db, hr_ch_mean, hr_ch_std)
    hr_img_db = norm_img(hr_img_db, hr_ch_mean, hr_ch_std)
    draw_batch = (lr_img_db, lr_up_img_db, hr_down_img_db, hr_down_up_img_db, hr_img_db, draw_projs, draw_geos)
    draw_geos = np.array([geo_to_numpy_array(draw_geo) for draw_geo in draw_geos]).T
    draw_stage_norm_imgs = build_stage_cache_from_batch(draw_batch, device=DEVICE)

    hr_mean_gpu = hr_ch_mean.to(DEVICE)
    hr_std_gpu = hr_ch_std.to(DEVICE)

    print(f"reproducible_eval={reproducible_eval}")

    if start_epoch > 0 and last_epoch is not None and last_loss_dict is not None:
        fallback_best_epoch_idx = max(int(last_epoch) - 1, 0)
        best_epoch_num_raw = ckpt_extra.get("best_epoch_num", last_loss_dict.get("best_epoch_num", None))
        if best_epoch_num_raw is not None:
            resume_best_epoch_idx = max(int(best_epoch_num_raw) - 1, 0)
        else:
            resume_best_epoch_idx = fallback_best_epoch_idx
        init_monitor_dict = {
            "min_loss": float(last_loss_dict.get("min_loss", last_loss_dict.get("valid_loss", float("inf")))),
            "best_epoch_idx": resume_best_epoch_idx,
            "patience_counter": int(ckpt_extra.get("patience_counter", last_loss_dict.get("patience_counter", 0))),
            "patience_counter_after_min_lr": int(
                ckpt_extra.get(
                    "patience_counter_after_min_lr",
                    last_loss_dict.get("patience_counter_after_min_lr", 0)
                )
            ),
        }
        training_monitor(
            epoch_idx=fallback_best_epoch_idx,
            min_lr=MIN_LEARNING_RATE,
            current_lr=optimizer.param_groups[0]["lr"],
            current_loss=float("inf"),
            patience_threshold=SCHEDULER_PATIENCE,
            resume_train=True,
            load_dict=init_monitor_dict
        )

    # 5) Training loop
    vae_model.train()
    for epoch_idx in range(start_epoch, TRAIN_MAX_EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        current_lr = optimizer.param_groups[0]["lr"]
        eval_seed = RANDOM_SEED if reproducible_eval else None

        (
            train_loss, train_l1_loss, train_l2_loss, train_kl_loss,
            train_ca_loss, train_sa_loss, train_cv_loss,
            train_updated_iterations,
            train_reached_max_iterations
        ) = train(
            train_loader,
            vae_model,
            optimizer,
            ema_model=ema_model,
            ema_decay=ema_decay,
            content_kl_weight=content_kl_weight,
            style_kl_weight=style_kl_weight,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            grad_clip_max_norm=grad_clip_max_norm,
            train_microbatch_size=train_microbatch_size,
            start_global_iteration=global_iteration,
            max_iterations=TRAIN_MAX_ITERATIONS,
        )
        global_iteration += train_updated_iterations
        (
            valid_loss, valid_l1_loss, valid_l2_loss, valid_kl_loss,
            valid_ca_loss, valid_sa_loss, valid_cv_loss
        ) = valid(
            valid_loader,
            ema_model,
            content_kl_weight=content_kl_weight,
            style_kl_weight=style_kl_weight,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            reproducible_eval=reproducible_eval,
            eval_seed=eval_seed,
            eval_microbatch_size=eval_microbatch_size,
        )

        is_break, is_save, pc, pc_min_lr = training_monitor(
            epoch_idx,
            MIN_LEARNING_RATE,
            current_lr,
            valid_loss,
            SCHEDULER_PATIENCE,
            resume_train=False,
            load_dict=None
        )
        scheduler.step(valid_loss)
        if train_reached_max_iterations:
            print(f"[Training] Reached max iterations ({TRAIN_MAX_ITERATIONS}), stopping.")
            is_break = True
        monitor_state = get_training_monitor_state()
        peak_vram_gb = (
            float(torch.cuda.max_memory_allocated()) / float(1024 ** 3)
            if torch.cuda.is_available()
            else float("nan")
        )

        epoch_num = epoch_idx + 1
        print(
            f"    Epoch: {epoch_num}, Iter: {global_iteration}, Pc: {pc}, Pc_min_lr: {pc_min_lr}, Lr: {current_lr:.6f}, "
            f"Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}\n"
            f"    L1_Loss: {valid_l1_loss:.6f}, L2_Loss: {valid_l2_loss:.6f}, "
            f"KL_Loss: {valid_kl_loss:.6f}, "
            f"CA_Loss: {valid_ca_loss:.6f}, SA_Loss: {valid_sa_loss:.6f}, CV_Loss: {valid_cv_loss:.6f}, "
            f"PeakVRAM(GB): {peak_vram_gb:.3f}"
        )

        info_path = CSUA_LDM_VAE_NOUNC_SAVEPATH.replace(".pth", "_TrainingInfo.txt")
        log_info(
            log_path=info_path,
            text=(
                f">>> Epoch: {epoch_num}, Iter: {global_iteration}, Pc: {pc}, Pc_min_lr: {pc_min_lr}, Lr: {current_lr:.6f}, "
                f"Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}\n"
                f"    L1_Loss: {valid_l1_loss:.6f}, L2_Loss: {valid_l2_loss:.6f}, "
                f"KL_Loss: {valid_kl_loss:.6f}, "
                f"CA_Loss: {valid_ca_loss:.6f}, SA_Loss: {valid_sa_loss:.6f}, CV_Loss: {valid_cv_loss:.6f}, "
                f"PeakVRAM(GB): {peak_vram_gb:.3f}"
            )
        )
        log_info(log_path=info_path, text='')

        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/train_loss', {"train_loss": train_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/train_l1_loss', {"train_l1_loss": train_l1_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/train_l2_loss', {"train_l2_loss": train_l2_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/train_kl_loss', {"train_kl_loss": train_kl_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/train_ca_loss', {"train_ca_loss": train_ca_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/train_sa_loss', {"train_sa_loss": train_sa_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/train_cv_loss', {"train_cv_loss": train_cv_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/valid_loss', {"valid_loss": valid_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/valid_l1_loss', {"valid_l1_loss": valid_l1_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/valid_l2_loss', {"valid_l2_loss": valid_l2_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/valid_kl_loss', {"valid_kl_loss": valid_kl_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/valid_ca_loss', {"valid_ca_loss": valid_ca_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/valid_sa_loss', {"valid_sa_loss": valid_sa_loss}, epoch_num)
        writer.add_scalars(f'{CSUA_LDM_VAE_NOUNC_TAG}/valid_cv_loss', {"valid_cv_loss": valid_cv_loss}, epoch_num)

        # Save fixed GT visualization once.
        if epoch_num == 1:
            draw_stage_names = ["lr_up", "hr_down_up", "hr"]
            draw_denorm_chunks = []
            for stage_name in draw_stage_names:
                draw_denorm_chunks.append(
                    denorm_mean(get_stage_img(draw_stage_norm_imgs, stage_name), hr_mean_gpu, hr_std_gpu)[:3]
                )
            draw_imgs = torch.cat(draw_denorm_chunks, dim=0)
            save_rgb_datas(
                draw_imgs[:, CSUA_LDM_VIS_BAND_ORDER, :, :].cpu(),
                nrow=3,
                savepath=os.path.join(CSUA_LDM_VAE_NOUNC_RGB_DIR, f'{CSUA_LDM_VAE_NOUNC_TAG}_True_RGB_0.jpg'),
                is_showminmax=True,
                use_sqrt=True
            )
            if save_tif_imgs:
                save_tif_datas(
                    draw_imgs.cpu(),
                    savepath=os.path.join(CSUA_LDM_VAE_NOUNC_TIF_DIR, f'{CSUA_LDM_VAE_NOUNC_TAG}_True_AllCH_0.tif'),
                    img_sources_num=len(draw_stage_names),
                    geotransforms=draw_geos,
                    projections=draw_projs,
                    is_showminmax=True
                )

        # 6) Save checkpoints
        if is_save or epoch_num % 25 == 0:
            checkpoint_loss_dict = {
                'train_loss': train_loss,
                'train_l1_loss': train_l1_loss,
                'train_l2_loss': train_l2_loss,
                'train_kl_loss': train_kl_loss,
                'train_ca_loss': train_ca_loss,
                'train_sa_loss': train_sa_loss,
                'train_cv_loss': train_cv_loss,
                'valid_loss': valid_loss,
                'valid_l1_loss': valid_l1_loss,
                'valid_l2_loss': valid_l2_loss,
                'valid_kl_loss': valid_kl_loss,
                'valid_ca_loss': valid_ca_loss,
                'valid_sa_loss': valid_sa_loss,
                'valid_cv_loss': valid_cv_loss,
                'min_loss': monitor_state['min_loss'],
            }
            checkpoint_extra_dict = {
                'ema_model_state_dict': ema_model.state_dict(),
                'ema_decay': ema_decay,
                'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                'patience_counter': monitor_state['patience_counter'],
                'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                'global_iteration': global_iteration,
            }
            save_model(
                CSUA_LDM_VAE_NOUNC_SAVEPATH + '.tmp',
                epoch_num,
                vae_model,
                optimizer,
                scheduler,
                checkpoint_loss_dict,
                extra=checkpoint_extra_dict
            )
            os.replace(CSUA_LDM_VAE_NOUNC_SAVEPATH + '.tmp', CSUA_LDM_VAE_NOUNC_SAVEPATH)

        if epoch_num in EXTRA_MODEL_SAVE_EPOCH_NUM:
            checkpoint_loss_dict = {
                'train_loss': train_loss,
                'train_l1_loss': train_l1_loss,
                'train_l2_loss': train_l2_loss,
                'train_kl_loss': train_kl_loss,
                'train_ca_loss': train_ca_loss,
                'train_sa_loss': train_sa_loss,
                'train_cv_loss': train_cv_loss,
                'valid_loss': valid_loss,
                'valid_l1_loss': valid_l1_loss,
                'valid_l2_loss': valid_l2_loss,
                'valid_kl_loss': valid_kl_loss,
                'valid_ca_loss': valid_ca_loss,
                'valid_sa_loss': valid_sa_loss,
                'valid_cv_loss': valid_cv_loss,
                'min_loss': monitor_state['min_loss'],
            }
            checkpoint_extra_dict = {
                'ema_model_state_dict': ema_model.state_dict(),
                'ema_decay': ema_decay,
                'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                'patience_counter': monitor_state['patience_counter'],
                'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                'global_iteration': global_iteration,
            }
            save_model(
                CSUA_LDM_VAE_NOUNC_SAVEPATH + f'_{epoch_num}e.tmp',
                epoch_num,
                vae_model,
                optimizer,
                scheduler,
                checkpoint_loss_dict,
                extra=checkpoint_extra_dict
            )
            os.replace(CSUA_LDM_VAE_NOUNC_SAVEPATH + f'_{epoch_num}e.tmp', CSUA_LDM_VAE_NOUNC_SAVEPATH.replace(".pth", "") + f'_{epoch_num}e.pth')

        # 7) Visualization (every 5 epochs)
        if epoch_num % 5 == 0:
            ema_model.eval()
            metrics_lines = []
            draw_metrics_list: list[dict[str, float]] = []
            recon_imgs_list = []
            stage_outputs = generate_stage_reconstruction_outputs(
                draw_stage_norm_imgs,
                ema_model,
                draw_microbatch_size=draw_microbatch_size,
                device=DEVICE
            )

            for stage_name in ["lr_up", "hr_down_up", "hr"]:
                stage_norm_imgs = get_stage_img(draw_stage_norm_imgs, stage_name)
                recon_pred_mean = stage_outputs[stage_name]["pred_mean"]

                target_denorm = denorm_mean(stage_norm_imgs, hr_mean_gpu, hr_std_gpu)
                recon_denorm = denorm_mean(recon_pred_mean, hr_mean_gpu, hr_std_gpu)
                metrics = evaluate_quality_metrics(recon_denorm, target_denorm)
                metrics_lines.append(format_metric_summary(stage_name, metrics))
                draw_metrics_list.append(metrics)

                recon_imgs_list.append(recon_denorm[:3])

            print(f"    Epoch: {epoch_num}, ===>> Quality Metrics")
            for metric_line in metrics_lines:
                print(f"    {metric_line}")

            log_info(log_path=info_path, text=f">>> Epoch: {epoch_num}, ===>> Quality Metrics")
            for metric_line in metrics_lines:
                log_info(log_path=info_path, text=f"    {metric_line}")
            log_info(log_path=info_path, text='')

            # Log averaged draw metrics to TensorBoard.
            if draw_metrics_list:
                agg_draw_metrics: dict[str, float] = {}
                for key in draw_metrics_list[0].keys():
                    valid_values = [m[key] for m in draw_metrics_list if not (isinstance(m[key], float) and math.isnan(m[key]))]
                    agg_draw_metrics[key] = sum(valid_values) / len(valid_values) if valid_values else float("nan")
                for key, value in agg_draw_metrics.items():
                    writer.add_scalar(f'{CSUA_LDM_VAE_NOUNC_TAG}/draw_{key}', value, epoch_num)

            recon_imgs = torch.cat(recon_imgs_list, dim=0)
            save_rgb_datas(
                recon_imgs[:, CSUA_LDM_VIS_BAND_ORDER, :, :].cpu(),
                nrow=3,
                savepath=os.path.join(CSUA_LDM_VAE_NOUNC_RGB_DIR, f'{CSUA_LDM_VAE_NOUNC_TAG}_RGB_{epoch_num}.jpg'),
                is_showminmax=True,
                use_sqrt=True,
            )

            if save_tif_imgs and epoch_num % 25 == 0:
                save_tif_datas(
                    recon_imgs.cpu(),
                    savepath=os.path.join(CSUA_LDM_VAE_NOUNC_TIF_DIR, f'{CSUA_LDM_VAE_NOUNC_TAG}_Recon_AllCH_{epoch_num}.tif'),
                    img_sources_num=3,
                    geotransforms=draw_geos,
                    projections=draw_projs,
                    is_showminmax=True
                )

            # Release visualization GPU memory
            stage_outputs = None
            recon_imgs = None
            recon_imgs_list = None
            metrics_lines = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()


        if train_reached_max_iterations:
            break

        if is_break:
            break

        time.sleep(5)

    writer.close()


if __name__ == '__main__':
    
    main(resume_train=False)
