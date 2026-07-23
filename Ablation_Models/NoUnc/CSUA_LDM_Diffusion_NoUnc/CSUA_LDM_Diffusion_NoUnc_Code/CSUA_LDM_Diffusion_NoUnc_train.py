import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = VARIANT_ROOT.parents[1]
VAE_ROOT = VARIANT_ROOT / "CSUA_LDM_VAE_NoUnc"
VAE_CODE_ROOT = VAE_ROOT / "CSUA_LDM_VAE_NoUnc_Code"

for candidate in (CURRENT_DIR, PROJECT_ROOT, VARIANT_ROOT, VAE_ROOT, VAE_CODE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import os
import time

import numpy as np
import torch
from osgeo import gdal
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from CSUA_LDM_NoUnc_Config import *  # type: ignore[reportMissingImports]
from CSUA_LDM_Dataset import (
    SuperResolutionDataset,
    validate_hr_image_sizes_multiple_of_8,
)
from CSUA_LDM_Evaluate_Common import (
    format_metric_summary as format_common_metric_summary,
)
from CSUA_LDM_Diffusion_NoUnc_helpers import (
    cosine_beta_scheduler,
    select_dual_branch_source,
)
from CSUA_LDM_Diffusion_NoUnc_model import CSUA_LDM_Diffusion_UNet
from CSUA_LDM_Diffusion_NoUnc_denoise import CSUA_LDM_Diffusion_NoUnc_Denoise
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
    build_ema_model,
    denorm_mean,
    ensure_batched_4d,
    geo_to_numpy_array,
    norm_img,
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
from CSUA_LDM_VAE_NoUnc_helpers import reproject_images_to_content_style_mu
from CSUA_LDM_VAE_NoUnc_model import LatentAutoEncoder

gdal.UseExceptions()

NOUNC_LOSS_METRIC_KEYS = (
    "total",
    "v_l1",
    "v_l2",
)


def _init_metric_sums(metric_keys: tuple[str, ...]) -> dict[str, float]:
    return {key: 0.0 for key in metric_keys}


def define_scheduler(opt, min_learning_rate, patience_threshold):
    scheduler = lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        min_lr=min_learning_rate,
        factor=0.5,
        threshold=0.0,
        patience=patience_threshold
    )
    return scheduler

class DiffusionLossCalculator(nn.Module):
    """Standard DDPM v-pred loss calculator for the NoUnc ablation."""

    def __init__(
        self,
        model,
        betas: torch.Tensor,
        lambda_l1: float = CSUA_LDM_DIFFUSION_NOUNC_V_L1_WEIGHT,
        lambda_l2: float = CSUA_LDM_DIFFUSION_NOUNC_V_L2_WEIGHT,
        latent_scaling_factor: float = LATENT_SCALING_FACTOR,
    ):
        super().__init__()
        self.model = model
        self.T = int(len(betas))
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)

        if latent_scaling_factor is None:
            latent_scaling_factor = LATENT_SCALING_FACTOR
        self.register_buffer(
            "latent_scaling_factor",
            torch.tensor(float(latent_scaling_factor), dtype=torch.float32),
        )

        betas = betas.float()
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def _sample_timesteps(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.T, (int(B),), device=device)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.view(timesteps.shape[0], *([1] * (ref.ndim - 1)))

    def _v_to_x0_eps(self, x_t: torch.Tensor, t: torch.Tensor, pred_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self._extract(self.alphas_cumprod, t, x_t)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-20))
        x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * pred_v
        eps = sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * pred_v
        return x0, eps

    def trainable_parameters(self):
        return list(self.model.parameters())


    def compute_loss_dict(self, cond_content, target_content):
        scale = self.latent_scaling_factor.to(device=cond_content.device, dtype=cond_content.dtype)
        y0 = cond_content * scale
        x0 = target_content * scale
        B = x0.shape[0]

        t = self._sample_timesteps(B, device=x0.device)
        eps = torch.randn_like(x0)

        alpha_bar = self._extract(self.alphas_cumprod, t, x0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-20))
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps
        v_target = sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * x0

        pred_v_mean = self.model(xt, t, y0)
        if pred_v_mean is None:
            raise KeyError("Model must output `pred_v_mean`.")

        diff = pred_v_mean - v_target
        loss_v_l1 = diff.abs().mean() if self.lambda_l1 > 0.0 else pred_v_mean.new_tensor(0.0)
        loss_v_l2 = diff.pow(2).mean() if self.lambda_l2 > 0.0 else pred_v_mean.new_tensor(0.0)
        total = self.lambda_l1 * loss_v_l1 + self.lambda_l2 * loss_v_l2

        x0_hat, _ = self._v_to_x0_eps(xt, t, pred_v_mean)
        return {
            "total": total,
            "loss_v_l1": loss_v_l1,
            "loss_v_l2": loss_v_l2,
            "x0_hat": x0_hat,
            "t": t,
        }

    def forward(self, cond_content, target_content):
        loss_dict = self.compute_loss_dict(cond_content, target_content)
        return (
            loss_dict["total"],
            loss_dict["loss_v_l1"],
            loss_dict["loss_v_l2"],
        )

def _get_module_param_dtype(module: nn.Module) -> torch.dtype:
    for p in module.parameters():
        return p.dtype
    return torch.float32


def _encode_image_to_latent(
    vae_model: LatentAutoEncoder,
    imgs: torch.Tensor,
    use_mu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = ensure_batched_4d(imgs)
    vae_dtype = _get_module_param_dtype(vae_model)
    imgs = norm_img(imgs, hr_ch_mean, hr_ch_std)
    imgs_in = imgs.to(device=imgs.device, dtype=vae_dtype)
    with torch.amp.autocast("cuda", enabled=False):
        z_params = vae_model.encode(imgs_in)
        z_content, z_style, mu_content, mu_style, _, _ = vae_model.reparameterize(z_params)
    if bool(use_mu):
        return mu_content, mu_style
    return z_content, z_style



def _compute_loss_for_batch(
    cond_imgs: torch.Tensor,
    target_imgs: torch.Tensor,
    Latent_model: LatentAutoEncoder,
    loss_calculator: DiffusionLossCalculator,
    use_amp: bool,
    use_reproj_cond: bool = False,
    epoch_idx: int = 0,
):
    device = cond_imgs.device

    with torch.no_grad():
        if use_reproj_cond:
            (
                _reproj_mean,
                cond_z_content,
                cond_z_style,
            ) = reproject_images_to_content_style_mu(
                cond_imgs,
                Latent_model,
                draw_microbatch_size=CSUA_LDM_DIFFUSION_NOUNC_DRAW_MICROBATCH_SIZE,
                device=device,
            )
        else:
            cond_z_content, cond_z_style = _encode_image_to_latent(
                Latent_model,
                cond_imgs,
                use_mu=False,
            )

        target_z_content, target_z_style = _encode_image_to_latent(
            Latent_model, target_imgs, use_mu=False,
        )

    if use_amp and torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_dict = loss_calculator.compute_loss_dict(
                cond_z_content,
                target_z_content,
            )
    else:
        loss_dict = loss_calculator.compute_loss_dict(
            cond_z_content,
            target_z_content,
        )

    loss = loss_dict["total"]
    loss_metrics = {
        "total": loss,
        "v_l1": loss_dict["loss_v_l1"],
        "v_l2": loss_dict["loss_v_l2"],
    }
    return loss_metrics


class OneStepSREvaluator:
    """Evaluate one-step SR for the NoUnc ablation (w/o uncertainty)."""

    def __init__(
        self,
        vae_model: LatentAutoEncoder,
        diffusion_model: CSUA_LDM_Diffusion_UNet,
        hr_mean: torch.Tensor,
        hr_std: torch.Tensor,
        draw_microbatch_size: int = CSUA_LDM_DIFFUSION_NOUNC_DRAW_MICROBATCH_SIZE,
    ):
        self.vae_model = vae_model
        self.diffusion_model = diffusion_model
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.draw_microbatch_size = draw_microbatch_size
        self._output_image_names = ["cond", "hr"]

    @torch.no_grad()
    def evaluate(
        self,
        lr_up_imgs: torch.Tensor,
        sr_targets: dict[str, torch.Tensor],
        use_reproj_condition: bool = False,
        decode_with_target_style: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
        lr_up_imgs, bsz, device = self._prepare_inputs(lr_up_imgs, sr_targets)
        self._init_accumulators()

        for mb_start, mb_end in iter_microbatch_slices(bsz, self.draw_microbatch_size):
            self._process_microbatch(
                mb_start, mb_end, lr_up_imgs, sr_targets, device,
                use_reproj_condition=use_reproj_condition,
                decode_with_target_style=decode_with_target_style,
            )

        sr_step_imgs = self._concatenate_images()
        sr_metrics = self._compute_metrics(sr_step_imgs, sr_targets)

        return sr_step_imgs, sr_metrics

    def _prepare_inputs(
        self,
        lr_up_imgs: torch.Tensor,
        sr_targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, int, torch.device]:
        lr_up_imgs = ensure_batched_4d(lr_up_imgs)
        try:
            device = next(self.vae_model.parameters()).device
        except StopIteration:
            device = lr_up_imgs.device
        lr_up_imgs = lr_up_imgs.to(device=device, non_blocking=True)
        bsz, _, h, w = lr_up_imgs.shape

        self.hr_mean = self.hr_mean.to(device=device, non_blocking=True)
        self.hr_std = self.hr_std.to(device=device, non_blocking=True)

        sr_targets["cond"] = lr_up_imgs
        if "hr" not in sr_targets:
            raise KeyError("Missing SR target: hr.")
        target_imgs = ensure_batched_4d(sr_targets["hr"]).to(device=device, non_blocking=True)
        sr_targets["hr"] = target_imgs

        return lr_up_imgs, bsz, device

    def _init_accumulators(self) -> None:
        self._image_chunks = {name: [] for name in self._output_image_names}

    def _process_microbatch(
        self,
        mb_start: int,
        mb_end: int,
        lr_up_imgs: torch.Tensor,
        sr_targets: dict[str, torch.Tensor],
        device: torch.device,
        use_reproj_condition: bool = False,
        decode_with_target_style: bool = False,
    ) -> None:
        lr_up_chunk = lr_up_imgs[mb_start:mb_end]

        if use_reproj_condition:
            lr_up_recon_mean, cond_z_content, cond_z_style = reproject_images_to_content_style_mu(
                lr_up_chunk,
                self.vae_model,
                draw_microbatch_size=self.draw_microbatch_size,
                device=device,
            )
        else:
            cond_z_content, cond_z_style = _encode_image_to_latent(
                self.vae_model, lr_up_chunk, use_mu=False,
            )
            lr_up_recon_mean = self.vae_model.decode(cond_z_content, cond_z_style)
        self._image_chunks["cond"].append(lr_up_recon_mean)

        batch_slice = slice(mb_start, mb_end)
        prev_target_imgs = sr_targets["hr"][batch_slice]
        prev_target_z_content, prev_target_z_style = _encode_image_to_latent(
            self.vae_model, prev_target_imgs, use_mu=False,
        )
        prev_target_mu_content, _ = _encode_image_to_latent(
            self.vae_model, prev_target_imgs, use_mu=True,
        )

        # For draw visualization, optionally decode predicted content with target (hr) style.
        decode_style_override = prev_target_z_style if decode_with_target_style else None

        sr_prev_imgs, sr_prev_imgs_e = CSUA_LDM_Diffusion_NoUnc_Denoise(
            y0_e_content=cond_z_content,
            y0_e_style=cond_z_style,
            x0_gt_e_content=prev_target_z_content,
            x0_gt_mu_content=prev_target_mu_content,
            init_seed=0,
            steps=CSUA_LDM_DIFFUSION_NOUNC_STEPS,
            vae_model=self.vae_model,
            diffusion_model=self.diffusion_model,
            show_pbar=False,
            x_init=None,
            decode_style_override=decode_style_override,
        )

        self._image_chunks["hr"].append(sr_prev_imgs)

    def _concatenate_images(self) -> dict[str, torch.Tensor]:
        return {
            img_name: torch.cat(chunks, dim=0)
            for img_name, chunks in self._image_chunks.items()
        }

    def _compute_metrics(
        self,
        sr_step_imgs: dict[str, torch.Tensor],
        sr_targets: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, float]]:
        sr_metrics: dict[str, dict[str, float]] = {}

        for img_name in self._output_image_names:
            sr_prev_imgs = sr_step_imgs[img_name]
            prev_target_imgs = sr_targets[img_name]
            sr_prev_denorm = torch.clamp(sr_prev_imgs * self.hr_std + self.hr_mean, min=0.0, max=1.0)
            prev_target_denorm = torch.clamp(prev_target_imgs, min=0.0, max=1.0)

            psnr_val = float(compute_PSNR(sr_prev_denorm, prev_target_denorm).item())
            ssim_val = float(compute_SSIM(sr_prev_denorm, prev_target_denorm).item())

            gmsd_rgb_val, gmsd_nir_val = compute_GMSD(
                sr_prev_denorm,
                prev_target_denorm,
                rgb_indices=CSUA_LDM_VIS_BAND_ORDER,
            )
            gmsd_rgb_val = float(gmsd_rgb_val.item())
            gmsd_nir_val = float(gmsd_nir_val.item()) if gmsd_nir_val is not None else float("nan")

            fsim_rgb_val, fsim_nir_val = compute_FSIM(
                sr_prev_denorm,
                prev_target_denorm,
                rgb_indices=CSUA_LDM_VIS_BAND_ORDER,
            )
            fsim_rgb_val = float(fsim_rgb_val.item())
            fsim_nir_val = float(fsim_nir_val.item()) if fsim_nir_val is not None else float("nan")

            lpips_rgb_val, lpips_nir_val = compute_LPIPS(
                sr_prev_denorm,
                prev_target_denorm,
                rgb_indices=CSUA_LDM_VIS_BAND_ORDER,
            )
            lpips_rgb_val = float(lpips_rgb_val.item())
            lpips_nir_val = float(lpips_nir_val.item()) if lpips_nir_val is not None else float("nan")

            sam_val = float(compute_SAM(sr_prev_denorm, prev_target_denorm).item())

            sr_metrics[img_name] = {
                "psnr": psnr_val,
                "ssim": ssim_val,
                "gmsd_rgb": gmsd_rgb_val,
                "gmsd_nir": gmsd_nir_val,
                "fsim_rgb": fsim_rgb_val,
                "fsim_nir": fsim_nir_val,
                "lpips_rgb": lpips_rgb_val,
                "lpips_nir": lpips_nir_val,
                "sam": sam_val,
            }

        return sr_metrics


def train(data_loader, Latent_model, optimizer, loss_calculator,
         l1_weight: float = CSUA_LDM_DIFFUSION_NOUNC_V_L1_WEIGHT,
         l2_weight: float = CSUA_LDM_DIFFUSION_NOUNC_V_L2_WEIGHT,
         grad_clip_max_norm: float | None = CSUA_LDM_GRAD_CLIP_MAX_NORM,
          epoch_idx: int = 0,
          ema_model=None,
          ema_decay=0.995,
          train_microbatch_size: int = CSUA_LDM_DIFFUSION_NOUNC_TRAIN_MICROBATCH_SIZE,
          start_global_iteration: int = 0,
          max_iterations: int = 0,):
    """
    Train loop for CSUA_LDM diffusion (NoUnc: w/o uncertainty).

    - VAE is frozen (eval mode, no_grad), only Diffusion is trained
    - returns: average loss per sample
    """
    device = DEVICE
    loss_calculator.train()
    loss_calculator.lambda_l1 = float(l1_weight)
    loss_calculator.lambda_l2 = float(l2_weight)
    use_amp = bool(torch.cuda.is_available())
    train_microbatch_size = max(int(train_microbatch_size), 1)

    metric_sums = _init_metric_sums(NOUNC_LOSS_METRIC_KEYS)
    num_samples = 0
    skipped_nonfinite_loss = 0
    skipped_nonfinite_grad = 0
    updated_iterations = 0
    reached_max_iterations = False

    pbar = tqdm(data_loader, dynamic_ncols=False, colour="#ff924a", leave=False, ncols=130)
    for batch in pbar:
        if max_iterations > 0 and start_global_iteration + updated_iterations >= max_iterations:
            reached_max_iterations = True
            break
        core_batch = batch[:7] if isinstance(batch, (list, tuple)) and len(batch) > 7 else batch
        _, lr_up_imgs, _, hr_down_up_imgs, hr_imgs, _, _ = core_batch
        batch_size = int(lr_up_imgs.shape[0])
        
        if batch_size <= 0:
            continue

        source_key = select_dual_branch_source(epoch_idx)
        cond_imgs = hr_down_up_imgs if source_key == "hr_down_up" else lr_up_imgs
        cond_imgs = cond_imgs.to(device=device, non_blocking=True)
        hr_imgs = hr_imgs.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        batch_metric_sums = _init_metric_sums(NOUNC_LOSS_METRIC_KEYS)
        skip_batch = False
        loss_terms = None
        weighted_loss = None
        loss = None

        for mb_start, mb_end in iter_microbatch_slices(batch_size, train_microbatch_size):
            microbatch_size = int(mb_end - mb_start)
            microbatch_weight = float(microbatch_size) / float(batch_size)
            cond_chunk = ensure_batched_4d(cond_imgs[mb_start:mb_end]).to(device, non_blocking=True)
            target_chunk = ensure_batched_4d(hr_imgs[mb_start:mb_end]).to(device, non_blocking=True)

            loss_terms = _compute_loss_for_batch(
                cond_imgs=cond_chunk,
                target_imgs=target_chunk,
                Latent_model=Latent_model,
                loss_calculator=loss_calculator,
                use_amp=bool(use_amp),
                use_reproj_cond=(source_key == "lr_up"),
                epoch_idx=epoch_idx,
            )
            weighted_loss = loss_terms["total"]
            if not bool(torch.isfinite(weighted_loss).all().item()):
                skipped_nonfinite_loss += 1
                print(f"[train] Skip non-finite loss batch, loss={float(weighted_loss.mean().cpu().item())}")
                optimizer.zero_grad(set_to_none=True)
                loss_terms = None
                chunk_step_imgs = None
                weighted_loss = None
                loss = None
                skip_batch = True
                break

            (weighted_loss * microbatch_weight).backward()
            for key in NOUNC_LOSS_METRIC_KEYS:
                batch_metric_sums[key] += float(loss_terms[key].item()) * microbatch_size

        if skip_batch:
            cond_chunk = None
            target_chunk = None
            loss_terms = None
            weighted_loss = None
            loss = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                loss_calculator.trainable_parameters(), max_norm=float(grad_clip_max_norm)
            )
            grad_norm_value = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if not np.isfinite(grad_norm_value):
                skipped_nonfinite_grad += 1
                optimizer.zero_grad(set_to_none=True)
                step_imgs = None
                chunk_step_imgs = None
                loss_terms = None
                weighted_loss = None
                loss = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        optimizer.step()

        updated_iterations += 1
        hit_max_iterations = (
            max_iterations > 0 and start_global_iteration + updated_iterations >= max_iterations
        )

        if ema_model is not None:
            update_ema_model(ema_model, loss_calculator.model, ema_decay)

        batch_metrics = {
            key: batch_metric_sums[key] / float(batch_size)
            for key in NOUNC_LOSS_METRIC_KEYS
        }

        for key in NOUNC_LOSS_METRIC_KEYS:
            metric_sums[key] += batch_metric_sums[key]
        num_samples += batch_size

        pbar.set_postfix(ordered_dict={
            "loss": f"{batch_metrics['total']:.5f}",
            "v_l1": f"{batch_metrics['v_l1']:.3f}",
            "v_l2": f"{batch_metrics['v_l2']:.3f}",
        })

        if hit_max_iterations:
            reached_max_iterations = True
            break

    if skipped_nonfinite_loss + skipped_nonfinite_grad > 0:
        print(
            f"[train] Skipped batches summary: "
            f"non_finite_loss={skipped_nonfinite_loss}, non_finite_grad_norm={skipped_nonfinite_grad}"
        )

    denom = max(num_samples, 1)
    avg_metrics = {
        key: metric_sums[key] / denom
        for key in NOUNC_LOSS_METRIC_KEYS
    }
    return avg_metrics, updated_iterations, reached_max_iterations



@torch.no_grad()
def valid(data_loader, Latent_model, loss_calculator,
         l1_weight: float = CSUA_LDM_DIFFUSION_NOUNC_V_L1_WEIGHT,
         l2_weight: float = CSUA_LDM_DIFFUSION_NOUNC_V_L2_WEIGHT,
         reproducible_eval: bool = CSUA_LDM_REPRODUCIBLE_EVAL,
         eval_seed: int | None = None,
         epoch_idx: int = 0,
         eval_microbatch_size: int = CSUA_LDM_DIFFUSION_NOUNC_VALID_MICROBATCH_SIZE):
    
    device = DEVICE
    loss_calculator.eval()
    loss_calculator.lambda_l1 = float(l1_weight)
    loss_calculator.lambda_l2 = float(l2_weight)
    use_amp = torch.cuda.is_available()
    eval_microbatch_size = max(int(eval_microbatch_size), 1)

    metric_sums = _init_metric_sums(NOUNC_LOSS_METRIC_KEYS)
    num_samples = 0
    skipped_nonfinite = 0

    rng_state = stash_rng_state() if reproducible_eval else None
    if reproducible_eval and eval_seed is not None:
        set_random_seed(eval_seed)

    try:
        for _, batch in enumerate(data_loader):
            core_batch = batch[:7] if isinstance(batch, (list, tuple)) and len(batch) > 7 else batch
            _, lr_up_imgs, _, hr_down_up_imgs, hr_imgs, _, _ = core_batch
            source_key = select_dual_branch_source(epoch_idx)
            cond_imgs = hr_down_up_imgs if source_key == "hr_down_up" else lr_up_imgs
            cond_imgs = cond_imgs.to(device=device, non_blocking=True)
            hr_imgs = hr_imgs.to(device=device, non_blocking=True)

            batch_size = int(lr_up_imgs.shape[0])
            if batch_size <= 0:
                continue

            for mb_start, mb_end in iter_microbatch_slices(batch_size, eval_microbatch_size):
                microbatch_size = int(mb_end - mb_start)
                cond_chunk = ensure_batched_4d(cond_imgs[mb_start:mb_end]).to(device, non_blocking=True)
                target_chunk = ensure_batched_4d(hr_imgs[mb_start:mb_end]).to(device, non_blocking=True)
                loss_terms = _compute_loss_for_batch(
                    cond_imgs=cond_chunk,
                    target_imgs=target_chunk,
                    Latent_model=Latent_model,
                    loss_calculator=loss_calculator,
                    use_amp=bool(use_amp),
                    use_reproj_cond=(source_key == "lr_up"),
                )
                weighted_loss = loss_terms["total"]

                if not bool(torch.isfinite(weighted_loss).all().item()):
                    skipped_nonfinite += 1
                    print(f"[valid] Skip non-finite loss batch.")
                    continue

                for key in NOUNC_LOSS_METRIC_KEYS:
                    metric_sums[key] += float(loss_terms[key].detach().float().cpu().item()) * microbatch_size
                num_samples += microbatch_size

        if skipped_nonfinite > 0:
            print(f"[valid] Skipped batches summary: non_finite_loss={skipped_nonfinite}")

        denom = max(num_samples, 1)
        return {
            key: metric_sums[key] / denom
            for key in NOUNC_LOSS_METRIC_KEYS
        }
    finally:
        restore_rng_state(rng_state)


def ensure_output_dirs():
    os.makedirs(os.path.dirname(CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH), exist_ok=True)
    os.makedirs(CSUA_LDM_DIFFUSION_NOUNC_LOG_DIR, exist_ok=True)
    os.makedirs(CSUA_LDM_DIFFUSION_NOUNC_RGB_DIR, exist_ok=True)
    os.makedirs(CSUA_LDM_DIFFUSION_NOUNC_TIF_DIR, exist_ok=True)


def main(dataset_dict: dict = CSUA_LDM_DATASET_DIRS,
         resume_train: bool = False,
         l1_weight: float = CSUA_LDM_DIFFUSION_NOUNC_V_L1_WEIGHT,
         l2_weight: float = CSUA_LDM_DIFFUSION_NOUNC_V_L2_WEIGHT,
         ema_decay: float = CSUA_LDM_EMA_DECAY,
         grad_clip_max_norm: float | None = CSUA_LDM_GRAD_CLIP_MAX_NORM,
         reproducible_eval: bool = CSUA_LDM_REPRODUCIBLE_EVAL,
         save_tif_imgs: bool = CSUA_LDM_SAVE_TIF_IMGS,
         train_microbatch_size: int = CSUA_LDM_DIFFUSION_NOUNC_TRAIN_MICROBATCH_SIZE,
         eval_microbatch_size: int = CSUA_LDM_DIFFUSION_NOUNC_VALID_MICROBATCH_SIZE,
         draw_microbatch_size: int = CSUA_LDM_DIFFUSION_NOUNC_DRAW_MICROBATCH_SIZE):
    
    set_random_seed(RANDOM_SEED)
    ensure_output_dirs()

    writer = SummaryWriter(CSUA_LDM_DIFFUSION_NOUNC_LOG_DIR)
    if CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED:
        print(
            f"[Soft-HR Target] ENABLED | kernel={CSUA_LDM_HR_GAUSSIAN_KERNEL}, "
            f"sigma={CSUA_LDM_HR_GAUSSIAN_SIGMA}, down={CSUA_LDM_HR_DOWN_MODE}, up={CSUA_LDM_HR_UP_MODE}"
        )
    else:
        print("[Soft-HR Target] DISABLED")

    # ---------- safety ----------
    DIFFUSION_NOUNC_CHECKPOINT_PATH = CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH
    is_exist = os.path.exists(DIFFUSION_NOUNC_CHECKPOINT_PATH)
    if is_exist and not resume_train:
        raise Exception(f"{CSUA_LDM_DIFFUSION_NOUNC_TAG} checkpoint exists and resume_train is False.")

    # ---------- build diffusion model ----------
    csua_diffusion_model = CSUA_LDM_Diffusion_UNet(
        ch=CSUA_LDM_DIFFUSION_NOUNC_UNET_CH,
        out_ch=CSUA_LDM_VAE_NOUNC_CONTENT_CHANNELS,
        ch_mult=CSUA_LDM_DIFFUSION_NOUNC_UNET_CH_MULT,
        attn_resolutions=CSUA_LDM_DIFFUSION_NOUNC_UNET_ATTN_RESOLUTIONS,
        dropout=CSUA_LDM_DIFFUSION_NOUNC_UNET_DROPOUT,
        resamp_with_conv=CSUA_LDM_DIFFUSION_NOUNC_UNET_RESAMP_WITH_CONV,
        in_channels=CSUA_LDM_VAE_NOUNC_CONTENT_CHANNELS,
        resolution=CSUA_LDM_DIFFUSION_NOUNC_UNET_RESOLUTION,
        cond_channels=CSUA_LDM_VAE_NOUNC_CONTENT_CHANNELS,
        cond_scale=CSUA_LDM_DIFFUSION_NOUNC_COND_SCALE,
        window_attn_min_resolution=CSUA_LDM_DIFFUSION_NOUNC_WINDOW_ATTN_MIN_RESOLUTION,
        attn_base_head_dim=CSUA_LDM_DIFFUSION_NOUNC_ATTN_BASE_HEAD_DIM,
        attn_heads_cap=CSUA_LDM_DIFFUSION_NOUNC_ATTN_HEADS_CAP,
    ).to(DEVICE)
    ema_model = build_ema_model(csua_diffusion_model)

    # ---------- Diffusion schedule (betas) ----------
    betas = cosine_beta_scheduler(
        timesteps=CSUA_LDM_DIFFUSION_NOUNC_STEPS,
        device=DEVICE,
        dtype=torch.float32,
    )
    loss_calculator = DiffusionLossCalculator(
        csua_diffusion_model,
        betas,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        csua_diffusion_model.parameters(),
        lr=CSUA_LDM_LEARNING_RATE,
        betas=(0.9, 0.999),
    )
    # ---------- resume ----------
    scheduler = define_scheduler(optimizer, MIN_LEARNING_RATE, SCHEDULER_PATIENCE)
    start_epoch = 0
    last_epoch = -1
    last_loss_dict = None
    ckpt_training_state = {}

    if resume_train:
        csua_diffusion_model, optimizer, scheduler, last_epoch, last_loss_dict, ckpt_extra, loaded_ema_decay = load_model(
            DIFFUSION_NOUNC_CHECKPOINT_PATH,
            csua_diffusion_model,
            optimizer,
            scheduler,
            ema_model=ema_model,
            ema_decay=ema_decay,
        )
        if loaded_ema_decay is not None:
            ema_decay = loaded_ema_decay
        if not isinstance(ckpt_extra, dict):
            ckpt_extra = {}
        if not isinstance(last_loss_dict, dict):
            last_loss_dict = {}
        start_epoch = max(int(last_epoch), 0)
        global_iteration = int(ckpt_extra.get("global_iteration", 0))
    else:
        last_epoch = None
        last_loss_dict = None
        ckpt_extra = {}
        global_iteration = 0
        start_epoch = 0
        ema_model.load_state_dict(csua_diffusion_model.state_dict(), strict=True)

    # ---------- dataset ----------
    if dataset_dict is None:
        raise Exception('DATASET_DICT is None !!!')

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
        generator=train_loader_generator,
        pin_memory=True,
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
        f"[{CSUA_LDM_DIFFUSION_NOUNC_TAG}] VRAM-friendly batch sizes => "
        f"train={CSUA_LDM_TRAIN_BATCH_SIZE}, valid={CSUA_LDM_VALID_BATCH_SIZE}, draw={CSUA_LDM_DRAW_BATCH_SIZE}"
    )
    print(
        f"[{CSUA_LDM_DIFFUSION_NOUNC_TAG}] Internal microbatch sizes => "
        f"train={train_microbatch_size}, valid={eval_microbatch_size}, draw={draw_microbatch_size}"
    )
    print(
        f"[{CSUA_LDM_DIFFUSION_NOUNC_TAG}] Dataset splits => "
        f"train={train_split_label}, valid={valid_split_label}, draw={draw_split_label}"
    )
    print(f"[{CSUA_LDM_DIFFUSION_NOUNC_TAG}] Using fixed one-step diffusion path: cond -> hr")

    # ---------- load latent VAE (once) ----------
    VAE_NOUNC_CHECKPOINT_PATH = CSUA_LDM_VAE_NOUNC_SAVEPATH
    if not os.path.exists(VAE_NOUNC_CHECKPOINT_PATH):
        raise Exception(f"{VAE_NOUNC_CHECKPOINT_PATH} does not exist")

    vae_model = LatentAutoEncoder(
        img_channels=CSUA_LDM_INPUT_CHANNELS,
        z_channels=CSUA_LDM_LATENT_HIDDENCHANNEL,
    ).to(DEVICE)
    vae_model, _, _, _, _, _, _ = load_model(
        VAE_NOUNC_CHECKPOINT_PATH,
        vae_model,
        None,
        None,
        resume_training=False,
        load_ema=True,
    )
    vae_model.eval()
    for p in vae_model.parameters():
        p.requires_grad = False





    # ---------- prepare draw batch (once) ----------
    draw_batch = next(iter(draw_loader))
    draw_core_batch = draw_batch[:7] if isinstance(draw_batch, (list, tuple)) and len(draw_batch) > 7 else draw_batch
    _, draw_lr_up_imgs, _, draw_hr_down_up_imgs, draw_hr_imgs, draw_projs, draw_geos = draw_core_batch
    draw_geos = np.array([geo_to_numpy_array(g) for g in draw_geos]).T

    hr_mean = hr_ch_mean.detach().cpu()
    hr_std = hr_ch_std.detach().cpu()

    draw_sr_names: list[str] = []
    draw_sr_imgs: dict[str, torch.Tensor] = {}
    draw_hr_down_up_sr_imgs: dict[str, torch.Tensor] = {}
    draw_sr_targets = {"cond": draw_lr_up_imgs, "hr": draw_hr_imgs}
    draw_hr_down_up_sr_targets = {"cond": draw_hr_down_up_imgs, "hr": draw_hr_imgs}

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
            load_dict=init_monitor_dict,
        )

    # ---------- training loop ----------
    csua_diffusion_model.train()
    for epoch_idx in range(start_epoch, TRAIN_MAX_EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        current_lr = optimizer.param_groups[0]['lr']
        eval_seed = RANDOM_SEED if reproducible_eval else None
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch_idx)

        # Training phase
        loss_calculator.model = csua_diffusion_model
        csua_diffusion_model.train()
        train_metrics, train_updated_iterations, train_reached_max_iterations = train(
            train_loader,
            vae_model,
            optimizer,
            loss_calculator,

            l1_weight=l1_weight,
            l2_weight=l2_weight,
            grad_clip_max_norm=grad_clip_max_norm,
            epoch_idx=epoch_idx,
            ema_model=ema_model,
            ema_decay=ema_decay,
            train_microbatch_size=train_microbatch_size,
            start_global_iteration=global_iteration,
            max_iterations=TRAIN_MAX_ITERATIONS,
        )
        global_iteration += train_updated_iterations

        # Evaluation/test phase
        csua_diffusion_model.eval()
        loss_calculator.model = ema_model
        try:
            valid_metrics = valid(
                valid_loader,
                vae_model,
                loss_calculator,

                l1_weight=l1_weight,
                l2_weight=l2_weight,
                reproducible_eval=reproducible_eval,
                eval_seed=eval_seed,
                epoch_idx=epoch_idx,
                eval_microbatch_size=eval_microbatch_size,
            )
        finally:
            loss_calculator.model = csua_diffusion_model
        csua_diffusion_model.train()


        epoch_num = epoch_idx + 1
        train_loss = train_metrics["total"]
        valid_loss = valid_metrics["total"]
        train_v_l1_loss = train_metrics["v_l1"]
        valid_v_l1_loss = valid_metrics["v_l1"]
        train_v_l2_loss = train_metrics["v_l2"]
        valid_v_l2_loss = valid_metrics["v_l2"]

        is_break, is_save, pc, pc_min_lr = training_monitor(
            epoch_idx,
            MIN_LEARNING_RATE,
            current_lr,
            float(valid_loss),
            SCHEDULER_PATIENCE,
            resume_train=False,
            load_dict=None,
        )

        if train_reached_max_iterations:
            print(f"[Training] Reached max iterations ({TRAIN_MAX_ITERATIONS}), stopping.")
            is_break = True

        # Update learning-rate scheduler
        scheduler.step(valid_loss)
        monitor_state = get_training_monitor_state()

        print(
            f'    Epoch: {epoch_num}, Iter: {global_iteration}, Pc: {pc}, Pc_min_lr: {pc_min_lr}, Lr: {current_lr:.6f}, '
            f'Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}\n'
            f'    V_L1_loss: {valid_v_l1_loss:.6f}, V_L2_Loss: {valid_v_l2_loss:.6f}, '
            f'Valid_Mode: one_step (cond -> hr)'
        )

        info_path = CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH.replace(".pth", "_TrainingInfo.txt")
        log_info(
            log_path=info_path,
            text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, Pc: {pc}, Pc_min_lr: {pc_min_lr}, Lr: {current_lr:.6f}, '
                 f'Train_Loss: {train_loss:.6f}, Valid_Loss: {valid_loss:.6f}, '
                 f"V_L1_Loss: {valid_v_l1_loss:.6f}, "
                 f"V_L2_Loss: {valid_v_l2_loss:.6f}, "
                 f"Valid_Mode: one_step (cond -> hr)"
        )
        log_info(log_path=info_path, text='')

        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/train_loss', train_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/valid_loss',  valid_loss,  epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/train_l1', train_v_l1_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/valid_l1', valid_v_l1_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/train_l2', train_v_l2_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/valid_l2', valid_v_l2_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/train_v_l1', train_v_l1_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/valid_v_l1', valid_v_l1_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/train_v_l2', train_v_l2_loss, epoch_num)
        writer.add_scalar(f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}/valid_v_l2', valid_v_l2_loss, epoch_num)

        draw_lr_up_denorm_imgs = draw_lr_up_imgs
        draw_hr_down_up_denorm_imgs = draw_hr_down_up_imgs
        draw_sr_denorm_imgs = {
            img_name: sr_imgs
            for img_name, sr_imgs in draw_sr_imgs.items()
        }
        draw_hr_down_up_sr_denorm_imgs = {
            img_name: sr_imgs
            for img_name, sr_imgs in draw_hr_down_up_sr_imgs.items()
        }
        draw_hr_denorm_imgs = draw_hr_imgs

        if epoch_num == 1:
            true_gt_imgs = torch.cat(
                [draw_lr_up_denorm_imgs[:3]]
                + [draw_sr_denorm_imgs[img_name][:3] for img_name in draw_sr_names]
                + [draw_hr_denorm_imgs[:3]],
                dim=0,
            )
            true_gt_rgb_layout_imgs = torch.cat(
                [
                    torch.cat(
                        [
                            left_stage_imgs[:3],
                            right_stage_imgs[:3],
                        ],
                        dim=0,
                    )
                    for left_stage_imgs, right_stage_imgs in zip(
                        [draw_lr_up_denorm_imgs] + [draw_sr_denorm_imgs[img_name] for img_name in draw_sr_names] + [draw_hr_denorm_imgs],
                        [draw_hr_down_up_denorm_imgs] + [draw_hr_down_up_sr_denorm_imgs[img_name] for img_name in draw_sr_names] + [draw_hr_denorm_imgs],
                    )
                ],
                dim=0,
            )
            save_rgb_datas(true_gt_rgb_layout_imgs[:, CSUA_LDM_VIS_BAND_ORDER, :, :].cpu(), 
                           nrow=6, 
                           savepath=os.path.join(CSUA_LDM_DIFFUSION_NOUNC_RGB_DIR, f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}_True_RGB_0.jpg'),
                           use_sqrt=True)
            if save_tif_imgs:
                save_tif_datas(true_gt_imgs.cpu(), 
                            savepath=os.path.join(CSUA_LDM_DIFFUSION_NOUNC_TIF_DIR, f'{CSUA_LDM_DIFFUSION_NOUNC_TAG}_True_TIF_0.tif'),
                            img_sources_num=len(draw_sr_names) + 2,
                            geotransforms=draw_geos,
                            projections=draw_projs,
                            is_showminmax=True)
        # save model
        if is_save or epoch_num % 25 == 0:
            loss_dict = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_v_l1_loss': train_v_l1_loss,
                'valid_v_l1_loss': valid_v_l1_loss,
                'train_v_l2_loss': train_v_l2_loss,
                'valid_v_l2_loss': valid_v_l2_loss,
                'min_loss': monitor_state['min_loss'],
            }
            extra_dict = {
                'ema_model_state_dict': ema_model.state_dict(),
                'ema_decay': ema_decay,
                'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                'patience_counter': monitor_state['patience_counter'],
                'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                'global_iteration': global_iteration,
            }
            save_model(
                CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH + '.tmp',
                epoch_num,
                csua_diffusion_model,
                optimizer,
                scheduler,
                loss_dict,
                extra=extra_dict,
            )
            os.replace(CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH + '.tmp', CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH)

        if epoch_num in EXTRA_MODEL_SAVE_EPOCH_NUM:
            loss_dict = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_v_l1_loss': train_v_l1_loss,
                'valid_v_l1_loss': valid_v_l1_loss,
                'train_v_l2_loss': train_v_l2_loss,
                'valid_v_l2_loss': valid_v_l2_loss,
                'min_loss': monitor_state['min_loss'],
            }
            extra_dict = {
                'ema_model_state_dict': ema_model.state_dict(),
                'ema_decay': ema_decay,
                'best_epoch_num': int(monitor_state['best_epoch_idx']) + 1,
                'patience_counter': monitor_state['patience_counter'],
                'patience_counter_after_min_lr': monitor_state['patience_counter_after_min_lr'],
                'global_iteration': global_iteration,
            }
            save_model(
                CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH + f'_{epoch_num}e.tmp',
                epoch_num,
                csua_diffusion_model,
                optimizer,
                scheduler,
                loss_dict,
                extra=extra_dict,
            )
            os.replace(
                CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH + f'_{epoch_num}e.tmp', 
                CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH.replace('.pth', '') + f'_{epoch_num}e.pth'
            )

        # visualization (every 5 epochs)
        if epoch_num % 5 == 0:
            evaluator = OneStepSREvaluator(
                vae_model=vae_model,
                diffusion_model=ema_model,
                hr_mean=hr_mean,
                hr_std=hr_std,
                draw_microbatch_size=draw_microbatch_size,
            )

            def _run_draw_visualization_for_mode(decode_with_target_style, mode_name):
                use_mode_suffix = mode_name != "cond_style"
                mode_file_suffix = f"_{mode_name}" if use_mode_suffix else ""
                mode_log_suffix = f"[{mode_name}]" if use_mode_suffix else ""
                mode_scalar_infix = f"_{mode_name}" if use_mode_suffix else ""
                mode_tag = CSUA_LDM_DIFFUSION_NOUNC_TAG

                sr_step_imgs, step_metric_vals = evaluator.evaluate(
                    lr_up_imgs=draw_lr_up_imgs,
                    sr_targets=draw_sr_targets,
                    use_reproj_condition=True,
                    decode_with_target_style=decode_with_target_style,
                )
                hr_down_up_sr_step_imgs, hr_down_up_step_metric_vals = evaluator.evaluate(
                    lr_up_imgs=draw_hr_down_up_imgs,
                    sr_targets=draw_hr_down_up_sr_targets,
                    use_reproj_condition=False,
                    decode_with_target_style=decode_with_target_style,
                )
                step_order = ["cond", "hr"]
                step_display_order = ["lr_up(reproj-cond)"] + step_order[1:]
                left_chain_labels = step_display_order
                right_chain_labels = ["hr_down_up"] + step_order[1:-1] + ["hr_down_up_sr"]
                sr_psnr_text = ", ".join(
                    [f"{display_name}_PSNR: {step_metric_vals[img_name]['psnr']:.6f}" for img_name, display_name in zip(step_order, step_display_order)]
                )
                sr_ssim_text = ", ".join(
                    [f"{display_name}_SSIM: {step_metric_vals[img_name]['ssim']:.6f}" for img_name, display_name in zip(step_order, step_display_order)]
                )
                sr_gmsd_text = ", ".join(
                    [
                        f"{display_name}_GMSD(RGB/NIR): "
                        f"{format_metric_value(step_metric_vals[img_name]['gmsd_rgb'])}/"
                        f"{format_metric_value(step_metric_vals[img_name]['gmsd_nir'])}"
                        for img_name, display_name in zip(step_order, step_display_order)
                    ]
                )
                sr_fsim_text = ", ".join(
                    [
                        f"{display_name}_FSIM(RGB/NIR): "
                        f"{format_metric_value(step_metric_vals[img_name]['fsim_rgb'])}/"
                        f"{format_metric_value(step_metric_vals[img_name]['fsim_nir'])}"
                        for img_name, display_name in zip(step_order, step_display_order)
                    ]
                )
                sr_sam_text = ", ".join(
                    [f"{display_name}_SAM: {step_metric_vals[img_name]['sam']:.6f}" for img_name, display_name in zip(step_order, step_display_order)]
                )
                sr_lpips_text = ", ".join(
                    [
                        f"{display_name}_LPIPS_RGB: {format_metric_value(step_metric_vals[img_name]['lpips_rgb'])}, {display_name}_LPIPS_NIR: {format_metric_value(step_metric_vals[img_name]['lpips_nir'])}"
                        for img_name, display_name in zip(step_order, step_display_order)
                    ]
                )
                hr_down_up_sr_psnr_text = ", ".join(
                    [f"{display_name}_PSNR: {hr_down_up_step_metric_vals[img_name]['psnr']:.6f}" for img_name, display_name in zip(step_order, right_chain_labels)]
                )
                hr_down_up_sr_ssim_text = ", ".join(
                    [f"{display_name}_SSIM: {hr_down_up_step_metric_vals[img_name]['ssim']:.6f}" for img_name, display_name in zip(step_order, right_chain_labels)]
                )
                hr_down_up_sr_gmsd_text = ", ".join(
                    [
                        f"{display_name}_GMSD(RGB/NIR): "
                        f"{format_metric_value(hr_down_up_step_metric_vals[img_name]['gmsd_rgb'])}/"
                        f"{format_metric_value(hr_down_up_step_metric_vals[img_name]['gmsd_nir'])}"
                        for img_name, display_name in zip(step_order, right_chain_labels)
                    ]
                )
                hr_down_up_sr_fsim_text = ", ".join(
                    [
                        f"{display_name}_FSIM(RGB/NIR): "
                        f"{format_metric_value(hr_down_up_step_metric_vals[img_name]['fsim_rgb'])}/"
                        f"{format_metric_value(hr_down_up_step_metric_vals[img_name]['fsim_nir'])}"
                        for img_name, display_name in zip(step_order, right_chain_labels)
                    ]
                )
                hr_down_up_sr_sam_text = ", ".join(
                    [f"{display_name}_SAM: {hr_down_up_step_metric_vals[img_name]['sam']:.6f}" for img_name, display_name in zip(step_order, right_chain_labels)]
                )
                hr_down_up_sr_lpips_text = ", ".join(
                    [
                        f"{display_name}_LPIPS_RGB: {format_metric_value(hr_down_up_step_metric_vals[img_name]['lpips_rgb'])}, {display_name}_LPIPS_NIR: {format_metric_value(hr_down_up_step_metric_vals[img_name]['lpips_nir'])}"
                        for img_name, display_name in zip(step_order, right_chain_labels)
                    ]
                )
                sr_quality_summary_text = " | ".join(
                    [
                        f"{display_name}[{format_common_metric_summary(step_metric_vals[img_name])}]"
                        for img_name, display_name in zip(step_order, step_display_order)
                    ]
                )
                hr_down_up_sr_quality_summary_text = " | ".join(
                    [
                        f"{display_name}[{format_common_metric_summary(hr_down_up_step_metric_vals[img_name])}]"
                        for img_name, display_name in zip(step_order, right_chain_labels)
                    ]
                )
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_psnr_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_ssim_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_gmsd_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_fsim_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_sam_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_lpips_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} Summary ===>> {sr_quality_summary_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_psnr_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_ssim_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_gmsd_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_fsim_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_sam_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_lpips_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} Summary ===>> {hr_down_up_sr_quality_summary_text}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, Left Path Row Order{mode_log_suffix} ===>> {", ".join(left_chain_labels)}')
                print(f'    Epoch: {epoch_num}, Iter: {global_iteration}, Right Path Row Order{mode_log_suffix} ===>> {", ".join(right_chain_labels)}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_psnr_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_ssim_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_gmsd_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_fsim_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_sam_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} ===>> {sr_lpips_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, ONE_STEP_SR{mode_log_suffix} Summary ===>> {sr_quality_summary_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_psnr_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_ssim_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_gmsd_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_fsim_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_sam_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} ===>> {hr_down_up_sr_lpips_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, HR_DOWN_UP_SR{mode_log_suffix} Summary ===>> {hr_down_up_sr_quality_summary_text}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, Left Path Row Order{mode_log_suffix} ===>> {", ".join(left_chain_labels)}')
                log_info(log_path=info_path, text=f'>>> Epoch: {epoch_num}, Iter: {global_iteration}, Right Path Row Order{mode_log_suffix} ===>> {", ".join(right_chain_labels)}')
                log_info(log_path=info_path, text='')
                metric_key_by_tag = {
                    "psnr": "psnr",
                    "ssim": "ssim",
                    "gmsd_rgb": "gmsd_rgb",
                    "gmsd_nir": "gmsd_nir",
                    "fsim_rgb": "fsim_rgb",
                    "fsim_nir": "fsim_nir",
                    "lpips_rgb": "lpips_rgb",
                    "lpips_nir": "lpips_nir",
                    "sam": "sam",
                }
                for tag_name, metric_key in metric_key_by_tag.items():
                    for img_name in step_order:
                        writer.add_scalar(
                            f'{mode_tag}_one_step_sr{mode_scalar_infix}_{tag_name}/{img_name}',
                            step_metric_vals[img_name][metric_key],
                            epoch_num,
                        )
                    for img_name, display_name in zip(step_order, right_chain_labels):
                        writer.add_scalar(
                            f'{mode_tag}_hr_down_up_sr{mode_scalar_infix}_{tag_name}/{display_name}',
                            hr_down_up_step_metric_vals[img_name][metric_key],
                            epoch_num,
                        )

                sr_step_denorm_imgs = {}
                for img_name in step_order:
                    sr_step_denorm_imgs[img_name] = denorm_mean(
                        sr_step_imgs[img_name], hr_mean, hr_std
                    )
                hr_down_up_sr_step_denorm_imgs = {}
                for img_name in step_order:
                    hr_down_up_sr_step_denorm_imgs[img_name] = denorm_mean(
                        hr_down_up_sr_step_imgs[img_name], hr_mean, hr_std
                    )

                # Build generated results in SR result order.
                sr_fulllevel_result_imgs = torch.cat(
                    [sr_step_denorm_imgs[img_name][:3] for img_name in step_order],
                    dim=0,
                )
                left_visual_denorm_imgs = {"lr_up": draw_lr_up_denorm_imgs}
                left_visual_denorm_imgs.update(
                    {
                        img_name: sr_step_denorm_imgs[img_name]
                        for img_name in step_order
                        if img_name != "lr_up"
                    }
                )
                right_visual_denorm_imgs = {"lr_up": draw_hr_down_up_imgs}
                right_visual_denorm_imgs.update(
                    {
                        img_name: hr_down_up_sr_step_denorm_imgs[img_name]
                        for img_name in step_order
                        if img_name != "lr_up"
                    }
                )
                sr_rgb_layout_imgs = torch.cat(
                    [
                        torch.cat(
                            [
                                left_visual_denorm_imgs[img_name][:3],
                                right_visual_denorm_imgs[img_name][:3],
                            ],
                            dim=0,
                        )
                        for img_name in step_order
                    ],
                    dim=0,
                )

                if epoch_num % 5 == 0:
                    save_rgb_datas(
                        sr_rgb_layout_imgs[:, CSUA_LDM_VIS_BAND_ORDER, :, :].cpu(),
                        nrow=6,
                        savepath=os.path.join(CSUA_LDM_DIFFUSION_NOUNC_RGB_DIR, f'{mode_tag}_RGB_{epoch_num}{mode_file_suffix}.jpg'),
                        is_showminmax=True,
                        use_sqrt=True,
                    )

                if epoch_num % 25 == 0 and save_tif_imgs:
                    save_tif_datas(sr_fulllevel_result_imgs.cpu(), 
                                   savepath=os.path.join(CSUA_LDM_DIFFUSION_NOUNC_TIF_DIR, f'{mode_tag}_TIF_{epoch_num}{mode_file_suffix}.tif'),
                                   img_sources_num=len(step_order),
                                   geotransforms=draw_geos,
                                   projections=draw_projs,
                                   is_showminmax=True)
                    log_info(log_path=info_path, text='')

            # Dispatch according to the configured report mode.
            if CSUA_LDM_DRAW_REPORT_MODE == "cond_style":
                _run_draw_visualization_for_mode(False, "cond_style")
            elif CSUA_LDM_DRAW_REPORT_MODE == "hr_style":
                _run_draw_visualization_for_mode(True, "hr_style")

            # Draw tensors are local to _run_draw_visualization_for_mode(...);
            # once the helper returns, their references are already released.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if train_reached_max_iterations:
            break

        if is_break:
            break

        time.sleep(5)

    writer.close()


if __name__=='__main__':

    main(resume_train=True)
