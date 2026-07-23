import argparse
import sys
from pathlib import Path

import torch


VARIANT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VARIANT_ROOT.parents[1]
VAE_CODE_ROOT = VARIANT_ROOT / "CSUA_LDM_VAE_NoUnc" / "CSUA_LDM_VAE_NoUnc_Code"
DIFFUSION_CODE_ROOT = VARIANT_ROOT / "CSUA_LDM_Diffusion_NoUnc" / "CSUA_LDM_Diffusion_NoUnc_Code"

for _path in (PROJECT_ROOT, VARIANT_ROOT, VAE_CODE_ROOT, DIFFUSION_CODE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


from CSUA_LDM_Diffusion_NoUnc_denoise import CSUA_LDM_Diffusion_NoUnc_Denoise
from CSUA_LDM_Diffusion_NoUnc_model import CSUA_LDM_Diffusion_UNet
from CSUA_LDM_Evaluate_Common import (
    build_standard_results_root,
    build_data_loader,
    build_dataset,
    ensure_output_dirs,
    evaluate_spectral_dataset,
    load_model_with_ema_fallback,
    save_draw_outputs,
    write_spectral_evaluation_info,
)
from CSUA_LDM_NoUnc_Config import *
from CSUA_LDM_Utils import ensure_batched_4d, set_random_seed
from CSUA_LDM_VAE_NoUnc_helpers import (
    build_vae_model,
    decode_content_with_style,
    reproject_images_to_content_style_mu,
)
from CSUA_LDM_VAE_NoUnc_model import LatentAutoEncoder


CSUA_LDM_EVAL_TAG = "CSUA_LDM_NoUnc"
CSUA_LDM_EVAL_OUTPUT_ROOT = build_standard_results_root("spectral_evaluate", CSUA_LDM_EVAL_TAG)
STYLE_SOURCE_CHOICES = ("lr_up", "hr_down_up", "hr")


def _normalize_style_source(style_source: str) -> str:
    style_source = str(style_source).strip().lower()
    if style_source not in STYLE_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported style source: {style_source!r}. "
            f"Expected one of {STYLE_SOURCE_CHOICES}."
        )
    return style_source


def _resolve_style_source_image(
    *,
    style_source: str,
    lr_up_img: torch.Tensor,
    hr_down_up_img: torch.Tensor | None,
    hr_img: torch.Tensor | None,
) -> torch.Tensor:
    style_source = _normalize_style_source(style_source)
    if style_source == "lr_up":
        return lr_up_img
    if style_source == "hr_down_up":
        if hr_down_up_img is None:
            raise ValueError("`hr_down_up_img` is required when style_source='hr_down_up'.")
        return hr_down_up_img
    if hr_img is None:
        raise ValueError("`hr_img` is required when style_source='hr'.")
    return hr_img


@torch.no_grad()
def run_diffusion_chain(
    lr_up_img: torch.Tensor,
    vae_model: LatentAutoEncoder,
    diffusion_model: CSUA_LDM_Diffusion_UNet,
    device: torch.device,
    hr_down_up_img: torch.Tensor | None = None,
    hr_img: torch.Tensor | None = None,
    style_source: str = "lr_up",
    show_pbar: bool = True,
    init_seed: int = 0,
) -> dict:
    """Run NoUnc inference from the LR-up content and the selected style source."""
    lr_up_img = ensure_batched_4d(lr_up_img).to(device)
    style_source = _normalize_style_source(style_source)
    style_img = ensure_batched_4d(
        _resolve_style_source_image(
            style_source=style_source,
            lr_up_img=lr_up_img,
            hr_down_up_img=hr_down_up_img,
            hr_img=hr_img,
        )
    ).to(device)

    input_mean, z_content, z_style = reproject_images_to_content_style_mu(
        imgs=lr_up_img,
        model=vae_model,
        device=device,
    )
    if style_source != "lr_up":
        _style_input_mean, _style_content, z_style = reproject_images_to_content_style_mu(
            imgs=style_img,
            model=vae_model,
            device=device,
        )
        input_mean = decode_content_with_style(
            content=z_content,
            style=z_style,
            model=vae_model,
            device=device,
        )

    pred_mean, full_latent = CSUA_LDM_Diffusion_NoUnc_Denoise(
        y0_e_content=z_content.to(device),
        y0_e_style=z_style.to(device),
        x0_gt_e_content=z_content.to(device),
        steps=CSUA_LDM_DIFFUSION_NOUNC_STEPS,
        vae_model=vae_model,
        diffusion_model=diffusion_model,
        show_pbar=show_pbar,
        x_init=None,
        init_seed=init_seed,
        print_recon_loss=False,
    )
    return {
        "images": {
            "lr_up(reproj-cond)": input_mean,
            "hr": pred_mean,
        },
        "latents": {
            "y0": torch.cat([z_content.to(device), z_style.to(device)], dim=1),
            "hr": full_latent,
        },
        "style_source": style_source,
    }


def build_diffusion_model(device: torch.device) -> tuple[CSUA_LDM_Diffusion_UNet, bool, int]:
    model = CSUA_LDM_Diffusion_UNet(
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
    ).to(device)
    return load_model_with_ema_fallback(model, CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH)


def main():
    parser = argparse.ArgumentParser(description="Spectrally evaluate CSUA_LDM_NoUnc with selectable style source.")
    parser.add_argument(
        "--style-source",
        type=str,
        choices=STYLE_SOURCE_CHOICES,
        default="lr_up",
        help="Choose which image provides the style latent: lr_up, hr_down_up, or hr.",
    )
    args = parser.parse_args()

    device = torch.device(DEVICE)
    set_random_seed(RANDOM_SEED, deterministic=False)
    style_source = str(args.style_source).strip().lower()
    eval_tag = CSUA_LDM_EVAL_TAG
    eval_output_root = CSUA_LDM_EVAL_OUTPUT_ROOT

    rgb_dir, tif_dir, report_dir = ensure_output_dirs(eval_output_root)

    artifact_stem = f"{Path(CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH).stem}_style-{style_source}"
    rgb_save_path = rgb_dir / f"{artifact_stem}_SpectralEvaluation_RGB_0.jpg"
    tif_save_path = tif_dir / f"{artifact_stem}_SpectralEvaluation_SR_AllCH_0.tif"
    report_path = report_dir / f"{artifact_stem}_SpectralEvaluationInfo.txt"

    test_dataset = build_dataset(
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
        lr_gaussian_blur_enabled=CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        lr_gaussian_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
        raw_scaling_kwargs=CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )
    draw_dataset = build_dataset(
        root_dir=CSUA_LDM_DATASET_DIRS["rootdir_list_forDraw"],
        split_name=CSUA_LDM_DATASET_DIRS["split_name_forDraw"],
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
        lr_gaussian_blur_enabled=CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        lr_gaussian_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
        raw_scaling_kwargs=CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )

    test_loader = build_data_loader(
        test_dataset,
        batch_size=CSUA_LDM_TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers_enabled=PERSISTENT_WORKERS_ENABLED,
    )
    test_reference_dataset = build_dataset(
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
        lr_gaussian_blur_enabled=False,
        lr_gaussian_kernel=CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=CSUA_LDM_LR_GAUSSIAN_SIGMA,
        raw_scaling_kwargs=CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )
    test_reference_loader = build_data_loader(
        test_reference_dataset,
        batch_size=CSUA_LDM_TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers_enabled=PERSISTENT_WORKERS_ENABLED,
    )
    draw_loader = build_data_loader(
        draw_dataset,
        batch_size=CSUA_LDM_DRAW_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers_enabled=PERSISTENT_WORKERS_ENABLED,
    )

    vae_model, vae_loaded_ema, vae_checkpoint_epoch = build_vae_model(
        Path(CSUA_LDM_VAE_NOUNC_SAVEPATH),
        device,
        load_ema=True,
    )
    diffusion_model, diffusion_loaded_ema, diffusion_checkpoint_epoch = build_diffusion_model(device)

    def predict_batch_fn(
        lr_up_mb: torch.Tensor,
        hr_down_up_mb: torch.Tensor,
        hr_mb: torch.Tensor,
        seed_offset: int,
    ) -> torch.Tensor:
        chain_out = run_diffusion_chain(
            lr_up_mb,
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            hr_down_up_img=hr_down_up_mb,
            hr_img=hr_mb,
            style_source=style_source,
            show_pbar=False,
            init_seed=int(RANDOM_SEED + seed_offset),
        )
        return chain_out["images"]["hr"].detach().float()

    test_loss, test_metrics = evaluate_spectral_dataset(
        test_loader,
        reference_data_loader=test_reference_loader,
        predict_batch_fn=predict_batch_fn,
        mean_t=hr_ch_mean,
        std_t=hr_ch_std,
        superres_scale=CSUA_LDM_SUPERRES_SCALE,
        gaussian_kernel=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        gaussian_sigma=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        down_mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
    )

    _, draw_lr_up_imgs, _, draw_hr_down_up_imgs, draw_hr_imgs, draw_projs, draw_geos = next(iter(draw_loader))
    save_draw_outputs(
        draw_lr_up_imgs=draw_lr_up_imgs,
        draw_hr_down_up_imgs=draw_hr_down_up_imgs,
        draw_hr_imgs=draw_hr_imgs,
        draw_projs=draw_projs,
        draw_geos=draw_geos,
        predict_batch_fn=predict_batch_fn,
        mean_t=hr_ch_mean,
        std_t=hr_ch_std,
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
        rgb_save_path=str(rgb_save_path),
        tif_save_path=str(tif_save_path),
        save_tif_imgs=CSUA_LDM_SAVE_TIF_IMGS,
    )

    write_spectral_evaluation_info(
        report_path=report_path,
        tag=eval_tag,
        dataset_dirs=CSUA_LDM_DATASET_DIRS,
        test_dataset_size=len(test_dataset),
        draw_dataset_size=len(draw_dataset),
        test_loss=test_loss,
        test_metrics=test_metrics,
        diffusion_checkpoint_path=CSUA_LDM_DIFFUSION_NOUNC_MODEL_SAVEPATH,
        diffusion_checkpoint_epoch=diffusion_checkpoint_epoch,
        diffusion_loaded_ema=diffusion_loaded_ema,
        diffusion_model=diffusion_model,
        vae_checkpoint_path=CSUA_LDM_VAE_NOUNC_SAVEPATH,
        vae_checkpoint_epoch=vae_checkpoint_epoch,
        vae_loaded_ema=vae_loaded_ema,
        vae_model=vae_model,
        save_tif_imgs=CSUA_LDM_SAVE_TIF_IMGS,
        rgb_dir=str(rgb_dir),
        tif_dir=str(tif_dir),
        superres_scale=CSUA_LDM_SUPERRES_SCALE,
        gaussian_kernel=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        gaussian_sigma=CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        down_mode=CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        spectral_reference_mode="D_hr(SR) vs LR_raw",
    )


if __name__ == "__main__":
    main()
