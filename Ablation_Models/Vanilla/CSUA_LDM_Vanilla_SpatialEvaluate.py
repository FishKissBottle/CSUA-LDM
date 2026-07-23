import sys
from pathlib import Path

import torch


VARIANT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VARIANT_ROOT.parents[1]
VAE_CODE_ROOT = VARIANT_ROOT / "CSUA_LDM_VAE_Vanilla" / "CSUA_LDM_VAE_Vanilla_Code"
DIFFUSION_CODE_ROOT = VARIANT_ROOT / "CSUA_LDM_Diffusion_Vanilla" / "CSUA_LDM_Diffusion_Vanilla_Code"

for _path in (PROJECT_ROOT, VARIANT_ROOT, VAE_CODE_ROOT, DIFFUSION_CODE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


from CSUA_LDM_Diffusion_Vanilla_denoise import CSUA_LDM_Diffusion_Vanilla_Denoise
from CSUA_LDM_Diffusion_Vanilla_model import CSUA_LDM_Diffusion_Vanilla_UNet
from CSUA_LDM_Evaluate_Common import (
    build_standard_results_root,
    build_data_loader,
    build_dataset,
    ensure_output_dirs,
    evaluate_dataset,
    load_model_with_ema_fallback,
    save_draw_outputs,
    write_evaluation_info,
)
from CSUA_LDM_Utils import ensure_batched_4d, set_random_seed
from CSUA_LDM_VAE_Vanilla_helpers import build_vae_model, reproject_images_to_latent_mu
from CSUA_LDM_VAE_Vanilla_model import LatentAutoEncoderVanilla
from CSUA_LDM_Vanilla_Config import *


CSUA_LDM_EVAL_TAG = "CSUA_LDM_Vanilla"
CSUA_LDM_EVAL_OUTPUT_ROOT = build_standard_results_root("spatial_evaluate", CSUA_LDM_EVAL_TAG)


@torch.no_grad()
def run_diffusion_chain(
    lr_up_img: torch.Tensor,
    vae_model: LatentAutoEncoderVanilla,
    diffusion_model: CSUA_LDM_Diffusion_Vanilla_UNet,
    device: torch.device,
    show_pbar: bool = True,
    init_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Vanilla inference in the complete latent space."""
    lr_up_img = ensure_batched_4d(lr_up_img).to(device)
    _, z = reproject_images_to_latent_mu(
        imgs=lr_up_img,
        model=vae_model,
        device=device,
    )
    pred_img, full_latent = CSUA_LDM_Diffusion_Vanilla_Denoise(
        y0_e=z.to(device),
        x0_gt_e=z.to(device),
        steps=CSUA_LDM_DIFFUSION_VANILLA_STEPS,
        vae_model=vae_model,
        diffusion_model=diffusion_model,
        show_pbar=show_pbar,
        x_init=None,
        init_seed=init_seed,
        print_recon_loss=False,
    )
    return pred_img, full_latent


def build_diffusion_model(device: torch.device) -> tuple[CSUA_LDM_Diffusion_Vanilla_UNet, bool, int]:
    model = CSUA_LDM_Diffusion_Vanilla_UNet(
        ch=CSUA_LDM_DIFFUSION_VANILLA_UNET_CH,
        out_ch=CSUA_LDM_LATENT_HIDDENCHANNEL,
        ch_mult=CSUA_LDM_DIFFUSION_VANILLA_UNET_CH_MULT,
        attn_resolutions=CSUA_LDM_DIFFUSION_VANILLA_UNET_ATTN_RESOLUTIONS,
        dropout=CSUA_LDM_DIFFUSION_VANILLA_UNET_DROPOUT,
        resamp_with_conv=CSUA_LDM_DIFFUSION_VANILLA_UNET_RESAMP_WITH_CONV,
        in_channels=CSUA_LDM_LATENT_HIDDENCHANNEL,
        resolution=CSUA_LDM_DIFFUSION_VANILLA_UNET_RESOLUTION,
        cond_channels=CSUA_LDM_LATENT_HIDDENCHANNEL,
        cond_scale=CSUA_LDM_DIFFUSION_VANILLA_COND_SCALE,
        window_attn_min_resolution=CSUA_LDM_DIFFUSION_VANILLA_WINDOW_ATTN_MIN_RESOLUTION,
        attn_base_head_dim=CSUA_LDM_DIFFUSION_VANILLA_ATTN_BASE_HEAD_DIM,
        attn_heads_cap=CSUA_LDM_DIFFUSION_VANILLA_ATTN_HEADS_CAP,
    ).to(device)
    return load_model_with_ema_fallback(model, CSUA_LDM_DIFFUSION_VANILLA_MODEL_SAVEPATH)


def main():
    device = torch.device(DEVICE)
    set_random_seed(RANDOM_SEED, deterministic=False)

    rgb_dir, tif_dir, report_dir = ensure_output_dirs(CSUA_LDM_EVAL_OUTPUT_ROOT)

    artifact_stem = Path(CSUA_LDM_DIFFUSION_VANILLA_MODEL_SAVEPATH).stem
    rgb_save_path = rgb_dir / f"{artifact_stem}_SpatialEvaluation_RGB_0.jpg"
    tif_save_path = tif_dir / f"{artifact_stem}_SpatialEvaluation_SR_AllCH_0.tif"
    report_path = report_dir / f"{artifact_stem}_SpatialEvaluationInfo.txt"

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
    draw_loader = build_data_loader(
        draw_dataset,
        batch_size=CSUA_LDM_DRAW_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers_enabled=PERSISTENT_WORKERS_ENABLED,
    )

    vae_model, vae_loaded_ema, vae_checkpoint_epoch = build_vae_model(
        Path(CSUA_LDM_VAE_VANILLA_SAVEPATH),
        device,
        load_ema=True,
    )
    diffusion_model, diffusion_loaded_ema, diffusion_checkpoint_epoch = build_diffusion_model(device)

    def predict_fn(lr_up_mb: torch.Tensor, seed_offset: int) -> torch.Tensor:
        pred_img, _ = run_diffusion_chain(
            lr_up_mb,
            vae_model=vae_model,
            diffusion_model=diffusion_model,
            device=device,
            show_pbar=False,
            init_seed=int(RANDOM_SEED + seed_offset),
        )
        return pred_img.detach().float()

    test_loss, test_metrics = evaluate_dataset(
        test_loader,
        predict_fn=predict_fn,
        mean_t=hr_ch_mean,
        std_t=hr_ch_std,
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
    )

    _, draw_lr_up_imgs, _, _, draw_hr_imgs, draw_projs, draw_geos = next(iter(draw_loader))
    save_draw_outputs(
        draw_lr_up_imgs=draw_lr_up_imgs,
        draw_hr_imgs=draw_hr_imgs,
        draw_projs=draw_projs,
        draw_geos=draw_geos,
        predict_fn=predict_fn,
        mean_t=hr_ch_mean,
        std_t=hr_ch_std,
        vis_band_order=CSUA_LDM_VIS_BAND_ORDER,
        rgb_save_path=str(rgb_save_path),
        tif_save_path=str(tif_save_path),
        save_tif_imgs=CSUA_LDM_SAVE_TIF_IMGS,
    )

    write_evaluation_info(
        report_path=report_path,
        tag=CSUA_LDM_EVAL_TAG,
        dataset_dirs=CSUA_LDM_DATASET_DIRS,
        test_dataset_size=len(test_dataset),
        draw_dataset_size=len(draw_dataset),
        test_loss=test_loss,
        test_metrics=test_metrics,
        diffusion_checkpoint_path=CSUA_LDM_DIFFUSION_VANILLA_MODEL_SAVEPATH,
        diffusion_checkpoint_epoch=diffusion_checkpoint_epoch,
        diffusion_loaded_ema=diffusion_loaded_ema,
        diffusion_model=diffusion_model,
        vae_checkpoint_path=CSUA_LDM_VAE_VANILLA_SAVEPATH,
        vae_checkpoint_epoch=vae_checkpoint_epoch,
        vae_loaded_ema=vae_loaded_ema,
        vae_model=vae_model,
        save_tif_imgs=CSUA_LDM_SAVE_TIF_IMGS,
        rgb_dir=str(rgb_dir),
        tif_dir=str(tif_dir),
    )


if __name__ == "__main__":
    main()
