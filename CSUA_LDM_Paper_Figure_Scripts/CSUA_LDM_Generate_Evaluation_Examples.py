"""Export the first CSUA-LDM test batch with the SpatialEvaluate pipeline.

Run this script once per dataset by setting ``CSUA_LDM_DATASET_YAML_OVERRIDE``.
The exported tensors are temporary inputs for
``CSUA_LDM_Package_GitHub_Examples.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_MODEL_ROOT = PROJECT_ROOT / "Main_Model"
VAE_CODE_ROOT = MAIN_MODEL_ROOT / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_Code"
DIFFUSION_CODE_ROOT = MAIN_MODEL_ROOT / "CSUA_LDM_Diffusion" / "CSUA_LDM_Diffusion_Code"

for _path in (PROJECT_ROOT, MAIN_MODEL_ROOT, VAE_CODE_ROOT, DIFFUSION_CODE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


from Main_Model.CSUA_LDM_Config import (  # noqa: E402
    CSUA_LDM_DATASET_DIRS,
    CSUA_LDM_DATASET_NAME,
    CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    CSUA_LDM_HR_CROP_SIZE,
    CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
    CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
    CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
    CSUA_LDM_HR_DEGRADATION_UP_MODE,
    CSUA_LDM_HR_DOWN_MODE,
    CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
    CSUA_LDM_HR_GAUSSIAN_KERNEL,
    CSUA_LDM_HR_GAUSSIAN_SIGMA,
    CSUA_LDM_HR_UP_MODE,
    CSUA_LDM_INPUT_CHANNELS,
    CSUA_LDM_LR_CROP_SIZE,
    CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
    CSUA_LDM_LR_GAUSSIAN_KERNEL,
    CSUA_LDM_LR_GAUSSIAN_SIGMA,
    CSUA_LDM_LR_UP_MODE,
    CSUA_LDM_SUPERRES_SCALE,
    CSUA_LDM_TEST_BATCH_SIZE,
    CSUA_LDM_VAE_SAVEPATH,
    CSUA_LDM_VIS_BAND_ORDER,
    DEVICE,
    NUM_WORKERS,
    PERSISTENT_WORKERS_ENABLED,
    RANDOM_SEED,
    hr_ch_mean,
    hr_ch_std,
)
from CSUA_LDM_Evaluate_Common import (  # noqa: E402
    build_data_loader,
    build_dataset,
    denormalize_imgs,
)
from Main_Model.CSUA_LDM_SpatialEvaluate import (  # noqa: E402
    build_diffusion_model,
    configure_sampling_uncertainty_guidance,
    run_diffusion_chain,
)
from CSUA_LDM_Utils import save_single_tif, set_random_seed  # noqa: E402
from CSUA_LDM_VAE_helpers import build_vae_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the first test batch using the SpatialEvaluate pipeline."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "assets" / "_generated_eval",
    )
    parser.add_argument(
        "--style-source",
        choices=("lr_up", "hr_down_up", "hr"),
        default="lr_up",
    )
    parser.add_argument(
        "--sampling-uncertainty-guidance",
        choices=("yaml", "on", "off"),
        default="yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(DEVICE)
    set_random_seed(RANDOM_SEED, deterministic=False)
    guidance_enabled = configure_sampling_uncertainty_guidance(
        args.sampling_uncertainty_guidance
    )

    dataset = build_dataset(
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
    data_loader = build_data_loader(
        dataset,
        batch_size=CSUA_LDM_TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers_enabled=PERSISTENT_WORKERS_ENABLED,
    )

    vae_model, _, _ = build_vae_model(
        Path(CSUA_LDM_VAE_SAVEPATH),
        device,
        load_ema=True,
    )
    diffusion_model, _, _ = build_diffusion_model(device)

    (
        _lr_imgs,
        lr_up_imgs,
        _hr_down_imgs,
        hr_down_up_imgs,
        hr_imgs,
        projections,
        geotransforms,
    ) = next(iter(data_loader))
    if int(lr_up_imgs.shape[0]) != int(CSUA_LDM_TEST_BATCH_SIZE):
        raise RuntimeError(
            "The first evaluation batch must contain "
            f"{CSUA_LDM_TEST_BATCH_SIZE} samples, got {lr_up_imgs.shape[0]}."
        )

    chain_out = run_diffusion_chain(
        lr_up_imgs,
        vae_model=vae_model,
        diffusion_model=diffusion_model,
        device=device,
        hr_down_up_img=hr_down_up_imgs,
        hr_img=hr_imgs,
        style_source=args.style_source,
        show_pbar=False,
        return_uncertainty=False,
        init_seed=RANDOM_SEED,
    )
    sr_imgs = denormalize_imgs(
        chain_out["images"]["hr"].detach().float(),
        hr_ch_mean.to(device),
        hr_ch_std.to(device),
    ).cpu()
    hr_imgs = hr_imgs.detach().float().cpu()

    if isinstance(geotransforms, (list, tuple)) and len(geotransforms) == 6:
        geotransforms = torch.stack(
            [torch.as_tensor(value) for value in geotransforms],
            dim=1,
        )
    elif (
        isinstance(geotransforms, torch.Tensor)
        and geotransforms.dim() == 2
        and geotransforms.shape[0] == 6
    ):
        geotransforms = geotransforms.transpose(0, 1)

    style_dir_name = f"style-{args.style_source}"
    output_dir = (
        args.output_root
        / CSUA_LDM_DATASET_NAME
        / style_dir_name
        / "TIF_Outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(int(CSUA_LDM_TEST_BATCH_SIZE)):
        save_single_tif(
            tensor=sr_imgs[index],
            save_path=str(output_dir / f"sr_{index:04d}.tif"),
            proj=projections[index],
            geo=geotransforms[index],
        )
        save_single_tif(
            tensor=hr_imgs[index],
            save_path=str(output_dir / f"gt_{index:04d}.tif"),
            proj=projections[index],
            geo=geotransforms[index],
        )

    print(
        f"Exported {CSUA_LDM_TEST_BATCH_SIZE} {CSUA_LDM_DATASET_NAME} samples "
        f"with style={args.style_source}, guidance={guidance_enabled}, "
        f"seed={RANDOM_SEED} to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
