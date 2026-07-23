import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from CSUA_LDM_ConfigLoader import load_yaml_config, merge_configs


VARIANT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VARIANT_ROOT.parents[1]

RANDOM_SEED = 999

# ---------------------------------------------------------------------------
# YAML configuration: dataset + runtime + variant
# ---------------------------------------------------------------------------
_DATASET_YAML_PATH = str(PROJECT_ROOT / "configs" / "datasets" / "sen2naip.yaml")
_RUNTIME_YAML_PATH = str(PROJECT_ROOT / "configs" / "runtime" / "default_train.yaml")
_VARIANT_YAML_PATH = str(PROJECT_ROOT / "configs" / "variants" / "vanilla.yaml")

_cfg = merge_configs(
    load_yaml_config(_DATASET_YAML_PATH),
    load_yaml_config(_RUNTIME_YAML_PATH),
    load_yaml_config(_VARIANT_YAML_PATH),
)

# Dataset geometry
CSUA_LDM_INPUT_CHANNELS = int(_cfg.geometry.input_channels)
CSUA_LDM_SUPERRES_SCALE = int(_cfg.geometry.superres_scale)
CSUA_LDM_LR_CROP_SIZE = int(_cfg.geometry.lr_crop_size)
CSUA_LDM_HR_CROP_SIZE = CSUA_LDM_LR_CROP_SIZE * CSUA_LDM_SUPERRES_SCALE

# Dataset raw-value scaling
_raw_scaling = getattr(_cfg, "raw_scaling", None)
CSUA_LDM_SCALE_LR_TO_UNIT_INTERVAL = bool(
    getattr(_raw_scaling, "scale_lr_to_unit_interval", False)
)
CSUA_LDM_LR_SCALE_DIVISOR = float(getattr(_raw_scaling, "lr_scale_divisor", 1.0))
CSUA_LDM_SCALE_HR_TO_UNIT_INTERVAL = bool(
    getattr(_raw_scaling, "scale_hr_to_unit_interval", False)
)
CSUA_LDM_HR_SCALE_DIVISOR = float(getattr(_raw_scaling, "hr_scale_divisor", 1.0))
if CSUA_LDM_LR_SCALE_DIVISOR <= 0.0 or CSUA_LDM_HR_SCALE_DIVISOR <= 0.0:
    raise ValueError(
        "Dataset raw-scaling divisors must be > 0. "
        f"Got lr={CSUA_LDM_LR_SCALE_DIVISOR}, hr={CSUA_LDM_HR_SCALE_DIVISOR}."
    )
CSUA_LDM_DATASET_RAW_SCALING_KWARGS = {
    "scale_lr_to_unit_interval": CSUA_LDM_SCALE_LR_TO_UNIT_INTERVAL,
    "lr_scale_divisor": CSUA_LDM_LR_SCALE_DIVISOR,
    "scale_hr_to_unit_interval": CSUA_LDM_SCALE_HR_TO_UNIT_INTERVAL,
    "hr_scale_divisor": CSUA_LDM_HR_SCALE_DIVISOR,
}

# Dataset roots
CSUA_LDM_DATASET_DIRS = {
    'rootdir_list_forTrain': str(_cfg.dataset.rootdir_list_forTrain),
    'split_name_forTrain': str(_cfg.dataset.split_name_forTrain),
    'rootdir_list_forValid': str(_cfg.dataset.rootdir_list_forValid),
    'split_name_forValid': str(_cfg.dataset.split_name_forValid),
    'rootdir_list_forTest': str(_cfg.dataset.rootdir_list_forTest),
    'split_name_forTest': str(_cfg.dataset.split_name_forTest),
    'rootdir_list_forDraw': str(_cfg.dataset.rootdir_list_forDraw),
    'split_name_forDraw': str(_cfg.dataset.split_name_forDraw),
}

# Visualization
CSUA_LDM_VIS_BAND_ORDER = list(_cfg.visualization.vis_band_order)

# Normalization (convert YAML lists back to torch tensors)
_hr_mean_list = list(_cfg.normalization.hr_mean)
_hr_std_list = list(_cfg.normalization.hr_std)
hr_ch_mean = torch.tensor(_hr_mean_list).view(1, len(_hr_mean_list), 1, 1)
hr_ch_std = torch.tensor(_hr_std_list).view(1, len(_hr_std_list), 1, 1)

# Batch sizes
CSUA_LDM_TRAIN_BATCH_SIZE = int(_cfg.batch_sizes.train)
CSUA_LDM_VALID_BATCH_SIZE = int(_cfg.batch_sizes.valid)
CSUA_LDM_TEST_BATCH_SIZE = int(_cfg.batch_sizes.test)
CSUA_LDM_DRAW_BATCH_SIZE = int(_cfg.batch_sizes.draw)

# Microbatch sizes
CSUA_LDM_VAE_VANILLA_TRAIN_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.vae.train)
CSUA_LDM_VAE_VANILLA_VALID_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.vae.valid)
CSUA_LDM_VAE_VANILLA_DRAW_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.vae.draw)
CSUA_LDM_DIFFUSION_VANILLA_TRAIN_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.diffusion.train)
CSUA_LDM_DIFFUSION_VANILLA_VALID_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.diffusion.valid)
CSUA_LDM_DIFFUSION_VANILLA_DRAW_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.diffusion.draw)

# Optimization
CSUA_LDM_LEARNING_RATE = float(_cfg.optimization.learning_rate)
MIN_LEARNING_RATE = float(_cfg.optimization.min_learning_rate)
CSUA_LDM_EMA_DECAY = float(_cfg.optimization.ema_decay)
CSUA_LDM_GRAD_CLIP_MAX_NORM = float(_cfg.optimization.grad_clip_max_norm)
SCHEDULER_PATIENCE = int(_cfg.optimization.scheduler_patience)

# Training budget
TRAIN_MAX_EPOCHS = int(_cfg.training_budget.max_epochs)
TRAIN_MAX_ITERATIONS = int(_cfg.training_budget.max_iterations)
EXTRA_MODEL_SAVE_EPOCH_NUM = list(_cfg.training_budget.extra_model_save_epoch_num)

# Hardware / runtime
NUM_WORKERS = int(_cfg.hardware.num_workers)
PERSISTENT_WORKERS_ENABLED = bool(getattr(_cfg.hardware, "persistent_workers_enabled", True))

# Logging
CSUA_LDM_SAVE_TIF_IMGS = bool(_cfg.logging.save_tif_images)

# Latent scaling (dataset-aware: lookup by dataset name in variant yaml)
_dataset_name = str(getattr(_cfg.dataset, "name", "unknown")).lower()
_scaling_map = getattr(_cfg.latent, "scaling_factor_by_dataset", {})
LATENT_SCALING_FACTOR = float(getattr(_scaling_map, _dataset_name, 1.0))
if LATENT_SCALING_FACTOR <= 0.0:
    raise ValueError(
        f"Invalid LATENT_SCALING_FACTOR for dataset '{_dataset_name}': {LATENT_SCALING_FACTOR}. "
        f"It must be > 0. Check configs/variants/vanilla.yaml."
    )

# ---------------------------------------------------------------------------
# Variant-specific configuration (Vanilla: w/o uncertainty and w/o disentangled latent design)
# ---------------------------------------------------------------------------

# VAE losses (Vanilla keeps only reconstruction + KL)
CSUA_LDM_VAE_VANILLA_KL_WEIGHT = float(_cfg.vae.losses.kl_weight)
CSUA_LDM_VAE_VANILLA_L1_WEIGHT = float(_cfg.vae.losses.l1_weight)
CSUA_LDM_VAE_VANILLA_L2_WEIGHT = float(_cfg.vae.losses.l2_weight)

# Diffusion conditioning
CSUA_LDM_DIFFUSION_VANILLA_COND_SCALE = float(_cfg.diffusion.conditioning.cond_scale)

# Diffusion losses
CSUA_LDM_DIFFUSION_VANILLA_V_L2_WEIGHT = float(_cfg.diffusion.losses.v_l2_weight)
CSUA_LDM_DIFFUSION_VANILLA_V_L1_WEIGHT = float(_cfg.diffusion.losses.v_l1_weight)

# Diffusion sampling
CSUA_LDM_DIFFUSION_VANILLA_STEPS = int(_cfg.diffusion.sampling.steps)
CSUA_LDM_DIFFUSION_VANILLA_SAMPLING_NOISE_SCALE = float(_cfg.diffusion.sampling.sampling_noise_scale)

# Diffusion dual branch
CSUA_LDM_DIFFUSION_VANILLA_DUAL_BRANCH_ENABLED = bool(_cfg.diffusion.dual_branch.enabled)
CSUA_LDM_DIFFUSION_VANILLA_DUAL_BRANCH_WARMUP_EPOCHS = int(_cfg.diffusion.dual_branch.warmup_epochs)
CSUA_LDM_DIFFUSION_VANILLA_DUAL_BRANCH_RAMP_EPOCHS = int(_cfg.diffusion.dual_branch.ramp_epochs)
CSUA_LDM_DIFFUSION_VANILLA_DUAL_BRANCH_MAX_LR_UP_PROB = float(_cfg.diffusion.dual_branch.max_lr_up_prob)

# Diffusion UNet
CSUA_LDM_DIFFUSION_VANILLA_UNET_CH = int(_cfg.diffusion.unet.ch)
CSUA_LDM_DIFFUSION_VANILLA_UNET_CH_MULT = tuple(_cfg.diffusion.unet.ch_mult)
CSUA_LDM_DIFFUSION_VANILLA_UNET_ATTN_RESOLUTIONS = list(_cfg.diffusion.unet.attn_resolutions)
CSUA_LDM_DIFFUSION_VANILLA_UNET_DROPOUT = float(_cfg.diffusion.unet.dropout)
CSUA_LDM_DIFFUSION_VANILLA_UNET_RESAMP_WITH_CONV = bool(_cfg.diffusion.unet.resamp_with_conv)
CSUA_LDM_DIFFUSION_VANILLA_UNET_RESOLUTION = CSUA_LDM_HR_CROP_SIZE // 8
CSUA_LDM_DIFFUSION_VANILLA_WINDOW_ATTN_MIN_RESOLUTION = int(_cfg.diffusion.unet.window_attn_min_resolution)
CSUA_LDM_DIFFUSION_VANILLA_ATTN_BASE_HEAD_DIM = int(_cfg.diffusion.unet.attn_base_head_dim)
CSUA_LDM_DIFFUSION_VANILLA_ATTN_HEADS_CAP = int(_cfg.diffusion.unet.attn_heads_cap)

# Degradation
CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL = int(_cfg.degradation.hr_gaussian_kernel)
CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA = float(_cfg.degradation.hr_gaussian_sigma)
CSUA_LDM_HR_DEGRADATION_DOWN_MODE = str(_cfg.degradation.hr_down_mode)
CSUA_LDM_HR_DEGRADATION_UP_MODE = str(_cfg.degradation.hr_up_mode)

CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED = bool(_cfg.degradation.soft_hr_gaussian_blur_enabled)
CSUA_LDM_HR_GAUSSIAN_KERNEL = int(_cfg.degradation.soft_hr_gaussian_kernel)
CSUA_LDM_HR_GAUSSIAN_SIGMA = float(_cfg.degradation.soft_hr_gaussian_sigma)
CSUA_LDM_HR_DOWN_MODE = str(_cfg.degradation.soft_hr_down_mode)
CSUA_LDM_HR_UP_MODE = str(_cfg.degradation.soft_hr_up_mode)

CSUA_LDM_LR_UP_MODE = str(_cfg.degradation.lr_up_mode)
CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED = bool(_cfg.degradation.lr_gaussian_blur_enabled)
CSUA_LDM_LR_GAUSSIAN_KERNEL = int(_cfg.degradation.lr_gaussian_kernel)
CSUA_LDM_LR_GAUSSIAN_SIGMA = float(_cfg.degradation.lr_gaussian_sigma)

# Image settings
CSUA_LDM_LATENT_HIDDENCHANNEL = int(_cfg.image.latent_hiddenchannel)

# Eval settings
CSUA_LDM_REPRODUCIBLE_EVAL = bool(_cfg.eval.reproducible_eval)

# ---------------------------------------------------------------------------
# Experiment naming: {prefix}_{variant}_{dataset}
# ---------------------------------------------------------------------------

CSUA_LDM_DATASET_NAME = str(getattr(_cfg.dataset, "name", "unknown")).lower()
CSUA_LDM_VARIANT_NAME = "vanilla"


def _build_experiment_id(prefix: str, variant_name: str, dataset_name: str) -> str:
    return f"{prefix}_{variant_name}_{dataset_name}"


CSUA_LDM_VAE_VANILLA_EXPERIMENT_ID = _build_experiment_id("CSUA_LDM_VAE", "Vanilla", CSUA_LDM_DATASET_NAME)
CSUA_LDM_DIFFUSION_VANILLA_EXPERIMENT_ID = _build_experiment_id("CSUA_LDM_Diffusion", "Vanilla", CSUA_LDM_DATASET_NAME)

# ---------------------------------------------------------------------------
# Project outputs based on experiment_id
# ---------------------------------------------------------------------------

CSUA_LDM_VAE_VANILLA_LOG_DIR = str(
    VARIANT_ROOT / "CSUA_LDM_VAE_Vanilla" / "CSUA_LDM_VAE_Vanilla_Logs" / CSUA_LDM_VAE_VANILLA_EXPERIMENT_ID
)
CSUA_LDM_VAE_VANILLA_RGB_DIR = str(
    VARIANT_ROOT / "CSUA_LDM_VAE_Vanilla" / "CSUA_LDM_VAE_Vanilla_RGBs" / CSUA_LDM_VAE_VANILLA_EXPERIMENT_ID
)
CSUA_LDM_VAE_VANILLA_TIF_DIR = str(
    VARIANT_ROOT / "CSUA_LDM_VAE_Vanilla" / "CSUA_LDM_VAE_Vanilla_TIFs" / CSUA_LDM_VAE_VANILLA_EXPERIMENT_ID
)
CSUA_LDM_VAE_VANILLA_SAVEPATH = str(
    VARIANT_ROOT
    / "CSUA_LDM_VAE_Vanilla"
    / "CSUA_LDM_VAE_Vanilla_Models_SaveFolder"
    / f"{CSUA_LDM_VAE_VANILLA_EXPERIMENT_ID}.pth"
)
CSUA_LDM_VAE_VANILLA_TAG = CSUA_LDM_VAE_VANILLA_EXPERIMENT_ID

CSUA_LDM_DIFFUSION_VANILLA_LOG_DIR = str(
    VARIANT_ROOT
    / "CSUA_LDM_Diffusion_Vanilla"
    / "CSUA_LDM_Diffusion_Vanilla_Logs"
    / CSUA_LDM_DIFFUSION_VANILLA_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_VANILLA_RGB_DIR = str(
    VARIANT_ROOT
    / "CSUA_LDM_Diffusion_Vanilla"
    / "CSUA_LDM_Diffusion_Vanilla_RGBs"
    / CSUA_LDM_DIFFUSION_VANILLA_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_VANILLA_TIF_DIR = str(
    VARIANT_ROOT
    / "CSUA_LDM_Diffusion_Vanilla"
    / "CSUA_LDM_Diffusion_Vanilla_TIFs"
    / CSUA_LDM_DIFFUSION_VANILLA_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_VANILLA_MODEL_SAVEPATH = str(
    VARIANT_ROOT
    / "CSUA_LDM_Diffusion_Vanilla"
    / "CSUA_LDM_Diffusion_Vanilla_Models_SaveFolder"
    / f"{CSUA_LDM_DIFFUSION_VANILLA_EXPERIMENT_ID}.pth"
)
CSUA_LDM_DIFFUSION_VANILLA_TAG = CSUA_LDM_DIFFUSION_VANILLA_EXPERIMENT_ID

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

all_train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ],
    additional_targets={
        "image0": "image",
        "image1": "image",
    }
)

all_valid_transforms = A.Compose(
    [
        ToTensorV2(),
    ],
    additional_targets={
        "image0": "image",
        "image1": "image",
    }
)
