import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from CSUA_LDM_ConfigLoader import load_yaml_config, merge_configs


MAIN_MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MAIN_MODEL_ROOT.parent

RANDOM_SEED = 999

# Microbatch encoding cache switch for VAE training
CSUA_LDM_VAE_ENABLE_MICROBATCH_ENCODING_CACHE = True

# ---------------------------------------------------------------------------
# Dataset, runtime, and model variant configuration
# ---------------------------------------------------------------------------
# Change these paths to select a dataset, training preset, or model variant.
_DATASET_YAML_PATH = str(PROJECT_ROOT / "configs" / "datasets" / "sen2naip.yaml")
_RUNTIME_YAML_PATH = str(PROJECT_ROOT / "configs" / "runtime" / "default_train.yaml")
_VARIANT_YAML_PATH = str(PROJECT_ROOT / "configs" / "variants" / "main.yaml")

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
CSUA_LDM_VAE_TRAIN_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.vae.train)
CSUA_LDM_VAE_VALID_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.vae.valid)
CSUA_LDM_VAE_DRAW_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.vae.draw)
CSUA_LDM_DIFFUSION_TRAIN_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.diffusion.train)
CSUA_LDM_DIFFUSION_VALID_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.diffusion.valid)
CSUA_LDM_DIFFUSION_DRAW_MICROBATCH_SIZE = int(_cfg.microbatch_sizes.diffusion.draw)

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
        f"It must be > 0. Check configs/variants/main.yaml."
    )

# ---------------------------------------------------------------------------
# Model and loss configuration
# ---------------------------------------------------------------------------

# VAE losses
CSUA_LDM_VAE_CONTENT_KL_WEIGHT = float(_cfg.vae.losses.content_kl_weight)
CSUA_LDM_VAE_STYLE_KL_WEIGHT = float(_cfg.vae.losses.style_kl_weight)
CSUA_LDM_VAE_GNLL_WEIGHT = float(_cfg.vae.losses.gnll_weight)
CSUA_LDM_VAE_L1_WEIGHT = float(_cfg.vae.losses.l1_weight)
CSUA_LDM_VAE_L2_WEIGHT = float(_cfg.vae.losses.l2_weight)
CSUA_LDM_VAE_CONFIDENCE_WEIGHT = float(_cfg.vae.losses.confidence_weight)

# VAE disentanglement
CSUA_LDM_VAE_CONTENT_CHANNELS = int(_cfg.vae.disentanglement.content_channels)
CSUA_LDM_VAE_STYLE_CHANNELS = int(_cfg.vae.disentanglement.style_channels)
CSUA_LDM_VAE_CONTENT_ALIGN_WEIGHT = float(_cfg.vae.disentanglement.content_align_weight)
CSUA_LDM_VAE_STYLE_ALIGN_WEIGHT = float(_cfg.vae.disentanglement.style_align_weight)
CSUA_LDM_VAE_CONTENT_VARIANCE_WEIGHT = float(_cfg.vae.disentanglement.content_variance_weight)
CSUA_LDM_VAE_CONTENT_VARIANCE_TARGET_STD = float(_cfg.vae.disentanglement.content_variance_target_std)
CSUA_LDM_VAE_STYLE_DROPOUT_RATE = float(_cfg.vae.disentanglement.style_dropout_rate)

# Diffusion conditioning
CSUA_LDM_DIFFUSION_COND_SCALE = float(_cfg.diffusion.conditioning.cond_scale)

# Diffusion losses
CSUA_LDM_DIFFUSION_GNLL_WEIGHT = float(_cfg.diffusion.losses.gnll_weight)
CSUA_LDM_DIFFUSION_V_L2_WEIGHT = float(_cfg.diffusion.losses.v_l2_weight)
CSUA_LDM_DIFFUSION_V_L1_WEIGHT = float(_cfg.diffusion.losses.v_l1_weight)
CSUA_LDM_DIFFUSION_V_CONFIDENCE_WEIGHT = float(_cfg.diffusion.losses.v_confidence_weight)

# Diffusion uncertainty
CSUA_LDM_DIFFUSION_UNC_SCALE = float(_cfg.diffusion.uncertainty.unc_scale)
CSUA_LDM_DIFFUSION_UNC_COND_LOGVAR_SOFTEN_LAMBDA = float(_cfg.diffusion.uncertainty.unc_cond_logvar_soften_lambda)

# Diffusion sampling
CSUA_LDM_DIFFUSION_STEPS = int(_cfg.diffusion.sampling.steps)
CSUA_LDM_DIFFUSION_SAVE_REVERSE_AND_LATENT_UNCERTAINTY = bool(_cfg.diffusion.sampling.save_reverse_and_latent_uncertainty)

CSUA_LDM_DIFFUSION_SAMPLING_NOISE_SCALE = float(_cfg.diffusion.sampling.sampling_noise_scale)

_diffusion_sampling_unc_guidance_cfg = getattr(_cfg.diffusion, "sampling_uncertainty_guidance", None)
CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_ENABLED = bool(
    getattr(_diffusion_sampling_unc_guidance_cfg, "enabled", True)
) if _diffusion_sampling_unc_guidance_cfg is not None else True
CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_RELIABILITY_ALPHA = float(
    getattr(_diffusion_sampling_unc_guidance_cfg, "reliability_alpha", 2.0)
) if _diffusion_sampling_unc_guidance_cfg is not None else 2.0
CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MIN = float(
    getattr(_diffusion_sampling_unc_guidance_cfg, "noise_gate_min", 0.3)
) if _diffusion_sampling_unc_guidance_cfg is not None else 0.3
CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_NOISE_GATE_MAX = float(
    getattr(_diffusion_sampling_unc_guidance_cfg, "noise_gate_max", 1.0)
) if _diffusion_sampling_unc_guidance_cfg is not None else 1.0
CSUA_LDM_DIFFUSION_INFERENCE_UNCERT_GUIDANCE_SAVE_MAPS = bool(
    getattr(_diffusion_sampling_unc_guidance_cfg, "save_reliability_maps", True)
) if _diffusion_sampling_unc_guidance_cfg is not None else True

# Diffusion dual branch
CSUA_LDM_DIFFUSION_DUAL_BRANCH_ENABLED = bool(_cfg.diffusion.dual_branch.enabled)
CSUA_LDM_DIFFUSION_DUAL_BRANCH_WARMUP_EPOCHS = int(_cfg.diffusion.dual_branch.warmup_epochs)
CSUA_LDM_DIFFUSION_DUAL_BRANCH_RAMP_EPOCHS = int(_cfg.diffusion.dual_branch.ramp_epochs)
CSUA_LDM_DIFFUSION_DUAL_BRANCH_MAX_LR_UP_PROB = float(_cfg.diffusion.dual_branch.max_lr_up_prob)

# Diffusion UNet
CSUA_LDM_DIFFUSION_UNET_CH = int(_cfg.diffusion.unet.ch)
CSUA_LDM_DIFFUSION_UNET_CH_MULT = tuple(_cfg.diffusion.unet.ch_mult)
CSUA_LDM_DIFFUSION_UNET_ATTN_RESOLUTIONS = list(_cfg.diffusion.unet.attn_resolutions)
CSUA_LDM_DIFFUSION_UNET_DROPOUT = float(_cfg.diffusion.unet.dropout)
CSUA_LDM_DIFFUSION_UNET_RESAMP_WITH_CONV = bool(_cfg.diffusion.unet.resamp_with_conv)
CSUA_LDM_DIFFUSION_UNET_RESOLUTION = CSUA_LDM_HR_CROP_SIZE // 8
CSUA_LDM_DIFFUSION_WINDOW_ATTN_MIN_RESOLUTION = int(_cfg.diffusion.unet.window_attn_min_resolution)
CSUA_LDM_DIFFUSION_ATTN_BASE_HEAD_DIM = int(_cfg.diffusion.unet.attn_base_head_dim)
CSUA_LDM_DIFFUSION_ATTN_HEADS_CAP = int(_cfg.diffusion.unet.attn_heads_cap)

# Degradation
CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL = int(_cfg.degradation.hr_gaussian_kernel)      # kernel_size = 2 * ceil(sigma * 3) + 1
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
CSUA_LDM_DRAW_REPORT_MODE = str(getattr(_cfg.diffusion.sampling, "draw_report_mode", "hr_style")).strip().lower()
if CSUA_LDM_DRAW_REPORT_MODE not in {"cond_style", "hr_style"}:
    raise ValueError(
        f"Invalid diffusion.sampling.draw_report_mode='{CSUA_LDM_DRAW_REPORT_MODE}'. "
        f"Supported values are: cond_style, hr_style."
    )

# ---------------------------------------------------------------------------
# Experiment naming: {prefix}_{variant}_{dataset}
# ---------------------------------------------------------------------------

CSUA_LDM_DATASET_NAME = str(getattr(_cfg.dataset, "name", "unknown")).lower()
CSUA_LDM_VARIANT_NAME = "main"


def _build_experiment_id(prefix: str, variant_name: str, dataset_name: str) -> str:
    return f"{prefix}_{variant_name}_{dataset_name}"


CSUA_LDM_VAE_EXPERIMENT_ID = _build_experiment_id("CSUA_LDM_VAE", "main", CSUA_LDM_DATASET_NAME)
CSUA_LDM_DIFFUSION_EXPERIMENT_ID = _build_experiment_id("CSUA_LDM_Diffusion", "main", CSUA_LDM_DATASET_NAME)

# ---------------------------------------------------------------------------
# Project outputs based on experiment_id
# ---------------------------------------------------------------------------

CSUA_LDM_VAE_LOG_DIR = str(
    MAIN_MODEL_ROOT / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_Logs" / CSUA_LDM_VAE_EXPERIMENT_ID
)
CSUA_LDM_VAE_RGB_DIR = str(
    MAIN_MODEL_ROOT / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_RGBs" / CSUA_LDM_VAE_EXPERIMENT_ID
)
CSUA_LDM_VAE_TIF_DIR = str(
    MAIN_MODEL_ROOT / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_TIFs" / CSUA_LDM_VAE_EXPERIMENT_ID
)
CSUA_LDM_VAE_UNC_RGB_DIR = str(
    MAIN_MODEL_ROOT / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_Unc_RGBs" / CSUA_LDM_VAE_EXPERIMENT_ID
)
CSUA_LDM_VAE_UNC_TIF_DIR = str(
    MAIN_MODEL_ROOT / "CSUA_LDM_VAE" / "CSUA_LDM_VAE_Unc_TIFs" / CSUA_LDM_VAE_EXPERIMENT_ID
)
CSUA_LDM_VAE_SAVEPATH = str(
    MAIN_MODEL_ROOT
    / "CSUA_LDM_VAE"
    / "CSUA_LDM_VAE_Models_SaveFolder"
    / f"{CSUA_LDM_VAE_EXPERIMENT_ID}.pth"
)
CSUA_LDM_VAE_TAG = CSUA_LDM_VAE_EXPERIMENT_ID

CSUA_LDM_DIFFUSION_LOG_DIR = str(
    MAIN_MODEL_ROOT
    / "CSUA_LDM_Diffusion"
    / "CSUA_LDM_Diffusion_Logs"
    / CSUA_LDM_DIFFUSION_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_RGB_DIR = str(
    MAIN_MODEL_ROOT
    / "CSUA_LDM_Diffusion"
    / "CSUA_LDM_Diffusion_RGBs"
    / CSUA_LDM_DIFFUSION_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_TIF_DIR = str(
    MAIN_MODEL_ROOT
    / "CSUA_LDM_Diffusion"
    / "CSUA_LDM_Diffusion_TIFs"
    / CSUA_LDM_DIFFUSION_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_UNC_RGB_DIR = str(
    MAIN_MODEL_ROOT
    / "CSUA_LDM_Diffusion"
    / "CSUA_LDM_Diffusion_Unc_RGBs"
    / CSUA_LDM_DIFFUSION_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_UNC_TIF_DIR = str(
    MAIN_MODEL_ROOT
    / "CSUA_LDM_Diffusion"
    / "CSUA_LDM_Diffusion_Unc_TIFs"
    / CSUA_LDM_DIFFUSION_EXPERIMENT_ID
)
CSUA_LDM_DIFFUSION_MODEL_SAVEPATH = str(
    MAIN_MODEL_ROOT
    / "CSUA_LDM_Diffusion"
    / "CSUA_LDM_Diffusion_Models_SaveFolder"
    / f"{CSUA_LDM_DIFFUSION_EXPERIMENT_ID}.pth"
)
CSUA_LDM_DIFFUSION_TAG = CSUA_LDM_DIFFUSION_EXPERIMENT_ID

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
