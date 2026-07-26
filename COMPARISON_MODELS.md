# Comparison Model Reproduction

This document records the official sources, required adaptations,
configurations, and evaluation protocol used for the comparison methods in the
CSUA-LDM experiments. The public repository does not redistribute the adapted
third-party implementations. Obtain each method from its official source and
apply the settings below for reproduction.

## Upstream Sources

| Method | Official source or implementation basis | License |
|---|---|---|
| Bilinear | Implemented locally with PyTorch interpolation | Root MIT License |
| DSen2 | [lanha/DSen2](https://github.com/lanha/DSen2) | GNU GPL v3.0 |
| EDSR | [sanghyun-son/EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) | MIT License |
| RCAN | RCAN implementation integrated in [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch), with method attribution to [yulunzhang/RCAN](https://github.com/yulunzhang/RCAN) | EDSR-PyTorch MIT License for the distributed implementation basis |
| ESRGAN | [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN) | Apache License 2.0 |
| SwinIR | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | Apache License 2.0 |
| ResShift | [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift) | S-Lab License 1.0, non-commercial use only |

Review the license in each official project before downloading, modifying, or
redistributing its code. In particular, ResShift is restricted to
non-commercial use under the S-Lab License 1.0.

## Required Adaptations

For the reported experiments, all learning-based baselines were adapted to the
same paired-data and evaluation protocol used by CSUA-LDM. Reproduction
requires the following changes to the official implementations:

- support for multi-band paired LR and HR remote sensing imagery;
- dataset-specific input geometry and scale factors;
- shared raw-value conversion, normalization, and band ordering;
- microbatch training, validation-based checkpointing, EMA, and gradient
  clipping where implemented;
- spatial evaluation against HR references and spectral evaluation after
  degrading SR outputs back to the LR domain;
- GeoTIFF output and RGB visualization through the common evaluation pipeline.

Model-specific adaptations are summarized below.

| Method | Required adaptation |
|---|---|
| DSen2 | PyTorch residual-refinement implementation with configurable channels, scale, and interpolation-based pre-upsampling |
| EDSR | Multi-band input/output support and dataset-specific upsampling scale |
| RCAN | Multi-band input/output support, dataset-specific scale, and integration with the shared EDSR-style training framework |
| ESRGAN | Multi-band RRDB generator and discriminator, two-stage PSNR pretraining and GAN finetuning, with perceptual supervision restricted to RGB bands |
| SwinIR | Configurable multi-band channels, dataset-specific upsampling, and integration with the paired-data evaluation pipeline |
| ResShift | Independent eight-channel first-stage autoencoder, dataset-aware latent scaling, and a 15-step latent restoration process |

## Shared Training Protocol

The shared settings used in the reported experiments are listed below.

| Setting | Value |
|---|---:|
| Random seed | 999 |
| Maximum epochs | 300 |
| Maximum optimizer updates | 75,000 |
| Default learning rate | 1e-4 |
| Minimum learning rate | 1e-6 |
| ReduceLROnPlateau patience | 3 |
| EMA decay | 0.995 |
| Gradient clipping norm | 3.0 |
| DataLoader workers | 4 |

DSen2, EDSR, RCAN, and SwinIR use Adam optimization with an L1 reconstruction
objective. ResShift uses Adam for its first stage and AdamW for latent
restoration. ESRGAN uses Adam and follows two stages: L1-oriented PSNR
pretraining, followed by perceptual and relativistic adversarial finetuning.

## Model Configurations

| Method | Configuration used in the reported experiments |
|---|---|
| Bilinear | `align_corners=false`; no trainable parameters |
| DSen2 | 6 residual blocks, 128 features, residual scale 0.1, bilinear pre-upsampling |
| EDSR | 16 residual blocks, 64 features, residual scale 1.0 |
| RCAN | 10 residual groups, 20 blocks per group, 64 features, reduction ratio 16 |
| ESRGAN | 16 RRDB blocks, 48 features, growth channels 24 |
| SwinIR | image size 64, patch size 1, window size 8, embedding dimension 60, depths `[6, 6, 6, 6]`, heads `[6, 6, 6, 6]`, direct pixel-shuffle upsampling |
| ResShift | 8 latent channels, base channels 160, channel multipliers `[1, 2, 2, 4]`, 15 diffusion steps, first-stage KL weight 5e-4 |

ESRGAN uses an L1 pixel-loss weight of 1.0, a perceptual-loss weight of 1.0
from VGG19 `conv5_4` on RGB bands, and a relativistic adversarial-loss weight
of 5e-3. GAN finetuning is limited to 25 epochs. ResShift uses an exponential
15-step schedule with power 0.3, terminal eta 0.99, minimum noise level 0.04,
and kappa 2.0. Its first-stage latent scaling factors are 1.015483, 1.043293,
and 1.097812 for OLI2MSI, SEN2NAIP, and FY3FS2, respectively.

## Reproduction Procedure

1. Obtain the method from the official source listed above and comply with its
   license.
2. Adapt the input and output layers to the dataset band count, and set the
   dataset-specific super-resolution scale.
3. Apply the architecture and optimization settings listed above.
4. Train the SR model from scratch on the same paired training split, select
   checkpoints with the same validation split, and evaluate on the same test
   split.
5. Export SR predictions in the original band order and spatial extent before
   applying the common evaluation protocol.

Bilinear interpolation has no training stage. ESRGAN is trained in two stages:
first set `train.stage: psnr_pretrain`; then set
`train.stage: gan_finetune` and initialize it with the locally trained
PSNR-oriented checkpoint. ResShift is also sequential: train the independent
first-stage autoencoder before the latent restoration model.

## Evaluation Protocol

Spatial and spectral performance must be evaluated separately using the metric
definitions in `CSUA_LDM_Quality_Metrics.py` and the common utilities in
`CSUA_LDM_Evaluate_Common.py`.

- Spatial evaluation compares each SR output directly with its paired HR
  reference using PSNR, SSIM, GMSD, FSIM, and LPIPS.
- Spectral evaluation maps each SR output back to the LR domain using the
  dataset-specific degradation settings, then computes SAM, ERGAS, and SCC
  against the original LR observation.
- Raw-value conversion, normalization, band order, and valid spatial extent
  must remain identical across all methods.

## Weight Provenance and Fairness

The SR models used in the comparison were trained within the local paired-data
pipeline rather than evaluated with upstream SR checkpoints. Bilinear has no
weights. ESRGAN finetuning starts from its locally trained PSNR-oriented
checkpoint; its perceptual feature extractor may use ImageNet-pretrained VGG19
weights as configured. ResShift restoration is trained after its locally
trained first-stage autoencoder.

For fair comparison, all methods use the same paired splits, dataset-specific
scale, raw-value conversion, band order, and evaluation metrics. Spatial
quality is measured against HR references. Spectral quality is measured after
mapping each SR output back to the LR domain with the same degradation
mechanism.

No third-party source implementation or upstream pretrained SR checkpoint is
distributed in this GitHub repository.
