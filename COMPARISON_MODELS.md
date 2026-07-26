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

| Method | Stage | Configuration used in the reported experiments |
|---|---|---|
| Bilinear | Interpolation | `align_corners=false`; no trainable parameters |
| DSen2 | Single-stage training | 6 residual blocks, 128 features, residual scale 0.1, bilinear pre-upsampling |
| EDSR | Single-stage training | 16 residual blocks, 64 features, residual scale 1.0 |
| RCAN | Single-stage training | 10 residual groups, 20 blocks per group, 64 features, reduction ratio 16 |
| ESRGAN | PSNR pretraining (training stage 1) | RRDB generator with 16 RRDB blocks, 48 features, and 24 growth channels, trained from scratch with L1 pixel loss (weight 1.0); discriminator not instantiated |
| ESRGAN | GAN finetuning (training stage 2) | Same RRDB generator initialized from the locally trained stage-1 checkpoint; VGG-style discriminator trained from scratch, with 48 initial feature channels and spectral normalization applied to its convolutional layers; pixel, VGG19 `conv5_4` perceptual, and relativistic adversarial loss weights of 1.0, 1.0, and 5e-3; perceptual loss restricted to RGB bands; maximum 25 epochs |
| SwinIR | Single-stage training | Image size 64, patch size 1, window size 8, embedding dimension 60, depths `[6, 6, 6, 6]`, heads `[6, 6, 6, 6]`, direct pixel-shuffle upsampling |
| ResShift | First-stage autoencoder (training stage 1) | KL autoencoder with 8 latent channels, downsampling factor 8, down-channel widths `[32, 64, 96, 128]`, and KL weight 5e-4; latent scaling factors of 1.015483, 1.043293, and 1.097812 for OLI2MSI, SEN2NAIP, and FY3FS2 |
| ResShift | Latent restoration (training stage 2) | Base channels 160, channel multipliers `[1, 2, 2, 4]`, and 15-step exponential schedule with power 0.3, terminal eta 0.99, minimum noise level 0.04, and kappa 2.0 |

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

## Release Scope

Public releases of this repository include the CSUA-LDM source code,
documentation, configurations, and curated examples. To comply with upstream
licenses and maintain clear licensing boundaries, they do not include adapted
third-party source code or pretrained SR weights released by upstream projects.
For further clarification about the reproduction settings or adaptation
details, contact
[1000550108@smail.shnu.edu.cn](mailto:1000550108@smail.shnu.edu.cn).
