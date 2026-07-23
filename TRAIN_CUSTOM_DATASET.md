# Training CSUA-LDM on a Custom Dataset

This guide describes how to train the main CSUA-LDM model on a paired
cross-sensor dataset. Run all commands from the repository root.

## 1. Prepare Paired Images

Use the following flat directory layout:

```text
MY_DATASET/
|-- train_lr/
|-- train_hr/
|-- valid_lr/
|-- valid_hr/
|-- test_lr/
|-- test_hr/
|-- draw_lr/
`-- draw_hr/
```

Each folder must contain GeoTIFF files (`.tif` or `.tiff`). An LR image and
its HR reference are paired only when their complete filenames, including the
extension, are identical:

```text
train_lr/sample_0001.tif
train_hr/sample_0001.tif
```

The data must satisfy the following requirements:

- LR and HR images must be spatially registered and describe the same scene.
- The LR-to-HR spatial ratio must be a fixed integer throughout the dataset.
- LR and HR images must contain the same number of bands in the same order.
- Every HR image height and width must be divisible by `8`.
- Each source image must be at least as large as its configured crop size.
- Invalid values should be handled before training. The loader replaces
  `NaN` and infinite values with zero and skips all-zero pairs.
- Training, validation, and test samples should be geographically and
  temporally separated where possible to prevent leakage.

The training loader applies aligned random crops. Validation, test, and draw
loaders use aligned center crops. If an LR crop begins at `(y, x)`, the HR
crop begins at `(scale * y, scale * x)`; accurate registration is therefore
essential.

The `draw` split is used for visual reports during training. It may contain a
small fixed subset, but the corresponding LR and HR directories must exist.

## 2. Create a Dataset Configuration

Copy an existing file from `configs/datasets/` and save it under a new,
filesystem-friendly name, for example:

```text
configs/datasets/my_dataset.yaml
```

A complete starting template is shown below. Forward slashes are recommended
in YAML paths on both Windows and Linux.

```yaml
dataset:
  name: MYDATASET
  rootdir_list_forTrain: "D:/datasets/MY_DATASET"
  split_name_forTrain: "train"
  rootdir_list_forValid: "D:/datasets/MY_DATASET"
  split_name_forValid: "valid"
  rootdir_list_forTest: "D:/datasets/MY_DATASET"
  split_name_forTest: "test"
  rootdir_list_forDraw: "D:/datasets/MY_DATASET"
  split_name_forDraw: "draw"

geometry:
  input_channels: 4
  superres_scale: 4
  lr_crop_size: 96

raw_scaling:
  scale_lr_to_unit_interval: true
  lr_scale_divisor: 10000.0
  scale_hr_to_unit_interval: true
  hr_scale_divisor: 10000.0

visualization:
  vis_band_order: [0, 1, 2]

degradation:
  hr_gaussian_kernel: 7
  hr_gaussian_sigma: 1.0
  hr_down_mode: area
  hr_up_mode: bicubic
  soft_hr_gaussian_blur_enabled: false
  soft_hr_gaussian_kernel: 3
  soft_hr_gaussian_sigma: 0.5
  soft_hr_down_mode: area
  soft_hr_up_mode: bicubic
  lr_up_mode: bicubic
  lr_gaussian_blur_enabled: false
  lr_gaussian_kernel: 3
  lr_gaussian_sigma: 0.5

normalization:
  hr_mean: [0.0, 0.0, 0.0, 0.0]
  hr_std: [1.0, 1.0, 1.0, 1.0]
```

Adjust these fields carefully:

- `dataset.name` identifies checkpoints, logs, and the dataset-specific latent
  scaling factor. Keep it unchanged throughout the experiment.
- `input_channels` must equal the band count in every LR and HR file.
- `superres_scale` is the integer LR-to-HR spatial ratio.
- The HR crop size is derived as
  `lr_crop_size * superres_scale`.
- The current main-model configuration is designed for `384 x 384` HR crops.
  To reuse it directly, choose `lr_crop_size * superres_scale = 384`, such as
  `128 x 3`, `96 x 4`, or `24 x 16`.
- If another HR crop size is used, review
  `diffusion.unet.attn_resolutions` in `configs/variants/main.yaml` and the
  resulting GPU memory use.
- Enable raw scaling only when division by the specified divisor converts the
  sensor values to the intended numerical range. Typical divisors include
  `10000` for scaled reflectance and `255` for 8-bit imagery.
- `vis_band_order` uses zero-based band indexes and controls RGB previews
  only. It does not reorder the model input.
- The placeholder `hr_mean` and `hr_std` arrays must have
  `input_channels` entries. They are replaced in Step 5.

Gaussian degradation parameters are dataset dependent. The values above are
initial placeholders rather than universal recommendations.

## 3. Activate the Custom Configuration

The selected dataset YAML can be overridden without editing Python files.

PowerShell:

```powershell
$env:CSUA_LDM_DATASET_YAML_OVERRIDE = (Resolve-Path "configs/datasets/my_dataset.yaml").Path
```

Bash:

```bash
export CSUA_LDM_DATASET_YAML_OVERRIDE="$(pwd)/configs/datasets/my_dataset.yaml"
```

Keep this variable set in every terminal used for preprocessing, training, or
evaluation. Verify the selected file before starting:

PowerShell:

```powershell
Write-Output $env:CSUA_LDM_DATASET_YAML_OVERRIDE
```

Bash:

```bash
echo "$CSUA_LDM_DATASET_YAML_OVERRIDE"
```

## 4. Inspect LR-HR Pairing

Before training, use `CSUA_LDM_Dataset.py` to inspect several LR-HR pairs:

```bash
python CSUA_LDM_Dataset.py --dataset-config configs/datasets/my_dataset.yaml --split train --num-samples 4 --show
```

The script prints the paired filenames and unmatched-file counts, and shows
`LR`, `LR up`, `HR down`, `HR down-up`, and `HR` side by side. Check that the
filenames correspond and that stable structures occur at the same locations
in LR-up and HR. The figure is also saved to
`tmp/mydataset_train_pair_check.jpg`. To inspect specific samples, use:

```bash
python CSUA_LDM_Dataset.py --dataset-config configs/datasets/my_dataset.yaml --split train --indices 0,12,35
```

Repeat the check for `valid`, `test`, and `draw`. Display stretching is used
only to make structures easier to inspect and should not be used to compare
radiometric values. Omit `--show` when only the saved figure is needed.

## 5. Compute Normalization Statistics

Compute channel statistics from the training split:

```bash
python Pre_Training/CSUA_LDM_Compute_Mean_Std.py --dataset-config configs/datasets/my_dataset.yaml
```

The script applies the configured raw-value scaling, computes per-band
statistics, and writes the HR mean and standard deviation back to
`normalization.hr_mean` and `normalization.hr_std` in the same YAML file.
Review the updated values before training.

## 6. Calibrate the HR Degradation

Search for a Gaussian blur strength on the validation split:

```bash
python Pre_Training/CSUA_LDM_Search_Best_HR_Gaussian_Params.py
```

The script reads the active dataset configuration and reports the validation
results for candidate values. It does not modify the YAML automatically.
After selecting `hr_gaussian_sigma`, set:

```text
hr_gaussian_kernel = 2 * ceil(3 * hr_gaussian_sigma) + 1
```

Then update `degradation.hr_gaussian_sigma` and
`degradation.hr_gaussian_kernel` manually.

## 7. Train the VAE

```bash
python Main_Model/CSUA_LDM_VAE/CSUA_LDM_VAE_Code/CSUA_LDM_VAE_train.py
```

The VAE must be trained first. It learns the content-style representation and
the latent uncertainty used by the subsequent diffusion stage. Check the
console output at startup to confirm the dataset name, paths, band count, crop
size, and super-resolution scale.

## 8. Estimate the Latent Scaling Factor

After VAE training, run:

```bash
python Main_Model/CSUA_LDM_VAE/CSUA_LDM_VAE_Code/CSUA_LDM_compute_scaling_factor.py
```

The script loads the dataset-specific VAE checkpoint and writes the estimated
factor to `latent.scaling_factor_by_dataset` in
`configs/variants/main.yaml`. Recompute this factor whenever the VAE,
dataset, band configuration, or normalization changes.

## 9. Train the Conditional Diffusion Model

```bash
python Main_Model/CSUA_LDM_Diffusion/CSUA_LDM_Diffusion_Code/CSUA_LDM_Diffusion_train.py
```

The diffusion model operates on the VAE content subspace. Do not start this
stage until the VAE checkpoint and the dataset-specific latent scaling factor
are available.

Training duration, batch sizes, optimizer settings, and resume behavior are
controlled by:

```text
configs/runtime/default_train.yaml
configs/variants/main.yaml
```

Reduce the physical or micro-batch size if GPU memory is insufficient.

## 10. Tune Uncertainty-Guided Sampling

This step is optional and requires valid VAE and diffusion checkpoints:

```bash
python Pre_Training/CSUA_LDM_Search_Best_Uncertainty_Guidance_Params.py --dataset current
```

The guidance strength is dataset dependent. Select parameters on the
validation split rather than assuming that stronger guidance improves every
metric.

## 11. Evaluate the Trained Model

Evaluate spatial reconstruction against paired HR references:

```bash
python Main_Model/CSUA_LDM_SpatialEvaluate.py --style-source hr --sampling-uncertainty-guidance yaml
```

Evaluate preservation of the source LR spectral characteristics:

```bash
python Main_Model/CSUA_LDM_SpectralEvaluate.py --style-source lr_up --sampling-uncertainty-guidance yaml
```

`--sampling-uncertainty-guidance` accepts `yaml`, `on`, or `off`.

The style source has different interpretations:

- `lr_up` is the deployable setting because it only requires the LR input.
- `hr` uses the paired HR style as a controlled diagnostic setting and is not
  available when no HR reference exists.
- `hr_down_up` is an additional paired-data diagnostic setting.

## Checklist

Before starting a long run, verify that:

- every LR file has an identically named HR file;
- stable structures occur at corresponding locations in LR-up and HR;
- all samples have the configured band count and band order;
- the scale ratio and spatial registration are correct;
- every HR image height and width is divisible by `8`;
- raw scaling matches the stored pixel values;
- `hr_mean` and `hr_std` were computed from the training split;
- the custom YAML override is visible in the current shell;
- the VAE is trained before computing the latent scaling factor;
- the latent scaling factor is computed before diffusion training; and
- evaluation reports the intended style source and sampling mode.

For the general environment setup and repository structure, see
[README.md](README.md).
