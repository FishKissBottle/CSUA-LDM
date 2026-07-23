# Main-Model Output Examples: Standard DDPM

This directory contains eight CSUA-LDM test examples for each of the
OLI2MSI, SEN2NAIP, and FY3FS2 datasets. The examples use the first
test batch and the same data construction, EMA checkpoints, random seed,
and sampling path as `Main_Model/CSUA_LDM_SpatialEvaluate.py`.

The configuration uses 1000-step v-prediction DDPM sampling and disables uncertainty-guided noise gating.
The random seed is `999`. Both `style-lr_up` and `style-hr` outputs are
provided for exactly the same test samples and center-crop regions.

- OLI2MSI: `128 x 128 -> 384 x 384`
- SEN2NAIP: `96 x 96 -> 384 x 384`
- FY3FS2: `24 x 24 -> 384 x 384`

Each sample includes a JPEG preview and two all-band, georeferenced
Float32 GeoTIFFs. Square-root stretching is used only in the preview.
The GeoTIFF values remain in the model output range `[0, 1]`.

## Gallery

| Test index | OLI2MSI | SEN2NAIP | FY3FS2 |
|---:|---|---|---|
| 0 | ![OLI2MSI test image 0](oli2msi/test_img_000_ddpm_preview.jpg) | ![SEN2NAIP test image 0](sen2naip/test_img_000_ddpm_preview.jpg) | ![FY3FS2 test image 0](fy3fs2/test_img_000_ddpm_preview.jpg) |
| 1 | ![OLI2MSI test image 1](oli2msi/test_img_001_ddpm_preview.jpg) | ![SEN2NAIP test image 1](sen2naip/test_img_001_ddpm_preview.jpg) | ![FY3FS2 test image 1](fy3fs2/test_img_001_ddpm_preview.jpg) |
| 2 | ![OLI2MSI test image 2](oli2msi/test_img_002_ddpm_preview.jpg) | ![SEN2NAIP test image 2](sen2naip/test_img_002_ddpm_preview.jpg) | ![FY3FS2 test image 2](fy3fs2/test_img_002_ddpm_preview.jpg) |
| 3 | ![OLI2MSI test image 3](oli2msi/test_img_003_ddpm_preview.jpg) | ![SEN2NAIP test image 3](sen2naip/test_img_003_ddpm_preview.jpg) | ![FY3FS2 test image 3](fy3fs2/test_img_003_ddpm_preview.jpg) |
| 4 | ![OLI2MSI test image 4](oli2msi/test_img_004_ddpm_preview.jpg) | ![SEN2NAIP test image 4](sen2naip/test_img_004_ddpm_preview.jpg) | ![FY3FS2 test image 4](fy3fs2/test_img_004_ddpm_preview.jpg) |
| 5 | ![OLI2MSI test image 5](oli2msi/test_img_005_ddpm_preview.jpg) | ![SEN2NAIP test image 5](sen2naip/test_img_005_ddpm_preview.jpg) | ![FY3FS2 test image 5](fy3fs2/test_img_005_ddpm_preview.jpg) |
| 6 | ![OLI2MSI test image 6](oli2msi/test_img_006_ddpm_preview.jpg) | ![SEN2NAIP test image 6](sen2naip/test_img_006_ddpm_preview.jpg) | ![FY3FS2 test image 6](fy3fs2/test_img_006_ddpm_preview.jpg) |
| 7 | ![OLI2MSI test image 7](oli2msi/test_img_007_ddpm_preview.jpg) | ![SEN2NAIP test image 7](sen2naip/test_img_007_ddpm_preview.jpg) | ![FY3FS2 test image 7](fy3fs2/test_img_007_ddpm_preview.jpg) |

Original test filenames, sampling settings, and GeoTIFF paths are listed
in [manifest.csv](manifest.csv). The source datasets are not redistributed.
