# OLI2MSI and SEN2NAIP Data Sources and Splits

This directory documents the sources and local split protocols of the OLI2MSI and SEN2NAIP datasets used in the CSUA-LDM experiments. Each CSV file lists image filenames only. The paired LR and HR images share the same filename and can be located in the corresponding `*_lr` and `*_hr` directories.

## OLI2MSI

### Original Data

OLI2MSI was released by Wang et al. The original project is available at [wjwjww/OLI2MSI](https://github.com/wjwjww/OLI2MSI). It contains real paired Landsat-8 OLI and Sentinel-2 MSI images. The 30-m OLI images serve as LR observations, whereas the 10-m MSI images serve as HR references, corresponding to a scale factor of 3. Each image pair contains three visible bands, with LR and HR image sizes of 160 x 160 and 480 x 480 pixels, respectively.

The original release provides only training and test splits, containing 5,225 and 100 image pairs, respectively.

### Processing in This Project

- The original 100 test pairs are retained without changing their membership.
- The original 5,225 training pairs are further divided into 4,739 training pairs and 486 validation pairs.
- LR and HR filenames are matched one-to-one, with no filename overlap among the training, validation, and test splits.
- The three image pairs in the `draw` directory are visualization samples copied from the validation split and do not constitute an additional data split.

The validation filenames are listed in [oli2msi_valid.csv](oli2msi_valid.csv).

## SEN2NAIP

### Original Data

SEN2NAIP was released by Aybar et al. The data record is available from [Science Data Bank](https://doi.org/10.57760/sciencedb.17395), while download scripts and a dataset mirror are provided at [isp-uv-es/SEN2NAIP](https://huggingface.co/datasets/isp-uv-es/SEN2NAIP).

The original SEN2NAIP release includes cross-sensor and synthetic subsets. The cross-sensor subset contains 2,851 real Sentinel-2/NAIP image pairs, whereas the synthetic subset uses a degradation model to generate Sentinel-2-like LR images. This project uses only the real cross-sensor subset and excludes all synthetic data. The LR images are acquired by Sentinel-2 at 10 m, and the HR images are acquired by NAIP at 2.5 m. Both contain RGB and near-infrared bands, corresponding to a scale factor of 4. The LR and HR image sizes are 120 x 120 and 480 x 480 pixels, respectively.

### Processing in This Project

The original cross-sensor data do not provide the training, validation, and test split used in this study. We therefore organize the data as follows:

| Split | Paired images |
|---|---:|
| Train | 2423 |
| Validation | 285 |
| Test | 142 |

In the current experimental split, `ROI_0131.tif` is included only in the `draw` directory for result visualization and is not used for model training, validation, or quantitative testing. The other two samples in `draw` are copies of test images. This split record is retained to ensure the reproducibility of the reported experiments. Together, the training, validation, and test splits and this independent visualization sample cover all 2,851 cross-sensor ROIs. LR and HR filenames are matched one-to-one, with no filename overlap among the formal splits.

The validation and test filenames are listed in [sen2naip_valid.csv](sen2naip_valid.csv) and [sen2naip_test.csv](sen2naip_test.csv), respectively.

## FY3FS2 Data Availability

Public release of FY3FS2 involves redistribution of the original remote sensing imagery and therefore requires the relevant approval. The dataset is currently under review and cannot yet be made publicly available. FY3FS2 will be released after approval is granted.

## Manifest Validation

| Dataset | CSV | Rows | LR/HR names matched |
|---|---|---:|---|
| OLI2MSI | `oli2msi_valid.csv` | 486 | Yes |
| SEN2NAIP | `sen2naip_valid.csv` | 285 | Yes |
| SEN2NAIP | `sen2naip_test.csv` | 142 | Yes |

These manifests reflect the actual splits in the current experimental directories and do not contain the original image data.
