"""Package compact CSUA-LDM examples for the public repository.

The script expects inference outputs produced with ``--save-tif true`` and
creates, for each selected test sample:

1. a four-panel JPEG preview (LR-up, style-lr_up SR, style-hr SR, and HR); and
2. two losslessly compressed, all-band SR GeoTIFFs, one per decoding style.

The packaged examples are intended for visualization and format inspection,
not as a replacement for the complete benchmark datasets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from osgeo import gdal
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_KEYS = ("oli2msi", "sen2naip", "fy3fs2")
SAMPLING_MODES = {
    "ddpm": {
        "file_tag": "ddpm",
        "display_name": "Standard DDPM",
        "display_name_zh": "标准 DDPM",
        "manifest_name": "standard_ddpm",
        "guidance_enabled": False,
    },
    "uncertainty_guided": {
        "file_tag": "uncertainty_guided",
        "display_name": "Uncertainty-guided DDPM",
        "display_name_zh": "不确定性引导 DDPM",
        "manifest_name": "uncertainty_guided_ddpm",
        "guidance_enabled": True,
    },
}

gdal.UseExceptions()


def _read_chw(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open raster: {path}")
    array = dataset.ReadAsArray()
    dataset = None
    if array.ndim == 2:
        array = array[None, ...]
    return array.astype(np.float32, copy=False)


def _resize_chw(array: np.ndarray, height: int, width: int) -> np.ndarray:
    resized = []
    for channel in array:
        image = Image.fromarray(channel, mode="F")
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        resized.append(np.asarray(image, dtype=np.float32))
    return np.stack(resized, axis=0)


def _center_crop_chw(array: np.ndarray, crop_size: int) -> tuple[np.ndarray, int, int]:
    _, height, width = array.shape
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"Center crop {crop_size} exceeds raster size {(height, width)}."
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return array[:, top : top + crop_size, left : left + crop_size], top, left


def _to_display_rgb(array: np.ndarray, band_order: list[int]) -> np.ndarray:
    if max(band_order) >= array.shape[0]:
        raise ValueError(
            f"Visualization band order {band_order} is invalid for "
            f"{array.shape[0]}-channel data."
        )
    rgb = np.clip(array[band_order], 0.0, 1.0)
    rgb = np.sqrt(rgb)
    return np.transpose(rgb, (1, 2, 0))


def _list_tifs(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def _save_preview(
    lr_up: np.ndarray,
    sr_style_lr_up: np.ndarray,
    sr_style_hr: np.ndarray,
    hr: np.ndarray,
    band_order: list[int],
    dataset_name: str,
    test_index: int,
    sampling_display_name: str,
    output_path: Path,
) -> None:
    panels = (
        (_to_display_rgb(lr_up, band_order), "LR-up"),
        (
            _to_display_rgb(sr_style_lr_up, band_order),
            "CSUA-LDM SR\n(style-lr_up)",
        ),
        (_to_display_rgb(sr_style_hr, band_order), "CSUA-LDM SR\n(style-hr)"),
        (_to_display_rgb(hr, band_order), "HR reference"),
    )

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.8))
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=12)
        axis.axis("off")

    fig.text(
        0.5,
        0.025,
        f"test_img_{test_index} from {dataset_name} | {sampling_display_name}",
        ha="center",
        va="bottom",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.11, top=0.89, wspace=0.035)
    fig.savefig(
        output_path,
        dpi=160,
        format="jpg",
        pil_kwargs={"quality": 90, "optimize": True, "progressive": True},
    )
    plt.close(fig)


def _copy_compressed_geotiff(
    source: Path,
    destination: Path,
    spatial_reference: Path,
    crop_top: int,
    crop_left: int,
) -> None:
    destination.unlink(missing_ok=True)
    translated = gdal.Translate(
        str(destination),
        str(source),
        format="GTiff",
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "BIGTIFF=IF_SAFER",
        ],
    )
    if translated is None:
        raise RuntimeError(f"Failed to create GeoTIFF: {destination}")
    translated.FlushCache()
    translated = None

    reference = gdal.Open(str(spatial_reference), gdal.GA_ReadOnly)
    if reference is None:
        raise FileNotFoundError(f"Unable to open spatial reference: {spatial_reference}")
    geotransform = reference.GetGeoTransform()
    projection = reference.GetProjection()
    reference = None

    adjusted_geotransform = (
        geotransform[0] + crop_left * geotransform[1] + crop_top * geotransform[2],
        geotransform[1],
        geotransform[2],
        geotransform[3] + crop_left * geotransform[4] + crop_top * geotransform[5],
        geotransform[4],
        geotransform[5],
    )
    output = gdal.Open(str(destination), gdal.GA_Update)
    if output is None:
        raise RuntimeError(f"Unable to update GeoTIFF metadata: {destination}")
    output.SetGeoTransform(adjusted_geotransform)
    output.SetProjection(projection)
    output.FlushCache()
    output = None


def _load_dataset_config(dataset_key: str) -> dict:
    config_path = PROJECT_ROOT / "configs" / "datasets" / f"{dataset_key}.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _package_dataset(
    dataset_key: str,
    generated_root: Path,
    output_root: Path,
    sample_count: int,
    sampling_mode: str,
) -> list[dict[str, str | int]]:
    mode_config = SAMPLING_MODES[sampling_mode]
    file_tag = str(mode_config["file_tag"])
    config = _load_dataset_config(dataset_key)
    dataset_config = config["dataset"]
    raw_scaling = config["raw_scaling"]
    dataset_name = dataset_config["name"]
    band_order = list(config["visualization"]["vis_band_order"])
    lr_crop_size = int(config["geometry"]["lr_crop_size"])
    superres_scale = int(config["geometry"]["superres_scale"])
    hr_crop_size = lr_crop_size * superres_scale

    test_root = Path(dataset_config["rootdir_list_forTest"])
    split_name = dataset_config["split_name_forTest"]
    lr_files = _list_tifs(test_root / f"{split_name}_lr")
    hr_files = _list_tifs(test_root / f"{split_name}_hr")

    style_lr_up_root = generated_root / dataset_key / "style-lr_up" / "TIF_Outputs"
    style_hr_root = generated_root / dataset_key / "style-hr" / "TIF_Outputs"
    sr_style_lr_up_files = sorted(style_lr_up_root.glob("sr_*.tif"))
    sr_style_hr_files = sorted(style_hr_root.glob("sr_*.tif"))
    gt_files = sorted(style_lr_up_root.glob("gt_*.tif"))
    if min(
        len(lr_files),
        len(hr_files),
        len(sr_style_lr_up_files),
        len(sr_style_hr_files),
        len(gt_files),
    ) < sample_count:
        raise RuntimeError(
            f"{dataset_name} does not contain {sample_count} complete examples: "
            f"LR={len(lr_files)}, HR={len(hr_files)}, "
            f"SR(style-lr_up)={len(sr_style_lr_up_files)}, "
            f"SR(style-hr)={len(sr_style_hr_files)}, GT={len(gt_files)}"
        )

    dataset_output = output_root / dataset_key
    dataset_output.mkdir(parents=True, exist_ok=True)
    for stale_output in dataset_output.glob("test_img_*"):
        if stale_output.is_file():
            stale_output.unlink()
    manifest_rows: list[dict[str, str | int]] = []

    for test_index in range(sample_count):
        lr = _read_chw(lr_files[test_index])
        if raw_scaling["scale_lr_to_unit_interval"]:
            lr = lr / float(raw_scaling["lr_scale_divisor"])

        sr_style_lr_up = _read_chw(sr_style_lr_up_files[test_index])
        sr_style_hr = _read_chw(sr_style_hr_files[test_index])
        hr = _read_chw(gt_files[test_index])
        for style_name, sr in (
            ("style-lr_up", sr_style_lr_up),
            ("style-hr", sr_style_hr),
        ):
            if tuple(sr.shape[1:]) != (hr_crop_size, hr_crop_size):
                raise ValueError(
                    f"{dataset_name} {style_name} SR size {tuple(sr.shape[1:])} "
                    f"does not match the configured HR crop "
                    f"{(hr_crop_size, hr_crop_size)}."
                )
        lr_crop, _lr_top, _lr_left = _center_crop_chw(lr, lr_crop_size)
        lr_up = _resize_chw(lr_crop, hr_crop_size, hr_crop_size)

        hr_reference = _read_chw(hr_files[test_index])
        _hr_crop, hr_top, hr_left = _center_crop_chw(
            hr_reference,
            hr_crop_size,
        )

        stem = f"test_img_{test_index:03d}_{file_tag}"
        preview_name = f"{stem}_preview.jpg"
        style_lr_up_tif_name = f"{stem}_sr_style-lr_up.tif"
        style_hr_tif_name = f"{stem}_sr_style-hr.tif"
        _save_preview(
            lr_up,
            sr_style_lr_up,
            sr_style_hr,
            hr,
            band_order,
            dataset_name,
            test_index,
            str(mode_config["display_name"]),
            dataset_output / preview_name,
        )
        _copy_compressed_geotiff(
            sr_style_lr_up_files[test_index],
            dataset_output / style_lr_up_tif_name,
            spatial_reference=hr_files[test_index],
            crop_top=hr_top,
            crop_left=hr_left,
        )
        _copy_compressed_geotiff(
            sr_style_hr_files[test_index],
            dataset_output / style_hr_tif_name,
            spatial_reference=hr_files[test_index],
            crop_top=hr_top,
            crop_left=hr_left,
        )

        manifest_rows.append(
            {
                "dataset": dataset_name,
                "test_index": test_index,
                "source_filename": lr_files[test_index].name,
                "preview_jpg": f"{dataset_key}/{preview_name}",
                "sr_style_lr_up_geotiff": (
                    f"{dataset_key}/{style_lr_up_tif_name}"
                ),
                "sr_style_hr_geotiff": f"{dataset_key}/{style_hr_tif_name}",
                "decode_styles": "style-lr_up|style-hr",
                "sampling_seed": 999,
                "sampling_mode": str(mode_config["manifest_name"]),
                "uncertainty_guidance": bool(mode_config["guidance_enabled"]),
                "generation_pipeline": (
                    f"SpatialEvaluate_first_test_batch_{file_tag}"
                ),
                "crop_mode": "center",
                "lr_crop_size": lr_crop_size,
                "hr_crop_size": hr_crop_size,
            }
        )

    return manifest_rows


def _write_readmes(
    output_root: Path,
    sampling_mode: str,
    sample_count: int,
) -> None:
    mode_config = SAMPLING_MODES[sampling_mode]
    file_tag = str(mode_config["file_tag"])
    display_name = str(mode_config["display_name"])
    display_name_zh = str(mode_config["display_name_zh"])
    guidance_enabled = bool(mode_config["guidance_enabled"])
    guidance_en = (
        "enables uncertainty-guided noise gating"
        if guidance_enabled
        else "disables uncertainty-guided noise gating"
    )
    guidance_zh = "启用不确定性噪声门控" if guidance_enabled else "关闭不确定性噪声门控"

    english_lines = [
        f"# Main-Model Output Examples: {display_name}",
        "",
        f"This directory contains {sample_count} CSUA-LDM test examples for each of the",
        "OLI2MSI, SEN2NAIP, and FY3FS2 datasets. The examples use the first",
        "test batch and the same data construction, EMA checkpoints, random seed,",
        "and sampling path as `Main_Model/CSUA_LDM_SpatialEvaluate.py`.",
        "",
        f"The configuration uses 1000-step v-prediction DDPM sampling and {guidance_en}.",
        "The random seed is `999`. Both `style-lr_up` and `style-hr` outputs are",
        "provided for exactly the same test samples and center-crop regions.",
        "",
        "- OLI2MSI: `128 x 128 -> 384 x 384`",
        "- SEN2NAIP: `96 x 96 -> 384 x 384`",
        "- FY3FS2: `24 x 24 -> 384 x 384`",
        "",
        "Each sample includes a JPEG preview and two all-band, georeferenced",
        "Float32 GeoTIFFs. Square-root stretching is used only in the preview.",
        "The GeoTIFF values remain in the model output range `[0, 1]`.",
        "",
        "## Gallery",
        "",
        "| Test index | OLI2MSI | SEN2NAIP | FY3FS2 |",
        "|---:|---|---|---|",
    ]
    chinese_lines = [
        f"# 主模型输出样例：{display_name_zh}",
        "",
        "本目录分别提供 CSUA-LDM 主模型在 OLI2MSI、SEN2NAIP 和 FY3FS2",
        f"测试集上的 {sample_count} 个输出样例。样例取自测试集首个批次，数据构建、EMA",
        "权重、随机种子及采样路径均与 `Main_Model/CSUA_LDM_SpatialEvaluate.py`",
        "保持一致。",
        "",
        f"当前结果采用 1000 步 v-prediction DDPM 采样并{guidance_zh}，随机种子为",
        "`999`。同一批测试样本和中心裁剪区域均同时提供 `style-lr_up` 与",
        "`style-hr` 两种解码结果。",
        "",
        "- OLI2MSI：`128 x 128 -> 384 x 384`",
        "- SEN2NAIP：`96 x 96 -> 384 x 384`",
        "- FY3FS2：`24 x 24 -> 384 x 384`",
        "",
        "每个样例包含一张 JPEG 预览图和两份保存全部波段的带地理参考",
        "Float32 GeoTIFF。平方根拉伸仅用于预览，GeoTIFF 像元值保持在模型",
        "输出的 `[0, 1]` 范围内。",
        "",
        "## 样例索引",
        "",
        "| 测试索引 | OLI2MSI | SEN2NAIP | FY3FS2 |",
        "|---:|---|---|---|",
    ]
    for test_index in range(sample_count):
        filename = f"test_img_{test_index:03d}_{file_tag}_preview.jpg"
        english_lines.append(
            f"| {test_index} | ![OLI2MSI test image {test_index}]"
            f"(oli2msi/{filename}) | ![SEN2NAIP test image {test_index}]"
            f"(sen2naip/{filename}) | ![FY3FS2 test image {test_index}]"
            f"(fy3fs2/{filename}) |"
        )
        chinese_lines.append(
            f"| {test_index} | ![OLI2MSI 测试影像 {test_index}]"
            f"(oli2msi/{filename}) | ![SEN2NAIP 测试影像 {test_index}]"
            f"(sen2naip/{filename}) | ![FY3FS2 测试影像 {test_index}]"
            f"(fy3fs2/{filename}) |"
        )
    english_lines.extend(
        [
            "",
            "Original test filenames, sampling settings, and GeoTIFF paths are listed",
            "in [manifest.csv](manifest.csv). The source datasets are not redistributed.",
            "",
        ]
    )
    chinese_lines.extend(
        [
            "",
            "原始测试文件名、采样设置及 GeoTIFF 路径见",
            "[manifest.csv](manifest.csv)。本仓库不重新分发原始数据集。",
            "",
        ]
    )
    (output_root / "README.md").write_text(
        "\n".join(english_lines), encoding="utf-8"
    )
    (output_root / "README_中文版.md").write_text(
        "\n".join(chinese_lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package compact main-model examples for GitHub."
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=None,
        help="Root containing per-dataset inference outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination for curated repository examples.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="Number of test samples packaged per dataset.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=tuple(SAMPLING_MODES),
        default="ddpm",
        help="Sampling mode represented by the generated inference outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_root = args.generated_root or (
        PROJECT_ROOT
        / (
            "assets/_generated_eval"
            if args.sampling_mode == "ddpm"
            else "assets/_generated_eval_uncertainty_guided"
        )
    )
    output_root = args.output_root or (
        PROJECT_ROOT
        / "assets"
        / "examples"
        / f"main_model_samples_{args.sampling_mode}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for dataset_key in DATASET_KEYS:
        rows.extend(
            _package_dataset(
                dataset_key,
                generated_root.resolve(),
                output_root.resolve(),
                args.sample_count,
                args.sampling_mode,
            )
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _write_readmes(output_root, args.sampling_mode, args.sample_count)
    print(f"Packaged {len(rows)} examples in {output_root.resolve()}")


if __name__ == "__main__":
    main()
