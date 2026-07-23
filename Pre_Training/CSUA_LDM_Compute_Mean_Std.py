from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CSUA_LDM_TifReader import Tif_Read_and_Write

DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"

gdal.UseExceptions()


def _normalize_root_paths(root_paths):
    if isinstance(root_paths, (list, tuple)):
        return list(root_paths)
    return [root_paths]


def _collect_split_image_paths(root_paths, split_name):
    hr_img_path_list = []
    lr_img_path_list = []

    for root_path in _normalize_root_paths(root_paths):
        hr_dir = os.path.join(root_path, f"{split_name}_hr")
        lr_dir = os.path.join(root_path, f"{split_name}_lr")

        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR split directory not found: {hr_dir}")
        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"LR split directory not found: {lr_dir}")

        hr_img_path_list.extend(
            [
                os.path.join(hr_dir, filename)
                for filename in sorted(os.listdir(hr_dir))
                if os.path.isfile(os.path.join(hr_dir, filename))
            ]
        )
        lr_img_path_list.extend(
            [
                os.path.join(lr_dir, filename)
                for filename in sorted(os.listdir(lr_dir))
                if os.path.isfile(os.path.join(lr_dir, filename))
            ]
        )

    if not hr_img_path_list:
        raise ValueError(f"No HR images found for split '{split_name}'.")
    if not lr_img_path_list:
        raise ValueError(f"No LR images found for split '{split_name}'.")

    return hr_img_path_list, lr_img_path_list


def compute_channel_mean_from_paths(
    img_paths,
    scale_to_unit_interval=False,
    scale_divisor=1.0,
):
    """
    img_paths: List[str]
    Returns:
        mean: (C,)
        std: (C,)
    """
    ch_sum = None
    ch_sumsq = None
    valid_cnt = None

    for path in tqdm(img_paths):
        img, _, _ = Tif_Read_and_Write().Tif_Read(path)  # (C, H, W)
        img = img.astype(np.float64)
        if bool(scale_to_unit_interval):
            img = img / float(scale_divisor)

        mask = np.isfinite(img)
        x = np.where(mask, img, 0.0)

        per_ch_sum = x.sum(axis=(1, 2))
        per_ch_sumsq = (x * x).sum(axis=(1, 2))
        per_ch_cnt = mask.sum(axis=(1, 2)).astype(np.float64)

        if ch_sum is None:
            ch_sum = per_ch_sum
            ch_sumsq = per_ch_sumsq
            valid_cnt = per_ch_cnt
        else:
            ch_sum += per_ch_sum
            ch_sumsq += per_ch_sumsq
            valid_cnt += per_ch_cnt

    valid_cnt = np.maximum(valid_cnt, 1.0)
    mean = ch_sum / valid_cnt
    var = ch_sumsq / valid_cnt - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, std


def _find_dataset_config(dataset_name: str) -> Path:
    """Locate the dataset YAML whose stem or dataset.name matches the selection."""
    matches = []
    expected_name = str(dataset_name).strip().casefold()

    for yaml_path in sorted(DATASET_CONFIG_DIR.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        configured_name = str(config.get("dataset", {}).get("name", "")).strip().casefold()
        if configured_name == expected_name or yaml_path.stem.casefold() == expected_name:
            matches.append(yaml_path)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one dataset config for '{dataset_name}', found {len(matches)}: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _resolve_dataset_config(args: argparse.Namespace) -> Path:
    if args.dataset_config is not None:
        yaml_path = args.dataset_config.expanduser()
        if not yaml_path.is_absolute():
            yaml_path = (Path.cwd() / yaml_path).resolve()
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Dataset config not found: {yaml_path}")
        return yaml_path
    return _find_dataset_config(args.dataset)


def _load_dataset_settings(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    dataset = config.get("dataset", {})
    geometry = config.get("geometry", {})
    raw_scaling = config.get("raw_scaling", {})
    required_fields = {
        "dataset.name": dataset.get("name"),
        "dataset.rootdir_list_forTrain": dataset.get("rootdir_list_forTrain"),
        "dataset.split_name_forTrain": dataset.get("split_name_forTrain"),
        "geometry.input_channels": geometry.get("input_channels"),
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(
            f"Dataset config {yaml_path} is missing required fields: {', '.join(missing)}"
        )

    lr_scale_divisor = float(raw_scaling.get("lr_scale_divisor", 1.0))
    hr_scale_divisor = float(raw_scaling.get("hr_scale_divisor", 1.0))
    if lr_scale_divisor <= 0.0 or hr_scale_divisor <= 0.0:
        raise ValueError(
            "raw_scaling divisors must be positive: "
            f"lr={lr_scale_divisor}, hr={hr_scale_divisor}"
        )

    return {
        "dataset_name": str(dataset["name"]),
        "train_root_paths": dataset["rootdir_list_forTrain"],
        "train_split_name": str(dataset["split_name_forTrain"]),
        "input_channels": int(geometry["input_channels"]),
        "scale_lr_to_unit_interval": bool(
            raw_scaling.get("scale_lr_to_unit_interval", False)
        ),
        "lr_scale_divisor": lr_scale_divisor,
        "scale_hr_to_unit_interval": bool(
            raw_scaling.get("scale_hr_to_unit_interval", False)
        ),
        "hr_scale_divisor": hr_scale_divisor,
    }


def _format_channel_values(values, precision: int = 8) -> str:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Normalization statistics contain non-finite values: {values}")
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _replace_yaml_list(text: str, field: str, values) -> str:
    pattern = re.compile(
        rf"^(?P<prefix>[ \t]*{re.escape(field)}[ \t]*:[ \t]*)"
        rf"(?P<value>\[[^\r\n]*\])(?P<suffix>[ \t]*(?:#[^\r\n]*)?)$",
        flags=re.MULTILINE,
    )
    replacement_value = _format_channel_values(values)
    updated_text, count = pattern.subn(
        lambda match: f"{match.group('prefix')}{replacement_value}{match.group('suffix')}",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not uniquely update normalization.{field} in dataset YAML.")
    return updated_text


def write_hr_normalization_to_dataset_config(
    yaml_path: Path,
    input_channels: int,
    hr_mean,
    hr_std,
) -> Path:
    """Update only the effective HR normalization fields in the dataset YAML."""
    hr_mean = np.asarray(hr_mean, dtype=np.float64).reshape(-1)
    hr_std = np.asarray(hr_std, dtype=np.float64).reshape(-1)
    if hr_mean.size != input_channels or hr_std.size != input_channels:
        raise ValueError(
            "Computed HR normalization channel count does not match geometry.input_channels: "
            f"mean={hr_mean.size}, std={hr_std.size}, expected={input_channels}."
        )
    if np.any(hr_std <= 0.0):
        raise ValueError(f"HR standard deviations must be positive, got: {hr_std}")

    with yaml_path.open("r", encoding="utf-8", newline="") as file:
        original_text = file.read()

    updated_text = _replace_yaml_list(original_text, "hr_mean", hr_mean)
    updated_text = _replace_yaml_list(updated_text, "hr_std", hr_std)

    if updated_text != original_text:
        temporary_path = yaml_path.with_name(f"{yaml_path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8", newline="") as file:
                file.write(updated_text)
            os.replace(temporary_path, yaml_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    with yaml_path.open("r", encoding="utf-8") as file:
        verified_config = yaml.safe_load(file) or {}
    normalization = verified_config.get("normalization", {})
    written_mean = np.asarray(normalization.get("hr_mean", []), dtype=np.float64)
    written_std = np.asarray(normalization.get("hr_std", []), dtype=np.float64)
    expected_mean = np.asarray([float(f"{value:.8f}") for value in hr_mean])
    expected_std = np.asarray([float(f"{value:.8f}") for value in hr_std])
    if not np.array_equal(written_mean, expected_mean):
        raise RuntimeError(
            f"HR mean write verification failed: expected {expected_mean}, got {written_mean}."
        )
    if not np.array_equal(written_std, expected_std):
        raise RuntimeError(
            f"HR std write verification failed: expected {expected_std}, got {written_std}."
        )

    return yaml_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute LR/HR channel statistics from a selected dataset YAML and "
            "write the HR statistics back to that YAML."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--dataset",
        help=(
            "Dataset name or YAML stem under configs/datasets, for example "
            "oli2msi, sen2naip, or fy3fs2."
        ),
    )
    selection.add_argument(
        "--dataset-config",
        type=Path,
        help="Path to a dataset YAML file.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    yaml_path = _resolve_dataset_config(args)
    settings = _load_dataset_settings(yaml_path)
    train_root_paths = settings["train_root_paths"]
    train_split_name = settings["train_split_name"]

    hr_img_path_list, lr_img_path_list = _collect_split_image_paths(
        train_root_paths,
        train_split_name,
    )

    hr_mean, hr_std = compute_channel_mean_from_paths(
        hr_img_path_list,
        scale_to_unit_interval=settings["scale_hr_to_unit_interval"],
        scale_divisor=settings["hr_scale_divisor"],
    )
    lr_mean, lr_std = compute_channel_mean_from_paths(
        lr_img_path_list,
        scale_to_unit_interval=settings["scale_lr_to_unit_interval"],
        scale_divisor=settings["lr_scale_divisor"],
    )

    print(f"Dataset: {settings['dataset_name']}")
    print(f"Dataset config: {yaml_path}")
    print(f"Computed split: {train_split_name}")
    print("HR mean:", hr_mean)
    print("HR std :", hr_std)
    print("LR mean:", lr_mean)
    print("LR std :", lr_std)

    config_path = write_hr_normalization_to_dataset_config(
        yaml_path,
        settings["input_channels"],
        hr_mean,
        hr_std,
    )
    print(f"Updated dataset config: {config_path}")


if __name__ == "__main__":
    main()
