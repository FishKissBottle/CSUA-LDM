"""Load and merge YAML configurations for CSUA_LDM."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml


def _flatten_dict(nested: dict, prefix: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into dot-separated keys."""
    items: dict[str, object] = {}
    for key, value in nested.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value
    return items


def _to_namespace(flat: dict, sep: str = ".") -> SimpleNamespace:
    """Convert a flat dict into a nested SimpleNamespace for dot-access."""
    root: dict[str, object] = {}
    for key, value in flat.items():
        parts = key.split(sep)
        node = root
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]  # type: ignore[assignment]
        node[parts[-1]] = value

    def _build(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _build(v) for k, v in obj.items()})
        return obj

    return _build(root)


def _namespace_to_dict(obj):
    """Recursively convert nested SimpleNamespace objects into plain dicts."""
    if isinstance(obj, SimpleNamespace):
        return {key: _namespace_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, dict):
        return {key: _namespace_to_dict(value) for key, value in obj.items()}
    return obj


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two nested dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(
    yaml_path: str | Path,
    *,
    validate: bool = True,
) -> SimpleNamespace:
    """Load a single YAML config file.

    Args:
        yaml_path: Path to the YAML file.
        validate: Whether to run field validation after loading.

    Returns:
        A SimpleNamespace with dot-access to all config values.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If validation fails.
    """
    yaml_path = Path(yaml_path)
    dataset_override = os.environ.get("CSUA_LDM_DATASET_YAML_OVERRIDE", "").strip()
    if dataset_override and yaml_path.parent.name.lower() == "datasets":
        yaml_path = Path(dataset_override)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}

    flat = _flatten_dict(raw)
    cfg = _to_namespace(flat)

    if validate:
        _validate_config(cfg)

    return cfg


def merge_configs(*configs: SimpleNamespace) -> SimpleNamespace:
    """Deep-merge multiple configs, later ones override earlier ones."""
    merged_dict: dict[str, object] = {}
    for cfg in configs:
        merged_dict = _deep_merge_dicts(merged_dict, _namespace_to_dict(cfg))
    merged_cfg = _to_namespace(_flatten_dict(merged_dict))
    _validate_config(merged_cfg)
    return merged_cfg


def _validate_config(cfg: SimpleNamespace) -> None:
    """Run basic field validation on dataset-related configs."""
    errors: list[str] = []

    # Dataset geometry
    if hasattr(cfg, "geometry"):
        geo = cfg.geometry
        lr_crop = getattr(geo, "lr_crop_size", None)
        scale = getattr(geo, "superres_scale", None)
        if lr_crop is not None and scale is not None:
            expected_hr = lr_crop * scale
            hr_crop = getattr(geo, "hr_crop_size", None)
            if hr_crop is not None and hr_crop != expected_hr:
                errors.append(
                    f"geometry.hr_crop_size ({hr_crop}) != "
                    f"lr_crop_size ({lr_crop}) * superres_scale ({scale}) = {expected_hr}"
                )

        if lr_crop is not None and lr_crop <= 0:
            errors.append("geometry.lr_crop_size must be positive")
        if scale is not None and scale <= 0:
            errors.append("geometry.superres_scale must be positive")

    # Dataset paths
    if hasattr(cfg, "dataset"):
        ds = cfg.dataset
        for key in ("rootdir_list_forTrain", "rootdir_list_forValid",
                    "rootdir_list_forTest", "rootdir_list_forDraw"):
            path = getattr(ds, key, None)
            if path is not None and not os.path.isdir(path):
                # Do not hard-fail on draw/test missing; only warn for train/valid
                if key in ("rootdir_list_forTrain", "rootdir_list_forValid"):
                    errors.append(f"dataset.{key} does not exist: {path}")

    # Normalization
    if hasattr(cfg, "normalization"):
        norm = cfg.normalization
        ch = getattr(getattr(cfg, "geometry", None), "input_channels", None)
        for key in ("hr_mean", "hr_std"):
            arr = getattr(norm, key, None)
            if arr is not None and ch is not None:
                if len(arr) != ch:
                    errors.append(
                        f"normalization.{key} length ({len(arr)}) != "
                        f"geometry.input_channels ({ch})"
                    )

    if errors:
        raise ValueError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )
