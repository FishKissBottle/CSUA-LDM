import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from osgeo import gdal
from torch.utils.data import Dataset

from CSUA_LDM_TifReader import Tif_Read_and_Write

gdal.UseExceptions()


def _is_tif_file(filename: str) -> bool:
    return filename.lower().endswith((".tif", ".tiff"))


def _list_tif_files(folder_path: str) -> list[str]:
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Dataset folder does not exist: {folder_path}")
    return sorted(
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if _is_tif_file(filename)
    )


def _collect_flat_split_pairs(root_dir: str, split_name: str | None) -> tuple[list[str], list[str]] | None:
    candidate_pairs: list[tuple[str, str]] = []
    if split_name:
        candidate_pairs.append(
            (os.path.join(root_dir, f"{split_name}_lr"), os.path.join(root_dir, f"{split_name}_hr"))
        )
    candidate_pairs.append((os.path.join(root_dir, "lr"), os.path.join(root_dir, "hr")))

    for lr_dir, hr_dir in candidate_pairs:
        if not (os.path.isdir(lr_dir) and os.path.isdir(hr_dir)):
            continue
        lr_paths = _list_tif_files(lr_dir)
        hr_paths = _list_tif_files(hr_dir)
        lr_by_name = {os.path.basename(path): path for path in lr_paths}
        hr_by_name = {os.path.basename(path): path for path in hr_paths}
        common_names = sorted(set(lr_by_name.keys()) & set(hr_by_name.keys()))
        if not common_names:
            raise ValueError(
                f"No paired LR/HR files found under split directories: {lr_dir} and {hr_dir}"
            )
        return [lr_by_name[name] for name in common_names], [hr_by_name[name] for name in common_names]
    return None


def _collect_legacy_nested_pairs(root_dir: str) -> tuple[list[str], list[str]]:
    lr_img_paths: list[str] = []
    hr_img_paths: list[str] = []
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.endswith("_FY"):
            lr_img_paths.extend(_list_tif_files(folder_path))
        elif folder_name.endswith("_S2"):
            hr_img_paths.extend(_list_tif_files(folder_path))

    if len(lr_img_paths) == 0 or len(hr_img_paths) == 0:
        raise ValueError(
            f"Could not find legacy FY/S2 paired folders under dataset root: {root_dir}"
        )
    if len(lr_img_paths) != len(hr_img_paths):
        raise ValueError(
            f"Legacy dataset pair count mismatch: {len(lr_img_paths)} LR files vs {len(hr_img_paths)} HR files."
        )
    return lr_img_paths, hr_img_paths


def _normalize_vis_band_order(vis_band_order, num_channels: int) -> list[int]:
    """Normalize and validate RGB visualization channel order."""
    if vis_band_order is None:
        resolved = list(range(min(3, int(num_channels))))
    else:
        resolved = [int(idx) for idx in vis_band_order]
    if len(resolved) != min(3, int(num_channels)):
        raise ValueError(
            f"`vis_band_order` must contain {min(3, int(num_channels))} channels, got {resolved}."
        )
    if min(resolved) < 0 or max(resolved) >= int(num_channels):
        raise ValueError(
            f"`vis_band_order`={resolved} is invalid for an input with {int(num_channels)} channels."
        )
    return resolved


def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Generate a 2D Gaussian kernel as a torch tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = g.unsqueeze(1) * g.unsqueeze(0)
    kernel /= kernel.sum()
    return kernel


def _gaussian_blur_2d(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a (C, H, W) image tensor using replicate padding.

    Args:
        img: Input image tensor of shape (C, H, W).
        kernel_size: Size of the Gaussian kernel (must be odd).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred image tensor of shape (C, H, W).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    c = img.shape[0]
    kernel = _gaussian_kernel_2d(kernel_size, sigma).to(device=img.device, dtype=img.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    img_padded = F.pad(img.unsqueeze(0), (padding, padding, padding, padding), mode='replicate')
    blurred = F.conv2d(img_padded, kernel, groups=c).squeeze(0)
    return blurred


def _infer_num_channels_from_tif(tif_path: str) -> int:
    """Infer channel count directly from a TIFF file header."""
    dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open TIFF for channel inference: {tif_path}")
    try:
        return int(dataset.RasterCount)
    finally:
        dataset = None


def _to_chw_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    """Convert an HWC numpy image into a float32 CHW tensor."""
    return torch.from_numpy(np.transpose(img_hwc, (2, 0, 1))).float()


class SuperResolutionDataset(Dataset):
    """Dataset for paired low-resolution and high-resolution remote sensing images."""

    def __init__(
        self,
        root_dir,
        is_train=True,
        train_transforms=None,
        test_transforms=None,
        split_name=None,
        vis_band_order=None,
        superres_scale=None,
        lr_crop_size=None,
        hr_crop_size=None,
        input_channels=None,
        hr_degradation_gaussian_kernel=5,
        hr_degradation_gaussian_sigma=1.0,
        hr_degradation_down_mode="area",
        hr_degradation_up_mode="bilinear",
        soft_hr_gaussian_blur_enabled=False,
        soft_hr_gaussian_kernel=11,
        soft_hr_gaussian_sigma=1.0,
        soft_hr_down_mode="area",
        soft_hr_up_mode="bicubic",
        lr_up_mode="bicubic",
        lr_gaussian_blur_enabled=False,
        lr_gaussian_kernel=3,
        lr_gaussian_sigma=0.8,
        scale_lr_to_unit_interval=False,
        lr_scale_divisor=1.0,
        scale_hr_to_unit_interval=False,
        hr_scale_divisor=1.0,
    ):
        """
        Args:
            root_dir: Root directory containing paired LR/HR subfolders.
            is_train: If True, use random crop coordinates; otherwise center crop.
        """
        if root_dir is None or str(root_dir).strip() == "":
            raise ValueError("`root_dir` must be a non-empty dataset directory path.")
        self.root_dir = os.fspath(root_dir)
        self.split_name = None if split_name in (None, "", "none", "None") else str(split_name)

        flat_pairs = _collect_flat_split_pairs(root_dir=self.root_dir, split_name=self.split_name)
        if flat_pairs is not None:
            self.lr_img_paths, self.hr_img_paths = flat_pairs
            self.dataset_layout = "flat_split_dirs"
        else:
            self.lr_img_paths, self.hr_img_paths = _collect_legacy_nested_pairs(root_dir=self.root_dir)
            self.dataset_layout = "legacy_nested_pairs"

        if len(self.hr_img_paths) == 0:
            raise ValueError(f"No HR images found under dataset root: {self.root_dir}")

        self.input_channels = int(
            _infer_num_channels_from_tif(self.hr_img_paths[0]) if input_channels is None else input_channels
        )
        self.multiplier = int(4 if superres_scale is None else superres_scale)
        self.lr_crop_size = int(40 if lr_crop_size is None else lr_crop_size)
        resolved_hr_crop_size = self.lr_crop_size * self.multiplier if hr_crop_size is None else hr_crop_size
        self.hr_crop_size = int(resolved_hr_crop_size)
        self.vis_band_order = _normalize_vis_band_order(
            vis_band_order=vis_band_order,
            num_channels=self.input_channels,
        )
        if self.multiplier <= 0:
            raise ValueError(f"`superres_scale` must be positive, got {self.multiplier}.")
        if self.lr_crop_size <= 0 or self.hr_crop_size <= 0:
            raise ValueError(
                f"Crop sizes must be positive, got lr={self.lr_crop_size}, hr={self.hr_crop_size}."
            )
        if self.hr_crop_size != self.lr_crop_size * self.multiplier:
            raise ValueError(
                "Configured HR crop size must equal LR crop size * super-resolution scale, "
                f"got lr={self.lr_crop_size}, scale={self.multiplier}, hr={self.hr_crop_size}."
            )
        self.scale_lr_to_unit_interval = bool(scale_lr_to_unit_interval)
        self.lr_scale_divisor = float(lr_scale_divisor)
        self.scale_hr_to_unit_interval = bool(scale_hr_to_unit_interval)
        self.hr_scale_divisor = float(hr_scale_divisor)
        if self.lr_scale_divisor <= 0.0 or self.hr_scale_divisor <= 0.0:
            raise ValueError(
                "Raw-scaling divisors must be > 0. "
                f"Got lr={self.lr_scale_divisor}, hr={self.hr_scale_divisor}."
            )

        self.is_train = is_train
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        # Degradation parameters for hr_down_img (from config)
        self.gaussian_kernel = int(hr_degradation_gaussian_kernel)
        self.gaussian_sigma = float(hr_degradation_gaussian_sigma)
        self.down_mode = str(hr_degradation_down_mode)
        self.up_mode = str(hr_degradation_up_mode)
        self.lr_up_mode = str(lr_up_mode)
        self.soft_hr_blur_enabled = bool(soft_hr_gaussian_blur_enabled)
        self.soft_hr_kernel = int(soft_hr_gaussian_kernel)
        self.soft_hr_sigma = float(soft_hr_gaussian_sigma)
        self.soft_hr_down_mode = str(soft_hr_down_mode)
        self.soft_hr_up_mode = str(soft_hr_up_mode)

        self.lr_blur_enabled = bool(lr_gaussian_blur_enabled)
        self.lr_gaussian_kernel = int(lr_gaussian_kernel)
        self.lr_gaussian_sigma = float(lr_gaussian_sigma)

    def __len__(self):
        """Return the number of HR image files available."""
        return len(self.hr_img_paths)

    def _build_soft_hr_target(self, hr_img: torch.Tensor) -> torch.Tensor:
        """Build a softened HR target image by Gaussian blur + downsample + upsample.

        Args:
            hr_img: High-resolution image tensor of shape (C, H_hr, W_hr).

        Returns:
            A tensor of the same shape as hr_img, representing the softened target.
        """
        c, h_hr, w_hr = hr_img.shape
        h_lr = h_hr // self.multiplier
        w_lr = w_hr // self.multiplier

        # 1) Gaussian blur on HR image
        hr_blurred = _gaussian_blur_2d(hr_img, self.soft_hr_kernel, self.soft_hr_sigma)

        # 2) Downsample to LR size
        hr_down = F.interpolate(
            hr_blurred.unsqueeze(0),
            size=(h_lr, w_lr),
            mode=self.soft_hr_down_mode if self.soft_hr_down_mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area') else 'area',
            align_corners=None if self.soft_hr_down_mode == 'area' else False,
        ).squeeze(0)

        # 3) Upsample back to HR size
        hr_soft = F.interpolate(
            hr_down.unsqueeze(0),
            size=(h_hr, w_hr),
            mode=self.soft_hr_up_mode if self.soft_hr_up_mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area') else 'bilinear',
            align_corners=False,
        ).squeeze(0)

        return hr_soft

    def _build_hr_down(self, hr_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build hr_down_img and hr_down_up_img from hr_img.

        Args:
            hr_img: High-resolution image tensor of shape (C, H_hr, W_hr).

        Returns:
            A tuple of (hr_down_img, hr_down_up_img), both torch.Tensor.
            hr_down_img has shape (C, H_lr, W_lr) where H_lr = H_hr / self.multiplier.
            hr_down_up_img has shape (C, H_hr, W_hr).
        """
        c, h_hr, w_hr = hr_img.shape
        h_lr = h_hr // self.multiplier
        w_lr = w_hr // self.multiplier

        # 1) Gaussian blur on HR image
        hr_blurred = _gaussian_blur_2d(hr_img, self.gaussian_kernel, self.gaussian_sigma)

        # 2) Downsample to LR size
        hr_down = F.interpolate(
            hr_blurred.unsqueeze(0),
            size=(h_lr, w_lr),
            mode=self.down_mode if self.down_mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area') else 'area',
            align_corners=None if self.down_mode == 'area' else False,
        ).squeeze(0)

        # 3) Upsample back to HR size
        hr_down_up = F.interpolate(
            hr_down.unsqueeze(0),
            size=(h_hr, w_hr),
            mode=self.up_mode if self.up_mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area') else 'bilinear',
            align_corners=False,
        ).squeeze(0)

        return hr_down, hr_down_up

    def __getitem__(self, index):
        """
        Returns:
            A tuple:
                lr_img: Low-resolution crop tensor, shape (C, lr_crop_size, lr_crop_size).
                hr_down_img: HR-derived low-resolution tensor, shape (C, lr_crop_size, lr_crop_size).
                hr_down_up_img: Upsampled hr_down_img tensor, shape (C, hr_crop_size, hr_crop_size).
                hr_img: Cropped high-resolution tensor, shape (C, hr_crop_size, hr_crop_size).
                hr_proj: Projection metadata from HR file.
                hr_geo: Geotransform metadata from HR file.
        """

        # Read paired low-resolution and high-resolution images.
        lr_img_path = self.lr_img_paths[index]
        lr_img, lr_proj, lr_geo = Tif_Read_and_Write().Tif_Read(lr_img_path)
        
        hr_img_path = self.hr_img_paths[index]
        hr_img, hr_proj, hr_geo = Tif_Read_and_Write().Tif_Read(hr_img_path)

        lr_img_name = os.path.basename(lr_img_path)
        hr_img_name = os.path.basename(hr_img_path)
        if lr_img_name != hr_img_name and self.dataset_layout == "legacy_nested_pairs":
            lr_img_infolist = lr_img_name.split('_')
            hr_img_infolist = hr_img_name.split('_')
            if len(lr_img_infolist) > 11 and len(hr_img_infolist) > 4:
                lr_img_date = lr_img_infolist[4]
                lr_img_lon = lr_img_infolist[9]
                lr_img_lat = lr_img_infolist[11]
                hr_img_date = hr_img_infolist[4]
                hr_img_lon = hr_img_infolist[2].replace('E', '')
                hr_img_lat = hr_img_infolist[3].replace('N', '')
                if lr_img_date != hr_img_date:
                    print('Warning, Date Mismatch !!!')
                if lr_img_lon != hr_img_lon:
                    print('Warning, Lon Mismatch !!!')
                if lr_img_lat != hr_img_lat:
                    print('Warning, Lat Mismatch !!!')
            else:
                print(f'Warning, Pair Name Mismatch !!! {lr_img_name} vs {hr_img_name}')

        if lr_img.ndim != 3 or hr_img.ndim != 3:
            raise ValueError(
                f"Expected CHW tensors from TIFF reader, got lr shape {getattr(lr_img, 'shape', None)} "
                f"and hr shape {getattr(hr_img, 'shape', None)}."
            )
        if int(lr_img.shape[0]) != self.input_channels or int(hr_img.shape[0]) != self.input_channels:
            raise ValueError(
                f"Channel mismatch for dataset sample {lr_img_name}: expected {self.input_channels} channels, "
                f"got lr={lr_img.shape[0]}, hr={hr_img.shape[0]}."
            )

        # Use random cropping for train and center cropping for eval/test.
        _, lr_height, lr_width = lr_img.shape
        _, hr_height, hr_width = hr_img.shape
        if lr_height < self.lr_crop_size or lr_width < self.lr_crop_size:
            raise ValueError(
                f"LR crop {self.lr_crop_size} exceeds LR image size {(lr_height, lr_width)} for {lr_img_name}."
            )
        if hr_height < self.hr_crop_size or hr_width < self.hr_crop_size:
            raise ValueError(
                f"HR crop {self.hr_crop_size} exceeds HR image size {(hr_height, hr_width)} for {hr_img_name}."
            )
        if self.is_train:
            lr_height_idx = random.randint(0, lr_height - self.lr_crop_size)
            lr_width_idx = random.randint(0, lr_width - self.lr_crop_size)
        else:
            lr_height_idx = max((lr_height - self.lr_crop_size) // 2, 0)
            lr_width_idx = max((lr_width - self.lr_crop_size) // 2, 0)
        hr_height_idx = lr_height_idx * self.multiplier
        hr_width_idx = lr_width_idx * self.multiplier

        lr_img = lr_img[:, lr_height_idx: lr_height_idx + self.lr_crop_size, lr_width_idx: lr_width_idx + self.lr_crop_size]
        hr_img = hr_img[:, hr_height_idx: hr_height_idx + self.hr_crop_size, hr_width_idx: hr_width_idx + self.hr_crop_size]
        if lr_img.size == 0 or hr_img.size == 0:
            raise ValueError(
                "Empty crop generated. "
                f"lr shape={lr_img.shape}, hr shape={hr_img.shape}, "
                f"scale={self.multiplier}, lr_crop={self.lr_crop_size}, hr_crop={self.hr_crop_size}, "
                f"lr_size={(lr_height, lr_width)}, hr_size={(hr_height, hr_width)}, "
                f"lr_idx={(lr_height_idx, lr_width_idx)}, hr_idx={(hr_height_idx, hr_width_idx)}, "
                f"files=({lr_img_name}, {hr_img_name})."
            )
        lr_img = np.nan_to_num(
            lr_img,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=False,
        ).astype(np.float32, copy=False)
        hr_img = np.nan_to_num(
            hr_img,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=False,
        ).astype(np.float32, copy=False)
        if self.scale_lr_to_unit_interval:
            lr_img = lr_img / self.lr_scale_divisor
        if self.scale_hr_to_unit_interval:
            hr_img = hr_img / self.hr_scale_divisor

        # Skip invalid samples with all-zero content.
        if np.max(lr_img) == 0.0 or np.max(hr_img) == 0.0:
            return self.__getitem__((index + 1) % len(self.hr_img_paths))

        lr_img = np.transpose(lr_img, (1, 2, 0))
        hr_img = np.transpose(hr_img, (1, 2, 0))

        # Place LR crop into an HR-sized canvas before augmentation.
        cut_idx_0 = self.hr_crop_size // 2 - self.lr_crop_size // 2
        cut_idx_1 = self.hr_crop_size // 2 + self.lr_crop_size // 2

        placeholder_map = np.full_like(hr_img, np.nan, dtype=np.float32)
        placeholder_map[cut_idx_0:cut_idx_1, cut_idx_0:cut_idx_1, :] = lr_img
        lr_img = placeholder_map

        if self.is_train:
            transforms = self.train_transforms
        else:
            transforms = self.test_transforms

        if transforms is not None:
            augmentations = transforms(image=lr_img, image0=hr_img)
            lr_img, hr_img = augmentations['image'], augmentations['image0']
        else:
            lr_img = _to_chw_tensor(
                np.nan_to_num(lr_img, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            )
            hr_img = _to_chw_tensor(
                np.nan_to_num(hr_img, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            )

        # After augmentation, crop LR tensor back to its original window.
        lr_img = lr_img[:, cut_idx_0:cut_idx_1, cut_idx_0:cut_idx_1]

        # Optional Gaussian blur on LR before upsampling
        if self.lr_blur_enabled and self.lr_gaussian_kernel > 0 and self.lr_gaussian_sigma > 0.0:
            lr_img = _gaussian_blur_2d(lr_img, self.lr_gaussian_kernel, self.lr_gaussian_sigma)

        # Upsample LR image to HR spatial size for the diffusion condition branch.
        lr_up_img = F.interpolate(
            lr_img.unsqueeze(0),
            size=hr_img.shape[-2:],
            mode=self.lr_up_mode if self.lr_up_mode in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area') else 'bicubic',
            align_corners=False,
        ).squeeze(0)

        # Build HR-derived low-resolution pair from augmented HR image (always from ORIGINAL hr_img).
        hr_down_img, hr_down_up_img = self._build_hr_down(hr_img)

        # Optional: replace hr_img with softened target (Option A)
        if self.soft_hr_blur_enabled:
            hr_img = self._build_soft_hr_target(hr_img)

        return lr_img, lr_up_img, hr_down_img, hr_down_up_img, hr_img, hr_proj, hr_geo


def validate_hr_image_sizes_multiple_of_8(
    root_dir: str,
    split_name: str | None,
    *,
    multiple: int = 8,
    max_report_items: int = 10,
) -> None:
    """Preflight check: ensure all HR images under the given split have dimensions divisible by 8.

    Args:
        root_dir: Dataset root directory.
        split_name: Split name (e.g., 'train', 'valid', 'draw'). None or empty disables flat-split lookup.
        multiple: The divisor to check against (default 8 for VAE 8x downsample).
        max_report_items: Maximum number of offending files to list in the error message.

    Raises:
        ValueError: If any HR image has height or width not divisible by ``multiple``.
    """
    root_dir = os.fspath(root_dir)
    split_name = None if split_name in (None, "", "none", "None") else str(split_name)

    flat_pairs = _collect_flat_split_pairs(root_dir=root_dir, split_name=split_name)
    if flat_pairs is not None:
        _, hr_paths = flat_pairs
    else:
        _, hr_paths = _collect_legacy_nested_pairs(root_dir=root_dir)

    invalid_items: list[tuple[str, int, int]] = []
    for hr_path in hr_paths:
        ds = gdal.Open(hr_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Failed to open HR image with GDAL: {hr_path}")
        height = ds.RasterYSize
        width = ds.RasterXSize
        ds = None
        if height % multiple != 0 or width % multiple != 0:
            invalid_items.append((hr_path, height, width))

    if invalid_items:
        examples = invalid_items[:max_report_items]
        example_text = "; ".join(
            f"{path} -> ({h}, {w})" for path, h, w in examples
        )
        raise ValueError(
            f"HR image size validation failed: expected both height and width to be multiples of {multiple}, "
            f"but found {len(invalid_items)} invalid file(s). "
            f"Examples: {example_text}"
        )


def _build_visual_check_dataset(dataset_yaml: Path, split: str):
    os.environ["CSUA_LDM_DATASET_YAML_OVERRIDE"] = str(dataset_yaml.resolve())
    import Main_Model.CSUA_LDM_Config as cfg

    root_key, split_key = {
        "train": ("rootdir_list_forTrain", "split_name_forTrain"),
        "valid": ("rootdir_list_forValid", "split_name_forValid"),
        "test": ("rootdir_list_forTest", "split_name_forTest"),
        "draw": ("rootdir_list_forDraw", "split_name_forDraw"),
    }[split]

    dataset = SuperResolutionDataset(
        root_dir=cfg.CSUA_LDM_DATASET_DIRS[root_key],
        is_train=False,
        split_name=cfg.CSUA_LDM_DATASET_DIRS[split_key],
        vis_band_order=cfg.CSUA_LDM_VIS_BAND_ORDER,
        superres_scale=cfg.CSUA_LDM_SUPERRES_SCALE,
        lr_crop_size=cfg.CSUA_LDM_LR_CROP_SIZE,
        hr_crop_size=cfg.CSUA_LDM_HR_CROP_SIZE,
        input_channels=cfg.CSUA_LDM_INPUT_CHANNELS,
        hr_degradation_gaussian_kernel=cfg.CSUA_LDM_HR_DEGRADATION_GAUSSIAN_KERNEL,
        hr_degradation_gaussian_sigma=cfg.CSUA_LDM_HR_DEGRADATION_GAUSSIAN_SIGMA,
        hr_degradation_down_mode=cfg.CSUA_LDM_HR_DEGRADATION_DOWN_MODE,
        hr_degradation_up_mode=cfg.CSUA_LDM_HR_DEGRADATION_UP_MODE,
        soft_hr_gaussian_blur_enabled=cfg.CSUA_LDM_HR_GAUSSIAN_BLUR_ENABLED,
        soft_hr_gaussian_kernel=cfg.CSUA_LDM_HR_GAUSSIAN_KERNEL,
        soft_hr_gaussian_sigma=cfg.CSUA_LDM_HR_GAUSSIAN_SIGMA,
        soft_hr_down_mode=cfg.CSUA_LDM_HR_DOWN_MODE,
        soft_hr_up_mode=cfg.CSUA_LDM_HR_UP_MODE,
        lr_up_mode=cfg.CSUA_LDM_LR_UP_MODE,
        lr_gaussian_blur_enabled=cfg.CSUA_LDM_LR_GAUSSIAN_BLUR_ENABLED,
        lr_gaussian_kernel=cfg.CSUA_LDM_LR_GAUSSIAN_KERNEL,
        lr_gaussian_sigma=cfg.CSUA_LDM_LR_GAUSSIAN_SIGMA,
        **cfg.CSUA_LDM_DATASET_RAW_SCALING_KWARGS,
    )
    return dataset, cfg.CSUA_LDM_DATASET_NAME


def _to_visual_image(image: torch.Tensor, band_order: list[int]) -> np.ndarray:
    image = image[band_order].detach().cpu().permute(1, 2, 0).numpy()
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    lower = np.percentile(image, 2.0, axis=(0, 1), keepdims=True)
    upper = np.percentile(image, 98.0, axis=(0, 1), keepdims=True)
    image = np.clip((image - lower) / np.maximum(upper - lower, 1e-8), 0.0, 1.0)
    return np.sqrt(image)


def _select_visual_check_indices(
    indices_text: str | None,
    dataset_size: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    if indices_text:
        indices = [int(value.strip()) for value in indices_text.split(",") if value.strip()]
    else:
        count = min(max(num_samples, 1), dataset_size)
        indices = sorted(random.Random(seed).sample(range(dataset_size), count))
    if not indices or any(index < 0 or index >= dataset_size for index in indices):
        raise ValueError(f"Sample indexes must be within 0-{dataset_size - 1}.")
    return list(dict.fromkeys(indices))


def _print_unmatched_file_counts(dataset: SuperResolutionDataset) -> None:
    if dataset.dataset_layout != "flat_split_dirs":
        return
    lr_dir = Path(dataset.lr_img_paths[0]).parent
    hr_dir = Path(dataset.hr_img_paths[0]).parent
    lr_names = {path.name for path in lr_dir.iterdir() if _is_tif_file(path.name)}
    hr_names = {path.name for path in hr_dir.iterdir() if _is_tif_file(path.name)}
    print(f"Unmatched LR files: {len(lr_names - hr_names)}")
    print(f"Unmatched HR files: {len(hr_names - lr_names)}")


def _parse_visual_check_args() -> argparse.Namespace:
    default_yaml = os.environ.get(
        "CSUA_LDM_DATASET_YAML_OVERRIDE",
        "configs/datasets/sen2naip.yaml",
    )
    parser = argparse.ArgumentParser(
        description="Visually inspect paired LR and HR samples."
    )
    parser.add_argument("--dataset-config", type=Path, default=Path(default_yaml))
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test", "draw"),
        default="train",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--indices", help="Comma-separated indexes, for example 0,12,35.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively in addition to saving it.",
    )
    return parser.parse_args()


def _run_visual_pair_check(args: argparse.Namespace) -> None:
    if not args.show:
        plt.switch_backend("Agg")

    dataset, dataset_name = _build_visual_check_dataset(
        args.dataset_config,
        args.split,
    )
    indices = _select_visual_check_indices(
        args.indices,
        len(dataset),
        args.num_samples,
        args.seed,
    )

    print(f"Dataset: {dataset_name} | split: {args.split}")
    print(f"Matched pairs: {len(dataset)}")
    _print_unmatched_file_counts(dataset)

    figure, axes = plt.subplots(
        len(indices),
        5,
        figsize=(18.0, 3.4 * len(indices)),
        squeeze=False,
    )
    titles = ("LR", "LR up", "HR down", "HR down-up", "HR")

    for row, index in enumerate(indices):
        sample = dataset[index]
        images = sample[:5]
        lr_name = Path(dataset.lr_img_paths[index]).name
        hr_name = Path(dataset.hr_img_paths[index]).name
        print(f"[{index}] LR={lr_name} | HR={hr_name}")

        for column, (title, image) in enumerate(zip(titles, images)):
            axes[row, column].imshow(
                _to_visual_image(image, dataset.vis_band_order)
            )
            axes[row, column].set_title(
                f"[{index}] {title}\n{lr_name if column < 2 else hr_name}"
            )
            axes[row, column].axis("off")

    output_path = args.output
    if output_path is None:
        output_path = Path("tmp") / f"{dataset_name}_{args.split}_pair_check.jpg"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved visual check: {output_path}")

    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    _run_visual_pair_check(_parse_visual_check_args())
