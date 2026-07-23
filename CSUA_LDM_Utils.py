import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from osgeo import osr
from torch.nn import Module
from torch.optim import Optimizer
from torchvision.utils import make_grid

from CSUA_LDM_TifReader import Tif_Read_and_Write

min_loss = float("inf")
best_epoch_idx = -1
patience_counter = 0
patience_counter_after_min_lr = 0


def geo_to_numpy_array(geo_item):
    if isinstance(geo_item, torch.Tensor):
        return geo_item.detach().cpu().numpy()
    return np.asarray(geo_item)



def get_training_monitor_state() -> Dict[str, float]:
    return {
        "min_loss": float(min_loss),
        "best_epoch_idx": int(best_epoch_idx),
        "patience_counter": int(patience_counter),
        "patience_counter_after_min_lr": int(patience_counter_after_min_lr),
    }


def _format_loss_for_log(value, precision: int = 6) -> str:
    """Format nested loss structures for stable console logging."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{repr(k)}: {_format_loss_for_log(v, precision)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
        return left + ", ".join(_format_loss_for_log(v, precision) for v in value) + right
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return repr(value)


def save_model(
    ckpt_path: str,
    epoch: int,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.Optimizer] = None,
    loss_dict: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    scaler: Optional[Any] = None,
):
    """
    Save training checkpoint.

    Args:
        ckpt_path: path to save checkpoint file (.pt / .pth).
        epoch: current epoch (int).
        model: model instance.
        optimizer: optimizer instance.
        scheduler: scheduler instance.
        scaler: AMP GradScaler instance.
        loss_dict: dictionary of losses.
        extra: any additional metadata.
        verbose: whether to print save info.

    Returns:
        ckpt_path
    """

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    extra = dict(extra or {})
    extra.setdefault("rng_state", stash_rng_state())

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict() if model else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss_dict": loss_dict,
        "extra": extra,
    }

    torch.save(checkpoint, ckpt_path)

    if verbose:
        print(f"[Checkpoint] Saved at epoch {epoch}")

    return ckpt_path


def load_model(    
    save_path: Union[str, Path],
    model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Optimizer] = None,
    resume_training: bool = True,
    preload_only: bool = False,
    skip_prefix: Optional[Union[str, Iterable[str]]] = None,
    load_ema: bool = False,
    scaler: Optional[Any] = None,
    strict: bool = True,
    ema_model: Optional[Module] = None,
    ema_decay: Optional[float] = None,
    map_location: Optional[Union[str, torch.device]] = None,
):
    """
    Load checkpoint and (optionally) restore model/optimizer.

    Args:
        save_path: checkpoint path (.pth / .pt)
        model: model instance to load weights into
        optimizer: optimizer instance to load state into
        scheduler: scheduler instance to load state into
        scaler: AMP GradScaler instance to load state into (optional)
        resume_training: whether you intend to resume training (print resume info)
        preload_only: whether it's just for preloading weights (no resume message)
        skip_prefix: prefix or prefixes to exclude when loading model weights
        load_ema: if True, load model weights from checkpoint["extra"]["ema_model_state_dict"].
                  if False, load from checkpoint["model_state_dict"].
        ema_model: optional EMA model to restore from checkpoint["extra"]["ema_model_state_dict"].
        ema_decay: optional EMA decay value; if checkpoint contains "ema_decay" in extra, it will be updated.
        map_location: optional torch.load map_location override. If omitted, the checkpoint
                     is restored onto the current model device when model is provided,
                     otherwise restored onto CPU for safe cross-device loading.

    Returns:
        (model, optimizer, scheduler, epoch_idx, loss_dict, extra, ema_decay)
    """
    save_path = Path(save_path)

    # ---- validate flags ----
    if not resume_training and preload_only:
        raise ValueError("Invalid flags: resume_training=False but preload_only=True")

    load_map_location: Union[str, torch.device]
    if map_location is not None:
        load_map_location = map_location
    elif model is not None:
        try:
            load_map_location = next(model.parameters()).device
        except StopIteration:
            try:
                load_map_location = next(model.buffers()).device
            except StopIteration:
                load_map_location = "cpu"
    else:
        load_map_location = "cpu"

    checkpoint = torch.load(
        save_path,
        map_location=load_map_location,
        weights_only=False,
    )

    # ---- read meta info safely ----
    epoch_idx = int(checkpoint.get("epoch", -1))
    loss_dict = checkpoint.get("loss_dict", {})
    extra = checkpoint.get("extra", {}) or {}
    if resume_training and not preload_only and extra.get("rng_state") is not None:
        restore_rng_state(extra.get("rng_state"))

    # ---- load model ----
    if model is not None:
        if load_ema:
            state = extra.get("ema_model_state_dict") if isinstance(extra, dict) else None
            if state is None:
                raise KeyError("Checkpoint missing key: 'extra.ema_model_state_dict' (load_ema=True)")
        else:
            state = checkpoint.get("model_state_dict")
            if state is None:
                raise KeyError("Checkpoint missing key: 'model_state_dict'")

        if skip_prefix is not None:
            # preload with partial weights
            prefixes = (skip_prefix,) if isinstance(skip_prefix, str) else tuple(skip_prefix)
            filtered = {k: v for k, v in state.items() if not k.startswith(prefixes)}
            # when filtering, strict must be False
            model.load_state_dict(filtered, strict=False)
        else:
            model.load_state_dict(state, strict=bool(strict))

    # ---- load optimizer ----
    if optimizer is not None:
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is None:
            raise KeyError("Checkpoint missing key: 'optimizer_state_dict'")
        optimizer.load_state_dict(opt_state)

    # ---- load scheduler ----
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            raise KeyError("Checkpoint missing key: 'scheduler_state_dict'")
        scheduler.load_state_dict(scheduler_state)

    # ---- load grad scaler when available ----
    if scaler is not None:
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is None:
            if resume_training and not preload_only:
                print("[Checkpoint] scaler_state_dict missing; using fresh GradScaler state.")
        else:
            scaler.load_state_dict(scaler_state)

    # ---- load ema model (optional) ----
    if ema_model is not None:
        ema_state = extra.get("ema_model_state_dict", None) if isinstance(extra, dict) else None
        if ema_state is not None:
            ema_model.load_state_dict(ema_state, strict=bool(strict))
            if ema_decay is not None:
                ema_decay = float(extra.get("ema_decay", ema_decay))
            print(f"Loaded EMA state from checkpoint. ema_decay={ema_decay}")
        else:
            ema_model.load_state_dict(model.state_dict(), strict=bool(strict))
            print("EMA state not found in checkpoint; initialized EMA from current model.")

    # ---- logging ----
    if resume_training and not preload_only:
        # epoch_idx: last finished epoch, so resume from epoch_idx + 1
        formatted_loss = _format_loss_for_log(loss_dict, precision=6)
        print(f"Resume from epoch: [{epoch_idx + 1}] (last checkpoint epoch_idx={epoch_idx}) | loss: {formatted_loss}")
    elif resume_training and preload_only:
        print(f"Preloading weights from: {save_path.stem}")
    elif not resume_training and preload_only:
        raise ValueError("Invalid flags: resume_training=False but preload_only=True")
    
    return model, optimizer, scheduler, epoch_idx, loss_dict, extra, ema_decay


@torch.no_grad()
def build_ema_model(model):
    """Create an EMA shadow model initialized from the current model.

    Args:
        model: Source model.

    Returns:
        A detached EMA model with gradients disabled.
    """
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    """Update EMA model parameters using current model parameters.

    Args:
        ema_model: Exponential moving average model to update.
        model: Current training model.
        decay: EMA decay factor in [0, 1).

    Returns:
        None.
    """
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()

    for k, ema_v in ema_state.items():
        model_v = model_state[k]
        if not torch.is_floating_point(ema_v):
            ema_v.copy_(model_v)
            continue
        ema_v.mul_(decay).add_(model_v, alpha=1.0 - decay)


def stash_rng_state():
    """Save RNG and cudnn backend states for temporary deterministic eval."""
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        rng_state["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        rng_state["deterministic_algorithms_warn_only"] = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
    return rng_state


def _coerce_rng_byte_tensor(state: Any) -> torch.Tensor:
    """Normalize a saved RNG state to a CPU torch.ByteTensor.

    Checkpoints may have their RNG tensors remapped onto CUDA when loaded
    with a model-device `map_location`. `torch.set_rng_state` and
    `torch.cuda.set_rng_state_all` expect CPU byte tensors, so we coerce them
    back here for robust resume across devices.
    """
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")


def restore_rng_state(rng_state):
    """Restore RNG states saved by `stash_rng_state`.

    Each state is restored only when present, which also supports checkpoints
    containing partial RNG metadata.
    """
    if rng_state is None:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "cpu" in rng_state:
        torch.set_rng_state(_coerce_rng_byte_tensor(rng_state["cpu"]))
    if torch.cuda.is_available() and "cuda" in rng_state:
        cuda_states = [_coerce_rng_byte_tensor(state) for state in rng_state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    if "cudnn_deterministic" in rng_state:
        torch.backends.cudnn.deterministic = bool(rng_state["cudnn_deterministic"])
    if "cudnn_benchmark" in rng_state:
        torch.backends.cudnn.benchmark = bool(rng_state["cudnn_benchmark"])
    if hasattr(torch, "use_deterministic_algorithms") and "deterministic_algorithms_enabled" in rng_state:
        enabled = bool(rng_state["deterministic_algorithms_enabled"])
        warn_only = bool(rng_state.get("deterministic_algorithms_warn_only", False))
        try:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(enabled)


def seed_dataloader_worker(worker_id: int):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)



def iter_microbatch_slices(batch_size: int, microbatch_size: int):
    batch_size = int(batch_size)
    step = max(int(microbatch_size), 1)
    for start in range(0, batch_size, step):
        end = min(start + step, batch_size)
        yield start, end


def ensure_batched_4d(imgs: torch.Tensor) -> torch.Tensor:
    """Convert one image tensor to a batch or validate an existing 4D batch."""
    if imgs.dim() == 3:
        return imgs.unsqueeze(0)
    if imgs.dim() != 4:
        raise ValueError(f"Expected 3D/4D tensor, got shape={tuple(imgs.shape)}.")
    return imgs


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string and validate CUDA availability when requested."""
    device_str = str(device_str).strip().lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    try:
        return torch.device(device_str)
    except RuntimeError as exc:
        raise RuntimeError(f"Unsupported device string: {device_str}") from exc



def training_monitor(
    epoch_idx: int,
    min_lr: float,
    current_lr: float,
    current_loss: float,
    patience_threshold: int,
    resume_train: bool = False,
    load_dict: dict | None = None,
):
    """
    Returns:
        stop_training (bool): whether to stop training
        save_model (bool): whether to save current model as best
        patience_counter (int)
        patience_counter_after_min_lr (int)
    """

    global min_loss, best_epoch_idx, patience_counter, patience_counter_after_min_lr

    # ---- 1) Handle checkpoint resume ----
    if resume_train:
        if load_dict is None:
            print("!!! Resume_Train is True but load_dict is None !!!")
            min_loss = float("inf")
            best_epoch_idx = epoch_idx
            patience_counter = 0
            patience_counter_after_min_lr = 0
        else:
            min_loss = load_dict.get("min_loss", float("inf"))
            best_epoch_idx = load_dict.get("best_epoch_idx", epoch_idx)
            patience_counter = int(load_dict.get("patience_counter", 0))
            patience_counter_after_min_lr = int(load_dict.get("patience_counter_after_min_lr", 0))
        return False, False, patience_counter, patience_counter_after_min_lr

    # ---- 2) First-epoch initialization ----
    if epoch_idx == 0:
        min_loss = current_loss
        best_epoch_idx = epoch_idx
        patience_counter = 0
        patience_counter_after_min_lr = 0
        return False, True, patience_counter, patience_counter_after_min_lr

    # ---- 3) Check for improvement ----
    improved = current_loss < min_loss
    lr_reached_min = current_lr <= min_lr

    if improved:
        min_loss = current_loss
        best_epoch_idx = epoch_idx
        patience_counter = 0
        patience_counter_after_min_lr = 0
        return False, True, patience_counter, patience_counter_after_min_lr

    # No improvement: counter logic
    patience_counter += 1

    if not lr_reached_min:
        # LR not at minimum: only accumulate patience_counter
        return False, False, patience_counter, patience_counter_after_min_lr

    # LR at minimum: additionally accumulate patience_counter_after_min_lr
    patience_counter_after_min_lr += 1

    remaining = patience_threshold - patience_counter_after_min_lr
    print(
        f"[TrainingMonitor] lr reached min={min_lr:.8f}; "
        f"continuing for {max(0, remaining)} more epoch(s) before stopping."
    )

    if patience_counter_after_min_lr > patience_threshold:
        print(
            f"[TrainingMonitor] no improvement after {patience_threshold} extra epoch(s) at min lr; stopping."
        )
        return True, False, patience_counter, patience_counter_after_min_lr

    return False, False, patience_counter, patience_counter_after_min_lr



def save_rgb_datas(
    imgs,
    nrow,
    savepath=None,
    format=None,
    is_showminmax=False,
    is_makegrid=True,
    use_sqrt=False,
    **kwargs,
):
    """
    Save RGB image(s) from a torch tensor.

    Args:
        imgs: torch.Tensor
            - if is_makegrid=True: expected shape (N, C, H, W)
            - if is_makegrid=False: expected shape (N, C, H, W)
        nrow: int
            Number of images per row when make_grid is used.
        savepath: str | None
            If provided, save image(s) to this path.
        format: str | None
            Passed to PIL.Image.save(format=...).
        is_showminmax: bool
            Whether to print min/max of imgs.
        is_makegrid: bool
            Whether to create a grid via torchvision.utils.make_grid.
        **kwargs:
            prompt_strs: Optional[list[str]] used for naming multiple files.

    Returns:
        - if is_makegrid=True: np.ndarray (H, W, 3), dtype=uint8
        - else: np.ndarray (N, H, W, 3), dtype=uint8
    """
    if is_showminmax:
        print(f"RGB_min: {torch.min(imgs)} | RGB_max: {torch.max(imgs)}")

    if is_makegrid:
        img_grid = make_grid(imgs, nrow=nrow)  # (C, H, W)

        if use_sqrt:
            img_grid = torch.sqrt(img_grid.clamp(0.0, 1.0))

        # (C, H, W) -> (H, W, C) and to uint8 numpy
        img_grid = (img_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())

        # Ensure RGB channel order
        img_grid = img_grid[:, :, [0, 1, 2]]

        im = Image.fromarray(img_grid)
        if savepath is not None:
            im.save(savepath, format=format)

        return img_grid

    # is_makegrid == False: save each image separately if needed
    prompt_strs = kwargs.get("prompt_strs", None)
        
    if use_sqrt:
        imgs = torch.sqrt(imgs.clamp(0.0, 1.0))

    imgs = (
        imgs.mul(255).add_(0.5).clamp_(0, 255)
        .permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        .to("cpu", torch.uint8).numpy()
    )

    if savepath is not None:
        if len(imgs) == 1:
            im = Image.fromarray(imgs[0])
            im.save(savepath, format=format)
        else:
            for idx, img in enumerate(imgs):
                im = Image.fromarray(img)

                if prompt_strs is not None:
                    prompt_str = prompt_strs[idx]
                    savepath_separate = savepath.replace(
                        ".jpg", f"_p{idx + 1}_{prompt_str}.jpg"
                    )
                else:
                    savepath_separate = savepath.replace(".jpg", f"_p{idx + 1}.jpg")

                im.save(savepath_separate, format=format)

    return imgs


def save_tif_datas(
    datas,
    projections=None,
    geotransforms=None,
    img_sources_num=4,
    savepath=None,
    is_showminmax: bool = False,
    **kwargs,
):
    """
    Save tensor/ndarray batches as GeoTIFF(s).
    """

    if is_showminmax:
        print(f"TIF_min: {torch.min(datas)} | TIF_max: {torch.max(datas)}")

    datas = datas.to("cpu").numpy()
    batch_size = int(datas.shape[0] / img_sources_num)

    # -------- geotransforms --------
    if geotransforms is None:
        regen_geotransforms = []
        for _ in range(img_sources_num):
            for _ in range(batch_size):
                top_left_lon = float(random.randint(0, 900000))
                top_left_lat = float(random.randint(0, 9000000))
                pixel_width = 10.0
                pixel_height = -10.0
                regen_geotransforms.append([top_left_lon, pixel_width, 0.0, top_left_lat, 0.0, pixel_height])
    else:
        regen_geotransforms = []
        for _ in range(img_sources_num):
            for idx in range(batch_size):
                regen_geotransforms.append(geotransforms[idx])

    # -------- epsg_codes --------
    if projections is None:
        epsg_codes = []
        for _ in range(img_sources_num):
            for _ in range(batch_size):
                num_1 = [32600, 32700][random.randint(0, 1)]
                num_2 = random.randint(1, 60)
                epsg_codes.append(num_1 + num_2)
    else:
        epsg_codes = []
        for _ in range(img_sources_num):
            for idx in range(batch_size):
                epsg_code = int(osr.SpatialReference(wkt=projections[idx]).GetAttrValue("AUTHORITY", 1))
                epsg_codes.append(epsg_code)

    # -------- prompt_strs --------
    prompt_strs = kwargs.get("prompt_strs", None)

    def _write_single_tif(data: np.ndarray, out_path: str, geotransform, epsg_code: int):
        # Preserve the original logic: only indices 0/3/1/5 are used as write parameters
        top_left_lon = geotransform[0]
        top_left_lat = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]
        Tif_Read_and_Write().Numpy_to_Tif(data, out_path, top_left_lon, top_left_lat, pixel_width, pixel_height, epsg_code, nodata_value=np.nan)

    # -------- Write file (preserve single/multi branch and naming logic) -------
    if len(datas) == 1:
        _write_single_tif(datas[0], savepath, regen_geotransforms[0], epsg_codes[0])
    else:
        for idx, (data, epsg_code, regen_geotransform) in enumerate(zip(datas, epsg_codes, regen_geotransforms)):
            source_idx = idx // batch_size
            individual_idx = idx % batch_size
            if prompt_strs is not None:
                prompt_str = prompt_strs[idx]
                savepath_separate = savepath.replace(".tif", f"_s{source_idx + 1}_i{individual_idx + 1}_{prompt_str}.tif")
            else:
                savepath_separate = savepath.replace(".tif", f"_s{source_idx + 1}_i{individual_idx + 1}.tif")

            _write_single_tif(data, savepath_separate, regen_geotransform, epsg_code)

    # -------- Restore original projections --------
    regen_projections = []
    for epsg_code in epsg_codes:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
        regen_projections.append(srs.ExportToWkt())

    return datas, regen_projections, regen_geotransforms


def save_single_tif(
    tensor: torch.Tensor,
    save_path: str,
    proj: str,
    geo: tuple,
) -> None:
    """Save one C,H,W tensor as a GeoTIFF."""
    data = tensor.detach().cpu().numpy()

    top_left_lon = float(geo[0])
    pixel_width = float(geo[1])
    top_left_lat = float(geo[3])
    pixel_height = float(geo[5])

    try:
        srs = osr.SpatialReference(wkt=proj)
        authority = srs.GetAttrValue("AUTHORITY", 1)
        epsg_code = int(authority) if authority is not None else 4326
    except Exception:
        epsg_code = 4326

    Tif_Read_and_Write().Numpy_to_Tif(
        array_data=data,
        output_path=save_path,
        top_left_lon=top_left_lon,
        top_left_lat=top_left_lat,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        epsg_code=epsg_code,
        nodata_value=np.nan,
    )


def log_info(log_path, text):
    """
    Append one line to a text log file.
    """
    log_path = Path(log_path)
    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write(text + '\n')  # Append the line to the text log



def set_random_seed(seed, deterministic: bool = False):
    """Set Python/NumPy/Torch RNG seeds and optionally enable deterministic eval."""
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    return seed


def denorm_mean(imgs: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    mean_t = mean_t.to(device=imgs.device, dtype=imgs.dtype)
    std_t = std_t.to(device=imgs.device, dtype=imgs.dtype)
    return torch.clamp(imgs * std_t + mean_t, min=0.0, max=1.0)


def denorm_std(stds: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    std_t = std_t.to(device=stds.device, dtype=stds.dtype)
    return stds * std_t


def norm_img(
    img: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> torch.Tensor:
    """Normalize an image tensor with explicitly supplied channel statistics."""
    mean_t = mean_t.to(device=img.device, dtype=img.dtype)
    std_t = std_t.to(device=img.device, dtype=img.dtype)
    return (img - mean_t) / std_t


def normalize_uncertainty_for_rgb_vis(unc: torch.Tensor, eps: float = 1e-8, per_image: bool = True) -> torch.Tensor:
    if per_image:
        max_val = unc.amax(dim=(1, 2, 3), keepdim=True)
    else:
        max_val = unc.amax()
    unc_vis = unc / (max_val + eps)
    return torch.clamp(unc_vis, min=0.0, max=1.0)


def soften_positive_logvar(logvar: torch.Tensor | None, soften_lambda: float) -> torch.Tensor | None:
    if logvar is None:
        return None

    soften_lambda = float(soften_lambda)
    if soften_lambda == 1.0:
        return logvar

    softened_logvar = logvar.clone()
    positive_mask = softened_logvar > 0
    softened_logvar[positive_mask] = softened_logvar[positive_mask] * soften_lambda
    return softened_logvar


def sigma_from_logvar(
    logvar: torch.Tensor | None,
    soften_lambda: float,
    scale: torch.Tensor | float | None = None,
) -> torch.Tensor | None:
    softened_logvar = soften_positive_logvar(logvar, soften_lambda)
    if softened_logvar is None:
        return None

    sigma = torch.exp(0.5 * softened_logvar)
    if scale is not None:
        scale_tensor = torch.as_tensor(scale, device=sigma.device, dtype=sigma.dtype)
        sigma = sigma * scale_tensor
    return sigma


def sigma_from_variance(
    variance: torch.Tensor | None,
    soften_lambda: float,
    min_variance: float = 1e-8,
) -> torch.Tensor | None:
    if variance is None:
        return None

    logvar = torch.log(torch.clamp(variance, min=min_variance))
    return sigma_from_logvar(logvar, soften_lambda=soften_lambda)


# ===================== Decoder Uncertainty Propagation =====================


def _clamp_var(var: torch.Tensor, min_val: float = 0.0, max_val: float = 1e6) -> torch.Tensor:
    return torch.clamp(torch.nan_to_num(var, nan=min_val, posinf=max_val, neginf=min_val), min=min_val, max=max_val)


class VAEDecoderPropagator:
    """
    Wrapper around VAE decode that computes both pixel mean and total pixel uncertainty.

    Total pixel variance = J * latent_var * J^T + decoder_cond_var,
    where J is the local Jacobian of the decoder mean path.
    """

    def __init__(self, vae_model):
        self.vae = vae_model
        self.content_ch = getattr(vae_model, "content_channels", vae_model.z_channels)
        self.has_disentangled_style = (
            hasattr(vae_model, "style_channels")
            and getattr(vae_model, "style_channels", 0) > 0
        )

    @torch.no_grad()
    def decode(self, z_content, z_style, logvar_content=None, logvar_style=None):
        """Decode latent to pixel mean and optional conditional variance."""
        if logvar_content is None:
            logvar_content = torch.zeros_like(z_content)
        if logvar_style is None:
            logvar_style = torch.zeros_like(z_style)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            if self.has_disentangled_style:
                dec_out = self.vae.decode(z_content, z_style, logvar_content, logvar_style)
            else:
                # NoDisent VAE decodes a joint latent tensor.
                z_full = z_content if z_style.shape[1] == 0 else torch.cat([z_content, z_style], dim=1)
                logvar_full = logvar_content if logvar_style.shape[1] == 0 else torch.cat([logvar_content, logvar_style], dim=1)
                dec_out = self.vae.decode(z_full, logvar_full)
            pred_mean = dec_out[0]
            pred_var = dec_out[1] if len(dec_out) >= 2 else None
        return pred_mean, pred_var

    @staticmethod
    def _conv2d_var(input_var: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        weight_sq = conv.weight.to(device=input_var.device, dtype=input_var.dtype).pow(2)
        out = F.conv2d(
            input_var,
            weight_sq,
            bias=None,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )
        return _clamp_var(out)

    @staticmethod
    def _silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    @staticmethod
    def _propagate_groupnorm_mean_var(mean: torch.Tensor, var: torch.Tensor, norm: nn.GroupNorm):
        mean_out = F.group_norm(mean, norm.num_groups, norm.weight, norm.bias, norm.eps)
        bsz, ch, h, w = mean.shape
        groups = int(norm.num_groups)
        mean_view = mean.reshape(bsz, groups, ch // groups, h, w)
        group_var = mean_view.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        scale = torch.rsqrt(group_var + norm.eps)
        scale = scale.expand(bsz, groups, ch // groups, 1, 1).reshape(bsz, ch, 1, 1)
        if norm.weight is not None:
            scale = scale * norm.weight.view(1, ch, 1, 1).to(device=mean.device, dtype=mean.dtype)
        var_out = _clamp_var(var * scale.pow(2))
        return mean_out, var_out

    def _propagate_module_mean_var(self, module, mean: torch.Tensor, var: torch.Tensor):
        if isinstance(module, nn.Sequential):
            mean_out, var_out = mean, var
            for sub in module:
                mean_out, var_out = self._propagate_module_mean_var(sub, mean_out, var_out)
            return mean_out, var_out

        if isinstance(module, nn.Conv2d):
            return module(mean), self._conv2d_var(var, module)

        if isinstance(module, nn.GroupNorm):
            return self._propagate_groupnorm_mean_var(mean, var, module)

        if isinstance(module, nn.SiLU):
            mean_out = module(mean)
            deriv = self._silu_derivative(mean)
            var_out = _clamp_var(var * deriv.pow(2))
            return mean_out, var_out

        if isinstance(module, nn.Upsample):
            mean_out = module(mean)
            var_out = module(var)
            return mean_out, _clamp_var(var_out)

        raise TypeError(f"Unsupported module for decoder variance propagation: {type(module).__name__}")

    def _propagate_residual_stack(self, mean: torch.Tensor, var: torch.Tensor, seq_first, seq_second, skip_conv):
        main_mean, main_var = self._propagate_module_mean_var(seq_first, mean, var)
        main_mean, main_var = self._propagate_module_mean_var(seq_second, main_mean, main_var)
        skip_mean, skip_var = self._propagate_module_mean_var(skip_conv, mean, var)
        out_mean = main_mean + skip_mean
        out_var = _clamp_var(main_var + skip_var)
        return out_mean, out_var

    def _propagate_mid_block_mean_var(self, block, mean: torch.Tensor, var: torch.Tensor):
        out_mean, out_var = self._propagate_residual_stack(
            mean, var,
            block.resnet_conv_first[0],
            block.resnet_conv_second[0],
            block.residual_input_conv[0],
        )
        for idx in range(block.num_layers):
            out_mean, out_var = self._propagate_residual_stack(
                out_mean, out_var,
                block.resnet_conv_first[idx + 1],
                block.resnet_conv_second[idx + 1],
                block.residual_input_conv[idx + 1],
            )
        return out_mean, out_var

    def _propagate_up_block_mean_var(self, block, mean: torch.Tensor, var: torch.Tensor):
        out_mean, out_var = self._propagate_module_mean_var(block.up_sample_conv, mean, var)
        for idx in range(block.num_layers):
            out_mean, out_var = self._propagate_residual_stack(
                out_mean, out_var,
                block.resnet_conv_first[idx],
                block.resnet_conv_second[idx],
                block.residual_input_conv[idx],
            )
        return out_mean, out_var

    @torch.no_grad()
    def propagate_pixel_uncertainty(self, latent_mean, latent_var, logvar_content=None, logvar_style=None):
        """Propagate latent variance through decoder to obtain total pixel uncertainty."""
        latent_var = _clamp_var(latent_var, min_val=1e-8, max_val=1e6)

        content_ch = self.content_ch
        z_mean_content = latent_mean[:, :content_ch, ...]
        z_var_content = latent_var[:, :content_ch, ...]

        z_proj = getattr(self.vae, "z_content_proj", getattr(self.vae, "z_proj", None))
        decoder_dtype = z_proj.weight.dtype
        z_mean_content = z_mean_content.to(dtype=decoder_dtype)
        z_var_content = z_var_content.to(dtype=decoder_dtype)

        h_mean = z_proj(z_mean_content)
        h_var = self._conv2d_var(z_var_content, z_proj)

        if self.has_disentangled_style:
            z_mean_style = latent_mean[:, content_ch:, ...]
            z_var_style = latent_var[:, content_ch:, ...]
            z_mean_style = z_mean_style.to(dtype=decoder_dtype)
            z_var_style = z_var_style.to(dtype=decoder_dtype)

            h_mean_style = self.vae.z_style_proj(z_mean_style)
            h_var_style = self._conv2d_var(z_var_style, self.vae.z_style_proj)
            h_mean = h_mean + h_mean_style
            h_var = _clamp_var(h_var + h_var_style)

        # FiLM modulation with logvar (mirrors vae.decode())
        if logvar_content is not None:
            logvar_content = logvar_content.to(dtype=decoder_dtype, device=h_mean.device)
            if self.has_disentangled_style:
                if logvar_style is None:
                    logvar_style = torch.zeros_like(z_mean_style)
                else:
                    logvar_style = logvar_style.to(dtype=decoder_dtype, device=h_mean.device)

            if hasattr(self.vae, "logvar_film_proj"):
                logvar_proj = getattr(self.vae, "logvar_content_proj", getattr(self.vae, "logvar_proj", None))
                logvar_feat = logvar_proj(logvar_content)

                if self.has_disentangled_style and logvar_style is not None:
                    logvar_feat = logvar_feat + self.vae.logvar_style_proj(logvar_style)

                gamma, beta = torch.chunk(self.vae.logvar_film_proj(logvar_feat), chunks=2, dim=1)
                film_scale = torch.tanh(self.vae.logvar_film_gate)
                if hasattr(film_scale, "to"):
                    film_scale = film_scale.to(dtype=h_mean.dtype, device=h_mean.device)
                a = 1.0 + film_scale * torch.tanh(gamma)
                b = film_scale * torch.tanh(beta)
                a = a.to(dtype=h_mean.dtype, device=h_mean.device)
                b = b.to(dtype=h_mean.dtype, device=h_mean.device)
                h_mean = h_mean * a + b
                h_var = _clamp_var(h_var * a.pow(2))

        for blk in self.vae.dec_mid_blocks:
            h_mean, h_var = self._propagate_mid_block_mean_var(blk, h_mean, h_var)

        for blk in self.vae.dec_up_blocks:
            h_mean, h_var = self._propagate_up_block_mean_var(blk, h_mean, h_var)

        h_mean, h_var = self._propagate_module_mean_var(self.vae.dec_norm_out, h_mean, h_var)
        h_mean, h_var = self._propagate_module_mean_var(self.vae.act, h_mean, h_var)

        pred_mean = self.vae.dec_conv_mean(h_mean)
        propagated_latent_var = self._conv2d_var(h_var, self.vae.dec_conv_mean)

        # Match the decoder's learned conditional-variance branch exactly.
        var_feat = self.vae.var_norm_out(h_mean)
        var_feat = self.vae.act(var_feat)
        var_feat = self.vae.var_fuse_proj(var_feat)
        var_feat = self.vae.act(var_feat + h_mean)
        raw_var = self.vae.dec_conv_var(var_feat)
        decoder_cond_var, _ = self.vae._variance_from_raw(raw_var)
        decoder_cond_var = _clamp_var(decoder_cond_var, min_val=1e-8, max_val=1e6)

        pixel_uncert_var = _clamp_var(propagated_latent_var + decoder_cond_var, min_val=1e-8, max_val=1e6)
        return {
            "pred_mean": pred_mean,
            "propagated_latent_var_px": propagated_latent_var,
            "decoder_cond_var_px": decoder_cond_var,
            "pixel_uncert_var_px": pixel_uncert_var,
        }

    @torch.no_grad()
    def decode_with_total_uncertainty(self, z_content, z_style, latent_mean, latent_var, logvar_content=None, logvar_style=None):
        """
        Decode the latent representation and return pixel mean and total variance.

        Parameters
        ----------
        z_content, z_style : torch.Tensor
            Latent tensors for content and style (used for VAE decode FiLM modulation).
        latent_mean : torch.Tensor
            Full latent mean (content + style concatenated). Used for uncertainty propagation.
        latent_var : torch.Tensor
            Full latent variance (content + style concatenated).
        logvar_content, logvar_style : torch.Tensor, optional
            Log-variance for FiLM modulation in decode. If None, zero tensors are used.

        Returns
        -------
        pred_mean : torch.Tensor
            Decoded pixel mean.
        total_pixel_var : torch.Tensor
            Total pixel variance (latent propagation + decoder conditional variance).
        """
        pred_mean, _ = self.decode(z_content, z_style, logvar_content, logvar_style)
        unc = self.propagate_pixel_uncertainty(latent_mean, latent_var, logvar_content=logvar_content, logvar_style=logvar_style)
        return pred_mean, unc["pixel_uncert_var_px"]

    def build_uncertainty_pack(self, latent_mean, reverse_var, reverse_logvar, latent_uncert_var, logvar_content=None, logvar_style=None):
        """Assemble a full uncertainty dict from latent/pixel/reverse variances."""
        unc_pack = self.propagate_pixel_uncertainty(latent_mean, latent_uncert_var, logvar_content=logvar_content, logvar_style=logvar_style)
        reverse_var = _clamp_var(reverse_var, min_val=1e-8, max_val=1e6)
        latent_uncert_var = _clamp_var(latent_uncert_var, min_val=1e-8, max_val=1e6)
        if reverse_logvar is None:
            reverse_logvar = torch.log(torch.clamp(reverse_var, min=1e-8))
        pixel_uncert_var = unc_pack["pixel_uncert_var_px"]

        return {
            "sr_mean": unc_pack["pred_mean"],
            "sr_var": pixel_uncert_var,
            "sr_std": torch.sqrt(torch.clamp(pixel_uncert_var, min=1e-8)),
            "reverse_uncert_var": reverse_var,
            "reverse_uncert_std": torch.sqrt(torch.clamp(reverse_var, min=1e-8)),
            "latent_uncert_var": latent_uncert_var,
            "latent_uncert_std": torch.sqrt(torch.clamp(latent_uncert_var, min=1e-8)),
            "pixel_uncert_var": pixel_uncert_var,
            "pixel_uncert_std": torch.sqrt(torch.clamp(pixel_uncert_var, min=1e-8)),
            "latent_mean": latent_mean,
            "latent_var": latent_uncert_var,
            "latent_logvar": torch.log(torch.clamp(latent_uncert_var, min=1e-8)),
            "reverse_logvar": reverse_logvar,
        }

    def build_input_uncertainty_pack(
        self,
        latent_mean,
        latent_var=None,
        logvar_content=None,
        logvar_style=None,
    ):
        """Build uncertainty pack for input latent before diffusion sampling."""
        content_ch = self.content_ch
        z_content = latent_mean[:, :content_ch, ...]
        z_style = latent_mean[:, content_ch:, ...] if self.has_disentangled_style else torch.empty(
            latent_mean.shape[0], 0, *latent_mean.shape[2:], device=latent_mean.device, dtype=latent_mean.dtype
        )

        if logvar_content is None:
            logvar_content = torch.zeros_like(z_content)
        if logvar_style is None:
            logvar_style = torch.zeros_like(z_style)

        has_input_latent_var = latent_var is not None
        if latent_var is None:
            latent_uncert_var = torch.zeros_like(latent_mean)
        else:
            latent_uncert_var = _clamp_var(latent_var, min_val=1e-8, max_val=1e6)

        unc_pack = self.propagate_pixel_uncertainty(
            latent_mean,
            latent_uncert_var,
            logvar_content=logvar_content,
            logvar_style=logvar_style,
        )
        pred_mean = unc_pack["pred_mean"]
        pixel_uncert_var = _clamp_var(unc_pack["pixel_uncert_var_px"], min_val=1e-8, max_val=1e6)
        reverse_uncert_var = torch.zeros_like(latent_mean)
        latent_logvar = (
            torch.log(torch.clamp(latent_uncert_var, min=1e-8))
            if has_input_latent_var
            else torch.full_like(latent_mean, fill_value=-30.0)
        )
        return {
            "sr_mean": pred_mean,
            "sr_var": pixel_uncert_var,
            "sr_std": torch.sqrt(pixel_uncert_var),
            "reverse_uncert_var": reverse_uncert_var,
            "reverse_uncert_std": torch.zeros_like(latent_mean),
            "latent_uncert_var": latent_uncert_var,
            "latent_uncert_std": torch.sqrt(torch.clamp(latent_uncert_var, min=1e-8)),
            "pixel_uncert_var": pixel_uncert_var,
            "pixel_uncert_std": torch.sqrt(torch.clamp(pixel_uncert_var, min=1e-8)),
            "latent_mean": latent_mean,
            "latent_var": latent_uncert_var,
            "latent_logvar": latent_logvar,
            "reverse_logvar": torch.full_like(latent_mean, fill_value=-30.0),
        }
