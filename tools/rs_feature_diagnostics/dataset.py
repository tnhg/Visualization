from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from timm.data import create_transform


@dataclass
class DatasetInfo:
    root: Path
    split_root: Path
    classes: list[str]
    class_to_idx: Dict[str, int]
    samples: list[tuple[str, int]]
    train_count: Optional[int]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def split_path(root: Path, split: str) -> Path:
    candidate = root / split if split else root
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)
    return candidate


def discover_dataset(root: Path, val_split: str, train_split: str = "train") -> DatasetInfo:
    root = root.expanduser().resolve()
    val_root = split_path(root, val_split)
    dataset = datasets.ImageFolder(val_root)
    train_count: Optional[int] = None
    train_root = root / train_split if train_split else root
    if train_root.is_dir():
        train = datasets.ImageFolder(train_root)
        if train.class_to_idx != dataset.class_to_idx:
            raise RuntimeError("train and validation class mappings differ")
        train_count = len(train)
    return DatasetInfo(
        root=root, split_root=val_root, classes=dataset.classes,
        class_to_idx=dataset.class_to_idx, samples=dataset.samples, train_count=train_count,
    )


def build_eval_dataset(info: DatasetInfo, data_config: Dict[str, object]):
    transform = create_transform(
        input_size=data_config["input_size"], is_training=False,
        mean=data_config["mean"], std=data_config["std"],
        interpolation=data_config["interpolation"], crop_pct=data_config["crop_pct"],
        crop_mode=data_config.get("crop_mode", "center"),
    )
    return datasets.ImageFolder(info.split_root, transform=transform)


def build_loader(dataset, batch_size: int, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_segmentation_mask(
    mask_root: Path, image_path: str | Path, split_root: Path,
    output_size: Tuple[int, int],
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Load a binary mask using an ImageFolder-relative, auditable convention."""
    image_path = Path(image_path).resolve()
    try:
        relative = image_path.relative_to(split_root.resolve())
    except ValueError:
        relative = Path(image_path.parent.name) / image_path.name
    candidates = [
        mask_root / relative,
        (mask_root / relative).with_suffix(".png"),
        mask_root / relative.parent / f"{relative.stem}.png",
        mask_root / f"{relative.stem}.png",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None, None
    with Image.open(path) as image:
        mask = image.convert("L")
        mask = TF.resize(mask, output_size, interpolation=InterpolationMode.NEAREST)
        tensor = TF.to_tensor(mask)[0] > .5
    return tensor, str(path.resolve())


def _interp_mode(name: str) -> InterpolationMode:
    name = name.lower()
    return {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }.get(name, InterpolationMode.BICUBIC)


def base_canvas(image: Image.Image, size: Tuple[int, int], crop_pct: float, interpolation: str) -> Image.Image:
    mode = _interp_mode(interpolation)
    resize_size = tuple(round(v / crop_pct) for v in size)
    image = TF.resize(image, resize_size, interpolation=mode, antialias=True)
    return TF.center_crop(image, size)


def _normalize_rgb(image: Image.Image, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(image), mean, std)


def _mean_canvas(width: int, height: int, mean: Sequence[float]) -> Image.Image:
    fill = tuple(round(255 * value) for value in mean)
    return Image.new("RGB", (width, height), fill)


def _position_crop(image: Image.Image, height: int, width: int, position: str) -> Image.Image:
    image_width, image_height = image.size
    if height > image_height or width > image_width:
        raise ValueError(
            f"crop {(height, width)} exceeds image {(image_height, image_width)}")
    offsets = {
        "center": ((image_height - height) // 2, (image_width - width) // 2),
        "top_left": (0, 0),
        "top_right": (0, image_width - width),
        "bottom_left": (image_height - height, 0),
        "bottom_right": (image_height - height, image_width - width),
    }
    if position not in offsets:
        raise ValueError(f"unknown crop position {position!r}")
    top, left = offsets[position]
    return TF.crop(image, top, left, height, width)


def content_scaled_tensor(
    image: Image.Image,
    scale: float,
    input_size: Tuple[int, int, int],
    mean: Sequence[float],
    std: Sequence[float],
    crop_pct: float,
    interpolation: str,
    crop_position: str = "center",
) -> tuple[torch.Tensor, str]:
    _, height, width = input_size
    canvas = base_canvas(image, (height, width), crop_pct, interpolation)
    mode = _interp_mode(interpolation)
    scaled_h, scaled_w = max(1, round(height * scale)), max(1, round(width * scale))
    scaled = TF.resize(canvas, (scaled_h, scaled_w), interpolation=mode, antialias=True)
    if scale < 1:
        if crop_position != "center":
            raise ValueError("zoom-out supports only centered placement")
        result = _mean_canvas(width, height, mean)
        result.paste(scaled, ((width - scaled_w) // 2, (height - scaled_h) // 2))
        method = f"resize({scaled_h},{scaled_w})+center_pad"
    elif scale > 1:
        result = _position_crop(scaled, height, width, crop_position)
        method = f"resize({scaled_h},{scaled_w})+{crop_position}_crop"
    else:
        if crop_position != "center":
            raise ValueError("unit content scale supports only center")
        result = canvas
        method = "identity_on_fixed_canvas"
    return _normalize_rgb(result, mean, std), method


def resolution_only_tensor(
    image: Image.Image,
    resolution_scale: float,
    input_size: Tuple[int, int, int],
    mean: Sequence[float],
    std: Sequence[float],
    crop_pct: float,
    interpolation: str,
) -> tuple[torch.Tensor, str]:
    """Reduce effective resolution while preserving occupancy, FOV, and context."""
    if not 0 < resolution_scale <= 1:
        raise ValueError("resolution_scale must be in (0, 1]")
    _, height, width = input_size
    canvas = base_canvas(image, (height, width), crop_pct, interpolation)
    mode = _interp_mode(interpolation)
    low_h = max(1, round(height * resolution_scale))
    low_w = max(1, round(width * resolution_scale))
    degraded = TF.resize(canvas, (low_h, low_w), interpolation=mode, antialias=True)
    restored = TF.resize(degraded, (height, width), interpolation=mode, antialias=True)
    method = f"resize({low_h},{low_w})+resize({height},{width})"
    return _normalize_rgb(restored, mean, std), method


def context_only_tensor(
    image: Image.Image,
    zoom_scale: float,
    input_size: Tuple[int, int, int],
    mean: Sequence[float],
    std: Sequence[float],
    crop_pct: float,
    interpolation: str,
) -> tuple[torch.Tensor, str]:
    """Remove peripheral context without resizing the retained center pixels."""
    if zoom_scale <= 1:
        raise ValueError("zoom_scale must be greater than one")
    _, height, width = input_size
    canvas = base_canvas(image, (height, width), crop_pct, interpolation)
    crop_h = max(1, round(height / zoom_scale))
    crop_w = max(1, round(width / zoom_scale))
    center = _position_crop(canvas, crop_h, crop_w, "center")
    result = _mean_canvas(width, height, mean)
    result.paste(center, ((width - crop_w) // 2, (height - crop_h) // 2))
    method = f"center_crop({crop_h},{crop_w})+native_pixel_center_pad"
    return _normalize_rgb(result, mean, std), method
