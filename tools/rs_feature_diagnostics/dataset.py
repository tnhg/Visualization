from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import numpy as np
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


@dataclass(frozen=True)
class PreprocessGeometry:
    """The exact geometry used to turn an original image into model input."""

    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    canvas_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    crop_pct: float
    crop_mode: str
    interpolation: str
    normalization: dict[str, tuple[float, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessedSample:
    tensor: torch.Tensor
    canvas: Image.Image
    geometry: PreprocessGeometry


class UnifiedPreprocessor:
    """Single source of truth for evaluation inputs, canvases, and masks."""

    def __init__(
        self,
        input_size: tuple[int, int, int],
        mean: Sequence[float],
        std: Sequence[float],
        crop_pct: float,
        interpolation: str,
        crop_mode: str = "center",
    ) -> None:
        if len(input_size) != 3 or input_size[0] != 3:
            raise ValueError(f"input_size must be (3,H,W), got {input_size}")
        if not 0 < float(crop_pct) <= 1:
            raise ValueError("crop_pct must be in (0, 1]")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each contain three values")
        if any(float(value) <= 0 for value in std):
            raise ValueError("std values must be positive")
        if crop_mode not in {"center", "resize"}:
            raise ValueError(f"unsupported evaluation crop_mode: {crop_mode!r}")
        self.input_size = tuple(int(value) for value in input_size)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.crop_pct = float(crop_pct)
        self.interpolation = str(interpolation)
        self.crop_mode = str(crop_mode)

    def process(self, image: Image.Image) -> PreprocessedSample:
        image = image.convert("RGB")
        _, height, width = self.input_size
        original_width, original_height = image.size
        mode = _interp_mode(self.interpolation)
        if self.crop_mode == "resize":
            resized = TF.resize(image, (height, width), interpolation=mode, antialias=True)
            resized_size = (height, width)
            crop_box = (0, 0, width, height)
            canvas = resized
        else:
            # Match torchvision/timm evaluation semantics: resize the shorter
            # edge while preserving aspect ratio, then take the requested crop.
            short_edge = max(1, round(min(height, width) / self.crop_pct))
            resized = TF.resize(image, short_edge, interpolation=mode, antialias=True)
            if resized.height < height or resized.width < width:
                scale = max(height / max(resized.height, 1), width / max(resized.width, 1))
                resized = TF.resize(
                    resized,
                    (max(height, round(resized.height * scale)), max(width, round(resized.width * scale))),
                    interpolation=mode, antialias=True,
                )
            resized_size = (resized.height, resized.width)
            top = max(0, (resized.height - height) // 2)
            left = max(0, (resized.width - width) // 2)
            canvas = TF.crop(resized, top, left, height, width)
            crop_box = (left, top, left + width, top + height)
        geometry = PreprocessGeometry(
            original_size=(original_width, original_height),
            resized_size=resized_size,
            canvas_size=(height, width), crop_box=crop_box,
            crop_pct=self.crop_pct, crop_mode=self.crop_mode,
            interpolation=self.interpolation,
            normalization={"mean": self.mean, "std": self.std},
        )
        tensor = _normalize_rgb(canvas, self.mean, self.std)
        return PreprocessedSample(tensor=tensor, canvas=canvas, geometry=geometry)

    def process_path(self, path: str | Path) -> PreprocessedSample:
        return self.process(open_rgb(path))

    def mask_canvas(self, mask: Image.Image, geometry: PreprocessGeometry) -> Image.Image:
        if mask.size != (geometry.original_size[0], geometry.original_size[1]):
            raise ValueError(
                f"mask size {mask.size} does not match image size {geometry.original_size}")
        mode = _interp_mode(geometry.interpolation)
        mask = mask.convert("L")
        if geometry.crop_mode == "resize":
            return TF.resize(mask, geometry.canvas_size, interpolation=InterpolationMode.NEAREST)
        resized = TF.resize(mask, geometry.resized_size, interpolation=InterpolationMode.NEAREST)
        left, top, right, bottom = geometry.crop_box
        return resized.crop((left, top, right, bottom))

    def mask_tensor(
        self, mask: Image.Image, geometry: PreprocessGeometry,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        canvas = self.mask_canvas(mask, geometry)
        if output_size is not None:
            canvas = TF.resize(canvas, output_size, interpolation=InterpolationMode.NEAREST)
        values = torch.from_numpy(np.array(canvas, dtype="uint8", copy=True))
        return values > 0


class _TensorTransform:
    """Pickle-safe ImageFolder transform backed by the unified preprocessor."""
    def __init__(self, preprocessor: UnifiedPreprocessor) -> None:
        self.preprocessor = preprocessor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.preprocessor.process(image).tensor


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
    preprocessor = UnifiedPreprocessor(
        tuple(data_config["input_size"]), data_config["mean"], data_config["std"],
        float(data_config["crop_pct"]), str(data_config["interpolation"]),
        str(data_config.get("crop_mode", "center")),
    )
    transform = _TensorTransform(preprocessor)
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
    output_size: Tuple[int, int], preprocessor: UnifiedPreprocessor | None = None,
    mask_encoding: str = "auto",
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
    existing = list(dict.fromkeys(candidate.resolve() for candidate in candidates if candidate.is_file()))
    if len(existing) > 1:
        raise RuntimeError(f"ambiguous segmentation mask for {image_path}: {existing}")
    path = existing[0] if existing else None
    if path is None:
        return None, None
    with Image.open(path) as image:
        values = set(np.unique(np.asarray(image.convert("L"), dtype="uint8")).tolist())
        if mask_encoding not in {"auto", "01", "0255"}:
            raise ValueError("mask_encoding must be auto, 01, or 0255")
        if mask_encoding == "01" and not values <= {0, 1}:
            raise ValueError(f"mask {path} is not encoded as 0/1: values={sorted(values)}")
        if mask_encoding == "0255" and not values <= {0, 255}:
            raise ValueError(f"mask {path} is not encoded as 0/255: values={sorted(values)}")
        if mask_encoding == "auto" and not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f"mask {path} has unsupported values: {sorted(values)}")
        if preprocessor is not None:
            geometry = preprocessor.process_path(image_path).geometry
            tensor = preprocessor.mask_tensor(image, geometry, output_size)
        else:
            tensor = TF.resize(image.convert("L"), output_size, interpolation=InterpolationMode.NEAREST)
            tensor = torch.from_numpy(np.array(tensor, dtype="uint8", copy=True)) > 0
    return tensor, str(path.resolve())


def _interp_mode(name: str) -> InterpolationMode:
    name = name.lower()
    return {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }.get(name, InterpolationMode.BICUBIC)


def base_canvas(image: Image.Image, size: Tuple[int, int], crop_pct: float, interpolation: str) -> Image.Image:
    preprocessor = UnifiedPreprocessor(
        (3, int(size[0]), int(size[1])), (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225), crop_pct, interpolation)
    return preprocessor.process(image).canvas


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
    crop_mode: str = "center",
) -> tuple[torch.Tensor, str]:
    _, height, width = input_size
    preprocessor = UnifiedPreprocessor(input_size, mean, std, crop_pct, interpolation, crop_mode)
    canvas = preprocessor.process(image).canvas
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
    crop_mode: str = "center",
) -> tuple[torch.Tensor, str]:
    """Reduce effective resolution while preserving occupancy, FOV, and context."""
    if not 0 < resolution_scale <= 1:
        raise ValueError("resolution_scale must be in (0, 1]")
    _, height, width = input_size
    canvas = UnifiedPreprocessor(input_size, mean, std, crop_pct, interpolation, crop_mode).process(image).canvas
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
    crop_mode: str = "center",
) -> tuple[torch.Tensor, str]:
    """Remove peripheral context without resizing the retained center pixels."""
    if zoom_scale <= 1:
        raise ValueError("zoom_scale must be greater than one")
    _, height, width = input_size
    canvas = UnifiedPreprocessor(input_size, mean, std, crop_pct, interpolation, crop_mode).process(image).canvas
    crop_h = max(1, round(height / zoom_scale))
    crop_w = max(1, round(width / zoom_scale))
    center = _position_crop(canvas, crop_h, crop_w, "center")
    result = _mean_canvas(width, height, mean)
    result.paste(center, ((width - crop_w) // 2, (height - crop_h) // 2))
    method = f"center_crop({crop_h},{crop_w})+native_pixel_center_pad"
    return _normalize_rgb(result, mean, std), method
