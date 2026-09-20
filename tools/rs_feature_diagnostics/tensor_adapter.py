from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


def tensor_candidates(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(tensor_candidates(item))
        return result
    if isinstance(value, dict):
        result: list[torch.Tensor] = []
        for item in value.values():
            result.extend(tensor_candidates(item))
        return result
    return []


def first_tensor(value: Any, strict: bool = False) -> Optional[torch.Tensor]:
    candidates = tensor_candidates(value)
    if strict and len(candidates) > 1:
        raise ValueError(
            f"ambiguous nested output contains {len(candidates)} tensors; "
            "declare an adapter selector instead of choosing the first")
    return candidates[0] if candidates else None


def replace_first_tensor(value: Any, tensor: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            if first_tensor(item) is not None:
                items[index] = replace_first_tensor(item, tensor)
                break
        return tuple(items)
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if first_tensor(item) is not None:
                items[index] = replace_first_tensor(item, tensor)
                break
        return items
    if isinstance(value, dict):
        items = dict(value)
        for key, item in items.items():
            if first_tensor(item) is not None:
                items[key] = replace_first_tensor(item, tensor)
                break
        return items
    raise TypeError("module output has no tensor to replace")


@dataclass
class AdaptedFeature:
    tensor: torch.Tensor
    source_layout: str
    grid_size: Optional[Tuple[int, int]]
    removed_class_token: bool = False

    @property
    def spatial(self) -> bool:
        return self.tensor.ndim == 4 and self.grid_size is not None


class FeatureTensorAdapter:
    """Convert NCHW, NHWC, or BNC features to NCHW without unsafe reshaping."""

    def adapt(
        self,
        value: Any,
        expected_grid: Optional[Tuple[int, int]] = None,
        layout_hint: Optional[str] = None,
        warn: bool = True,
    ) -> AdaptedFeature:
        tensor = first_tensor(value, strict=True)
        if tensor is None:
            raise TypeError("feature output does not contain a tensor")
        if tensor.ndim == 4:
            return self._adapt_4d(tensor, expected_grid, layout_hint)
        if tensor.ndim == 3:
            return self._adapt_tokens(tensor, expected_grid, warn)
        if warn:
            warnings.warn(f"spatial layout unavailable for shape {tuple(tensor.shape)}", stacklevel=2)
        return AdaptedFeature(tensor=tensor, source_layout=f"{tensor.ndim}D", grid_size=None)

    @staticmethod
    def _adapt_4d(
        tensor: torch.Tensor,
        expected_grid: Optional[Tuple[int, int]],
        layout_hint: Optional[str],
    ) -> AdaptedFeature:
        if layout_hint:
            hint = layout_hint.upper()
            if hint == "NCHW":
                return AdaptedFeature(tensor, hint, tuple(tensor.shape[-2:]))
            if hint == "NHWC":
                converted = tensor.permute(0, 3, 1, 2).contiguous()
                return AdaptedFeature(converted, hint, tuple(converted.shape[-2:]))
            raise ValueError(f"unsupported layout hint: {layout_hint}")
        if expected_grid:
            if tuple(tensor.shape[-2:]) == expected_grid:
                return AdaptedFeature(tensor, "NCHW", expected_grid)
            if tuple(tensor.shape[1:3]) == expected_grid:
                converted = tensor.permute(0, 3, 1, 2).contiguous()
                return AdaptedFeature(converted, "NHWC", expected_grid)
        if tensor.shape[2] == tensor.shape[3] and tensor.shape[1] != tensor.shape[2]:
            return AdaptedFeature(tensor, "NCHW", tuple(tensor.shape[-2:]))
        if tensor.shape[1] == tensor.shape[2] and tensor.shape[3] != tensor.shape[1]:
            converted = tensor.permute(0, 3, 1, 2).contiguous()
            return AdaptedFeature(converted, "NHWC", tuple(converted.shape[-2:]))
        raise ValueError(f"ambiguous 4D tensor layout for shape {tuple(tensor.shape)}; provide an adapter override")

    @staticmethod
    def _adapt_tokens(
        tensor: torch.Tensor,
        expected_grid: Optional[Tuple[int, int]],
        warn: bool,
    ) -> AdaptedFeature:
        token_count = tensor.shape[1]
        removed_cls = False
        if expected_grid:
            spatial_count = expected_grid[0] * expected_grid[1]
            if token_count == spatial_count + 1:
                tensor = tensor[:, 1:]
                removed_cls = True
            elif token_count != spatial_count:
                if warn:
                    warnings.warn(
                        f"token count {token_count} does not match expected grid {expected_grid}", stacklevel=2)
                return AdaptedFeature(tensor, "BNC", None)
            grid = expected_grid
        else:
            side = math.isqrt(token_count)
            if side * side == token_count:
                grid = (side, side)
            else:
                side = math.isqrt(token_count - 1)
                if side * side == token_count - 1:
                    tensor = tensor[:, 1:]
                    removed_cls = True
                    grid = (side, side)
                else:
                    if warn:
                        warnings.warn(
                            f"cannot safely infer a spatial grid from {token_count} tokens", stacklevel=2)
                    return AdaptedFeature(tensor, "BNC", None)
        b, _, c = tensor.shape
        converted = tensor.transpose(1, 2).reshape(b, c, *grid).contiguous()
        return AdaptedFeature(converted, "BNC", grid, removed_cls)


def restore_feature_layout(tensor: torch.Tensor, original: torch.Tensor, adapted: AdaptedFeature) -> torch.Tensor:
    if adapted.source_layout == "NCHW":
        return tensor
    if adapted.source_layout == "NHWC":
        return tensor.permute(0, 2, 3, 1).contiguous()
    if adapted.source_layout == "BNC":
        tokens = tensor.flatten(2).transpose(1, 2).contiguous()
        if adapted.removed_class_token:
            tokens = torch.cat((original[:, :1], tokens), dim=1)
        return tokens
    raise ValueError(f"cannot restore unsupported layout {adapted.source_layout}")
