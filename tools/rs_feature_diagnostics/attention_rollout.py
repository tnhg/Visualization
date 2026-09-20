from __future__ import annotations

"""Attention-rollout visualisation for global-token Transformer classifiers.

The implementation captures the probability tensor emitted by the ``attn_drop``
submodule used by timm attention blocks.  This avoids model-specific forward
rewrites and deliberately declines windowed / cross-attention tensors that
cannot be projected back to an input image without an architecture-specific
mapping.
"""

import math
from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from .dataset import open_rgb
from .sample_selector import Prediction
from .tensor_adapter import first_tensor
from .utils import artifact_path, safe_name, write_csv
from .visualization import heatmap, multi_panel, normalize_map, overlay


@dataclass
class AttentionRolloutResult:
    rollout: np.ndarray
    final_head_maps: np.ndarray
    layer_names: list[str]
    grid_size: tuple[int, int]
    token_count: int
    prefix_tokens: int
    captured_layer_count: int


class AttentionCapture(AbstractContextManager):
    """Capture square attention-probability tensors during one model forward.

    timm enables fused scaled-dot-product attention on supported PyTorch builds,
    which bypasses ``attn_drop``.  Fusing is temporarily disabled only while the
    context is active, then every original module value is restored.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.candidate_names: list[str] = []
        self._fused_states: list[tuple[object, object]] = []

    def __enter__(self):
        for _name, module in self.model.named_modules():
            if not hasattr(module, "fused_attn"):
                continue
            try:
                value = getattr(module, "fused_attn")
                setattr(module, "fused_attn", False)
                self._fused_states.append((module, value))
            except (AttributeError, RuntimeError, TypeError):
                # Some third-party modules expose a read-only implementation
                # detail. They remain usable if they already emit attn_drop.
                continue

        for name, module in self.model.named_modules():
            if name.endswith("attn_drop"):
                self.candidate_names.append(name)
                self.handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None or tensor.ndim != 4:
                return
            if tensor.shape[-1] != tensor.shape[-2]:
                return
            # CPU storage keeps each selected sample bounded on GPU and means
            # that rollout arithmetic cannot hold on to the inference graph.
            self.outputs[name] = tensor.detach().float().cpu()
        return hook

    @torch.inference_mode()
    def capture(self, image: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        self.outputs.clear()
        self.model(image)
        return list(self.outputs.items())

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, value in reversed(self._fused_states):
            try:
                setattr(module, "fused_attn", value)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._fused_states.clear()

    def __exit__(self, *args):
        self.close()
        return False


def _infer_grid(patch_tokens: int, input_size: tuple[int, int]) -> Optional[tuple[int, int]]:
    if patch_tokens < 1:
        return None
    target_ratio = input_size[0] / max(input_size[1], 1)
    candidates = []
    for height in range(1, math.isqrt(patch_tokens) + 1):
        if patch_tokens % height:
            continue
        width = patch_tokens // height
        for first, second in ((height, width), (width, height)):
            candidates.append((abs(math.log((first / max(second, 1)) / target_ratio)), first, second))
    if not candidates:
        return None
    _, height, width = min(candidates)
    return height, width


def _compatible_group(
    captured: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], int, int]:
    """Select the largest same-token-count group without mixing model stages."""
    groups: dict[tuple[int, int], list[tuple[str, torch.Tensor]]] = {}
    captured_count = 0
    for name, tensor in captured:
        if tensor.ndim != 4 or tensor.shape[-1] != tensor.shape[-2] or tensor.shape[0] < 1:
            continue
        captured_count += 1
        groups.setdefault((int(tensor.shape[-1]), int(tensor.shape[1])), []).append((name, tensor))
    if not groups:
        return [], 0, captured_count
    # Layer count is primary; token count breaks ties in favour of global over
    # smaller local windows.
    key, group = max(groups.items(), key=lambda item: (len(item[1]), item[0][0]))
    return group, key[0], captured_count


def build_attention_rollout(
    captured: Sequence[tuple[str, torch.Tensor]],
    prefix_tokens: int,
    input_size: tuple[int, int],
    expected_patch_grid: Optional[tuple[int, int]] = None,
) -> AttentionRolloutResult:
    """Build residual attention rollout and final-layer per-head maps.

    ``prefix_tokens`` must include a class token.  Attention-only architectures
    that pool patch tokens have no unambiguous class-query row, so callers get a
    descriptive error rather than a misleading heatmap.
    """
    if prefix_tokens < 1:
        raise ValueError("attention rollout requires a class/prefix token")
    group, token_count, captured_count = _compatible_group(captured)
    if not group:
        raise ValueError("no square attention-probability tensors were captured")
    if token_count <= prefix_tokens:
        raise ValueError(f"attention has {token_count} tokens but {prefix_tokens} prefix tokens")
    grid_size = _infer_grid(token_count - prefix_tokens, input_size)
    if grid_size is None:
        raise ValueError("patch-token count cannot be projected to an image grid")
    if expected_patch_grid is not None:
        expected_patch_grid = tuple(int(value) for value in expected_patch_grid)
        if math.prod(expected_patch_grid) != token_count - prefix_tokens:
            raise ValueError(
                "captured attention does not span the model's full patch grid "
                f"({token_count - prefix_tokens} tokens vs {expected_patch_grid})")
        grid_size = expected_patch_grid

    rollout = torch.eye(token_count, dtype=torch.float32)
    for _name, tensor in group:
        attention = tensor[0].mean(dim=0)
        identity = torch.eye(token_count, dtype=attention.dtype)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = attention @ rollout

    final_attention = group[-1][1][0]
    rollout_map = rollout[0, prefix_tokens:].reshape(grid_size).numpy()
    head_maps = final_attention[:, 0, prefix_tokens:].reshape(-1, *grid_size).numpy()
    return AttentionRolloutResult(
        rollout=rollout_map,
        final_head_maps=head_maps,
        layer_names=[name for name, _ in group],
        grid_size=grid_size,
        token_count=token_count,
        prefix_tokens=prefix_tokens,
        captured_layer_count=captured_count,
    )


def analyze_attention_rollout_samples(
    model,
    dataset,
    predictions: Sequence[Prediction],
    device: torch.device,
    output_dir: Path,
    input_size: tuple[int, int, int],
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    crop_pct: float = 1.0,
    interpolation: str = "bilinear",
    crop_mode: str = "center",
) -> list[dict]:
    """Create rollout and final-head figures for selected samples when supported."""
    rows: list[dict] = []
    prefix_tokens = model_prefix_tokens(model)
    from .dataset import UnifiedPreprocessor
    preprocessor = UnifiedPreprocessor(input_size, mean, std, crop_pct, interpolation, crop_mode)
    model.eval()
    with AttentionCapture(model) as capture:
        if not capture.candidate_names:
            reason = "skipped: model exposes no attn_drop probability modules"
            for sample_no, prediction in enumerate(predictions):
                rows.append({
                    "sample_id": _sample_id(sample_no, prediction),
                    "image_path": prediction.image_path,
                    "status": reason,
                })
        else:
            for sample_no, prediction in enumerate(predictions):
                sample_id = _sample_id(sample_no, prediction)
                processed = preprocessor.process_path(prediction.image_path)
                image = processed.tensor
                expected_patch_grid = model_patch_grid(model, tuple(image.shape[-2:]))
                captured = capture.capture(image.unsqueeze(0).to(device))
                try:
                    result = build_attention_rollout(
                        captured, prefix_tokens, tuple(input_size[-2:]), expected_patch_grid)
                except ValueError as error:
                    rows.append({
                        "sample_id": sample_id, "image_path": prediction.image_path,
                        "captured_attention_layers": len(captured),
                        "status": f"skipped: {error}",
                    })
                    continue
                base = output_dir / "attention" / f"{sample_id}_rollout"
                base.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    artifact_path(base, ".npz"), rollout=result.rollout,
                    final_head_maps=result.final_head_maps,
                    layer_names=np.asarray(result.layer_names),
                )
                title = f"Attention rollout ({len(result.layer_names)} layers)"
                heatmap(result.rollout, base.with_name(base.name + "_map"), title, "class-token attention")
                overlay(processed.canvas, result.rollout,
                        base.with_name(base.name + "_overlay"), title)
                visible_heads = min(16, result.final_head_maps.shape[0])
                if visible_heads:
                    multi_panel(
                        [normalize_map(item) for item in result.final_head_maps[:visible_heads]],
                        [f"final attention head {index}" for index in range(visible_heads)],
                        base.with_name(base.name + "_final_heads"),
                        cmaps=["magma"] * visible_heads,
                        columns=min(4, visible_heads),
                    )
                rows.append({
                    "sample_id": sample_id, "image_path": prediction.image_path,
                    "true_class": prediction.true_class, "pred_class": prediction.pred_class,
                    "status": "ok", "prefix_tokens": result.prefix_tokens,
                    "token_count": result.token_count, "grid_height": result.grid_size[0],
                    "grid_width": result.grid_size[1], "rollout_layers": len(result.layer_names),
                    "captured_attention_layers": result.captured_layer_count,
                    "used_attention_layers": ";".join(result.layer_names),
                    "final_head_count": int(result.final_head_maps.shape[0]),
                })
    write_csv(output_dir / "tables" / "attention_rollout_manifest.csv", rows)
    return rows


def _sample_id(sample_no: int, prediction: Prediction) -> str:
    return prediction.sample_id or f"{sample_no:04d}_{safe_name(prediction.true_class)}_to_{safe_name(prediction.pred_class)}"


def model_prefix_tokens(model) -> int:
    """Return a declared class-token count; do not invent one for window models."""
    value = getattr(model, "num_prefix_tokens", None)
    if value is None:
        # A few third-party ViTs retain ``cls_token`` but do not expose timm's
        # ``num_prefix_tokens``. A window Transformer has neither and must skip.
        value = 1 if getattr(model, "cls_token", None) is not None else 0
    return int(value or 0)


def model_patch_grid(model, image_size: Optional[tuple[int, int]] = None) -> Optional[tuple[int, int]]:
    """Return the full patch grid, respecting timm dynamic-image-size ViTs."""
    patch_embed = getattr(model, "patch_embed", None)
    if image_size is not None and getattr(model, "dynamic_img_size", False):
        dynamic_feature_size = getattr(patch_embed, "dynamic_feat_size", None)
        if callable(dynamic_feature_size):
            try:
                grid = dynamic_feature_size(tuple(int(value) for value in image_size))
                if len(grid) == 2:
                    return int(grid[0]), int(grid[1])
            except (TypeError, ValueError):
                # Fall through to the nominal grid; the captured-token check
                # will still reject an incompatible attention layout.
                pass
    grid = getattr(patch_embed, "grid_size", None)
    if grid is None or len(grid) != 2:
        return None
    return int(grid[0]), int(grid[1])
