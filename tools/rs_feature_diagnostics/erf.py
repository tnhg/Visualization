from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .hook_manager import CaptureHookManager
from .metrics import erf_radii
from .sample_selector import Prediction, extract_logits
from .tensor_adapter import FeatureTensorAdapter
from .dataset import UnifiedPreprocessor
from .utils import artifact_path, safe_name, semantic_stage_label, write_csv
from .visualization import heatmap, overlay


def stage_erf(
    model,
    image: torch.Tensor,
    layer_name: str,
    location: str = "center",
    channel_index: Optional[int] = None,
    layout_hint: Optional[str] = None,
) -> tuple[np.ndarray, Dict[str, float]]:
    model.zero_grad(set_to_none=True)
    image = image.detach().float().requires_grad_(True)
    with CaptureHookManager(model, [layer_name], detach=False) as hooks:
        model(image)
        adapted = FeatureTensorAdapter().adapt(hooks.outputs[layer_name], layout_hint=layout_hint)
        if not adapted.spatial:
            raise RuntimeError(f"stage ERF requires a spatial feature: {layer_name}")
        feature = adapted.tensor
        energy = feature.abs().mean(dim=1)
        if location == "max":
            flat = int(energy[0].argmax())
            y, x = divmod(flat, energy.shape[-1])
        else:
            y, x = energy.shape[-2] // 2, energy.shape[-1] // 2
        objective = feature[0, channel_index, y, x] if channel_index is not None else feature[0, :, y, x].square().sum().sqrt()
        objective.backward()
    if image.grad is None or not torch.isfinite(image.grad).all():
        gradient = torch.zeros(image.shape[-2:], device=image.device)
        radii = erf_radii(gradient, center=(float(y), float(x)))
        radii.update({"query_y": int(y), "query_x": int(x), "status": "unavailable",
                      "reason": "gradient_missing_or_nonfinite"})
        return gradient.cpu().numpy(), radii
    gradient = image.grad.detach().abs().mean(dim=1)[0]
    maximum = gradient.max()
    if not torch.isfinite(maximum) or float(maximum) <= 1e-12:
        radii = erf_radii(gradient, center=(float(y), float(x)))
        radii.update({"query_y": int(y), "query_x": int(x), "status": "unavailable",
                      "reason": "zero_gradient"})
        return gradient.cpu().numpy(), radii
    gradient = gradient / maximum
    return gradient.cpu().numpy(), {
        **erf_radii(gradient, center=(float(y), float(x))),
        "query_y": int(y), "query_x": int(x),
    }


def class_sensitive_gradient(model, image: torch.Tensor, class_index: int) -> tuple[np.ndarray, Dict[str, float]]:
    model.zero_grad(set_to_none=True)
    image = image.detach().float().requires_grad_(True)
    logits = extract_logits(model(image))
    logits[0, class_index].backward()
    if image.grad is None or not torch.isfinite(image.grad).all():
        gradient = torch.zeros(image.shape[-2:], device=image.device)
        return gradient.cpu().numpy(), {**erf_radii(gradient), "status": "unavailable",
                                       "reason": "gradient_missing_or_nonfinite"}
    gradient = image.grad.detach().abs().mean(dim=1)[0]
    maximum = gradient.max()
    if not torch.isfinite(maximum) or float(maximum) <= 1e-12:
        return gradient.cpu().numpy(), {**erf_radii(gradient), "status": "unavailable",
                                       "reason": "zero_gradient"}
    gradient = gradient / maximum
    return gradient.cpu().numpy(), erf_radii(gradient)


def analyze_erf_samples(
    model,
    dataset,
    predictions: Sequence[Prediction],
    stage_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    location: str,
    layouts: Optional[Dict[str, str]] = None,
    input_size: tuple[int, int, int] | None = None,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    crop_pct: float = 1.0,
    interpolation: str = "bilinear",
    crop_mode: str = "center",
) -> list[dict]:
    layouts = layouts or {}
    rows = []
    model.eval()
    from .dataset import open_rgb
    preprocessor = (
        UnifiedPreprocessor(input_size, mean, std, crop_pct, interpolation, crop_mode)
        if input_size is not None else None)
    for sample_no, prediction in enumerate(predictions[:max(1, min(4, len(predictions)))]):
        processed = preprocessor.process_path(prediction.image_path) if preprocessor else None
        tensor = processed.tensor if processed is not None else dataset[prediction.dataset_index][0]
        tensor = tensor.unsqueeze(0).to(device)
        original = processed.canvas if processed is not None else open_rgb(prediction.image_path)
        sample_id = prediction.sample_id
        for stage_index, stage in enumerate(stage_names):
            stage_label = semantic_stage_label(stage, stage_index)
            gradient, radii = stage_erf(model, tensor, stage, location, layout_hint=layouts.get(stage))
            base = output_dir / "erf" / f"{sample_id}_{stage_label}"
            np.save(artifact_path(base, ".npy"), gradient)
            heatmap(gradient, base.with_name(base.name + "_map"), f"ERF: {stage_label} ({stage})")
            overlay(original, gradient, base.with_name(base.name + "_overlay"), f"ERF: {stage_label} ({stage})")
            rows.append({"sample_id": sample_id, "image_path": prediction.image_path,
                         "kind": "stage_erf", "stage": stage, "stage_label": stage_label, **radii})
        gradient, radii = class_sensitive_gradient(model, tensor, prediction.pred_index)
        base = output_dir / "erf" / f"{sample_id}_class_sensitive_pred"
        np.save(artifact_path(base, ".npy"), gradient)
        heatmap(gradient, base.with_name(base.name + "_map"), "Class-sensitive input gradient")
        rows.append({"sample_id": sample_id, "image_path": prediction.image_path,
                     "kind": "class_sensitive_input_gradient", "stage": "classifier", **radii})
    write_csv(output_dir / "tables" / "erf_radii.csv", rows)
    return rows
