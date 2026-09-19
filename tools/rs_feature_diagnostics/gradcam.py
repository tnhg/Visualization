from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import content_scaled_tensor, open_rgb
from .sample_selector import Prediction, extract_logits
from .tensor_adapter import FeatureTensorAdapter, first_tensor
from .utils import safe_name
from .visualization import heatmap, multi_panel, overlay


@dataclass
class GradCamResult:
    class_index: int
    cam: np.ndarray


class GradCAM:
    def __init__(self, model, layer_name: str, layout_hint: Optional[str] = None):
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Grad-CAM layer not found: {layer_name}")
        self.model = model
        self.layer_name = layer_name
        self.layout_hint = layout_hint
        self.activation: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None
        self.handle = modules[layer_name].register_forward_hook(self._hook)
        self.adapter = FeatureTensorAdapter()

    def _hook(self, _module, _inputs, output):
        tensor = first_tensor(output)
        if tensor is None:
            raise TypeError(f"Grad-CAM layer {self.layer_name} output contains no tensor")
        self.activation = tensor
        tensor.register_hook(lambda gradient: setattr(self, "gradient", gradient))

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def compute(self, image: torch.Tensor, class_index: int) -> GradCamResult:
        self.model.zero_grad(set_to_none=True)
        self.activation = self.gradient = None
        logits = extract_logits(self.model(image.float()))
        logits[0, class_index].backward()
        if self.activation is None or self.gradient is None:
            raise RuntimeError("Grad-CAM hook did not capture activation/gradient")
        activation = self.adapter.adapt(self.activation, layout_hint=self.layout_hint)
        gradient = self.adapter.adapt(self.gradient, expected_grid=activation.grid_size, layout_hint=self.layout_hint)
        if not activation.spatial or not gradient.spatial:
            raise RuntimeError(f"Grad-CAM layer {self.layer_name} has no recoverable spatial layout")
        weights = gradient.tensor.float().mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation.tensor.float()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-12)
        return GradCamResult(class_index, cam.detach().cpu().numpy())


def analyze_gradcam_samples(
    model,
    dataset,
    predictions: Sequence[Prediction],
    layer_name: str,
    device: torch.device,
    output_dir: Path,
    target_mode: str,
    class_index: Optional[int],
    layout_hint: Optional[str] = None,
    scales: Optional[Sequence[float]] = None,
    input_size: Optional[tuple[int, int, int]] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    crop_pct: float = 1.0,
    interpolation: str = "bilinear",
) -> list[dict]:
    rows = []
    model.eval()
    with GradCAM(model, layer_name, layout_hint) as gradcam:
        for sample_no, prediction in enumerate(predictions):
            tensor, _ = dataset[prediction.dataset_index]
            tensor = tensor.unsqueeze(0).to(device)
            original = open_rgb(prediction.image_path)
            if target_mode == "predicted":
                targets = [("pred", prediction.pred_index)]
            elif target_mode == "true":
                targets = [("true", prediction.true_index)]
            elif target_mode == "index":
                if class_index is None:
                    raise ValueError("--gradcam-target index requires --gradcam-class-index")
                targets = [("index", class_index)]
            else:
                targets = [("pred", prediction.pred_index), ("true", prediction.true_index)]
            sample_id = f"{sample_no:04d}_{safe_name(prediction.true_class)}_to_{safe_name(prediction.pred_class)}"
            for target_name, target in targets:
                result = gradcam.compute(tensor, target)
                base = output_dir / "gradcam" / f"{sample_id}_{target_name}"
                np.save(base.with_suffix(".npy"), result.cam)
                heatmap(result.cam, base.with_name(base.name + "_map"), f"Grad-CAM ({target_name} class)")
                overlay(original, result.cam, base.with_name(base.name + "_overlay"), f"Grad-CAM ({target_name} class)")
                rows.append({"sample_id": sample_id, "image_path": prediction.image_path, "target": target_name,
                             "class_index": target, "layer": layer_name, "content_scale": 1.0,
                             "scale_method": "standard evaluation transform"})

            # Content scale changes while the model canvas remains fixed.  The
            # raw CAM arrays are retained so Figure B can be regenerated without
            # running the model again.
            if scales and input_size and mean and std:
                for target_name, target in targets:
                    scale_cams = []
                    scale_titles = []
                    for scale in scales:
                        scaled, method = content_scaled_tensor(
                            original, scale, input_size, mean, std, crop_pct, interpolation)
                        result = gradcam.compute(scaled.unsqueeze(0).to(device), target)
                        suffix = str(scale).replace(".", "p")
                        base = output_dir / "gradcam" / f"{sample_id}_scale_{suffix}_{target_name}"
                        np.save(base.with_suffix(".npy"), result.cam)
                        heatmap(result.cam, base.with_name(base.name + "_map"),
                                f"Grad-CAM ({target_name}), scale={scale:g}")
                        scale_cams.append(result.cam)
                        scale_titles.append(f"scale={scale:g}")
                        rows.append({
                            "sample_id": sample_id, "image_path": prediction.image_path,
                            "target": target_name, "class_index": target, "layer": layer_name,
                            "content_scale": scale, "scale_method": method,
                        })
                    multi_panel(
                        scale_cams, scale_titles,
                        output_dir / "gradcam" / f"{sample_id}_multiscale_{target_name}",
                        columns=min(3, len(scale_cams)))
    return rows
