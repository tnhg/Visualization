from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .hook_manager import InterventionHook
from .sample_selector import extract_logits
from .tensor_adapter import FeatureTensorAdapter, restore_feature_layout
from .utils import safe_name
from .visualization import save_figure


def evaluate_indices(
    model, dataset, indices: Sequence[int], device: torch.device, batch_size: int, workers: int,
) -> tuple[Dict[str, float], list[dict]]:
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False, num_workers=workers)
    correct = 0
    total = 0
    true_probability = 0.
    sample_rows: list[dict] = []
    offset = 0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = extract_logits(model(images)).float()
            probability = logits.softmax(dim=1)
            predicted = logits.argmax(1)
            correct += int((predicted == targets).sum())
            true_probability += float(probability.gather(1, targets[:, None]).sum())
            top2 = logits.topk(min(2, logits.shape[1]), dim=1).values
            margins = top2[:, 0] - top2[:, 1] if logits.shape[1] > 1 else top2[:, 0]
            for index in range(targets.numel()):
                sample_rows.append({
                    "dataset_index": int(indices[offset + index]),
                    "true_index": int(targets[index]),
                    "pred_index": int(predicted[index]),
                    "true_class_probability": float(probability[index, targets[index]]),
                    "predicted_class_probability": float(probability[index, predicted[index]]),
                    "logit_margin": float(margins[index]),
                    "correct": bool(predicted[index] == targets[index]),
                })
            offset += targets.numel()
            total += targets.numel()
    return ({"top1": 100. * correct / max(total, 1),
             "mean_true_class_probability": true_probability / max(total, 1),
             "samples": total}, sample_rows)


def spatial_mask_function(keep_ratio: float, replacement: str, layout_hint: Optional[str] = None):
    adapter = FeatureTensorAdapter()

    def function(output: torch.Tensor) -> torch.Tensor:
        adapted = adapter.adapt(output, layout_hint=layout_hint)
        if not adapted.spatial:
            raise RuntimeError("spatial intervention requires a recoverable feature grid")
        feature = adapted.tensor
        energy = feature.square().sum(dim=1).sqrt().flatten(1)
        keep = max(1, round(energy.shape[1] * keep_ratio))
        threshold = energy.kthvalue(max(1, energy.shape[1] - keep + 1), dim=1).values[:, None]
        mask = (energy >= threshold).reshape(feature.shape[0], 1, *feature.shape[-2:])
        if replacement == "mean":
            fill = feature.mean(dim=(2, 3), keepdim=True)
            modified = torch.where(mask, feature, fill)
        else:
            modified = feature * mask
        return restore_feature_layout(modified, output, adapted)
    return function


def spatial_deletion_curve(
    model, dataset, indices: Sequence[int], stage: str, keep_ratios: Sequence[float],
    replacement: str, device: torch.device, batch_size: int, workers: int,
    layout_hint: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    rows = []
    sample_rows = []
    for ratio in keep_ratios:
        with InterventionHook(model, stage, spatial_mask_function(ratio, replacement, layout_hint)):
            metrics, details = evaluate_indices(model, dataset, indices, device, batch_size, workers)
        rows.append({"stage": stage, "keep_ratio": ratio, "replacement": replacement, **metrics})
        sample_rows.extend({"stage": stage, "keep_ratio": ratio, "replacement": replacement, **row}
                           for row in details)
    return rows, sample_rows


def greedy_clusters(similarity: torch.Tensor, threshold: float) -> List[List[int]]:
    remaining = set(range(similarity.shape[0]))
    clusters: List[List[int]] = []
    while remaining:
        anchor = min(remaining)
        cluster = sorted(index for index in remaining if float(similarity[anchor, index]) >= threshold)
        for index in cluster:
            remaining.discard(index)
        clusters.append(cluster)
    return clusters


def channel_mask_function(clusters: Sequence[Sequence[int]], keep_ratio: float, replacement: str, layout_hint: Optional[str] = None):
    adapter = FeatureTensorAdapter()

    def function(output: torch.Tensor) -> torch.Tensor:
        adapted = adapter.adapt(output, layout_hint=layout_hint)
        if not adapted.spatial:
            raise RuntimeError("channel intervention requires a recoverable feature grid")
        modified = adapted.tensor.clone()
        for cluster in clusters:
            keep = max(1, round(len(cluster) * keep_ratio)) if keep_ratio > 0 else 1
            retained, removed = list(cluster[:keep]), list(cluster[keep:])
            if not removed:
                continue
            if replacement == "cluster_mean":
                fill = modified[:, retained].mean(dim=1, keepdim=True)
                modified[:, removed] = fill
            else:
                modified[:, removed] = 0
        return restore_feature_layout(modified, output, adapted)
    return function


def channel_deletion_curve(
    model, dataset, indices: Sequence[int], stage: str, similarity: torch.Tensor,
    threshold: float, keep_ratios: Sequence[float], replacement: str,
    device: torch.device, batch_size: int, workers: int, layout_hint: Optional[str] = None,
) -> tuple[list[dict], List[List[int]], list[dict]]:
    clusters = greedy_clusters(similarity, threshold)
    rows = []
    sample_rows = []
    for ratio in keep_ratios:
        function = channel_mask_function(clusters, ratio, replacement, layout_hint)
        with InterventionHook(model, stage, function):
            metrics, details = evaluate_indices(model, dataset, indices, device, batch_size, workers)
        effective = float(np.mean([max(1, round(len(cluster) * ratio)) / len(cluster)
                                   for cluster in clusters])) if clusters else 1.0
        rows.append({"stage": stage, "retained_ratio_per_cluster": ratio, "replacement": replacement,
                     "effective_mean_retained_ratio": effective, "clusters": len(clusters), **metrics})
        sample_rows.extend({
            "stage": stage, "retained_ratio_per_cluster": ratio,
            "effective_mean_retained_ratio": effective, "replacement": replacement, **row,
        } for row in details)
    return rows, clusters, sample_rows


def plot_sample_intervention_curves(
    rows: Sequence[dict], ratio_key: str, kind: str, output_dir: Path,
) -> None:
    """Render the three requested sample-level causal intervention curves."""
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: Dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dataset_index"]), str(row["stage"]))].append(row)
    for (dataset_index, stage), group in grouped.items():
        group = sorted(group, key=lambda row: float(row[ratio_key]), reverse=True)
        x = [float(row[ratio_key]) for row in group]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, key, title in (
            (axes[0], "true_class_probability", "True-class probability"),
            (axes[1], "predicted_class_probability", "Predicted-class probability"),
            (axes[2], "logit_margin", "Logit margin"),
        ):
            ax.plot(x, [float(row[key]) for row in group], marker="o")
            ax.set_title(title)
            ax.set_xlabel("retained ratio")
            ax.set_ylabel(key.replace("_", " "))
            ax.grid(alpha=.2)
        base = output_dir / "interventions" / (
            f"sample_{dataset_index:06d}_{safe_name(stage).replace('.', '_')}_{kind}_curves")
        save_figure(fig, base)
