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
from .utils import safe_name, stable_sample_id
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
            top1_top2 = top2[:, 0] - top2[:, 1] if logits.shape[1] > 1 else top2[:, 0]
            for index in range(targets.numel()):
                target = int(targets[index])
                dataset_index = int(indices[offset + index])
                sample_path = dataset.samples[dataset_index][0] if hasattr(dataset, "samples") else ""
                try:
                    relative_path = str(Path(sample_path).resolve().relative_to(Path(dataset.root).resolve()).as_posix())
                except (ValueError, OSError):
                    relative_path = str(sample_path)
                wrong_logits = logits[index].clone()
                wrong_logits[target] = -torch.inf
                competitor = int(wrong_logits.argmax()) if logits.shape[1] > 1 else target
                gt_margin = float(logits[index, target] - wrong_logits[competitor]) if logits.shape[1] > 1 else float("inf")
                sample_rows.append({
                    "dataset_index": dataset_index,
                    "sample_id": stable_sample_id(relative_path),
                    "relative_path": relative_path,
                    "true_index": int(targets[index]),
                    "pred_index": int(predicted[index]),
                    "true_class_probability": float(probability[index, targets[index]]),
                    "predicted_class_probability": float(probability[index, predicted[index]]),
                    "logit_margin": float(top1_top2[index]),
                    "top1_top2_margin": float(top1_top2[index]),
                    "gt_margin": gt_margin,
                    "fixed_competitor_index": competitor,
                    "fixed_competitor_margin": gt_margin,
                    "correct": bool(predicted[index] == targets[index]),
                })
            offset += targets.numel()
            total += targets.numel()
    return ({"top1": 100. * correct / max(total, 1),
             "mean_true_class_probability": true_probability / max(total, 1),
             "samples": total}, sample_rows)


def spatial_mask_function(keep_ratio: float, replacement: str, layout_hint: Optional[str] = None):
    if not 0 <= float(keep_ratio) <= 1:
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
    adapter = FeatureTensorAdapter()

    def function(output: torch.Tensor) -> torch.Tensor:
        adapted = adapter.adapt(output, layout_hint=layout_hint)
        if not adapted.spatial:
            raise RuntimeError("spatial intervention requires a recoverable feature grid")
        feature = adapted.tensor
        energy = feature.square().sum(dim=1).sqrt().flatten(1)
        total = energy.shape[1]
        function.last_total_positions = int(total)
        keep = int(round(total * keep_ratio))
        mask_flat = torch.zeros_like(energy, dtype=torch.bool)
        if keep:
            # Stable descending sort makes ties deterministic and retains exactly k.
            order = torch.argsort(energy, dim=1, descending=True, stable=True)
            mask_flat.scatter_(1, order[:, :keep], True)
        mask = mask_flat.reshape(feature.shape[0], 1, *feature.shape[-2:])
        if replacement == "mean":
            fill = feature.mean(dim=(2, 3), keepdim=True)
            modified = torch.where(mask, feature, fill)
        else:
            modified = feature * mask
        return restore_feature_layout(modified, output, adapted)
    function.last_total_positions = None
    return function


def spatial_random_mask_function(
    keep_ratio: float, replacement: str, seed: int, layout_hint: Optional[str] = None,
):
    """Random spatial control with the same exact retained count as the target mask."""
    if not 0 <= float(keep_ratio) <= 1:
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
    adapter = FeatureTensorAdapter()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def function(output: torch.Tensor) -> torch.Tensor:
        adapted = adapter.adapt(output, layout_hint=layout_hint)
        if not adapted.spatial:
            raise RuntimeError("spatial intervention requires a recoverable feature grid")
        feature = adapted.tensor
        total = feature.shape[-2] * feature.shape[-1]
        function.last_total_positions = int(total)
        keep = int(round(total * keep_ratio))
        mask_flat = torch.zeros((feature.shape[0], total), dtype=torch.bool, device=feature.device)
        for batch_index in range(feature.shape[0]):
            if keep:
                chosen = torch.randperm(total, generator=generator)[:keep].to(feature.device)
                mask_flat[batch_index, chosen] = True
        mask = mask_flat.reshape(feature.shape[0], 1, *feature.shape[-2:])
        if replacement == "mean":
            fill = feature.mean(dim=(2, 3), keepdim=True)
            modified = torch.where(mask, feature, fill)
        else:
            modified = feature * mask
        return restore_feature_layout(modified, output, adapted)
    function.last_total_positions = None
    return function


def spatial_deletion_curve(
    model, dataset, indices: Sequence[int], stage: str, keep_ratios: Sequence[float],
    replacement: str, device: torch.device, batch_size: int, workers: int,
    layout_hint: Optional[str] = None,
    strategy: str = "response",
    seed: int = 1234,
) -> tuple[list[dict], list[dict]]:
    rows = []
    sample_rows = []
    for ratio in keep_ratios:
        mask_fn = (spatial_mask_function(ratio, replacement, layout_hint)
                   if strategy == "response" else
                   spatial_random_mask_function(ratio, replacement, seed + int(round(ratio * 1000)), layout_hint))
        with InterventionHook(model, stage, mask_fn):
            metrics, details = evaluate_indices(model, dataset, indices, device, batch_size, workers)
        total_positions = getattr(mask_fn, "last_total_positions", None)
        rows.append({"stage": stage, "keep_ratio": ratio, "replacement": replacement,
                     "strategy": strategy, "seed": seed if strategy != "response" else None,
                     "retained_positions": int(round(total_positions * ratio)) if total_positions is not None else None,
                     **metrics})
        sample_rows.extend({"stage": stage, "keep_ratio": ratio, "replacement": replacement,
                            "strategy": strategy, "seed": seed if strategy != "response" else None, **row}
                           for row in details)
    return rows, sample_rows


def greedy_clusters(similarity: torch.Tensor, threshold: float) -> List[List[int]]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if not np.isfinite(float(threshold)) or not -1 <= float(threshold) <= 1:
        raise ValueError("cluster threshold must be finite and in [-1, 1]")
    remaining = set(range(similarity.shape[0]))
    clusters: List[List[int]] = []
    while remaining:
        anchor = min(remaining)
        row = similarity[anchor]
        cluster = sorted(
            index for index in remaining
            if index == anchor and torch.isfinite(row[index]) or
            index != anchor and torch.isfinite(row[index]) and float(row[index]) >= threshold
        )
        if not cluster:
            cluster = [anchor]
        for index in cluster:
            remaining.discard(index)
        clusters.append(cluster)
    return clusters


def channel_mask_function(clusters: Sequence[Sequence[int]], keep_ratio: float, replacement: str, layout_hint: Optional[str] = None):
    if not 0 <= float(keep_ratio) <= 1:
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
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


def channel_random_mask_function(
    total_channels: int, keep_ratio: float, replacement: str, seed: int,
    layout_hint: Optional[str] = None,
):
    """Random channel control retaining exactly the target number of channels."""
    if total_channels < 1 or not 0 <= float(keep_ratio) <= 1:
        raise ValueError("total_channels must be positive and keep_ratio must be in [0, 1]")
    keep = int(round(total_channels * keep_ratio))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    adapter = FeatureTensorAdapter()

    def function(output: torch.Tensor) -> torch.Tensor:
        adapted = adapter.adapt(output, layout_hint=layout_hint)
        if not adapted.spatial or adapted.tensor.shape[1] != total_channels:
            raise RuntimeError("random channel intervention shape does not match declared channels")
        selected = torch.zeros(total_channels, dtype=torch.bool, device=adapted.tensor.device)
        if keep:
            selected[torch.randperm(total_channels, generator=generator)[:keep].to(selected.device)] = True
        modified = adapted.tensor.clone()
        if replacement in {"mean", "cluster_mean"}:
            fill = modified.mean(dim=1, keepdim=True)
            modified[:, ~selected] = fill
        else:
            modified[:, ~selected] = 0
        return restore_feature_layout(modified, output, adapted)
    return function


def channel_deletion_curve(
    model, dataset, indices: Sequence[int], stage: str, similarity: torch.Tensor,
    threshold: float, keep_ratios: Sequence[float], replacement: str,
    device: torch.device, batch_size: int, workers: int, layout_hint: Optional[str] = None,
    strategy: str = "cluster", seed: int = 1234,
) -> tuple[list[dict], List[List[int]], list[dict]]:
    clusters = greedy_clusters(similarity, threshold)
    rows = []
    sample_rows = []
    for ratio in keep_ratios:
        function = (channel_mask_function(clusters, ratio, replacement, layout_hint)
                    if strategy == "cluster" else
                    channel_random_mask_function(similarity.shape[0], ratio, replacement, seed + int(round(ratio * 1000)), layout_hint))
        with InterventionHook(model, stage, function):
            metrics, details = evaluate_indices(model, dataset, indices, device, batch_size, workers)
        retained = sum(
            (max(1, round(len(cluster) * ratio)) if ratio > 0 else 1)
            for cluster in clusters)
        total_channels = sum(len(cluster) for cluster in clusters)
        effective = float(np.mean([max(1, round(len(cluster) * ratio)) / len(cluster)
                                   for cluster in clusters])) if clusters else 1.0
        rows.append({"stage": stage, "retained_ratio_per_cluster": ratio, "replacement": replacement,
                     "effective_mean_retained_ratio": effective,
                     "strategy": strategy, "seed": seed if strategy != "cluster" else None,
                     "retained_channels": retained, "total_channels": total_channels,
                     "effective_retained_ratio": retained / total_channels if total_channels else 1.0,
                     "ratio_zero_strategy": "one_representative_per_cluster" if ratio == 0 else "proportional",
                     "clusters": len(clusters), **metrics})
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
            (axes[2], "gt_margin", "GT margin (GT - fixed strongest wrong)"),
        ):
            ax.plot(x, [float(row[key]) for row in group], marker="o")
            ax.set_title(title)
            ax.set_xlabel("retained ratio")
            ax.set_ylabel(key.replace("_", " "))
            ax.grid(alpha=.2)
        base = output_dir / "interventions" / (
            f"sample_{dataset_index:06d}_{safe_name(stage).replace('.', '_')}_{kind}_curves")
        save_figure(fig, base)
