from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torchvision.transforms import functional as TF

from .dataset import base_canvas, load_segmentation_mask, open_rgb
from .hook_manager import CaptureHookManager
from .metrics import (
    effective_rank, homogeneous_proxy, spatial_maps, subset_similarity,
    token_similarity,
)
from .sample_selector import Prediction
from .tensor_adapter import FeatureTensorAdapter
from .utils import safe_name, semantic_stage_label, write_csv, write_json
from .visualization import heatmap, histogram, multi_panel, overlay


def analyze_spatial(
    model,
    dataset,
    predictions: Sequence[Prediction],
    stage_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    input_size: tuple[int, int, int],
    crop_pct: float,
    interpolation: str,
    max_tokens: int,
    similarity_samples: int,
    seed: int,
    temperature: float,
    texture_quantile: float,
    segmentation_mask_dir: Optional[Path] = None,
    split_root: Optional[Path] = None,
    layouts: Optional[Dict[str, str]] = None,
) -> list[dict]:
    layouts = layouts or {}
    adapter = FeatureTensorAdapter()
    rows: list[dict] = []
    model.eval()
    for sample_no, prediction in enumerate(predictions):
        image_tensor, _ = dataset[prediction.dataset_index]
        batch = image_tensor.unsqueeze(0).to(device)
        with CaptureHookManager(model, stage_names, detach=True) as hooks:
            with torch.inference_mode():
                model(batch)
        original = open_rgb(prediction.image_path)
        canvas = base_canvas(original, input_size[-2:], crop_pct, interpolation)
        raw_tensor = TF.to_tensor(canvas).to(device)
        sample_id = f"{sample_no:04d}_{safe_name(prediction.true_class)}_to_{safe_name(prediction.pred_class)}"
        sample_arrays = [np.asarray(canvas)]
        sample_titles = [
            f"Input image\nTrue: {prediction.true_class} | Pred: {prediction.pred_class}"
        ]
        sample_payload = {}
        for stage_index, stage in enumerate(stage_names):
            stage_label = semantic_stage_label(stage, stage_index)
            try:
                adapted = adapter.adapt(hooks.outputs[stage], layout_hint=layouts.get(stage))
            except (ValueError, TypeError) as error:
                rows.append({"sample_id": sample_id, "stage": stage, "status": f"skipped: {error}"})
                continue
            if not adapted.spatial:
                rows.append({"sample_id": sample_id, "stage": stage, "status": "skipped: spatial layout unavailable"})
                continue
            feature = adapted.tensor.float()
            maps = spatial_maps(feature, temperature)
            rank = effective_rank(feature)
            homogeneous = homogeneous_proxy(raw_tensor, adapted.grid_size, texture_quantile)
            segmentation_mask = None
            segmentation_path = None
            if segmentation_mask_dir is not None and split_root is not None:
                segmentation_mask, segmentation_path = load_segmentation_mask(
                    segmentation_mask_dir, prediction.image_path, split_root, adapted.grid_size)
            if sample_no < similarity_samples:
                similarity_matrix, similarity, sampled_indices = token_similarity(
                    feature, max_tokens, seed + sample_no)
                homogeneous_similarity = subset_similarity(feature, homogeneous)
                textured_similarity = subset_similarity(feature, ~homogeneous)
                foreground_similarity = (
                    subset_similarity(feature, segmentation_mask.to(feature.device))
                    if segmentation_mask is not None else float("nan"))
                background_similarity = (
                    subset_similarity(feature, ~segmentation_mask.to(feature.device))
                    if segmentation_mask is not None else float("nan"))
                similarity_status = "computed"
            else:
                similarity_matrix = feature.new_empty((0, 0))
                sampled_indices = torch.empty(0, dtype=torch.long, device=feature.device)
                similarity = {
                    "mean_off_diagonal": float("nan"), "median": float("nan"),
                    "p90": float("nan"), "sampled_tokens": 0,
                    "total_tokens": int(feature.shape[-2] * feature.shape[-1]),
                }
                homogeneous_similarity = textured_similarity = float("nan")
                foreground_similarity = background_similarity = float("nan")
                similarity_status = "skipped_by_similarity_samples_limit"
            energy = maps["energy"][0].cpu().numpy()
            variance = maps["variance"][0].cpu().numpy()
            entropy = maps["entropy"][0].cpu().numpy()
            row = {
                "sample_id": sample_id, "image_path": prediction.image_path,
                "true_class": prediction.true_class, "pred_class": prediction.pred_class,
                "correct": prediction.correct, "stage_index": stage_index, "stage": stage,
                "stage_label": stage_label,
                "feature_shape": str(tuple(feature.shape)), "layout": adapted.source_layout,
                "mean_energy": float(maps["energy"].mean()),
                "mean_spatial_variance": float(maps["variance"].mean()),
                "mean_spatial_entropy": float(maps["entropy"].mean()),
                "homogeneous_similarity": homogeneous_similarity,
                "non_homogeneous_similarity": textured_similarity,
                "foreground_similarity": foreground_similarity,
                "background_similarity": background_similarity,
                "segmentation_mask_path": segmentation_path or "not_available",
                "similarity_status": similarity_status,
                "image_texture_variance": float(raw_tensor.var()),
                "status": "ok", **rank, **similarity,
            }
            rows.append(row)
            stage_dir = output_dir / "spatial" / sample_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            base = stage_dir / stage_label
            np.savez_compressed(
                base.with_suffix(".npz"), energy=energy, variance=variance, entropy=entropy,
                similarity=similarity_matrix.cpu().numpy(), sampled_token_indices=sampled_indices.cpu().numpy(),
                homogeneous_mask=homogeneous.cpu().numpy(),
                segmentation_mask=(segmentation_mask.cpu().numpy() if segmentation_mask is not None
                                   else np.empty((0, 0), dtype=np.uint8)),
            )
            title_suffix = f"{stage_label} ({stage})"
            heatmap(energy, base.with_name(base.name + "_energy"), f"Spatial energy — {title_suffix}", "L2 energy")
            heatmap(variance, base.with_name(base.name + "_variance"), f"Spatial variance — {title_suffix}", "channel variance")
            heatmap(entropy, base.with_name(base.name + "_entropy"), f"Spatial entropy — {title_suffix}", "entropy")
            overlay(original, energy, base.with_name(base.name + "_energy_overlay"), f"Spatial energy — {title_suffix}")
            if sample_no == 0 and similarity_matrix.numel():
                heatmap(similarity_matrix.cpu().numpy(), base.with_name(base.name + "_token_similarity"),
                        f"Token cosine similarity — {title_suffix}", "cosine", cmap="coolwarm")
                off_diag = similarity_matrix[~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=device)]
                histogram(off_diag.cpu().numpy(), base.with_name(base.name + "_similarity_hist"),
                          f"Token similarity distribution — {title_suffix}", "cosine similarity")
            sample_arrays.append(energy)
            sample_titles.append(f"{stage_label} energy\n({stage})")
            sample_payload[stage] = row
        if sample_arrays:
            multi_panel(sample_arrays, sample_titles, output_dir / "samples" / f"{sample_id}_stage_energy", columns=2)
            write_json(output_dir / "samples" / f"{sample_id}_spatial_metrics.json", sample_payload)
    write_csv(output_dir / "tables" / "spatial_metrics.csv", rows)
    aggregate_spatial(rows, output_dir)
    return rows


def aggregate_spatial(rows: Sequence[dict], output_dir: Path) -> list[dict]:
    numeric = (
        "mean_energy", "mean_spatial_variance", "mean_spatial_entropy",
        "mean_off_diagonal", "median", "p90", "entropy_effective_rank",
        "energy_rank_90", "energy_rank_95", "homogeneous_similarity",
        "non_homogeneous_similarity", "image_texture_variance",
        "foreground_similarity", "background_similarity",
    )
    grouped: Dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(row["stage"], bool(row["correct"]))].append(row)
    output = []
    for (stage, correct), group in grouped.items():
        result = {"stage": stage, "group": "correct" if correct else "misclassified", "samples": len(group)}
        for key in numeric:
            values = np.asarray([item[key] for item in group], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"mean_{key}"] = float(finite.mean()) if finite.size else float("nan")
            result[f"std_{key}"] = float(finite.std()) if finite.size else float("nan")
        output.append(result)
    # Dataset-level relation between image texture and feature redundancy.
    for stage in sorted({row.get("stage") for row in rows if row.get("status") == "ok"}):
        group = [row for row in rows if row.get("stage") == stage and row.get("status") == "ok"]
        if len(group) >= 3:
            texture = np.asarray([row["image_texture_variance"] for row in group])
            redundancy = np.asarray([row["mean_off_diagonal"] for row in group])
            correlation = float(np.corrcoef(texture, redundancy)[0, 1])
            output.append({"stage": stage, "group": "texture_vs_redundancy", "samples": len(group),
                           "pearson_correlation": correlation})
    write_csv(output_dir / "tables" / "spatial_metrics_aggregated.csv", output)
    return output
