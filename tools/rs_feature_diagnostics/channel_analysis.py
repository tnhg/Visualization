from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .erf import stage_erf
from .hook_manager import CaptureHookManager
from .metrics import channel_similarity, similarity_summary
from .sample_selector import Prediction
from .scale_analysis import ScaleAnalysisResult
from .tensor_adapter import FeatureTensorAdapter
from .utils import semantic_stage_label, write_csv
from .visualization import heatmap, histogram, save_figure


def analyze_channel(
    model,
    dataset,
    predictions: Sequence[Prediction],
    stage_names: Sequence[str],
    scale_result: ScaleAnalysisResult,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    erf_channels_per_stage: int,
    all_channel_erf: bool,
    compute_channel_erf: bool,
    layouts: Optional[Dict[str, str]] = None,
) -> tuple[list[dict], Dict[str, torch.Tensor], list[dict]]:
    layouts = layouts or {}
    adapter = FeatureTensorAdapter()
    matrix_sum: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = defaultdict(int)
    summary_rows: list[dict] = []
    first_features: Dict[str, torch.Tensor] = {}
    model.eval()
    for prediction in predictions:
        image, _ = dataset[prediction.dataset_index]
        with CaptureHookManager(model, stage_names, detach=True) as hooks:
            with torch.inference_mode():
                model(image.unsqueeze(0).to(device))
        for stage in stage_names:
            try:
                adapted = adapter.adapt(hooks.outputs[stage], layout_hint=layouts.get(stage))
            except (ValueError, TypeError):
                continue
            if not adapted.spatial:
                continue
            feature = adapted.tensor.float()
            cosine = channel_similarity(feature, "cosine").cpu()
            pearson = channel_similarity(feature, "pearson").cpu()
            if stage not in matrix_sum:
                matrix_sum[stage] = cosine
                first_features[stage] = feature.detach().cpu()
            else:
                matrix_sum[stage] += cosine
            counts[stage] += 1
            stats_cos = similarity_summary(cosine, threshold)
            stats_pearson = similarity_summary(pearson, threshold, absolute=True)
            summary_rows.append({
                "image_path": prediction.image_path, "correct": prediction.correct, "stage": stage,
                **{f"cosine_{key}": value for key, value in stats_cos.items()},
                **{f"pearson_{key}": value for key, value in stats_pearson.items()},
            })
    mean_matrices = {stage: value / max(counts[stage], 1) for stage, value in matrix_sum.items()}
    joint_rows: list[dict] = []
    preferences_by_stage: Dict[str, Dict[int, float]] = defaultdict(dict)
    for row in scale_result.preference_rows:
        preferences_by_stage[row["stage"]][int(row["channel"])] = float(row["preferred_scale"])
    for stage_index, (stage, matrix) in enumerate(mean_matrices.items()):
        stage_label = semantic_stage_label(stage, stage_index)
        np.save(output_dir / "channel" / f"{stage_label}_cosine_similarity.npy", matrix.numpy())
        heatmap(matrix.numpy(), output_dir / "channel" / f"{stage_label}_cosine_similarity",
                f"Channel cosine similarity — {stage_label} ({stage})", "cosine", cmap="coolwarm")
        offdiag = matrix[~torch.eye(matrix.shape[0], dtype=torch.bool)]
        histogram(offdiag.numpy(), output_dir / "channel" / f"{stage_label}_similarity_hist",
                  f"Channel similarity distribution — {stage_label} ({stage})", "cosine similarity")
        preference = preferences_by_stage.get(stage, {})
        order = sorted(range(matrix.shape[0]), key=lambda channel: (preference.get(channel, float("inf")), channel))
        clustered = matrix[order][:, order]
        np.save(output_dir / "channel" / f"{stage_label}_clustered_cosine_similarity.npy",
                clustered.numpy())
        heatmap(clustered.numpy(), output_dir / "channel" / f"{stage_label}_clustered_similarity",
                f"Scale-preference ordered channel similarity — {stage_label} ({stage})", "cosine", cmap="coolwarm")
        within, between = [], []
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[0]):
                if i not in preference or j not in preference:
                    continue
                target = within if preference[i] == preference[j] else between
                target.append(float(matrix[i, j]))
        joint_rows.append({
            "stage": stage, "stage_label": stage_label,
            "within_scale_group_similarity": float(np.mean(within)) if within else float("nan"),
            "between_scale_group_similarity": float(np.mean(between)) if between else float("nan"),
            "within_pairs": len(within), "between_pairs": len(between),
        })
    write_csv(output_dir / "tables" / "channel_redundancy.csv", summary_rows)
    write_csv(output_dir / "tables" / "scale_group_channel_similarity.csv", joint_rows)

    channel_erf_rows: list[dict] = []
    if compute_channel_erf and predictions:
        image, _ = dataset[predictions[0].dataset_index]
        image = image.unsqueeze(0).to(device)
        for stage in stage_names:
            feature = first_features.get(stage)
            if feature is None:
                continue
            activation = feature.abs().mean(dim=(0, 2, 3))
            channels = torch.arange(len(activation)) if all_channel_erf else torch.topk(
                activation, min(erf_channels_per_stage, len(activation))).indices
            if all_channel_erf:
                print(f"WARNING: computing {len(channels)} channel ERFs for {stage}")
            for channel in channels.tolist():
                gradient, radii = stage_erf(model, image, stage, channel_index=channel, layout_hint=layouts.get(stage))
                channel_erf_rows.append({
                    "stage": stage, "channel": channel,
                    "preferred_scale": preferences_by_stage.get(stage, {}).get(channel, float("nan")),
                    **radii,
                })
        write_csv(output_dir / "tables" / "channel_erf.csv", channel_erf_rows)
        if channel_erf_rows:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for stage_index, stage in enumerate(stage_names):
                stage_label = semantic_stage_label(stage, stage_index)
                rows = [row for row in channel_erf_rows if row["stage"] == stage]
                if not rows:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].scatter([row["preferred_scale"] for row in rows],
                                [row["r80_normalized"] for row in rows], alpha=.8)
                axes[0].set(xlabel="preferred content scale", ylabel="normalized ERF r80",
                            title="Scale preference vs channel ERF")
                axes[1].scatter([row["channel"] for row in rows],
                                [row["r80_normalized"] for row in rows], alpha=.8)
                axes[1].set(xlabel="channel index", ylabel="normalized ERF r80",
                            title="Channel-wise ERF")
                for ax in axes:
                    ax.grid(alpha=.2)
                save_figure(fig, output_dir / "channel" / f"{stage_label}_channel_erf_scatter")
    return summary_rows, mean_matrices, channel_erf_rows
