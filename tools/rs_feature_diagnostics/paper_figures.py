from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .scale_analysis import ScaleAnalysisResult
from .visualization import save_figure


def figure_a(spatial_rows: Sequence[dict], deletion_rows: Sequence[dict], output_dir: Path) -> None:
    valid = [row for row in spatial_rows if row.get("status") == "ok"]
    if not valid:
        return
    stages = list(dict.fromkeys(row["stage"] for row in valid))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for correct, label, color in ((True, "correct", "#4472c4"), (False, "misclassified", "#c44e52")):
        subset = [row for row in valid if bool(row["correct"]) == correct]
        if subset:
            axes[0, 0].plot(range(len(stages)), [np.mean([r["mean_off_diagonal"] for r in subset if r["stage"] == s]) for s in stages], marker="o", label=label, color=color)
            axes[0, 1].plot(range(len(stages)), [np.mean([r["entropy_effective_rank"] for r in subset if r["stage"] == s]) for s in stages], marker="o", label=label, color=color)
    axes[0, 0].set_title("Stage-wise token similarity")
    axes[0, 1].set_title("Spatial entropy effective rank")
    for key, label in (("mean_energy", "energy"), ("mean_spatial_variance", "variance"),
                       ("mean_spatial_entropy", "entropy")):
        axes[0, 2].plot(range(len(stages)),
                        [np.mean([r[key] for r in valid if r["stage"] == s]) for s in stages],
                        marker="o", label=label)
    axes[0, 2].set_title("Spatial-map statistics")
    axes[1, 0].plot(range(len(stages)), [np.nanmean([r["homogeneous_similarity"] for r in valid if r["stage"] == s]) for s in stages], marker="o", label="homogeneous proxy")
    axes[1, 0].plot(range(len(stages)), [np.nanmean([r["non_homogeneous_similarity"] for r in valid if r["stage"] == s]) for s in stages], marker="o", label="non-homogeneous")
    axes[1, 0].set_title("Texture-region control")
    if deletion_rows:
        for stage in dict.fromkeys(row["stage"] for row in deletion_rows):
            rows = [row for row in deletion_rows if row["stage"] == stage]
            axes[1, 1].plot([r["keep_ratio"] for r in rows], [r["top1"] for r in rows], marker="o", label=stage)
            axes[1, 2].plot([r["keep_ratio"] for r in rows],
                            [r["mean_true_class_probability"] for r in rows], marker="o", label=stage)
    axes[1, 1].set_title("Spatial deletion: Top-1")
    axes[1, 2].set_title("Spatial deletion: true-class confidence")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    save_figure(fig, output_dir / "paper_figures" / "figure_A_spatial_redundancy_diagnostic")


def figure_b(scale_result: Optional[ScaleAnalysisResult], erf_rows: Sequence[dict], output_dir: Path) -> None:
    if scale_result is None:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    rows = scale_result.rows
    stages = list(dict.fromkeys(row["stage"] for row in rows))
    scales = sorted({float(row["scale"]) for row in rows})
    matrix = np.asarray([[np.mean([r["rms_activation"] for r in rows if r["stage"] == stage and r["scale"] == scale]) for scale in scales] for stage in stages])
    image = axes[0, 0].imshow(matrix, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Scale × stage response")
    axes[0, 0].set_xticks(range(len(scales)), [str(x) for x in scales])
    axes[0, 0].set_yticks(range(len(stages)), [f"S{i+1}" for i in range(len(stages))])
    fig.colorbar(image, ax=axes[0, 0], label="RMS activation")
    if erf_rows:
        stage_erf_rows = [row for row in erf_rows if row.get("kind") == "stage_erf"]
        for key, label in (("r50_normalized", "r50"), ("r80_normalized", "r80"),
                           ("r90_normalized", "r90")):
            axes[0, 2].plot(range(len(stages)), [
                np.mean([row[key] for row in stage_erf_rows if row["stage"] == stage])
                if any(row["stage"] == stage for row in stage_erf_rows) else np.nan
                for stage in stages], marker="o", label=label)
    axes[0, 2].set_title("Stage ERF radii")
    if erf_rows:
        axes[0, 2].legend(fontsize=8)
    cam_paths = sorted((output_dir / "gradcam").glob("*_scale_*_pred.npy"))
    if cam_paths:
        # Keep one sample/target and tile all configured scales in a single
        # panel so the aggregate figure shows a genuine multi-scale comparison.
        prefix = cam_paths[0].name.split("_scale_", 1)[0]
        sample_paths = [path for path in cam_paths if path.name.startswith(prefix + "_scale_")]
        cams = [np.load(path) for path in sample_paths]
        axes[0, 1].imshow(np.concatenate(cams, axis=1), cmap="jet")
        width = cams[0].shape[1]
        for index in range(1, len(cams)):
            axes[0, 1].axvline(index * width - .5, color="white", linewidth=.7)
        axes[0, 1].set_title("Multi-scale Grad-CAM example")
        axes[0, 1].axis("off")
    erf_paths = sorted((output_dir / "erf").glob("*_stage*.npy"))
    if erf_paths:
        axes[1, 0].imshow(np.load(erf_paths[-1]), cmap="magma")
        axes[1, 0].set_title("Stage ERF map example")
        axes[1, 0].axis("off")
    sensitivity_path = output_dir / "tables" / "scale_sensitivity.csv"
    if sensitivity_path.exists():
        import csv
        with sensitivity_path.open() as handle:
            sensitivity = list(csv.DictReader(handle))
        values = [float(r.get("misclassification_correction_rate", r["top1"])) for r in sensitivity]
        axes[1, 1].plot([float(r["scale"]) for r in sensitivity], values, marker="o")
    axes[1, 1].set_title("Content-scale intervention")
    axes[1, 1].set_xlabel("content scale")
    axes[1, 1].set_ylabel("Original-error correction (%)")
    axes[1, 2].axis("off")
    axes[1, 2].text(.02, .95, "Content scaling changes occupancy, resolution, and context.\n"
                    "Dedicated controls separate these factors;\n"
                    "Stage ERF uses a stage-local objective;\n"
                    "Grad-CAM is class-sensitive.", va="top", fontsize=10)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    save_figure(fig, output_dir / "paper_figures" / "figure_B_scale_receptive_field_diagnostic")


def figure_c(
    scale_result: Optional[ScaleAnalysisResult], channel_rows: Sequence[dict],
    joint_rows_path: Path, deletion_rows: Sequence[dict], output_dir: Path,
    mean_matrices: Optional[Dict[str, object]] = None,
    channel_erf_rows: Sequence[dict] = (),
) -> None:
    if scale_result is None or not scale_result.channel_responses:
        return
    stage, response = next(iter(scale_result.channel_responses.items()))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes[0, 0].imshow(response.T, aspect="auto", cmap="viridis")
    axes[0, 0].set_title(f"Channel × scale — {stage}")
    preferences = [row["preferred_scale"] for row in scale_result.preference_rows if row["stage"] == stage]
    selectivity = [row["scale_selectivity_index"] for row in scale_result.preference_rows if row["stage"] == stage]
    axes[0, 1].hist(preferences, bins=max(3, len(set(preferences))), alpha=.65, label="preferred scale")
    axes[0, 1].hist(selectivity, bins=12, alpha=.55, label="selectivity")
    axes[0, 1].set_title("Scale preference / selectivity")
    axes[0, 1].legend(fontsize=8)
    if mean_matrices and stage in mean_matrices:
        channel_matrix = mean_matrices[stage]
        if hasattr(channel_matrix, "numpy"):
            channel_matrix = channel_matrix.numpy()
        axes[0, 2].imshow(channel_matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 2].set_title("Channel similarity matrix")
    joint = []
    if joint_rows_path.exists():
        import csv
        with joint_rows_path.open() as handle:
            joint = list(csv.DictReader(handle))
    if joint:
        names = [f"S{i+1}" for i in range(len(joint))]
        x = np.arange(len(names))
        axes[1, 0].bar(x - .18, [float(row["within_scale_group_similarity"]) for row in joint], .36, label="within")
        axes[1, 0].bar(x + .18, [float(row["between_scale_group_similarity"]) for row in joint], .36, label="between")
        axes[1, 0].set_xticks(x, names)
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Within vs between scale groups")
    if channel_erf_rows:
        axes[1, 1].scatter([row["preferred_scale"] for row in channel_erf_rows],
                           [row["r80_normalized"] for row in channel_erf_rows], alpha=.7)
    axes[1, 1].set_title("Scale preference vs channel ERF")
    axes[1, 1].set_xlabel("preferred scale")
    axes[1, 1].set_ylabel("normalized r80")
    if deletion_rows:
        for stage_name in dict.fromkeys(row["stage"] for row in deletion_rows):
            rows = [row for row in deletion_rows if row["stage"] == stage_name]
            axes[1, 2].plot([r["effective_mean_retained_ratio"] for r in rows], [r["top1"] for r in rows], marker="o", label=stage_name)
    axes[1, 2].set_title("Channel deletion curve")
    axes[1, 2].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    save_figure(fig, output_dir / "paper_figures" / "figure_C_channel_specialization_redundancy_diagnostic")
