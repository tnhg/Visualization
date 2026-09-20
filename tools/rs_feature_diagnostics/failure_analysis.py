from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import open_rgb
from .sample_selector import Prediction
from .utils import artifact_path, safe_name
from .visualization import save_figure


def failure_panels(
    predictions: Sequence[Prediction], spatial_rows: Sequence[dict], scale_rows: Sequence[dict],
    output_dir: Path,
) -> None:
    for sample_no, prediction in enumerate(predictions):
        if prediction.correct:
            continue
        sample_id = prediction.sample_id
        rows = [row for row in spatial_rows if row.get("sample_id") == sample_id and row.get("status") == "ok"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes[0, 0].imshow(open_rgb(prediction.image_path))
        axes[0, 0].set_title(
            f"Input\ntrue={prediction.true_class}, pred={prediction.pred_class}\nconfidence={prediction.confidence:.3f}")
        for column, target in ((1, "pred"), (2, "true")):
            path = artifact_path(output_dir / "gradcam" / f"{sample_id}_{target}", ".npy")
            if path.exists():
                axes[0, column].imshow(np.load(path), cmap="jet")
                axes[0, column].set_title(f"Grad-CAM ({target})")
            else:
                axes[0, column].text(.5, .5, "not generated", ha="center")
                axes[0, column].set_title(f"Grad-CAM ({target})")
        if rows:
            last_label = rows[-1].get("stage_label", f"stage{len(rows)}")
            last = artifact_path(output_dir / "spatial" / sample_id / last_label, ".npz")
            if last.exists():
                axes[0, 3].imshow(np.load(last)["energy"], cmap="viridis")
            axes[0, 3].set_title("Late-stage spatial energy")
            x = np.arange(len(rows)) + 1
            axes[1, 0].plot(x, [row["entropy_effective_rank"] for row in rows], marker="o")
            axes[1, 0].set_title("Stage effective rank")
            axes[1, 1].plot(x, [row["mean_off_diagonal"] for row in rows], marker="o")
            axes[1, 1].set_title("Stage token similarity")
            labels = [row.get("stage_label", f"feature{index + 1}") for index, row in enumerate(rows)]
            for axis in (axes[1, 0], axes[1, 1]):
                axis.set_xticks(x)
                axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        sample_scale = [row for row in scale_rows if row.get("sample_id") == sample_id]
        if sample_scale:
            stages = list(dict.fromkeys(row["stage"] for row in sample_scale))
            scales = sorted({float(row["scale"]) for row in sample_scale})
            matrix = np.asarray([[np.mean([r["rms_activation"] for r in sample_scale if r["stage"] == s and r["scale"] == scale]) for scale in scales] for s in stages])
            axes[1, 2].imshow(matrix, aspect="auto", cmap="magma")
            axes[1, 2].set_title("Scale × stage response")
        erf_path = artifact_path(output_dir / "erf" / f"{sample_id}_stem", ".npy")
        if erf_path.exists():
            axes[1, 3].imshow(np.load(erf_path), cmap="inferno")
            axes[1, 3].set_title("Early-stage ERF")
        else:
            axes[1, 3].text(.5, .5, "not generated", ha="center")
            axes[1, 3].set_title("ERF")
        for ax in axes.flat:
            ax.grid(alpha=.15)
            if not ax.lines:
                ax.set_xticks([])
                ax.set_yticks([])
        name = (
            f"true_{safe_name(prediction.true_class)}_pred_{safe_name(prediction.pred_class)}_"
            f"conf_{prediction.confidence:.3f}_{sample_no:04d}"
        )
        save_figure(fig, output_dir / "samples" / name)
