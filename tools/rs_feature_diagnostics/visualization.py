from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .utils import artifact_path


def normalize_map(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32))
    low, high = float(array.min()), float(array.max())
    return (array - low) / (high - low + 1e-12)


def save_figure(fig, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(artifact_path(base_path, ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(artifact_path(base_path, ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(array: np.ndarray, base_path: Path, title: str, colorbar_label: str = "value", cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(array, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x / token")
    ax.set_ylabel("y / token")
    bar = fig.colorbar(image, ax=ax)
    bar.set_label(colorbar_label)
    save_figure(fig, base_path)

def overlay(image: Image.Image, array: np.ndarray, base_path: Path, title: str, alpha: float = .45) -> None:
    array = normalize_map(array)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = Image.fromarray(np.uint8(255 * array)).resize(image.size, resampling)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(image)
    ax.imshow(np.asarray(resized), cmap="jet", alpha=alpha, vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    save_figure(fig, base_path)


def histogram(values: np.ndarray, base_path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(np.asarray(values).ravel(), bins=40, color="#4472c4", alpha=.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=.2)
    save_figure(fig, base_path)


def line_plot(
    x: Sequence[float], series: dict[str, Sequence[float]], base_path: Path,
    title: str, xlabel: str, ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for label, values in series.items():
        ax.plot(x, values, marker="o", linewidth=1.8, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.25)
    if len(series) > 1:
        ax.legend(fontsize=8)
    save_figure(fig, base_path)


def grouped_bar_plot(
    categories: Sequence[str], series: dict[str, Sequence[float]], base_path: Path,
    title: str, ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6.0, .8 * len(categories)), 4.4))
    x = np.arange(len(categories), dtype=np.float32)
    width = .8 / max(len(series), 1)
    for series_index, (label, values) in enumerate(series.items()):
        offset = (series_index - (len(series) - 1) / 2) * width
        ax.bar(x + offset, np.asarray(values, dtype=np.float32), width=width, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.grid(axis="y", alpha=.25)
    if len(series) > 1:
        ax.legend(fontsize=8)
    save_figure(fig, base_path)


def matrix_heatmap(
    array: np.ndarray, x_labels: Sequence[str], y_labels: Sequence[str],
    base_path: Path, title: str, colorbar_label: str, cmap: str = "viridis",
    annotate: bool = False,
) -> None:
    fig_width = min(14.0, max(5.2, .45 * len(x_labels) + 2.0))
    fig_height = min(14.0, max(4.4, .35 * len(y_labels) + 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(array, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=11)
    x_step = max(1, int(np.ceil(len(x_labels) / 30)))
    y_step = max(1, int(np.ceil(len(y_labels) / 30)))
    x_ticks = np.arange(0, len(x_labels), x_step)
    y_ticks = np.arange(0, len(y_labels), y_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([x_labels[index] for index in x_ticks], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([y_labels[index] for index in y_ticks], fontsize=8)
    if annotate and array.size <= 100:
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    bar = fig.colorbar(image, ax=ax)
    bar.set_label(colorbar_label)
    save_figure(fig, base_path)


def multi_panel(
    arrays: Sequence[np.ndarray], titles: Sequence[str], base_path: Path,
    cmaps: Optional[Sequence[str]] = None, columns: int = 3,
) -> None:
    if not arrays:
        return
    rows = int(np.ceil(len(arrays) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.7 * rows), squeeze=False)
    cmaps = cmaps or ["viridis"] * len(arrays)
    for ax, array, title, cmap in zip(axes.flat, arrays, titles, cmaps):
        ax.imshow(array, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes.flat[len(arrays):]:
        ax.axis("off")
    save_figure(fig, base_path)
