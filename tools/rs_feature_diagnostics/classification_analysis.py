from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .sample_selector import Prediction
from .utils import write_csv, write_json
from .visualization import grouped_bar_plot, line_plot, matrix_heatmap


EPS = 1e-12


@dataclass
class ClassificationAnalysisResult:
    confusion_counts: np.ndarray
    confusion_row_normalized: np.ndarray
    per_class_rows: list[dict]
    top_confusions: list[dict]
    reliability_rows: list[dict]
    risk_coverage_rows: list[dict]
    summary: dict


def confusion_counts(predictions: Sequence[Prediction], num_classes: int) -> np.ndarray:
    """Return an auditable true-label-row, predicted-label-column matrix."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for prediction in predictions:
        true_index, pred_index = prediction.true_index, prediction.pred_index
        if not (0 <= true_index < num_classes and 0 <= pred_index < num_classes):
            raise ValueError(
                "prediction class index is outside the supplied class mapping: "
                f"true={true_index}, predicted={pred_index}, classes={num_classes}")
        matrix[true_index, pred_index] += 1
    return matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize confusion rows while keeping absent classes finite and explicit."""
    support = matrix.sum(axis=1, keepdims=True)
    result = np.zeros_like(matrix, dtype=np.float64)
    np.divide(matrix, support, out=result, where=support > 0)
    return result


def per_class_metrics(matrix: np.ndarray, classes: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for index, class_name in enumerate(classes):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class_index": index,
            "class": class_name,
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def ranked_confusions(matrix: np.ndarray, classes: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for true_index in range(matrix.shape[0]):
        support = int(matrix[true_index].sum())
        for pred_index in range(matrix.shape[1]):
            count = int(matrix[true_index, pred_index])
            if true_index == pred_index or count == 0:
                continue
            rows.append({
                "true_index": true_index,
                "true_class": classes[true_index],
                "pred_index": pred_index,
                "pred_class": classes[pred_index],
                "count": count,
                "true_class_error_rate": count / support if support else 0.0,
            })
    return sorted(rows, key=lambda row: (-row["count"], row["true_index"], row["pred_index"]))


def reliability_metrics(predictions: Sequence[Prediction], bins: int = 10) -> tuple[list[dict], dict]:
    if bins < 2:
        raise ValueError("reliability bins must be at least two")
    if not predictions:
        return [], {"samples": 0, "top1_accuracy": float("nan"), "ece": float("nan"), "nll": float("nan")}
    confidence = np.clip(np.asarray([item.confidence for item in predictions], dtype=np.float64), 0.0, 1.0)
    correct = np.asarray([item.pred_index == item.true_index for item in predictions], dtype=np.float64)
    true_probability = np.clip(
        np.asarray([item.true_class_probability for item in predictions], dtype=np.float64), EPS, 1.0)
    bin_index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    rows: list[dict] = []
    ece = 0.0
    for index in range(bins):
        selected = bin_index == index
        count = int(selected.sum())
        lower, upper = index / bins, (index + 1) / bins
        if count:
            mean_confidence = float(confidence[selected].mean())
            accuracy = float(correct[selected].mean())
            gap = accuracy - mean_confidence
            ece += abs(gap) * count / len(predictions)
        else:
            mean_confidence = accuracy = gap = float("nan")
        rows.append({
            "bin_index": index,
            "bin_lower": lower,
            "bin_upper": upper,
            "samples": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "accuracy_minus_confidence": gap,
        })
    return rows, {
        "samples": len(predictions),
        "top1_accuracy": float(correct.mean()),
        "top1_accuracy_percent": float(100.0 * correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "ece": float(ece),
        "nll": float(-np.log(true_probability).mean()),
        "reliability_bins": bins,
    }


def risk_coverage_curve(predictions: Sequence[Prediction], points: int = 101) -> list[dict]:
    """Evaluate selective risk as increasingly lower-confidence samples are retained."""
    if points < 2:
        raise ValueError("risk-coverage points must be at least two")
    if not predictions:
        return []
    ordered = sorted(predictions, key=lambda item: (-item.confidence, item.dataset_index))
    confidence = np.asarray([item.confidence for item in ordered], dtype=np.float64)
    correct = np.asarray([item.pred_index == item.true_index for item in ordered], dtype=np.float64)
    retained_counts = np.unique(np.ceil(np.linspace(1, len(ordered), min(points, len(ordered)))).astype(np.int64))
    cumulative_correct = np.cumsum(correct)
    rows: list[dict] = []
    for retained in retained_counts:
        accuracy = float(cumulative_correct[retained - 1] / retained)
        rows.append({
            "retained_samples": int(retained),
            "coverage": float(retained / len(ordered)),
            "confidence_threshold": float(confidence[retained - 1]),
            "accuracy": accuracy,
            "risk": float(1.0 - accuracy),
        })
    return rows


def analyze_classification(
    predictions: Sequence[Prediction],
    classes: Sequence[str],
    output_dir: Path,
    reliability_bins: int = 10,
    risk_coverage_points: int = 101,
) -> ClassificationAnalysisResult:
    """Write dataset-level classification diagnostics from existing inference records only."""
    if not classes:
        raise ValueError("classification analysis requires a non-empty class mapping")
    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_counts(predictions, len(classes))
    normalized = row_normalize(matrix)
    class_rows = per_class_metrics(matrix, classes)
    confusions = ranked_confusions(matrix, classes)
    reliability_rows, summary = reliability_metrics(predictions, reliability_bins)
    risk_rows = risk_coverage_curve(predictions, risk_coverage_points)
    summary.update({
        "classes": len(classes),
        "macro_precision": float(np.mean([row["precision"] for row in class_rows])),
        "macro_recall": float(np.mean([row["recall"] for row in class_rows])),
        "macro_f1": float(np.mean([row["f1"] for row in class_rows])),
        "confusion_matrix_layout": "rows=true_class, columns=predicted_class",
    })

    np.save(classification_dir / "confusion_counts.npy", matrix)
    np.save(classification_dir / "confusion_row_normalized.npy", normalized)
    write_csv(output_dir / "tables" / "classification_per_class.csv", class_rows)
    write_csv(output_dir / "tables" / "classification_top_confusions.csv", confusions)
    write_csv(output_dir / "tables" / "classification_reliability.csv", reliability_rows)
    write_csv(output_dir / "tables" / "classification_risk_coverage.csv", risk_rows)
    write_json(output_dir / "tables" / "classification_summary.json", summary)

    matrix_heatmap(
        matrix, classes, classes, classification_dir / "confusion_counts",
        "Confusion matrix counts (rows=true, columns=predicted)", "samples", cmap="Blues",
        annotate=matrix.size <= 100)
    matrix_heatmap(
        normalized, classes, classes, classification_dir / "confusion_row_normalized",
        "Row-normalized confusion matrix (rows=true, columns=predicted)",
        "fraction of true class", cmap="Blues", annotate=matrix.size <= 100)
    metric_plot_rows = sorted(class_rows, key=lambda row: (row["f1"], row["class_index"]))[:50]
    metric_plot_title = (
        "Per-class classification metrics"
        if len(metric_plot_rows) == len(class_rows)
        else f"Lowest-F1 classes ({len(metric_plot_rows)} of {len(class_rows)})")
    grouped_bar_plot(
        [row["class"] for row in metric_plot_rows],
        {
            "precision": [row["precision"] for row in metric_plot_rows],
            "recall": [row["recall"] for row in metric_plot_rows],
            "F1": [row["f1"] for row in metric_plot_rows],
        },
        classification_dir / "per_class_metrics",
        metric_plot_title, "score")
    valid_reliability = [row for row in reliability_rows if row["samples"]]
    if valid_reliability:
        confidence = [float(row["mean_confidence"]) for row in valid_reliability]
        line_plot(
            confidence,
            {
                "perfect calibration": confidence,
                "observed accuracy": [float(row["accuracy"]) for row in valid_reliability],
            },
            classification_dir / "reliability_diagram",
            f"Reliability diagram (ECE={summary['ece']:.4f}, NLL={summary['nll']:.4f})",
            "mean confidence", "accuracy")
    if risk_rows:
        line_plot(
            [float(row["coverage"]) for row in risk_rows],
            {"selective risk": [float(row["risk"]) for row in risk_rows]},
            classification_dir / "risk_coverage",
            "Risk-coverage curve", "coverage", "risk (1 - accuracy)")
    return ClassificationAnalysisResult(
        confusion_counts=matrix,
        confusion_row_normalized=normalized,
        per_class_rows=class_rows,
        top_confusions=confusions,
        reliability_rows=reliability_rows,
        risk_coverage_rows=risk_rows,
        summary=summary,
    )
