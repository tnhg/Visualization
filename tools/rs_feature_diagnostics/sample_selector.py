from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .tensor_adapter import first_tensor
from .utils import autocast_context, write_csv


@dataclass
class Prediction:
    dataset_index: int
    image_path: str
    true_class: str
    true_index: int
    pred_class: str
    pred_index: int
    confidence: float
    true_class_probability: float
    logit_margin: float
    correct: bool


def extract_logits(output) -> torch.Tensor:
    tensor = first_tensor(output)
    if tensor is None or tensor.ndim != 2:
        raise ValueError(f"model output is not [B, classes]: {type(output)}")
    return tensor


def run_inference(
    model,
    loader: DataLoader,
    classes: Sequence[str],
    samples: Sequence[tuple[str, int]],
    device: torch.device,
    amp: bool,
    output_path: Path,
) -> List[Prediction]:
    model.eval()
    rows: List[Prediction] = []
    offset = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, amp):
                logits = extract_logits(model(images))
            probabilities = logits.float().softmax(dim=1)
            top_values, top_indices = probabilities.topk(min(2, probabilities.shape[1]), dim=1)
            top_logits = logits.float().topk(min(2, logits.shape[1]), dim=1).values
            for index in range(images.shape[0]):
                target = int(targets[index])
                pred = int(top_indices[index, 0])
                margin = float(top_logits[index, 0] - (
                    top_logits[index, 1] if top_logits.shape[1] > 1 else 0.))
                path, sample_target = samples[offset + index]
                if sample_target != target:
                    raise RuntimeError("dataset order changed between discovery and inference")
                rows.append(Prediction(
                    dataset_index=offset + index, image_path=path,
                    true_class=classes[target], true_index=target,
                    pred_class=classes[pred], pred_index=pred,
                    confidence=float(top_values[index, 0]),
                    true_class_probability=float(probabilities[index, target]),
                    logit_margin=margin, correct=pred == target,
                ))
            offset += images.shape[0]
    write_csv(output_path, [asdict(row) for row in rows])
    return rows


def select_predictions(
    predictions: Sequence[Prediction],
    mode: str,
    max_samples: int,
    class_filter: Sequence[str] = (),
    top_k_errors: int = 50,
) -> List[Prediction]:
    allowed = set(class_filter)
    rows = list(predictions)
    if allowed:
        rows = [row for row in rows if row.true_class in allowed or str(row.true_index) in allowed]
    if mode == "misclassified":
        rows = sorted((row for row in rows if not row.correct), key=lambda row: row.confidence, reverse=True)
        rows = rows[:top_k_errors]
    elif mode == "correct":
        rows = sorted((row for row in rows if row.correct), key=lambda row: row.confidence, reverse=True)
    elif mode == "class":
        if not allowed:
            raise ValueError("--sample-mode class requires --class-filter")
        rows = sorted(rows, key=lambda row: row.confidence, reverse=True)
    elif mode != "all":
        raise ValueError(mode)
    return rows[:max_samples]
