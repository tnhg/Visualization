from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .tensor_adapter import first_tensor
from .utils import autocast_context, stable_sample_id, write_csv


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
    sample_id: str = ""
    top1_top2_margin: float | None = None
    gt_margin: float | None = None
    fixed_competitor_index: int | None = None
    fixed_competitor_margin: float | None = None
    logits: tuple[float, ...] = ()
    input_fingerprint: str = ""
    relative_path: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id:
            self.sample_id = stable_sample_id(self.image_path)
        if self.top1_top2_margin is None:
            self.top1_top2_margin = self.logit_margin


def extract_logits(output) -> torch.Tensor:
    tensor = first_tensor(output, strict=True)
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
    sample_root: Path | None = None,
) -> List[Prediction]:
    model.eval()
    rows: List[Prediction] = []
    seen_ids: dict[str, str] = {}
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
                top1_top2_margin = float(top_logits[index, 0] - (
                    top_logits[index, 1] if top_logits.shape[1] > 1 else 0.))
                target_logit = logits.float()[index, target]
                wrong_logits = logits.float()[index].clone()
                wrong_logits[target] = -torch.inf
                competitor = int(wrong_logits.argmax()) if wrong_logits.numel() > 1 else target
                gt_margin = float(target_logit - wrong_logits[competitor]) if wrong_logits.numel() > 1 else float("inf")
                path, sample_target = samples[offset + index]
                if sample_target != target:
                    raise RuntimeError("dataset order changed between discovery and inference")
                relative = Path(path).resolve()
                if sample_root is not None:
                    try:
                        relative_key = relative.relative_to(Path(sample_root).resolve()).as_posix()
                    except ValueError as error:
                        raise RuntimeError(f"sample path is outside split root: {path}") from error
                else:
                    relative_key = relative.as_posix()
                sample_id = stable_sample_id(relative_key)
                if sample_id in seen_ids and seen_ids[sample_id] != relative_key:
                    raise RuntimeError(f"stable sample_id collision: {sample_id}: {seen_ids[sample_id]} vs {relative_key}")
                seen_ids[sample_id] = relative_key
                input_fingerprint = hashlib.sha1(
                    images[index].detach().float().cpu().numpy().tobytes()).hexdigest()
                rows.append(Prediction(
                    dataset_index=offset + index, image_path=path,
                    true_class=classes[target], true_index=target,
                    pred_class=classes[pred], pred_index=pred,
                    confidence=float(top_values[index, 0]),
                    true_class_probability=float(probabilities[index, target]),
                    logit_margin=top1_top2_margin, correct=pred == target,
                    sample_id=sample_id, top1_top2_margin=top1_top2_margin,
                    gt_margin=gt_margin, fixed_competitor_index=competitor,
                    fixed_competitor_margin=gt_margin,
                    logits=tuple(float(value) for value in logits[index].detach().cpu()),
                    input_fingerprint=input_fingerprint,
                    relative_path=relative_key,
                ))
            offset += images.shape[0]
    write_csv(output_path, [
        {key: value for key, value in asdict(row).items() if key not in {"logits"}}
        for row in rows])
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
    if mode == "paired_error_correct":
        return select_paired_error_correct(predictions, max_samples, seed=1234, class_filter=class_filter)
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


def select_paired_error_correct(
    predictions: Sequence[Prediction], max_samples: int, seed: int = 1234,
    class_filter: Sequence[str] = (),
) -> List[Prediction]:
    """Select a deterministic, error-enriched pool with same-class correct controls."""
    if max_samples < 2:
        raise ValueError("paired_error_correct requires max_samples >= 2")
    allowed = set(class_filter)
    rows = [row for row in predictions if not allowed or row.true_class in allowed or str(row.true_index) in allowed]
    errors: dict[int, list[Prediction]] = {}
    correct: dict[int, list[Prediction]] = {}
    for row in rows:
        (correct if row.correct else errors).setdefault(row.true_index, []).append(row)
    for values in (*errors.values(), *correct.values()):
        values.sort(key=lambda item: (item.confidence, item.sample_id))
    selected: list[Prediction] = []
    classes = sorted(set(errors) | set(correct))
    # Round-robin across classes prevents the largest class from consuming the pool.
    for source in (errors, correct):
        cursor = 0
        while len(selected) < max_samples // 2 and classes:
            added = False
            for class_index in classes:
                values = source.get(class_index, [])
                if cursor < len(values):
                    selected.append(values[cursor])
                    added = True
                    if len(selected) >= max_samples // 2:
                        break
            cursor += 1
            if not added:
                break
    if len(selected) < max_samples:
        remaining = [row for row in rows if row not in selected]
        remaining.sort(key=lambda item: (item.correct, item.confidence, item.sample_id))
        selected.extend(remaining[:max_samples - len(selected)])
    return selected[:max_samples]
