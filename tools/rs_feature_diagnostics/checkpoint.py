from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn


LOG = logging.getLogger("rs_feature_diagnostics")
PREFIXES = ("module.", "model.", "_orig_mod.")
STATE_KEYS = ("state_dict", "model", "model_state_dict", "state_dict_ema", "model_ema")


@dataclass
class CheckpointReport:
    path: str
    selected_key: str
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    skipped_head_keys: list[str] = field(default_factory=list)


def classifier_module_names(model: nn.Module) -> list[str]:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        return []
    ids = {id(module) for module in classifier.modules()}
    return [name for name, module in model.named_modules() if id(module) in ids and name]


def classifier_shapes(model: nn.Module) -> Dict[str, Tuple[int, ...]]:
    names = classifier_module_names(model)
    return {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if any(key == name or key.startswith(name + ".") for name in names)
    }


def _load_raw(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        LOG.warning("weights_only checkpoint load failed (%s); falling back to legacy pickle load", error)
        return torch.load(path, map_location="cpu", weights_only=False)


def _select_state_dict(payload: Any, use_ema: bool) -> tuple[Mapping[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping or state_dict")
    priority = ("state_dict_ema", "model_ema", "state_dict", "model", "model_state_dict") if use_ema else (
        "state_dict", "model", "model_state_dict", "state_dict_ema", "model_ema")
    for key in priority:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate and all(isinstance(v, torch.Tensor) for v in candidate.values()):
            return candidate, key
    if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload, "root"
    raise KeyError(f"no model weights found; supported keys: {STATE_KEYS}")


def _strip_prefixes(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def load_checkpoint(
    model: nn.Module,
    path: Path,
    num_classes: int,
    allow_head_mismatch: bool = False,
    use_ema: bool = False,
) -> CheckpointReport:
    payload = _load_raw(path)
    selected, selected_key = _select_state_dict(payload, use_ema)
    state = _strip_prefixes(selected)
    target = model.state_dict()
    classifier_names = classifier_module_names(model)

    mismatches = {
        key: (tuple(value.shape), tuple(target[key].shape))
        for key, value in state.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    }
    head_mismatches = {
        key: shapes for key, shapes in mismatches.items()
        if any(key == name or key.startswith(name + ".") for name in classifier_names)
    }
    non_head = {key: shapes for key, shapes in mismatches.items() if key not in head_mismatches}
    if non_head:
        raise RuntimeError(f"non-classifier checkpoint shape mismatch: {non_head}")
    if head_mismatches and not allow_head_mismatch:
        raise RuntimeError(
            "classifier mismatch: "
            f"checkpoint shapes={head_mismatches}, dataset num_classes={num_classes}, "
            f"current classifier shapes={classifier_shapes(model)}. "
            "Use --allow-head-mismatch only when intentionally discarding the checkpoint head."
        )
    skipped = list(head_mismatches)
    for key in skipped:
        state.pop(key)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_prefixes = tuple(name + "." for name in classifier_names)
    missing_non_head = [key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)]
    missing_head = [key for key in incompatible.missing_keys if key.startswith(allowed_prefixes)]
    if missing_head and not allow_head_mismatch:
        raise RuntimeError(
            "classifier keys are missing from the checkpoint: "
            f"{missing_head}; dataset num_classes={num_classes}, "
            f"current classifier shapes={classifier_shapes(model)}. "
            "Use --allow-head-mismatch only when intentionally using a newly initialized head."
        )
    if missing_non_head or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint key mismatch: missing={missing_non_head}, unexpected={incompatible.unexpected_keys}"
        )
    return CheckpointReport(
        path=str(path), selected_key=selected_key,
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        skipped_head_keys=skipped,
    )
