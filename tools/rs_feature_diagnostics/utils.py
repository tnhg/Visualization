from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import logging
import os
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


LOG = logging.getLogger("rs_feature_diagnostics")
OUTPUT_SUBDIRS = (
    "manifest", "predictions", "classification", "samples", "gradcam", "attention", "spatial", "scale",
    "erf", "channel", "interventions", "paper_figures", "tables",
)


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def semantic_stage_label(module_name: str, sequence_index: int) -> str:
    """Return an artifact label without counting the stem as a network stage."""
    if module_name == "stem" or module_name.startswith("stem."):
        return "stem"
    match = re.search(r"(?:^|\.)stages\.(\d+)(?:\.|$)", module_name)
    if match:
        return f"stage{int(match.group(1)) + 1}"
    return f"feature{sequence_index + 1}_{safe_name(module_name)}"


def prepare_output(root: Path, dataset: str, model: str, run_id: str | None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output = root.expanduser().resolve() / safe_name(dataset) / safe_name(model) / safe_name(run_id)
    for name in OUTPUT_SUBDIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(repo: Path, device: torch.device) -> Dict[str, Any]:
    import timm
    result: Dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(repo),
        "torch_version": torch.__version__,
        "timm_version": getattr(timm, "__version__", "unknown"),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["gpu_capability"] = torch.cuda.get_device_capability(device)
    else:
        result["gpu_name"] = None
    return result


def resolve_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {spec}")
    return device


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def release_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
