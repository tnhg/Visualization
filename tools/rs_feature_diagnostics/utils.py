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
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


LOG = logging.getLogger("rs_feature_diagnostics")
OUTPUT_SUBDIRS = (
    "manifest", "predictions", "classification", "samples", "gradcam", "attention", "spatial", "scale",
    "erf", "channel", "interventions", "paper_figures", "tables", "metrics", "figures",
    "cases", "raw", "prompts",
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


def artifact_id(value: str) -> str:
    """Return a filesystem identifier separate from the human-readable label.

    Dots are deliberately replaced so a module such as ``blocks.5`` can never
    be interpreted as a file suffix by callers constructing artifact paths.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value)).strip("._") or "artifact"


def artifact_path(base: Path, extension: str) -> Path:
    """Append an extension without treating dots in a display name as suffixes."""
    extension = extension if extension.startswith(".") else f".{extension}"
    return base.parent / f"{base.name}{extension}"


def stable_sample_id(relative_path: str) -> str:
    """Create a deterministic ID from a split-relative POSIX path.

    This is an association key, not a security digest.  The normalized path is
    retained separately in prediction records so the ID remains auditable.
    """
    normalized = Path(str(relative_path).replace("\\", "/")).as_posix()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"s_{digest}"


def semantic_stage_label(module_name: str, sequence_index: int) -> str:
    """Return an artifact label without counting the stem as a network stage."""
    if module_name == "stem" or module_name.startswith("stem."):
        return "stem"
    match = re.search(r"(?:^|\.)stages\.(\d+)(?:\.|$)", module_name)
    if match:
        return f"stage{int(match.group(1)) + 1}"
    return f"feature{sequence_index + 1}_{safe_name(module_name)}"


def prepare_output(root: Path, dataset: str, model: str, run_id: str | None, resume: bool = False) -> Path:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = root.expanduser().resolve() / safe_name(dataset) / safe_name(model) / safe_name(run_id)
    if output.exists() and not resume:
        base = output
        index = 1
        while output.exists():
            output = base.parent / f"{base.name}_{index:02d}"
            index += 1
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


def _json_safe(value: Any) -> Any:
    """Recursively make values legal JSON without turning invalid numbers into 0."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                    dir=path.parent, prefix=f".{path.name}.",
                                    suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, data: Any) -> None:
    payload = _json_safe(data)
    _atomic_write(path, json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: Sequence[str] | None = None,
) -> None:
    """Write a deterministic union-schema CSV without silently dropping fields."""
    if schema is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row.keys():
                key = str(key)
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        if not fields:
            fields = ["status"]
    else:
        fields = list(dict.fromkeys(str(item) for item in schema))
    rendered = []
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        normalized = {field: _csv_value(row.get(field)) for field in fields}
        extra = set(row.keys()) - set(fields)
        if extra:
            raise ValueError(f"CSV row contains fields outside schema: {sorted(extra)}")
        writer.writerow(normalized)
    _atomic_write(path, buffer.getvalue())


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, (np.generic, torch.Tensor)):
        value = _json_safe(value)
    return value


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
        "timm_file": getattr(timm, "__file__", None),
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
