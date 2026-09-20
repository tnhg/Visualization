from __future__ import annotations

from pathlib import Path
import inspect
import subprocess
from typing import Any, Dict, Optional

import torch

from .checkpoint import CheckpointReport, classifier_shapes
from .config import DiagnosticConfig
from .dataset import DatasetInfo
from .model_inspector import InspectionResult
from .utils import environment_manifest, git_commit, sha256_file, write_json


def build_manifest(
    cfg: DiagnosticConfig,
    repo_root: Path,
    output_dir: Path,
    device: torch.device,
    model,
    data_config: Dict[str, Any],
    dataset_info: DatasetInfo,
    inspection: InspectionResult,
    checkpoint_report: Optional[CheckpointReport],
) -> Dict[str, Any]:
    manifest = environment_manifest(repo_root, device)
    hook_names = list(dict.fromkeys(
        ([inspection.stem] if inspection.stem else []) + inspection.stages))
    manifest.update({
        "model_name": cfg.model,
        "model_kwargs": cfg.model_kwargs,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "classifier_shapes": classifier_shapes(model),
        "checkpoint_path": str(cfg.checkpoint) if cfg.checkpoint else None,
        "checkpoint_sha256": sha256_file(cfg.checkpoint) if cfg.checkpoint else None,
        "checkpoint_report": checkpoint_report.__dict__ if checkpoint_report else None,
        "dataset_path": str(dataset_info.root),
        "dataset_split_path": str(dataset_info.split_root),
        "dataset_class_mapping": dataset_info.class_to_idx,
        "dataset_sample_count": len(dataset_info.samples),
        "dataset_train_count": dataset_info.train_count,
        "input_size": data_config["input_size"],
        "mean": data_config["mean"],
        "std": data_config["std"],
        "crop_pct": data_config["crop_pct"],
        "crop_mode": data_config.get("crop_mode"),
        "interpolation": data_config["interpolation"],
        "sample_selection": {
            "mode": cfg.sample_mode, "max_samples": cfg.max_samples,
            "class_filter": cfg.class_filter, "top_k_errors": cfg.top_k_errors,
        },
        "random_seed": cfg.seed,
        "analysis_settings": cfg.to_dict(),
        "analysis_structure": "deployment" if cfg.deploy else "training/analysis",
        "hooked_module_names": hook_names,
        "feature_shapes": {
            name: next((record.output_shape for record in reversed(inspection.records) if record.name == name), None)
            for name in hook_names
        },
        "missing_semantic_ops": [key for key, value in inspection.semantic_ops.items() if value == "not_available"],
        "geometry_version": "unified_preprocessor_v1",
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_class_source": inspect.getsourcefile(type(model)),
        "model_source": {
            "python_paths": [str(path) for path in cfg.python_paths],
            "registration_modules": list(cfg.registration_modules),
            "git_commits": {
                str(path): git_commit(Path(path)) for path in cfg.python_paths if Path(path).is_dir()
            },
        },
        "tool_repo_dirty": _git_dirty(repo_root),
        "model_kwargs": cfg.model_kwargs,
        "numeric_mode": {"amp": cfg.amp, "dtype": "float16_autocast" if cfg.amp else "float32"},
    })
    write_json(output_dir / "manifest" / "run_manifest.json", manifest)
    return manifest


def _git_dirty(repo: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True)
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None
