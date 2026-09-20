"""Command-line entry point for the schema-versioned diagnostics workflow."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from .config import load_yaml_config
from .reporting import build_evidence_package, compare_prediction_runs, render_run


TEMPLATE = """schema_version: 2
experiment: aid_baseline_diagnosis
model_source:
  python_paths: ["${TIMM_REPO}"]
  registration_modules: []
models:
  baseline:
    name: rs_dinet_baseline_l1_d4
    checkpoint: "${BASELINE_CKPT}"
    state: model
    kwargs: {}
  candidate: null
data:
  root: "${AID_DATA_DIR}"
  train_split: train
  val_split: val
preprocess:
  input_size: [3, 224, 224]
  crop_pct: 0.875
  crop_mode: center
  interpolation: bicubic
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
sampling:
  mode: misclassified
  display_pairs: 16
  seed: 824
analysis:
  preset: standard
  stages: [stage2, stage3]
  max_diagnostic_samples: 40
  random_control_repeats: 5
runtime:
  device: cpu
  batch_size: 16
  workers: 2
report:
  language: zh
  detail: standard
  pdf: false
  max_case_pages: 20
output:
  root: ./diagnostic_results
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsdiag", description="离线、可审计的遥感分类诊断工具")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="写入 schema_version=2 配置模板")
    init.add_argument("--output", type=Path, required=True)
    for name in ("doctor", "run"):
        command = sub.add_parser(name, help=f"{name} YAML 配置")
        command.add_argument("--config", type=Path, required=True)
    compare = sub.add_parser("compare", help="按 stable sample_id 比较两个已完成运行")
    compare.add_argument("--config", type=Path, required=True)
    render = sub.add_parser("render", help="离线重绘已有运行，不加载模型")
    render.add_argument("--run", type=Path, required=True)
    summarize = sub.add_parser("summarize", help="生成确定性摘要和外部模型提示词")
    summarize.add_argument("--run", type=Path, required=True)
    summarize.add_argument("--prompt-only", action="store_true")
    return parser


def _doctor(config_path: Path) -> int:
    try:
        cfg, raw = load_yaml_config(config_path)
        checks = {
            "config": "ok", "data_root": str(cfg.data_dir),
            "data_exists": cfg.data_dir.is_dir(),
            "checkpoint": str(cfg.checkpoint) if cfg.checkpoint else None,
            "checkpoint_exists": cfg.checkpoint is None or cfg.checkpoint.is_file(),
            "python_paths": [str(path) for path in cfg.python_paths],
            "python_paths_exist": all(path.is_dir() for path in cfg.python_paths),
            "registration_modules": cfg.registration_modules,
            "device": cfg.device, "analysis": cfg.analysis,
            "plan": {
                "population_forwards": 1,
                "diagnostic_sample_cap": cfg.max_diagnostic_samples,
                "display_sample_cap": cfg.max_samples,
                "amp": cfg.amp, "output_root": str(cfg.output_dir),
            },
        }
        if not checks["data_exists"] or not checks["checkpoint_exists"] or not checks["python_paths_exist"]:
            checks["config"] = "failed"
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if checks["config"] == "ok" else 2
    except Exception as error:
        print(json.dumps({"config": "failed", "reason": str(error)}, ensure_ascii=False, indent=2))
        return 2


def _compare(config_path: Path) -> int:
    import yaml
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    comparison = raw.get("comparison") or {}
    required = {"baseline_run", "candidate_run", "output"}
    missing = sorted(required - set(comparison))
    if missing:
        raise ValueError(f"comparison config missing keys: {missing}")
    base = path.parent
    resolve = lambda value: (base / Path(value)).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    summary = compare_prediction_runs(resolve(comparison["baseline_run"]), resolve(comparison["candidate_run"]), resolve(comparison["output"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        target = args.output.expanduser().resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(TEMPLATE, encoding="utf-8")
        print(target)
        return 0
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "render":
        print(render_run(args.run))
        return 0
    if args.command == "summarize":
        build_evidence_package(args.run)
        prompt = Path(args.run).resolve() / "prompts" / "SUMMARIZE_THIS_RUN.md"
        if args.prompt_only:
            print(prompt.read_text(encoding="utf-8"))
        else:
            print(Path(args.run).resolve() / "SUMMARY.md")
        return 0
    if args.command == "compare":
        return _compare(args.config)
    if args.command == "run":
        cfg, _ = load_yaml_config(args.config)
        # Make external model source paths visible before importing the legacy
        # entrypoint (which imports timm at module load time).
        for source_path in reversed(cfg.python_paths):
            if str(source_path) not in sys.path:
                sys.path.insert(0, str(source_path))
        for module_name in cfg.registration_modules:
            importlib.import_module(module_name)
        from .main import main as legacy_main
        return legacy_main(config=cfg)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
