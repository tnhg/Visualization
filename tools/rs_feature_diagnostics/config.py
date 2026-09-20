from __future__ import annotations

import argparse
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ANALYSES = (
    "structure", "inference", "classification", "gradcam", "attention", "spatial", "scale", "erf",
    "channel", "interventions", "all",
)


def _kv_pair(value: str) -> Tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model kwargs must use key=value")
    key, raw = value.split("=", 1)
    if raw.lower() in ("true", "false"):
        parsed: Any = raw.lower() == "true"
    elif raw.lower() in ("none", "null"):
        parsed = None
    else:
        try:
            parsed = int(raw)
        except ValueError:
            try:
                parsed = float(raw)
            except ValueError:
                parsed = raw
    return key, parsed


@dataclass
class DiagnosticConfig:
    model: str
    data_dir: Path
    checkpoint: Optional[Path] = None
    pretrained: bool = False
    train_split: str = "train"
    val_split: str = "val"
    num_classes: Optional[int] = None
    input_size: Optional[Tuple[int, int, int]] = None
    batch_size: int = 32
    workers: int = 4
    device: str = "cuda:0"
    output_dir: Path = Path("./visualization_results")
    run_id: Optional[str] = None
    analysis: List[str] = field(default_factory=lambda: ["structure"])
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    mean: Optional[Tuple[float, ...]] = None
    std: Optional[Tuple[float, ...]] = None
    interpolation: Optional[str] = None
    crop_pct: Optional[float] = None
    crop_mode: Optional[str] = None
    adapter_config: Optional[Path] = None
    segmentation_mask_dir: Optional[Path] = None
    allow_ambiguous_stages: bool = False
    allow_head_mismatch: bool = False
    use_ema: bool = False
    trust_checkpoint: bool = False
    deploy: bool = False
    amp: bool = False
    seed: int = 1234
    sample_mode: str = "misclassified"
    max_samples: int = 16
    max_diagnostic_samples: int = 128
    class_filter: List[str] = field(default_factory=list)
    top_k_errors: int = 50
    max_tokens: int = 512
    similarity_samples: int = 64
    intervention_samples: int = 128
    keep_ratios: Tuple[float, ...] = (1.0, .9, .75, .5, .25, .1)
    channel_keep_ratios: Tuple[float, ...] = (1.0, .75, .5, .25, .0)
    replacement: str = "zero"
    scales: Tuple[float, ...] = (.5, .75, 1.0, 1.5, 2.0)
    scale_eval_scope: str = "full"
    scale_batch_size: Optional[int] = None
    entropy_temperature: float = 1.0
    texture_quantile: float = .3
    channel_correlation_threshold: float = .9
    erf_channels_per_stage: int = 16
    all_channel_erf: bool = False
    erf_location: str = "center"
    gradcam_layer: Optional[str] = None
    gradcam_target: str = "both"
    gradcam_class_index: Optional[int] = None
    resume: bool = False
    python_paths: List[Path] = field(default_factory=list)
    registration_modules: List[str] = field(default_factory=list)
    schema_version: int = 1
    stages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
        return result


def _input_size(values: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int]]:
    if values is None:
        return None
    if len(values) == 1:
        return 3, values[0], values[0]
    if len(values) == 3:
        return tuple(values)  # type: ignore[return-value]
    raise ValueError("--input-size accepts H or C H W")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--num-classes", type=int)
    p.add_argument("--input-size", type=int, nargs="+")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=Path("./visualization_results"))
    p.add_argument("--run-id")
    p.add_argument("--analysis", nargs="+", choices=ANALYSES, default=["structure"])
    p.add_argument("--model-kwargs", nargs="*", type=_kv_pair, default=[])
    p.add_argument("--mean", type=float, nargs="+")
    p.add_argument("--std", type=float, nargs="+")
    p.add_argument("--interpolation")
    p.add_argument("--crop-pct", type=float)
    p.add_argument("--crop-mode", choices=("center", "resize"))
    p.add_argument("--adapter-config", type=Path)
    p.add_argument("--segmentation-mask-dir", type=Path)
    p.add_argument("--allow-ambiguous-stages", action="store_true")
    p.add_argument("--allow-head-mismatch", action="store_true")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--trust-checkpoint", action="store_true",
                   help="allow legacy pickle loading after weights_only fails")
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sample-mode", choices=("misclassified", "correct", "all", "class", "paired_error_correct"), default="misclassified")
    p.add_argument("--max-samples", type=int, default=16)
    p.add_argument("--max-diagnostic-samples", type=int, default=128)
    p.add_argument("--class-filter", nargs="*", default=[])
    p.add_argument("--top-k-errors", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-tokens-for-similarity", type=int, dest="max_tokens")
    p.add_argument("--similarity-samples", type=int, default=64)
    p.add_argument("--intervention-samples", type=int, default=128)
    p.add_argument("--keep-ratios", type=float, nargs="+", default=(1., .9, .75, .5, .25, .1))
    p.add_argument("--channel-keep-ratios", type=float, nargs="+", default=(1., .75, .5, .25, 0.))
    p.add_argument("--replacement", choices=("zero", "mean", "cluster_mean"), default="zero")
    p.add_argument("--scales", type=float, nargs="+", default=(.5, .75, 1., 1.5, 2.))
    p.add_argument("--scale-eval-scope", choices=("selected", "full"), default="full")
    p.add_argument("--scale-batch-size", type=int)
    p.add_argument("--entropy-temperature", type=float, default=1.)
    p.add_argument("--texture-quantile", type=float, default=.3)
    p.add_argument("--channel-correlation-threshold", type=float, default=.9)
    p.add_argument("--erf-channels-per-stage", type=int, default=16)
    p.add_argument("--all-channel-erf", action="store_true")
    p.add_argument("--erf-location", choices=("center", "max"), default="center")
    p.add_argument("--gradcam-layer")
    p.add_argument("--gradcam-target", choices=("predicted", "true", "both", "index"), default="both")
    p.add_argument("--gradcam-class-index", type=int)
    p.add_argument("--stages", nargs="*", default=[])
    p.add_argument("--resume", action="store_true")
    return p


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"environment variable {name} is not defined")
            return os.environ[name]
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _reject_unknown(mapping: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown config key(s) at {where}: {unknown}")


def _resolve_config_path(value: Any, base: Path, *, required: bool = False) -> Optional[Path]:
    if value in (None, ""):
        if required:
            raise ValueError("required path is empty")
        return None
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _preset_analyses(preset: str) -> list[str]:
    presets = {
        "quick": ["inference", "classification"],
        "standard": ["inference", "classification", "spatial", "scale"],
        "deep": ["inference", "classification", "gradcam", "attention", "spatial", "scale", "erf", "channel", "interventions"],
    }
    if preset not in presets:
        raise ValueError(f"analysis.preset must be quick, standard, or deep, got {preset!r}")
    return presets[preset]


def load_yaml_config(path: Path) -> tuple[DiagnosticConfig, dict[str, Any]]:
    """Load schema-versioned YAML while preserving the legacy CLI config."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("YAML configs require PyYAML; install it in an isolated environment") from error
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _expand_env(raw)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    version = int(raw.get("schema_version", 0))
    if version != 2:
        raise ValueError(f"unsupported schema_version={version}; expected 2")
    _reject_unknown(raw, {"schema_version", "experiment", "model_source", "models", "data", "preprocess", "sampling", "analysis", "runtime", "report", "output", "comparison"}, "root")
    source = raw.get("model_source") or {}
    _reject_unknown(source, {"python_paths", "registration_modules"}, "model_source")
    models = raw.get("models") or {}
    _reject_unknown(models, {"baseline", "candidate"}, "models")
    baseline = models.get("baseline") or {}
    _reject_unknown(baseline, {"name", "checkpoint", "state", "kwargs", "pretrained", "trust_checkpoint"}, "models.baseline")
    if not baseline.get("name"):
        raise ValueError("models.baseline.name is required")
    if baseline.get("state", "model") not in {"model", "ema"}:
        raise ValueError("models.baseline.state must be model or ema")
    data = raw.get("data") or {}
    _reject_unknown(data, {"root", "train_split", "val_split", "mask_dir", "num_classes"}, "data")
    preprocess = raw.get("preprocess") or {}
    _reject_unknown(preprocess, {"input_size", "crop_pct", "crop_mode", "interpolation", "mean", "std"}, "preprocess")
    sampling = raw.get("sampling") or {}
    _reject_unknown(sampling, {"mode", "display_pairs", "seed", "max_samples", "top_k_errors", "class_filter"}, "sampling")
    analysis = raw.get("analysis") or {}
    _reject_unknown(analysis, {"preset", "tasks", "stages", "max_diagnostic_samples", "random_control_repeats"}, "analysis")
    runtime = raw.get("runtime") or {}
    _reject_unknown(runtime, {"device", "batch_size", "workers", "amp", "deploy"}, "runtime")
    output = raw.get("output") or {}
    _reject_unknown(output, {"root", "run_id", "resume"}, "output")
    base = path.parent
    input_size = preprocess.get("input_size", [3, 224, 224])
    if len(input_size) == 2:
        input_size = [3, *input_size]
    if len(input_size) != 3:
        raise ValueError("preprocess.input_size must contain C,H,W or H,W")
    if any(int(value) <= 0 for value in input_size):
        raise ValueError("preprocess.input_size values must be positive")
    mean = tuple(float(x) for x in preprocess.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in preprocess.get("std", [0.229, 0.224, 0.225]))
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("preprocess mean/std must contain three values and positive std")
    if not 0 < float(preprocess.get("crop_pct", .875)) <= 1:
        raise ValueError("preprocess.crop_pct must be in (0, 1]")
    if str(preprocess.get("crop_mode", "center")) not in {"center", "resize"}:
        raise ValueError("preprocess.crop_mode must be center or resize")
    sample_mode = str(sampling.get("mode", "misclassified"))
    if sample_mode not in {"misclassified", "correct", "all", "class", "paired_error_correct"}:
        raise ValueError(f"sampling.mode is unsupported: {sample_mode}")
    tasks = analysis.get("tasks") or _preset_analyses(str(analysis.get("preset", "standard")))
    if not tasks or any(item not in ANALYSES[:-1] for item in tasks):
        raise ValueError(f"analysis.tasks contains unsupported task: {tasks}")
    checkpoint = _resolve_config_path(baseline.get("checkpoint"), base)
    cfg = DiagnosticConfig(
        model=str(baseline["name"]), data_dir=_resolve_config_path(data.get("root"), base, required=True),
        checkpoint=checkpoint, pretrained=bool(baseline.get("pretrained", False)),
        train_split=str(data.get("train_split", "train")), val_split=str(data.get("val_split", "val")),
        num_classes=data.get("num_classes"), input_size=tuple(int(x) for x in input_size),
        batch_size=int(runtime.get("batch_size", 16)), workers=int(runtime.get("workers", 2)),
        device=str(runtime.get("device", "cpu")), output_dir=_resolve_config_path(output.get("root", "./diagnostic_results"), base, required=True),
        run_id=output.get("run_id"), analysis=list(dict.fromkeys(tasks)),
        model_kwargs=dict(baseline.get("kwargs") or {}), mean=mean, std=std,
        interpolation=str(preprocess.get("interpolation", "bicubic")), crop_pct=float(preprocess.get("crop_pct", .875)),
        crop_mode=str(preprocess.get("crop_mode", "center")),
        segmentation_mask_dir=_resolve_config_path(data.get("mask_dir"), base),
        allow_ambiguous_stages=False, amp=bool(runtime.get("amp", False)), deploy=bool(runtime.get("deploy", False)),
        use_ema=str(baseline.get("state", "model")) == "ema",
        trust_checkpoint=bool(baseline.get("trust_checkpoint", False)),
        seed=int(sampling.get("seed", 1234)), sample_mode=sample_mode,
        max_samples=int(sampling.get("max_samples", sampling.get("display_pairs", 16))),
        max_diagnostic_samples=int(analysis.get("max_diagnostic_samples", 128)),
        class_filter=list(sampling.get("class_filter", [])), top_k_errors=int(sampling.get("top_k_errors", 50)),
        python_paths=[_resolve_config_path(item, base, required=True) for item in source.get("python_paths", [])],
        registration_modules=list(source.get("registration_modules", [])), schema_version=2,
        resume=bool(output.get("resume", False)),
        stages=list(analysis.get("stages", [])),
    )
    # YAML users expect a failed configuration before model/data startup.
    parse_config_validation(cfg)
    return cfg, raw


def parse_config_validation(cfg: DiagnosticConfig) -> None:
    if cfg.checkpoint and not cfg.checkpoint.is_file():
        raise FileNotFoundError(cfg.checkpoint)
    if not cfg.data_dir.is_dir():
        raise NotADirectoryError(cfg.data_dir)
    if cfg.segmentation_mask_dir and not cfg.segmentation_mask_dir.is_dir():
        raise NotADirectoryError(cfg.segmentation_mask_dir)
    if cfg.max_samples < 1 or cfg.max_diagnostic_samples < 1 or cfg.batch_size < 1 or cfg.workers < 0:
        raise ValueError("sample limits and batch_size must be positive; workers cannot be negative")
    if any(not 0 <= float(x) <= 1 for x in (*cfg.keep_ratios, *cfg.channel_keep_ratios)):
        raise ValueError("keep ratios must be in [0, 1]")
    if any(float(x) <= 0 for x in cfg.scales):
        raise ValueError("scales must be positive")


def parse_config(argv: Optional[Sequence[str]] = None) -> DiagnosticConfig:
    args = build_parser().parse_args(argv)
    analyses: List[str] = []
    for item in args.analysis:
        analyses.extend(x.strip() for x in item.split(",") if x.strip())
    if "all" in analyses:
        analyses = [
            "inference", "classification", "gradcam", "attention", "spatial", "scale", "erf",
            "channel", "interventions",
        ]
    cfg = DiagnosticConfig(
        model=args.model,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        data_dir=args.data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        num_classes=args.num_classes,
        input_size=_input_size(args.input_size),
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        output_dir=args.output_dir,
        run_id=args.run_id,
        analysis=analyses,
        model_kwargs=dict(args.model_kwargs),
        mean=tuple(args.mean) if args.mean else None,
        std=tuple(args.std) if args.std else None,
        interpolation=args.interpolation,
        crop_pct=args.crop_pct,
        crop_mode=args.crop_mode,
        adapter_config=args.adapter_config,
        segmentation_mask_dir=args.segmentation_mask_dir,
        allow_ambiguous_stages=args.allow_ambiguous_stages,
        allow_head_mismatch=args.allow_head_mismatch,
        use_ema=args.use_ema,
        trust_checkpoint=args.trust_checkpoint,
        deploy=args.deploy,
        amp=args.amp,
        seed=args.seed,
        sample_mode=args.sample_mode,
        max_samples=args.max_samples,
        max_diagnostic_samples=args.max_diagnostic_samples,
        class_filter=args.class_filter,
        top_k_errors=args.top_k_errors,
        max_tokens=args.max_tokens,
        similarity_samples=args.similarity_samples,
        intervention_samples=args.intervention_samples,
        keep_ratios=tuple(args.keep_ratios),
        channel_keep_ratios=tuple(args.channel_keep_ratios),
        replacement=args.replacement,
        scales=tuple(args.scales),
        scale_eval_scope=args.scale_eval_scope,
        scale_batch_size=args.scale_batch_size,
        entropy_temperature=args.entropy_temperature,
        texture_quantile=args.texture_quantile,
        channel_correlation_threshold=args.channel_correlation_threshold,
        erf_channels_per_stage=args.erf_channels_per_stage,
        all_channel_erf=args.all_channel_erf,
        erf_location=args.erf_location,
        gradcam_layer=args.gradcam_layer,
        gradcam_target=args.gradcam_target,
        gradcam_class_index=args.gradcam_class_index,
        resume=args.resume,
        stages=list(args.stages),
    )
    if cfg.checkpoint and not cfg.checkpoint.is_file():
        raise FileNotFoundError(cfg.checkpoint)
    if cfg.segmentation_mask_dir and not cfg.segmentation_mask_dir.is_dir():
        raise NotADirectoryError(cfg.segmentation_mask_dir)
    if cfg.max_samples < 1 or cfg.max_diagnostic_samples < 1 or cfg.max_tokens < 2:
        raise ValueError("sample/token limits must be positive")
    if cfg.scale_batch_size is not None and cfg.scale_batch_size < 1:
        raise ValueError("--scale-batch-size must be positive")
    for name, ratios in (("keep-ratios", cfg.keep_ratios), ("channel-keep-ratios", cfg.channel_keep_ratios)):
        if any(not 0 <= float(value) <= 1 for value in ratios):
            raise ValueError(f"--{name} values must be in [0, 1]")
    if any(float(scale) <= 0 for scale in cfg.scales):
        raise ValueError("--scales values must be positive")
    if not cfg.analysis:
        raise ValueError("at least one analysis is required")
    return cfg
