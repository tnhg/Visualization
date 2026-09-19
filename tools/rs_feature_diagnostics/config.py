from __future__ import annotations

import argparse
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
    adapter_config: Optional[Path] = None
    segmentation_mask_dir: Optional[Path] = None
    allow_ambiguous_stages: bool = False
    allow_head_mismatch: bool = False
    use_ema: bool = False
    deploy: bool = False
    amp: bool = False
    seed: int = 1234
    sample_mode: str = "misclassified"
    max_samples: int = 16
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
    p.add_argument("--adapter-config", type=Path)
    p.add_argument("--segmentation-mask-dir", type=Path)
    p.add_argument("--allow-ambiguous-stages", action="store_true")
    p.add_argument("--allow-head-mismatch", action="store_true")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sample-mode", choices=("misclassified", "correct", "all", "class"), default="misclassified")
    p.add_argument("--max-samples", type=int, default=16)
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
    return p


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
        adapter_config=args.adapter_config,
        segmentation_mask_dir=args.segmentation_mask_dir,
        allow_ambiguous_stages=args.allow_ambiguous_stages,
        allow_head_mismatch=args.allow_head_mismatch,
        use_ema=args.use_ema,
        deploy=args.deploy,
        amp=args.amp,
        seed=args.seed,
        sample_mode=args.sample_mode,
        max_samples=args.max_samples,
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
    )
    if cfg.checkpoint and not cfg.checkpoint.is_file():
        raise FileNotFoundError(cfg.checkpoint)
    if cfg.segmentation_mask_dir and not cfg.segmentation_mask_dir.is_dir():
        raise NotADirectoryError(cfg.segmentation_mask_dir)
    if cfg.max_samples < 1 or cfg.max_tokens < 2:
        raise ValueError("sample/token limits must be positive")
    if cfg.scale_batch_size is not None and cfg.scale_batch_size < 1:
        raise ValueError("--scale-batch-size must be positive")
    return cfg
