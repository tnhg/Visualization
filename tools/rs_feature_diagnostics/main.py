#!/usr/bin/env python3
from __future__ import annotations

# Allow the documented `python tools/rs_feature_diagnostics/main.py` entrypoint.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from dataclasses import asdict
from typing import Optional

import torch
import timm
from timm.data import resolve_model_data_config

from tools.rs_feature_diagnostics.adapters import GenericTimmAdapter
from tools.rs_feature_diagnostics.attention_rollout import analyze_attention_rollout_samples
from tools.rs_feature_diagnostics.channel_analysis import analyze_channel
from tools.rs_feature_diagnostics.classification_analysis import analyze_classification
from tools.rs_feature_diagnostics.checkpoint import classifier_module_names, load_checkpoint
from tools.rs_feature_diagnostics.config import DiagnosticConfig, parse_config
from tools.rs_feature_diagnostics.dataset import build_eval_dataset, build_loader, discover_dataset
from tools.rs_feature_diagnostics.erf import analyze_erf_samples
from tools.rs_feature_diagnostics.failure_analysis import failure_panels
from tools.rs_feature_diagnostics.gradcam import analyze_gradcam_samples
from tools.rs_feature_diagnostics.intervention import (
    channel_deletion_curve, plot_sample_intervention_curves, spatial_deletion_curve,
)
from tools.rs_feature_diagnostics.manifest import build_manifest
from tools.rs_feature_diagnostics.model_inspector import InspectionResult, ModelInspector
from tools.rs_feature_diagnostics.paper_figures import figure_a, figure_b, figure_c
from tools.rs_feature_diagnostics.sample_selector import run_inference, select_predictions
from tools.rs_feature_diagnostics.scale_analysis import ScaleAnalysisResult, analyze_scale
from tools.rs_feature_diagnostics.spatial_analysis import analyze_spatial
from tools.rs_feature_diagnostics.utils import (
    LOG, prepare_output, release_cuda, resolve_device, safe_name, set_seed,
    setup_logging, write_csv, write_json,
)


def _data_overrides(cfg: DiagnosticConfig) -> dict:
    result = {}
    for key in ("input_size", "mean", "std", "interpolation", "crop_pct"):
        value = getattr(cfg, key)
        if value is not None:
            result[key] = value
    return result


def print_preflight(
    cfg: DiagnosticConfig,
    dataset_info,
    data_config: dict,
    model,
    inspection: InspectionResult,
    checkpoint_report,
) -> None:
    record_map = {record.name: record for record in inspection.records}
    print("\n=== REMOTE-SENSING FEATURE DIAGNOSTICS PREFLIGHT ===")
    print(f"MODEL: {cfg.model} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"DATASET: {dataset_info.split_root} ({len(dataset_info.samples)} samples, {dataset_info.num_classes} classes)")
    print(f"CHECKPOINT: {cfg.checkpoint or ('ImageNet pretrained' if cfg.pretrained else 'none / random initialization')}")
    print(f"CHECKPOINT STATE: {checkpoint_report.selected_key if checkpoint_report else 'not applicable'}")
    print(f"CLASSIFIER: {classifier_module_names(model) or ['not_available']}")
    print(f"INPUT: {data_config['input_size']}, mean={data_config['mean']}, std={data_config['std']}")
    print(f"STEM: {inspection.stem or 'not_available'}")
    print(f"STAGES ({inspection.source}):")
    for index, stage in enumerate(inspection.stages, 1):
        record = record_map.get(stage)
        print(f"  Stage {index}: {stage} -> {record.output_shape if record else 'shape unavailable'}")
    hook_points = list(dict.fromkeys(
        ([inspection.stem] if inspection.stem else []) + inspection.stages))
    print(f"HOOK POINTS: {hook_points}")
    print(f"REPRESENTATIVE BLOCKS: {inspection.representative_blocks or ['not_available']}")
    print("SEMANTIC OPS:")
    for key, value in inspection.semantic_ops.items():
        print(f"  {key}: {value}")
    print(f"MISSING SEMANTIC OPS: {[key for key, value in inspection.semantic_ops.items() if value == 'not_available'] or 'none'}")
    print("========================================================\n")


def main(argv=None) -> int:
    setup_logging()
    cfg = parse_config(argv)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    dataset_info = discover_dataset(cfg.data_dir, cfg.val_split, cfg.train_split)
    if cfg.num_classes is not None and cfg.num_classes != dataset_info.num_classes:
        raise ValueError(
            f"--num-classes={cfg.num_classes} disagrees with ImageFolder class count={dataset_info.num_classes}")
    num_classes = cfg.num_classes or dataset_info.num_classes
    model = timm.create_model(
        cfg.model, pretrained=cfg.pretrained, num_classes=num_classes, **cfg.model_kwargs)
    checkpoint_report = None
    if cfg.checkpoint:
        checkpoint_report = load_checkpoint(
            model, cfg.checkpoint, num_classes,
            allow_head_mismatch=cfg.allow_head_mismatch, use_ema=cfg.use_ema)
    elif not cfg.pretrained:
        LOG.warning("no checkpoint and pretrained=False: predictions will use random weights")

    adapter = GenericTimmAdapter.from_model(model, cfg.adapter_config)
    if cfg.deploy:
        LOG.warning("deploy mode requested: reparameterizing before inspection; pre-fusion hook names are discarded")
        model = adapter.apply_deploy(model)
        adapter = GenericTimmAdapter.from_model(model, cfg.adapter_config)
    model.eval().to(device)
    data_config = resolve_model_data_config(model, args=_data_overrides(cfg), verbose=False)
    data_config["input_size"] = tuple(data_config["input_size"])

    output_dir = prepare_output(cfg.output_dir, cfg.data_dir.name, cfg.model, cfg.run_id)
    inspector = ModelInspector(model, data_config["input_size"], device)
    inspection = inspector.inspect(adapter)
    inspector.save(inspection, output_dir)
    print_preflight(cfg, dataset_info, data_config, model, inspection, checkpoint_report)
    build_manifest(
        cfg, REPO_ROOT, output_dir, device, model, data_config,
        dataset_info, inspection, checkpoint_report)
    if inspection.ambiguous and not cfg.allow_ambiguous_stages:
        message = (
            "automatic stage detection is ambiguous; inspect model_structure.txt and provide "
            "--adapter-config. Refusing to run feature analyses silently."
        )
        if cfg.analysis == ["structure"]:
            LOG.warning(message)
            return 0
        raise RuntimeError(message)
    if cfg.analysis == ["structure"]:
        print(f"Preflight artifacts: {output_dir}")
        return 0

    dataset = build_eval_dataset(dataset_info, data_config)
    loader = build_loader(dataset, cfg.batch_size, cfg.workers, device)
    predictions = run_inference(
        model, loader, dataset_info.classes, dataset_info.samples, device, cfg.amp,
        output_dir / "predictions" / "predictions.csv")
    if "classification" in cfg.analysis:
        analyze_classification(predictions, dataset_info.classes, output_dir)
    selected = select_predictions(
        predictions, cfg.sample_mode, cfg.max_samples, cfg.class_filter, cfg.top_k_errors)
    write_csv(output_dir / "predictions" / "selected_samples.csv", [asdict(item) for item in selected])
    if not selected:
        LOG.warning("sample selection returned no samples; sample-level analyses will be skipped")

    spatial_rows: list[dict] = []
    scale_result: Optional[ScaleAnalysisResult] = None
    erf_rows: list[dict] = []
    channel_rows: list[dict] = []
    channel_erf_rows: list[dict] = []
    spatial_deletion_rows: list[dict] = []
    channel_deletion_rows: list[dict] = []
    spatial_deletion_sample_rows: list[dict] = []
    channel_deletion_sample_rows: list[dict] = []
    mean_channel_matrices = {}
    feature_layers = list(dict.fromkeys(
        ([inspection.stem] if inspection.stem else []) + inspection.stages))

    gradcam_layer = cfg.gradcam_layer or (inspection.stages[-1] if inspection.stages else None)
    if "gradcam" in cfg.analysis and selected and gradcam_layer:
        rows = analyze_gradcam_samples(
            model, dataset, selected, gradcam_layer, device, output_dir,
            cfg.gradcam_target, cfg.gradcam_class_index, adapter.layouts.get(gradcam_layer),
            cfg.scales if "scale" in cfg.analysis else None, data_config["input_size"],
            data_config["mean"], data_config["std"], data_config["crop_pct"],
            data_config["interpolation"])
        write_csv(output_dir / "tables" / "gradcam_manifest.csv", rows)
        release_cuda()

    if "attention" in cfg.analysis and selected:
        analyze_attention_rollout_samples(
            model, dataset, selected, device, output_dir, data_config["input_size"])
        release_cuda()

    if "spatial" in cfg.analysis and selected:
        spatial_rows = analyze_spatial(
            model, dataset, selected, feature_layers, device, output_dir,
            data_config["input_size"], data_config["crop_pct"], data_config["interpolation"],
            cfg.max_tokens, cfg.similarity_samples, cfg.seed, cfg.entropy_temperature,
            cfg.texture_quantile, cfg.segmentation_mask_dir, dataset_info.split_root,
            adapter.layouts)
        release_cuda()

    need_scale = "scale" in cfg.analysis or "channel" in cfg.analysis or "interventions" in cfg.analysis
    if need_scale and selected:
        scale_population = predictions if cfg.scale_eval_scope == "full" else selected
        scale_result = analyze_scale(
            model, selected, scale_population, dataset_info.classes,
            feature_layers, device, output_dir,
            data_config["input_size"], data_config["mean"], data_config["std"],
            data_config["crop_pct"], data_config["interpolation"], cfg.scales,
            cfg.scale_batch_size or cfg.batch_size, cfg.amp, adapter.layouts)
        release_cuda()

    if "erf" in cfg.analysis and selected:
        erf_rows = analyze_erf_samples(
            model, dataset, selected, feature_layers, device, output_dir,
            cfg.erf_location, adapter.layouts)
        release_cuda()

    need_channel = "channel" in cfg.analysis or "interventions" in cfg.analysis
    if need_channel and selected and scale_result is not None:
        channel_rows, mean_channel_matrices, channel_erf_rows = analyze_channel(
            model, dataset, selected, feature_layers, scale_result, device, output_dir,
            cfg.channel_correlation_threshold, cfg.erf_channels_per_stage,
            cfg.all_channel_erf, compute_channel_erf="channel" in cfg.analysis and "erf" in cfg.analysis,
            layouts=adapter.layouts)
        release_cuda()

    if "interventions" in cfg.analysis:
        indices = [item.dataset_index for item in selected[:cfg.intervention_samples]]
        if not indices:
            indices = list(range(min(cfg.intervention_samples, len(dataset))))
        for stage in inspection.stages:
            rows, sample_details = spatial_deletion_curve(
                model, dataset, indices, stage, cfg.keep_ratios,
                "mean" if cfg.replacement == "mean" else "zero",
                device, cfg.batch_size, cfg.workers, adapter.layouts.get(stage))
            spatial_deletion_rows.extend(rows)
            spatial_deletion_sample_rows.extend(sample_details)
            if stage in mean_channel_matrices:
                rows, clusters, sample_details = channel_deletion_curve(
                    model, dataset, indices, stage, mean_channel_matrices[stage],
                    cfg.channel_correlation_threshold, cfg.channel_keep_ratios,
                    "cluster_mean" if cfg.replacement == "cluster_mean" else "zero",
                    device, cfg.batch_size, cfg.workers, adapter.layouts.get(stage))
                channel_deletion_rows.extend(rows)
                channel_deletion_sample_rows.extend(sample_details)
                write_json(output_dir / "interventions" / f"{safe_name(stage)}_channel_clusters.json", clusters)
        write_csv(output_dir / "tables" / "spatial_deletion_curve.csv", spatial_deletion_rows)
        write_csv(output_dir / "tables" / "spatial_deletion_samples.csv", spatial_deletion_sample_rows)
        write_csv(output_dir / "tables" / "channel_deletion_curve.csv", channel_deletion_rows)
        write_csv(output_dir / "tables" / "channel_deletion_samples.csv", channel_deletion_sample_rows)
        plot_sample_intervention_curves(
            spatial_deletion_sample_rows, "keep_ratio", "spatial", output_dir)
        plot_sample_intervention_curves(
            channel_deletion_sample_rows, "effective_mean_retained_ratio", "channel", output_dir)
        release_cuda()

    figure_a(spatial_rows, spatial_deletion_rows, output_dir)
    figure_b(scale_result, erf_rows, output_dir)
    figure_c(
        scale_result, channel_rows,
        output_dir / "tables" / "scale_group_channel_similarity.csv",
        channel_deletion_rows, output_dir, mean_channel_matrices, channel_erf_rows)
    if any(name in cfg.analysis for name in ("gradcam", "spatial", "scale", "erf")):
        failure_panels(selected, spatial_rows, scale_result.rows if scale_result else [], output_dir)
    write_json(output_dir / "manifest" / "run_status.json", {
        "status": "complete", "analyses": cfg.analysis,
        "prediction_count": len(predictions), "selected_sample_count": len(selected),
        "output_dir": str(output_dir),
    })
    print(f"Diagnostics complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
