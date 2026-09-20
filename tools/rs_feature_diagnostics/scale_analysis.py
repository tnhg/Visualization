from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .dataset import (
    content_scaled_tensor,
    context_only_tensor,
    open_rgb,
    resolution_only_tensor,
)
from .hook_manager import CaptureHookManager
from .metrics import effective_rank, scale_preferences, spatial_maps
from .sample_selector import Prediction, extract_logits
from .tensor_adapter import FeatureTensorAdapter
from .utils import autocast_context, safe_name, semantic_stage_label, write_csv
from .visualization import grouped_bar_plot, line_plot, matrix_heatmap, multi_panel


@dataclass
class ScaleAnalysisResult:
    rows: list[dict]
    channel_responses: Dict[str, np.ndarray]
    preference_rows: list[dict]


@dataclass(frozen=True)
class _Intervention:
    name: str
    family: str
    scale: float
    crop_position: str
    transform: Callable[[Image.Image], tuple[torch.Tensor, str]]


@dataclass
class _Evaluation:
    intervention: _Intervention
    logits: torch.Tensor
    predicted: torch.Tensor
    confidence: torch.Tensor
    true_probability: torch.Tensor
    gt_margin: torch.Tensor
    correct: torch.Tensor
    method: str
    input_fingerprints: list[str]
    sample_ids: list[str]
    targets: list[int]


def _scale_name(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def _transition(before_correct: bool, after_correct: bool) -> str:
    if before_correct and after_correct:
        return "stable_correct"
    if before_correct:
        return "degraded"
    if after_correct:
        return "corrected"
    return "unchanged_wrong"


def _evaluate_intervention(
    model,
    predictions: Sequence[Prediction],
    intervention: _Intervention,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> _Evaluation:
    all_logits: list[torch.Tensor] = []
    methods: set[str] = set()
    input_fingerprints: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(predictions), batch_size):
            chunk = predictions[start:start + batch_size]
            tensors = []
            for prediction in chunk:
                tensor, method = intervention.transform(open_rgb(prediction.image_path))
                tensors.append(tensor)
                input_fingerprints.append(hashlib.sha1(tensor.float().numpy().tobytes()).hexdigest())
                methods.add(method)
            batch = torch.stack(tensors).to(device, non_blocking=True)
            with autocast_context(device, amp):
                logits = extract_logits(model(batch))
            all_logits.append(logits.float().cpu())
    logits = torch.cat(all_logits) if all_logits else torch.empty((0, 0))
    targets = torch.tensor([item.true_index for item in predictions], dtype=torch.long)
    probabilities = logits.softmax(dim=1)
    predicted = logits.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    true_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    wrong_logits = logits.clone()
    wrong_logits.scatter_(1, targets[:, None], -torch.inf)
    strongest_wrong = wrong_logits.max(dim=1).values
    gt_margin = logits.gather(1, targets[:, None]).squeeze(1) - strongest_wrong
    return _Evaluation(
        intervention=intervention,
        logits=logits,
        predicted=predicted,
        confidence=confidence,
        true_probability=true_probability,
        gt_margin=gt_margin,
        correct=predicted.eq(targets),
        method=";".join(sorted(methods)),
        input_fingerprints=input_fingerprints,
        sample_ids=[item.sample_id for item in predictions],
        targets=[item.true_index for item in predictions],
    )


def _summarize_evaluation(
    evaluation: _Evaluation,
    baseline: _Evaluation,
    predictions: Sequence[Prediction],
) -> dict:
    before = baseline.correct
    after = evaluation.correct
    originally_wrong = ~before
    corrected = int((originally_wrong & after).sum())
    degraded = int((before & ~after).sum())
    wrong_count = int(originally_wrong.sum())
    return {
        "condition": evaluation.intervention.name,
        "family": evaluation.intervention.family,
        "scale": evaluation.intervention.scale,
        "crop_position": evaluation.intervention.crop_position,
        "method": evaluation.method,
        "samples": len(predictions),
        "correct": int(after.sum()),
        "top1": 100. * float(after.float().mean()),
        "baseline_correct": int(before.sum()),
        "corrected": corrected,
        "degraded": degraded,
        "stable_correct": int((before & after).sum()),
        "unchanged_wrong": int((originally_wrong & ~after).sum()),
        "prediction_disagreement": int(evaluation.predicted.ne(baseline.predicted).sum()),
        "net_gain": corrected - degraded,
        "originally_wrong": wrong_count,
        "misclassification_correction_rate": 100. * corrected / max(wrong_count, 1),
        "mean_gt_margin": float(evaluation.gt_margin.mean()),
        "mean_delta_gt_margin": float((evaluation.gt_margin - baseline.gt_margin).mean()),
        "mean_gt_probability": float(evaluation.true_probability.mean()),
        "mean_confidence": float(evaluation.confidence.mean()),
    }


def _write_prediction_rows(
    evaluations: Sequence[_Evaluation],
    baseline: _Evaluation,
    predictions: Sequence[Prediction],
    classes: Sequence[str],
    output_dir: Path,
) -> None:
    rows: list[dict] = []
    for evaluation in evaluations:
        for row_index, prediction in enumerate(predictions):
            before_correct = bool(baseline.correct[row_index])
            after_correct = bool(evaluation.correct[row_index])
            pred_index = int(evaluation.predicted[row_index])
            rows.append({
                "condition": evaluation.intervention.name,
                "family": evaluation.intervention.family,
                "scale": evaluation.intervention.scale,
                "crop_position": evaluation.intervention.crop_position,
                "dataset_index": prediction.dataset_index,
                "sample_id": prediction.sample_id,
                "relative_path": prediction.relative_path,
                "image_path": prediction.image_path,
                "true_class": prediction.true_class,
                "true_index": prediction.true_index,
                "baseline_pred_class": classes[int(baseline.predicted[row_index])],
                "baseline_pred_index": int(baseline.predicted[row_index]),
                "baseline_correct": before_correct,
                "pred_class": classes[pred_index],
                "pred_index": pred_index,
                "correct": after_correct,
                "transition": _transition(before_correct, after_correct),
                "confidence": float(evaluation.confidence[row_index]),
                "gt_probability": float(evaluation.true_probability[row_index]),
                "gt_margin": float(evaluation.gt_margin[row_index]),
                "delta_gt_margin": float(evaluation.gt_margin[row_index] - baseline.gt_margin[row_index]),
            })
    write_csv(output_dir / "tables" / "scale_intervention_predictions.csv", rows)


def _plot_factor_decomposition(summary_rows: Sequence[dict], output_dir: Path) -> None:
    by_condition = {row["condition"]: row for row in summary_rows}
    labels: list[str] = []
    values: list[float] = []
    for scale in (.5, .75):
        main = f"content_scale_{_scale_name(scale)}"
        control = f"resolution_only_{_scale_name(scale)}"
        if main in by_condition and control in by_condition:
            labels.extend((f"scale {scale:g}", f"resolution {scale:g}"))
            values.extend((by_condition[main]["misclassification_correction_rate"],
                           by_condition[control]["misclassification_correction_rate"]))
    for scale in (1.5, 2.0):
        main = f"content_scale_{_scale_name(scale)}"
        control = f"context_only_{_scale_name(scale)}"
        if main in by_condition and control in by_condition:
            labels.extend((f"zoom {scale:g}", f"context {scale:g}"))
            values.extend((by_condition[main]["misclassification_correction_rate"],
                           by_condition[control]["misclassification_correction_rate"]))
    if labels:
        grouped_bar_plot(
            labels, {"Correction rate": values}, output_dir / "scale" / "factor_decomposition",
            "Scale / resolution / context factor decomposition",
            "Originally misclassified samples corrected (%)")


def _correction_overlap(
    main_evaluations: Sequence[_Evaluation],
    baseline: _Evaluation,
    output_dir: Path,
) -> None:
    labels = [f"{item.intervention.scale:g}" for item in main_evaluations]
    corrected_sets = [set(torch.where((~baseline.correct) & item.correct)[0].tolist())
                      for item in main_evaluations]
    rows: list[dict] = []
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)
    for first_index, (first_label, first_set) in enumerate(zip(labels, corrected_sets)):
        for second_index, (second_label, second_set) in enumerate(zip(labels, corrected_sets)):
            intersection = len(first_set & second_set)
            union = len(first_set | second_set)
            jaccard = intersection / union if union else np.nan
            matrix[first_index, second_index] = jaccard
            rows.append({
                "scale_a": first_label,
                "scale_b": second_label,
                "corrected_a": len(first_set),
                "corrected_b": len(second_set),
                "intersection": intersection,
                "union": union,
                "jaccard": jaccard,
                "fraction_a_contained_in_b": intersection / len(first_set) if first_set else np.nan,
                "fraction_b_contained_in_a": intersection / len(second_set) if second_set else np.nan,
            })
    write_csv(output_dir / "tables" / "scale_correction_overlap.csv", rows)
    matrix_heatmap(
        matrix, labels, labels, output_dir / "scale" / "correction_set_jaccard",
        "Corrected-sample set overlap", "Jaccard", annotate=True)


def _classwise_analysis(
    main_evaluations: Sequence[_Evaluation],
    baseline: _Evaluation,
    predictions: Sequence[Prediction],
    classes: Sequence[str],
    output_dir: Path,
) -> None:
    targets = torch.tensor([item.true_index for item in predictions], dtype=torch.long)
    rows: list[dict] = []
    correction_matrix = np.full((len(classes), len(main_evaluations)), np.nan, dtype=np.float32)
    for scale_index, evaluation in enumerate(main_evaluations):
        for class_index, class_name in enumerate(classes):
            mask = targets.eq(class_index)
            before = baseline.correct[mask]
            after = evaluation.correct[mask]
            originally_wrong = ~before
            corrected = int((originally_wrong & after).sum())
            degraded = int((before & ~after).sum())
            wrong_count = int(originally_wrong.sum())
            rate = 100. * corrected / wrong_count if wrong_count else np.nan
            correction_matrix[class_index, scale_index] = rate
            rows.append({
                "scale": evaluation.intervention.scale,
                "class_index": class_index,
                "class_name": class_name,
                "samples": int(mask.sum()),
                "baseline_correct": int(before.sum()),
                "correct": int(after.sum()),
                "corrected": corrected,
                "degraded": degraded,
                "net_gain": corrected - degraded,
                "originally_wrong": wrong_count,
                "misclassification_correction_rate": rate,
                "mean_delta_gt_margin": float(
                    (evaluation.gt_margin[mask] - baseline.gt_margin[mask]).mean()) if mask.any() else np.nan,
            })
    write_csv(output_dir / "tables" / "scale_classwise.csv", rows)
    matrix_heatmap(
        correction_matrix,
        [f"{item.intervention.scale:g}" for item in main_evaluations], classes,
        output_dir / "scale" / "classwise_correction_rate",
        "Class-wise correction of original errors", "Correction rate (%)")


def _crop_bias_analysis(
    evaluations_by_name: Dict[str, _Evaluation],
    baseline: _Evaluation,
    predictions: Sequence[Prediction],
    zoom_scales: Sequence[float],
    output_dir: Path,
) -> None:
    positions = ("center", "top_left", "top_right", "bottom_left", "bottom_right")
    rows: list[dict] = []
    plot_series: dict[str, list[float]] = {}
    aggregate_rows: list[dict] = []
    for scale in zoom_scales:
        summaries = []
        for position in positions:
            name = (f"content_scale_{_scale_name(scale)}" if position == "center" else
                    f"content_scale_{_scale_name(scale)}_{position}")
            if name not in evaluations_by_name:
                continue
            summary = _summarize_evaluation(evaluations_by_name[name], baseline, predictions)
            summaries.append(summary)
            rows.append(summary)
        if summaries:
            correction = [row["misclassification_correction_rate"] for row in summaries]
            top1 = [row["top1"] for row in summaries]
            plot_series[f"scale={scale:g}"] = correction
            aggregate_rows.append({
                "scale": scale,
                "crop_count": len(summaries),
                "mean_top1": float(np.mean(top1)),
                "std_top1": float(np.std(top1)),
                "mean_correction_rate": float(np.mean(correction)),
                "std_correction_rate": float(np.std(correction)),
                "center_top1": summaries[0]["top1"],
                "center_correction_rate": summaries[0]["misclassification_correction_rate"],
            })
    write_csv(output_dir / "tables" / "scale_crop_bias.csv", rows)
    write_csv(output_dir / "tables" / "scale_crop_bias_aggregate.csv", aggregate_rows)
    if plot_series:
        grouped_bar_plot(
            positions, plot_series, output_dir / "scale" / "center_bias_control",
            "Position-controlled zoom intervention",
            "Originally misclassified samples corrected (%)")


def _run_intervention_analysis(
    model,
    predictions: Sequence[Prediction],
    classes: Sequence[str],
    device: torch.device,
    output_dir: Path,
    input_size: tuple[int, int, int],
    mean: Sequence[float],
    std: Sequence[float],
    crop_pct: float,
    interpolation: str,
    scales: Sequence[float],
    batch_size: int,
    amp: bool,
    crop_mode: str = "center",
) -> None:
    if not any(abs(scale - 1.) < 1e-8 for scale in scales):
        raise ValueError("content-scale intervention requires scale=1 as its baseline")

    def content_transform(scale: float, position: str = "center"):
        return lambda image: content_scaled_tensor(
            image, scale, input_size, mean, std, crop_pct, interpolation, position, crop_mode)

    def resolution_transform(scale: float):
        return lambda image: resolution_only_tensor(
            image, scale, input_size, mean, std, crop_pct, interpolation, crop_mode)

    def context_transform(scale: float):
        return lambda image: context_only_tensor(
            image, scale, input_size, mean, std, crop_pct, interpolation, crop_mode)

    main_interventions = [
        _Intervention(
            f"content_scale_{_scale_name(scale)}", "content_scale", scale, "center",
            content_transform(scale))
        for scale in scales
    ]
    interventions = list(main_interventions)
    interventions.extend(
        _Intervention(
            f"resolution_only_{_scale_name(scale)}", "resolution_only", scale, "center",
            resolution_transform(scale))
        for scale in scales if scale < 1
    )
    zoom_scales = [scale for scale in scales if scale > 1]
    interventions.extend(
        _Intervention(
            f"context_only_{_scale_name(scale)}", "context_only", scale, "center",
            context_transform(scale))
        for scale in zoom_scales
    )
    for scale in zoom_scales:
        for position in ("top_left", "top_right", "bottom_left", "bottom_right"):
            interventions.append(_Intervention(
                f"content_scale_{_scale_name(scale)}_{position}", "crop_bias", scale, position,
                content_transform(scale, position)))

    evaluations = [
        _evaluate_intervention(model, predictions, intervention, device, batch_size, amp)
        for intervention in interventions
    ]
    evaluations_by_name = {item.intervention.name: item for item in evaluations}
    baseline = evaluations_by_name["content_scale_1"]
    main_evaluations = [evaluations_by_name[item.name] for item in main_interventions]

    summary_rows = [_summarize_evaluation(item, baseline, predictions) for item in evaluations]
    write_csv(output_dir / "tables" / "scale_intervention_summary.csv", summary_rows)
    _write_prediction_rows(evaluations, baseline, predictions, classes, output_dir)
    np.savez_compressed(
        output_dir / "scale" / "scale_intervention_logits.npz",
        dataset_indices=np.asarray([item.dataset_index for item in predictions], dtype=np.int64),
        **{safe_name(item.intervention.name): item.logits.numpy() for item in evaluations},
    )

    original_predictions = torch.tensor([item.pred_index for item in predictions])
    original_mismatch = int(baseline.predicted.ne(original_predictions).sum())
    expected_sample_ids = [item.sample_id for item in predictions]
    sample_id_mismatch = sum(
        current != expected
        for current, expected in zip(baseline.sample_ids, expected_sample_ids)
    )
    expected_targets = [item.true_index for item in predictions]
    target_mismatch = sum(
        int(current) != int(expected)
        for current, expected in zip(baseline.targets, expected_targets)
    )
    expected_logits = [item.logits for item in predictions]
    logit_deltas = [
        float(torch.tensor(current).sub(torch.tensor(expected)).abs().max())
        for current, expected in zip(baseline.logits.tolist(), expected_logits)
        if expected
    ]
    input_mismatch = sum(
        bool(expected and expected != current)
        for current, expected in zip(baseline.input_fingerprints, [item.input_fingerprint for item in predictions])
    )
    identity_pass = (
        original_mismatch == 0 and sample_id_mismatch == 0 and target_mismatch == 0
        and input_mismatch == 0 and all(delta <= 1e-5 for delta in logit_deltas)
    )
    write_csv(output_dir / "tables" / "scale_identity_audit.csv", [{
        "samples": len(predictions),
        "prediction_mismatch_vs_initial_inference": original_mismatch,
        "sample_id_mismatch": sample_id_mismatch,
        "target_mismatch": target_mismatch,
        "input_tensor_mismatch": input_mismatch,
        "max_abs_logit_delta": max(logit_deltas) if logit_deltas else None,
        "numeric_mode": "amp" if amp else "float32",
        "pass": identity_pass,
        "status": "complete" if identity_pass else "blocked",
        "reason": "" if identity_pass else "scale=1 is not identical to the population inference baseline",
    }])
    if not identity_pass:
        raise RuntimeError("scale identity audit failed; refusing to compare intervention conditions")

    originally_wrong = ~baseline.correct
    correction_curve = [
        100. * float(item.correct[originally_wrong].float().mean()) if originally_wrong.any() else np.nan
        for item in main_evaluations
    ]
    sensitivity_rows = [{
        "scale": item.intervention.scale,
        "misclassification_correction_rate": correction_curve[index],
        "top1": correction_curve[index],
        "originally_misclassified_samples": int(originally_wrong.sum()),
        "corrected": int((originally_wrong & item.correct).sum()),
        "full_top1": 100. * float(item.correct.float().mean()),
        "samples": len(predictions),
    } for index, item in enumerate(main_evaluations)]
    write_csv(output_dir / "tables" / "scale_sensitivity.csv", sensitivity_rows)
    line_plot(
        list(scales), {"Correction rate": correction_curve},
        output_dir / "scale" / "scale_sensitivity",
        "Content-scale intervention on originally misclassified samples",
        "content scale", "Misclassification correction rate (%)")

    mean_delta_all = [float((item.gt_margin - baseline.gt_margin).mean()) for item in main_evaluations]
    mean_delta_wrong = [
        float((item.gt_margin[originally_wrong] - baseline.gt_margin[originally_wrong]).mean())
        if originally_wrong.any() else np.nan for item in main_evaluations
    ]
    line_plot(
        list(scales), {"All validation": mean_delta_all, "Originally misclassified": mean_delta_wrong},
        output_dir / "scale" / "gt_margin_change",
        "GT-vs-strongest-wrong logit margin change", "content scale", "Mean delta GT margin")

    _plot_factor_decomposition(summary_rows, output_dir)
    _correction_overlap(main_evaluations, baseline, output_dir)
    _classwise_analysis(main_evaluations, baseline, predictions, classes, output_dir)
    _crop_bias_analysis(evaluations_by_name, baseline, predictions, zoom_scales, output_dir)


def analyze_scale(
    model,
    selected_predictions: Sequence[Prediction],
    population_predictions: Sequence[Prediction],
    classes: Sequence[str],
    stage_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    input_size: tuple[int, int, int],
    mean: Sequence[float],
    std: Sequence[float],
    crop_pct: float,
    interpolation: str,
    scales: Sequence[float],
    batch_size: int,
    amp: bool,
    layouts: Optional[Dict[str, str]] = None,
    crop_mode: str = "center",
) -> ScaleAnalysisResult:
    layouts = layouts or {}
    adapter = FeatureTensorAdapter()
    rows: list[dict] = []
    response_sum: Dict[str, list[Optional[torch.Tensor]]] = {
        stage: [None for _ in scales] for stage in stage_names}
    response_count: Dict[str, list[int]] = {stage: [0 for _ in scales] for stage in stage_names}
    first_sample_maps: Dict[str, list[np.ndarray]] = {stage: [] for stage in stage_names}
    model.eval()

    for sample_no, prediction in enumerate(selected_predictions):
        image = open_rgb(prediction.image_path)
        for scale_index, scale in enumerate(scales):
            tensor, method = content_scaled_tensor(
                image, scale, input_size, mean, std, crop_pct, interpolation, "center", crop_mode)
            batch = tensor.unsqueeze(0).to(device)
            with CaptureHookManager(model, stage_names, detach=True) as hooks:
                with torch.inference_mode():
                    extract_logits(model(batch))
            for stage_index, stage in enumerate(stage_names):
                stage_label = semantic_stage_label(stage, stage_index)
                try:
                    adapted = adapter.adapt(hooks.outputs[stage], layout_hint=layouts.get(stage))
                except (ValueError, TypeError):
                    continue
                if not adapted.spatial:
                    continue
                feature = adapted.tensor.float()
                maps = spatial_maps(feature)
                rank = effective_rank(feature)
                channel_rms = feature.square().mean(dim=(0, 2, 3)).sqrt().cpu()
                if response_sum[stage][scale_index] is None:
                    response_sum[stage][scale_index] = channel_rms
                else:
                    response_sum[stage][scale_index] += channel_rms
                response_count[stage][scale_index] += 1
                concentration = float(maps["energy"].amax() / (maps["energy"].sum() + 1e-12))
                rows.append({
                    "sample_id": prediction.sample_id,
                    "dataset_index": prediction.dataset_index,
                    "image_path": prediction.image_path,
                    "true_class": prediction.true_class,
                    "scale": scale,
                    "scale_method": method,
                    "stage_index": stage_index,
                    "stage": stage,
                    "mean_absolute_activation": float(feature.abs().mean()),
                    "rms_activation": float(feature.square().mean().sqrt()),
                    "spatial_concentration": concentration,
                    **rank,
                })
                if sample_no == 0:
                    first_sample_maps[stage].append(maps["energy"][0].cpu().numpy())
    write_csv(output_dir / "tables" / "scale_stage_response.csv", rows)

    _run_intervention_analysis(
        model, population_predictions, classes, device, output_dir,
        input_size, mean, std, crop_pct, interpolation, scales, batch_size, amp, crop_mode)

    responses: Dict[str, np.ndarray] = {}
    preference_rows: list[dict] = []
    scale_tensor = torch.tensor(scales)
    for stage_index, stage in enumerate(stage_names):
        stage_label = semantic_stage_label(stage, stage_index)
        if any(value is None for value in response_sum[stage]):
            continue
        response = torch.stack([
            value / max(response_count[stage][index], 1)
            for index, value in enumerate(response_sum[stage]) if value is not None
        ])
        responses[stage] = response.numpy()
        preference, selectivity = scale_preferences(response, scale_tensor)
        for channel in range(response.shape[1]):
            preference_rows.append({
                "stage_index": stage_index,
                "stage": stage,
                "stage_label": stage_label,
                "channel": channel,
                "preferred_scale": float(preference[channel]),
                "scale_selectivity_index": float(selectivity[channel]),
            })
        matrix_heatmap(
            response.T.numpy(), [f"{scale:g}" for scale in scales],
            [str(channel) for channel in range(response.shape[1])],
            output_dir / "scale" / f"{stage_label}_channel_scale_response",
            f"Channel x content-scale response - {stage_label} ({stage})", "RMS activation")
        if first_sample_maps[stage]:
            multi_panel(
                first_sample_maps[stage], [f"scale={scale:g}" for scale in scales],
                output_dir / "scale" / f"{stage_label}_multiscale_energy", columns=3)
    np.savez_compressed(output_dir / "scale" / "channel_scale_responses.npz", **responses)
    write_csv(output_dir / "tables" / "channel_scale_preferences.csv", preference_rows)
    return ScaleAnalysisResult(rows, responses, preference_rows)
