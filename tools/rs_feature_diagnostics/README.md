# Remote-Sensing Feature Diagnostics

A model- and ImageFolder-dataset-independent diagnostic toolkit for spatial redundancy,
scale/receptive-field mismatch, channel scale specialization, channel redundancy, and
classification failure analysis.

The documented script entrypoint intentionally uses the current repository source:

```bash
python tools/rs_feature_diagnostics/main.py --help
```

## Preflight first

```bash
python tools/rs_feature_diagnostics/main.py \
  --model repvit_m0_9 \
  --checkpoint /path/to/model_best.pth.tar \
  --data-dir /path/to/UCM-82 \
  --val-split val \
  --device cuda:0 \
  --analysis structure
```

Inspect `manifest/model_structure.txt`. If stage discovery is not semantically correct,
copy `adapter_example.yaml`, edit its module paths, and pass `--adapter-config`.

## Full analysis

```bash
python tools/rs_feature_diagnostics/main.py \
  --model repvit_m0_9 \
  --checkpoint /path/to/model_best.pth.tar \
  --data-dir /path/to/UCM-82 \
  --val-split val \
  --sample-mode misclassified \
  --input-size 224 \
  --device cuda:0 \
  --output-dir ./visualization_results \
  --analysis all
```

Expensive analyses have bounded defaults. Override them with `--max-samples`,
`--max-tokens`, `--similarity-samples`, `--intervention-samples`, and
`--erf-channels-per-stage`.

## Dataset-level classification diagnostics

Use `--analysis classification` (included in `--analysis all`) to generate a full-validation
confusion matrix (counts and true-class-row normalized), per-class precision/recall/F1,
ranked error pairs, a reliability diagram with ECE and NLL, and a risk-coverage curve.
It reuses the ordinary inference records (also written to `predictions/predictions.csv`), so it
adds no model forward pass or model-specific hook. Confusion-matrix rows are true classes and
columns are predicted classes; CSV/JSON tables are written under `tables/` and PNG/PDF figures
under `classification/`.

## Transformer attention rollout

Use `--analysis attention` (also included in `--analysis all`) for class-token attention
rollout and final-layer per-head maps on supported global-token Transformer models. The
implementation temporarily turns off timm fused attention while it captures the existing
`attn_drop` probability tensors, then restores every module setting. It intentionally skips
windowed/cross-attention or token layouts that cannot be mapped faithfully to the input grid;
the reason is recorded in `tables/attention_rollout_manifest.csv` rather than fabricating an
image-space heatmap. CNN runs therefore remain valid and simply produce explicit skipped rows.

## Content-scale intervention analysis

The scale analysis separates three factors instead of treating zoom as a pure
object-scale test:

- content-scale: resize plus center padding for zoom-out, or resize plus crop for zoom-in;
- resolution-only: downsample to 112/168 (for a 224 input) and resize back without changing
  occupancy, field of view, or visible context;
- context-only: retain the native-pixel center crop and mean-pad it back without magnification.

Zoom-in also runs center/top-left/top-right/bottom-left/bottom-right crops. Reports include
full-validation corrected/degraded transitions, GT-vs-strongest-wrong logit margins,
class-wise effects, corrected-sample Jaccard overlap, per-image predictions, and compressed
logits. The headline `scale_sensitivity` curve is explicitly the correction rate among
originally misclassified samples, not validation Top-1. Full-validation controls are the
default; use `--scale-eval-scope selected` for a bounded exploratory run, or
`--scale-batch-size` to reduce inference memory.

Every run writes `manifest/`, complete validation predictions, sample selections,
CSV aggregate and sample-level intervention curves, raw `.npy/.npz` arrays, 300-dpi
PNG/PDF plots, failure panels, and Figures A/B/C below
`<output-dir>/<dataset>/<model>/<run-id>/`.

Artifact stage labels follow network semantics: `stem` is named separately and modules
`stages.0` through `stages.3` are named `stage1` through `stage4`. Plot titles include both
the semantic label and the concrete module path so the hook source remains auditable.

Spatial entropy is defined as Shannon entropy after
`softmax(abs(feature) / temperature)` across channels. Effective rank uses centered
token features and singular-value entropy; energy ranks report the dimensions needed
for 90% and 95% squared-singular-value energy. The homogeneous-region result is only
an RGB local-variance/Sobel texture proxy, never a foreground/background annotation.
When real binary masks are available, pass `--segmentation-mask-dir`: masks should
mirror validation paths (`<mask-dir>/<class>/<image-name>.png`, with the original
suffix also accepted). Foreground/background similarity is then reported separately;
the texture proxy is not relabeled as ground truth.

## Design boundaries

- ImageFolder class count and mapping are inferred before model construction.
- Checkpoint classifier mismatches fail explicitly unless `--allow-head-mismatch` is used.
- Transformer tokens are reshaped only for `N=H*W` or `N=H*W+1`; otherwise spatial
  analyses are skipped with a warning.
- Deploy/reparameterization is opt-in because fusion changes the semantic hook tree.
- Ordinary inference may use AMP; Grad-CAM, ERF, SVD, and correlations use FP32.
- Homogeneous-region masks are image-texture proxies, not ground-truth background masks.
- Auto stage discovery cannot infer semantic residual outputs for every non-standard model;
  use an adapter override when preflight is ambiguous or semantically wrong.
- Dataset-wide interventions and all-channel ERF are intentionally expensive.
- `--channel-keep-ratios 0` means retain one representative per similarity cluster;
  the CSV also records the resulting effective retained ratio.
