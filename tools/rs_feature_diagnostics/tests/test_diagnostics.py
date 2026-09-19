from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from PIL import Image

import timm

from tools.rs_feature_diagnostics.adapters import GenericTimmAdapter
from tools.rs_feature_diagnostics.attention_rollout import (
    AttentionCapture, build_attention_rollout, model_patch_grid, model_prefix_tokens,
)
from tools.rs_feature_diagnostics.checkpoint import load_checkpoint
from tools.rs_feature_diagnostics.classification_analysis import analyze_classification
from tools.rs_feature_diagnostics.config import parse_config
from tools.rs_feature_diagnostics.dataset import load_segmentation_mask
from tools.rs_feature_diagnostics.erf import class_sensitive_gradient, stage_erf
from tools.rs_feature_diagnostics.gradcam import GradCAM
from tools.rs_feature_diagnostics.hook_manager import InterventionHook
from tools.rs_feature_diagnostics.intervention import channel_mask_function, spatial_mask_function
from tools.rs_feature_diagnostics.metrics import effective_rank, token_similarity
from tools.rs_feature_diagnostics.model_inspector import ModelInspector
from tools.rs_feature_diagnostics.sample_selector import Prediction
from tools.rs_feature_diagnostics.tensor_adapter import FeatureTensorAdapter


class TinyCNN(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3, padding=1)
        self.block = nn.Sequential(nn.ReLU(), nn.Conv2d(6, 8, 3, stride=2, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, classes)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        return self.head(self.pool(x).flatten(1))


class CheckpointTests(unittest.TestCase):
    def test_prefix_and_wrapper_loading(self):
        source = TinyCNN(3)
        state = {f"module.model.{key}": value.clone() for key, value in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"model_state_dict": state}, path)
            target = TinyCNN(3)
            report = load_checkpoint(target, path, 3)
            self.assertEqual(report.selected_key, "model_state_dict")
            for left, right in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_classifier_mismatch_is_explicit(self):
        source = TinyCNN(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"state_dict": source.state_dict()}, path)
            with self.assertRaisesRegex(RuntimeError, "classifier mismatch"):
                load_checkpoint(TinyCNN(3), path, 3)
            report = load_checkpoint(TinyCNN(3), path, 3, allow_head_mismatch=True)
            self.assertTrue(report.skipped_head_keys)


class TensorAndMetricTests(unittest.TestCase):
    def test_segmentation_mask_relative_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "val"
            image_path = split / "class_a" / "sample.tif"
            mask_path = root / "masks" / "class_a" / "sample.png"
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(image_path)
            Image.new("L", (8, 8), 255).save(mask_path)
            mask, source = load_segmentation_mask(root / "masks", image_path, split, (4, 4))
            self.assertEqual(mask.shape, (4, 4))
            self.assertTrue(mask.all())
            self.assertEqual(source, str(mask_path.resolve()))

    def test_layout_conversion(self):
        adapter = FeatureTensorAdapter()
        nchw = torch.randn(2, 8, 7, 7)
        nhwc = torch.randn(2, 7, 7, 8)
        tokens = torch.randn(2, 50, 8)
        self.assertEqual(adapter.adapt(nchw).tensor.shape, (2, 8, 7, 7))
        self.assertEqual(adapter.adapt(nhwc).tensor.shape, (2, 8, 7, 7))
        adapted = adapter.adapt(tokens)
        self.assertEqual(adapted.tensor.shape, (2, 8, 7, 7))
        self.assertTrue(adapted.removed_class_token)

    def test_effective_rank_and_similarity(self):
        feature = torch.randn(1, 12, 8, 8)
        rank = effective_rank(feature)
        self.assertGreaterEqual(rank["energy_rank_95"], rank["energy_rank_90"])
        matrix, stats, indices = token_similarity(feature, max_tokens=20, seed=7)
        self.assertEqual(matrix.shape, (20, 20))
        self.assertEqual(stats["sampled_tokens"], 20)
        self.assertEqual(indices.numel(), 20)


class ClassificationAnalysisTests(unittest.TestCase):
    @staticmethod
    def _predictions():
        return [
            Prediction(0, "a0.jpg", "a", 0, "a", 0, .90, .90, 1.0, True),
            Prediction(1, "a1.jpg", "a", 0, "b", 1, .80, .10, .8, False),
            Prediction(2, "b0.jpg", "b", 1, "b", 1, .70, .70, .5, True),
            Prediction(3, "b1.jpg", "b", 1, "c", 2, .60, .20, .4, False),
            Prediction(4, "c0.jpg", "c", 2, "c", 2, .95, .95, 1.2, True),
        ]

    def test_dataset_level_classification_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = analyze_classification(self._predictions(), ["a", "b", "c"], output)
            self.assertEqual(result.confusion_counts.tolist(), [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
            self.assertAlmostEqual(result.confusion_row_normalized[0, 0], .5)
            self.assertAlmostEqual(result.per_class_rows[0]["precision"], 1.0)
            self.assertAlmostEqual(result.per_class_rows[0]["recall"], .5)
            self.assertEqual(
                [(row["true_class"], row["pred_class"]) for row in result.top_confusions],
                [("a", "b"), ("b", "c")])
            self.assertAlmostEqual(result.summary["top1_accuracy"], .6)
            self.assertAlmostEqual(result.summary["ece"], .37)
            self.assertEqual(result.risk_coverage_rows[0]["retained_samples"], 1)
            self.assertAlmostEqual(result.risk_coverage_rows[-1]["risk"], .4)
            for path in (
                output / "classification" / "confusion_counts.png",
                output / "classification" / "confusion_row_normalized.pdf",
                output / "classification" / "reliability_diagram.png",
                output / "classification" / "risk_coverage.pdf",
                output / "tables" / "classification_per_class.csv",
                output / "tables" / "classification_summary.json",
            ):
                self.assertTrue(path.is_file(), path)

    def test_analysis_choice_includes_classification(self):
        cfg = parse_config(["--model", "tiny", "--data-dir", ".", "--analysis", "all"])
        self.assertIn("classification", cfg.analysis)


class AttentionRolloutTests(unittest.TestCase):
    def test_timm_vit_rollout_and_fused_flag_restoration(self):
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=3).eval()
        original_fused_flags = [module.fused_attn for module in model.modules() if hasattr(module, "fused_attn")]
        with AttentionCapture(model) as capture:
            captured = capture.capture(torch.randn(1, 3, 224, 224))
            result = build_attention_rollout(
                captured, int(getattr(model, "num_prefix_tokens", 1)), (224, 224))
            self.assertEqual(len(captured), 12)
            self.assertEqual(result.rollout.shape, (14, 14))
            self.assertEqual(result.final_head_maps.shape, (3, 14, 14))
            self.assertEqual(len(result.layer_names), 12)
        restored_fused_flags = [module.fused_attn for module in model.modules() if hasattr(module, "fused_attn")]
        self.assertEqual(restored_fused_flags, original_fused_flags)

    def test_analysis_choice_includes_attention(self):
        cfg = parse_config(["--model", "tiny", "--data-dir", ".", "--analysis", "all"])
        self.assertIn("attention", cfg.analysis)

    def test_window_attention_is_not_treated_as_class_token_rollout(self):
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=3).eval()
        self.assertEqual(model_prefix_tokens(model), 0)
        with AttentionCapture(model) as capture:
            captured = capture.capture(torch.randn(1, 3, 224, 224))
        with self.assertRaisesRegex(ValueError, "class/prefix token"):
            build_attention_rollout(captured, model_prefix_tokens(model), (224, 224), model_patch_grid(model))

    def test_dynamic_vit_uses_the_actual_patch_grid(self):
        model = timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, num_classes=3, dynamic_img_size=True).eval()
        image = torch.randn(1, 3, 256, 256)
        with AttentionCapture(model) as capture:
            captured = capture.capture(image)
        result = build_attention_rollout(
            captured, model_prefix_tokens(model), (256, 256), model_patch_grid(model, (256, 256)))
        self.assertEqual(result.grid_size, (16, 16))
        self.assertEqual(result.rollout.shape, (16, 16))


class InterventionAndGradientTests(unittest.TestCase):
    def test_spatial_and_channel_hooks(self):
        model = TinyCNN().eval()
        image = torch.randn(2, 3, 32, 32)
        baseline = model(image)
        with InterventionHook(model, "block.1", spatial_mask_function(.5, "zero")):
            spatial_output = model(image)
        clusters = [[0, 1, 2], [3, 4], [5, 6, 7]]
        with InterventionHook(model, "block.1", channel_mask_function(clusters, .5, "zero")):
            channel_output = model(image)
        self.assertEqual(spatial_output.shape, baseline.shape)
        self.assertEqual(channel_output.shape, baseline.shape)
        self.assertFalse(torch.equal(spatial_output, baseline))

    def test_gradcam_and_erf_shapes(self):
        model = TinyCNN().eval()
        image = torch.randn(1, 3, 32, 32)
        with GradCAM(model, "block.1") as cam:
            result = cam.compute(image, 1)
        self.assertEqual(result.cam.shape, (32, 32))
        gradient, radii = stage_erf(model, image, "block.1")
        self.assertEqual(gradient.shape, (32, 32))
        self.assertIn("r80_pixels", radii)
        class_gradient, _ = class_sensitive_gradient(model, image, 1)
        self.assertEqual(class_gradient.shape, (32, 32))


class LocalTimmDryRunTests(unittest.TestCase):
    def test_cnn_and_transformer_inspection(self):
        for model_name in ("mobilenetv2_100", "vit_tiny_patch16_224"):
            model = timm.create_model(model_name, pretrained=False, num_classes=3).eval()
            adapter = GenericTimmAdapter.from_model(model)
            result = ModelInspector(model, (3, 224, 224), torch.device("cpu")).inspect(adapter)
            self.assertGreaterEqual(len(result.stages), 2, model_name)
            self.assertFalse(result.ambiguous, model_name)


if __name__ == "__main__":
    unittest.main()
