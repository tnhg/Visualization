from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch import nn

from timm.utils.model import reparameterize_model

from .base import ModelAdapter, SemanticOps


class GenericTimmAdapter(ModelAdapter):
    @classmethod
    def from_model(cls, model: nn.Module, override_path: Optional[Path] = None) -> "GenericTimmAdapter":
        adapter = cls(source="generic_timm")
        modules = dict(model.named_modules())
        for stem_candidate in ("stem", "conv_stem", "patch_embed"):
            if stem_candidate in modules:
                adapter.stem = stem_candidate
                break
        feature_info = getattr(model, "feature_info", None)
        dicts: List[Dict[str, Any]] = []
        if feature_info is not None:
            if hasattr(feature_info, "get_dicts"):
                dicts = feature_info.get_dicts()
            elif isinstance(feature_info, (list, tuple)):
                dicts = list(feature_info)
        adapter.stages = [str(item["module"]) for item in dicts if item.get("module") in modules]
        if not adapter.stages and hasattr(model, "blocks"):
            blocks = getattr(model, "blocks")
            try:
                count = len(blocks)
            except TypeError:
                count = 0
            if count:
                indices = sorted(set((max(0, count // 4 - 1), max(0, count // 2 - 1), max(0, 3 * count // 4 - 1), count - 1)))
                adapter.stages = [f"blocks.{index}" for index in indices if f"blocks.{index}" in modules]
                adapter.stem = adapter.stem or ("patch_embed" if "patch_embed" in modules else None)
                adapter.source = "generic_timm_transformer_blocks"
        if not adapter.stages and hasattr(model, "layers"):
            layers = getattr(model, "layers")
            try:
                adapter.stages = [f"layers.{index}" for index in range(len(layers)) if f"layers.{index}" in modules]
            except TypeError:
                pass
        if adapter.stages:
            adapter.stem = adapter.stem or adapter.stages[0]
        adapter.semantic_ops = cls._semantic_scan(model)
        adapter.representative_blocks = cls._representative_blocks(model)
        if override_path:
            payload = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
            adapter.source = f"override:{override_path}"
            adapter.stem = payload.get("stem", adapter.stem)
            adapter.stages = list(payload.get("stages", adapter.stages))
            adapter.representative_blocks = list(payload.get("representative_blocks", adapter.representative_blocks))
            adapter.layouts = dict(payload.get("layouts", {}))
            missing = [name for name in ([adapter.stem] if adapter.stem else []) + adapter.stages if name not in modules]
            if missing:
                raise KeyError(f"adapter override references missing modules: {missing}")
        return adapter

    @staticmethod
    def _semantic_scan(model: nn.Module) -> SemanticOps:
        result = SemanticOps()
        for name, module in model.named_modules():
            if not name:
                continue
            lowered = f"{name} {type(module).__name__}".lower()
            if isinstance(module, nn.Conv2d):
                kernel = module.kernel_size
                if max(kernel) > 1:
                    result.spatial_mixers.append(name)
                if module.groups == module.in_channels == module.out_channels and max(kernel) > 1:
                    result.depthwise_convs.append(name)
                if kernel == (1, 1):
                    result.channel_mixers.append(name)
            elif isinstance(module, nn.Linear):
                result.channel_mixers.append(name)
            if not isinstance(module, nn.Identity) and any(
                    token in lowered for token in ("squeezeexcite", "semodule", ".se ", "channelattention")):
                result.se_modules.append(name)
            if any(token in lowered for token in ("attention", "attn", "tokenmixer", "token_mixer", "windowattention")):
                result.attention_modules.append(name)
            if any(token in type(module).__name__.lower() for token in ("block", "invertedresidual")) and list(module.children()):
                result.residual_blocks.append(name)
        return result

    @staticmethod
    def _representative_blocks(model: nn.Module) -> List[str]:
        blocks = [name for name, module in model.named_modules() if name and list(module.children()) and any(
            token in type(module).__name__.lower()
            for token in ("block", "invertedresidual", "depthwiseseparableconv"))]
        if not blocks:
            return []
        picks = [blocks[0], blocks[len(blocks) // 2], blocks[-1]]
        return list(dict.fromkeys(picks))

    def apply_deploy(self, model: nn.Module) -> nn.Module:
        return reparameterize_model(model, inplace=False)
