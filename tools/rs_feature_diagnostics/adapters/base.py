from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torch import nn


@dataclass
class SemanticOps:
    spatial_mixers: List[str] = field(default_factory=list)
    depthwise_convs: List[str] = field(default_factory=list)
    channel_mixers: List[str] = field(default_factory=list)
    se_modules: List[str] = field(default_factory=list)
    attention_modules: List[str] = field(default_factory=list)
    residual_blocks: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, str]:
        return {
            "spatial_mixer": self.spatial_mixers[0] if self.spatial_mixers else "not_available",
            "dwconv": self.depthwise_convs[0] if self.depthwise_convs else "not_available",
            "pointwise_or_mlp": self.channel_mixers[0] if self.channel_mixers else "not_available",
            "se": self.se_modules[0] if self.se_modules else "not_available",
            "attention": self.attention_modules[0] if self.attention_modules else "not_available",
            "residual_output": self.residual_blocks[0] if self.residual_blocks else "not_available",
        }


@dataclass
class ModelAdapter:
    stem: Optional[str] = None
    stages: List[str] = field(default_factory=list)
    representative_blocks: List[str] = field(default_factory=list)
    layouts: Dict[str, str] = field(default_factory=dict)
    semantic_ops: SemanticOps = field(default_factory=SemanticOps)
    source: str = "auto"

    def apply_deploy(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError
