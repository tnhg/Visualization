from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from .adapters import GenericTimmAdapter
from .tensor_adapter import FeatureTensorAdapter, first_tensor
from .utils import write_json


@dataclass
class ModuleRecord:
    order: int
    name: str
    module_type: str
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    parameter_count: int
    kernel_size: Any = None
    stride: Any = None
    groups: Any = None
    layout: Optional[str] = None
    grid_size: Optional[Tuple[int, int]] = None


@dataclass
class InspectionResult:
    records: List[ModuleRecord]
    stem: Optional[str]
    stages: List[str]
    representative_blocks: List[str]
    semantic_ops: Dict[str, str]
    source: str
    ambiguous: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stem": self.stem,
            "stages": self.stages,
            "representative_blocks": self.representative_blocks,
            "semantic_ops": self.semantic_ops,
            "source": self.source,
            "ambiguous": self.ambiguous,
            "records": [asdict(record) for record in self.records],
        }


class ModelInspector:
    def __init__(self, model: nn.Module, input_size: Tuple[int, int, int], device: torch.device):
        self.model = model
        self.input_size = input_size
        self.device = device
        self.tensor_adapter = FeatureTensorAdapter()

    def inspect(self, adapter: GenericTimmAdapter) -> InspectionResult:
        records: List[ModuleRecord] = []
        handles = []
        order = 0

        def make_hook(name: str, module: nn.Module):
            def hook(_module, inputs, output):
                nonlocal order
                tensor = first_tensor(output)
                input_tensor = first_tensor(inputs)
                output_shape = tuple(tensor.shape) if tensor is not None else None
                grid = None
                layout = None
                if tensor is not None and tensor.ndim in (3, 4):
                    try:
                        adapted = self.tensor_adapter.adapt(
                            tensor, layout_hint=adapter.layouts.get(name), warn=False)
                        grid, layout = adapted.grid_size, adapted.source_layout
                    except ValueError:
                        pass
                records.append(ModuleRecord(
                    order=order, name=name, module_type=type(module).__name__,
                    input_shape=tuple(input_tensor.shape) if input_tensor is not None else None,
                    output_shape=output_shape,
                    parameter_count=sum(p.numel() for p in module.parameters(recurse=False)),
                    kernel_size=getattr(module, "kernel_size", None),
                    stride=getattr(module, "stride", None),
                    groups=getattr(module, "groups", None),
                    layout=layout, grid_size=grid,
                ))
                order += 1
            return hook

        for name, module in self.model.named_modules():
            if name:
                handles.append(module.register_forward_hook(make_hook(name, module)))
        try:
            example = torch.zeros((1,) + self.input_size, device=self.device)
            with torch.inference_mode():
                self.model(example)
        finally:
            for handle in handles:
                handle.remove()

        stages = list(adapter.stages)
        source = adapter.source
        if not stages:
            stages = self._auto_stages(records)
            source = "auto_resolution_hierarchy"
        modules = dict(self.model.named_modules())
        stages = [name for name in stages if name in modules]
        distinct_grids = {self._grid_for(name, records) for name in stages}
        ambiguous = len(stages) < 2 or (source == "auto_resolution_hierarchy" and len(distinct_grids) < 2)
        stem = adapter.stem or (stages[0] if stages else None)
        return InspectionResult(
            records=records, stem=stem, stages=stages,
            representative_blocks=adapter.representative_blocks,
            semantic_ops=adapter.semantic_ops.summary(), source=source, ambiguous=ambiguous,
        )

    @staticmethod
    def _grid_for(name: str, records: List[ModuleRecord]):
        matches = [r.grid_size for r in records if r.name == name and r.grid_size]
        return matches[-1] if matches else None

    @staticmethod
    def _auto_stages(records: List[ModuleRecord]) -> List[str]:
        by_grid: Dict[Tuple[int, int], ModuleRecord] = {}
        for record in records:
            if record.grid_size and min(record.grid_size) > 1 and record.output_shape:
                channels = record.output_shape[1] if record.layout == "NCHW" else record.output_shape[-1]
                if channels >= 4:
                    by_grid[record.grid_size] = record
        ordered = sorted(by_grid.values(), key=lambda r: r.grid_size[0] * r.grid_size[1], reverse=True)
        if len(ordered) > 5:
            ordered = ordered[-5:]
        return [record.name for record in ordered]

    @staticmethod
    def save(result: InspectionResult, output_dir: Path) -> None:
        write_json(output_dir / "manifest" / "model_structure.json", result.to_dict())
        lines = [
            f"source: {result.source}", f"stem: {result.stem}",
            "stages:", *[f"  - {name}" for name in result.stages],
            "representative_blocks:", *[f"  - {name}" for name in result.representative_blocks],
            "semantic_ops:", *[f"  {key}: {value}" for key, value in result.semantic_ops.items()],
            "", "module records:",
        ]
        lines.extend(
            f"{r.order:04d} {r.name:<60} {r.module_type:<28} {r.input_shape} -> {r.output_shape} "
            f"grid={r.grid_size} layout={r.layout} params={r.parameter_count}"
            for r in result.records
        )
        (output_dir / "manifest" / "model_structure.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
