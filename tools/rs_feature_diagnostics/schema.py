"""Small, serialisable evidence records shared by reports and renderers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


SCHEMA_VERSION = 2


class RecordMixin:
    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass
class SampleRecord(RecordMixin):
    sample_id: str
    relative_path: str
    target: int
    source: str = "population"


@dataclass
class PredictionRecord(RecordMixin):
    run_id: str
    model: str
    sample_id: str
    pred_index: int
    target_index: int
    confidence: Optional[float]
    true_probability: Optional[float]
    top1_top2_margin: Optional[float]
    gt_margin: Optional[float]
    status: str = "complete"


@dataclass
class ArtifactRecord(RecordMixin):
    artifact_id: str
    artifact_type: str
    path: str
    sample_id: Optional[str] = None
    model: Optional[str] = None
    stage: Optional[str] = None
    condition: Optional[str] = None
    scope: str = "diagnostic"
    status: str = "complete"
    reason: str = ""
    normalization: dict[str, Any] = field(default_factory=dict)
    raw_array: Optional[str] = None


@dataclass
class InterventionRecord(RecordMixin):
    method: str
    sample_id: str
    requested_ratio: Optional[float]
    retained_count: Optional[int]
    total_count: Optional[int]
    replacement: str
    seed: Optional[int]
    baseline_pred: Optional[int]
    after_pred: Optional[int]
    transition: str
    gt_margin_before: Optional[float] = None
    gt_margin_after: Optional[float] = None


@dataclass
class EvidenceRecord(RecordMixin):
    evidence_id: str
    kind: str
    fact: str
    n: Optional[int]
    denominator: Optional[int]
    comparison: str
    source_file: str
    row_key: str
    figures: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    support: str = "observation"

