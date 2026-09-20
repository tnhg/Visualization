"""Exact signed contribution maps for verified global-average-pool + Linear heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class HeadContributionResult:
    status: str
    reason: str
    contribution: torch.Tensor | None = None
    target_index: int | None = None
    competitor_index: int | None = None
    reconstruction_error: float | None = None


def exact_head_contribution(
    feature: torch.Tensor, classifier: nn.Module, target_index: int, competitor_index: int,
    bias: torch.Tensor | None = None, tolerance: float = 1e-5,
) -> HeadContributionResult:
    """Return a signed map only when the supplied head is exactly Linear over GAP.

    The caller must provide the feature tensor immediately before global average
    pooling and the corresponding Linear classifier.  The reconstruction check
    prevents a visually plausible map from being reported for a head with an
    extra norm, activation, or non-average pooling operation.
    """
    if feature.ndim != 4:
        return HeadContributionResult("skipped", "feature must be NCHW")
    if not isinstance(classifier, nn.Linear):
        return HeadContributionResult("skipped", "classifier is not nn.Linear")
    if feature.shape[1] != classifier.in_features:
        return HeadContributionResult("skipped", "feature channels do not match classifier")
    classes = classifier.out_features
    if not (0 <= target_index < classes and 0 <= competitor_index < classes) or target_index == competitor_index:
        return HeadContributionResult("skipped", "invalid target/competitor indices")
    x = feature.float()
    delta_w = classifier.weight[target_index].float() - classifier.weight[competitor_index].float()
    contribution = (x * delta_w[None, :, None, None]).sum(dim=1) / (x.shape[-2] * x.shape[-1])
    delta_bias = (classifier.bias[target_index] - classifier.bias[competitor_index]).float() if classifier.bias is not None else torch.tensor(0., device=x.device)
    reconstructed = contribution.sum(dim=(-2, -1)) + delta_bias
    pooled = x.mean(dim=(-2, -1))
    expected = pooled @ delta_w + delta_bias
    error = float((reconstructed - expected).abs().max().detach().cpu())
    if error > tolerance:
        return HeadContributionResult(
            "skipped", f"exact reconstruction failed (max_abs_error={error:g})",
            target_index=target_index, competitor_index=competitor_index,
            reconstruction_error=error)
    return HeadContributionResult(
        "ok", "verified GAP + Linear signed contribution", contribution.detach(),
        target_index, competitor_index, error)

