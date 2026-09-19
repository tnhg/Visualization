from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-12


def spatial_maps(feature: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
    x = feature.float()
    energy = torch.sqrt(torch.sum(x.square(), dim=1) + EPS)
    variance = torch.var(x, dim=1, unbiased=False)
    probability = torch.softmax(x.abs() / max(temperature, EPS), dim=1)
    entropy = -(probability * torch.log(probability + EPS)).sum(dim=1)
    return {"energy": energy, "variance": variance, "entropy": entropy}


def token_matrix(feature: torch.Tensor) -> torch.Tensor:
    return feature.flatten(2).transpose(1, 2).float()


def token_similarity(
    feature: torch.Tensor,
    max_tokens: int = 512,
    seed: int = 1234,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    tokens = token_matrix(feature)[0]
    total = tokens.shape[0]
    if total > max_tokens:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(total, generator=generator)[:max_tokens].sort().values.to(tokens.device)
        tokens = tokens[indices]
    else:
        indices = torch.arange(total, device=tokens.device)
    normalized = F.normalize(tokens, dim=1, eps=EPS)
    matrix = normalized @ normalized.T
    if matrix.shape[0] > 1:
        values = matrix[~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)]
    else:
        values = matrix.flatten()
    stats = {
        "mean_off_diagonal": values.mean().item(),
        "median": values.median().item(),
        "p90": torch.quantile(values, .9).item(),
        "sampled_tokens": int(tokens.shape[0]),
        "total_tokens": int(total),
    }
    return matrix, stats, indices


def effective_rank(feature: torch.Tensor) -> Dict[str, float]:
    tokens = token_matrix(feature)[0]
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(tokens)
    if singular.numel() == 0 or singular.sum() <= EPS:
        return {"entropy_effective_rank": 0., "energy_rank_90": 0., "energy_rank_95": 0.}
    p = singular / singular.sum()
    entropy_rank = torch.exp(-(p * torch.log(p + EPS)).sum()).item()
    energy = singular.square()
    cumulative = torch.cumsum(energy / (energy.sum() + EPS), dim=0)
    rank90 = int(torch.searchsorted(cumulative, torch.tensor(.9, device=cumulative.device)).item() + 1)
    rank95 = int(torch.searchsorted(cumulative, torch.tensor(.95, device=cumulative.device)).item() + 1)
    return {"entropy_effective_rank": entropy_rank, "energy_rank_90": rank90, "energy_rank_95": rank95}


def homogeneous_proxy(image: torch.Tensor, output_size: Tuple[int, int], quantile: float = .3) -> torch.Tensor:
    x = image.float().mean(dim=0, keepdim=True).unsqueeze(0)
    local_mean = F.avg_pool2d(x, 5, stride=1, padding=2)
    local_var = F.avg_pool2d(x.square(), 5, stride=1, padding=2) - local_mean.square()
    sobel_x = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gradient = torch.sqrt(F.conv2d(x, sobel_x, padding=1).square() + F.conv2d(x, sobel_y, padding=1).square() + EPS)
    texture = local_var.clamp_min(0).sqrt() + gradient
    texture = F.interpolate(texture, output_size, mode="bilinear", align_corners=False)[0, 0]
    threshold = torch.quantile(texture.flatten(), quantile)
    return texture <= threshold


def subset_similarity(feature: torch.Tensor, mask: torch.Tensor) -> float:
    tokens = token_matrix(feature)[0]
    selected = tokens[mask.flatten()]
    if selected.shape[0] < 2:
        return float("nan")
    selected = F.normalize(selected, dim=1, eps=EPS)
    matrix = selected @ selected.T
    values = matrix[~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)]
    return values.mean().item()


def channel_similarity(feature: torch.Tensor, method: str = "cosine") -> torch.Tensor:
    channels = feature[0].flatten(1).float()
    if method == "cosine":
        channels = F.normalize(channels, dim=1, eps=EPS)
        return channels @ channels.T
    if method == "pearson":
        channels = channels - channels.mean(dim=1, keepdim=True)
        channels = F.normalize(channels, dim=1, eps=EPS)
        return channels @ channels.T
    raise ValueError(method)


def similarity_summary(matrix: torch.Tensor, threshold: float = .9, absolute: bool = False) -> Dict[str, float]:
    n = matrix.shape[0]
    if n < 2:
        return {"mean_off_diagonal": 0., "p90": 0., "high_pair_ratio": 0.}
    values = matrix[~torch.eye(n, dtype=torch.bool, device=matrix.device)]
    scored = values.abs() if absolute else values
    return {
        "mean_off_diagonal": values.mean().item(),
        "p90": torch.quantile(values, .9).item(),
        "high_pair_ratio": (scored >= threshold).float().mean().item(),
    }


def scale_preferences(response: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # response: scale x channel
    maxima, indices = response.max(dim=0)
    preference = scales.to(response.device)[indices]
    selectivity = (maxima - response.mean(dim=0)) / (maxima.abs() + EPS)
    return preference, selectivity


def erf_radii(energy: torch.Tensor, fractions=(.5, .8, .9)) -> Dict[str, float]:
    energy = energy.float().clamp_min(0)
    h, w = energy.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=energy.device), torch.arange(w, device=energy.device), indexing="ij")
    cy, cx = (h - 1) / 2., (w - 1) / 2.
    distance = torch.sqrt((yy - cy).square() + (xx - cx).square()).flatten()
    values = energy.flatten()
    order = torch.argsort(distance)
    cumulative = torch.cumsum(values[order], 0) / (values.sum() + EPS)
    sorted_distance = distance[order]
    result: Dict[str, float] = {}
    normalizer = float(max(h, w))
    for fraction in fractions:
        index = min(int(torch.searchsorted(cumulative, torch.tensor(fraction, device=energy.device))), len(order) - 1)
        radius = sorted_distance[index].item()
        key = int(round(100 * fraction))
        result[f"r{key}_pixels"] = radius
        result[f"r{key}_normalized"] = radius / normalizer
    return result
