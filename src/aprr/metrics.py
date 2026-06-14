"""Evaluation metrics for ACCR vs. baselines."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeResult:
    router: str
    query_id: int
    split: str           # "G1" | "G2" | "G3"
    path: List[int]
    success: bool
    latency_ms: float
    hops: int
    iteration: int       # training/evaluation step


def aggregate_metrics(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-router metrics suitable for the paper's Table I."""
    agg: Dict[str, Dict[str, float]] = {}
    by_router: Dict[str, List[EpisodeResult]] = {}
    for r in results:
        by_router.setdefault(r.router, []).append(r)
    for name, rs in by_router.items():
        succ = np.array([r.success for r in rs], dtype=np.float64)
        lat  = np.array([r.latency_ms for r in rs])
        hops = np.array([r.hops for r in rs])
        agg[name] = {
            "n": float(len(rs)),
            "success_rate": float(succ.mean()),
            "success_rate_ci95": float(1.96 * succ.std(ddof=1) / max(np.sqrt(len(rs)), 1)),
            "mean_latency_ms": float(lat.mean()),
            "p50_latency_ms": float(np.percentile(lat, 50)),
            "p95_latency_ms": float(np.percentile(lat, 95)),
            "mean_hops": float(hops.mean()),
            "median_hops": float(np.median(hops)),
            "successful_mean_latency_ms": float(lat[succ.astype(bool)].mean()
                                                if succ.any() else float("nan")),
        }
    return agg
