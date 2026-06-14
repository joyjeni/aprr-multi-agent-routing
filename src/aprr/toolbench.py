"""
ToolBench-derived benchmark harness.

ToolBench (Qin et al., 2024) provides three splits of API-tool usage queries:

    G1  — single-tool, single-step
    G2  — multi-tool,  single-step
    G3  — multi-tool,  multi-step

Each split implies a different optimal routing depth through our 8-agent
topology. We construct a *deterministic* simulator that emits queries with
ground-truth optimal paths so that:

    - Oracle achieves 100% success at minimum hops
    - Random/RoundRobin/StaticSemantic/LLMRouter/APRR are scored against the
      same ground-truth distribution
    - Results are exactly reproducible across runs (fixed RNG)

Reference: Qin et al., "ToolLLM: Facilitating Large Language Models to Master
16000+ Real-World APIs", ICLR 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .agents import AgentTopology


# Canonical reference paths through the 8-agent topology
#   intent=0, search=1, retrieve=2, rerank=3, tool=4, compose=5, verify=6, respond=7
GT_PATHS: Dict[str, List[List[int]]] = {
    "G1": [
        [0, 7],
        [0, 5, 7],
        [0, 3, 7],
        [0, 2, 7],
    ],
    "G2": [
        [0, 2, 3, 4, 7],
        [0, 1, 3, 4, 7],
        [0, 2, 4, 5, 7],
    ],
    "G3": [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 4, 5, 6, 7],
        [0, 2, 3, 4, 5, 6, 7],
    ],
}


@dataclass
class ToolBenchQuery:
    qid: int
    split: str
    text: str
    embedding: np.ndarray
    gt_path: List[int]


# ---------------------------------------------------------------------------
class ToolBenchSimulator:
    """Reproducible ToolBench-style query generator + agent simulator."""

    def __init__(self, topology: AgentTopology, seed: int = 42,
                 split_mix: Tuple[float, float, float] = (0.5, 0.3, 0.2)):
        self.topo = topology
        self.rng = np.random.default_rng(seed)
        self.split_mix = split_mix
        self._queries: List[ToolBenchQuery] = []

    def generate(self, n: int = 1000) -> List[ToolBenchQuery]:
        splits = self.rng.choice(["G1", "G2", "G3"], size=n, p=self.split_mix)
        queries: List[ToolBenchQuery] = []
        d = self.topo.embed_dim
        for qid, split in enumerate(splits):
            gt = self._pick_path(split)
            emb = self.topo.embeddings[gt].mean(axis=0)
            emb = emb + self.rng.normal(0, 0.15, size=d).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            text = f"[{split}] synthesised query #{qid} (path={gt})"
            queries.append(ToolBenchQuery(qid, split, text, emb, gt))
        self._queries = queries
        return queries

    def _pick_path(self, split: str) -> List[int]:
        candidates = GT_PATHS[split]
        return list(candidates[int(self.rng.integers(0, len(candidates)))])

    def rollout(self, path: List[int], gt_path: List[int]) -> Tuple[bool, float]:
        """
        Simulate executing `path` and return (success, latency_ms).

        Success model (ToolBench-style):
            1. order-preserving subsequence coverage of gt_path
            2. terminal = path ends at a terminal agent
            3. P(success) = lcs_coverage × terminal × mean(success_bias)
        Order-preserving coverage rewards routers that learn the correct
        sequence of agents, not just the set — closer to ToolBench's
        ``pass_rate`` which checks both API selection and call order.
        """
        latency = 0.0
        for idx in path:
            latency += self.topo.agents[idx].sample_latency(self.rng)
        # longest common subsequence between path and gt_path
        m, k = len(path), len(gt_path)
        dp = [[0] * (k + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(k):
                dp[i + 1][j + 1] = (dp[i][j] + 1
                                    if path[i] == gt_path[j]
                                    else max(dp[i][j + 1], dp[i + 1][j]))
        lcs = dp[m][k]
        coverage = lcs / len(gt_path)
        terminal_ok = float(self.topo.agents[path[-1]].is_terminal)
        on_path = [n for n in path if n in set(gt_path)]
        avg_bias = float(np.mean([self.topo.agents[n].success_bias
                                   for n in on_path] or [0.5]))
        p_success = (coverage ** 1.5) * terminal_ok * avg_bias
        success = bool(self.rng.random() < p_success)
        return success, float(latency)

    def gt_path_of(self, qid: int) -> List[int]:
        return self._queries[qid].gt_path
