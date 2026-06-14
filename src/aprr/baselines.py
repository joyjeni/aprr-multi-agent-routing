"""
Baseline routers compared against APRR.

    1. RandomRouter         — uniform sampling over agents
    2. RoundRobinRouter     — fixed cyclic order
    3. StaticSemanticRouter — greedy semantic similarity (no learning)
    4. LLMRouter            — REINFORCE-trained softmax router
                              (proxy for an LLM-based routing controller)
    5. OracleRouter         — privileged ground-truth path (upper bound)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .agents import AgentTopology


class _BaseRouter:
    name = "Base"
    def __init__(self, topology: AgentTopology, seed: int = 0):
        self.topo = topology
        self.rng = np.random.default_rng(seed)

    def update_trail(self, path, success, latency_ms):
        return

    def snapshot(self):
        return None


class RandomRouter(_BaseRouter):
    name = "Random"

    def route(self, query_emb: Optional[np.ndarray], start: int = 0,
              require_terminal: bool = True, max_hops: int = 8) -> List[int]:
        path, visited = [start], {start}
        for _ in range(max_hops - 1):
            cands = [j for j in range(self.topo.n) if j != path[-1] and j not in visited]
            if not cands:
                break
            j = int(self.rng.choice(cands))
            path.append(j); visited.add(j)
            if require_terminal and self.topo.agents[j].is_terminal:
                break
        return path


class RoundRobinRouter(_BaseRouter):
    name = "RoundRobin"
    def __init__(self, topology, seed=0):
        super().__init__(topology, seed)
        self._counter = 0

    def route(self, query_emb=None, start=0, require_terminal=True, max_hops=8):
        order = [(start + 1 + (self._counter + k) % (self.topo.n - 1)) % self.topo.n
                 for k in range(self.topo.n - 1)]
        self._counter += 1
        path = [start]
        for j in order[: max_hops - 1]:
            path.append(j)
            if require_terminal and self.topo.agents[j].is_terminal:
                break
        return path


class StaticSemanticRouter(_BaseRouter):
    name = "StaticSemantic"

    def route(self, query_emb: Optional[np.ndarray], start=0,
              require_terminal=True, max_hops=8) -> List[int]:
        path, visited = [start], {start}
        for _ in range(max_hops - 1):
            i = path[-1]
            cands = np.array([j for j in range(self.topo.n) if j != i and j not in visited])
            if cands.size == 0:
                break
            eta = self.topo.heuristic[i, cands].copy()
            if query_emb is not None:
                e = self.topo.embeddings[cands]
                sim = (e @ query_emb) / ((np.linalg.norm(e, axis=1) + 1e-9)
                                          * (np.linalg.norm(query_emb) + 1e-9))
                eta = 0.5 * eta + 0.5 * np.clip(sim, 1e-3, 1.0)
            j = int(cands[int(np.argmax(eta))])
            path.append(j); visited.add(j)
            if require_terminal and self.topo.agents[j].is_terminal:
                break
        return path


class LLMRouter(_BaseRouter):
    """REINFORCE-trained softmax router (proxy for an LLM-based controller)."""
    name = "LLMRouter"

    def __init__(self, topology, seed=0, lr: float = 0.05):
        super().__init__(topology, seed)
        self.W = self.rng.normal(0, 0.1, size=(topology.n, topology.n))
        self.lr = lr

    def route(self, query_emb=None, start=0, require_terminal=True, max_hops=8):
        path, visited = [start], {start}
        self._last_logprobs = []
        for _ in range(max_hops - 1):
            i = path[-1]
            cands = np.array([j for j in range(self.topo.n) if j != i and j not in visited])
            if cands.size == 0:
                break
            logits = self.W[i, cands]
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            j_idx = int(self.rng.choice(len(cands), p=probs))
            j = int(cands[j_idx])
            self._last_logprobs.append((i, j, np.log(probs[j_idx] + 1e-12)))
            path.append(j); visited.add(j)
            if require_terminal and self.topo.agents[j].is_terminal:
                break
        return path

    def update_trail(self, path, success, latency_ms):
        reward = (1.0 if success else -0.2) - min(1.0, latency_ms / 1500.0)
        for (i, j, lp) in getattr(self, "_last_logprobs", []):
            self.W[i, j] += self.lr * reward


class OracleRouter(_BaseRouter):
    """Upper-bound router with privileged access to the ground-truth path."""
    name = "Oracle"
    def __init__(self, topology, ground_truth_fn, seed=0):
        super().__init__(topology, seed)
        self.gt = ground_truth_fn
        self._next_qid = None

    def set_query(self, qid: int):
        self._next_qid = qid

    def route(self, query_emb=None, start=0, require_terminal=True, max_hops=8):
        return list(self.gt(self._next_qid))[:max_hops]
