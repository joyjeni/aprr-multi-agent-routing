"""
APRR — Adaptive Probabilistic Routing Reinforcement.

Core algorithm (Equations 1–4 in the manuscript).

    (1)  W_ij  ← initialised to W0 for all (i, j) ∈ E
    (2)  P(a_j | a_i, q) ∝ (W_ij^α · η_ij^β · ψ_j(q)^γ)
                       where η_ij = sim(e_i, e_j),  ψ_j(q) = sim(q, e_j)
    (3)  ΔW_ij = κ · (success / cost)        for successful trajectories
    (4)  W_ij ← (1 − λ) · W_ij + ΔW_ij       (decay-regularised reinforcement)

This is a decay-regularised online policy update with a multiplicative
semantic prior — equivalent to a REINFORCE step with a learned baseline
when α=γ=1 and β=0 (see Appendix A in the manuscript).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .agents import AgentTopology


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RouterConfig:
    alpha: float = 2.0       # learned-affinity weight
    beta: float = 1.0        # semantic-prior weight (role-role)
    gamma: float = 2.5       # query-relevance weight (query-role)
    lam: float = 0.005       # affinity decay rate λ (decay-regularisation)
    kappa: float = 5.0       # reinforcement scale κ
    W0: float = 0.1          # initial routing affinity
    W_min: float = 1e-3      # clamp
    W_max: float = 20.0      # clamp
    max_hops: int = 8        # episode horizon
    epsilon: float = 0.15    # ε-greedy exploration
    epsilon_decay: float = 0.98   # multiplicative epsilon schedule
    epsilon_min: float = 0.01
    seed: int = 0


# ---------------------------------------------------------------------------
# APRR Router
# ---------------------------------------------------------------------------
class APRRRouter:
    """Adaptive Probabilistic Routing Reinforcement over an :class:`AgentTopology`."""

    name = "APRR"

    def __init__(self, topology: AgentTopology, config: RouterConfig | None = None):
        self.topo = topology
        self.cfg = config or RouterConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        n = topology.n
        # routing-affinity matrix W_ij
        self.W = np.full((n, n), self.cfg.W0, dtype=np.float64)
        np.fill_diagonal(self.W, 0.0)
        # bookkeeping for figures and dashboards
        self.W_history: List[np.ndarray] = []
        self.iter_metrics: List[dict] = []

    # ------------------------------------------------------------------ API
    def select_next(
        self,
        current: int,
        visited: set[int],
        query_emb: Optional[np.ndarray] = None,
    ) -> int:
        """Sample next agent j from policy P(a_j | a_current, q)."""
        n = self.topo.n
        candidates = [j for j in range(n) if j != current and j not in visited]
        if not candidates:
            candidates = [j for j in range(n) if j != current]

        cand_arr = np.array(candidates)
        W_ij  = self.W[current, cand_arr]
        eta   = self.topo.heuristic[current, cand_arr]

        if query_emb is not None and self.cfg.gamma > 0.0:
            agent_emb = self.topo.embeddings[cand_arr]
            an = np.linalg.norm(agent_emb, axis=1) + 1e-9
            qn = np.linalg.norm(query_emb) + 1e-9
            psi = np.clip((agent_emb @ query_emb) / (an * qn), 1e-3, 1.0)
        else:
            psi = np.ones_like(eta)

        # ε-greedy exploration to escape local optima early in training
        if self.rng.random() < self.cfg.epsilon:
            return int(self.rng.choice(cand_arr))

        weights = (W_ij ** self.cfg.alpha) \
                * (eta  ** self.cfg.beta)  \
                * (psi  ** self.cfg.gamma)
        s = weights.sum()
        if not np.isfinite(s) or s <= 0.0:
            return int(self.rng.choice(cand_arr))
        probs = weights / s
        return int(self.rng.choice(cand_arr, p=probs))

    # --------------------------------------------------------------- update
    def update_trail(
        self,
        path: List[int],
        success: bool,
        latency_ms: float,
        coverage: float | None = None,
    ) -> None:
        """Decay W then reinforce edges along a successful trajectory (Eq. 3-4).

        Reinforcement signal:
            r = (success ? 1 : -0.05) * (1 / L)² * (1 / latency_norm)
        Short, fast, successful paths receive quadratically larger deposits
        than long ones — this is what enables APRR to discover "shortcuts".
        """
        # 1) global decay (temporal discount)
        self.W *= (1.0 - self.cfg.lam)

        # 2) deposit ΔW on traversed edges
        if len(path) >= 2:
            L = len(path) - 1                     # number of edges
            lat_norm = max(latency_ms, 1.0) / 200.0  # scale to ~O(1)
            reward = 1.0 if success else -0.05
            deposit = self.cfg.kappa * reward * (1.0 / (L * L)) * (1.0 / lat_norm)
            for i, j in zip(path[:-1], path[1:]):
                self.W[i, j] += deposit

        # 3) decay exploration
        self.cfg.epsilon = max(self.cfg.epsilon_min,
                               self.cfg.epsilon * self.cfg.epsilon_decay)

        # 4) clamp
        np.clip(self.W, self.cfg.W_min, self.cfg.W_max, out=self.W)
        np.fill_diagonal(self.W, 0.0)

    # --------------------------------------------------------------- route
    def route(
        self,
        query_emb: Optional[np.ndarray],
        start: int = 0,
        require_terminal: bool = True,
        max_hops: int | None = None,
    ) -> List[int]:
        """Sampled routing producing a hop-sequence starting at `start`."""
        max_hops = max_hops or self.cfg.max_hops
        path = [start]
        visited = {start}
        for _ in range(max_hops - 1):
            j = self.select_next(path[-1], visited, query_emb)
            path.append(j)
            visited.add(j)
            if require_terminal and self.topo.agents[j].is_terminal:
                break
        return path

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> np.ndarray:
        return self.W.copy()

    # backward-compat alias for any external caller
    @property
    def tau(self):  # pragma: no cover
        return self.W
