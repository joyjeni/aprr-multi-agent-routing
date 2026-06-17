"""
crow_router.py — Chain-of-Reasoning Over Workload (CROW) Router
===============================================================
An extension to the APRR multi-agent routing system that gates routing
decisions through a lightweight chain-of-thought deliberation mechanism.

- Low-complexity queries: greedy argmax over the affinity matrix W.
- High-complexity queries: multi-agent voting with CoT justifications,
  weighted by a reasoning-quality score.

No LLM calls — all complexity estimation and reasoning traces are
heuristic / template-based.

Dependencies: numpy, math, random, time, re  (all stdlib or numpy)
"""

from __future__ import annotations

import math
import random
import re
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------

_MULTI_STEP_KEYWORDS: list[str] = [
    "then", "after", "also", "compare", "and then", "followed by",
    "next", "finally", "step", "first", "second", "third",
    "additionally", "moreover", "furthermore",
]

_TOOL_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "weather": ["weather", "rain", "temperature", "forecast", "humidity", "wind"],
    "market_prices": ["price", "market", "commodity", "rate", "cost", "value", "rupee"],
    "crop_advisory": ["crop", "seed", "fertilizer", "pest", "irrigation", "harvest"],
    "schemes": ["scheme", "subsidy", "loan", "insurance", "government", "policy"],
    "soil_health": ["soil", "ph", "nutrient", "nitrogen", "organic", "moisture"],
}

# Domain keyword map for source-agent inference
_DOMAIN_KWS: dict[str, list[str]] = {
    "weather":       ["weather", "rain", "temperature", "forecast", "humidity", "wind", "monsoon"],
    "market_prices": ["price", "market", "mandi", "msp", "commodity", "rate", "cost"],
    "crop_advisory": ["crop", "seed", "fertilizer", "pest", "irrigation", "harvest", "variety"],
    "schemes":       ["scheme", "subsidy", "loan", "insurance", "kisan", "credit", "apply"],
    "soil_health":   ["soil", "ph", "nutrient", "nitrogen", "organic", "moisture", "carbon"],
}
_DOMAIN_ORDER = ["weather", "market_prices", "crop_advisory", "schemes", "soil_health"]


def _infer_source_row(query: str, n_agents: int) -> int:
    """Infer source row in W from query domain keywords."""
    q = query.lower()
    best_idx = 0
    best_hits = 0
    for idx, domain in enumerate(_DOMAIN_ORDER[:n_agents]):
        hits = sum(1 for kw in _DOMAIN_KWS.get(domain, []) if kw in q)
        if hits > best_hits:
            best_hits = hits
            best_idx = idx
    return best_idx


_COMPLEXITY_CATEGORY_LABELS: dict[str, str] = {
    "weather": "meteorological analysis",
    "market_prices": "market data retrieval",
    "crop_advisory": "agronomic advisory",
    "schemes": "government scheme lookup",
    "soil_health": "soil science analysis",
    "general": "general-purpose reasoning",
}


# ---------------------------------------------------------------------------
# CROWRouter
# ---------------------------------------------------------------------------

class CROWRouter:
    """
    Chain-of-Reasoning Over Workload (CROW) Router.

    Extends APRR by introducing a deliberation gate: queries whose
    heuristic complexity score exceeds ``thought_budget_threshold`` trigger
    a multi-round voting process among the top-k candidate agents, each
    producing a lightweight reasoning trace. The winning agent is chosen
    by reasoning-quality-weighted vote aggregation.

    Parameters
    ----------
    agents : list[str]
        Human-readable agent names (length N).
    W_init : np.ndarray, optional
        Initial N×N affinity matrix.  Defaults to the identity scaled by 0.5.
    lambda_decay : float
        Multiplicative decay rate applied after every update step.
    alpha : float
        Learning-rate multiplier on the raw APRR delta (not used in the
        base formula but exposed for downstream experimentation).
    thought_budget_threshold : float
        Complexity score threshold in [0, 1].  Queries above this value
        enter deliberation mode.
    max_deliberation_rounds : int
        Maximum rounds of voting in deliberation mode.
    """

    # Reasoning-quality weight applied to success delta
    _BETA: float = 0.3
    # Number of top-k agents considered during deliberation
    _TOP_K: int = 3
    # Normalisation constant for latency (ms)
    _LATENCY_NORM: float = 500.0

    def __init__(
        self,
        agents: list[str],
        W_init: Optional[np.ndarray] = None,
        lambda_decay: float = 0.05,
        alpha: float = 0.3,
        thought_budget_threshold: float = 0.5,
        max_deliberation_rounds: int = 3,
    ) -> None:
        self.agents = agents
        self.n_agents = len(agents)
        self.lambda_decay = lambda_decay
        self.alpha = alpha
        self.thought_budget_threshold = thought_budget_threshold
        self.max_deliberation_rounds = max_deliberation_rounds

        if W_init is not None:
            if W_init.shape != (self.n_agents, self.n_agents):
                raise ValueError(
                    f"W_init shape {W_init.shape} does not match "
                    f"({self.n_agents}, {self.n_agents})."
                )
            self.W = W_init.astype(float).copy()
        else:
            self.W = np.eye(self.n_agents) * 0.5

        # Per-agent success histories  {agent_idx: [bool, ...]}
        self._success_history: dict[int, list[bool]] = {
            i: [] for i in range(self.n_agents)
        }

    # ------------------------------------------------------------------
    # Complexity estimation
    # ------------------------------------------------------------------

    def predict_complexity(self, query: str) -> float:
        """
        Heuristic complexity score in [0, 1].

        Components
        ----------
        1. Normalised average token length  (longer tokens → richer vocab)
        2. Fraction of tool-category buckets mentioned
        3. Multi-step keyword density

        Returns
        -------
        float
            Complexity score in [0, 1].
        """
        tokens = query.lower().split()
        if not tokens:
            return 0.0

        # --- Component 1: avg token length (normalised to ~0..1)
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        len_score = min(1.0, avg_len / 10.0)  # 10-char avg → score 1.0

        # --- Component 2: tool-category coverage
        categories_hit = 0
        for kws in _TOOL_CATEGORY_PATTERNS.values():
            if any(kw in query.lower() for kw in kws):
                categories_hit += 1
        cat_score = categories_hit / len(_TOOL_CATEGORY_PATTERNS)

        # --- Component 3: multi-step keyword density
        ms_hits = sum(
            1 for kw in _MULTI_STEP_KEYWORDS if re.search(r"\b" + kw + r"\b", query.lower())
        )
        ms_score = min(1.0, ms_hits / 3.0)  # 3+ keywords → score 1.0

        # Weighted combination
        complexity = 0.3 * len_score + 0.4 * cat_score + 0.3 * ms_score
        return float(np.clip(complexity, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Reasoning trace generation
    # ------------------------------------------------------------------

    def generate_reasoning_trace(self, query: str, agent_idx: int) -> dict:
        """
        Produce a template-based reasoning trace for ``agent_idx`` on ``query``.

        Returns
        -------
        dict with keys:
            agent         : int
            rationale     : str
            confidence    : float
            reasoning_steps : int
        """
        # Determine dominant category
        best_cat = "general"
        best_hits = 0
        for cat, kws in _TOOL_CATEGORY_PATTERNS.items():
            hits = sum(1 for kw in kws if kw in query.lower())
            if hits > best_hits:
                best_hits = hits
                best_cat = cat
        category_label = _COMPLEXITY_CATEGORY_LABELS.get(best_cat, "general reasoning")

        # Affinity score for this agent (use max of row as proxy for "this agent's W")
        w_val = float(self.W[agent_idx, agent_idx]) if self.n_agents > 0 else 0.0
        # Also pull the max off-diagonal for this agent's incoming weight
        col_max = float(np.max(self.W[:, agent_idx]))
        effective_w = max(w_val, col_max)

        # Success history summary
        hist = self._success_history[agent_idx]
        if hist:
            sr = sum(hist) / len(hist)
            hist_str = f"{sr:.0%} ({len(hist)} recent calls)"
        else:
            hist_str = "no history"

        # Confidence: blend affinity and success rate
        hist_sr = (sum(hist) / len(hist)) if hist else 0.5
        confidence = float(np.clip(0.6 * effective_w + 0.4 * hist_sr, 0.0, 1.0))

        # Reasoning steps: proportional to complexity
        complexity = self.predict_complexity(query)
        reasoning_steps = max(1, min(5, int(math.ceil(complexity * 5))))

        rationale = (
            f"Query '{query[:60]}{'...' if len(query) > 60 else ''}' requires "
            f"{category_label}. Agent {self.agents[agent_idx]} has affinity "
            f"W={effective_w:.3f}. Success history: {hist_str}. "
            f"Route confidence: {confidence:.2f}."
        )

        return {
            "agent": agent_idx,
            "rationale": rationale,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
        }

    # ------------------------------------------------------------------
    # Reasoning quality score
    # ------------------------------------------------------------------

    @staticmethod
    def score_reasoning_quality(trace: dict) -> float:
        """
        Score a reasoning trace in [0, 1].

        Formula
        -------
        ``min(1.0, confidence * (1 + 0.1 * min(steps, 5)))``

        Penalises low confidence; rewards more reasoning steps up to a cap
        of 5.

        Parameters
        ----------
        trace : dict
            Output of :meth:`generate_reasoning_trace`.

        Returns
        -------
        float
        """
        confidence: float = float(trace.get("confidence", 0.0))
        steps: int = int(trace.get("reasoning_steps", 1))
        score = confidence * (1.0 + 0.1 * min(steps, 5))
        return float(min(1.0, score))

    # ------------------------------------------------------------------
    # Deliberation
    # ------------------------------------------------------------------

    def deliberate(self, query: str, top_k_agents: list[int]) -> int:
        """
        Multi-agent voting with CoT justifications.

        Each agent in ``top_k_agents`` generates a reasoning trace.
        Votes are weighted by the reasoning-quality score of each trace.
        The process repeats for up to ``max_deliberation_rounds`` rounds;
        early-exit if the leading candidate accumulates ≥ 60 % of votes.

        Parameters
        ----------
        query : str
        top_k_agents : list[int]
            Indices of candidate agents.

        Returns
        -------
        int
            Index of the winning agent.
        """
        if not top_k_agents:
            return 0

        vote_accumulator: dict[int, float] = {a: 0.0 for a in top_k_agents}

        for round_idx in range(self.max_deliberation_rounds):
            round_votes: dict[int, float] = {a: 0.0 for a in top_k_agents}

            for candidate in top_k_agents:
                trace = self.generate_reasoning_trace(query, candidate)
                quality = self.score_reasoning_quality(trace)
                # Each agent votes for itself weighted by its own reasoning quality
                round_votes[candidate] += quality

            # Accumulate
            for a, v in round_votes.items():
                vote_accumulator[a] += v

            # Early exit check
            total = sum(vote_accumulator.values())
            if total > 0:
                winner = max(vote_accumulator, key=vote_accumulator.__getitem__)
                if vote_accumulator[winner] / total >= 0.60:
                    return winner

        # Final winner
        return max(vote_accumulator, key=vote_accumulator.__getitem__)

    # ------------------------------------------------------------------
    # Main routing function
    # ------------------------------------------------------------------

    def route(self, query: str) -> dict:
        """
        Route a query to an agent.

        - Complexity < threshold  → greedy APRR-style (argmax W column sum)
        - Complexity ≥ threshold  → deliberation among top-k agents

        Parameters
        ----------
        query : str

        Returns
        -------
        dict with keys:
            agent       : int
            latency_ms  : float
            reasoning   : dict  (last trace for the chosen agent)
            mode        : "greedy" | "deliberation"
        """
        t0 = time.perf_counter()

        complexity = self.predict_complexity(query)
        mode: str

        # Infer source row from query domain (APRR: W[i,:] where i = source agent)
        src_row = _infer_source_row(query, self.n_agents)

        if complexity < self.thought_budget_threshold:
            # --- Greedy: pick agent with highest affinity in source row
            row_scores = self.W[src_row]
            chosen_agent = int(np.argmax(row_scores))
            reasoning = self.generate_reasoning_trace(query, chosen_agent)
            mode = "greedy"
        else:
            # --- Deliberation: top-k candidates by source-row scores
            row_scores = self.W[src_row]
            top_k = min(self._TOP_K, self.n_agents)
            top_k_agents = list(np.argsort(row_scores)[-top_k:][::-1])
            chosen_agent = self.deliberate(query, top_k_agents)
            reasoning = self.generate_reasoning_trace(query, chosen_agent)
            mode = "deliberation"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "agent": chosen_agent,
            "agent_name": self.agents[chosen_agent],
            "latency_ms": round(latency_ms, 3),
            "reasoning": reasoning,
            "mode": mode,
            "complexity_score": round(complexity, 4),
        }

    # ------------------------------------------------------------------
    # W update
    # ------------------------------------------------------------------

    def update(
        self,
        agent_from: int,
        agent_to: int,
        success: bool,
        latency_ms: float,
        reasoning_quality: float,
    ) -> None:
        """
        Update the affinity matrix after an observed routing outcome.

        Formula
        -------
        ``ΔW[i,j] += 𝟙[success] · (1/L²) · (1/latency_norm) · (1 + β·reasoning_quality)``

        Then multiplicative decay: ``W *= (1 − λ)``

        Parameters
        ----------
        agent_from : int
            Source agent index (row in W).
        agent_to : int
            Destination agent index (column in W).
        success : bool
            Whether the routing outcome was successful.
        latency_ms : float
            Observed latency in milliseconds.
        reasoning_quality : float
            Quality score in [0, 1] returned by :meth:`score_reasoning_quality`.
        """
        # Update success history
        self._success_history[agent_to].append(success)
        if len(self._success_history[agent_to]) > 50:
            self._success_history[agent_to].pop(0)

        if success:
            # Hop count L: we treat a single hop as L=1 (extensible)
            L = 1
            latency_norm = max(latency_ms, 1.0) / self._LATENCY_NORM
            delta = (1.0 / (L ** 2)) * (1.0 / latency_norm) * (1.0 + self._BETA * reasoning_quality)
            self.W[agent_from, agent_to] += delta

        # Multiplicative decay
        self.W *= (1.0 - self.lambda_decay)

        # Clip to non-negative
        np.clip(self.W, 0.0, None, out=self.W)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CROWRouter(n_agents={self.n_agents}, "
            f"threshold={self.thought_budget_threshold}, "
            f"lambda={self.lambda_decay})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _make_synthetic_queries() -> list[str]:
    """Generate 20 ToolBench-style synthetic queries."""
    templates = [
        "What is the current temperature in Delhi?",
        "Show me the price of wheat and rice in Mandis, then compare with last week's rates.",
        "My soil pH is 6.2. Which crop should I plant after comparing nitrogen levels?",
        "List all government schemes for marginal farmers and also explain PM-KISAN.",
        "Forecast rain for the next 3 days.",
        "What fertilizer should I use for cotton after soil testing?",
        "Get the MSP for paddy this season.",
        "Which pesticide is effective against aphids in wheat? Then check availability.",
        "Soil moisture is low. Recommend irrigation schedule and then check weather.",
        "Explain the PM Fasal Bima Yojana scheme step by step.",
        "What is the current humidity in Rajasthan?",
        "Compare market prices of tomatoes across Mumbai, Delhi, and Kolkata.",
        "Best seed variety for drip irrigation in Maharashtra.",
        "How to apply for Kisan Credit Card loan?",
        "Nitrogen deficiency symptoms in maize and recommended treatment.",
        "Temperature forecast for wheat harvest period in Punjab.",
        "What is organic carbon percentage and how to improve it in black soil?",
        "Compare subsidy rates for solar pumps across different states.",
        "Is it going to rain tomorrow? Also advise on spraying schedule.",
        "Explain integrated pest management for paddy, then list approved chemicals.",
    ]
    return templates


if __name__ == "__main__":
    print("=" * 70)
    print("CROW Router — 20-Query Demo (ToolBench-style synthetic data)")
    print("=" * 70)

    agents = [
        "WeatherAgent", "MarketAgent", "CropAgent",
        "SchemeAgent", "SoilAgent",
    ]
    router = CROWRouter(
        agents=agents,
        lambda_decay=0.05,
        alpha=0.3,
        thought_budget_threshold=0.4,
        max_deliberation_rounds=3,
    )

    queries = _make_synthetic_queries()
    random.seed(42)

    header = (
        f"{'#':>3}  {'Mode':<14} {'Agent':<14} {'Complexity':>10} "
        f"{'Confidence':>10} {'Latency(ms)':>12}"
    )
    print(header)
    print("-" * len(header))

    for idx, q in enumerate(queries, 1):
        result = router.route(q)

        # Simulate outcome: success if agent name matches query domain keyword
        agent_name: str = result["agent_name"].lower()
        success = any(kw in q.lower() for kw in agent_name.replace("agent", "").split())
        sim_latency = random.uniform(50, 400)
        rq = router.score_reasoning_quality(result["reasoning"])

        router.update(
            agent_from=result["agent"],
            agent_to=result["agent"],
            success=success,
            latency_ms=sim_latency,
            reasoning_quality=rq,
        )

        print(
            f"{idx:>3}  {result['mode']:<14} {result['agent_name']:<14} "
            f"{result['complexity_score']:>10.4f} "
            f"{result['reasoning']['confidence']:>10.4f} "
            f"{result['latency_ms']:>12.3f}"
        )
        print(f"     Q: {q[:70]}")
        print(f"     R: {result['reasoning']['rationale'][:90]}...")
        print()

    print("Final affinity matrix W:")
    print(np.round(router.W, 4))
