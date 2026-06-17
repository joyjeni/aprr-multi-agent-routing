"""
octoroute_router.py — OctoRoute: Octopus-Inspired Distributed Routing
======================================================================
Two-layer routing architecture inspired by the octopus nervous system:

  - OctoArms (peripheral):  semi-autonomous routing units, each
    specialising in one query domain via a local affinity matrix.
  - OctoRouteRouter (brain): global coordinator that dispatches queries
    to the most suitable arm using fast chromatophore-style signals,
    then delegates intra-arm agent selection.

No LLM calls.  All signals are heuristic / token-overlap based.

Dependencies: numpy, math, random, time, collections  (stdlib + numpy)
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Domain keyword vocabulary
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "weather": [
        "weather", "rain", "temperature", "forecast", "humidity",
        "wind", "cloud", "storm", "monsoon", "climate", "cold", "hot",
    ],
    "market_prices": [
        "price", "market", "commodity", "rate", "cost", "value",
        "rupee", "mandi", "msp", "wholesale", "retail", "trade",
    ],
    "crop_advisory": [
        "crop", "seed", "fertilizer", "pest", "irrigation", "harvest",
        "sow", "plant", "variety", "yield", "agronomic", "spray",
    ],
    "schemes": [
        "scheme", "subsidy", "loan", "insurance", "government", "policy",
        "kisan", "credit", "benefit", "apply", "eligibility", "fasal",
    ],
    "soil_health": [
        "soil", "ph", "nutrient", "nitrogen", "organic", "moisture",
        "carbon", "potassium", "phosphorus", "deficiency", "texture",
    ],
}

_LATENCY_NORM: float = 500.0


# ---------------------------------------------------------------------------
# OctoArm
# ---------------------------------------------------------------------------

class OctoArm:
    """
    Semi-autonomous routing arm specialising in a single query domain.

    Each arm maintains its own local APRR affinity matrix (``W_local``)
    over the shared agent pool and emits a fast 1-bit chromatophore signal
    indicating whether it claims the incoming query.

    Parameters
    ----------
    arm_id : int
        Unique integer identifier for this arm.
    domain : str
        Human-readable domain name (e.g. ``"weather"``).
    n_agents : int
        Number of agents in the shared pool.
    functional_token : str
        String token used to identify this arm (e.g. ``"<octo_0>"``).
    lambda_decay : float
        Multiplicative decay rate for ``W_local``.
    """

    def __init__(
        self,
        arm_id: int,
        domain: str,
        n_agents: int,
        functional_token: str,
        lambda_decay: float = 0.05,
    ) -> None:
        self.arm_id = arm_id
        self.domain = domain
        self.functional_token = functional_token
        self.n_agents = n_agents
        self.lambda_decay = lambda_decay

        # Local affinity matrix; initialised with a soft bias toward the
        # arm's own domain agent (arm_id % n_agents), plus a small uniform
        # background so other agents remain reachable.
        self.W_local: np.ndarray = np.ones((n_agents, n_agents)) * 0.1
        preferred = arm_id % n_agents
        self.W_local[:, preferred] += 0.4  # stronger column for domain agent

        # Rolling confidence history (last 20 routing outcomes)
        self.confidence_history: deque[float] = deque(maxlen=20)

        # Domain keyword set for fast Jaccard lookup
        self._domain_keywords: set[str] = set(
            _DOMAIN_KEYWORDS.get(domain, [domain.lower()])
        )

    # ------------------------------------------------------------------
    # Chromatophore signal
    # ------------------------------------------------------------------

    def emit_chromatophore_signal(self, query: str) -> dict:
        """
        Emit a fast 1-bit domain-match signal for ``query``.

        The domain_match score is the Jaccard similarity between the
        set of lowercased query tokens and the arm's domain keyword set.

        Returns
        -------
        dict
            ``{"arm_id": int, "signal": 0|1, "domain_match": float}``
        """
        query_tokens = set(query.lower().split())
        # Strip punctuation from tokens
        query_tokens = {t.strip(".,?!;:\"'()[]") for t in query_tokens}

        intersection = query_tokens & self._domain_keywords
        union = query_tokens | self._domain_keywords

        domain_match = len(intersection) / len(union) if union else 0.0
        signal = 1 if domain_match > 0.5 else 0

        return {
            "arm_id": self.arm_id,
            "signal": signal,
            "domain_match": round(domain_match, 4),
            "functional_token": self.functional_token,
        }

    # ------------------------------------------------------------------
    # Local routing
    # ------------------------------------------------------------------

    def local_route(self, query: str) -> dict:
        """
        Route ``query`` to an agent using the arm's local affinity matrix.

        The agent with the highest column sum in ``W_local`` is chosen
        (same greedy argmax logic as APRR).

        Returns
        -------
        dict
            ``{"agent": int, "confidence": float, "latency_ms": float}``
        """
        t0 = time.perf_counter()

        col_scores = self.W_local.sum(axis=0)
        chosen_agent = int(np.argmax(col_scores))

        # Confidence: normalised score of chosen vs max-possible
        total = col_scores.sum()
        confidence = float(col_scores[chosen_agent] / total) if total > 0 else 1.0 / self.n_agents

        # Record confidence
        self.confidence_history.append(confidence)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "agent": chosen_agent,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 4),
        }

    # ------------------------------------------------------------------
    # Local W update (APRR rule)
    # ------------------------------------------------------------------

    def update_local(
        self,
        agent_to: int,
        success: bool,
        latency_ms: float,
    ) -> None:
        """
        Update ``W_local`` using the standard APRR delta rule.

        ``ΔW[arm_id_proxy, agent_to] += 𝟙[success] · (1/L²) · (1/latency_norm)``

        Parameters
        ----------
        agent_to : int
            Destination agent index.
        success : bool
            Whether the routed call succeeded.
        latency_ms : float
            Observed end-to-end latency in milliseconds.
        """
        # Use arm_id % n_agents as the "from" row to avoid index errors
        row = self.arm_id % self.n_agents

        if success:
            L = 1
            latency_norm = max(latency_ms, 1.0) / _LATENCY_NORM
            delta = (1.0 / (L ** 2)) * (1.0 / latency_norm)
            self.W_local[row, agent_to] += delta

        # Multiplicative decay
        self.W_local *= (1.0 - self.lambda_decay)
        np.clip(self.W_local, 0.0, None, out=self.W_local)

    # ------------------------------------------------------------------
    # Arm Specialisation Index (ASI) helper
    # ------------------------------------------------------------------

    def specialisation_index(self) -> float:
        """
        Compute the Arm Specialisation Index for the arm's W_local.

        ``ASI = 1 - entropy(row_distribution) / log(N)``

        where the row distribution is the column-normalised first row.

        Returns
        -------
        float
            ASI in [0, 1]. Higher → more specialised.
        """
        if self.n_agents <= 1:
            return 1.0
        row = self.W_local[self.arm_id % self.n_agents]
        total = row.sum()
        if total <= 0:
            return 0.0
        p = row / total
        # Entropy (base e); clip to avoid log(0)
        p_clipped = np.clip(p, 1e-12, None)
        entropy = -float(np.sum(p_clipped * np.log(p_clipped)))
        max_entropy = math.log(self.n_agents)
        return float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"OctoArm(id={self.arm_id}, domain='{self.domain}', "
            f"token='{self.functional_token}')"
        )


# ---------------------------------------------------------------------------
# OctoRouteRouter
# ---------------------------------------------------------------------------

class OctoRouteRouter:
    """
    Octopus-Inspired Distributed Router.

    Coordinates a set of ``OctoArm`` instances (peripheral intelligence)
    through a global affinity matrix (central brain).

    Routing is performed in two layers:

    1. **Arm dispatch** — each arm emits a chromatophore signal; the arm
       with the highest ``domain_match`` score is selected via the
       functional-token dispatch mechanism.
    2. **Agent selection** — the selected arm runs ``local_route`` to
       pick the final agent from its local affinity matrix.

    Parameters
    ----------
    agents : list[str]
        Human-readable agent names (length N).
    arm_domains : list[str]
        One domain string per arm (e.g. ``["weather", "market_prices", ...]``).
    lambda_decay : float
        Decay rate applied to both local and global affinity matrices.
    chromatophore_threshold : float
        Minimum ``domain_match`` score for an arm to be considered a
        valid candidate.  Falls back to global W if no arm qualifies.
    """

    def __init__(
        self,
        agents: list[str],
        arm_domains: list[str],
        lambda_decay: float = 0.05,
        chromatophore_threshold: float = 0.6,
    ) -> None:
        self.agents = agents
        self.n_agents = len(agents)
        self.n_arms = len(arm_domains)
        self.lambda_decay = lambda_decay
        self.chromatophore_threshold = chromatophore_threshold

        # Create one OctoArm per domain
        self.arms: list[OctoArm] = [
            OctoArm(
                arm_id=i,
                domain=domain,
                n_agents=self.n_agents,
                functional_token=f"<octo_{i}>",
                lambda_decay=lambda_decay,
            )
            for i, domain in enumerate(arm_domains)
        ]

        # Global coordinator matrix (brain): arms × agents
        self.W_global: np.ndarray = (
            np.ones((self.n_arms, self.n_agents)) / self.n_agents
        )

    # ------------------------------------------------------------------
    # Arm dispatch via functional tokens
    # ------------------------------------------------------------------

    def dispatch_via_functional_token(self, query: str) -> OctoArm:
        """
        Select the most domain-appropriate arm for ``query``.

        Step 1: Collect chromatophore signals from all arms (fast O(N·|V|)).
        Step 2: Select arm with highest ``domain_match``.
        Step 3: If best match < ``chromatophore_threshold``, fall back to
                the arm whose ``W_global`` row has the highest column sum
                (global brain routing).

        Parameters
        ----------
        query : str

        Returns
        -------
        OctoArm
            The selected arm.
        """
        signals = [arm.emit_chromatophore_signal(query) for arm in self.arms]
        best_idx = int(np.argmax([s["domain_match"] for s in signals]))
        best_match = signals[best_idx]["domain_match"]

        if best_match >= self.chromatophore_threshold:
            return self.arms[best_idx]

        # Fallback: use global W (brain routing)
        global_scores = self.W_global.sum(axis=1)
        fallback_idx = int(np.argmax(global_scores))
        return self.arms[fallback_idx]

    # ------------------------------------------------------------------
    # Main routing function
    # ------------------------------------------------------------------

    def route(self, query: str) -> dict:
        """
        Two-layer routing: arm dispatch → agent selection.

        Returns
        -------
        dict with keys:
            agent                : int
            agent_name           : str
            arm                  : int
            functional_token     : str
            latency_ms           : float
            chromatophore_signals: list[dict]
        """
        t0 = time.perf_counter()

        # Collect all signals (stored for transparency)
        chromatophore_signals = [
            arm.emit_chromatophore_signal(query) for arm in self.arms
        ]

        # Layer 1: dispatch
        selected_arm = self.dispatch_via_functional_token(query)

        # Layer 2: local routing within selected arm
        local_result = selected_arm.local_route(query)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "agent": local_result["agent"],
            "agent_name": self.agents[local_result["agent"]],
            "arm": selected_arm.arm_id,
            "functional_token": selected_arm.functional_token,
            "arm_domain": selected_arm.domain,
            "arm_confidence": local_result["confidence"],
            "latency_ms": round(latency_ms, 3),
            "chromatophore_signals": chromatophore_signals,
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        arm_id: int,
        agent_to: int,
        success: bool,
        latency_ms: float,
    ) -> None:
        """
        Update both the local arm W and the global W_global.

        Parameters
        ----------
        arm_id : int
            The arm that handled this routing decision.
        agent_to : int
            The agent that was selected.
        success : bool
        latency_ms : float
        """
        # Update arm's local W
        arm = self.arms[arm_id]
        arm.update_local(agent_to, success, latency_ms)

        # Update global W (brain)
        if success:
            L = 1
            latency_norm = max(latency_ms, 1.0) / _LATENCY_NORM
            delta = (1.0 / (L ** 2)) * (1.0 / latency_norm)
            self.W_global[arm_id, agent_to] += delta

        self.W_global *= (1.0 - self.lambda_decay)
        np.clip(self.W_global, 0.0, None, out=self.W_global)

    # ------------------------------------------------------------------
    # Arm Specialisation Index
    # ------------------------------------------------------------------

    def get_arm_specialization_index(self) -> dict:
        """
        Compute the Arm Specialisation Index (ASI) for every arm.

        ``ASI = 1 - H(row_distribution) / log(N)``

        where H is the Shannon entropy of the normalised W_local row.

        Returns
        -------
        dict
            ``{"arm_0": float, "arm_1": float, ...}``
        """
        result: dict[str, float] = {}
        for arm in self.arms:
            key = f"arm_{arm.arm_id}"
            result[key] = round(arm.specialisation_index(), 4)
        return result

    def __repr__(self) -> str:
        return (
            f"OctoRouteRouter(n_arms={self.n_arms}, n_agents={self.n_agents}, "
            f"threshold={self.chromatophore_threshold})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _make_synthetic_queries_octo(n: int = 50) -> list[tuple[str, str]]:
    """
    Return ``n`` (query, expected_domain) pairs spread across 5 domains.
    """
    templates: dict[str, list[str]] = {
        "weather": [
            "What is today's temperature in Lucknow?",
            "Will it rain in the next 48 hours in Maharashtra?",
            "Humidity levels this week for Rajasthan.",
            "Forecast wind speed for coastal Karnataka.",
            "Is there a storm warning for Gujarat?",
            "How cold will it get in Himachal Pradesh this week?",
            "Monsoon arrival date prediction for this year.",
            "Check cloud cover over Uttar Pradesh today.",
            "Alert me if temperature drops below 10°C in Punjab.",
            "What is the dew point in Tamil Nadu right now?",
        ],
        "market_prices": [
            "Current wholesale price of wheat in Hapur Mandi.",
            "MSP rate for paddy this kharif season.",
            "Tomato prices across Delhi, Mumbai, and Pune.",
            "How much is onion selling for in Lasalgaon today?",
            "Compare cotton rates across Gujarat mandis.",
            "What is the export price of basmati rice?",
            "Potato commodity rates in Agra mandi.",
            "Gold and silver rates in Indian market today.",
            "Diesel price for agricultural pump sets.",
            "Soybean futures rate on NCDEX.",
        ],
        "crop_advisory": [
            "Best crop to plant in black soil after monsoon.",
            "Which fertilizer is good for cotton at flowering stage?",
            "How to control leaf curl virus in chili crop?",
            "Irrigation schedule for drip-irrigated sugarcane.",
            "Seed rate per hectare for wheat cultivation.",
            "Pesticide recommendation for aphids in mustard.",
            "Top maize hybrid varieties for short-season planting.",
            "How to improve yield of sunflower in semi-arid regions?",
            "Post-harvest storage tips for onion to reduce losses.",
            "Inter-cropping options for groundnut fields.",
        ],
        "schemes": [
            "How to apply for PM-KISAN installment?",
            "Eligibility criteria for Pradhan Mantri Fasal Bima Yojana.",
            "Kisan Credit Card interest rate and limit.",
            "Solar pump subsidy scheme in Rajasthan.",
            "Documents needed for crop loan waiver.",
            "State-wise subsidy for drip irrigation installation.",
            "How to register for e-NAM trading portal?",
            "Women farmer special schemes in Tamil Nadu.",
            "Soil health card scheme application process.",
            "NABARD refinance scheme details for FPOs.",
        ],
        "soil_health": [
            "What does pH 5.5 mean for paddy cultivation?",
            "How to increase organic carbon in loamy soil?",
            "Potassium deficiency symptoms in tomato plants.",
            "Nitrogen fixing bacteria for black soil improvement.",
            "How to interpret soil test results from KVK lab?",
            "Best green manure crop for improving sandy soil.",
            "Micronutrient deficiency indicators in wheat.",
            "Steps to reduce soil compaction in irrigated fields.",
            "Soil moisture measurement methods for small farmers.",
            "Phosphorus availability at different pH levels.",
        ],
    }

    all_pairs: list[tuple[str, str]] = []
    for domain, qs in templates.items():
        for q in qs:
            all_pairs.append((q, domain))

    random.shuffle(all_pairs)
    return all_pairs[:n]


if __name__ == "__main__":
    print("=" * 70)
    print("OctoRoute Router — 50-Query Demo (5 Domains)")
    print("=" * 70)

    agents = [
        "WeatherAgent", "MarketAgent", "CropAgent",
        "SchemeAgent", "SoilAgent",
    ]
    arm_domains = ["weather", "market_prices", "crop_advisory", "schemes", "soil_health"]

    router = OctoRouteRouter(
        agents=agents,
        arm_domains=arm_domains,
        lambda_decay=0.05,
        chromatophore_threshold=0.05,   # low threshold for demo
    )

    queries = _make_synthetic_queries_octo(50)
    random.seed(42)

    domain_agent_map: dict[str, int] = {
        "weather": 0,
        "market_prices": 1,
        "crop_advisory": 2,
        "schemes": 3,
        "soil_health": 4,
    }

    correct = 0
    print(
        f"{'#':>3}  {'Arm':<14} {'Agent':<14} {'Token':<10} "
        f"{'Confidence':>10} {'Latency(ms)':>12} {'Hit?':>6}"
    )
    print("-" * 72)

    for idx, (query, expected_domain) in enumerate(queries, 1):
        result = router.route(query)
        expected_agent = domain_agent_map[expected_domain]
        hit = result["agent"] == expected_agent
        if hit:
            correct += 1

        sim_latency = random.uniform(30, 300)
        router.update(
            arm_id=result["arm"],
            agent_to=result["agent"],
            success=hit,
            latency_ms=sim_latency,
        )

        print(
            f"{idx:>3}  {result['arm_domain']:<14} {result['agent_name']:<14} "
            f"{result['functional_token']:<10} "
            f"{result['arm_confidence']:>10.4f} "
            f"{result['latency_ms']:>12.3f} "
            f"{'✓' if hit else '✗':>6}"
        )

    print(f"\nAccuracy: {correct}/{len(queries)} = {correct/len(queries):.1%}")
    print("\nArm Specialisation Index:")
    for arm_key, asi in router.get_arm_specialization_index().items():
        print(f"  {arm_key}: {asi:.4f}")
    print("\nGlobal W_global (brain matrix):")
    print(np.round(router.W_global, 4))
