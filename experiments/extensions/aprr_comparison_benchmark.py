"""
aprr_comparison_benchmark.py — APRR Extension Comparison Benchmark
====================================================================
Compares five routing strategies across synthetic ToolBench-style queries:

  1. APRR            — baseline W-matrix router (greedy argmax)
  2. CROWRouter      — Chain-of-Reasoning Over Workload
  3. OctoRouteRouter — Octopus-Inspired Distributed Routing
  4. CROW-OctoRoute  — Hybrid: OctoArm domain dispatch + CROW deliberation
  5. StaticSemantic  — random-choice baseline (no learning)

Evaluation protocol
-------------------
- 5 seeds × 20 warm-up iterations × 100 evaluation queries
- Synthetic query generator: 100 queries × 5 domains, each domain mapped to
  a "correct" agent; success = (routed_agent == correct_agent)

Metrics collected per router
----------------------------
- success_rate  : fraction of queries routed to the correct agent
- latency_ms    : mean simulated end-to-end latency
- avg_hops      : mean routing hops (always 1 for these single-hop routers)
- interpretability: qualitative label
- overhead      : qualitative label

Outputs
-------
- Formatted comparison table printed to stdout
- /home/user/workspace/aprr_extension_results.json

Dependencies: numpy, random, math, time, json, collections  (stdlib + numpy)
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from typing import Any, Optional

import numpy as np

# Re-use implementations from sibling files
import sys
import os

# Make sure sibling files are importable
sys.path.insert(0, os.path.dirname(__file__))
from crow_router import CROWRouter
from octoroute_router import OctoArm, OctoRouteRouter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENTS = ["WeatherAgent", "MarketAgent", "CropAgent", "SchemeAgent", "SoilAgent"]
N_AGENTS = len(AGENTS)
ARM_DOMAINS = ["weather", "market_prices", "crop_advisory", "schemes", "soil_health"]
LATENCY_NORM = 500.0

# Domain → correct agent index
DOMAIN_AGENT_MAP: dict[str, int] = {
    "weather": 0,
    "market_prices": 1,
    "crop_advisory": 2,
    "schemes": 3,
    "soil_health": 4,
}

# ---------------------------------------------------------------------------
# Synthetic query generator
# ---------------------------------------------------------------------------

_QUERY_TEMPLATES: dict[str, list[str]] = {
    "weather": [
        "What is the temperature today in {city}?",
        "Will it rain tomorrow in {city}?",
        "Humidity forecast for {city} this week.",
        "Is there a storm warning near {city}?",
        "Wind speed forecast for {city}.",
        "Expected temperature drop in {city} tonight.",
        "Monsoon prediction for {region} this year.",
        "Cloud cover report for {city}.",
        "Weather alert threshold for {city}.",
        "Dew point and heat index for {city}.",
        "Rainfall probability for next 3 days in {city}.",
        "Cold wave advisory for {region}.",
        "UV index for {city} tomorrow.",
        "Fog probability for {city} morning.",
        "Sunrise and sunset time for {city} today.",
        "Frost risk for {region} orchards tonight.",
        "Thunderstorm likelihood for {city} afternoon.",
        "Seasonal weather outlook for {region}.",
        "Hailstorm history for {region}.",
        "Is it safe to harvest today given the weather in {city}?",
    ],
    "market_prices": [
        "Current wholesale price of {crop} in {city} mandi.",
        "MSP for {crop} this season.",
        "Compare {crop} rates across three mandis.",
        "Export price of {crop} today.",
        "Commodity futures for {crop} on NCDEX.",
        "Retail vs wholesale difference for {crop} in {city}.",
        "Price trend of {crop} over last month.",
        "Best market to sell {crop} near {city}.",
        "Daily arrivals of {crop} at {city} mandi.",
        "Average price realisation for {crop} farmers.",
        "Agri input cost index for {region}.",
        "Diesel price for agriculture in {state}.",
        "Fertilizer bag price today.",
        "Seed cost per bag for {crop}.",
        "Irrigation equipment prices in {city}.",
        "Cold storage charges for {crop} in {region}.",
        "Transport cost from {city} mandi to nearby city.",
        "Processor buying price for {crop}.",
        "Commodity insurance premium for {crop}.",
        "Gold rate as collateral benchmark today.",
    ],
    "crop_advisory": [
        "Best variety of {crop} for {region} climate.",
        "Fertilizer schedule for {crop} at jointing stage.",
        "Pest management for {crop} during monsoon.",
        "Irrigation frequency for {crop} in drip system.",
        "Seed rate per hectare for {crop}.",
        "Disease-resistant {crop} varieties available.",
        "How to manage waterlogging in {crop} field?",
        "Post-harvest handling tips for {crop}.",
        "Inter-crop suitable with {crop} in {region}.",
        "Recommended herbicide for weed control in {crop}.",
        "Micronutrient spray schedule for {crop}.",
        "Gap-filling techniques for poor germination in {crop}.",
        "Thinning and staking advice for {crop}.",
        "Biostimulant use for {crop} yield improvement.",
        "Crop residue management after {crop} harvest.",
        "Nursery preparation for {crop} transplanting.",
        "Ratooning potential of {crop} in {region}.",
        "Drone-based spraying dosage for {crop}.",
        "Growth regulator application timing for {crop}.",
        "Nitrogen toxicity symptoms in {crop}.",
    ],
    "schemes": [
        "How to register for PM-KISAN scheme?",
        "PMFBY eligibility criteria for {crop} in {state}.",
        "Kisan Credit Card loan limit for small farmers.",
        "Solar pump subsidy available in {state}.",
        "State government crop loan waiver process.",
        "NABARD scheme for FPO registration.",
        "e-NAM portal onboarding steps.",
        "Soil health card renewal procedure.",
        "Women farmer special benefit schemes in {state}.",
        "Crop insurance claim filing procedure.",
        "Agri infrastructure fund for cold storage.",
        "PM-KUSUM component B eligibility.",
        "DBT transfer status check for farmer subsidy.",
        "Rashtriya Krishi Vikas Yojana application.",
        "MIDH scheme for horticulture in {state}.",
        "National mission for oilseed and oil palm support.",
        "Farmer registration on AgriStack.",
        "Pradhan Mantri Krishi Sinchai Yojana details.",
        "Assistance under Per Drop More Crop scheme.",
        "Scheme for organic certification cost reimbursement.",
    ],
    "soil_health": [
        "Soil pH interpretation for {crop} cultivation.",
        "How to increase organic matter in {region} soil?",
        "Potassium deficiency visual symptoms in {crop}.",
        "Nitrogen fixation bacteria for {region} black soil.",
        "Soil texture classification for field in {region}.",
        "Phosphorus availability at different pH values.",
        "Steps to reduce soil compaction in {region} fields.",
        "Micronutrient test package from KVK lab.",
        "Green manure options for improving {region} soil.",
        "Soil moisture retention methods for clay soil.",
        "Cation exchange capacity meaning for farmers.",
        "Boron and zinc deficiency treatment for {crop}.",
        "Benefits of vermicompost in {region} soil.",
        "How to interpret soil health card numbers?",
        "Carbon sequestration farming practices for {region}.",
        "Salinity remediation steps for {region} farmland.",
        "Soil aggregate stability test significance.",
        "Acidic soil correction using lime application.",
        "Residual herbicide impact on soil biology.",
        "Biochar application benefits for sandy {region} soil.",
    ],
}

_CITIES = ["Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur", "Bhopal", "Nagpur", "Hyderabad"]
_REGIONS = ["North India", "South India", "Central India", "Western India", "Eastern India"]
_STATES = ["Maharashtra", "Rajasthan", "Punjab", "Tamil Nadu", "Uttar Pradesh"]
_CROPS = ["wheat", "paddy", "cotton", "onion", "tomato", "soybean", "maize", "sugarcane"]


def generate_synthetic_queries(n: int = 100, seed: int = 0) -> list[tuple[str, str]]:
    """
    Generate ``n`` (query_text, correct_domain) pairs from templates.

    Parameters
    ----------
    n : int
        Number of queries to generate.
    seed : int
        Random seed.

    Returns
    -------
    list of (query_str, domain_str)
    """
    rng = random.Random(seed)
    domains = list(_QUERY_TEMPLATES.keys())
    per_domain = n // len(domains)
    remainder = n % len(domains)

    pairs: list[tuple[str, str]] = []

    for i, domain in enumerate(domains):
        count = per_domain + (1 if i < remainder else 0)
        templates = _QUERY_TEMPLATES[domain]
        for _ in range(count):
            tmpl = rng.choice(templates)
            query = tmpl.format(
                city=rng.choice(_CITIES),
                region=rng.choice(_REGIONS),
                state=rng.choice(_STATES),
                crop=rng.choice(_CROPS),
            )
            pairs.append((query, domain))

    rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# APRR stub (self-contained, no external import needed)
# ---------------------------------------------------------------------------

# Domain keyword map for source-agent inference (mirrors OctoRoute)
_APRR_DOMAIN_KWS: dict[str, list[str]] = {
    "weather":       ["weather", "rain", "temperature", "forecast", "humidity", "wind", "monsoon"],
    "market_prices": ["price", "market", "mandi", "msp", "commodity", "rate", "cost"],
    "crop_advisory": ["crop", "seed", "fertilizer", "pest", "irrigation", "harvest", "variety"],
    "schemes":       ["scheme", "subsidy", "loan", "insurance", "kisan", "credit", "apply"],
    "soil_health":   ["soil", "ph", "nutrient", "nitrogen", "organic", "moisture", "carbon"],
}
_APRR_DOMAIN_ORDER = ["weather", "market_prices", "crop_advisory", "schemes", "soil_health"]


def _infer_source_agent(query: str, n_agents: int) -> int:
    """Infer source row index in W from query domain keywords."""
    q = query.lower()
    best_idx = 0
    best_hits = 0
    for idx, domain in enumerate(_APRR_DOMAIN_ORDER[:n_agents]):
        hits = sum(1 for kw in _APRR_DOMAIN_KWS.get(domain, []) if kw in q)
        if hits > best_hits:
            best_hits = hits
            best_idx = idx
    return best_idx


class APRRRouter:
    """
    Minimal APRR implementation for benchmarking.

    Implements the standard APRR update rule:
        ΔW[i,j] += 𝟙[success] · (1/L²) · (1/latency_norm)
        W       *= (1 − λ)

    Routing selects the row corresponding to the inferred source domain
    (via keyword match), then takes argmax across that row.
    """

    _LATENCY_NORM = 500.0

    def __init__(
        self,
        agents: list[str],
        lambda_decay: float = 0.05,
        W_init: Optional[np.ndarray] = None,
    ) -> None:
        self.agents = agents
        self.n_agents = len(agents)
        self.lambda_decay = lambda_decay
        if W_init is not None:
            self.W = W_init.astype(float).copy()
        else:
            self.W = np.eye(self.n_agents) * 0.5

    def route(self, query: str) -> dict:
        t0 = time.perf_counter()
        src = _infer_source_agent(query, self.n_agents)
        row_scores = self.W[src]
        chosen = int(np.argmax(row_scores))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "agent": chosen,
            "agent_name": self.agents[chosen],
            "latency_ms": round(latency_ms, 3),
            "mode": "greedy",
            "source_row": src,
        }

    def update(
        self,
        agent_from: int,
        agent_to: int,
        success: bool,
        latency_ms: float,
    ) -> None:
        if success:
            L = 1
            lat_norm = max(latency_ms, 1.0) / self._LATENCY_NORM
            delta = (1.0 / (L ** 2)) * (1.0 / lat_norm)
            self.W[agent_from, agent_to] += delta
        self.W *= (1.0 - self.lambda_decay)
        np.clip(self.W, 0.0, None, out=self.W)


# ---------------------------------------------------------------------------
# StaticSemantic (random baseline)
# ---------------------------------------------------------------------------

class StaticSemanticRouter:
    """Random-choice router — no learning, no reasoning."""

    def __init__(self, agents: list[str], seed: int = 0) -> None:
        self.agents = agents
        self._rng = random.Random(seed)

    def route(self, query: str) -> dict:
        t0 = time.perf_counter()
        chosen = self._rng.randint(0, len(self.agents) - 1)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "agent": chosen,
            "agent_name": self.agents[chosen],
            "latency_ms": round(latency_ms, 3),
            "mode": "static",
        }

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass  # No learning


# ---------------------------------------------------------------------------
# CROW-OctoRoute Hybrid
# ---------------------------------------------------------------------------

class CROWOctoRouteHybrid:
    """
    Hybrid router: OctoArm domain dispatch (Layer 1) + CROW deliberation
    within the selected arm (Layer 2).

    An OctoRouteRouter selects the arm.  The arm's W_local is used to
    seed the top-k candidates, which are then evaluated by a CROWRouter
    instance for final deliberation.
    """

    def __init__(
        self,
        agents: list[str],
        arm_domains: list[str],
        lambda_decay: float = 0.05,
        thought_budget_threshold: float = 0.4,
        chromatophore_threshold: float = 0.05,
    ) -> None:
        self.agents = agents
        self.n_agents = len(agents)
        self.arm_domains = arm_domains

        self._octo = OctoRouteRouter(
            agents=agents,
            arm_domains=arm_domains,
            lambda_decay=lambda_decay,
            chromatophore_threshold=chromatophore_threshold,
        )
        self._crow = CROWRouter(
            agents=agents,
            lambda_decay=lambda_decay,
            thought_budget_threshold=thought_budget_threshold,
            max_deliberation_rounds=2,
        )

    def route(self, query: str) -> dict:
        t0 = time.perf_counter()

        # Layer 1: OctoArm dispatch
        selected_arm = self._octo.dispatch_via_functional_token(query)
        chroma_signals = [
            arm.emit_chromatophore_signal(query) for arm in self._octo.arms
        ]

        # Temporarily override CROWRouter's W with arm's local W
        saved_W = self._crow.W.copy()
        self._crow.W = selected_arm.W_local.copy()

        # Layer 2: CROW routing with deliberation
        crow_result = self._crow.route(query)

        # Restore CROW's W
        self._crow.W = saved_W

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "agent": crow_result["agent"],
            "agent_name": self.agents[crow_result["agent"]],
            "arm": selected_arm.arm_id,
            "arm_domain": selected_arm.domain,
            "functional_token": selected_arm.functional_token,
            "latency_ms": round(latency_ms, 3),
            "mode": crow_result["mode"],
            "chromatophore_signals": chroma_signals,
            "crow_reasoning": crow_result.get("reasoning", {}),
        }

    def update(
        self,
        arm_id: int,
        agent_from: int,
        agent_to: int,
        success: bool,
        latency_ms: float,
        reasoning_quality: float = 0.5,
    ) -> None:
        self._octo.update(arm_id, agent_to, success, latency_ms)
        self._crow.update(agent_from, agent_to, success, latency_ms, reasoning_quality)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _simulate_latency(rng: random.Random) -> float:
    """Simulate a realistic latency in ms."""
    return rng.uniform(20, 600)


def _domain_biased_W(n_agents: int, bias: float = 0.4) -> np.ndarray:
    """
    Create an N×N W matrix with a domain-biased diagonal.

    Each agent ``i`` is the "home" agent for domain ``i``.  The diagonal
    entry ``W[i, i]`` gets a +``bias`` boost so each agent has a higher
    incoming-column score than competing agents, creating a weak prior
    towards correct routing without requiring extensive warm-up.
    """
    W = np.ones((n_agents, n_agents)) * 0.1
    W += np.eye(n_agents) * bias   # diagonal gets extra bias
    return W


def run_benchmark(
    n_seeds: int = 5,
    n_warmup: int = 20,
    n_eval: int = 100,
) -> dict:
    """
    Run the full benchmark.

    Parameters
    ----------
    n_seeds : int
        Number of independent random seeds.
    n_warmup : int
        Warm-up iterations (router learns but results not counted).
    n_eval : int
        Evaluation queries per seed.

    Returns
    -------
    dict
        Full results keyed by router name.
    """

    router_names = ["APRR", "CROW", "OctoRoute", "CROW-OctoRoute", "StaticSemantic"]
    accumulators: dict[str, dict[str, list[float]]] = {
        name: {"success": [], "latency_ms": [], "hops": []}
        for name in router_names
    }

    for seed in range(n_seeds):
        rng = random.Random(seed * 137 + 42)
        np.random.seed(seed)

        # Domain-biased W init (gives all W-matrix routers same starting prior
        # as OctoRoute arms, enabling fair comparison)
        W_init = _domain_biased_W(N_AGENTS, bias=0.4)

        # Instantiate routers fresh each seed
        routers: dict[str, Any] = {
            "APRR": APRRRouter(agents=AGENTS, lambda_decay=0.05, W_init=W_init.copy()),
            "CROW": CROWRouter(
                agents=AGENTS,
                W_init=W_init.copy(),
                lambda_decay=0.05,
                thought_budget_threshold=0.4,
                max_deliberation_rounds=3,
            ),
            "OctoRoute": OctoRouteRouter(
                agents=AGENTS,
                arm_domains=ARM_DOMAINS,
                lambda_decay=0.05,
                chromatophore_threshold=0.05,
            ),
            "CROW-OctoRoute": CROWOctoRouteHybrid(
                agents=AGENTS,
                arm_domains=ARM_DOMAINS,
                lambda_decay=0.05,
                thought_budget_threshold=0.4,
                chromatophore_threshold=0.05,
            ),
            "StaticSemantic": StaticSemanticRouter(agents=AGENTS, seed=seed),
        }

        # --- Warm-up phase
        warmup_queries = generate_synthetic_queries(n=n_warmup * 5, seed=seed)
        warmup_queries = warmup_queries[: n_warmup]

        for q_text, domain in warmup_queries:
            correct_agent = DOMAIN_AGENT_MAP[domain]
            sim_lat = _simulate_latency(rng)

            # APRR
            r = routers["APRR"].route(q_text)
            success = r["agent"] == correct_agent
            routers["APRR"].update(r["agent"], r["agent"], success, sim_lat)

            # CROW
            r = routers["CROW"].route(q_text)
            success = r["agent"] == correct_agent
            rq = routers["CROW"].score_reasoning_quality(r.get("reasoning", {"confidence": 0.5, "reasoning_steps": 1}))
            routers["CROW"].update(r["agent"], r["agent"], success, sim_lat, rq)

            # OctoRoute
            r = routers["OctoRoute"].route(q_text)
            success = r["agent"] == correct_agent
            routers["OctoRoute"].update(r["arm"], r["agent"], success, sim_lat)

            # CROW-OctoRoute
            r = routers["CROW-OctoRoute"].route(q_text)
            success = r["agent"] == correct_agent
            rq = 0.5
            routers["CROW-OctoRoute"].update(
                r.get("arm", 0), r["agent"], r["agent"], success, sim_lat, rq
            )

            # StaticSemantic
            r = routers["StaticSemantic"].route(q_text)

        # --- Evaluation phase
        eval_queries = generate_synthetic_queries(n=n_eval * 2, seed=seed + 1000)
        eval_queries = eval_queries[:n_eval]

        # Per-router base latency offsets (ms) reflecting architectural overhead
        ROUTER_BASE_LATENCY = {
            "APRR":           rng.uniform(200, 320),   # simple W-matrix lookup
            "CROW":           rng.uniform(380, 520),   # CoT trace + possible deliberation
            "OctoRoute":      rng.uniform(150, 260),   # fast functional-token dispatch
            "CROW-OctoRoute": rng.uniform(420, 600),   # arm dispatch + CROW deliberation
            "StaticSemantic": rng.uniform(350, 450),   # random but with semantic overhead
        }

        for q_text, domain in eval_queries:
            correct_agent = DOMAIN_AGENT_MAP[domain]

            for name in router_names:
                router = routers[name]
                t0 = time.perf_counter()
                r = router.route(q_text)
                route_cpu_ms = (time.perf_counter() - t0) * 1000.0
                # Simulated end-to-end latency = base network latency + router overhead
                sim_lat = ROUTER_BASE_LATENCY[name] + route_cpu_ms * 50
                success = int(r["agent"] == correct_agent)

                accumulators[name]["success"].append(success)
                accumulators[name]["latency_ms"].append(sim_lat)
                # hops: CROW deliberation adds hops for complex queries
                hops = r.get("deliberation_rounds", 1) if name in ("CROW", "CROW-OctoRoute") else 1
                accumulators[name]["hops"].append(hops)

                # Apply learning update
                if name == "APRR":
                    router.update(r["agent"], r["agent"], bool(success), sim_lat)
                elif name == "CROW":
                    rq = router.score_reasoning_quality(
                        r.get("reasoning", {"confidence": 0.5, "reasoning_steps": 1})
                    )
                    router.update(r["agent"], r["agent"], bool(success), sim_lat, rq)
                elif name == "OctoRoute":
                    router.update(r.get("arm", 0), r["agent"], bool(success), sim_lat)
                elif name == "CROW-OctoRoute":
                    router.update(
                        r.get("arm", 0), r["agent"], r["agent"],
                        bool(success), sim_lat, 0.5
                    )
                # StaticSemantic: no update

    # --- Aggregate results
    qualitative: dict[str, dict[str, str]] = {
        "APRR":           {"interpretability": "Low",  "overhead": "Low"},
        "CROW":           {"interpretability": "High", "overhead": "Med"},
        "OctoRoute":      {"interpretability": "Med",  "overhead": "Low"},
        "CROW-OctoRoute": {"interpretability": "High", "overhead": "High"},
        "StaticSemantic": {"interpretability": "None", "overhead": "None"},
    }

    results: dict[str, Any] = {}
    for name in router_names:
        acc = accumulators[name]
        n = len(acc["success"])
        results[name] = {
            "success_rate": round(sum(acc["success"]) / n, 4) if n > 0 else 0.0,
            "avg_latency_ms": round(sum(acc["latency_ms"]) / n, 2) if n > 0 else 0.0,
            "avg_hops": round(sum(acc["hops"]) / n, 2) if n > 0 else 0.0,
            "n_queries": n,
            **qualitative[name],
        }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Router':<20} | {'Success':>7} | {'Latency(ms)':>11} | "
        f"{'Hops':>4} | {'Interp.':>8} | {'Overhead':>8}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    order = ["APRR", "CROW", "OctoRoute", "CROW-OctoRoute", "StaticSemantic"]
    for name in order:
        r = results[name]
        print(
            f"{name:<20} | {r['success_rate']:>7.3f} | "
            f"{r['avg_latency_ms']:>11.1f} | "
            f"{r['avg_hops']:>4.1f} | "
            f"{r['interpretability']:>8} | "
            f"{r['overhead']:>8}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("APRR Extension Benchmark")
    print("Routers: APRR, CROW, OctoRoute, CROW-OctoRoute, StaticSemantic")
    print("Protocol: 5 seeds × 20 warm-up × 100 eval queries")
    print("=" * 70)

    t_start = time.perf_counter()
    results = run_benchmark(n_seeds=5, n_warmup=20, n_eval=100)
    elapsed = time.perf_counter() - t_start

    print_comparison_table(results)
    print(f"Benchmark completed in {elapsed:.2f}s")

    # Save JSON
    out_path = "/home/user/workspace/aprr_extension_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {out_path}")
