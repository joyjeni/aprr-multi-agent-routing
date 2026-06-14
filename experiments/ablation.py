"""Ablation study over (alpha, beta, rho) and query-fusion weight."""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np

from aprr.agents import AgentTopology
from aprr.router import APRRRouter, RouterConfig
from aprr.toolbench import ToolBenchSimulator


def run_one(cfg: RouterConfig, n_queries=600, n_iterations=25, seed=42):
    topo = AgentTopology.default(seed=seed)
    sim = ToolBenchSimulator(topo, seed=seed); queries = sim.generate(n_queries)
    router = APRRRouter(topo, cfg)
    successes, latencies = [], []
    for it in range(n_iterations):
        for q in queries:
            path = router.route(query_emb=q.embedding, max_hops=8)
            s, l = sim.rollout(path, q.gt_path)
            router.update_trail(path, s, l)
            if it >= n_iterations - 3:   # last 3 iterations only
                successes.append(int(s)); latencies.append(l)
    return float(np.mean(successes)), float(np.mean(latencies))


def grid(out_path="results/ablation.json"):
    grid = {
        "alpha": [0.5, 1.0, 1.5, 2.0],
        "beta":  [1.0, 2.0, 2.5, 3.0],
        "lam":   [0.05, 0.10, 0.20, 0.40],
        "gamma": [0.0, 0.5, 1.0, 1.5],
    }
    records = []
    for k, values in grid.items():
        for v in values:
            cfg = RouterConfig(**{k: v})
            sr, lat = run_one(cfg)
            records.append({"param": k, "value": v, "success_rate": sr, "latency_ms": lat})
            print(f"{k}={v}  succ={sr:.3f}  lat={lat:.1f}ms")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(records, indent=2))
    print(f"✔ {out_path}")


if __name__ == "__main__":
    grid()
