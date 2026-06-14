"""
Experiment driver — runs all routers on the ToolBench simulator and emits
JSON / CSV artifacts consumed by the figure-generation script and the
Vercel dashboard.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .agents import AgentTopology
from .baselines import (LLMRouter, OracleRouter, RandomRouter,
                        RoundRobinRouter, StaticSemanticRouter)
from .metrics import EpisodeResult, aggregate_metrics
from .router import APRRRouter, RouterConfig
from .toolbench import ToolBenchSimulator


def build_routers(topology: AgentTopology, sim: ToolBenchSimulator,
                  seeds: Dict[str, int] | None = None) -> Dict[str, object]:
    seeds = seeds or {}
    return {
        "Random":         RandomRouter(topology, seed=seeds.get("Random", 1)),
        "RoundRobin":     RoundRobinRouter(topology, seed=seeds.get("RoundRobin", 2)),
        "StaticSemantic": StaticSemanticRouter(topology, seed=seeds.get("StaticSemantic", 3)),
        "LLMRouter":      LLMRouter(topology, seed=seeds.get("LLMRouter", 4), lr=0.05),
        "APRR":           APRRRouter(topology, RouterConfig(seed=seeds.get("APRR", 5))),
        "Oracle":         OracleRouter(topology, ground_truth_fn=sim.gt_path_of,
                                       seed=seeds.get("Oracle", 6)),
    }


def run_experiment(
    n_queries: int = 1500,
    n_iterations: int = 50,
    seed: int = 42,
    out_dir: str | Path = "results",
    save_W_every: int = 5,
) -> Dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    topo = AgentTopology.default(seed=seed)
    sim = ToolBenchSimulator(topo, seed=seed)
    queries = sim.generate(n_queries)
    routers = build_routers(topo, sim, seeds={k: seed + i for i, k in enumerate(
        ["Random","RoundRobin","StaticSemantic","LLMRouter","APRR","Oracle"])})

    results: List[EpisodeResult] = []
    convergence: Dict[str, List[Dict]] = {k: [] for k in routers}
    W_snaps: List[Dict] = []
    t0 = time.time()

    for it in range(n_iterations):
        idx_perm = np.random.default_rng(seed + it).permutation(len(queries))
        per_router_succ: Dict[str, List[int]] = {k: [] for k in routers}
        per_router_lat:  Dict[str, List[float]] = {k: [] for k in routers}

        for q_i in idx_perm:
            q = queries[q_i]
            for name, router in routers.items():
                if name == "Oracle":
                    router.set_query(q.qid)
                path = router.route(query_emb=q.embedding, start=0,
                                    require_terminal=True, max_hops=8)
                success, latency = sim.rollout(path, q.gt_path)
                router.update_trail(path, success, latency)

                results.append(EpisodeResult(
                    router=name, query_id=int(q.qid), split=q.split,
                    path=list(map(int, path)),
                    success=bool(success), latency_ms=float(latency),
                    hops=len(path), iteration=it,
                ))
                per_router_succ[name].append(int(success))
                per_router_lat[name].append(latency)

        for name in routers:
            convergence[name].append({
                "iteration": it,
                "success_rate": float(np.mean(per_router_succ[name])),
                "mean_latency_ms": float(np.mean(per_router_lat[name])),
                "mean_hops": float(np.mean([len(r.path) for r in results
                                            if r.router == name and r.iteration == it])),
            })

        if it % save_W_every == 0 or it == n_iterations - 1:
            snap = routers["APRR"].snapshot()
            W_snaps.append({"iteration": it, "W": snap.tolist()})
        if it % 10 == 0:
            elapsed = time.time() - t0
            print(f"[iter {it:>3}/{n_iterations}]  elapsed={elapsed:6.1f}s  "
                  f"APRR succ={convergence['APRR'][-1]['success_rate']:.3f}  "
                  f"lat={convergence['APRR'][-1]['mean_latency_ms']:.1f}ms")

    elapsed = time.time() - t0
    agg = aggregate_metrics(results)

    artifact = {
        "config": {
            "n_queries": n_queries,
            "n_iterations": n_iterations,
            "seed": seed,
            "elapsed_sec": elapsed,
            "topology": {"n_agents": topo.n,
                          "roles": [a.role for a in topo.agents]},
        },
        "aggregate": agg,
        "convergence": convergence,
        "W_snapshots": W_snaps,
    }
    (out / "results.json").write_text(json.dumps(artifact, indent=2))
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(asdict(r)) for r in results)
    )
    print(f"\n✔ wrote {out/'results.json'}  ({elapsed:.1f}s, "
          f"{len(results)} episodes)")
    return artifact


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_queries", type=int, default=1500)
    p.add_argument("--n_iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results")
    a = p.parse_args()
    run_experiment(a.n_queries, a.n_iterations, a.seed, a.out_dir)
