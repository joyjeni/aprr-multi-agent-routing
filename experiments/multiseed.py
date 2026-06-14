"""Multi-seed evaluation for the main results table (5 seeds → 95% CI)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aprr.runner import run_experiment


def main(n_queries=500, n_iterations=40, seeds=(42, 43, 44, 45, 46),
         out_path="results/multiseed.json"):
    Path("results").mkdir(exist_ok=True)
    per_seed_agg: List[Dict] = []
    per_seed_conv: Dict[int, Dict] = {}
    for s in seeds:
        print(f"\n========= SEED {s} =========")
        art = run_experiment(n_queries=n_queries, n_iterations=n_iterations,
                              seed=s, out_dir=f"results/seed_{s}",
                              save_W_every=10)
        per_seed_agg.append(art["aggregate"])
        per_seed_conv[s] = art["convergence"]

    # combine
    routers = list(per_seed_agg[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for r in routers:
        vals = defaultdict(list)
        for agg in per_seed_agg:
            for k, v in agg[r].items():
                vals[k].append(v)
        summary[r] = {
            k: {"mean": float(np.mean(vs)), "std": float(np.std(vs, ddof=1)),
                "ci95": float(1.96 * np.std(vs, ddof=1) / np.sqrt(len(vs)))}
            for k, vs in vals.items()
        }

    # convergence: avg across seeds per iteration
    conv_mean: Dict[str, List[Dict]] = {}
    n_iter = n_iterations
    for r in routers:
        rows = []
        for it in range(n_iter):
            sr = np.mean([per_seed_conv[s][r][it]["success_rate"] for s in seeds])
            la = np.mean([per_seed_conv[s][r][it]["mean_latency_ms"] for s in seeds])
            ho = np.mean([per_seed_conv[s][r][it]["mean_hops"] for s in seeds])
            sr_std = np.std([per_seed_conv[s][r][it]["success_rate"] for s in seeds], ddof=1)
            la_std = np.std([per_seed_conv[s][r][it]["mean_latency_ms"] for s in seeds], ddof=1)
            rows.append({"iteration": it, "success_rate": float(sr),
                         "mean_latency_ms": float(la), "mean_hops": float(ho),
                         "success_rate_std": float(sr_std),
                         "mean_latency_ms_std": float(la_std)})
        conv_mean[r] = rows

    Path(out_path).write_text(json.dumps({
        "seeds": list(seeds),
        "n_queries": n_queries,
        "n_iterations": n_iterations,
        "summary": summary,
        "convergence_mean": conv_mean,
    }, indent=2))
    print(f"\n✔ wrote {out_path}")
    print("\nMain results (mean ± 95% CI over seeds):")
    for r in routers:
        sr = summary[r]["success_rate"]
        la = summary[r]["mean_latency_ms"]
        ho = summary[r]["mean_hops"]
        print(f"  {r:16s}  succ={sr['mean']:.3f} ± {sr['ci95']:.3f}   "
              f"lat={la['mean']:6.1f} ± {la['ci95']:.1f}ms   "
              f"hops={ho['mean']:.2f} ± {ho['ci95']:.2f}")


if __name__ == "__main__":
    main()
