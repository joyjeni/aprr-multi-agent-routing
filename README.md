# APRR: Adaptive Probabilistic Routing Reinforcement for Multi-Agent LLM Systems

> Online policy-iteration routing for multi-agent LLM pipelines with **35.7% latency reduction** and **23.9% hop reduction** over semantic baselines, Pareto-optimal on the latency–success frontier.

[![Live Dashboard](https://img.shields.io/badge/Dashboard-aprr.vercel.app-d93025)](https://aprr-multi-agent-routing.vercel.app)
[![Colab](https://img.shields.io/badge/Colab-Reproduce-orange)](https://colab.research.google.com/github/joyjeni/aprr-multi-agent-routing/blob/main/notebooks/APRR_Reproducible_Benchmark.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Author:** Jenisha T, MS Ramaiah University of Applied Sciences — PhD candidate, Computer Science Engineering.

---

## Highlights

- **Algorithm.** APRR is a decay-regularized, online policy-iteration router over a graph of LLM agents. A routing-affinity matrix **W** is updated by success-weighted, cost-quadratic reinforcement (ΔW ∝ 𝟙[success] · 1/L² · 1/latency_norm) with multiplicative decay (1 − λ).
- **Theory.** Proposition 1 establishes a REINFORCE-equivalence under stated conditions. Theorem 1 bounds the W matrix and proves geometric decay of unrewarded edges.
- **Empirics.** 5 seeds × 40 iterations × 500 ToolBench queries (G1/G2/G3). APRR is Pareto-optimal vs. Random, RoundRobin, StaticSemantic, and an LLM-Router baseline.
- **Reproducibility.** Single-command reproduction; free-tier T4 Colab notebook; JSON artifacts shipped to dashboard.

## Headline Results (5-seed mean ± 95% CI)

| Router | Success | Latency (ms) | Hops |
|---|---|---|---|
| Random | 0.323 ± 0.004 | 393.9 | 3.50 |
| RoundRobin | 0.380 ± 0.002 | 546.7 | 4.33 |
| StaticSemantic | 0.499 ± 0.024 | 406.6 | 3.64 |
| LLMRouter | 0.406 ± 0.010 | 151.0 | 2.07 |
| **APRR (ours)** | **0.470 ± 0.020** | **261.3 ± 31.7** | **2.77** |
| Oracle (upper bound) | 0.900 ± 0.001 | 448.9 | 4.30 |

## Project layout

```
aprr-multi-agent-routing/
├── src/aprr/              # Algorithm + 5 baselines + ToolBench harness
├── experiments/           # multiseed, ablation, figure generators
├── results/               # multiseed.json, ablation.json, per-seed JSON
├── figures/               # fig1–fig7 (PDF + PNG)
├── tables/                # table1_main_results.{csv,tex}
├── paper/                 # IEEE manuscript (main.tex, main.pdf) + peer-review pack
├── notebooks/             # APRR_Reproducible_Benchmark.ipynb
├── vercel-app/            # Next.js 14 dashboard
└── tests/                 # pytest suite (4/4 pass)
```

## Reproduce in ~10 minutes (local CPU)

```bash
git clone https://github.com/joyjeni/aprr-multi-agent-routing.git
cd aprr-multi-agent-routing
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src

python experiments/multiseed.py          # 5 seeds × 40 iters × 500 queries
python experiments/make_figures.py       # fig1–fig5
python experiments/ablation.py           # α, β, λ, γ ablation grid
python experiments/make_ablation_figure.py
python -m pytest tests/ -v               # 4/4 tests
```

## Reproduce on Colab / Kaggle (free T4)

Open `notebooks/APRR_Reproducible_Benchmark.ipynb` in Colab. Runs end-to-end on free T4 in ~12 minutes including an optional Gemma-2-2B-it real-LLM case study.

## Live dashboard

Interactive results, charts, and downloadable artifacts at
**https://aprr-multi-agent-routing.vercel.app** (API: `/api/results`).

## Manuscript

- `paper/main.pdf` — 8-page IEEE TNNLS-style submission.
- `paper/manuscript.md` — readable Markdown mirror.
- `paper/review_harvard.md`, `paper/review_mit.md` — simulated peer reviews.
- `paper/author_response.md` — point-by-point response and revision plan.
- `paper/peer_review_package.pdf` — bundled review report.

## Citation

```bibtex
@article{jenisha2026aprr,
  title={APRR: Adaptive Probabilistic Routing Reinforcement for Multi-Agent LLM Systems},
  author={Jenisha, T.},
  journal={IEEE Transactions on Neural Networks and Learning Systems (under review)},
  year={2026},
  note={Code: https://github.com/joyjeni/aprr-multi-agent-routing}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

## Objective 2 Extension: CROW and OctoRoute Routing Variants

> **New (June 2026)** — Two novel routing paradigms benchmarked against APRR.

### CROW — Chain-of-Reasoning Over Workload

CROW augments APRR with a Chain-of-Thought deliberation layer. Before routing, a complexity scorer gates whether the query needs greedy dispatch (low complexity) or multi-agent CoT deliberation (high complexity). The routing affinity update is weighted by *reasoning quality* ρ(T_q) ∈ [0,1]:

```
ΔW[i,j] += 𝟙[success] · (1/L²) · (1/latency_norm) · (1 + β·ρ(T_q))
```

**Key properties:** interpretable trace per routing decision; deliberation round budget θ_q; penalises low-confidence traces in the W update.

### OctoRoute — Octopus-Inspired Distributed Routing

OctoRoute replaces the centralised W matrix with a two-layer architecture inspired by the octopus nervous system (2/3 of octopus neurons are in the arms — decentralised intelligence):

- **Layer 1 — Functional token dispatch** (`<octo_0>`…`<octo_N-1>`): central coordinator selects an arm via 1-bit chromatophore signals (Jaccard domain-match), reducing context by ~80%.
- **Layer 2 — Arm-local W_local routing**: each arm maintains its own affinity matrix over the shared agent pool, updated independently.

### Benchmark Results (5 seeds × 20 warm-up × 100 eval queries)

| Router | Success | Latency (ms) | Interpretability | Best For |
|---|---|---|---|---|
| APRR (baseline) | 0.660 | 268.7 | Low | General-purpose |
| **CROW** | 0.660 | 467.4 | **High** | Complex multi-step queries |
| **OctoRoute** | 0.660 | **207.2** | Med | Latency-sensitive edge deployment |
| CROW-OctoRoute | 0.658 | 497.8 | High | Research demonstrator |
| StaticSemantic | 0.212 | 385.2 | None | Baseline |

> OctoRoute achieves **~22% lower latency** than APRR via direct functional-token dispatch. CROW is expected to show larger success gains on real ToolBench G3 (out-of-distribution) splits where deliberation quality separates from greedy routing.

### Source

```
src/aprr_extensions/
  crow_router.py          # CROWRouter: CoT-gated routing with quality-weighted ΔW
  octoroute_router.py     # OctoRouteRouter: functional tokens + arm-local W
  __init__.py

experiments/extensions/
  aprr_comparison_benchmark.py  # 5-router benchmark runner

notebooks/extensions/
  APRR_CROW_OctoRoute_Benchmark.ipynb  # Kaggle/Colab-ready

paper/extensions/
  APRR_CROW_OctoRoute_Extension.md    # Full research extension document

results/
  extension_results.json
```

### PhD Integration Map

| Extension | Objective 1 (SessionRerank) | Objective 3 (MNCD Mesh) | Objective 4 (FCNP Pruning) |
|---|---|---|---|
| **CROW** | Reasoning trace → session priority score | Negative-quality traces → distress signal | Trace relevance → context chunk filter |
| **OctoRoute** | Arm domain → session category hint | CSA drops → mesh distress broadcast | Arm dispatch → domain-gated pruning |

