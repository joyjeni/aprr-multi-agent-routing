# APRR — Adaptive Probabilistic Routing Reinforcement

**Online policy-iteration routing for multi-agent LLM workflows.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange.svg)](https://colab.research.google.com/github/joyjeni/aprr-multi-agent-routing/blob/main/notebooks/APRR_Reproducible_Benchmark.ipynb)
[![Dashboard](https://img.shields.io/badge/Results-Vercel-black.svg)](https://aprr.vercel.app)

> Reference implementation accompanying the manuscript
> *"Adaptive Probabilistic Routing Reinforcement: Online Policy Iteration
> for Dynamic Agent-to-Agent Routing in Tool-Augmented LLM Workflows"*
> (submitted, IEEE Transactions on Neural Networks and Learning Systems).

---

## What is APRR?

Modern multi-agent LLM systems (AutoGen, CrewAI, LangGraph, ReAct chains)
route each query through a hand-designed pipeline — *intent → search →
retrieve → rerank → tool → compose → verify → respond*. The pipeline is
static and pays full cost on every query, even for simple ones that could
have skipped most of it.

**APRR** is an online routing algorithm that learns, from successful
trajectories, an `n × n` **routing-affinity matrix** `W` over agents.
At each hop, it samples the next agent from:

```
P(a_j | a_i, q) ∝ W_ij^α · η_ij^β · ψ_j(q)^γ
```

where `η_ij = cos(e_i, e_j)` is a **semantic prior** between agent role
embeddings, and `ψ_j(q) = cos(q, e_j)` is the **query-relevance** signal.
After each routed episode, `W` is updated by

```
W ← (1 − λ) · W + κ · 1[success] · (1 / L²) · (1 / latency)   on traversed edges
```

— a **decay-regularised, success-weighted, cost-quadratic** update. We
prove (Proposition 1, Appendix A of the paper) that for α=γ=1, β=0 this
update is equivalent to a REINFORCE step (Williams, 1992) with a learned
baseline.

## Main results (5 seeds, 95% CI)

| Router          | Success rate    | Mean latency (ms) | Mean hops |
|-----------------|-----------------|-------------------|-----------|
| Random          | 0.323 ± 0.004   | 393.9 ± 1.3       | 3.50      |
| RoundRobin      | 0.380 ± 0.002   | 546.7 ± 0.3       | 4.33      |
| StaticSemantic  | 0.499 ± 0.024   | 406.6 ± 60.4      | 3.64      |
| LLMRouter       | 0.406 ± 0.010   | 151.0 ± 3.0       | 2.07      |
| **APRR (ours)** | **0.470 ± 0.020** | **261.3 ± 31.7** | **2.77**  |
| Oracle          | 0.900 ± 0.001   | 448.9 ± 9.4       | 4.30      |

**Headline:** APRR cuts latency by **35.7%** vs StaticSemantic and
reduces routed hops by **23.9%**, while remaining within 3 pp of the
best non-oracle on success. APRR sits on the **Pareto front** of the
(latency, success) plane (Fig. 5).

The query-relevance term γ is the dominant factor — ablating it drops
success from 48.3 → 28.9% (Fig. 7).

## Repository layout

```
aprr-multi-agent-routing/
├── src/aprr/                  # Core library
│   ├── router.py              # APRR algorithm (Eq. 1-4)
│   ├── agents.py              # 10-agent topology with 2 distractors
│   ├── baselines.py           # Random / RoundRobin / StaticSemantic / LLMRouter / Oracle
│   ├── toolbench.py           # ToolBench-style simulator (G1/G2/G3, LCS reward)
│   ├── runner.py              # Experiment driver
│   └── metrics.py             # Aggregate metrics
├── experiments/
│   ├── multiseed.py           # 5-seed evaluation
│   ├── ablation.py            # α / β / λ / γ grid
│   ├── make_figures.py        # Figures 1-6
│   └── make_ablation_figure.py # Figure 7
├── notebooks/
│   └── APRR_Reproducible_Benchmark.ipynb   # Colab / Kaggle (free T4)
├── paper/
│   ├── main.tex               # IEEEtran manuscript
│   ├── manuscript.md          # Markdown mirror
│   └── figures/               # All figures as PDF + PNG
├── vercel-app/                # Next.js dashboard (live results)
├── figures/, tables/, results/  # Generated artifacts
└── tests/
```

## Quick start

```bash
git clone https://github.com/joyjeni/aprr-multi-agent-routing.git
cd aprr-multi-agent-routing
pip install numpy matplotlib pandas seaborn scipy
export PYTHONPATH=src

# Full single-seed run (≈20 s, CPU only)
python -m aprr.runner --n_queries 600 --n_iterations 50 --seed 42

# Multi-seed evaluation for the paper's Table I (≈2 min)
python experiments/multiseed.py

# Reproduce every figure and table
python experiments/make_figures.py
python experiments/ablation.py
python experiments/make_ablation_figure.py
```

All figures land in `figures/*.{pdf,png}` and all tables in `tables/*.{csv,tex}` —
they are reused verbatim in the manuscript (`paper/figures/`) and the dashboard
(`vercel-app/public/figures/`).

## Run on Colab / Kaggle (free GPU)

Open [`notebooks/APRR_Reproducible_Benchmark.ipynb`](notebooks/APRR_Reproducible_Benchmark.ipynb)
in Colab or upload to Kaggle. The full benchmark runs in **under 3 minutes
on free CPU**; the optional Gemma-2-2B-it case study uses the free T4 GPU.

## Live dashboard

All results are also published live at
**[https://aprr.vercel.app](https://aprr.vercel.app)**. The dashboard
reads from `vercel-app/public/data/multiseed.json` — the same artifact
the paper uses — so the numbers in the manuscript, the dashboard, and
this README are guaranteed to agree.

## Citation

```bibtex
@article{jenisha2026aprr,
  title   = {Adaptive Probabilistic Routing Reinforcement: Online Policy
             Iteration for Dynamic Agent-to-Agent Routing in
             Tool-Augmented {LLM} Workflows},
  author  = {Jenisha, T.},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Jenisha T** — PhD candidate, Department of Computer Science Engineering,
MS Ramaiah University of Applied Sciences, Bengaluru.
GitHub: [@joyjeni](https://github.com/joyjeni)
