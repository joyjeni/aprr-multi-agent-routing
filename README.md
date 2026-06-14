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
