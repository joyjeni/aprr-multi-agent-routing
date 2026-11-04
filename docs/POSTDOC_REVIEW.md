# Postdoctoral review — O2 — APRR (multi-agent routing)

Reviewer stance: a postdoctoral researcher preparing this chapter for a Q1 journal (November 2026).
Severity: **Critical** blocks publication · **Major** a reviewer will require it · **Fixed** done in this commit.

| Severity | Finding | Evidence | Action |
|---|---|---|---|
| Critical | Main results come from a circular simulator | `ToolBenchSimulator.generate` builds each query embedding as the mean of its ground-truth agents' embeddings plus noise, so an embedding-similarity router recovers the answer almost by construction. The repo name and tables say "ToolBench" but these queries are synthetic. | Make the real-query benchmark (`experiments/extensions`, n = 165 real ToolBench eval queries) the primary result; label simulator results as sanity checks. |
| Critical | Real-data gains are within noise and unstable | On 165 real ToolBench queries (one seed), APRR-core scores 74.5 % vs StaticSemantic 70.9 % — a 3.6 pp gap with a ≈ ±10 pp 95 % interval. CTGR scores 74.5 % in one table and 55.8 % in the pipeline notebook. On the synthetic benchmark APRR (0.449) and StaticSemantic (0.451) are tied. | Run ≥ 500 real queries × ≥ 5 seeds with a paired McNemar test; add LinUCB and Thompson-sampling baselines; report one canonical real-data table. |
| Major | Theory does not cover performance | Proposition 1 (REINFORCE equivalence at α = γ = 1, β = λ = 0) and Theorem 1 (bounded W, geometric decay of unrewarded edges) are about stability, not about routing quality; there is no regret or convergence-to-optimal statement. | Add a regret bound (e.g. the risk-signed LinUCB formulation in ACRS master `docs/THEORY.md`, Prop. 2) or prove convergence of the affinity update to the best agent. |
| Fixed | Tests could not be collected | `tests/test_router.py` imported `aprr` without the `src/` layout on the path. | Added `tests/conftest.py`; 4/4 tests pass. |
| Major | 33 MB raw log in the repository | `results/episodes.jsonl` is committed uncompressed. | `make_figures.py` now also reads `results/episodes.jsonl.gz`; gzip the log (3.7 MB) and delete the raw file in a follow-up commit. |

## What this chapter needs to be journal-ready

1. Real LLM agents (open-weight 7–8B class) instead of simulated ones, with ≥ 300 evaluation items per arm.
2. At least two recognised public benchmarks (see the dataset plan in `phd-agentic-ai-master/docs/POSTDOC_ROADMAP.md`).
3. Pre-registered primary metric and one headline statistical test; all other comparisons labelled exploratory.
4. Leak-free splits and a reproducibility command that regenerates every table.
