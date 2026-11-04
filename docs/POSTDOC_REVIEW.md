# Postdoctoral review — O2 — APRR (multi-agent routing)

Reviewer stance: a postdoctoral researcher preparing this chapter for a Q1 journal (November 2026).
Severity: **Critical** blocks publication · **Major** a reviewer will require it · **Fixed** done in this commit.

| Severity | Finding | Evidence | Action |
|---|---|---|---|
| Critical | Main results come from a circular simulator | `ToolBenchSimulator.generate` builds each query embedding as the mean of its ground-truth agents' embeddings plus noise, so an embedding-similarity router recovers the answer almost by construction. The repo name and tables say "ToolBench" but these queries are synthetic. | Make the real-query benchmark (`experiments/extensions`, n = 165 real ToolBench eval queries) the primary result; label simulator results as sanity checks. |
| Critical | The core APRR router is not in the real-data table | On real queries only Random, StaticSemantic, CTGR, FTDR and the hybrid are reported. CTGR 74.5 % vs StaticSemantic 70.9 % (n = 165) is a 3.6 pp gap with a ≈ ±10 pp 95 % interval — not significant. | Add APRR-core, LinUCB and Thompson sampling to the real-data run; use ≥ 500 queries and a paired McNemar test. |
| Major | No theory for the routing update | The affinity-matrix update has no convergence or regret statement. | Either adopt the risk-signed LinUCB formulation with its regret statement (ACRS master `docs/THEORY.md`, Prop. 2) or prove convergence of the affinity update. |
| Fixed | Tests could not be collected | `tests/test_router.py` imported `aprr` without the `src/` layout on the path. | Added `tests/conftest.py`; 4/4 tests pass. |
| Major | 33 MB raw log in the repository | `results/episodes.jsonl` is committed uncompressed. | `make_figures.py` now also reads `results/episodes.jsonl.gz`; gzip the log (3.7 MB) and delete the raw file in a follow-up commit. |

## What this chapter needs to be journal-ready

1. Real LLM agents (open-weight 7–8B class) instead of simulated ones, with ≥ 300 evaluation items per arm.
2. At least two recognised public benchmarks (see the dataset plan in `phd-agentic-ai-master/docs/POSTDOC_ROADMAP.md`).
3. Pre-registered primary metric and one headline statistical test; all other comparisons labelled exploratory.
4. Leak-free splits and a reproducibility command that regenerates every table.
