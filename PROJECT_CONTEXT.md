# APRR Project — Shared Context for Subagents

This file is the **single source of truth** for any subagent working on this
project. Read it first, then read the code and results files referenced
below.

## TL;DR

**APRR** (Adaptive Probabilistic Routing Reinforcement) is an online
policy-iteration router for multi-agent LLM workflows. It maintains a
routing-affinity matrix W ∈ ℝ^{n×n} between specialised agents, updated by
decay-regularised success-weighted reinforcement on traversed edges, and
samples next-agent decisions from a softmax-style policy combining:

  1. learned affinity W_ij (α)
  2. semantic prior η_ij = cos(e_i, e_j) (β)
  3. query relevance ψ_j(q) = cos(q, e_j) (γ)

P(a_j | a_i, q) ∝ W_ij^α · η_ij^β · ψ_j(q)^γ

After every routed episode:
  W ← (1 − λ) · W + κ · 1[success]/(L² · latency_norm) on edges of the path.

**IMPORTANT — no bio-mimic vocabulary.** This is a CSE/Agentic-AI / RL
paper. Do NOT use the words: ant, colony, swarm, weight-signal, trail,
evaporation, parallel search, biological, nature-inspired, indirect coordination, hive,
sting, larva. Use: routing affinity, online policy iteration,
reinforcement, decay/decay-regularisation, edge reinforcement, success-
weighted update, multi-trajectory sampling.

## Repository layout (workspace: /home/user/workspace/aprr-multi-agent-routing)

    src/aprr/                  core library
      __init__.py
      agents.py                AgentTopology, Agent dataclass
      router.py                APRRRouter + RouterConfig (the algorithm)
      baselines.py             Random / RoundRobin / StaticSemantic / LLMRouter / Oracle
      toolbench.py             ToolBench-style simulator (G1/G2/G3, LCS reward)
      runner.py                full experiment driver
      metrics.py               EpisodeResult + aggregate_metrics

    experiments/
      multiseed.py             5-seed eval → results/multiseed.json (main table)
      ablation.py              α/β/λ/γ grid → results/ablation.json
      make_figures.py          generates Fig. 1-6
      make_ablation_figure.py  generates Fig. 7

    results/
      seed_{42..46}/results.json    per-seed artifacts
      seed_{42..46}/episodes.jsonl  per-seed episodes
      multiseed.json           combined 5-seed summary  ← USE FOR PAPER TABLES
      ablation.json            ablation table  ← USE FOR FIG. 7 & TAB. III

    figures/                   PNG + PDF artifacts (paper/figures/ has a copy)
    tables/                    CSV + LaTeX
    paper/                     LaTeX manuscript directory
    notebooks/                 Colab/Kaggle reproducible notebook
    vercel-app/                Next.js dashboard

## Main numerical results (5 seeds, 500 queries × 40 iterations)

| Router          | Success (95% CI) | Mean latency (ms) | Mean hops |
|-----------------|-------------------|-------------------|-----------|
| Random          | 0.324 ± 0.006     | 395.5             | 3.51      |
| RoundRobin      | 0.379 ± 0.007     | 546.1             | 4.33      |
| StaticSemantic  | 0.451 ± 0.007     | 414.5             | 3.70      |
| LLMRouter       | 0.404 ± 0.007     | 148.3             | 2.05      |
| **APRR (ours)** | **0.449 ± 0.007** | **315.8**         | **3.03**  |
| Oracle          | 0.899 ± 0.004     | 452.7             | 4.32      |

Source of truth for these numbers: `tables/table1_main_results.csv` — always regenerate this table from that CSV rather than hand-editing it; do not let this file drift from the CSV again (that drift happened once already and was corrected in August 2026).

**Headline claims (corrected August 2026 — see README's "Review-driven corrections" section for full history):**
- APRR is **Pareto-optimal**, not Pareto-dominant: its 95% CI on success (0.449 ± 0.007) overlaps StaticSemantic's (0.451 ± 0.007), so APRR does not strictly dominate StaticSemantic on success rate. APRR's advantage is a large **mean-latency reduction** (315.8 ms vs StaticSemantic's 414.5 ms, ≈23.8% lower) at statistically indistinguishable success rate — i.e. APRR sits on/near the Pareto frontier, trading a small amount of success-rate headroom for materially lower latency and hop count, rather than beating every baseline on every axis.
- LLMRouter has lower latency (148.3 ms) but meaningfully lower success (0.404) than APRR — APRR is not latency-optimal, it is the best success/latency **trade-off** among learned/adaptive routers.
- γ (query-relevance) ablation and distractor-avoidance findings: see `results/ablation.json` and Fig. 3 — re-verify exact numbers against that file before quoting them; do not carry forward unverified ablation figures from earlier drafts.

## Base paper and differentiation from concurrent SOTA (updated August 2026)

**Base/SOTA architecture this work is positioned against:** MasRouter (Yue, Zhang, Liu, Wan, Wang, Cheng, Qi. *MasRouter: Learning to Route LLMs for Multi-Agent Systems.* ACL 2025, pp. 15549–15572. https://aclanthology.org/2025.acl-long.757/) — a cascaded, policy-gradient-trained 3-stage neural controller (collaboration-mode → role allocation → LLM-backbone routing). See README.md § "Base Paper & Design Architecture (Objective 2)" for the full delta table, ASCII architecture diagram, and property comparison — that section is the canonical, defense-ready version; do not duplicate/diverge from it here.

Other verified, closest concurrent works (all citations independently checked against publisher pages — title/authors/venue/year/URL — before being used anywhere in this repo):
- **DyTopo** (Lu, Hu, Zhao, Cao, 2026, arXiv:2602.06039): manager-guided per-round sparse graph reconstruction via semantic matching. [NOTE: earlier drafts of this file incorrectly attributed this to "Hong et al." — corrected.]
- **Differentiable MoA (DMoA)** (Wu, Lu, Yan, Qiu, Hu, Guo, Yang, 2026, arXiv:2605.15706): differentiable routing trained with predictive-entropy self-supervision.
- **NeuroMAS** (Lu, Fang, Zhong, Ma, 2026, arXiv:2605.16757): MAS as a trainable neural net via joint RL. [NOTE: earlier drafts cited this with no author list — corrected.]
- **FlowBank** (Yu et al., 2026, arXiv:2606.11290): query-adaptive workflow precompute-and-reuse.
- **MetaCogAgent** (2026, arXiv:2605.17292, submitted IEEE SMC 2026): metacognitive self-assessment for delegation.

[REMOVED] A prior draft of this file cited an unverifiable "Stigmergy for capability selection (Hanke, 2026)" as the "closest competitor." No such paper could be verified against any publisher/arXiv record — it has been removed rather than kept as an unverified claim. If a genuinely verified stigmergy/pheromone-style capability-routing paper is found later, add it back with its real citation; until then, do not reintroduce it.

**APRR's distinct contribution:**
  1. Closed-form policy gradient interpretation (Appendix A of the paper).
  2. Free-tier-friendly (no LLM gradient updates required for the core algorithm — only the LLMRouter baseline trains).
  3. Distractor-aware: provably robust to semantically-similar but useless agents.
  4. Order-preserving LCS reward enables learning correct call **sequences** (not just sets).
  5. Cost-quadratic deposit ∝ 1/L² makes shorter paths exponentially preferred (this is novel — prior work uses 1/L or constant).

## Key files for subagents

- Numerical results: `results/multiseed.json` and `results/ablation.json`
- Figures (PNG for web, PDF for paper): `figures/*` and `paper/figures/*`
- LaTeX table: `tables/table1_main_results.tex`
- Algorithm reference: `src/aprr/router.py` (read its docstring + the `update_trail` method for Equations 1-4)
- Baselines: `src/aprr/baselines.py`
- Simulator: `src/aprr/toolbench.py` (read the `rollout` docstring for the success model)

## Target venue

**IEEE Transactions on Neural Networks and Learning Systems** (TNNLS), or
**IEEE Transactions on Pattern Analysis and Machine Intelligence** (TPAMI).
Length: 8-10 pages double-column, IEEEtran class.

## Author

Jenisha T (PhD candidate, MS Ramaiah University of Applied Sciences)
GitHub: joyjeni
