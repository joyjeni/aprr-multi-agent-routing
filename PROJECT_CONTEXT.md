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

| Router          | Success rate    | Mean latency (ms) | Mean hops |
|-----------------|-----------------|-------------------|-----------|
| Random          | 0.323 ± 0.004   | 393.9 ± 1.3       | 3.50      |
| RoundRobin      | 0.380 ± 0.002   | 546.7 ± 0.3       | 4.33      |
| StaticSemantic  | 0.499 ± 0.024   | 406.6 ± 60.4      | 3.64      |
| LLMRouter       | 0.406 ± 0.010   | 151.0 ± 3.0       | 2.07      |
| **APRR (ours)** | **0.470 ± 0.020** | **261.3 ± 31.7** | **2.77**  |
| Oracle          | 0.900 ± 0.001   | 448.9 ± 9.4       | 4.30      |

**Headline claims:**
- 35.7% latency reduction vs StaticSemantic (406.6 → 261.3 ms)
- 23.9% hop reduction (3.64 → 2.77)
- Pareto-dominates all non-oracle baselines on the (latency, success) frontier
- γ (query-relevance) ablation: success 0.289 → 0.483 as γ: 0 → 1.5
- W matrix learns to avoid distractor agents (Fig. 3 shows near-zero affinity to distractor_a/_b)

## Differentiation from concurrent SOTA (June 2026)

The closest concurrent works are:
- **DyTopo** (Hong et al., 2026): manager-guided per-round sparse graph reconstruction.
- **Differentiable MoA (DMoA)** (Wu et al., 2026): differentiable routing trained with predictive-entropy self-supervision.
- **NeuroMAS** (2026): MAS as a trainable neural net via RL.
- **FlowBank** (2026): precompute-and-reuse portfolio of workflows.
- **MetaCogAgent** (2026): metacognitive self-assessment for delegation.
- **Stigmergy for capability selection** (Hanke, 2026): weight-signal-inspired capability layer (closest competitor — APRR must differentiate by being decay-regularised online policy iteration with provable REINFORCE-equivalence, not a signal-weighted field).

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
