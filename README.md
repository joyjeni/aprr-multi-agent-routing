# APRR — Adaptive Probabilistic Routing & Reinforcement

> **PhD Objective 2** | Multi-Agent Routing for Agricultural AI

[![GitHub](https://img.shields.io/badge/GitHub-Public-black)](https://github.com/joyjeni/aprr-multi-agent-routing)
[![Vercel](https://img.shields.io/badge/Vercel-Live-brightgreen)](https://aprr-multi-agent-routing.vercel.app)
[![Kaggle](https://img.shields.io/badge/Kaggle-Benchmark-blue)](https://www.kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**GitHub**: https://github.com/joyjeni/aprr-multi-agent-routing (PUBLIC)
**Vercel Demo**: https://aprr-multi-agent-routing.vercel.app

---

## What is APRR?

APRR (**Adaptive Probabilistic Routing & Reinforcement**) is a multi-agent routing framework that learns to dispatch queries to the correct agent using REINFORCE-style online updates on a weighted routing matrix W. Each routing decision is a stochastic sample from a probability distribution over agent transitions; successful routes strengthen the W-matrix while penalising for latency and path length.

**Novel Contribution**: APRR is the first system combining:
1. **REINFORCE-equivalent online routing** — W updated at every episode, no offline batch.
2. **CoT quality-weighted W update** — the weight increment is scaled by the quality of the chain-of-thought reasoning trace.
3. **Bio-inspired arm dispatch** (OctoRoute) — octopus-inspired functional token routing for parallel agent arms.

### Sub-Components

APRR contains two novel sub-components (not separate objectives):

| Sub-Component | Full Name                          | Role                                          |
|---------------|------------------------------------|-----------------------------------------------|
| **CROW**      | Chain-of-Reasoning Over Workload   | Complexity-gated deliberation before routing  |
| **OctoRoute** | Octopus-inspired functional tokens | `<octo_k>` tokens dispatch parallel agent arms|

---

## Deployment

**GitHub Repository**: https://github.com/joyjeni/aprr-multi-agent-routing

**Vercel Deployment**: https://aprr-multi-agent-routing.vercel.app

### Kaggle Notebooks

| Notebook                                   | Purpose                              |
|--------------------------------------------|--------------------------------------|
| `APRR_Reproducible_Benchmark.ipynb`        | Core APRR benchmark (baseline suite) |
| `APRR_CROW_OctoRoute_Benchmark.ipynb`      | CROW + OctoRoute extension benchmark |

---

## Core Equations

### Agent Transition Probability

$$P(a_j \mid a_i, q) \;\propto\; W_{ij}^\alpha \cdot \eta_{ij}^\beta \cdot \psi_j(q)^\gamma$$

where:
- $W_{ij}$ — learned edge weight from agent $i$ to agent $j$
- $\eta_{ij}$ — heuristic suitability (domain overlap between $a_i$'s output and $a_j$'s input type)
- $\psi_j(q)$ — query-agent affinity score for agent $j$ given query $q$
- $\alpha, \beta, \gamma$ — exponent hyperparameters (control exploration vs. exploitation)

### W-Matrix Update (REINFORCE-equivalent)

$$W \;\leftarrow\; (1 - \lambda) \cdot W \;\;+\;\; \frac{\kappa \cdot \mathbf{1}[\text{success}]}{L^2 \cdot \text{lat\_norm}}$$

where:
- $\lambda$ — global decay rate (forgetting)
- $\kappa$ — learning rate
- $L$ — number of hops in the routing path (penalises long paths)
- $\text{lat\_norm}$ — normalised end-to-end latency (penalises slow routes)

### CROW: Complexity-Gated Deliberation

$$\Delta W \;\mathrel{*}=\; \left(1 + \beta \cdot \rho(T_q)\right)$$

where $\rho(T_q)$ is the **CoT quality score** of reasoning trace $T_q$ and $\beta$ controls how much quality amplifies the weight update. High-quality reasoning traces produce larger W updates; low-quality traces are discounted.

CROW gates deliberation: if query complexity $c(q) > \theta$, CROW generates a full reasoning trace before routing. If $c(q) \leq \theta$, the query routes directly (fast path).

### OctoRoute: Functional Token Dispatch

OctoRoute introduces `<octo_k>` functional tokens (inspired by octopus arm autonomy). Each token maps to a specific agent arm, enabling **parallel dispatch** of query sub-tasks.

- Reduces end-to-end latency by **−22%** vs sequential routing.
- Arm label `<octo_k>` gates domain-specific pruning in Obj4/FCNP.

---

## Results

### Core APRR Benchmark

| Metric              | Value       | Notes                              |
|---------------------|-------------|-------------------------------------|
| Success Rate        | **0.470**   | Episode-level task completion       |
| Mean Latency        | **261.3 ms**| End-to-end (routing + agent calls)  |
| Mean Hops           | **2.77**    | Average routing path length         |
| Pareto Optimality   | ✓           | Dominates all baselines on success/latency frontier |

The APRR system is **Pareto-optimal**: no baseline achieves higher success rate at lower latency simultaneously.

### CROW + OctoRoute Extension

| Component   | Metric       | Delta vs APRR baseline |
|-------------|--------------|------------------------|
| CROW        | CoT quality  | +ρ(T_q) amplification  |
| OctoRoute   | Latency      | **−22%**               |

---

## Repository Structure

```
aprr-multi-agent-routing/
├── src/
│   ├── aprr/                    # Core APRR routing engine
│   │   ├── router.py            # P(a_j|a_i,q), W-matrix update
│   │   ├── agent_registry.py    # Agent catalogue (domain, I/O types)
│   │   └── episode.py           # Episode logging (user_language field)
│   └── aprr_extensions/         # CROW + OctoRoute sub-components
│       ├── crow.py              # CROW deliberation, ΔW *= (1+β·ρ)
│       └── octoroute.py         # <octo_k> functional token dispatch
├── experiments/
│   ├── benchmark_core.py        # Core APRR benchmark
│   └── extensions/
│       └── benchmark_extensions.py  # CROW + OctoRoute benchmark
├── kaggle/
│   ├── APRR_Reproducible_Benchmark.ipynb
│   └── APRR_CROW_OctoRoute_Benchmark.ipynb
├── dashboard/                   # Vercel deployment frontend
└── docs/
    └── README_OBJ2.md           # This file
```

---

## Episode Metadata Schema

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Episode:
    episode_id: str
    query_en: str               # English query (post-translation)
    user_language: str          # ISO code: en, kn, ta, ml, hi, mr
    agent_path: List[str]       # Sequence of agent IDs visited
    success: bool
    latency_ms: float
    hops: int
    crow_used: bool             # Was CROW deliberation triggered?
    cot_quality: Optional[float]  # ρ(T_q) if CROW used
    octo_arm: Optional[str]     # <octo_k> token if OctoRoute used
```

---

## Integration with PhD Pipeline

APRR is **Objective 2** in the four-component PhD pipeline:

```
[Obj1: SessionRerank+] ──priority scores──► [Obj2: APRR]
                                                   │
                          routing confidence ──►  [Obj3: MNCD]
                                                   │
                           octo_arm label ──────► [Obj4: FCNP]
```

### Incoming Signals

| Source | Signal                         | Usage in APRR                              |
|--------|--------------------------------|--------------------------------------------|
| Obj1   | API priority ranking vector    | Initialise W-matrix for current session    |

### Outgoing Signals

| Destination | Signal                      | Purpose                                         |
|-------------|-----------------------------|-------------------------------------------------|
| Obj3/MNCD   | Routing confidence score    | Triggers distress channel if confidence < τ     |
| Obj4/FCNP   | OctoRoute arm label         | Gates domain-specific context pruning strategy  |

---

## Multilingual Support

APRR's routing logic is **language-agnostic**. All routing decisions are made on English queries (post-translation from Obj1's language detection layer).

Each episode logs `user_language` in metadata for:
- Per-language performance analysis
- Identifying if certain languages route to suboptimal agents
- Feeding back to multilingual evaluation benchmarks

See [`/docs/multilingual_integration.md`](./multilingual_integration.md) for the full multilingual design.

---

## Running Locally

```bash
git clone https://github.com/joyjeni/aprr-multi-agent-routing
cd aprr-multi-agent-routing
pip install -r requirements.txt

# Run core benchmark
python experiments/benchmark_core.py

# Run CROW + OctoRoute benchmark
python experiments/extensions/benchmark_extensions.py

# Launch Vercel dashboard locally
cd dashboard && npm install && npm run dev
```

---

## Citation

```bibtex
@misc{aprr2026,
  title  = {APRR: Adaptive Probabilistic Routing and Reinforcement for Multi-Agent LLMs},
  author = {Jeni, Joy},
  year   = {2026},
  note   = {PhD Objective 2. https://github.com/joyjeni/aprr-multi-agent-routing}
}
```

---

*Part of the PhD Agricultural AI pipeline. See also: [Obj1 SessionRerank+](./README_OBJ1.md) | [Obj3 MNCD](./README_OBJ3.md) | [Obj4 FCNP](./README_OBJ4.md) | [Multilingual Design](./multilingual_integration.md)*
