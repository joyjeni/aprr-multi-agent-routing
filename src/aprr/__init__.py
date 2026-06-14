"""
APRR — Adaptive Probabilistic Routing Reinforcement
for multi-agent LLM workflows.

Reference implementation accompanying the manuscript:

    "Adaptive Probabilistic Routing Reinforcement: An Online
     Policy-Iteration Method for Dynamic Agent-to-Agent Routing
     in Tool-Augmented LLM Systems"
    (Submitted, IEEE Transactions on Neural Networks and Learning Systems)

Algorithm summary
-----------------
APRR maintains a routing-affinity matrix W ∈ R^{n×n} over n specialised
agents. At each hop, the router samples the next agent from a softmax-like
policy that combines:

    1. Learned affinity   W_ij              (online success-reinforced weight)
    2. Semantic prior     η_ij = sim(e_i,e_j) (cosine similarity of role embeddings)
    3. Query relevance    sim(q, e_j)        (cosine similarity of query embedding)

After every routed episode the affinity matrix is updated by a temporal
discount (decay) and a success-weighted reinforcement on the traversed
edges.  Under mild assumptions the update is equivalent to a policy-
gradient step with a baseline-subtracted reward (Williams, 1992) and is
guaranteed to converge to a locally-optimal routing policy.
"""
from .router import APRRRouter, RouterConfig
from .baselines import (
    RandomRouter,
    RoundRobinRouter,
    StaticSemanticRouter,
    LLMRouter,
    OracleRouter,
)
from .agents import Agent, AgentTopology
from .metrics import EpisodeResult, aggregate_metrics

__all__ = [
    "APRRRouter",
    "RouterConfig",
    "RandomRouter",
    "RoundRobinRouter",
    "StaticSemanticRouter",
    "LLMRouter",
    "OracleRouter",
    "Agent",
    "AgentTopology",
    "EpisodeResult",
    "aggregate_metrics",
]

__version__ = "1.0.0"
