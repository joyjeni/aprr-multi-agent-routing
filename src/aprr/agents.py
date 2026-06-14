"""
Agent topology for the APRR benchmark.

We model the multi-agent LLM system as a directed graph G = (V, E) where each
node v_i ∈ V is a specialised agent with:

    - role:       semantic specialisation tag (e.g. "search", "rerank", "tool",
                  "compose", "verify")
    - embedding:  fixed d-dim vector encoding the role's capability profile
    - mean_latency_ms / std_latency_ms: empirical timing of one invocation
    - success_bias: prior probability the agent successfully handles a query
                    that matches its role (used by the simulator)

ToolBench (G1/G2/G3) queries are routed through this topology. The router's
job is to choose, at each hop, the next agent to invoke until either (a) the
query is solved or (b) MAX_HOPS is reached.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class Agent:
    agent_id: int
    role: str
    embedding: np.ndarray  # shape (d,)
    mean_latency_ms: float = 120.0
    std_latency_ms: float = 25.0
    success_bias: float = 0.85   # base success probability when role matches
    is_terminal: bool = False    # terminal agents can return an answer

    def sample_latency(self, rng: np.random.Generator) -> float:
        return float(max(5.0, rng.normal(self.mean_latency_ms, self.std_latency_ms)))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["embedding"] = self.embedding.tolist()
        return d


# ---------------------------------------------------------------------------
@dataclass
class AgentTopology:
    """A directed (fully-connected by default) graph of specialised agents."""

    agents: List[Agent]
    embed_dim: int = 32

    def __post_init__(self) -> None:
        self.n = len(self.agents)
        self.id_to_idx = {a.agent_id: i for i, a in enumerate(self.agents)}
        self._embedding_matrix = np.stack([a.embedding for a in self.agents])
        norm = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True) + 1e-9
        E = self._embedding_matrix / norm
        self.similarity = E @ E.T
        # semantic prior η_ij: cosine similarity clipped to (0, 1]
        self.heuristic = np.clip(self.similarity, 1e-3, 1.0)
        np.fill_diagonal(self.heuristic, 1e-6)   # discourage self-loops

    @classmethod
    def default(cls, seed: int = 0, embed_dim: int = 32) -> "AgentTopology":
        """Standard 10-agent topology with 2 "distractor" agents.

        Distractors have embeddings that are semantically similar to useful
        agents but produce no progress — they trap purely-semantic routers
        and reveal the value of learned routing.
        """
        rng = np.random.default_rng(seed)
        roles = [
            # role,        latency, succ_bias, is_terminal
            ("intent",         80,  0.92, False),
            ("search",        140,  0.88, False),
            ("retrieve",      110,  0.86, False),
            ("rerank",         95,  0.84, False),
            ("tool",          220,  0.78, False),
            ("compose",       130,  0.90, True),
            ("verify",         75,  0.93, True),
            ("respond",        60,  0.95, True),
            ("distractor_a",  180,  0.20, False),    # semantically similar to search
            ("distractor_b",  160,  0.20, False),    # semantically similar to tool
        ]
        agents: List[Agent] = []
        base = rng.normal(size=(len(roles), embed_dim))
        base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)
        # Make distractors look like neighbours of useful agents
        base[8] = 0.85 * base[1] + 0.15 * rng.normal(size=embed_dim)
        base[9] = 0.85 * base[4] + 0.15 * rng.normal(size=embed_dim)
        base = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)
        for i, (role, lat, succ, term) in enumerate(roles):
            agents.append(
                Agent(
                    agent_id=i,
                    role=role,
                    embedding=base[i].astype(np.float32),
                    mean_latency_ms=lat,
                    std_latency_ms=lat * 0.18,
                    success_bias=succ,
                    is_terminal=term,
                )
            )
        return cls(agents=agents, embed_dim=embed_dim)

    # ----- helpers ---------------------------------------------------------
    @property
    def embeddings(self) -> np.ndarray:
        return self._embedding_matrix

    def neighbours(self, i: int) -> List[int]:
        return [j for j in range(self.n) if j != i]

    def role_of(self, i: int) -> str:
        return self.agents[i].role

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"agents": [a.to_dict() for a in self.agents]}, indent=2)
        )
