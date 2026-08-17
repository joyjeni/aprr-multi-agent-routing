"""Real ToolBench loaders for the Objective 2 routing benchmark.

Adapted from Objective 1's loader
(``session-aware-toolbench-rerank/src/sessionrerank/data/loader.py``) so
Objective 1 and Objective 2 read the identical dataset with the identical
schema, letting Objective 2's router consume Objective 1's ranked
candidate-API output directly.

Benchmark eval splits (``benchmark/g{1,2,3}_instruction-*.parquet``): each
row has

  - query_id: int
  - query:    str
  - api_list: list[dict]      candidate APIs (JSON-encoded string on disk)
  - relevant_apis: list[[tool_name, api_name]]   ground truth

This is real, human/model-annotated ToolBench data (Qin et al., ToolLLM,
ICLR 2024) — not a synthetic simulator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "data" / "toolbench_hf"
BENCH = ROOT / "benchmark"

BENCHMARK_FILES = {
    "G1": "g1_instruction-00000-of-00001.parquet",
    "G2": "g2_instruction-00000-of-00001.parquet",
    "G3": "g3_instruction-00000-of-00001.parquet",
}


@dataclass(frozen=True)
class API:
    tool_name: str
    api_name: str
    category: str
    description: str

    @property
    def uid(self) -> str:
        return f"{self.tool_name}::{self.api_name}"


def _parse_api(d: dict) -> API:
    return API(
        tool_name=d.get("tool_name", ""),
        api_name=d.get("api_name", ""),
        category=d.get("category_name", "") or "Other",
        description=d.get("api_description", "") or "",
    )


@dataclass
class EvalQuery:
    split: str
    query_id: int
    query: str
    candidate_apis: list[API]
    relevant_keys: set[tuple[str, str]]  # set of (tool_name, api_name)

    @property
    def relevant_categories(self) -> set[str]:
        """Categories of the ground-truth relevant APIs for this query."""
        by_key = {(a.tool_name, a.api_name): a.category for a in self.candidate_apis}
        cats = set()
        for k in self.relevant_keys:
            if k in by_key:
                cats.add(by_key[k])
        return cats


def _coerce_list(x):
    if isinstance(x, str):
        return json.loads(x)
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return list(x) if x is not None else []


def load_eval_split(split: str) -> list[EvalQuery]:
    if not BENCH.exists():
        raise FileNotFoundError(
            f"ToolBench benchmark data not found at {BENCH}. "
            "Run `python -m aprr.data.download_toolbench` first."
        )
    fname = BENCHMARK_FILES[split]
    df = pd.read_parquet(BENCH / fname)
    out: list[EvalQuery] = []
    for _, row in df.iterrows():
        cands = [_parse_api(d) for d in _coerce_list(row["api_list"])]
        rel_raw = _coerce_list(row["relevant_apis"])
        rel = set()
        for r in rel_raw:
            r = _coerce_list(r) if not isinstance(r, (list, tuple)) else r
            if len(r) >= 2:
                rel.add((str(r[0]), str(r[1])))
        out.append(
            EvalQuery(
                split=split,
                query_id=int(row["query_id"]),
                query=str(row["query"]),
                candidate_apis=cands,
                relevant_keys=rel,
            )
        )
    return out


@lru_cache(maxsize=1)
def load_all_eval() -> list[EvalQuery]:
    """All eval queries across G1+G2+G3, in a single flat list."""
    out: list[EvalQuery] = []
    for split in BENCHMARK_FILES:
        out.extend(load_eval_split(split))
    return out


def top_categories(k: int = 5) -> list[str]:
    """Return the ``k`` most frequent real ToolBench API categories
    (by number of distinct eval queries whose candidate pool includes
    that category) across G1+G2+G3."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for q in load_all_eval():
        for cat in {a.category for a in q.candidate_apis}:
            counts[cat] += 1
    return [c for c, _ in counts.most_common(k)]


def category_keywords(categories: list[str], top_n_terms: int = 40) -> dict[str, list[str]]:
    """Derive a keyword vocabulary per category from real API names and
    descriptions in that category (used to seed the CTGR/FTDR routers'
    domain-keyword vocabulary with genuine ToolBench terminology instead
    of hand-written agri-domain keyword lists)."""
    import re
    from collections import Counter

    _STOP = {
        "the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "with",
        "is", "this", "that", "be", "are", "by", "as", "it", "from", "will",
        "can", "you", "your", "api", "data", "get", "returns", "return",
        "parameter", "parameters", "value", "default", "type", "string",
        "request", "response", "used", "use", "given",
    }

    per_cat_terms: dict[str, Counter] = {c: Counter() for c in categories}
    for q in load_all_eval():
        for a in q.candidate_apis:
            if a.category not in per_cat_terms:
                continue
            text = f"{a.tool_name} {a.api_name} {a.description}".lower()
            for tok in re.findall(r"[a-z]{4,}", text):
                if tok not in _STOP:
                    per_cat_terms[a.category][tok] += 1

    # Discriminative filter: drop terms that appear in the top-N list of
    # more than half the categories (generic API vocabulary like "search",
    # "details", "info") so each category keeps distinctive vocabulary.
    raw_top: dict[str, list[str]] = {
        cat: [w for w, _ in counter.most_common(top_n_terms * 2)]
        for cat, counter in per_cat_terms.items()
    }
    doc_freq: Counter = Counter()
    for terms in raw_top.values():
        for t in set(terms):
            doc_freq[t] += 1
    n_cats = max(1, len(raw_top))

    result: dict[str, list[str]] = {}
    for cat, terms in raw_top.items():
        filtered = [t for t in terms if doc_freq[t] <= max(1, n_cats // 2)]
        result[cat] = (filtered or terms)[:top_n_terms]
    return result
