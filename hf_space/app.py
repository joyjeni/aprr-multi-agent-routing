"""
APRR / CTGR / FTDR — Live Multi-Agent Routing Demo
====================================================
Interactive Hugging Face Space for the PhD Objective 2 research
(Adaptive Probabilistic Routing Reinforcement and its two extensions,
CTGR = Chain-of-Thought Gated Router and FTDR = Functional-Token
Dispatch Router).

At startup this Space:
  1. Downloads the real ToolBench-v1 dataset from the Hugging Face Hub
     (tuandunghcmut/toolbench-v1).
  2. Installs and imports the actual `aprr` / `aprr_extensions` source
     code from the Objective 2 GitHub repo (no logic is re-implemented
     here — this Space calls the same router classes used in the
     validated Kaggle notebook).
  3. Runs the same 165-query online (prequential) evaluation loop used
     in the notebook to warm up each router's live affinity matrix, so
     the routers you interact with below have already been reinforced
     on real ToolBench traffic exactly as in the paper's results.

Then you can type your own query and watch APRR-core / CTGR / FTDR make
a live routing decision, with an optional "Objective 1 signal" toggle
that simulates the top-1 category Objective 1's session-aware reranker
would hand off (the real fusion boost formula from the notebook is
applied exactly: an additive boost of
max(BOOST_FLOOR, BOOST_FRACTION * live_score_spread) on the matching
agent's score before the router's own argmax / deliberation step).

Sources
-------
- Base paper: Qin et al., "ToolLLM: Facilitating Large Language Models
  to Master 16000+ Real-world APIs", ICLR 2024.
  https://arxiv.org/abs/2307.16789
- Dataset: https://huggingface.co/datasets/tuandunghcmut/toolbench-v1
- Objective 2 (this Space's routing code):
  https://github.com/joyjeni/aprr-multi-agent-routing
- Objective 1 (session-aware reranking, referenced conceptually):
  https://github.com/joyjeni/session-aware-toolbench-rerank
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import spaces  # noqa: E402  (must be imported before any CUDA-touching libs; required for ZeroGPU startup check)
import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# 1. Fetch the real Objective 2 source code (aprr + aprr_extensions).
#    Installed at Space build/startup time rather than vendored, so this
#    Space always reflects the current state of the GitHub repo.
# ---------------------------------------------------------------------------
REPO_DIR = Path("/tmp/aprr-multi-agent-routing")
if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/joyjeni/aprr-multi-agent-routing.git", str(REPO_DIR)],
        check=True,
    )
sys.path.insert(0, str(REPO_DIR / "src"))

from aprr_extensions import CTGRRouter, FTDRRouter  # noqa: E402
from aprr_extensions.ctgr_router import _infer_source_row  # noqa: E402
import aprr.data.toolbench_loader as obj2_loader  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Download the real ToolBench-v1 benchmark parquet files from the Hub.
# ---------------------------------------------------------------------------
from huggingface_hub import snapshot_download  # noqa: E402

DATA_DIR = Path("/tmp/toolbench_hf")
if not (DATA_DIR / "benchmark").exists():
    snapshot_download(
        repo_id="tuandunghcmut/toolbench-v1",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=["benchmark/*.parquet"],
    )
obj2_loader.ROOT = DATA_DIR
obj2_loader.BENCH = DATA_DIR / "benchmark"

# ---------------------------------------------------------------------------
# 3. Build the real agent pool + routers, exactly as in the notebook.
# ---------------------------------------------------------------------------
N_AGENTS = 5
SEED = 42
BOOST_FRACTION = 0.6
BOOST_FLOOR = 0.15

categories = obj2_loader.top_categories(N_AGENTS)
domain_keywords = obj2_loader.category_keywords(categories, top_n_terms=30)
agent_names = [f"{c.replace(' ', '')}Agent" for c in categories]
cat_to_idx = {c: i for i, c in enumerate(categories)}


def _boost(scores: np.ndarray, idx: int) -> np.ndarray:
    spread = float(scores.max() - scores.min())
    boosted = scores.copy()
    boosted[idx] += max(BOOST_FLOOR, BOOST_FRACTION * spread)
    return boosted


def make_ctgr(threshold: float) -> CTGRRouter:
    return CTGRRouter(
        agents=agent_names, domain_keywords=domain_keywords, domain_order=categories,
        thought_budget_threshold=threshold,
    )


def make_ftdr() -> FTDRRouter:
    return FTDRRouter(
        agents=agent_names, arm_domains=categories, domain_keywords=domain_keywords,
        lambda_decay=0.05, confidence_threshold=0.05,
    )


aprr_core = make_ctgr(1.1)   # APRR-core: high threshold -> effectively always greedy argmax
ctgr = make_ctgr(0.4)        # CTGR: lower threshold -> deliberation branch triggers more often
ftdr = make_ftdr()

# ---------------------------------------------------------------------------
# 4. Warm-up: the same 165-query real ToolBench online evaluation loop used
#    in the validated notebook, so the routers in this demo carry the exact
#    same reinforced state reported in the paper's results table.
# ---------------------------------------------------------------------------
_STARTUP_LOG: list[str] = []


def _log(msg: str) -> None:
    print(msg)
    _STARTUP_LOG.append(msg)


obj2_queries_all = [q for q in obj2_loader.load_all_eval() if q.relevant_categories & set(categories)]
_rng_shuffle = np.random.default_rng(SEED)
_order = _rng_shuffle.permutation(len(obj2_queries_all))
warmup_stream = [obj2_queries_all[i] for i in _order]
_log(f"[warmup] {len(warmup_stream)} real ToolBench eval queries, agent pool: {categories}")


def _warmup_ctgr(router: CTGRRouter) -> dict:
    successes = []
    for q in warmup_stream:
        result = router.route(q.query)
        chosen, trace, mode = result["agent"], result["reasoning"], result["mode"]
        success = categories[chosen] in q.relevant_categories
        quality = router.score_reasoning_quality(trace) if isinstance(trace, dict) else 0.5
        src_row = _infer_source_row(q.query, router.n_agents, router._domain_kws, router._domain_order)
        router.update(agent_from=src_row, agent_to=chosen, success=success,
                       latency_ms=0.0, reasoning_quality=quality)
        successes.append(1.0 if success else 0.0)
    return {"success_rate": round(statistics.fmean(successes), 4), "n": len(successes)}


def _warmup_ftdr(router: FTDRRouter) -> dict:
    successes = []
    for q in warmup_stream:
        result = router.route(q.query)
        chosen, arm_id = result["agent"], result["arm"]
        success = categories[chosen] in q.relevant_categories
        router.update(arm_id=arm_id, agent_to=chosen, success=success, latency_ms=0.0)
        successes.append(1.0 if success else 0.0)
    return {"success_rate": round(statistics.fmean(successes), 4), "n": len(successes)}


t0 = time.time()
_stats_aprr = _warmup_ctgr(aprr_core)
_stats_ctgr = _warmup_ctgr(ctgr)
_stats_ftdr = _warmup_ftdr(ftdr)
_log(f"[warmup] APRR-core online success_rate={_stats_aprr['success_rate']}")
_log(f"[warmup] CTGR online success_rate={_stats_ctgr['success_rate']}")
_log(f"[warmup] FTDR online success_rate={_stats_ftdr['success_rate']}")
_log(f"[warmup] done in {time.time() - t0:.1f}s — routers are now live and reinforced.")

STARTUP_SUMMARY = "\n".join(_STARTUP_LOG)

# ---------------------------------------------------------------------------
# 5. Live single-query routing for the Gradio UI.
# ---------------------------------------------------------------------------


def route_live(query: str, obj1_signal_category: str):
    if not query or not query.strip():
        return "Type a query above first.", "", "", ""

    use_fusion = obj1_signal_category != "(none — no Objective 1 signal)"
    rows = []

    # --- APRR-core ---
    src_row = _infer_source_row(query, aprr_core.n_agents, aprr_core._domain_kws, aprr_core._domain_order)
    row_scores = aprr_core.W[src_row].copy()
    if use_fusion and obj1_signal_category in cat_to_idx:
        row_scores = _boost(row_scores, cat_to_idx[obj1_signal_category])
    complexity = aprr_core.predict_complexity(query)
    if complexity < aprr_core.thought_budget_threshold:
        chosen_aprr, mode_aprr = int(np.argmax(row_scores)), "greedy"
    else:
        top_k = min(aprr_core._TOP_K, aprr_core.n_agents)
        top_k_agents = list(np.argsort(row_scores)[-top_k:][::-1])
        chosen_aprr, mode_aprr = aprr_core.deliberate(query, top_k_agents), "deliberation"
    rows.append(("APRR-core", agent_names[chosen_aprr], categories[chosen_aprr], mode_aprr, row_scores.tolist()))

    # --- CTGR ---
    src_row_c = _infer_source_row(query, ctgr.n_agents, ctgr._domain_kws, ctgr._domain_order)
    row_scores_c = ctgr.W[src_row_c].copy()
    if use_fusion and obj1_signal_category in cat_to_idx:
        row_scores_c = _boost(row_scores_c, cat_to_idx[obj1_signal_category])
    complexity_c = ctgr.predict_complexity(query)
    if complexity_c < ctgr.thought_budget_threshold:
        chosen_ctgr, mode_ctgr = int(np.argmax(row_scores_c)), "greedy"
    else:
        top_k = min(ctgr._TOP_K, ctgr.n_agents)
        top_k_agents = list(np.argsort(row_scores_c)[-top_k:][::-1])
        chosen_ctgr, mode_ctgr = ctgr.deliberate(query, top_k_agents), "deliberation"
    rows.append(("CTGR", agent_names[chosen_ctgr], categories[chosen_ctgr], mode_ctgr, row_scores_c.tolist()))

    # --- FTDR ---
    result = ftdr.route(query)
    chosen_ftdr, arm_id = result["agent"], result["arm"]
    local_scores = ftdr.arms[arm_id].W_local.sum(axis=0)
    if use_fusion and obj1_signal_category in cat_to_idx:
        local_scores = _boost(local_scores, cat_to_idx[obj1_signal_category])
        chosen_ftdr = int(np.argmax(local_scores))
    rows.append(("FTDR", agent_names[chosen_ftdr], categories[chosen_ftdr], f"arm={arm_id}", local_scores.tolist()))

    # Format as a markdown table + per-router score bars
    header = "| Router | Chosen agent | Category | Decision mode |\n|---|---|---|---|\n"
    body = "\n".join(f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} |" for r in rows)
    table_md = header + body

    def _score_md(name, scores):
        pairs = sorted(zip(agent_names, scores), key=lambda p: -p[1])
        return f"**{name} live scores** (higher = preferred)\n\n" + "\n".join(
            f"- {a}: `{s:.3f}`" for a, s in pairs
        )

    aprr_md = _score_md("APRR-core", rows[0][4])
    ctgr_md = _score_md("CTGR", rows[1][4])
    ftdr_md = _score_md("FTDR", rows[2][4])

    fusion_note = (
        f"Objective 1 fusion signal applied: boosting **{obj1_signal_category}** "
        f"(+{BOOST_FRACTION} × live score spread, floor {BOOST_FLOOR})."
        if use_fusion else
        "No Objective 1 fusion signal — routers use their own live-learned affinity only."
    )

    return table_md, aprr_md, ctgr_md, ftdr_md, fusion_note


@spaces.GPU(duration=15)
def _zerogpu_keepalive():
    """No-op GPU-eligible function so Hugging Face ZeroGPU accepts this Space at
    startup. The routers themselves are pure-CPU (numpy-based), so no actual GPU
    work happens here; this only satisfies ZeroGPU's static @spaces.GPU check."""
    return True


def route_live_wrapper(query, obj1_signal_category):
    table_md, aprr_md, ctgr_md, ftdr_md, fusion_note = route_live(query, obj1_signal_category)
    return table_md, fusion_note, aprr_md, ctgr_md, ftdr_md


EXAMPLE_QUERIES = [
    "Find me a comedy movie streaming tonight and also recommend a similar TV show",
    "I need a tool that converts currency and then books a flight",
    "Search for trending music playlists and download the top track",
    "Compare weather APIs and then generate a chart of this week's forecast",
]

with gr.Blocks(title="APRR / CTGR / FTDR — Live Multi-Agent Routing") as demo:
    gr.Markdown(
        """
        # APRR · CTGR · FTDR — Live Multi-Agent Routing Demo
        Objective 2 of a PhD research project on adaptive multi-agent LLM routing.
        This Space runs the **actual router source code** from the
        [aprr-multi-agent-routing](https://github.com/joyjeni/aprr-multi-agent-routing) repo,
        warmed up online on the real
        [ToolBench-v1](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1) dataset
        (base paper: [Qin et al., ToolLLM, ICLR 2024](https://arxiv.org/abs/2307.16789)),
        against a live **5-agent pool** built from ToolBench's own top categories.

        Type a query below and watch **APRR-core**, **CTGR** (Chain-of-Thought Gated
        Router), and **FTDR** (Functional-Token Dispatch Router) each make a live
        routing decision. Optionally simulate the fusion signal that Objective 1's
        session-aware reranker would hand off, and see how each router's decision
        (and internal scores) shift.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            query_box = gr.Textbox(
                label="Your query", placeholder="e.g. Find me a comedy movie and a similar TV show",
                lines=2,
            )
            signal_dropdown = gr.Dropdown(
                choices=["(none — no Objective 1 signal)"] + categories,
                value="(none — no Objective 1 signal)",
                label="Simulated Objective 1 signal (top-1 category from session-aware reranking)",
            )
            run_btn = gr.Button("Route this query", variant="primary")
            gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_box)
        with gr.Column(scale=1):
            gr.Markdown(f"**Live agent pool** (real ToolBench top-{N_AGENTS} categories):")
            gr.Markdown("\n".join(f"- {a} → `{c}`" for a, c in zip(agent_names, categories)))

    fusion_note_box = gr.Markdown()
    result_table = gr.Markdown(label="Routing decisions")
    with gr.Row():
        aprr_scores = gr.Markdown()
        ctgr_scores = gr.Markdown()
        ftdr_scores = gr.Markdown()

    run_btn.click(
        route_live_wrapper,
        inputs=[query_box, signal_dropdown],
        outputs=[result_table, fusion_note_box, aprr_scores, ctgr_scores, ftdr_scores],
    )

    with gr.Accordion("Startup log (data download + warm-up on 165 real ToolBench queries)", open=False):
        gr.Markdown(f"```\n{STARTUP_SUMMARY}\n```")

    gr.Markdown(
        """
        ---
        **Sources:** [ToolLLM paper (Qin et al., ICLR 2024)](https://arxiv.org/abs/2307.16789) ·
        [ToolBench-v1 dataset](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1) ·
        [Objective 2 repo (this Space's code)](https://github.com/joyjeni/aprr-multi-agent-routing) ·
        [Objective 1 repo](https://github.com/joyjeni/session-aware-toolbench-rerank) ·
        [Live results dashboard](https://aprr-multi-agent-routing.vercel.app)
        """
    )

if __name__ == "__main__":
    demo.launch()
