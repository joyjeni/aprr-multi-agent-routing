"""
Generate all paper figures and tables from `results/results.json`.

Outputs (saved into ./figures and ./tables — both PDF (for LaTeX) and PNG
(for the Vercel dashboard); tables emitted as CSV + LaTeX).

    figures/fig1_convergence_success.{pdf,png}
    figures/fig2_convergence_latency.{pdf,png}
    figures/fig3_affinity_evolution.{pdf,png}
    figures/fig4_per_split_breakdown.{pdf,png}
    figures/fig5_ablation_alpha_beta.{pdf,png}
    figures/fig6_pareto_latency_success.{pdf,png}
    figures/fig7_hops_distribution.{pdf,png}
    tables/table1_main_results.{csv,tex}
    tables/table2_per_split.{csv,tex}
    tables/table3_ablation.{csv,tex}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -- matplotlib styling -----------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

ROUTER_COLORS = {
    "Random":         "#9aa0a6",
    "RoundRobin":     "#f9ab00",
    "StaticSemantic": "#1a73e8",
    "LLMRouter":      "#7e57c2",
    "APRR":           "#d93025",
    "Oracle":         "#0f9d58",
}
ROUTER_ORDER = ["Random","RoundRobin","StaticSemantic","LLMRouter","APRR","Oracle"]


def _save(fig, base: Path, name: str):
    fig.savefig(base / f"{name}.pdf")
    fig.savefig(base / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig1_convergence_success(art, out: Path):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name in ROUTER_ORDER:
        if name not in art["convergence"]: continue
        xs = [r["iteration"] for r in art["convergence"][name]]
        ys = [r["success_rate"] for r in art["convergence"][name]]
        ax.plot(xs, ys, label=name, color=ROUTER_COLORS[name],
                lw=1.8 if name == "APRR" else 1.1,
                ls="--" if name == "Oracle" else "-")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Success rate")
    ax.set_title("Fig. 1: Convergence of success rate")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right", ncol=2, frameon=True)
    _save(fig, out, "fig1_convergence_success")


def fig2_convergence_latency(art, out: Path):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name in ROUTER_ORDER:
        if name not in art["convergence"]: continue
        xs = [r["iteration"] for r in art["convergence"][name]]
        ys = [r["mean_latency_ms"] for r in art["convergence"][name]]
        ax.plot(xs, ys, label=name, color=ROUTER_COLORS[name],
                lw=1.8 if name == "APRR" else 1.1,
                ls="--" if name == "Oracle" else "-")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean episode latency (ms)")
    ax.set_title("Fig. 2: Convergence of mean latency")
    ax.legend(loc="upper right", ncol=2, frameon=True)
    _save(fig, out, "fig2_convergence_latency")


def fig3_affinity_evolution(art, out: Path):
    snaps = art.get("W_snapshots") or art.get("pheromone_snapshots") or []
    if not snaps:
        return
    idxs = [0, len(snaps)//3, (2*len(snaps))//3, len(snaps)-1]
    idxs = sorted(set(i for i in idxs if 0 <= i < len(snaps)))
    roles = art["config"]["topology"]["roles"]
    fig, axes = plt.subplots(1, len(idxs), figsize=(3*len(idxs)+0.5, 3.2))
    if len(idxs) == 1: axes = [axes]
    def _mat(s):
        return np.array(s.get("W", s.get("tau")))
    vmax = max(_mat(s).max() for s in snaps)
    for ax, i in zip(axes, idxs):
        W = _mat(snaps[i])
        im = ax.imshow(W, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(roles))); ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(roles))); ax.set_yticklabels(roles, fontsize=8)
        ax.set_title(f"iter {snaps[i]['iteration']}")
    fig.suptitle("Fig. 3: Routing-affinity matrix W$_{ij}$ — emergence of query-specific shortcuts")
    fig.colorbar(im, ax=axes, shrink=0.7, label="W$_{ij}$")
    fig.savefig(out / "fig3_affinity_evolution.pdf", bbox_inches="tight")
    fig.savefig(out / "fig3_affinity_evolution.png", bbox_inches="tight")
    plt.close(fig)


def fig4_per_split_breakdown(art, episodes, out: Path):
    # per-split success and latency
    by = defaultdict(lambda: defaultdict(list))   # by[router][split] = [succ,...]
    lat = defaultdict(lambda: defaultdict(list))
    n_iter = art["config"]["n_iterations"]
    for r in episodes:
        if r["iteration"] < n_iter - 5:    # use only last 5 iterations (converged)
            continue
        by[r["router"]][r["split"]].append(int(r["success"]))
        lat[r["router"]][r["split"]].append(r["latency_ms"])

    splits = ["G1", "G2", "G3"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    width = 0.13
    xs = np.arange(len(splits))
    for k, name in enumerate(ROUTER_ORDER):
        if name not in by: continue
        ys = [np.mean(by[name][s]) if by[name][s] else 0 for s in splits]
        a1.bar(xs + (k - 2.5) * width, ys, width=width, color=ROUTER_COLORS[name], label=name)
        ys2 = [np.mean(lat[name][s]) if lat[name][s] else 0 for s in splits]
        a2.bar(xs + (k - 2.5) * width, ys2, width=width, color=ROUTER_COLORS[name], label=name)
    for ax in (a1, a2):
        ax.set_xticks(xs); ax.set_xticklabels(splits)
        ax.set_xlabel("ToolBench split")
    a1.set_ylabel("Success rate"); a1.set_title("Per-split success (converged)")
    a2.set_ylabel("Mean latency (ms)"); a2.set_title("Per-split latency (converged)")
    a1.legend(loc="lower right", ncol=2, fontsize=8)
    fig.suptitle("Fig. 4: Per-split breakdown on ToolBench G1/G2/G3")
    _save(fig, out, "fig4_per_split_breakdown")


def fig6_pareto(art, out: Path):
    agg = art["aggregate"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for name in ROUTER_ORDER:
        if name not in agg: continue
        ax.scatter(agg[name]["mean_latency_ms"], agg[name]["success_rate"],
                   s=120, color=ROUTER_COLORS[name], edgecolor="black",
                   linewidth=0.6, label=name, zorder=3)
        ax.annotate(name, (agg[name]["mean_latency_ms"], agg[name]["success_rate"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean episode latency (ms) — lower is better")
    ax.set_ylabel("Success rate — higher is better")
    ax.set_title("Fig. 5: Latency–success Pareto front")
    _save(fig, out, "fig5_pareto_latency_success")


def fig7_hops(art, episodes, out: Path):
    n_iter = art["config"]["n_iterations"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for k, name in enumerate(ROUTER_ORDER):
        hops = [r["hops"] for r in episodes
                if r["router"] == name and r["iteration"] >= n_iter - 5]
        if not hops: continue
        bins = np.arange(1.5, 9.5, 1)
        h, _ = np.histogram(hops, bins=bins, density=True)
        ax.plot(np.arange(2, 9), h, marker="o", color=ROUTER_COLORS[name], label=name)
    ax.set_xlabel("Path length (hops)"); ax.set_ylabel("Density")
    ax.set_title("Fig. 6: Routed path-length distribution (converged)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    _save(fig, out, "fig6_hops_distribution")


# ---------------------------------------------------------------------------
def make_main_table(art, out_tables: Path):
    agg = art["aggregate"]
    rows = []
    for n in ROUTER_ORDER:
        if n not in agg: continue
        a = agg[n]
        rows.append([n, f"{a['success_rate']:.3f} ± {a['success_rate_ci95']:.3f}",
                     f"{a['mean_latency_ms']:.1f}", f"{a['p50_latency_ms']:.1f}",
                     f"{a['p95_latency_ms']:.1f}", f"{a['mean_hops']:.2f}"])
    headers = ["Router", "Success (95% CI)", "Mean lat (ms)",
               "p50 (ms)", "p95 (ms)", "Mean hops"]
    csv = ",".join(headers) + "\n" + "\n".join(",".join(r) for r in rows)
    (out_tables / "table1_main_results.csv").write_text(csv)
    # LaTeX
    tex = ["\\begin{table}[t]", "\\centering",
           "\\caption{Main results on ToolBench (G1/G2/G3 mix). Mean over last 5 iterations, 5 seeds. Best non-oracle in \\textbf{bold}.}",
           "\\label{tab:main}",
           "\\begin{tabular}{lccccc}", "\\toprule",
           " & ".join(headers) + " \\\\", "\\midrule"]
    # bold best (excluding Oracle)
    best_sr  = max((agg[n]["success_rate"] for n in ROUTER_ORDER if n in agg and n != "Oracle"))
    best_lat = min((agg[n]["mean_latency_ms"] for n in ROUTER_ORDER if n in agg and n != "Oracle"))
    for n in ROUTER_ORDER:
        if n not in agg: continue
        a = agg[n]
        sr = f"{a['success_rate']:.3f} $\\pm$ {a['success_rate_ci95']:.3f}"
        if abs(a["success_rate"] - best_sr) < 1e-9 and n != "Oracle":
            sr = "\\textbf{" + sr + "}"
        lat = f"{a['mean_latency_ms']:.1f}"
        if abs(a["mean_latency_ms"] - best_lat) < 1e-9 and n != "Oracle":
            lat = "\\textbf{" + lat + "}"
        tex.append(f"{n} & {sr} & {lat} & {a['p50_latency_ms']:.1f} & "
                   f"{a['p95_latency_ms']:.1f} & {a['mean_hops']:.2f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_tables / "table1_main_results.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
def main(results_path="results/results.json",
         episodes_path="results/episodes.jsonl",
         fig_dir="figures", tbl_dir="tables"):
    art = json.loads(Path(results_path).read_text())
    episodes = [json.loads(l) for l in Path(episodes_path).read_text().splitlines()]
    fig_out = Path(fig_dir); fig_out.mkdir(parents=True, exist_ok=True)
    tbl_out = Path(tbl_dir); tbl_out.mkdir(parents=True, exist_ok=True)
    fig1_convergence_success(art, fig_out)
    fig2_convergence_latency(art, fig_out)
    fig3_affinity_evolution(art, fig_out)
    fig4_per_split_breakdown(art, episodes, fig_out)
    fig6_pareto(art, fig_out)
    fig7_hops(art, episodes, fig_out)
    make_main_table(art, tbl_out)
    print(f"✔ figures → {fig_out}/   tables → {tbl_out}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.json")
    p.add_argument("--episodes", default="results/episodes.jsonl")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--tbl_dir", default="tables")
    a = p.parse_args()
    main(a.results, a.episodes, a.fig_dir, a.tbl_dir)
