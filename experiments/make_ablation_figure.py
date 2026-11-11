"""Generate the ablation figure (Fig. 7) from `results/ablation.json`."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "figure.dpi": 130,
    "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25,
})

PARAM_LABELS = {
    "alpha": r"$\alpha$ (learned-affinity weight)",
    "beta":  r"$\beta$ (semantic-prior weight)",
    "lam":   r"$\lambda$ (decay rate)",
    "gamma": r"$\gamma$ (query-relevance weight)",
}


def main(in_path="results/ablation.json", out_dir="figures"):
    records = json.loads(Path(in_path).read_text())
    by_param = defaultdict(list)
    for r in records:
        by_param[r["param"]].append(r)

    # Two rows of small multiples (success above, latency below) instead of a second y-axis.
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharex="col")
    for j, param in enumerate(["alpha", "beta", "lam", "gamma"]):
        rs = sorted(by_param[param], key=lambda r: r["value"])
        xs = [r["value"] for r in rs]
        ys = [r["success_rate"] for r in rs]
        ls = [r["latency_ms"] for r in rs]
        top, bottom = axes[0][j], axes[1][j]
        top.plot(xs, ys, "o-", color="#2a78d6", lw=2)
        top.set_ylim(0.25, 0.55)
        bottom.plot(xs, ls, "s-", color="#6e6d68", lw=1.6)
        bottom.set_xlabel(PARAM_LABELS[param])
        if j == 0:
            top.set_ylabel("Success rate")
            bottom.set_ylabel("Latency (ms)")
    fig.suptitle("Fig. 7: APRR hyperparameter ablation")
    fig.tight_layout()
    out = Path(out_dir)
    fig.savefig(out / "fig7_ablation.pdf")
    fig.savefig(out / "fig7_ablation.png")
    plt.close(fig)
    print(f"✔ {out}/fig7_ablation.{{pdf,png}}")


if __name__ == "__main__":
    main()
