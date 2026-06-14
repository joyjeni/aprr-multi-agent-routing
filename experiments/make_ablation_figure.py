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

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, param in zip(axes, ["alpha", "beta", "lam", "gamma"]):
        rs = sorted(by_param[param], key=lambda r: r["value"])
        xs = [r["value"] for r in rs]
        ys = [r["success_rate"] for r in rs]
        ls = [r["latency_ms"] for r in rs]
        ax.plot(xs, ys, "o-", color="#d93025", lw=2, label="Success rate")
        ax.set_xlabel(PARAM_LABELS[param])
        ax.set_ylabel("Success rate", color="#d93025")
        ax.tick_params(axis="y", labelcolor="#d93025")
        ax.set_ylim(0.25, 0.55)
        ax2 = ax.twinx()
        ax2.plot(xs, ls, "s--", color="#1a73e8", lw=1.4, label="Latency")
        ax2.set_ylabel("Latency (ms)", color="#1a73e8")
        ax2.tick_params(axis="y", labelcolor="#1a73e8")
        ax2.grid(False)
    fig.suptitle("Fig. 7: APRR hyperparameter ablation")
    fig.tight_layout()
    out = Path(out_dir)
    fig.savefig(out / "fig7_ablation.pdf")
    fig.savefig(out / "fig7_ablation.png")
    plt.close(fig)
    print(f"✔ {out}/fig7_ablation.{{pdf,png}}")


if __name__ == "__main__":
    main()
