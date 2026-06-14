import Image from "next/image";

interface Fig {
  png: string;
  pdf: string;
  caption: string;
  title: string;
}

const FIGS: Fig[] = [
  {
    png: "/figures/fig1_convergence_success.png",
    pdf: "/figures/fig1_convergence_success.pdf",
    title: "Fig. 1 — Convergence (success)",
    caption:
      "Success rate vs iteration across routers. APRR converges above all non-oracle baselines.",
  },
  {
    png: "/figures/fig3_affinity_evolution.png",
    pdf: "/figures/fig3_affinity_evolution.pdf",
    title: "Fig. 3 — Affinity evolution",
    caption:
      "Routing-affinity matrix W over training; near-zero affinity learned toward distractor agents.",
  },
  {
    png: "/figures/fig4_per_split_breakdown.png",
    pdf: "/figures/fig4_per_split_breakdown.pdf",
    title: "Fig. 4 — Per-split breakdown",
    caption:
      "Performance across difficulty splits (G1/G2/G3) showing consistent gains on harder queries.",
  },
  {
    png: "/figures/fig5_pareto_latency_success.png",
    pdf: "/figures/fig5_pareto_latency_success.pdf",
    title: "Fig. 5 — Pareto frontier",
    caption:
      "Latency–success trade-off; APRR Pareto-dominates every non-oracle baseline.",
  },
  {
    png: "/figures/fig7_ablation.png",
    pdf: "/figures/fig7_ablation.pdf",
    title: "Fig. 7 — Hyperparameter ablation",
    caption:
      "Sensitivity to α, β, λ, and γ. Query relevance (γ) yields the dominant effect.",
  },
];

export function Figures() {
  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
      {FIGS.map((f, i) => (
        <figure
          key={f.png}
          className={`card overflow-hidden ${
            i === FIGS.length - 1 && FIGS.length % 2 === 1 ? "md:col-span-2" : ""
          }`}
        >
          <div className="relative aspect-[16/10] w-full bg-white">
            <Image
              src={f.png}
              alt={f.caption}
              fill
              sizes="(max-width: 768px) 100vw, 50vw"
              className="object-contain p-2"
            />
          </div>
          <figcaption className="flex items-start justify-between gap-3 border-t border-line p-4">
            <div>
              <div className="text-sm font-medium text-fg">{f.title}</div>
              <p className="mt-1 text-xs leading-relaxed text-fg-muted">
                {f.caption}
              </p>
            </div>
            <a
              href={f.pdf}
              target="_blank"
              rel="noreferrer"
              className="shrink-0 rounded-md border border-line px-2.5 py-1 text-xs font-medium text-fg-muted transition-colors hover:border-aprr hover:text-aprr-soft"
            >
              PDF
            </a>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
