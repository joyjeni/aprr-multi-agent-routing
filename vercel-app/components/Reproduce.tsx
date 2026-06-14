interface ResLink {
  title: string;
  desc: string;
  href: string;
  cta: string;
  external: boolean;
  accent?: boolean;
  disabled?: boolean;
}

const LINKS: ResLink[] = [
  {
    title: "GitHub repository",
    desc: "Full source: router, baselines, simulator, experiment drivers, and figure scripts.",
    href: "https://github.com/joyjeni/aprr-multi-agent-routing",
    cta: "joyjeni/aprr-multi-agent-routing",
    external: true,
    accent: true,
  },
  {
    title: "Colab notebook",
    desc: "Reproducible end-to-end benchmark — runs the 5-seed evaluation on a free tier.",
    href: "https://github.com/joyjeni/aprr-multi-agent-routing/blob/main/notebooks/APRR_Reproducible_Benchmark.ipynb",
    cta: "Open notebook",
    external: true,
  },
  {
    title: "Kaggle kernel",
    desc: "Alternative reproducible runtime (placeholder — link to be added on publication).",
    href: "#",
    cta: "Coming soon",
    external: false,
    disabled: true,
  },
];

const DATASETS: { label: string; href: string }[] = [
  { label: "multiseed.json", href: "/data/multiseed.json" },
  { label: "ablation.json", href: "/data/ablation.json" },
  { label: "API · /api/results", href: "/api/results" },
];

const FIG_PDFS: { label: string; href: string }[] = [
  { label: "Fig. 1 (PDF)", href: "/figures/fig1_convergence_success.pdf" },
  { label: "Fig. 3 (PDF)", href: "/figures/fig3_affinity_evolution.pdf" },
  { label: "Fig. 4 (PDF)", href: "/figures/fig4_per_split_breakdown.pdf" },
  { label: "Fig. 5 (PDF)", href: "/figures/fig5_pareto_latency_success.pdf" },
  { label: "Fig. 7 (PDF)", href: "/figures/fig7_ablation.pdf" },
];

export function Reproduce() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        {LINKS.map((l) => (
          <a
            key={l.title}
            href={l.disabled ? undefined : l.href}
            target={l.external ? "_blank" : undefined}
            rel={l.external ? "noreferrer" : undefined}
            className={`card group flex flex-col p-5 transition-colors ${
              l.disabled
                ? "cursor-not-allowed opacity-60"
                : l.accent
                ? "hover:border-aprr"
                : "hover:border-fg-muted"
            }`}
          >
            <div className="text-sm font-semibold text-fg">{l.title}</div>
            <p className="mt-1.5 flex-1 text-xs leading-relaxed text-fg-muted">
              {l.desc}
            </p>
            <span
              className={`mt-4 inline-flex items-center gap-1 font-mono text-xs ${
                l.accent ? "text-aprr-soft" : "text-base-soft"
              }`}
            >
              {l.cta}
              {!l.disabled && (
                <span className="transition-transform group-hover:translate-x-0.5">
                  →
                </span>
              )}
            </span>
          </a>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div className="card p-5">
          <h3 className="text-sm font-semibold text-fg">Result artifacts</h3>
          <p className="mt-1 text-xs text-fg-dim">
            Raw benchmark JSON. The API route returns the same payload for
            programmatic GET/POST from notebooks.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {DATASETS.map((d) => (
              <a
                key={d.href}
                href={d.href}
                target="_blank"
                rel="noreferrer"
                className="rounded-md border border-line bg-ink-900/40 px-3 py-1.5 font-mono text-xs text-fg-muted transition-colors hover:border-base hover:text-base-soft"
              >
                {d.label}
              </a>
            ))}
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-semibold text-fg">Figure PDFs</h3>
          <p className="mt-1 text-xs text-fg-dim">
            Vector PDFs sized for IEEE double-column submission.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {FIG_PDFS.map((f) => (
              <a
                key={f.href}
                href={f.href}
                target="_blank"
                rel="noreferrer"
                className="rounded-md border border-line bg-ink-900/40 px-3 py-1.5 font-mono text-xs text-fg-muted transition-colors hover:border-aprr hover:text-aprr-soft"
              >
                {f.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
