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
    title: "ACRS on AppWorld",
    desc: "The agent behind the AppWorld results (branch o4-provenance): all four objectives, the policy check, the Kaggle notebook and offline tests.",
    href: "https://github.com/joyjeni/acrs-appworld/tree/o4-provenance",
    cta: "joyjeni/acrs-appworld",
    external: true,
    accent: true,
  },
  {
    title: "GitHub repository",
    desc: "APRR source: router, baselines, simulator, experiment drivers and figure scripts.",
    href: "https://github.com/joyjeni/aprr-multi-agent-routing",
    cta: "joyjeni/aprr-multi-agent-routing",
    external: true,
  },
  {
    title: "Colab notebook",
    desc: "Reproducible end-to-end benchmark — runs the 5-seed evaluation on a free tier.",
    href: "https://github.com/joyjeni/aprr-multi-agent-routing/blob/main/notebooks/APRR_Reproducible_Benchmark.ipynb",
    cta: "Open notebook",
    external: true,
  },
  {
    title: "O1 → O3 pipeline (earlier)",
    desc: "Kaggle notebook: session-aware reranking feeds APRR/CTGR/FTDR online routing on real ToolBench data.",
    href: "https://github.com/joyjeni/aprr-multi-agent-routing/blob/main/notebooks/APRR_Objective1_Objective2_ToolBench.ipynb",
    cta: "Open notebook",
    external: true,
  },
];

const DATASETS: { label: string; href: string }[] = [
  { label: "appworld_dev.json", href: "/data/appworld_dev.json" },
  { label: "multiseed.json", href: "/data/multiseed.json" },
  { label: "ablation.json", href: "/data/ablation.json" },
  { label: "pipeline_results.json", href: "/data/pipeline_results.json" },
  { label: "affinity_seed42.json", href: "/data/affinity_seed42.json" },
  { label: "per_split_seed42.json", href: "/data/per_split_seed42.json" },
  { label: "API · /api/results", href: "/api/results" },
];

const SCRIPTS: { label: string; href: string }[] = [
  {
    label: "make_figures.py",
    href: "https://github.com/joyjeni/aprr-multi-agent-routing/blob/main/experiments/make_figures.py",
  },
  {
    label: "make_ablation_figure.py",
    href: "https://github.com/joyjeni/aprr-multi-agent-routing/blob/main/experiments/make_ablation_figure.py",
  },
  {
    label: "appworld_dev.json",
    href: "/data/appworld_dev.json",
  },
];

export function Reproduce() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
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
            Aggregate result files (AppWorld: no task text or API output).
            The API route returns the same payload for programmatic access.
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
          <h3 className="text-sm font-semibold text-fg">Figure scripts</h3>
          <p className="mt-1 text-xs text-fg-dim">
            Regenerate the manuscript figures (neutral palette) from the
            result files; AppWorld numbers come from the aggregate file.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {SCRIPTS.map((f) => (
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
