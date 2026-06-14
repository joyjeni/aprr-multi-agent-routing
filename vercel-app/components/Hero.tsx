import { Logo } from "./Logo";

function Metric({
  value,
  label,
  accent,
}: {
  value: string;
  label: string;
  accent?: boolean;
}) {
  return (
    <div className="card p-5">
      <div
        className={`font-mono text-3xl font-semibold tracking-tight sm:text-4xl ${
          accent ? "text-aprr-soft" : "text-fg"
        }`}
      >
        {value}
      </div>
      <div className="mt-1.5 text-sm text-fg-muted">{label}</div>
    </div>
  );
}

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      <div className="section-pad pb-14 pt-16 sm:pt-24">
        <div className="flex items-center gap-2.5">
          <Logo className="h-8 w-8 text-fg" />
          <span className="eyebrow">Multi-agent LLM routing · RL</span>
        </div>

        <h1 className="mt-6 max-w-4xl text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
          <span className="text-aprr-soft">APRR</span>: Adaptive Probabilistic
          Routing Reinforcement
        </h1>

        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-fg-muted sm:text-xl">
          Online policy-iteration routing for multi-agent LLM workflows. A
          decay-regularised, success-weighted affinity matrix learns shorter,
          higher-quality call sequences — without LLM gradient updates.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-fg-muted">
          <span>
            <span className="text-fg">Jenisha&nbsp;T</span> · MS Ramaiah
            University of Applied Sciences
          </span>
          <span className="hidden h-3 w-px bg-line sm:inline-block" />
          <span>5 seeds · 500 queries × 40 iterations</span>
        </div>

        <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Metric value="−35.7%" label="Mean latency vs StaticSemantic" accent />
          <Metric value="−23.9%" label="Mean hops per query" accent />
          <Metric value="Pareto-optimal" label="On the latency–success frontier" />
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <a
            href="#results"
            className="rounded-md bg-aprr px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-aprr-soft"
          >
            View benchmark
          </a>
          <a
            href="#reproduce"
            className="rounded-md border border-line px-5 py-2.5 text-sm font-medium text-fg transition-colors hover:border-fg-muted"
          >
            Reproduce results
          </a>
        </div>
      </div>
    </section>
  );
}
