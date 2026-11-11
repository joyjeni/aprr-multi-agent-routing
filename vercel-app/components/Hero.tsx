import { Logo } from "./Logo";
import type { AppWorldResults } from "@/lib/appworld";
import { cfg } from "@/lib/appworld";

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
      <div className="mt-1.5 text-sm leading-snug text-fg-muted">{label}</div>
    </div>
  );
}

export function Hero({ aw }: { aw: AppWorldResults }) {
  const full = cfg(aw, "full");
  const ref = cfg(aw, "reference");
  const smallShare = 100 - (full.large_share_pct ?? 0);
  const fewer = Math.round(
    (100 * ((ref.large_steps ?? 0) - (full.large_steps ?? 0))) / (ref.large_steps ?? 1)
  );
  const pairRef = aw.paired_mcnemar.find(
    (p) => p.a === "Full ACRS" && p.b === "Reference agent"
  );
  const noO3 = cfg(aw, "no_o3");
  const fewerNoO3 = Math.round(
    (100 * ((noO3.large_steps ?? 0) - (full.large_steps ?? 0))) / (noO3.large_steps ?? 1)
  );

  return (
    <section id="top" className="relative overflow-hidden">
      <div className="section-pad pb-14 pt-16 sm:pt-24">
        <div className="flex items-center gap-2.5">
          <Logo className="h-8 w-8 text-fg" />
          <span className="eyebrow">ACRS · Objective O3 · Risk-weighted policy routing</span>
        </div>

        <h1 className="mt-6 max-w-4xl text-4xl font-semibold leading-[1.08] tracking-tight sm:text-6xl">
          Choosing a <span className="text-aprr-soft">4B or an 8B model</span> for every step of a tool agent
        </h1>

        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-fg-muted sm:text-xl">
          APRR started as a routing method for multi-agent workflows on simulated ToolBench queries. In ACRS it is
          objective O3: a contextual Thompson sampler that picks Qwen3-4B or Qwen3-8B for each step of an AppWorld
          agent, and trusts the small model on a risky step only when it is clearly better.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-fg-muted">
          <span>
            <span className="text-fg">Jenisha&nbsp;T</span> · Ph.D. research, Ramaiah University of Applied
            Sciences
          </span>
          <span className="hidden h-3 w-px bg-line sm:inline-block" />
          <span>
            AppWorld {aw.benchmark.split} split · {aw.benchmark.tasks} tasks · {aw.benchmark.seeds} seed · updated{" "}
            {aw.updated}
          </span>
        </div>

        <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Metric
            value={`${smallShare.toFixed(1)}%`}
            label={`of full-ACRS steps written by the 4B model (${full.small_steps} of ${
              (full.small_steps ?? 0) + (full.large_steps ?? 0)
            })`}
            accent
          />
          <Metric
            value={`−${fewer}%`}
            label={`calls to the 8B model against the reference agent (${full.large_steps} vs ${ref.large_steps})`}
            accent
          />
          <Metric
            value={`${full.tgc_pct}% / ${ref.tgc_pct}%`}
            label={`task success, full ACRS / reference agent: no significant difference (McNemar p = ${pairRef?.p.toFixed(
              2
            )})`}
          />
          <Metric
            value={`−${fewerNoO3}%`}
            label={`calls to the 8B model against the same system without O3 (${full.large_steps} vs ${noO3.large_steps}); the per-step risk check is still pending`}
          />
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <a
            href="#appworld"
            className="rounded-md bg-aprr px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-aprr-dim"
          >
            AppWorld results
          </a>
          <a
            href="#earlier"
            className="rounded-md border border-line px-5 py-2.5 text-sm font-medium text-fg transition-colors hover:border-fg-muted"
          >
            Earlier experiments
          </a>
        </div>
      </div>
    </section>
  );
}
