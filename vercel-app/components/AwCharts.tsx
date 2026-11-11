import type { AppWorldResults, AwConfig } from "@/lib/appworld";
import { AW } from "@/lib/appworld";

function ChartCard({
  title,
  subtitle,
  legend,
  children,
}: {
  title: string;
  subtitle: string;
  legend?: { label: string; color: string }[];
  children: React.ReactNode;
}) {
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-fg">{title}</h3>
      <p className="mt-0.5 text-xs leading-relaxed text-fg-dim">{subtitle}</p>
      {legend && (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
          {legend.map((l) => (
            <span key={l.label} className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: l.color }} />
              {l.label}
            </span>
          ))}
        </div>
      )}
      <div className="mt-4">{children}</div>
    </div>
  );
}

function RowLabel({ c }: { c: AwConfig }) {
  return (
    <div
      className={`w-36 shrink-0 truncate text-xs sm:w-40 ${
        c.id === "full" ? "font-medium text-aprr-soft" : "text-fg-muted"
      }`}
      title={c.description}
    >
      {c.label}
    </div>
  );
}

/* Task success with a 95% Wilson interval, one row per configuration. */
function SuccessIntervals({ rows, refPct }: { rows: AwConfig[]; refPct: number }) {
  const max = 40;
  const x = (v: number) => `${(100 * v) / max}%`;
  return (
    <ChartCard
      title="Task success with 95% interval"
      subtitle="Every interval overlaps the reference agent (dashed line): on 57 tasks no configuration is significantly better or worse."
    >
      <div className="space-y-2.5">
        {rows.map((c) => {
          const isFull = c.id === "full";
          const color = isFull ? AW.accent : AW.neutralSoft;
          const [lo, hi] = c.tgc_ci95 ?? [0, 0];
          return (
            <div key={c.id} className="flex items-center gap-3">
              <RowLabel c={c} />
              <div
                className="relative h-5 flex-1"
                title={`${c.label}: ${c.tgc_pct}% (${c.tgc_solved}/57), 95% interval ${lo}–${hi}%`}
              >
                <div className="absolute inset-y-0 border-l border-dashed" style={{ left: x(refPct), borderColor: AW.reference }} />
                <div
                  className="absolute top-1/2 h-[2px] -translate-y-1/2 rounded"
                  style={{ left: x(lo), width: `calc(${x(hi)} - ${x(lo)})`, background: color }}
                />
                <div
                  className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-ink-800"
                  style={{ left: x(c.tgc_pct ?? 0), background: color }}
                />
              </div>
              <div className="w-12 shrink-0 text-right font-mono text-xs tabular-nums text-fg">
                {c.tgc_pct?.toFixed(1)}%
              </div>
            </div>
          );
        })}
        <div className="flex items-center gap-3 pt-1">
          <div className="w-36 shrink-0 sm:w-40" />
          <div className="relative h-4 flex-1 text-[10px] text-fg-dim">
            {[0, 10, 20, 30, 40].map((t) => (
              <span key={t} className="absolute -translate-x-1/2" style={{ left: x(t) }}>
                {t}%
              </span>
            ))}
          </div>
          <div className="w-12 shrink-0" />
        </div>
      </div>
    </ChartCard>
  );
}

/* Which model wrote each step: stacked 8B + 4B steps per configuration. */
function RouterMix({ rows }: { rows: AwConfig[] }) {
  const max = Math.max(...rows.map((c) => (c.large_steps ?? 0) + (c.small_steps ?? 0)));
  return (
    <ChartCard
      title="Which model wrote each step"
      subtitle="Steps over all 57 tasks. With O3 active, 19–25% of steps go to the 4B model; episodes are also shorter, so full ACRS calls the 8B model 49% less often than the reference agent."
      legend={[
        { label: "Qwen3-8B (large)", color: AW.accent },
        { label: "Qwen3-4B (small)", color: AW.accentSoft },
      ]}
    >
      <div className="space-y-2.5">
        {rows.map((c) => {
          const L = c.large_steps ?? 0;
          const S = c.small_steps ?? 0;
          return (
            <div key={c.id} className="flex items-center gap-3">
              <RowLabel c={c} />
              <div
                className="flex h-4 flex-1 gap-[2px]"
                title={`${c.label}: ${L} steps on the 8B model, ${S} on the 4B model`}
              >
                <div className="h-full rounded-l" style={{ width: `${(100 * L) / max}%`, background: AW.accent }} />
                {S > 0 && (
                  <div className="h-full rounded-r" style={{ width: `${(100 * S) / max}%`, background: AW.accentSoft }} />
                )}
              </div>
              <div className="w-20 shrink-0 text-right font-mono text-xs tabular-nums text-fg">
                {L}
                {S > 0 && <span className="text-fg-muted"> + {S}</span>}
              </div>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}

/* Mean prompt tokens per task. */
function PromptTokens({ rows }: { rows: AwConfig[] }) {
  const max = Math.max(...rows.map((c) => c.prompt_tokens_mean_per_task ?? 0));
  return (
    <ChartCard
      title="Prompt tokens per task (mean)"
      subtitle="O2 alone cuts the median prompt per task by 23% against the reference agent (Wilcoxon p = 0.0013). Full ACRS spends tokens on O4's candidate programs."
    >
      <div className="space-y-2.5">
        {rows.map((c) => {
          const v = c.prompt_tokens_mean_per_task ?? 0;
          const isFull = c.id === "full";
          return (
            <div key={c.id} className="flex items-center gap-3">
              <RowLabel c={c} />
              <div className="h-4 flex-1" title={`${c.label}: ${v.toLocaleString("en-GB")} prompt tokens per task`}>
                <div
                  className="h-full rounded"
                  style={{ width: `${(100 * v) / max}%`, background: isFull ? AW.accent : AW.neutral }}
                />
              </div>
              <div className="w-20 shrink-0 text-right font-mono text-xs tabular-nums text-fg">
                {(v / 1000).toFixed(1)}k
              </div>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}

export function AwCharts({ aw }: { aw: AppWorldResults }) {
  const done = aw.configs.filter((c) => c.status === "done");
  const ref = done.find((c) => c.id === "reference");
  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
      <SuccessIntervals rows={done} refPct={ref?.tgc_pct ?? 0} />
      <RouterMix rows={done} />
      <div className="lg:col-span-2">
        <PromptTokens rows={done} />
      </div>
    </div>
  );
}
