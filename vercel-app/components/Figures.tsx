// Earlier-experiment figures, drawn from data rather than images so that the colours follow the site
// palette (blue accent, amber oracle, grey baselines; no red or green).
import fs from "node:fs";
import path from "node:path";
import { ROUTER_COLORS, ROUTER_ORDER } from "@/lib/types";

interface Affinity {
  source: string;
  roles: string[];
  snapshots: { iteration: number; W: number[][] }[];
}

interface PerSplit {
  source: string;
  splits: string[];
  routers: { router: string; success: number[]; latency_ms: number[] }[];
}

function load<T>(name: string): T {
  return JSON.parse(fs.readFileSync(path.join(process.cwd(), "public", "data", name), "utf-8")) as T;
}

const LABEL: Record<string, string> = {
  Random: "Random",
  RoundRobin: "Round-Robin",
  StaticSemantic: "Static Semantic",
  LLMRouter: "LLM Router",
  APRR: "APRR",
  Oracle: "Oracle",
};

// Sequential ramp for the heatmap: from the card surface (no affinity) to light blue (strongest).
const RAMP = ["#1b1b1f", "#173456", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"];

function rampColor(t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = RAMP[i];
  const b = RAMP[i + 1];
  const ch = (h: string, k: number) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);
  const mix = [0, 1, 2].map((k) => Math.round(ch(a, k) + f * (ch(b, k) - ch(a, k))));
  return `rgb(${mix.join(",")})`;
}

function Heatmaps({ data }: { data: Affinity }) {
  const vmax = Math.max(...data.snapshots.flatMap((s) => s.W.flat()));
  const n = data.roles.length;
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-fg">Routing-affinity matrix W during training</h3>
      <p className="mt-0.5 text-xs text-fg-dim">
        Rows: from-agent, columns: to-agent. Brighter cells carry more learned affinity; little affinity goes to
        the two distractor agents. Seed 42.
      </p>
      <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
        {data.snapshots.map((s) => (
          <div key={s.iteration}>
            <div className="mb-1.5 text-xs text-fg-muted">iteration {s.iteration}</div>
            <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
              {s.W.map((row, i) =>
                row.map((v, j) => (
                  <div
                    key={`${i}-${j}`}
                    className="aspect-square rounded-[2px]"
                    style={{ background: rampColor(v / vmax) }}
                    title={`${data.roles[i]} → ${data.roles[j]}: ${v}`}
                  />
                ))
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3 text-[11px] text-fg-dim">
        <span>0</span>
        <div
          className="h-2 w-40 rounded"
          style={{ background: `linear-gradient(90deg, ${RAMP.join(", ")})` }}
        />
        <span>{vmax}</span>
        <span className="ml-2">Agents in order: {data.roles.join(", ")}</span>
      </div>
    </div>
  );
}

function SplitPanel({
  title,
  data,
  pick,
  fmt,
}: {
  title: string;
  data: PerSplit;
  pick: (r: PerSplit["routers"][number]) => number[];
  fmt: (v: number) => string;
}) {
  const max = Math.max(...data.routers.flatMap((r) => pick(r)));
  return (
    <div>
      <div className="mb-2 text-xs font-medium text-fg-muted">{title}</div>
      <div className="grid grid-cols-3 gap-4">
        {data.splits.map((split, si) => (
          <div key={split}>
            <div className="flex h-40 items-end gap-[2px] border-b border-ink-500">
              {ROUTER_ORDER.map((name) => {
                const r = data.routers.find((x) => x.router === name);
                if (!r) return null;
                const v = pick(r)[si];
                return (
                  <div
                    key={name}
                    className="flex-1 rounded-t"
                    style={{ height: `${(100 * v) / max}%`, background: ROUTER_COLORS[name] }}
                    title={`${LABEL[name]} · ${split}: ${fmt(v)}`}
                  />
                );
              })}
            </div>
            <div className="mt-1 text-center text-xs text-fg-dim">{split}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PerSplitChart({ data }: { data: PerSplit }) {
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-fg">Per-split breakdown on ToolBench G1/G2/G3 (after convergence)</h3>
      <p className="mt-0.5 text-xs text-fg-dim">
        APRR is marginally ahead of the other non-oracle routers on G1 and G2 but below three of them on G3; the
        LLM Router is fastest on every split. Hover a bar for its value.
      </p>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
        {ROUTER_ORDER.map((r) => (
          <span key={r} className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: ROUTER_COLORS[r] }} />
            {LABEL[r]}
          </span>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-1 gap-6 md:grid-cols-2">
        <SplitPanel title="Success rate" data={data} pick={(r) => r.success} fmt={(v) => v.toFixed(3)} />
        <SplitPanel
          title="Mean latency (ms)"
          data={data}
          pick={(r) => r.latency_ms}
          fmt={(v) => `${v.toFixed(0)} ms`}
        />
      </div>
      <p className="mt-4 text-[11px] leading-relaxed text-fg-dim">{data.source}</p>
    </div>
  );
}

export function Figures() {
  const affinity = load<Affinity>("affinity_seed42.json");
  const split = load<PerSplit>("per_split_seed42.json");
  return (
    <div className="space-y-5">
      <Heatmaps data={affinity} />
      <PerSplitChart data={split} />
    </div>
  );
}
