"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
  LabelList,
  ReferenceLine,
} from "recharts";
import type { MultiSeed, AblationRow } from "@/lib/types";
import { ROUTER_ORDER, ROUTER_COLORS } from "@/lib/types";

const AXIS = "#6b6b73";
const GRID = "#26262c";

const LABEL: Record<string, string> = {
  Random: "Random",
  RoundRobin: "Round-Robin",
  StaticSemantic: "Static Semantic",
  LLMRouter: "LLM Router",
  APRR: "APRR",
  Oracle: "Oracle",
};

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-fg">{title}</h3>
      <p className="mt-0.5 text-xs text-fg-dim">{subtitle}</p>
      <div className="mt-4 h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          {children as React.ReactElement}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const tooltipStyle = {
  background: "#121214",
  border: "1px solid #26262c",
  borderRadius: 8,
  fontSize: 12,
  color: "#f5f5f7",
};

/* ---------- (a) Convergence: success rate vs iteration ---------- */
function ConvergenceChart({ data }: { data: MultiSeed }) {
  const merged = useMemo(() => {
    const n = data.n_iterations;
    const rows: Record<string, number>[] = [];
    for (let i = 0; i < n; i++) {
      const row: Record<string, number> = { iteration: i };
      for (const r of ROUTER_ORDER) {
        const series = data.convergence_mean[r];
        if (series && series[i]) row[r] = series[i].success_rate;
      }
      rows.push(row);
    }
    return rows;
  }, [data]);

  return (
    <ChartCard
      title="Convergence — success rate vs iteration"
      subtitle="Online learning curve; APRR rises above non-oracle baselines as affinities adapt."
    >
      <LineChart data={merged} margin={{ top: 5, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis
          dataKey="iteration"
          stroke={AXIS}
          tick={{ fontSize: 11 }}
          label={{ value: "iteration", position: "insideBottom", offset: -2, fill: AXIS, fontSize: 11 }}
        />
        <YAxis
          stroke={AXIS}
          tick={{ fontSize: 11 }}
          domain={[0.25, 0.95]}
          tickFormatter={(v) => v.toFixed(2)}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          formatter={(v: number, name: string) => [v.toFixed(3), LABEL[name] ?? name]}
          labelFormatter={(l) => `Iteration ${l}`}
        />
        <Legend
          wrapperStyle={{ fontSize: 11 }}
          formatter={(v) => LABEL[v] ?? v}
        />
        {ROUTER_ORDER.map((r) => (
          <Line
            key={r}
            type="monotone"
            dataKey={r}
            stroke={ROUTER_COLORS[r]}
            strokeWidth={r === "APRR" ? 2.6 : 1.4}
            dot={false}
            opacity={r === "APRR" || r === "Oracle" ? 1 : 0.75}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartCard>
  );
}

/* ---------- (b) Pareto scatter: latency vs success ---------- */
function ParetoChart({ data }: { data: MultiSeed }) {
  const points = useMemo(
    () =>
      ROUTER_ORDER.filter((r) => data.summary[r]).map((r) => ({
        router: r,
        latency: data.summary[r].mean_latency_ms.mean,
        success: data.summary[r].success_rate.mean,
      })),
    [data]
  );

  return (
    <ChartCard
      title="Pareto frontier — latency vs success"
      subtitle="Lower latency and higher success are better. APRR dominates all non-oracle baselines."
    >
      <ScatterChart margin={{ top: 8, right: 12, left: -12, bottom: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="latency"
          name="latency"
          stroke={AXIS}
          tick={{ fontSize: 11 }}
          domain={[100, 600]}
          label={{ value: "mean latency (ms)", position: "insideBottom", offset: -2, fill: AXIS, fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="success"
          name="success"
          stroke={AXIS}
          tick={{ fontSize: 11 }}
          domain={[0.25, 0.95]}
          tickFormatter={(v) => v.toFixed(2)}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          cursor={{ stroke: GRID }}
          formatter={(v: number, name: string) =>
            name === "latency" ? [`${v.toFixed(1)} ms`, "latency"] : [v.toFixed(3), "success"]
          }
          labelFormatter={() => ""}
        />
        <Scatter data={points} isAnimationActive={false}>
          {points.map((p) => (
            <Cell key={p.router} fill={ROUTER_COLORS[p.router]} />
          ))}
          <LabelList
            dataKey="router"
            position="top"
            formatter={(v: string) => LABEL[v] ?? v}
            style={{ fill: "#a1a1aa", fontSize: 10 }}
          />
        </Scatter>
      </ScatterChart>
    </ChartCard>
  );
}

/* ---------- (c) Bar chart: latency by router ---------- */
function LatencyBarChart({ data }: { data: MultiSeed }) {
  const bars = useMemo(
    () =>
      ROUTER_ORDER.filter((r) => data.summary[r]).map((r) => ({
        router: LABEL[r] ?? r,
        key: r,
        latency: data.summary[r].mean_latency_ms.mean,
      })),
    [data]
  );

  return (
    <ChartCard
      title="Mean latency by router"
      subtitle="End-to-end milliseconds per query. APRR trims path length without LLM-router overhead."
    >
      <BarChart data={bars} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="router" stroke={AXIS} tick={{ fontSize: 10 }} interval={0} angle={-12} textAnchor="end" height={48} />
        <YAxis stroke={AXIS} tick={{ fontSize: 11 }} />
        <Tooltip
          contentStyle={tooltipStyle}
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          formatter={(v: number) => [`${v.toFixed(1)} ms`, "latency"]}
        />
        <Bar dataKey="latency" radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {bars.map((b) => (
            <Cell key={b.key} fill={ROUTER_COLORS[b.key]} />
          ))}
          <LabelList
            dataKey="latency"
            position="top"
            formatter={(v: number) => v.toFixed(0)}
            style={{ fill: "#a1a1aa", fontSize: 10 }}
          />
        </Bar>
      </BarChart>
    </ChartCard>
  );
}

/* ---------- (d) Ablation 2x2 small-multiples ---------- */
const ABLATION_META: { param: string; label: string; sym: string }[] = [
  { param: "alpha", label: "Learned affinity", sym: "α" },
  { param: "beta", label: "Semantic prior", sym: "β" },
  { param: "lam", label: "Decay regularisation", sym: "λ" },
  { param: "gamma", label: "Query relevance", sym: "γ" },
];

function AblationMini({
  rows,
  meta,
}: {
  rows: AblationRow[];
  meta: { param: string; label: string; sym: string };
}) {
  const series = rows
    .filter((r) => r.param === meta.param)
    .map((r) => ({ value: r.value, success: r.success_rate }));

  return (
    <div className="rounded-lg border border-line bg-ink-900/40 p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-fg">
          <span className="font-mono text-aprr-soft">{meta.sym}</span> · {meta.label}
        </span>
      </div>
      <div className="mt-2 h-[150px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 6, right: 8, left: -22, bottom: -4 }}>
            <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
            <XAxis
              dataKey="value"
              stroke={AXIS}
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => String(v)}
            />
            <YAxis
              stroke={AXIS}
              tick={{ fontSize: 10 }}
              domain={[0.25, 0.55]}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v: number) => [v.toFixed(3), "success"]}
              labelFormatter={(l) => `${meta.sym} = ${l}`}
            />
            <Line
              type="monotone"
              dataKey="success"
              stroke="#d93025"
              strokeWidth={2.2}
              dot={{ r: 2.5, fill: "#d93025" }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function AblationGrid({ rows }: { rows: AblationRow[] }) {
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold text-fg">
        Hyperparameter ablation — success rate
      </h3>
      <p className="mt-0.5 text-xs text-fg-dim">
        Query relevance (γ) drives the largest gain: success climbs 0.289 → 0.483 as γ: 0 → 1.5.
      </p>
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        {ABLATION_META.map((m) => (
          <AblationMini key={m.param} rows={rows} meta={m} />
        ))}
      </div>
    </div>
  );
}

export function Charts({
  data,
  ablation,
}: {
  data: MultiSeed;
  ablation: AblationRow[];
}) {
  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
      <ConvergenceChart data={data} />
      <ParetoChart data={data} />
      <LatencyBarChart data={data} />
      <AblationGrid rows={ablation} />
    </div>
  );
}
