"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";
import type { PipelineResults } from "@/lib/types";
import { PIPELINE_COLORS } from "@/lib/types";

const AXIS = "#6b6b73";
const GRID = "#26262c";

const SHORT_LABEL: Record<string, string> = {
  Random: "Random",
  StaticSemantic: "Static Semantic",
  "APRR-core (online, no Obj1 signal)": "APRR-core",
  "CTGR (online, no Obj1 signal)": "CTGR",
  "FTDR (online, no Obj1 signal)": "FTDR",
  "APRR-core + Obj1 fusion (online)": "APRR-core + O1",
  "CTGR + Obj1 fusion (online)": "CTGR + O1",
  "FTDR + Obj1 fusion (online)": "FTDR + O1",
};

const tooltipStyle = {
  background: "#121214",
  border: "1px solid #26262c",
  borderRadius: 8,
  fontSize: 12,
  color: "#f5f5f7",
};

export function Pipeline({ data }: { data: PipelineResults }) {
  const chartRows = useMemo(
    () =>
      data.results.map((r) => ({
        router: SHORT_LABEL[r.router] ?? r.router,
        fullName: r.router,
        success_rate: r.success_rate,
      })),
    [data]
  );

  const obj1 = data.objective1_validation;

  return (
    <div className="space-y-6">
      {/* Stage 1: Objective 1 hand-off */}
      <div className="card p-5">
        <h3 className="text-sm font-semibold text-fg">
          Stage 1 — O1 session-aware reranking (hand-off signal)
        </h3>
        <p className="mt-1 text-xs text-fg-dim">
          Computed live from{" "}
          <a
            href={data.objective1_repo}
            target="_blank"
            rel="noreferrer"
            className="text-base-soft underline-offset-2 hover:underline"
          >
            session-aware-toolbench-rerank
          </a>{" "}
          on {obj1.n_queries} real ToolBench eval queries across{" "}
          {obj1.n_sessions} synthetic sessions. O1&apos;s top-1
          session-reranked category per query is the signal handed to the
          O3 routers below (named Objective 1 and Objective 2 in the notebook).
        </p>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-md border border-line bg-ink-900/40 p-4">
            <div className="text-xs font-medium text-fg-dim">BM25 only</div>
            <div className="mt-2 flex gap-4 font-mono text-sm tabular-nums text-fg">
              <span>Hit@1 {obj1.BM25_only["Hit@1"].toFixed(4)}</span>
              <span>Hit@5 {obj1.BM25_only["Hit@5"].toFixed(4)}</span>
              <span>NDCG@5 {obj1.BM25_only["NDCG@5"].toFixed(4)}</span>
            </div>
          </div>
          <div className="rounded-md border border-aprr/30 bg-aprr/10 p-4">
            <div className="text-xs font-medium text-aprr-soft">
              BM25 + SessionRerank
            </div>
            <div className="mt-2 flex gap-4 font-mono text-sm tabular-nums text-fg">
              <span>
                Hit@1 {obj1.BM25_plus_SessionRerank["Hit@1"].toFixed(4)}
              </span>
              <span>
                Hit@5 {obj1.BM25_plus_SessionRerank["Hit@5"].toFixed(4)}
              </span>
              <span>
                NDCG@5 {obj1.BM25_plus_SessionRerank["NDCG@5"].toFixed(4)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Stage 2: online routing chart */}
      <div className="card p-5">
        <h3 className="text-sm font-semibold text-fg">
          Stage 2 — O3 online routing ({data.n_eval_queries} real
          ToolBench eval queries, seed 42)
        </h3>
        <p className="mt-1 text-xs text-fg-dim">
          Each router routes → is scored against ground truth → calls its
          own real <code className="font-mono">update()</code> to reinforce
          its affinity matrix (genuine prequential online learning, not a
          frozen inference pass). Agent pool: {data.categories.join(", ")}.
        </p>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: "#3987e5" }} />
            with the O1 fusion signal
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: "#a1a1aa" }} />
            without it
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: "#5b5b63" }} />
            fixed baselines
          </span>
        </div>
        <div className="mt-4 h-[340px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartRows}
              margin={{ top: 24, right: 8, left: -16, bottom: 64 }}
            >
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis
                dataKey="router"
                stroke={AXIS}
                tick={{ fontSize: 10 }}
                angle={-30}
                textAnchor="end"
                interval={0}
              />
              <YAxis
                stroke={AXIS}
                tick={{ fontSize: 11 }}
                domain={[0, 1]}
                tickFormatter={(v) => v.toFixed(1)}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v: number) => v.toFixed(4)}
                labelFormatter={(_, payload) =>
                  payload?.[0]?.payload?.fullName ?? ""
                }
              />
              <Bar dataKey="success_rate" radius={[4, 4, 0, 0]}>
                {chartRows.map((row) => (
                  <Cell
                    key={row.fullName}
                    fill={PIPELINE_COLORS[row.fullName] ?? "#a1a1aa"}
                  />
                ))}
                <LabelList
                  dataKey="success_rate"
                  position="top"
                  formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                  style={{ fontSize: 10, fill: "#a1a1aa" }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Results table */}
      <div className="card overflow-hidden">
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[720px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left text-fg-dim">
                <th className="px-4 py-3 font-medium">Router</th>
                <th className="px-4 py-3 text-right font-medium">
                  Success rate
                </th>
                <th className="px-4 py-3 text-right font-medium">
                  1st half
                </th>
                <th className="px-4 py-3 text-right font-medium">
                  2nd half
                </th>
                <th className="px-4 py-3 text-right font-medium">
                  Routing compute (ms)
                </th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((r) => {
                const isFusionCtgr = r.router.startsWith("CTGR + Obj1");
                return (
                  <tr
                    key={r.router}
                    className={`border-b border-line/60 last:border-0 transition-colors ${
                      isFusionCtgr ? "bg-aprr/10" : "hover:bg-ink-700/40"
                    }`}
                  >
                    <td className="px-4 py-3 font-medium">
                      <span className="inline-flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{
                            background:
                              PIPELINE_COLORS[r.router] ?? "#a1a1aa",
                          }}
                        />
                        <span className={isFusionCtgr ? "text-aprr-soft" : "text-fg"}>
                          {r.router}
                        </span>
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums">
                      {r.success_rate.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-fg-muted">
                      {r.success_rate_first_half.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-fg-muted">
                      {r.success_rate_second_half.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-fg-muted">
                      {r.mean_routing_compute_ms.toFixed(4)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="border-t border-line px-4 py-3 text-xs text-fg-dim">
          CTGR is the only router whose success rate visibly changes with
          the O1 fusion signal (0.5576 → 0.6606), because its
          deliberation branch reshuffles the candidate shortlist using the
          boosted scores. APRR-core and FTDR make a single greedy argmax
          decision with no exploration, so once online reinforcement pushes
          an incumbent choice&apos;s score high enough, a single query&apos;s
          fusion boost is mathematically too small to overturn it — a
          documented exploration/exploitation limitation of greedy,
          no-exploration online routers, not a bug being hidden.
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <a
          href="https://github.com/joyjeni/aprr-multi-agent-routing/blob/main/notebooks/APRR_Objective1_Objective2_ToolBench.ipynb"
          target="_blank"
          rel="noreferrer"
          className="rounded-md border border-line bg-ink-900/40 px-3 py-1.5 font-mono text-xs text-fg-muted transition-colors hover:border-aprr hover:text-aprr-soft"
        >
          Open pipeline notebook →
        </a>
        <a
          href="/data/pipeline_results.json"
          target="_blank"
          rel="noreferrer"
          className="rounded-md border border-line bg-ink-900/40 px-3 py-1.5 font-mono text-xs text-fg-muted transition-colors hover:border-base hover:text-base-soft"
        >
          pipeline_results.json
        </a>
      </div>
    </div>
  );
}
