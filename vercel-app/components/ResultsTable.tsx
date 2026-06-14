"use client";

import { useMemo, useState } from "react";
import type { MultiSeed, RouterSummary } from "@/lib/types";
import { ROUTER_ORDER } from "@/lib/types";

interface Row {
  router: string;
  success: number;
  successCi: number;
  latency: number;
  latencyCi: number;
  hops: number;
  p95: number;
}

type SortKey = "router" | "success" | "latency" | "hops" | "p95";

function buildRows(summary: Record<string, RouterSummary>): Row[] {
  return ROUTER_ORDER.filter((r) => summary[r]).map((r) => {
    const s = summary[r];
    return {
      router: r,
      success: s.success_rate.mean,
      successCi: s.success_rate.ci95,
      latency: s.mean_latency_ms.mean,
      latencyCi: s.mean_latency_ms.ci95,
      hops: s.mean_hops.mean,
      p95: s.p95_latency_ms.mean,
    };
  });
}

const LABEL: Record<string, string> = {
  Random: "Random",
  RoundRobin: "Round-Robin",
  StaticSemantic: "Static Semantic",
  LLMRouter: "LLM Router",
  APRR: "APRR (ours)",
  Oracle: "Oracle",
};

export function ResultsTable({ data }: { data: MultiSeed }) {
  const rows = useMemo(() => buildRows(data.summary), [data]);
  const [sortKey, setSortKey] = useState<SortKey>("success");
  const [asc, setAsc] = useState(false);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      if (sortKey === "router") {
        return asc
          ? a.router.localeCompare(b.router)
          : b.router.localeCompare(a.router);
      }
      const av = a[sortKey] as number;
      const bv = b[sortKey] as number;
      return asc ? av - bv : bv - av;
    });
    return copy;
  }, [rows, sortKey, asc]);

  function toggle(key: SortKey) {
    if (key === sortKey) setAsc(!asc);
    else {
      setSortKey(key);
      setAsc(key === "latency" || key === "hops" || key === "p95");
    }
  }

  const head: { key: SortKey; label: string; right?: boolean }[] = [
    { key: "router", label: "Router" },
    { key: "success", label: "Success ± 95% CI", right: true },
    { key: "latency", label: "Mean latency (ms)", right: true },
    { key: "p95", label: "P95 latency (ms)", right: true },
    { key: "hops", label: "Mean hops", right: true },
  ];

  return (
    <div className="card overflow-hidden">
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-line text-left text-fg-dim">
              {head.map((h) => (
                <th
                  key={h.key}
                  className={`px-4 py-3 font-medium ${h.right ? "text-right" : ""}`}
                >
                  <button
                    onClick={() => toggle(h.key)}
                    className={`inline-flex items-center gap-1 transition-colors hover:text-fg ${
                      sortKey === h.key ? "text-fg" : ""
                    } ${h.right ? "flex-row-reverse" : ""}`}
                  >
                    {h.label}
                    <span className="text-[10px]">
                      {sortKey === h.key ? (asc ? "▲" : "▼") : "↕"}
                    </span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const isAprr = r.router === "APRR";
              const isOracle = r.router === "Oracle";
              return (
                <tr
                  key={r.router}
                  className={`border-b border-line/60 last:border-0 transition-colors ${
                    isAprr
                      ? "bg-aprr/10"
                      : "hover:bg-ink-700/40"
                  }`}
                >
                  <td className="px-4 py-3 font-medium">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{
                          background: isAprr
                            ? "#d93025"
                            : isOracle
                            ? "#1e9e6a"
                            : "#1a73e8",
                        }}
                      />
                      <span className={isAprr ? "text-aprr-soft" : "text-fg"}>
                        {LABEL[r.router] ?? r.router}
                      </span>
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums">
                    {r.success.toFixed(3)}
                    <span className="text-fg-dim">
                      {" "}
                      ± {r.successCi.toFixed(3)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums">
                    {r.latency.toFixed(1)}
                    <span className="text-fg-dim"> ± {r.latencyCi.toFixed(1)}</span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums text-fg-muted">
                    {r.p95.toFixed(1)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono tabular-nums">
                    {r.hops.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-line px-4 py-3 text-xs text-fg-dim">
        <span>
          Aggregated over seeds {data.seeds.join(", ")} · {data.n_queries}{" "}
          queries × {data.n_iterations} iterations.
        </span>
        <span>Click a column header to sort.</span>
      </div>
    </div>
  );
}
