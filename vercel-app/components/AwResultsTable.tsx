import type { AppWorldResults, AwConfig } from "@/lib/appworld";

const HEAD = [
  { label: "Configuration", right: false },
  { label: "Task success (TGC)", right: true },
  { label: "95% interval", right: true },
  { label: "Scenarios (SGC)", right: true },
  { label: "Steps / task", right: true },
  { label: "Tokens (M)", right: true },
  { label: "Steps on 8B", right: true },
  { label: "Collateral damage", right: true },
  { label: "Wall (min)", right: true },
];

function Row({ c, tasks, scen }: { c: AwConfig; tasks: number; scen: number }) {
  const isFull = c.id === "full";
  const cell = "px-3 py-2.5 text-right font-mono tabular-nums";
  if (c.status === "pending") {
    return (
      <tr className="border-b border-line/60 last:border-0">
        <td className="px-3 py-2.5">
          <div className="font-medium text-fg">{c.label}</div>
          <div className="text-[11px] text-fg-dim">{c.description}</div>
        </td>
        <td colSpan={8} className="px-3 py-2.5 text-xs text-fg-dim">
          Scheduled on Kaggle; added here when the run finishes.
        </td>
      </tr>
    );
  }
  return (
    <tr
      className={`border-b border-line/60 last:border-0 ${
        isFull ? "bg-aprr/10" : "hover:bg-ink-700/40"
      }`}
    >
      <td className="px-3 py-2.5">
        <div className={`font-medium ${isFull ? "text-aprr-soft" : "text-fg"}`}>{c.label}</div>
        <div className="text-[11px] text-fg-dim">{c.description}</div>
      </td>
      <td className={cell}>
        {c.tgc_pct?.toFixed(1)}%<span className="text-fg-dim"> ({c.tgc_solved}/{tasks})</span>
      </td>
      <td className={`${cell} text-fg-muted`}>
        {c.tgc_ci95?.[0].toFixed(1)}–{c.tgc_ci95?.[1].toFixed(1)}
      </td>
      <td className={cell}>
        {c.sgc_solved}/{scen}
      </td>
      <td className={cell}>{c.steps_per_task?.toFixed(1)}</td>
      <td className={cell}>{c.tokens_m?.toFixed(2)}</td>
      <td className={cell}>{c.large_share_pct?.toFixed(1)}%</td>
      <td className={cell}>
        {c.collateral_damage_tasks}/{tasks}
      </td>
      <td className={`${cell} text-fg-muted`}>{c.wall_min}</td>
    </tr>
  );
}

export function AwResultsTable({ aw }: { aw: AppWorldResults }) {
  return (
    <div className="card overflow-hidden">
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[960px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-fg-dim">
              {HEAD.map((h) => (
                <th key={h.label} className={`px-3 py-3 font-medium ${h.right ? "text-right" : ""}`}>
                  {h.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {aw.configs.map((c) => (
              <Row key={c.id} c={c} tasks={aw.benchmark.tasks} scen={aw.benchmark.scenarios} />
            ))}
          </tbody>
        </table>
      </div>
      <div className="space-y-1 border-t border-line px-4 py-3 text-xs text-fg-dim">
        <p>
          TGC: tasks whose unit tests all pass. SGC: scenarios (three related tasks) fully solved. 95% interval:
          Wilson score interval for TGC. Collateral damage: tasks in which the agent changed state it was not asked
          to change. Wall time includes the O4 candidate executions.
        </p>
        <p>
          Configuration names in the released harness follow the earlier objective numbering: {" "}
          <span className="font-mono">o4_only</span> is O2 only, <span className="font-mono">no_o4</span> is without
          O2, <span className="font-mono">no_o3</span> is without O4 and <span className="font-mono">no_o2</span> is
          without O3.
        </p>
      </div>
    </div>
  );
}
