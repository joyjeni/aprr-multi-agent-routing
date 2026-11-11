import type { AppWorldResults } from "@/lib/appworld";

export function AwEvidence({ aw, focus }: { aw: AppWorldResults; focus: string }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <div className="card overflow-hidden">
          <div className="p-5 pb-3">
            <h3 className="text-sm font-semibold text-fg">Paired comparisons on the 57 tasks</h3>
            <p className="mt-0.5 text-xs text-fg-dim">
              McNemar exact test on per-task success: only the tasks that one side solves and the other does not
              carry information.
            </p>
          </div>
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full min-w-[480px] border-collapse text-sm">
              <thead>
                <tr className="border-y border-line text-left text-xs text-fg-dim">
                  <th className="px-4 py-2.5 font-medium">A vs B</th>
                  <th className="px-4 py-2.5 text-right font-medium">A only</th>
                  <th className="px-4 py-2.5 text-right font-medium">B only</th>
                  <th className="px-4 py-2.5 text-right font-medium">p</th>
                </tr>
              </thead>
              <tbody>
                {aw.paired_mcnemar.map((p) => (
                  <tr key={`${p.a}-${p.b}`} className="border-b border-line/60 last:border-0">
                    <td className="px-4 py-2.5 text-fg">
                      {p.a} <span className="text-fg-dim">vs</span> {p.b}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums">{p.a_only}</td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums">{p.b_only}</td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums text-fg-muted">
                      {p.p.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="border-t border-line px-4 py-3 text-xs text-fg-dim">
            Prompt tokens per task (median), Wilcoxon signed-rank:{" "}
            {aw.wilcoxon_prompt_tokens.map((w, i) => (
              <span key={w.a + w.b}>
                {i > 0 && "; "}
                {w.a} {w.median_a.toLocaleString("en-GB")} vs {w.b} {w.median_b.toLocaleString("en-GB")}, p ={" "}
                {w.p}
              </span>
            ))}
            .
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-semibold text-fg">Reading these numbers</h3>
          <ul className="mt-3 space-y-2.5">
            {aw.caveats.map((c) => (
              <li key={c} className="flex gap-3 text-sm leading-relaxed text-fg-muted">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-fg-dim" />
                <span>{c}</span>
              </li>
            ))}
            <li className="flex gap-3 text-sm leading-relaxed text-fg-muted">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-fg-dim" />
              <span>{aw.context}</span>
            </li>
          </ul>
          <h4 className="mt-5 text-xs font-semibold uppercase tracking-[0.14em] text-fg-dim">Next runs</h4>
          <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-fg-muted">
            {aw.next_runs.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ol>
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="p-5 pb-3">
          <h3 className="text-sm font-semibold text-fg">What the development run supports, by research question</h3>
        </div>
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[820px] border-collapse text-sm">
            <thead>
              <tr className="border-y border-line text-left text-xs text-fg-dim">
                <th className="px-4 py-2.5 font-medium">Question</th>
                <th className="px-4 py-2.5 font-medium">Evidence</th>
                <th className="px-4 py-2.5 font-medium">Status</th>
                <th className="px-4 py-2.5 font-medium">Next</th>
              </tr>
            </thead>
            <tbody>
              {aw.findings.map((f) => {
                const isFocus = f.objective.includes(focus);
                return (
                  <tr
                    key={f.rq}
                    className={`border-b border-line/60 align-top last:border-0 ${isFocus ? "bg-aprr/10" : ""}`}
                  >
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className={`font-mono ${isFocus ? "text-aprr-soft" : "text-fg"}`}>{f.rq}</span>
                      <span className="text-fg-dim"> · {f.objective}</span>
                    </td>
                    <td className="px-4 py-3 text-fg-muted">{f.evidence}</td>
                    <td className="px-4 py-3 text-fg">{f.status}</td>
                    <td className="px-4 py-3 text-xs text-fg-muted">{f.next}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
