import type { AppWorldResults } from "@/lib/appworld";
import { cfg } from "@/lib/appworld";

const CONTEXT = [
  { k: "Last step failed", v: "the previous step raised an error", risky: true },
  { k: "Write likely", v: "the top APIs O1 retrieves for the task include a state-changing (non-GET) call", risky: true },
  { k: "Context lost", v: "O2 had to cut an output that no live variable still holds", risky: true },
  { k: "Late in the episode", v: "step 15 or later", risky: false },
];

export function AwRouting({ aw }: { aw: AppWorldResults }) {
  const full = cfg(aw, "full");
  const ref = cfg(aw, "reference");
  const finding = aw.findings.find((f) => f.objective === "O3");
  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
      <div className="card p-5 lg:col-span-3">
        <h3 className="text-sm font-semibold text-fg">The router on AppWorld</h3>
        <p className="mt-2 text-sm leading-relaxed text-fg-muted">
          Before each step a two-arm contextual Thompson sampler chooses the small or the large model. It keeps a
          Beta posterior per arm and context, so it learns online which model writes steps that run cleanly. Four
          binary features describe the context:
        </p>
        <ul className="mt-3 space-y-2">
          {CONTEXT.map((c) => (
            <li key={c.k} className="flex gap-3 text-sm">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-aprr-soft" />
              <span>
                <span className="text-fg">{c.k}</span>
                <span className="text-fg-muted"> — {c.v}</span>
                {c.risky && <span className="ml-1.5 text-[11px] text-aprr-soft">risky</span>}
              </span>
            </li>
          ))}
        </ul>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-line bg-ink-900/40 p-3">
            <div className="font-mono text-sm text-aprr-soft">δ = 0.35</div>
            <div className="mt-1 text-xs text-fg-muted">
              risk-signed margin: in a risky context the 4B model must beat the 8B model by this much before it is chosen
            </div>
          </div>
          <div className="rounded-lg border border-line bg-ink-900/40 p-3">
            <div className="font-mono text-sm text-aprr-soft">λ = 0.15</div>
            <div className="mt-1 text-xs text-fg-muted">cost penalty on the 8B model</div>
          </div>
          <div className="rounded-lg border border-line bg-ink-900/40 p-3">
            <div className="font-mono text-sm text-aprr-soft">reward</div>
            <div className="mt-1 text-xs text-fg-muted">
              1 if the step ran without error, G1 allowed it and, if it changed state, O4 committed it; otherwise 0
            </div>
          </div>
        </div>
      </div>

      <div className="card p-5 lg:col-span-2">
        <h3 className="text-sm font-semibold text-fg">What the first run shows</h3>
        <dl className="mt-3 space-y-3 text-sm">
          <div>
            <dt className="text-xs text-fg-dim">Steps on the 4B model (full ACRS)</dt>
            <dd className="font-mono text-fg">
              {full.small_steps} of {(full.small_steps ?? 0) + (full.large_steps ?? 0)} ·{" "}
              {(100 - (full.large_share_pct ?? 0)).toFixed(1)}%
            </dd>
          </div>
          <div>
            <dt className="text-xs text-fg-dim">Calls to the 8B model</dt>
            <dd className="font-mono text-fg">
              {full.large_steps} against {ref.large_steps} for the reference agent
            </dd>
          </div>
          <div>
            <dt className="text-xs text-fg-dim">Task success</dt>
            <dd className="font-mono text-fg">
              {full.tgc_pct}% against {ref.tgc_pct}% (7 tasks against 8, p = 1.0)
            </dd>
          </div>
          <div>
            <dt className="text-xs text-fg-dim">Status of RQ3</dt>
            <dd className="text-fg">{finding?.status}</dd>
          </div>
        </dl>
        <p className="mt-4 rounded-lg border border-line bg-ink-900/40 p-3 text-xs leading-relaxed text-fg-muted">
          Part of the drop in 8B calls comes from shorter episodes, not only from routing. The ablation without O3
          separates the two, and the per-step logs will show whether risky steps went to the 8B model, which is
          the risk-signed part of the claim. {finding?.next}
        </p>
      </div>
    </div>
  );
}
