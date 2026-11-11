import type { AppWorldResults } from "@/lib/appworld";

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line/60 py-2 last:border-0">
      <span className="text-xs text-fg-dim">{k}</span>
      <span className="text-right text-sm text-fg">{v}</span>
    </div>
  );
}

export function AwOverview({ aw, focus }: { aw: AppWorldResults; focus: string }) {
  const b = aw.benchmark;
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        <div className="card p-5">
          <h3 className="text-sm font-semibold text-fg">Benchmark</h3>
          <div className="mt-3">
            <Fact k="Suite" v={`${b.name} (${b.citation})`} />
            <Fact k="Environment" v={`${b.apps} apps, ${b.apis} APIs`} />
            <Fact k="Split" v={`${b.split}: ${b.tasks} tasks in ${b.scenarios} scenarios`} />
            <Fact k="Graded by" v={b.grading} />
          </div>
        </div>
        <div className="card p-5">
          <h3 className="text-sm font-semibold text-fg">Models</h3>
          <div className="mt-3">
            <Fact k="Large tier" v={aw.models.large} />
            <Fact k="Small tier" v={aw.models.small} />
            <Fact k="Serving" v={aw.models.serving} />
            <Fact k="Worked examples" v={`${b.few_shot} from the train split`} />
          </div>
        </div>
        <div className="card p-5">
          <h3 className="text-sm font-semibold text-fg">Protocol</h3>
          <div className="mt-3">
            <Fact k="Seeds" v={`${b.seeds} (development run)`} />
            <Fact k="Success test" v="McNemar exact, paired by task" />
            <Fact k="Token test" v="Wilcoxon signed-rank, paired" />
            <Fact k="Next" v="test splits, three seeds" />
          </div>
        </div>
      </div>

      <div className="card p-5">
        <h3 className="text-sm font-semibold text-fg">How each objective acts on AppWorld</h3>
        <p className="mt-1 text-xs text-fg-dim">
          Within one step, O1 chooses the API documentation shown, O3 chooses the model, G1 checks the proposed
          code and O4 certifies any state change; O2 rewrites the history after the step. Each part has a switch;
          switched off, it becomes the usual analogue, so one agent gives the reference, the full system and every
          leave-one-out ablation.
        </p>
        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
          {aw.objectives.map((o) => {
            const isFocus = o.id === focus;
            return (
              <div
                key={o.id}
                className={`rounded-lg border p-4 ${
                  isFocus ? "border-aprr/60 bg-aprr/10" : "border-line bg-ink-900/40"
                }`}
              >
                <div className="flex items-baseline gap-2">
                  <span className={`font-mono text-sm font-semibold ${isFocus ? "text-aprr-soft" : "text-fg"}`}>
                    {o.id}
                  </span>
                  <span className="text-sm font-medium text-fg">{o.name}</span>
                  {isFocus && <span className="ml-auto text-[11px] text-aprr-soft">this repository</span>}
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-fg-muted">{o.appworld}</p>
              </div>
            );
          })}
        </div>
      </div>

      <p className="text-xs text-fg-dim">
        {aw.privacy} Source: {aw.source}.
      </p>
    </div>
  );
}
