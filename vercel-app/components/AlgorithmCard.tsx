import katex from "katex";

interface Eq {
  n: string;
  tex: string;
  note: string;
}

const EQUATIONS: Eq[] = [
  {
    n: "1",
    tex: "P(a_j \\mid a_i, q) \\;\\propto\\; W_{ij}^{\\alpha}\\,\\eta_{ij}^{\\beta}\\,\\psi_j(q)^{\\gamma}",
    note: "Routing policy — softmax-style sampling combining learned affinity, semantic prior, and query relevance.",
  },
  {
    n: "2",
    tex: "\\eta_{ij} = \\cos(e_i, e_j), \\qquad \\psi_j(q) = \\cos(q, e_j)",
    note: "Semantic prior between agent embeddings and query-relevance of agent j to query q.",
  },
  {
    n: "3",
    tex: "W \\;\\leftarrow\\; (1-\\lambda)\\,W \\;+\\; \\kappa\\,\\frac{\\mathbf{1}[\\text{success}]}{L^{2}\\,\\cdot\\,\\text{latency}_{\\text{norm}}}",
    note: "Decay-regularised, success-weighted edge reinforcement applied to the traversed path after every episode.",
  },
  {
    n: "4",
    tex: "\\nabla_{W}\\,\\mathbb{E}[R] \\;=\\; \\mathbb{E}\\!\\left[\\,R\\,\\nabla_{W}\\log P(\\tau)\\,\\right]",
    note: "Closed-form policy-gradient interpretation — the update is REINFORCE-equivalent (Appendix A).",
  },
];

const SYMBOLS: { sym: string; meaning: string }[] = [
  { sym: "W_{ij}", meaning: "Routing affinity from agent i to agent j" },
  { sym: "\\alpha,\\beta,\\gamma", meaning: "Exponents on affinity / prior / relevance" },
  { sym: "\\lambda", meaning: "Decay-regularisation rate" },
  { sym: "L", meaning: "Path length (number of hops)" },
  { sym: "\\kappa", meaning: "Reinforcement gain" },
];

function tex(src: string, displayMode = true) {
  return katex.renderToString(src, {
    displayMode,
    throwOnError: false,
    output: "html",
  });
}

export function AlgorithmCard() {
  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
      <div className="card p-6 lg:col-span-2">
        <h3 className="text-sm font-semibold text-fg">Governing equations</h3>
        <p className="mt-1 text-xs text-fg-dim">
          The router maintains an affinity matrix W ∈ ℝ<sup>n×n</sup> updated by online policy iteration.
        </p>
        <ol className="mt-5 space-y-5">
          {EQUATIONS.map((eq) => (
            <li key={eq.n} className="flex gap-4">
              <span className="mt-1 select-none font-mono text-xs text-fg-dim">
                ({eq.n})
              </span>
              <div className="min-w-0 flex-1">
                <div
                  className="scroll-thin overflow-x-auto py-1 text-fg"
                  dangerouslySetInnerHTML={{ __html: tex(eq.tex) }}
                />
                <p className="mt-1.5 text-xs leading-relaxed text-fg-muted">
                  {eq.note}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>

      <div className="card p-6">
        <h3 className="text-sm font-semibold text-fg">Notation</h3>
        <dl className="mt-4 space-y-3">
          {SYMBOLS.map((s) => (
            <div key={s.sym} className="flex items-baseline gap-3">
              <dt
                className="shrink-0 text-fg"
                dangerouslySetInnerHTML={{ __html: tex(s.sym, false) }}
              />
              <dd className="text-xs leading-relaxed text-fg-muted">
                {s.meaning}
              </dd>
            </div>
          ))}
        </dl>
        <div className="mt-6 rounded-lg border border-aprr/30 bg-aprr/5 p-4">
          <p className="text-xs leading-relaxed text-fg-muted">
            The cost-quadratic deposit{" "}
            <span className="font-mono text-aprr-soft">∝ 1/L²</span> makes
            shorter call sequences exponentially preferred — distinct from prior
            work that uses 1/L or constant deposits.
          </p>
        </div>
      </div>
    </div>
  );
}
