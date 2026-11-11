import { getMultiSeed, getAblation, getPipelineResults } from "@/lib/data";
import { getAppWorld } from "@/lib/appworld";
import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { Section } from "@/components/Section";
import { AwOverview } from "@/components/AwOverview";
import { AwResultsTable } from "@/components/AwResultsTable";
import { AwCharts } from "@/components/AwCharts";
import { AwRouting } from "@/components/AwRouting";
import { AwEvidence } from "@/components/AwEvidence";
import { ResultsTable } from "@/components/ResultsTable";
import { Charts } from "@/components/Charts";
import { Figures } from "@/components/Figures";
import { AlgorithmCard } from "@/components/AlgorithmCard";
import { Pipeline } from "@/components/Pipeline";
import { Reproduce } from "@/components/Reproduce";
import { Logo } from "@/components/Logo";

export default function Home() {
  const aw = getAppWorld();
  const data = getMultiSeed();
  const ablation = getAblation();
  const pipeline = getPipelineResults();

  return (
    <>
      <Nav />
      <main>
        <Hero aw={aw} />

        <div className="hairline" />
        <Section
          id="appworld"
          eyebrow={`Latest · AppWorld · updated ${aw.updated}`}
          title="ACRS on AppWorld: development split, first full run"
          intro="All four objectives and the policy check run in one agent loop with open-weight models on free GPUs, scored by AppWorld's own unit tests. Each configuration switches one part off, so every difference below is attributable to that part."
        >
          <AwOverview aw={aw} focus="O3" />
        </Section>

        <div className="hairline" />
        <Section
          id="aw-results"
          eyebrow="AppWorld · results"
          title="Outcome and cost of every configuration"
          intro="57 development tasks, one seed. The full-ACRS row is shaded."
        >
          <div className="space-y-5">
            <AwResultsTable aw={aw} />
            <AwCharts aw={aw} />
          </div>
        </Section>

        <div className="hairline" />
        <Section
          id="o3"
          eyebrow="This repository · O3"
          title="Risk-weighted policy routing inside ACRS"
          intro="The AppWorld form of APRR: per-step choice between two model tiers, biased to the large model when a step is risky, rewarded only by steps that actually work."
        >
          <AwRouting aw={aw} />
        </Section>

        <div className="hairline" />
        <Section
          id="evidence"
          eyebrow="AppWorld · statistics"
          title="Paired tests and what the run supports"
          intro="Reported as they came out, including the results that are not significant."
        >
          <AwEvidence aw={aw} focus="O3" />
        </Section>

        <div id="earlier" className="scroll-mt-20 border-y border-line bg-ink-800/40">
          <div className="section-pad py-12">
            <span className="eyebrow">Earlier experiments · June to October 2026</span>
            <h2 className="mt-2 max-w-3xl text-2xl font-semibold tracking-tight sm:text-3xl">
              APRR on simulated ToolBench workflows
            </h2>
            <p className="mt-3 max-w-3xl text-base leading-relaxed text-fg-muted">
              Before AppWorld, APRR was developed and tested on a simulated multi-agent workflow over ToolBench
              queries (6 routers, 5 seeds, 500 queries × 40 iterations), where it reduced mean latency by 35.7% and
              hops by 23.9% against a static semantic router. Those agents were simulated, not language models; the
              AppWorld results above are the current evidence. In the thesis numbering this work is objective O3,
              and the session-aware reranker it was paired with is O1 (both were called Objective 1 and Objective 2
              in the original notebooks).
            </p>
          </div>
        </div>

        <Section
          id="results"
          eyebrow="Earlier · benchmark"
          title="Six routers on the simulated workflow"
          intro="Aggregated over 5 seeds. Success rate is reported with 95% confidence intervals; latency and hops are per-query means. The APRR row is shaded."
        >
          <ResultsTable data={data} />
        </Section>

        <div className="hairline" />
        <Section
          id="charts"
          eyebrow="Earlier · interactive"
          title="Charts from the benchmark JSON"
          intro="Convergence, the latency–success trade-off, latency by router, and the hyperparameter ablation, rendered from the result files."
        >
          <Charts data={data} ablation={ablation} />
        </Section>

        <div className="hairline" />
        <Section
          id="figures"
          eyebrow="Earlier · manuscript figures"
          title="Affinity learning and per-split results"
          intro="The two manuscript figures not covered by the charts above, redrawn from their data."
        >
          <Figures />
        </Section>

        <div className="hairline" />
        <Section
          id="algorithm"
          eyebrow="Earlier · method"
          title="The APRR update, in four equations"
          intro="A softmax-style routing policy with a decay-regularised, success-weighted reinforcement rule, interpretable as online policy iteration."
        >
          <AlgorithmCard />
        </Section>

        <div className="hairline" />
        <Section
          id="pipeline"
          eyebrow="Earlier · O1 → O3"
          title="Session-aware reranking feeding multi-agent routing"
          intro="A Kaggle notebook chains the session-aware ToolBench reranker (O1) into the APRR, CTGR and FTDR routers (O3) with online reinforcement on real ToolBench evaluation queries."
        >
          <Pipeline data={pipeline} />
        </Section>

        <div className="hairline" />
        <Section
          id="reproduce"
          eyebrow="Open science"
          title="Reproduce everything"
          intro="Source code, notebooks, the aggregate result files behind every number on this page, and a JSON API."
        >
          <Reproduce />
        </Section>
      </main>

      <footer className="hairline">
        <div className="section-pad flex flex-col gap-4 py-10 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2.5">
            <Logo className="h-5 w-5 text-fg-muted" />
            <span className="text-sm text-fg-muted">APRR · objective O3 of ACRS</span>
          </div>
          <div className="text-xs text-fg-dim">
            Jenisha&nbsp;T · Ramaiah University of Applied Sciences ·{" "}
            <a
              href="https://github.com/joyjeni/aprr-multi-agent-routing"
              target="_blank"
              rel="noreferrer"
              className="text-fg-muted underline-offset-2 hover:text-fg hover:underline"
            >
              github.com/joyjeni
            </a>
          </div>
        </div>
      </footer>
    </>
  );
}
