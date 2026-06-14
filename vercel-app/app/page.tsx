import { getMultiSeed, getAblation } from "@/lib/data";
import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { Section } from "@/components/Section";
import { ResultsTable } from "@/components/ResultsTable";
import { Charts } from "@/components/Charts";
import { Figures } from "@/components/Figures";
import { AlgorithmCard } from "@/components/AlgorithmCard";
import { Reproduce } from "@/components/Reproduce";
import { Logo } from "@/components/Logo";

export default function Home() {
  const data = getMultiSeed();
  const ablation = getAblation();

  return (
    <>
      <Nav />
      <main>
        <Hero />

        <div className="hairline" />
        <Section
          id="results"
          eyebrow="Benchmark"
          title="Main results across six routers"
          intro="Aggregated over 5 seeds. Success rate is reported with 95% confidence intervals; latency and hops are per-query means. The APRR row is highlighted."
        >
          <ResultsTable data={data} />
        </Section>

        <div className="hairline" />
        <Section
          id="charts"
          eyebrow="Interactive"
          title="Live charts from the benchmark JSON"
          intro="Every series is rendered directly from the embedded results — convergence, the latency–success Pareto frontier, latency by router, and the hyperparameter ablation."
        >
          <Charts data={data} ablation={ablation} />
        </Section>

        <div className="hairline" />
        <Section
          id="figures"
          eyebrow="Manuscript"
          title="Publication figures"
          intro="High-resolution figures from the paper, with vector PDFs for IEEE double-column submission."
        >
          <Figures />
        </Section>

        <div className="hairline" />
        <Section
          id="algorithm"
          eyebrow="Method"
          title="The APRR update, in four equations"
          intro="A softmax-style routing policy with a decay-regularised, success-weighted reinforcement rule — interpretable as online policy iteration with a closed-form policy gradient."
        >
          <AlgorithmCard />
        </Section>

        <div className="hairline" />
        <Section
          id="reproduce"
          eyebrow="Open science"
          title="Reproduce everything"
          intro="Source, a free-tier reproducible notebook, raw result artifacts, and a JSON API for programmatic access."
        >
          <Reproduce />
        </Section>
      </main>

      <footer className="hairline">
        <div className="section-pad flex flex-col gap-4 py-10 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2.5">
            <Logo className="h-5 w-5 text-fg-muted" />
            <span className="text-sm text-fg-muted">
              APRR — Adaptive Probabilistic Routing Reinforcement
            </span>
          </div>
          <div className="text-xs text-fg-dim">
            Jenisha&nbsp;T · MS Ramaiah University of Applied Sciences ·{" "}
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
