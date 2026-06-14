# Peer Review — MIT CSAIL Perspective

**Manuscript:** Adaptive Probabilistic Routing Reinforcement: Online Policy Iteration for Dynamic Agent-to-Agent Routing in Tool-Augmented LLM Workflows

**Reviewer:** Senior Principal Investigator, Computer Science and Artificial Intelligence Laboratory (CSAIL), MIT  
Expertise: LLM Routing Systems, Systems Benchmarking, Reproducibility in ML, Inference-Time Optimisation

**Review Round:** 1  
**Venue Target:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

---

## Summary

This paper presents APRR, a three-factor multiplicative routing policy combined with a decay-regularised online affinity update, targeting the problem of dynamic agent-to-agent routing in tool-augmented multi-agent LLM workflows. The authors construct a ToolBench-style deterministic simulator and compare APRR against five baselines (Random, RoundRobin, StaticSemantic, LLMRouter, Oracle) across a 5-seed protocol, reporting a 35.7% mean latency reduction and 23.9% hop reduction relative to the best static baseline. The paper has genuine practical relevance: the routing bottleneck in multi-agent LLM systems is real and insufficiently studied. However, from a systems benchmarking and reproducibility standpoint, the evaluation is built almost entirely on a synthetic simulator whose fidelity to real deployments is unvalidated, the latency model is almost certainly unrealistic, the LLMRouter baseline is inadequately described, and multiple key numbers in the paper do not reconcile with the raw JSON data. I recommend Major Revision with the understanding that the simulation-only scope is the paper's most fundamental limitation for a TNNLS audience.

---

## Strengths

- **Practically motivated problem.** Dynamic routing across heterogeneous agent pools is an underexplored systems-level problem in the LLM inference literature. The formulation as an online policy-iteration problem over a routing-affinity matrix is elegant and computationally cheap — O(n²) per episode with n ≤ ~100 is deployable in real systems without a dedicated accelerator.

- **Concise algorithm specification.** The pseudocode in §III is unambiguous and complete; a reader can reimplement APRR from the paper without the released code. This is not always the case in the systems routing literature.

- **JSON artifact transparency.** The authors provide multiseed.json and ablation.json as reproducibility artifacts, allowing independent verification of reported numbers. This level of data transparency is above average for this class of paper.

- **Multi-dimensional performance reporting.** Reporting mean, p50, and p95 latency alongside success rate and mean hops in Table I captures the tail-latency behaviour that is operationally critical for deployed services. Most routing papers report only mean performance.

- **Query-complexity stratification.** Separating results by G1/G2/G3 query complexity (§V.D, Fig. 4) is methodologically sound: a single aggregate metric conceals the fact that APRR's advantage is concentrated in G1 (single-step) queries, which is important for understanding the algorithm's actual deployment profile.

- **Explicit threat-to-validity section.** The paper enumerates external, internal, and construct validity threats (§VI.D). This is unusual and commendable; most ML papers omit this analysis entirely.

---

## Weaknesses

- **The latency model is a proxy, not a measurement.** The simulator assigns latency as a deterministic function of path length, normalised by a fixed 200 ms constant. Real API call latencies are stochastic, heavy-tailed, correlated with query content, and subject to cold-start, rate-limiting, and network partitioning effects. The reported p95 APRR latency of 525.8 ms (std ≈ 187.8 ms across seeds) is driven by hop count variance, not by any realistic model of execution time. Every latency number in the paper must be labelled as "simulated" throughout — currently they are presented as if measured on a real system.

- **LLMRouter baseline is under-specified.** LLMRouter is described as "REINFORCE-trained softmax policy" in one line of Table IV. No architecture, training budget, input representation, or reward design is given. LLMRouter achieves the lowest latency (151.0 ms, mean hops 2.07) by collapsing to near-uniform 2-hop routing (confirmed in multiseed.json: median hops = 2.0 from iteration 3 onward), suggesting it has converged to a degenerate policy that ignores query content. This is a critical baseline: if LLMRouter is implemented carelessly (e.g., insufficient exploration), it becomes a strawman rather than a fair competitor. The paper must fully specify LLMRouter's architecture, hyperparameters, and training protocol.

- **Discrepancy between paper text and JSON data for the ablation.** The paper (§V.E, ablation table) reports α=2.0 → success 0.421, latency 279.3 ms. The ablation.json confirms success 0.421 and latency 279.3 ms. However, the paper also reports γ=1.5 → success 0.483, latency 361.9 ms, while ablation.json shows success 0.4828 and latency 361.9 ms — consistent. But γ=0.0 in the paper reads success 0.289, latency 306.6 ms, while ablation.json gives success 0.2889 and latency 306.6 ms. These are rounded consistently but the ablation protocol is not stated: the multiseed.json full-run APRR mean is 0.470, while the best γ in the ablation is 0.483. This implies the ablation was run with *different* hyperparameters, iteration counts, or seeds than the main experiment — a methodological inconsistency that must be resolved and explicitly documented.

- **Five seeds are insufficient for high-variance methods.** APRR's mean latency standard deviation across seeds is ±36.2 ms (from multiseed.json), giving a 95% CI of ±31.7 ms on a mean of 261.3 ms — a coefficient of variation of ~13.8%. For a routing algorithm whose claimed advantage is latency reduction, this degree of variance across only five seeds is concerning. The paper should run at least 20 seeds for latency comparisons, or provide a power analysis justifying why five seeds are sufficient to detect the claimed 35.7% effect.

- **No ablation on agent pool size or distractor count.** All experiments use exactly 10 agents with 2 distractors (20% distractor rate). The paper provides no evidence that APRR's distractor suppression or Pareto dominance generalises to different pool sizes or distractor fractions. For TNNLS, a scalability analysis — even on the simulator — covering n ∈ {5, 10, 20, 50} agents and distractor fractions ∈ {0%, 20%, 40%} is the minimum required to support the paper's claims about the algorithm's practical utility in real agent pools.

- **The success ceiling (Oracle = 0.900, not 1.0) is unexplained.** In an ideal simulator where the Oracle follows ground-truth paths, a ceiling of 0.900 success implies that 10% of correctly-routed queries fail the LCS criterion even with privileged path knowledge. Either the LCS^1.5 criterion is mis-calibrated (the 1.5 exponent systematically rejects near-correct paths), or the ground-truth paths themselves are sometimes infeasible. This 10% irreducible failure rate is never acknowledged or explained, which calls the validity of the entire success metric into question.

---

## Detailed Comments

1. **Table I: APRR success rate (0.470) is *lower* than StaticSemantic (0.499) — the paper obfuscates this.** The abstract states APRR "Pareto-dominates all non-oracle baselines." Pareto dominance requires superiority on *all* objectives simultaneously. APRR achieves 0.470 success vs. StaticSemantic's 0.499 — APRR is *worse* on success rate by approximately 2.9 pp (comparing to the point estimates). The paper defends this via Pareto-front analysis (Fig. 5), arguing that no other method achieves >0.46 AND <300 ms simultaneously. This is a valid argument for Pareto optimality, not Pareto dominance. The abstract and §V must be reworded to accurately characterise APRR as Pareto-optimal (lying on the frontier) rather than Pareto-dominant (strictly superior on all dimensions), to avoid misleading readers.

2. **Section IV, Success Criterion (Eq. 5): The b̄(π) term is undefined.** The success criterion reads P(success | π, π*) = (LCS(π, π*)/|π*|)^1.5 · 1[terminal] · b̄(π). The paper never defines b̄(π). This is an incomplete specification of the evaluation criterion. If b̄(π) is a binary indicator (e.g., path budget not exceeded), it must be stated. If it is a soft penalty, its form must be given.

3. **Section IV, Protocol: The 500-query evaluation set is fixed across all 40 iterations.** The paper states "5 seeds × 40 iterations × 500 queries = 100,000 episodes per router." If the same 500 queries are evaluated in every iteration (i.e., the query distribution is identical across iterations), this is not independent evaluation: APRR's affinity matrix is being updated on the same queries it is tested on, creating an online overfitting regime. The paper must clarify whether queries are sampled fresh per iteration or reused; if reused, a held-out test set is required to measure generalisation.

4. **Section V, LLMRouter convergence (multiseed.json inspection): LLMRouter reaches 2.0 median hops by iteration 3 and never recovers.** The convergence data in multiseed.json shows that LLMRouter's mean hops collapses from 3.45 at iteration 0 to exactly 2.0 by iteration 3 and remains constant through iteration 39. This is a textbook mode collapse in a policy-gradient method — the policy has saturated to deterministic two-hop routing. The authors characterise this as "LLMRouter collapses to ≈2 hops regardless of query complexity" (§V.F) but treat it as a baseline property rather than a training failure. A properly trained LLMRouter with appropriate entropy regularisation should not exhibit this collapse; as implemented, it is not a fair upper-bound for the REINFORCE class.

5. **Section III, Hyperparameter Table: γ default = 2.5, but best ablation value = 1.5.** The hyperparameter table (§III) lists γ = 2.5 as the default value used in the main experiment. The ablation (Fig. 7) sweeps γ ∈ {0.0, 0.5, 1.0, 1.5} and reports γ = 1.5 as best. The default of γ = 2.5 is *not included in the ablation sweep*. The main result is therefore obtained with an un-ablated hyperparameter value. The paper must extend the ablation to include γ = 2.5 and justify why a value outside the ablated range was chosen as the default, or revise the main experiment to use γ = 1.5.

6. **Section V, Pareto Front (Fig. 5): The frontier is constructed from aggregate statistics, not from individual query traces.** Mean latency vs. mean success rate are aggregate statistics computed over 20,000 queries per router. The Pareto front over aggregate means does not imply Pareto efficiency over individual query distributions — a router that excels on G1 (50% of queries) but degrades severely on G3 (20%) can appear on the Pareto front of aggregates while being strictly dominated on the G3 subpopulation. Per-split Pareto analysis should be provided.

7. **Section II, Related Work: Citations dated 2026 are unverifiable.** DyTopo [2026], DMoA [2026], NeuroMAS [2026], FlowBank [2026], MetaCogAgent [2026], and MAPoRL [2026] are cited as positioning references but provide placeholder arXiv IDs ("arXiv:2602.xxxxx"). If these are intended as future or concurrent work that predates the submission, the authors must provide working DOIs or arXiv links. If these papers do not exist, the entire competitive positioning in §II is unverifiable and possibly fabricated.

8. **Section IV, LCS Criterion: The 1.5 exponent introduces nonlinearity that has system-level implications.** An LCS ratio of 0.9 (90% of ground-truth steps matched) yields a score of 0.9^1.5 ≈ 0.854; a ratio of 0.7 yields 0.7^1.5 ≈ 0.586. This superlinear penalty means that a single missing step from a 10-hop path drops the success probability by 41 pp. This creates a systematic bias *against* shorter paths (which have fewer opportunities to accumulate LCS matches) and in favour of longer, more complete traversals — which is precisely the opposite of what the paper claims to optimise. The criterion should be validated against human preference judgements on a sample of ToolBench tasks, or replaced with the standard binary pass_rate used in the original ToolBench benchmark.

9. **Section VI, Discussion (λ): The "optimal at λ=0.1" claim contradicts the ablation data.** §VI states "Optimal at λ=0.1 (success 0.469)." The ablation.json confirms lam=0.1 → success 0.4689. However, the full multiseed experiment reports APRR success of 0.470 with λ=0.005 (the default). This means the ablation with λ=0.1 slightly underperforms the default λ=0.005. The text should be rewritten: λ=0.005 (default) achieves the best success rate in the full protocol; λ=0.1 achieves a similar rate but with higher latency (313.8 ms vs. 261.3 ms). The claim that λ=0.1 is "optimal" is unsupported and internally inconsistent.

10. **Reproducibility: The Colab/Kaggle notebooks are referenced but no commit hash or frozen environment is provided.** The paper links to a GitHub repository but specifies no commit hash, environment snapshot (requirements.txt, conda env), or Docker image. Given that the LLM embedding models used to compute role embeddings (η_{ij}) and query relevances (ψ_j) are not specified anywhere in the paper, full reproduction is impossible without knowing the embedding model, its version, and whether it was accessed via API or local weights. This is a critical reproducibility gap.

---

## Recommendation

**Major Revision**

The problem is well-motivated and APRR is a technically coherent, computationally lightweight algorithm. The transparency of the JSON data artifacts is genuinely commendable. However, the evaluation is entirely simulation-based with an unrealistic latency model; the LLMRouter baseline is under-specified and likely exhibits training failure; the γ default is outside the ablated range; the Pareto-dominance claim overstates the results; b̄(π) in the success criterion is undefined; and the 2026 forward-dated citations are unverifiable. These are not minor polish issues — they collectively undermine the paper's empirical validity. A revised submission should include at minimum: (a) at least one experiment on real ToolBench REST API calls with real latency measurements, (b) a fully specified LLMRouter baseline, (c) the γ=2.5 ablation point, (d) resolution of the 2026 citation issue, and (e) a definition of b̄(π).

| Criterion | Score (1–10) |
|-----------|-------------|
| Novelty | 6 |
| Technical Soundness | 5 |
| Clarity | 6 |
| Reproducibility | 5 |

**Overall Score: 5.5 / 10 — Major Revision**
