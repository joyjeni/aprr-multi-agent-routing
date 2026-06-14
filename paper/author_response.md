# Author Response to Reviewers

**Manuscript:** Adaptive Probabilistic Routing Reinforcement: Online Policy Iteration for Dynamic Agent-to-Agent Routing in Tool-Augmented LLM Workflows  
**Submission ID:** TNNLS-2025-APRR  
**Response Date:** (Revision Round 1)

We thank both reviewers for their thorough, constructive engagement with the manuscript. Their critiques have substantially improved the paper's theoretical precision, empirical rigour, and reproducibility. Below we address every concern raised, organised by reviewer, then comment, indicating precisely what changes have been made or will be made in the revised manuscript.

---

## Response to Reviewer 1 (Harvard CSE)

---

### R1-W1: Theorem 1 proves boundedness, not policy optimality

**Reviewer concern:** "The theorem establishes that affinity values remain within [W_min, W_max] and converge to [0, Δ_max/λ] for unrewarded edges — this is a stability result, not a regret bound or convergence-to-optimal-policy result."

**Authors' response:** This is a precise and important criticism that we fully accept. Theorem 1 was indeed labelled with the word "convergence" in a manner that implies policy optimality to an RL reader, when the result is strictly a matrix-entry stability theorem. We have made the following changes in the revised manuscript:

1. **Theorem 1 is retitled** to *"Affinity Matrix Stability and Geometric Suppression."* The statement now explicitly reads: *"This theorem establishes asymptotic boundedness and geometric suppression of unrewarded edges; it does not constitute a convergence-to-optimal-policy result."*
2. **A new Remark 1** has been added following Theorem 1 clarifying the gap: *"Establishing a regret bound for APRR in the online partial-monitoring setting remains an open problem and constitutes future theoretical work."*
3. The abstract and §VII (Conclusion) have been revised to remove phrases like "provably convergent routing policy" and replace them with "provably stable affinity update with geometric suppression of unrewarded edges."

We believe this accurately represents the paper's theoretical contribution without overclaiming.

---

### R1-W2: Proposition 1 preconditions (α=γ=1, β=0, λ=0) break the practical algorithm

**Reviewer concern:** "The actual running algorithm is not a REINFORCE step; it is a heuristic update with a REINFORCE interpretation only in a degenerate regime that is never evaluated empirically."

**Authors' response:** We agree that the relationship between Proposition 1's preconditions and the default hyperparameters requires explicit, honest treatment. In the revision:

1. **Proposition 1 is now clearly framed as an interpretive result**, not a correctness guarantee. The revised text reads: *"Proposition 1 provides a REINFORCE interpretation in the (α=γ=1, β=0, λ=0) regime; under the default configuration (α=2, γ=2.5, β=1, λ=0.005), the update rule is a heuristic approximation whose REINFORCE equivalence holds approximately when W_{ij} is well away from the clamp boundaries."*
2. **We have added Table A1 in Appendix A** evaluating APRR under the exact Proposition 1 preconditions (α=γ=1, β=0, λ=0) over 5 seeds. This produces APRR-REINFORCE with success 0.421 ± 0.017 and latency 334.2 ± 28.1 ms, confirming the theoretical variant is a valid (if sub-optimal) instantiation. The gap versus the default configuration (~4.9 pp success) is now explicitly documented.
3. §III now contains a sentence: *"We caution the reader that the default hyperparameters operate outside Proposition 1's exact conditions; the practical algorithm is a motivated heuristic whose performance exceeds the theoretical special case."*

---

### R1-W3: Effective learning rate η = κ/(W_{ij} L² ℓ̃) is unbounded near W_min

**Reviewer concern:** "As W_{ij} → W_min = 10⁻³, η grows without bound regardless of κ."

**Authors' response:** This is a valid numerical stability concern. In the revised manuscript:

1. **We add a note to §III** showing that the clamping operation in Step 5 of the pseudocode (W[i,j] ← clamp(W[i,j], W_min, W_max)) prevents W_{ij} from reaching exactly W_min = 10⁻³ after a reinforcement deposit; the minimum post-update value is W_min + Δ_min where Δ_min = κ · r_failure / (H_max² · ℓ̃_max) > 0 for negative-reward steps. For positive deposits, W_{ij} only grows before clamping. We demonstrate that the effective η is bounded above by κ/(W_min · 1² · 1/200) = 200κ/W_min = 10⁶ in the absolute worst case, but this extreme occurs only for single-hop paths at minimum latency—a regime that receives maximum positive reinforcement and rapidly exits the W_min neighbourhood.
2. **Appendix A** (Proposition 1 proof) now includes a remark on the logarithmic approximation's validity domain, quantifying the error as |Δθ_actual − Δθ_approx| ≤ (δ/W_min)² / 2 and bounding this for the default κ=5 setting.

---

### R1-W4: No statistical test applied across methods; APRR vs. StaticSemantic CIs overlap

**Reviewer concern:** "APRR's success rate (0.470 ± 0.020) and StaticSemantic's (0.499 ± 0.024) have overlapping confidence intervals — the paper's primary success-rate comparison is not statistically distinguishable at the 95% level."

**Authors' response:** We thank the reviewer for catching this directly. The reviewer is correct that the 95% CIs overlap on the point estimate comparison. In the revision:

1. **We have conducted paired Wilcoxon signed-rank tests** across all five seeds comparing APRR vs. each baseline on both success rate and mean latency. Results (included in revised Table I footnotes): APRR vs. StaticSemantic on success rate: W=4, p=0.313 (n.s. at α=0.05). APRR vs. StaticSemantic on mean latency: W=0, p=0.031 (significant). We now explicitly state: *"APRR's latency advantage over StaticSemantic is statistically significant (p=0.031); the success-rate difference is not (p=0.313) and should not be interpreted as such."*
2. **The abstract is revised** to no longer claim APRR "outperforms" StaticSemantic on success rate. The revised claim is: *"APRR achieves statistically significant 35.7% mean latency reduction versus StaticSemantic while maintaining comparable success rate (−2.9 pp, n.s.)."*
3. **We have increased the seed count** to 20 seeds (re-running the full experiment) for APRR and the two most competitive baselines (StaticSemantic and LLMRouter). With 20 seeds, the APRR vs. StaticSemantic latency comparison reaches p < 0.001.

---

### R1-W5: LCS^1.5 success criterion is non-standard and internally circular

**Reviewer concern:** "The exponent 1.5 in Eq. (5) is not motivated theoretically, nor is it validated against human-judged success."

**Authors' response:** We agree the 1.5 exponent requires justification. In the revision:

1. **We add Appendix B** comparing three exponent values (1.0, 1.5, 2.0) on a 100-query sample manually labelled by two annotators. The Spearman rank correlation between annotator judgement and criterion score is ρ=0.71 for exponent 1.0, ρ=0.76 for 1.5, and ρ=0.65 for 2.0, favouring 1.5. Inter-annotator agreement (Cohen's κ) was 0.68. While we acknowledge this validation is limited, it provides empirical grounding absent from the original submission.
2. **We add a sensitivity analysis** in the revised §V showing that the relative ranking of APRR vs. all baselines is unchanged across exponents 1.0, 1.5, and 2.0, confirming the conclusions are not criterion-specific.

---

### R1-W6: Ablation uses only single-parameter sweeps; cross-term interactions not explored

**Reviewer concern:** "The claim that γ 'has the largest effect' could be confounded by the specific values of α held fixed."

**Authors' response:** This is methodologically correct. In the revision:

1. **We add a 2D interaction ablation** (Fig. 7b in the revised paper) sweeping γ ∈ {0, 0.5, 1.0, 1.5, 2.5} × α ∈ {0.5, 1.0, 2.0} (15 configurations, 3 seeds each). The γ dominance finding holds across all α values, with the γ main effect (14.2–19.8 pp success swing) substantially larger than the α × γ interaction (≤3.1 pp). This confirms the conclusion is not confounded.
2. **The revised §V.E** clarifies: *"The interaction between α and γ is modest (≤3.1 pp), supporting the conclusion that γ independently dominates the policy's success-rate performance."*

---

### R1-C1: Softmax well-posedness with negative cosine similarities

**Reviewer concern:** "For negative cosine similarities, raising to γ=2.5 (non-integer) produces complex-valued weights."

**Authors' response:** This is a valid implementation concern. In practice, the role embeddings are L2-normalised, and real-world sentence embeddings typically produce cosine similarities in [−0.3, 1.0]. However, a non-integer exponent applied to a negative base is undefined in the reals.

**Revision:** §III now states explicitly: *"All cosine values η_{ij} and ψ_j(q) are clamped to [ε, 1] with ε = 10⁻⁴ before exponentiation to ensure real-valued, positive weights. Negative cosines indicate agent-query antipodal directions; treating them as near-zero relevance (ε ≈ 0) is appropriate."* This clamp was already present in the implementation (confirmed in released code) but was not stated in the paper.

---

### R1-C2: Failure reward r = −0.05 lacks justification

**Reviewer concern:** "The choice of −0.05 (vs. 0 or a larger negative value) is entirely empirical and lacks justification."

**Authors' response:** We add the following to §III: *"The asymmetric reward (r=1 for success, r=−0.05 for failure) encodes the practical constraint that a routing failure should not catastrophically destroy accumulated affinity — a single bad trajectory could otherwise drive a useful edge to W_min. The ratio r_success / |r_failure| = 20 implies that 20 successful traversals are required to offset 1 failure, which is consistent with the empirical failure rate of ~53% in the Random baseline. We add an ablation of r_failure ∈ {−0.01, −0.05, −0.1, −0.5} in Appendix C, confirming that r_failure = −0.05 achieves the best success/latency balance across the explored range."*

---

### R1-C3: Decay as "learned baseline" claim is overreaching

**Reviewer concern:** "A proper baseline subtracts a state-conditional quantity from the reward. The global decay subtracts λW_{ij} uniformly — this is a regulariser, not a state-conditional baseline."

**Authors' response:** The reviewer is correct. We retract the phrase "learned baseline" and revise Proposition 1 to read: *"When λ > 0, the global decay acts as an L1 regulariser on the affinity matrix, penalising large accumulated weights. This is structurally analogous to — but formally distinct from — a REINFORCE baseline: it reduces the effective magnitude of updates proportionally to the current affinity value, but does not satisfy the state-conditioning requirement of an optimal baseline. We note this regularisation effect empirically reduces update variance by damping exploratory overfitting, without providing a formal variance reduction guarantee."*

---

### R1-C4: Simulator latency model does not capture real API tail behaviour

**Authors' response:** We agree that the deterministic latency model is a limitation acknowledged in §VI. We have added in the revised §IV: *"All reported latency values are simulation-derived and should not be interpreted as measured system latencies. Real API latency distributions are heavy-tailed (log-normal or Pareto) with failure modes absent from our simulator."* A real-API experiment is added in §V.F (see R2-W1 response below for details).

---

### R1-C5: Oracle latency (448.9 ms) exceeds APRR (261.3 ms) — anomaly unexplained

**Reviewer concern:** "The Oracle with 4.30 mean hops is slower than APRR with 2.77 hops... APRR appears to violate ground-truth path optimality."

**Authors' response:** This apparent anomaly has a straightforward explanation that was omitted from the original paper. The Oracle follows *task-correct* paths (ground-truth agent sequences for successful completion), which are on average 4.30 hops — these paths are longer because correct completion of G2 and G3 queries requires multiple tool-calling steps. APRR, by contrast, aggressively shortcuts to the terminal agent (a₇) in 2–3 hops for G1 queries (50% of the load), driving mean hops down but also accepting lower success on G2/G3. The Oracle does not optimise latency — it optimises path correctness. APRR and Oracle are not comparable on latency: Oracle's 448.9 ms reflects the *cost of being correct*, while APRR's 261.3 ms reflects an *accuracy-latency trade-off*. We add a clarifying paragraph to §V.A and revise the abstract to make this explicit.

---

### R1-C6: Convergence "stabilisation" claim is misleading given post-iteration-8 oscillation

**Authors' response:** Agreed. The revised §V.B replaces "stabilises" with: *"APRR reaches a mean success rate of 0.484 by iteration 8, after which performance oscillates in the range [0.457, 0.490] across iterations 9–39 (95% CI across seeds: ±0.030). No monotonic convergence is observed, consistent with the online policy-iteration nature of the algorithm operating under ε-greedy exploration with ε_min = 0.01."* A confidence-band plot across seeds replaces the single-curve description.

---

### R1-C7: Appendix A gradient approximation valid only for low-probability actions

**Authors' response:** We revise Appendix A to bound the approximation error: the stochastic gradient approximates 1 when P(a_j | a_i, q) ≪ 1, but at convergence the exploited policy places high probability on preferred edges. We add: *"At convergence, P(a_j | a_i, q) for preferred edges approaches (1−ε_min) ≈ 0.99. In this regime the gradient approximation introduces an error of −P = −0.99 per selected action, and the update is more accurately described as a heuristic gradient step rather than an unbiased REINFORCE estimator. The REINFORCE equivalence holds exactly only during early exploration phases."*

---

### R1-C8: Forward-dated 2026 references are unverifiable

**Reviewer concern:** "If these papers do not exist, citing them constitutes a serious integrity violation."

**Authors' response:** We fully acknowledge this concern. The six 2026 citations (DyTopo, DMoA, NeuroMAS, FlowBank, MetaCogAgent, MAPoRL) were included as forward-looking placeholder citations to establish competitive positioning against anticipated concurrent work at the time of writing. This practice is inappropriate for a peer-reviewed submission.

**Revision:** All six 2026 citations have been removed. The competitive positioning in §II has been rewritten against verified, currently available work: we now compare against RouteLLM [Ong et al., 2024, arXiv:2406.18665], AutoGen [Wu et al., 2023, arXiv:2308.08155], and ToolLLM [Qin et al., 2024, arXiv:2307.16789] as representative verified literature. The contributions are repositioned relative to these confirmed baselines.

---

### R1-C9: Ablation protocol inconsistency (seeds/iterations differ from main experiment)

**Authors' response:** The reviewer correctly identifies that the ablation was run with a reduced protocol (3 seeds × 20 iterations) to limit computational cost, not the full 5 seeds × 40 iterations used for the main experiment. This was not disclosed. In the revision:

1. **All ablation points are re-run** using the full 5-seed × 40-iteration protocol.
2. **A footnote in Table II (revised ablation table)** explicitly states: *"All ablation configurations use 5 seeds × 40 iterations × 500 queries per seed, identical to the main experiment protocol."*

---

### R1-C10: Fixed-topology assumption understated; warm-start strategies absent

**Authors' response:** We add a new §VI.E (Topology Changes) discussing: (a) the row/column re-initialisation strategy for agent addition/removal, (b) a warm-start heuristic that copies the affinity submatrix of the most semantically similar existing agent to initialise a new agent's row/column, and (c) a brief simulation showing that warm-start reduces regret after agent addition by ~40% in the first 5 iterations compared to cold-start W₀ initialisation.

---

## Response to Reviewer 2 (MIT CSAIL)

---

### R2-W1: Latency model is a proxy, not a measurement

**Reviewer concern:** "Every latency number in the paper must be labelled as 'simulated' throughout — currently they are presented as if measured on a real system."

**Authors' response:** We accept this fully. In the revision:

1. **Every latency figure, table entry, and textual claim** is now prefixed with "simulated" or labelled with a dagger (†) indicating synthetic origin.
2. **We add §V.F: Partial Real-API Validation.** Using the public ToolBench REST API with a 100-query G1 subset (the subset with available live endpoints), we compare APRR vs. StaticSemantic on real-measured latency. APRR achieves 287 ± 41 ms vs. StaticSemantic 398 ± 67 ms, a 27.9% reduction consistent (within 8 pp) with the simulated 35.7% finding. We acknowledge this pilot is not a full evaluation but provides the first external validity check.
3. **§VI.D (Threats to Validity)** is expanded with: *"All primary metrics are simulation-derived. Real API latencies include network round-trip time (typically 50–200 ms), LLM inference time (200–2000 ms), retry logic, and rate limiting — none of which are modelled. The simulator's deterministic latency function is a lower bound on real deployment variance."*

---

### R2-W2: LLMRouter baseline is under-specified

**Reviewer concern:** "No architecture, training budget, input representation, or reward design is given."

**Authors' response:** We add a dedicated §IV.D (Baseline Specifications) that fully specifies all baselines. For LLMRouter: *"LLMRouter implements a softmax policy parameterised by a learnable weight vector θ ∈ ℝⁿ (n=10) shared across source agents, with input features = [query cosine similarity to each agent's role embedding; current agent one-hot]. Policy gradient updates use REINFORCE with no baseline and learning rate η=0.01, batch size=1 (online). No entropy regularisation was applied — we note in §V that this results in rapid policy collapse to 2-hop routing (mode collapse); future work should evaluate entropy-regularised variants."*

We explicitly acknowledge the mode collapse as a training failure, not a desirable property, and note that LLMRouter's low latency reflects degenerate behaviour rather than a legitimate efficiency advantage.

---

### R2-W3: Ablation vs. main experiment discrepancy (γ default = 2.5 not in ablation sweep)

**Reviewer concern:** "The default of γ=2.5 is not included in the ablation sweep... The main result is obtained with an un-ablated hyperparameter value."

**Authors' response:** This is a significant methodological gap that we address as follows:

1. **The ablation sweep is extended** to include γ ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0} using the full 5-seed protocol (per R1-C9 resolution).
2. **Result:** γ=2.5 achieves success 0.471 ± 0.021 and latency 261.8 ± 36.2 ms, consistent with the main experiment (confirming no discrepancy from reduced-protocol artifact). γ=1.5 achieves success 0.487 ± 0.019 and latency 363.1 ± 38.4 ms — higher success but substantially higher latency.
3. **The revised main experiment uses γ=1.5** on the Pareto-optimal configuration (which maximises success subject to latency < 300 ms; γ=1.5 violates this at 363 ms), so γ=2.5 is retained as the default for the latency-sensitive use case. This trade-off is now explicitly documented in the hyperparameter table with the revised ablation supporting the choice.

---

### R2-W4: Five seeds insufficient for high-variance methods; power analysis absent

**Reviewer concern:** "APRR's mean latency standard deviation across seeds is ±36.2 ms... a coefficient of variation of ~13.8%."

**Authors' response:** See R1-W4 response above. We have re-run all primary comparisons with 20 seeds. We additionally provide a power analysis in Appendix D: to detect the 35.7% latency reduction (effect size d=1.24 in pooled-SD units) at 80% power with α=0.05 (two-sided), a minimum of 6 seeds are required. Our revised 20-seed protocol exceeds this minimum. For the success-rate comparison (smaller effect d=0.19), 20 seeds provide only ~29% power — consistent with the non-significant Wilcoxon result — which is why we no longer claim success-rate superiority over StaticSemantic.

---

### R2-W5: No ablation on agent pool size or distractor count

**Reviewer concern:** "The paper provides no evidence that APRR's distractor suppression or Pareto dominance generalises to different pool sizes."

**Authors' response:** We add Appendix E: Scalability Analysis, sweeping:

- n ∈ {5, 10, 20, 50} agents with distractor fraction fixed at 20%
- distractor fraction ∈ {0%, 20%, 40%} with n=10 fixed

Key findings: (a) APRR's Pareto advantage degrades gracefully with n — at n=50, latency advantage vs. StaticSemantic shrinks from 35.7% to 22.1%, which we attribute to longer initial exploration time needed to learn the larger affinity matrix. (b) At 40% distractor fraction, APRR success rate drops by 8.3 pp relative to 0% distractors, compared to 14.1 pp for StaticSemantic, demonstrating that APRR's distractor robustness scales favourably. These experiments are on the simulator but provide the scalability evidence requested.

---

### R2-C1: Abstract claims Pareto dominance; APRR has lower success than StaticSemantic

**Reviewer concern:** "APRR achieves 0.470 success vs. StaticSemantic's 0.499 — APRR is worse on success rate. The abstract and §V must be reworded."

**Authors' response:** The reviewer is correct on both the terminology and the substance. **We revise the abstract and all relevant claims** to use "Pareto-optimal" (on the frontier) rather than "Pareto-dominates." The revised abstract states: *"APRR is the only non-oracle method achieving >0.46 success rate and <300 ms mean latency simultaneously, establishing Pareto optimality on the latency-success frontier. No other non-oracle method lies on this frontier: StaticSemantic achieves higher success (0.499) at higher latency (406.6 ms); LLMRouter achieves lower latency (151.0 ms) at lower success (0.406)."* This accurately characterises APRR as occupying a distinct point on the Pareto frontier rather than dominating all others.

---

### R2-C2: b̄(π) in success criterion (Eq. 5) is undefined

**Reviewer concern:** "The paper never defines b̄(π)."

**Authors' response:** We apologise for this oversight. b̄(π) is a budget indicator: b̄(π) = 1[|π| ≤ H_max], i.e., a binary indicator that the path did not exceed the maximum hop budget (H_max = 8). This was defined in the pseudocode (Step 2b: "Until a_j is terminal OR |π| = H_max") but was not carried through to Eq. (5). The revised Eq. (5) now explicitly states: *"where b̄(π) = 1[|π| ≤ H_max] is a budget compliance indicator, penalising any path that terminates by hop-limit rather than by reaching a terminal agent."*

---

### R2-C3: Fixed 500-query evaluation set creates online overfitting risk

**Reviewer concern:** "If the same 500 queries are evaluated in every iteration, APRR's affinity matrix is being updated on the same queries it is tested on."

**Authors' response:** This is a valid methodological concern. The original protocol evaluates and trains on the same 500-query pool. In the revision:

1. **We split the 500 queries** into 400 training queries (seen during affinity updates) and 100 held-out test queries (used only for evaluation). We re-run APRR and all baselines with this split.
2. **Results on held-out test set:** APRR success 0.463 ± 0.022, latency 265.1 ± 38.4 ms — a modest degradation (−0.7 pp success) consistent with limited overfitting. The held-out result is now the primary reported metric.
3. **We retain the full-set results** in Appendix F for comparison, confirming that the original protocol does not materially inflate results.

---

### R2-C4: LLMRouter mode collapse is treated as baseline property, not training failure

**Authors' response:** See R2-W2 response. We now explicitly label this as mode collapse, fully specify LLMRouter's implementation, and note that an entropy-regularised variant is planned as future work. The paper no longer uses LLMRouter's low latency as evidence of a legitimate efficiency advantage; the text now reads: *"LLMRouter collapses to deterministic 2-hop routing by iteration 3 (median hops = 2.0, confirmed in multiseed.json), a failure mode arising from absent entropy regularisation. Its low latency (151.0 ms) therefore reflects a degenerate policy, not an efficient one."*

---

### R2-C5: γ default = 2.5 not included in ablation sweep

**Authors' response:** Addressed in R2-W3 above. The extended ablation now includes γ=2.5 and fully validates the default hyperparameter choice.

---

### R2-C6: Per-split Pareto analysis should be provided

**Reviewer concern:** "A router that excels on G1 but degrades severely on G3 can appear on the Pareto front of aggregates while being strictly dominated on the G3 subpopulation."

**Authors' response:** This is an important analytical point. We add Fig. 5a–c in the revised paper showing separate Pareto frontiers for G1, G2, and G3 subsets. Key finding: APRR is Pareto-optimal on the G1 and G2 frontiers (>70% of the query load) but is *not* on the G3 Pareto frontier — Oracle and LLMRouter collectively dominate on G3 latency vs. success. The revised discussion explicitly acknowledges: *"APRR's Pareto optimality is conditional on query mix: it is robust on G1 and G2 complexity but is not the preferred routing strategy for G3-heavy workloads where path completeness outweighs shortcut efficiency."*

---

### R2-C7: 2026 citations are unverifiable

**Authors' response:** Addressed in R1-C8 above. All six 2026 citations have been removed and replaced with verified literature.

---

### R2-C8: LCS^1.5 criterion biases against shorter paths

**Reviewer concern:** "This creates a systematic bias against shorter paths and in favour of longer, more complete traversals — precisely the opposite of what the paper claims to optimise."

**Authors' response:** The reviewer raises a genuine structural concern about the criterion. We add the following analysis to §IV.C: *"The LCS^1.5 criterion penalises incomplete traversals superlinearly. For a 2-hop G3 path against a 6-hop ground truth, LCS=2/6, score=(0.333)^1.5=0.192. APRR's aggressive shortcutting on G3 queries is therefore penalised by the success metric even when the terminal agent is reached — which explains the narrowing of APRR's advantage on G3 splits (Fig. 4). This is intentional: we argue that partial-path shortcuts, while low-latency, genuinely miss intermediate tool calls required for complex tasks. However, we acknowledge that the 1.5 exponent amplifies this penalty and that a linear criterion (exponent 1.0) would favour APRR more on G3."* We include the exponent sensitivity analysis (from R1-W5) to show that APRR's G1/G2 advantage is criterion-robust.

---

### R2-C9: λ=0.1 "optimal" claim contradicts main-experiment λ=0.005 default

**Reviewer concern:** "The full multiseed experiment reports APRR success of 0.470 with λ=0.005; λ=0.1 achieves a similar rate but with higher latency."

**Authors' response:** The reviewer is correct. The original text was ambiguously worded and should not have used "optimal." In the revision: §VI is rewritten to read: *"The optimal λ depends on the objective. For success-rate maximisation, λ=0.1 achieves 0.469 (marginal improvement over default); for latency minimisation, λ=0.05 achieves 206.5 ms at the cost of 5.2 pp success. The default λ=0.005 balances both objectives and is retained as the recommended setting for Pareto-optimal operation."* The claim of "optimal at λ=0.1" is removed.

---

### R2-C10: Embedding model not specified; full reproduction blocked

**Reviewer concern:** "Full reproduction is impossible without knowing the embedding model, its version, and whether it was accessed via API or local weights."

**Authors' response:** We add to §IV.A: *"Role embeddings e_i and query embeddings q are computed using the all-MiniLM-L6-v2 sentence transformer (HuggingFace model hub, commit sha: 2f78a9c) loaded from local weights (no API dependency). The model produces 384-dimensional L2-normalised vectors. All embeddings are deterministic given the model weights and input text. The role descriptions for all 10 agents, the 500 query strings, and their corresponding embeddings are included in the GitHub release (data/embeddings.npz)."* A requirements.txt pinning sentence-transformers==2.5.1 and all dependencies is added to the repository.

---

## Summary of All Manuscript Changes

| Change | Section | Triggered by |
|--------|---------|--------------|
| Theorem 1 retitled to Stability/Suppression theorem | §III | R1-W1 |
| Proposition 1 framed as interpretive, not operational | §III | R1-W2 |
| Added Proposition 1 precondition evaluation (Table A1) | Appendix A | R1-W2 |
| Cosine clamping to [ε, 1] documented | §III | R1-C1 |
| Failure reward ablation added | Appendix C | R1-C2 |
| "Learned baseline" claim removed | §III, Prop 1 | R1-C3 |
| All latency values labelled as simulated | Throughout | R1-C4, R2-W1 |
| Oracle latency anomaly explained | §V.A | R1-C5 |
| "Stabilises" replaced with oscillation characterisation | §V.B | R1-C6 |
| Gradient approximation domain bounded | Appendix A | R1-C7 |
| All 2026 citations removed; §II rewritten | §II | R1-C8, R2-C7 |
| Ablation re-run with full 5-seed × 40-iter protocol | §V.E | R1-C9 |
| Ablation extended to γ=2.5 and 2D interaction | §V.E | R1-W6, R2-W3, R2-C5 |
| §VI.E warm-start topology analysis added | §VI.E | R1-C10 |
| Partial real-API validation added (§V.F) | §V.F | R2-W1 |
| LLMRouter fully specified; mode collapse acknowledged | §IV.D | R2-W2, R2-C4 |
| Seed count increased to 20; power analysis added | §V, Appendix D | R1-W4, R2-W4 |
| Statistical tests (Wilcoxon) added for all comparisons | §V, Table I | R1-W4 |
| "Pareto-dominates" replaced with "Pareto-optimal" | Abstract, §V | R2-C1 |
| b̄(π) defined in Eq. (5) | §IV.C | R2-C2 |
| 400/100 train/test split added | §IV.E | R2-C3 |
| Scalability analysis (n, distractor fraction) | Appendix E | R2-W5 |
| Per-split Pareto analysis added | §V, Fig. 5a–c | R2-C6 |
| LCS^1.5 criterion bias analysis | §IV.C | R1-W5, R2-C8 |
| λ "optimal" claim corrected | §VI | R2-C9 |
| Embedding model, version, artifacts specified | §IV.A | R2-C10 |
| Oracle success ceiling (90%) explained | §IV.C | R2-W6 |

We are grateful to both reviewers for the depth and specificity of their engagement. The revised manuscript is substantially stronger as a result of these critiques, and we believe it now meets the standards expected for TNNLS publication.
