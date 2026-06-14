# Peer Review — Harvard CSE Perspective

**Manuscript:** Adaptive Probabilistic Routing Reinforcement: Online Policy Iteration for Dynamic Agent-to-Agent Routing in Tool-Augmented LLM Workflows

**Reviewer:** Senior Professor, School of Engineering and Applied Sciences, Harvard University  
Expertise: Multi-Agent Systems, Reinforcement Learning Theory, Distributed Decision-Making

**Review Round:** 1  
**Venue Target:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

---

## Summary

This paper introduces APRR, a lightweight online policy-iteration algorithm for routing queries across a heterogeneous pool of specialised LLM agents. The core mechanism is a routing-affinity matrix **W** updated by a decay-regularised, success-weighted additive rule; a three-factor multiplicative policy (learned affinity × semantic prior × query relevance) then selects the next agent via a softmax. The authors prove convergence of the affinity matrix entries under bounded reward (Theorem 1) and establish equivalence to a REINFORCE gradient step with baseline subtraction under specific hyperparameter settings (Proposition 1). Empirical evaluation on a ToolBench-derived simulator with 10 agents and five random seeds demonstrates a 35.7% reduction in mean end-to-end latency and a 23.9% reduction in mean hop count relative to the strongest static baseline. The paper is well-structured, the algorithm is clearly described, and the reproducibility artifacts are commendable; however, substantial theoretical gaps and the exclusive reliance on a synthetic, closed-form simulator limit the paper's current standing for publication in TNNLS.

---

## Strengths

- **Clear theoretical framing.** The authors explicitly connect APRR to classical REINFORCE [Williams, 1992] via a log-space reparameterisation, making the algorithm's behaviour interpretable to an RL theorist. Proposition 1 provides a principled derivation, not merely an analogy.

- **Novel 1/L² path-length penalty.** The quadratic inverse path-length reward shaping is a concise, practical contribution that incentivises path compression without requiring any additional critic network or value function approximation. The mechanical simplicity is a genuine virtue.

- **Distractor suppression without explicit labelling.** The demonstration that affinity matrix columns corresponding to distractor agents converge to near-zero weight (consistent with Theorem 1) is non-trivial and practically relevant in agent deployments where tool-role overlap is inevitable.

- **Reproducible experimental protocol.** Five seeds × 40 iterations × 500 queries = 100,000 episodes per router, all on free-tier hardware, with pre-computed JSON artifacts. The community benefit of such accessibility is real.

- **Pareto-front analysis.** Presenting results simultaneously on the (latency, success) plane (Fig. 5) is the correct way to characterise a router that trades success rate against latency; APRR's Pareto dominance over non-oracle baselines is the paper's strongest empirical claim.

- **Informative ablation.** The single-parameter sweep for α, β, γ, and λ directly quantifies the marginal contribution of each factor. The finding that γ (query-relevance exponent) is the dominant hyperparameter (19.4 pp swing from γ=0 to γ=1.5) is insightful and well-explained.

---

## Weaknesses

- **Theorem 1 proves boundedness, not policy optimality.** The theorem establishes that affinity values remain within [W_min, W_max] and converge to [0, Δ_max/λ] for unrewarded edges — this is a stability result, not a regret bound or convergence-to-optimal-policy result. TNNLS readers in the RL community will immediately note the absence of any formal statement about convergence of the *induced routing policy* to a local or global optimum. The paper's theoretical contribution is significantly weaker than implied by the word "convergence."

- **Proposition 1's preconditions break the practical algorithm.** The REINFORCE equivalence requires α = γ = 1, β = 0, λ = 0 — but the default hyperparameters are α = 2.0, γ = 2.5, β = 1.0, λ = 0.005. The actual running algorithm is *not* a REINFORCE step; it is a heuristic update with a REINFORCE interpretation only in a degenerate regime that is never evaluated empirically. This gap between theory and practice is a significant concern.

- **The effective learning rate η = κ/(W_{ij} L² ℓ̃) is state-dependent and unbounded near W_min.** As W_{ij} → W_min = 10⁻³, η grows without bound regardless of κ, which can cause numerical instability. The paper acknowledges clamping but provides no analysis of how this clamp interacts with the REINFORCE equivalence or the convergence bound.

- **No statistical test is applied across methods.** Table I reports 95% CIs computed over five seeds. APRR's success rate (0.470 ± 0.020) and StaticSemantic's (0.499 ± 0.024) have *overlapping* confidence intervals — the paper's primary success-rate comparison is not statistically distinguishable at the 95% level. A paired t-test or Wilcoxon signed-rank test over seeds is mandatory before claiming superiority.

- **The LCS^1.5 success criterion is non-standard and its validity is not established.** The exponent 1.5 in Eq. (5) is not motivated theoretically, nor is it validated against human-judged success on actual ToolBench tasks. The criterion is strictly internal to the simulator, which creates a circularity: the reward function and the evaluation metric share structural assumptions, potentially inflating reported performance.

- **Incomplete ablation: only single-parameter sweeps are reported.** Given that the policy is multiplicative in three exponents (α, β, γ), the reported ablation treats parameters as independent. Cross-term interactions — especially between α and γ, whose product determines the relative emphasis of learned versus query-specific routing — could significantly alter the conclusions. The claim that γ "has the largest effect" could be confounded by the specific values of α held fixed during that sweep.

---

## Detailed Comments

1. **Section III, Eq. (2): Softmax normalisation and numerical stability.** The policy is defined as P(a_j | a_i, q) ∝ W_{ij}^α · η_{ij}^β · ψ_j(q)^γ. For negative cosine similarities (η_{ij} < 0 or ψ_j < 0), raising to a non-integer exponent γ = 2.5 produces complex-valued weights. The paper should explicitly state whether cosine similarities are clamped to [0,1] or [−1,1], and if negative values can arise in the role-embedding geometry (they can, generically). This is not a minor implementation detail; it affects the well-posedness of the policy.

2. **Section III, Affinity Update (Eq. 4): Reward definition for failures.** The deposit Δ W_{ij} uses r = −0.05 on failure. The choice of −0.05 (vs. 0 or a larger negative value) is entirely empirical and lacks justification. Given that the affinity clamp W_min = 10⁻³ is several orders of magnitude below W₀ = 0.1, a small negative r may be insufficient to drive distractor edges to their lower bound within 40 iterations. An analysis of the expected number of episodes required for distractor convergence as a function of r_failure would strengthen the paper considerably.

3. **Section III, Proposition 1: Baseline subtraction claim.** The paper states "When λ > 0, the decay acts as a learned baseline that subtracts expected future reward, reducing variance of the REINFORCE estimator." This is the most overreaching claim in the paper. A proper baseline in REINFORCE subtracts a *state-conditional* quantity from the reward to reduce variance while preserving expectation. The global decay (Eq. 3) subtracts λW_{ij} from *every* edge uniformly — this is a regulariser, not a state-conditional baseline. The variance reduction argument requires proof that E[λW_{ij}^{(t)}] = E[R | state] at the edges in question, which is not established.

4. **Section IV, Benchmark: Simulator fidelity.** The ToolBench-style simulator assigns latencies as a deterministic function of hop count. Real API call latencies are highly non-Gaussian (heavy-tailed, with retries, timeouts, and network jitter). The p95 latency for APRR (525.8 ms, std ≈ 187.8 ms from the JSON data) already shows large variance even within the deterministic simulator — a signal that tail behaviour is an important regime that is not adequately modelled.

5. **Section V, Table I: Oracle has higher mean latency than APRR (448.9 ms vs. 261.3 ms) despite following ground-truth paths.** This is counterintuitive: the Oracle with 4.30 mean hops is slower than APRR with 2.77 hops. The paper implies APRR achieves shortcuts that even the Oracle cannot, which would require APRR to violate the ground-truth path optimality assumption. The paper must explicitly explain this anomaly — either the Oracle is defined incorrectly (it uses the ground-truth *path* but not the *shortest* path), or the success criterion penalises shorter paths in ways that are not transparent.

6. **Section V, Convergence (text, §V.B): Claimed APRR improvement trajectory.** The manuscript states "APRR improves rapidly over the first 5 iterations (success: 0.443 → ~0.484 by iteration 8)." Inspection of the multiseed.json convergence data confirms iteration 0 ≈ 0.443 and iteration 8 ≈ 0.484 for APRR, which is accurate. However, iterations 9–39 show substantial oscillation (range: 0.457–0.490) rather than monotonic improvement or stable convergence. The claim of "stabilisation" after iteration 8 is misleading; the policy has not converged in any rigorous sense — it fluctuates within a band. A more honest characterisation with confidence bands across the 5 seeds should be provided.

7. **Appendix A, Proof of Proposition 1: Approximation P(a_j | a_i, q) ≪ 1.** The proof relies on "when P(a_j | a_i, q) ≪ 1, the stochastic gradient ≈ 1." This approximation is quantitatively valid only when the policy is nearly uniform. After convergence, the policy concentrates on optimal edges, making this approximation invalid precisely when the algorithm is most useful. The proof should either bound the approximation error or remove this step and provide an exact derivation.

8. **Section II, Related Work: Forward-dated references (2026).** The paper cites DyTopo [Hong et al., 2026], DMoA [Wu et al., 2026], NeuroMAS [Li et al., 2026], FlowBank [Zhang et al., 2026], MetaCogAgent [Chen et al., 2026], and MAPoRL [Tan et al., 2026]. These appear to be placeholder or speculative future citations. No arXiv IDs are provided (e.g., "arXiv:2602.xxxxx"). If these papers do not exist, citing them constitutes a serious integrity violation. The authors must provide verifiable URLs or arXiv identifiers, or remove these citations entirely and reposition the contributions relative to existing verified literature.

9. **Section VI, Discussion: The γ ablation uses a reduced evaluation protocol.** The ablation table (Fig. 7 / §V.E) reports values that differ from the full multiseed.json results. Specifically, the best γ=1.5 entry in the ablation (success 0.483) approximately matches the multiseed APRR mean (0.470), but the ablation values for γ=0 (success 0.289) appear to come from a single-seed or reduced-iteration run. The paper should clarify the exact protocol (seeds, iterations, queries) used for the ablation, as inconsistency would undermine the reliability of the sensitivity analysis.

10. **Section VII, Conclusion and Limitations: The fixed-topology assumption is understated.** The paper acknowledges "Agent addition/removal requires row/column re-initialisation in W," but does not discuss what happens to accumulated affinity information in the remaining rows/columns when even a single agent is swapped. In production multi-agent systems, agent pool composition changes frequently. A brief analysis of warm-start strategies (e.g., preserving affinity submatrix structure) would significantly increase the paper's practical relevance.

---

## Recommendation

**Major Revision**

The algorithmic contribution (APRR) is interesting and practically motivated. The Pareto-front dominance result and the distractor-suppression property are the paper's two strongest empirical claims. However, the theoretical section overstates its results (Theorem 1 ≠ policy optimality; Proposition 1 preconditions ≠ running algorithm), the success-rate comparison between APRR and StaticSemantic is not statistically significant at the 95% level, the 2026 forward-dated references must be resolved, and the simulator's construct validity is insufficiently addressed. These issues require substantial revision before the paper meets TNNLS standards.

| Criterion | Score (1–10) |
|-----------|-------------|
| Novelty | 6 |
| Technical Soundness | 5 |
| Clarity | 7 |
| Reproducibility | 8 |

**Overall Score: 6.5 / 10 — Major Revision**
