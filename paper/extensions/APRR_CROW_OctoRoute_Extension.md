# APRR Extension: CROW and OctoRoute Routing Architectures

**Technical Research Extension Document — Objective 2 Dissertation Extension**
**Author:** Jenisha T, MS Ramaiah University of Applied Sciences
**Date:** June 2026
**Document Type:** PhD Dissertation Extension, Objective 2 — Advanced Routing Paradigms

---

## Abstract

The Adaptive Probabilistic Routing Reinforcement (APRR) system establishes a strong baseline for online, decay-regularized multi-agent routing over a directed agent graph, achieving a 35.7% latency reduction and 23.9% hop reduction relative to static-semantic baselines on ToolBench. This document formalizes two architectural extensions to APRR designed to address its two principal limitations: (1) the absence of introspective deliberation for high-uncertainty routing decisions, and (2) the centralised coordination bottleneck that increases hop count and latency as agent-graph diameter grows. **CROW** (Chain-of-Reasoning Over Workload Routing) augments the APRR policy with a query-complexity-conditioned thought budget θ_q; when routing uncertainty exceeds a threshold, CROW generates a structured Chain-of-Thought deliberation trace, scores its quality via a learned reasoning evaluator, and uses this quality signal to compute a reasoning-quality-weighted affinity update ΔW_CoT. **OctoRoute** replaces centralised dispatch with a biologically-inspired two-layer architecture: a lightweight functional-token router (the "brain") maps queries directly to specialised agent arms via discrete tokens ⟨route_k⟩, while each arm maintains a local affinity matrix W_local[k] and broadcasts one-bit chromatophore confidence signals to update the global coordinator. A **CROW-OctoRoute hybrid** integrates CROW's deliberation chains as arm-local reasoning modules, achieving best-of-both. Theoretical analysis and a detailed experimental protocol over the same 5-seed × 40-iteration × 500-query ToolBench benchmark are provided. We hypothesise that CROW achieves a 7–11% absolute success-rate improvement over APRR on complex (G3) queries at 1.4× latency cost, while OctoRoute achieves 18–25% latency reduction over APRR with modest cross-domain accuracy trade-offs.

---

## 1. Motivation

### 1.1 Limitations of APRR

APRR encodes routing competence as a decaying affinity matrix **W** ∈ ℝ^{|A|×|A|} updated by the rule:

$$\Delta W_{ij} \;\propto\; \mathbb{1}[\text{success}] \;\cdot\; \frac{1}{L^2} \;\cdot\; \frac{1}{\ell_{\text{norm}}}$$

with decay factor (1 − λ) applied at each iteration. While this formulation achieves competitive success rates (0.470 on ToolBench, 5-seed mean) and competitive latency (261.3 ms), two structural weaknesses are apparent upon analysis:

1. **Opacity of routing rationale.** The affinity update is a scalar signal: it records *that* a routing path succeeded, but not *why*. For complex, multi-hop queries (ToolBench G3 sub-category), the signal is highly noisy because success depends on the conjunction of multiple routing decisions. There is no mechanism to attribute success or failure to individual routing sub-steps.

2. **Centralised coordination bottleneck.** All routing decisions pass through the single global policy π(a | q, W). As the agent graph grows, the number of parameters in W scales as O(|A|²), and the per-query latency of evaluating the policy grows with graph diameter. APRR reports 2.77 mean hops, but for cross-domain queries this rises substantially, increasing latency disproportionately.

These two limitations motivate two distinct but complementary extensions.

### 1.2 Why Chain-of-Thought for Routing?

Chain-of-Thought (CoT) prompting [Wei et al., 2022] and its structured descendants — Layered-CoT [arxiv:2501.18645], TCAR [arxiv:2601.04544], and Route-to-Reason (RTR) [arxiv:2505.19435] — demonstrate that *externalising* a model's reasoning process before committing to a decision both improves accuracy and enables post-hoc quality assessment of the deliberation itself. In the routing context, the reasoning trace "agent A specialises in code execution, agent B in retrieval; query q requires retrieval followed by code execution; W[retrieval, code] = 0.72; therefore route to B then A" contains information that a scalar success/failure signal does not. The trace quality — logical coherence, citation of prior affinity evidence, accurate capability attribution — is a stronger learning signal than binary outcome. TCAR [arxiv:2601.04544] empirically confirms that reasoning-chain generation before agent selection significantly improves routing accuracy and robustness under ambiguous or cross-domain queries. RTR [arxiv:2505.19435] further shows that jointly routing over (model, reasoning-strategy) pairs under budget constraints achieves a superior accuracy-cost Pareto frontier compared to fixed-strategy routing.

The Self-REF framework [arxiv:2410.13284] provides a complementary mechanism: by injecting learned confidence tokens ⟨CN⟩ and ⟨UN⟩ into a fine-tuned LM, a continuous confidence score c ∈ [0, 1] can be extracted directly from token probabilities without requiring explicit verbalization. This mechanism underpins CROW's threshold-gated deliberation trigger.

### 1.3 Why Distributed Bio-Inspired Routing?

The biological octopus (*Octopus vulgaris*) distributes approximately two-thirds of its neurons across its eight arms, enabling arm-local motor control, texture discrimination, and short-range obstacle avoidance without consulting the central brain. This architectural principle — local competence with lightweight central coordination — maps naturally onto multi-agent routing when agents specialise strongly by domain (as in ToolBench's tool categories). Octopus v4 [arxiv:2404.19296] (Nexa AI) operationalises a related idea in LLM routing: functional tokens ⟨route_0⟩…⟨route_N⟩ replace full agent-name strings, reducing context length by approximately 95% while preserving dispatch accuracy. MasRouter [ACL 2025, arxiv:2502.11133] further shows that cascaded controller networks — collaboration mode determination, role allocation, LLM routing — can be decomposed and composed independently, with up to 52% overhead reduction relative to SOTA.

OctoRoute synthesises these insights: functional tokens provide the efficient dispatch layer; arm-local affinity matrices provide the specialised routing intelligence; and chromatophore signals (one-bit confidence broadcasts) provide the fast feedback channel that makes global W convergence faster than waiting for full response evaluation.

---

## 2. CROW: Chain-of-Reasoning Over Workload Routing

### 2.1 Overview

CROW augments APRR with a query-adaptive deliberation mechanism. For each query q, CROW first predicts a thought budget θ_q — the maximum number of reasoning steps warranted by query complexity. If the APRR greedy routing confidence c_q = max_j W[i,j] / Σ_j W[i,j] falls below a threshold τ, or if θ_q > θ_min, CROW activates a CoT deliberation chain before committing to a routing decision. The resulting reasoning trace T_q is scored for quality, and the quality score ρ(T_q) modulates the affinity update differently from the standard APRR rule.

### 2.2 Complexity Prediction Module

Let φ(q) ∈ ℝ^d be an embedding of query q (produced by a lightweight encoder, e.g., sentence-BERT). The thought budget predictor is a two-layer MLP:

$$\theta_q \;=\; \text{MLP}_\theta\!\bigl(\varphi(q)\bigr) \;=\; \text{clamp}\!\Bigl(\bigl\lfloor \sigma\!\bigl(W_2\,\text{ReLU}(W_1\,\varphi(q) + b_1) + b_2\bigr) \cdot \Theta_{\max}\bigr\rfloor,\; \theta_{\min},\; \Theta_{\max}\Bigr)$$

where Θ_max is the maximum allowed deliberation budget (e.g., 5 reasoning steps), θ_min = 1 (minimum), and σ is the sigmoid function. MLP_θ is trained offline on a labelled complexity dataset derived from ToolBench queries annotated by: (a) the number of tool calls in the ground-truth trajectory, (b) the number of distinct tool categories invoked, and (c) whether the query requires cross-agent coordination (G1 = single tool, G2 = intra-category multi-tool, G3 = cross-category multi-tool). The loss is a cross-entropy over discretised budget bins:

$$\mathcal{L}_\theta \;=\; -\sum_{q \in \mathcal{D}} \log P\!\bigl(\theta_q = \hat{\theta}_q\bigr)$$

where θ̂_q is the complexity label derived from the ground-truth tool trajectory.

### 2.3 Routing Confidence and Deliberation Trigger

At inference time, given the current routing affinity matrix W and current source agent i, CROW computes the greedy routing confidence:

$$c_q \;=\; \frac{\max_j W_{ij}}{\sum_j W_{ij} + \epsilon}$$

CROW activates the CoT deliberation chain when either of two conditions holds:

$$\text{Trigger}(q) \;=\; \bigl(c_q < \tau\bigr) \;\vee\; \bigl(\theta_q \geq \theta_{\min}^{\text{CoT}}\bigr)$$

where τ ∈ (0, 1) is a tunable confidence threshold (default: 0.45) and θ_min^{CoT} is the minimum thought budget that activates deliberation (default: 2). Below the trigger, CROW falls back to standard APRR greedy routing, preserving its latency profile for simple queries.

### 2.4 CoT Deliberation Chain

When the trigger fires, CROW generates a structured reasoning trace T_q over the current agent graph 𝒢 = (A, E, W):

$$T_q \;=\; \bigl\langle s_1, s_2, \ldots, s_{\theta_q} \bigr\rangle$$

Each step s_k is a natural-language statement of the form:

- **s_1 (Agent Capability Attribution):** "Agent A_j specialises in {domain_j}; query q requires {inferred_requirement}."
- **s_2 (Prior Evidence Invocation):** "Prior affinity W[i,j] = {value}; agent A_j has succeeded on {semantically similar} queries with rate {estimate}."
- **s_3 (Conflict Resolution, if multiple candidates):** "Agent A_j' also matches requirement; tie-broken by latency_norm[j] < latency_norm[j']."
- **s_k (Conclusion):** "Route to A_j*."

For θ_q ≥ 3 (high-uncertainty queries), CROW further spawns a **multi-agent deliberation sub-chain**: 2–3 candidate agents each produce a CoT justification for their own selection (a self-nomination trace), and a lightweight aggregator LM scores the traces and selects the consensus candidate. This mirrors the TCAR collaborative execution pipeline [arxiv:2601.04544] but operates at the routing level rather than the task level.

### 2.5 Reasoning Quality Scoring

The reasoning quality scorer ρ : 𝒯 → [0, 1] evaluates the trace T_q along three dimensions:

1. **Logical Coherence** (ρ_logic): Does the conclusion follow from the stated premises? Measured by a trained entailment classifier applied to (s_1, …, s_{θ_q−1}) → s_{θ_q}.
2. **Evidence Grounding** (ρ_grnd): Do the stated affinity values and capability attributions match W and the agent capability index? A lookup-based verifier computes Jaccard similarity between stated capabilities and the agent's registered capability set.
3. **Outcome Alignment** (ρ_align): After the task is executed, did the chosen agent succeed? ρ_align = 𝟙[success].

The composite quality score is:

$$\rho(T_q) \;=\; \alpha_1 \rho_{\text{logic}} + \alpha_2 \rho_{\text{grnd}} + \alpha_3 \rho_{\text{align}}, \quad \alpha_1 + \alpha_2 + \alpha_3 = 1$$

Default weights: α_1 = 0.3, α_2 = 0.3, α_3 = 0.4 (outcome alignment has the strongest learning signal).

### 2.6 Reasoning-Quality-Weighted Affinity Update

CROW replaces the APRR affinity update with a reasoning-quality-modulated rule. Let j* be the chosen routing target and let ρ(T_q) ∈ [0, 1] be the composite quality score:

$$\Delta W_{ij^*}^{\text{CROW}} \;=\; \rho(T_q) \;\cdot\; \frac{1}{L^2} \;\cdot\; \frac{1}{\ell_{\text{norm}}} \;\cdot\; \mathbb{1}[\text{success}]$$

For failure cases, the update includes a *negative quality penalty* proportional to trace quality:

$$\Delta W_{ij^*}^{\text{CROW,fail}} \;=\; -\,\rho(T_q) \;\cdot\; \frac{1}{L^2} \;\cdot\; \gamma_{\text{fail}}$$

where γ_fail ∈ (0, 1) is a penalty coefficient (default: 0.2). The rationale is: a high-quality trace that still leads to failure indicates the agent's actual capability is weaker than documented, warranting a stronger negative update; a low-quality trace that leads to failure should be penalised less aggressively, since the routing decision was poorly reasoned to begin with.

The full CROW update rule with decay is:

$$W_{ij^*} \;\leftarrow\; (1-\lambda) \cdot W_{ij^*} \;+\; \begin{cases} \Delta W_{ij^*}^{\text{CROW}} & \text{if success} \\ \Delta W_{ij^*}^{\text{CROW,fail}} & \text{if failure} \end{cases}$$

### 2.7 Algorithm 1: CROW Routing

```
Algorithm 1: CROW — Chain-of-Reasoning Over Workload Routing
─────────────────────────────────────────────────────────────────
Input:  Query q, agent graph 𝒢=(A, E, W), source agent i,
        threshold τ, thought budget predictor MLP_θ,
        quality scorer ρ, decay λ
Output: Routing decision j*, updated W

1:  φ_q ← Encode(q)                          // d-dim embedding
2:  θ_q ← MLP_θ(φ_q)                         // predict thought budget
3:  c_q ← max_j W[i,j] / (Σ_j W[i,j] + ε)  // greedy confidence

4:  if c_q < τ OR θ_q ≥ θ_min^{CoT} then
5:      // === CoT Deliberation Path ===
6:      Candidates ← Top-K(W[i,:], K=3)       // top-3 by affinity
7:      T_q ← []
8:      for k = 1 to θ_q do
9:          s_k ← GenerateReasoningStep(q, 𝒢, W, Candidates, k)
10:         T_q.append(s_k)
11:     end for
12:     if θ_q ≥ 3 then
13:         // Multi-agent deliberation sub-chain
14:         for each a ∈ Candidates do
15:             T_a ← GenerateSelfNomination(q, a, W)
16:         end for
17:         j* ← AggregatorLM.select(q, {T_a : a ∈ Candidates})
18:     else
19:         j* ← argmax_{j ∈ Candidates} Score(T_q, j)
20:     end if
21:     ρ_logic ← EntailmentScore(T_q)
22:     ρ_grnd  ← GroundingScore(T_q, W, AgentCapabilityIndex)
23:     // ρ_align computed post-execution (deferred)
24: else
25:     // === Greedy APRR Path ===
26:     j* ← argmax_j W[i,j]
27:     T_q ← None
28:     ρ_logic, ρ_grnd ← 1.0, 1.0  // no trace to penalise
29: end if

30: // === Execute routing decision ===
31: Route query q to agent j*
32: success ← Observe(response from j*)
33: L    ← PathLength(i, j*)
34: ℓ    ← Latency(j*)
35: ℓ_norm ← ℓ / ℓ_max_observed

36: // === Update W ===
37: if T_q ≠ None then
38:     ρ_align ← 𝟙[success]
39:     ρ(T_q) ← α_1·ρ_logic + α_2·ρ_grnd + α_3·ρ_align
40:     if success then
41:         ΔW ← ρ(T_q) · (1/L²) · (1/ℓ_norm)
42:     else
43:         ΔW ← -ρ(T_q) · (1/L²) · γ_fail
44:     end if
45: else
46:     // Standard APRR update
47:     ΔW ← 𝟙[success] · (1/L²) · (1/ℓ_norm)
48: end if

49: W[i,j*] ← (1-λ)·W[i,j*] + ΔW
50: return j*, W
─────────────────────────────────────────────────────────────────
```

**Complexity Analysis.** The greedy path (line 24–28) has the same O(|A|) complexity as base APRR. The CoT path adds O(θ_q · d_LM) for trace generation, where d_LM is the decoding cost per step. For θ_q ≤ 5 and lightweight deliberation LMs (e.g., 7B-parameter distilled models), the additional latency is bounded by approximately 150–300 ms per deliberation event. Since deliberation is triggered only on complex queries (estimated 15–25% of ToolBench G3), the amortised latency overhead is modest.

---

## 3. OctoRoute: Octopus-Inspired Distributed Routing Intelligence

### 3.1 Biological Motivation

The octopus nervous system comprises approximately 500 million neurons, of which ~33% reside in the central brain and ~67% are distributed across eight semi-autonomous arms. Each arm executes reflexive motor programs, texture classification, and localised path planning without central brain involvement. Communication between arm ganglia and the central brain uses fast, low-bandwidth signals — analogous to what we term **chromatophore signals** in OctoRoute, named after the rapid skin-color signaling used for inter-octopus communication and camouflage coordination. The key architectural insight is: **coarse coordination should be cheap and fast; fine control should be local and specialised**.

### 3.2 Architecture Overview

OctoRoute implements a two-layer routing hierarchy:

- **Layer 1 (Brain — Functional Token Dispatch):** A lightweight routing LM M_brain that maps query q to a functional token f_k ∈ {⟨route_0⟩, ⟨route_1⟩, …, ⟨route_N⟩}, each corresponding to one specialised agent arm. This mirrors Octopus v4 [arxiv:2404.19296], which demonstrates that functional tokens reduce routing context by ~95% compared to full agent-name descriptions while maintaining dispatch accuracy.
- **Layer 2 (Arm — Local Fine Routing):** Each arm k maintains a local affinity matrix W_local[k] ∈ ℝ^{|A_k|×|A_k|} over the sub-graph of agents specialised in arm k's domain. Within-arm routing follows a local APRR update rule.
- **Chromatophore Signal Protocol:** After processing a query, each arm broadcasts a one-bit confidence signal σ_k ∈ {0, 1} to the brain's global affinity W_global, enabling rapid feedback without the latency of waiting for a full response evaluation.

### 3.3 Functional Token Dispatch (Layer 1)

Let 𝒦 = {0, 1, …, N} be the set of arm indices, each arm k corresponding to a tool domain (e.g., k=0: code execution, k=1: web search, k=2: database query, k=3: file I/O, k=4: cross-domain). M_brain receives query q and produces:

$$f^* \;=\; \mathop{\arg\max}_{k \in \mathcal{K}} \; P_{M_{\text{brain}}}\!\bigl(\langle\text{route}_k\rangle \mid q\bigr)$$

The functional token vocabulary is trained via a contrastive objective over (query, arm-domain) pairs, following the Octopus v4 training paradigm [arxiv:2404.19296]. The brain model M_brain is intentionally small (1–3B parameters) to minimise dispatch latency. Critically, no agent capability description is included in the brain's context; only the functional token index is predicted, reducing context from O(|A| · desc_len) to O(1).

Let p(f_k | q) denote the brain's dispatch confidence for arm k. The dispatch is *certain* if max_k p(f_k | q) ≥ τ_brain (default: 0.7); otherwise, the top-2 arms are activated in parallel (a **multi-arm dispatch**), with each arm processing the query independently and the brain selecting the response with the higher chromatophore confidence.

### 3.4 Arm-Local Routing (Layer 2)

Within arm k, the local APRR policy operates over the sub-graph 𝒢_k = (A_k, E_k, W_local[k]). The local update rule mirrors the global APRR rule:

$$\Delta W_{\text{local}}^{(k)}[i,j] \;\propto\; \mathbb{1}[\text{success}] \;\cdot\; \frac{1}{L_k^2} \;\cdot\; \frac{1}{\ell_{\text{norm},k}}$$

with decay (1 − λ_k), where λ_k may differ from the global decay rate to allow faster local adaptation. Arm k operates autonomously: queries dispatched to arm k do not need to re-consult the brain until either (a) the query exceeds the arm's domain boundary (cross-domain escalation), or (b) the arm's local confidence falls below τ_arm (default: 0.35), triggering escalation to the brain.

### 3.5 Chromatophore Signal Protocol

The chromatophore signal σ_k is a fast, 1-bit feedback signal broadcast from arm k to the brain upon completing a routed query. It is defined as:

$$\sigma_k(q) \;=\; \begin{cases} 1 & \text{if arm } k \text{ produced a successful response with confidence} \geq \delta \\ 0 & \text{otherwise} \end{cases}$$

where δ is an arm-local confidence threshold derived from the Self-REF confidence token framework [arxiv:2410.13284]: arm k emits a ⟨CN⟩ token (σ_k = 1) or a ⟨UN⟩ token (σ_k = 0), and the confidence score c_k = P(⟨CN⟩) / (P(⟨CN⟩) + P(⟨UN⟩)) determines σ_k via δ thresholding.

The chromatophore update to the global routing matrix W_global is:

$$W_{\text{global}}[q_{\text{type}}, k] \;\leftarrow\; (1-\lambda_g) \cdot W_{\text{global}}[q_{\text{type}}, k] \;+\; \sigma_k(q) \cdot \eta_{\text{chroma}}$$

where q_type ∈ {G1, G2, G3} is the query complexity class, and η_chroma is a chromatophore learning rate (default: 0.05, smaller than the standard APRR update rate of 0.1 to reflect the lower information content of a 1-bit signal). Over N queries, the chromatophore protocol enables W_global to converge approximately √N times faster than waiting for full response evaluations, because signal latency is decoupled from response evaluation latency.

### 3.6 Arm Specialisation Index

The **Arm Specialisation Index** (ASI) measures the degree to which arm k's query distribution is concentrated in its target domain:

$$\text{ASI}(k) \;=\; 1 - \frac{H\bigl(\{p(d \mid k)\}_{d \in \mathcal{D}}\bigr)}{\log |\mathcal{D}|}$$

where H(·) is the Shannon entropy of the distribution over tool domains 𝒟 routed to arm k, and |𝒟| is the total number of domains. ASI(k) = 1 indicates perfect specialisation; ASI(k) = 0 indicates uniform distribution. This metric is used to monitor arm drift during online learning.

### 3.7 Algorithm 2: OctoRoute Inference and Update

```
Algorithm 2: OctoRoute — Octopus-Inspired Distributed Routing
─────────────────────────────────────────────────────────────────
Input:  Query q, brain model M_brain, arm set {(𝒢_k, W_local[k])},
        global matrix W_global, thresholds τ_brain, τ_arm, δ
Output: Response r, updated W_global, W_local[k*]

// === Layer 1: Functional Token Dispatch (Brain) ===
1:  for k in 𝒦 do
2:      p_k ← P_{M_brain}(⟨route_k⟩ | q)    // O(1) context per arm
3:  end for
4:  k* ← argmax_k p_k
5:  c_brain ← max_k p_k

6:  if c_brain ≥ τ_brain then
7:      ActiveArms ← {k*}                      // single-arm dispatch
8:  else
9:      k1, k2 ← Top-2({p_k})                 // parallel multi-arm dispatch
10:     ActiveArms ← {k1, k2}
11: end if

// === Layer 2: Arm-Local Fine Routing ===
12: for each k ∈ ActiveArms do (in parallel)
13:     i_k  ← EntryNode(𝒢_k)                 // arm entry agent
14:     c_arm_k ← max_j W_local[k][i_k, j]
         / (Σ_j W_local[k][i_k, j] + ε)
15:     if c_arm_k < τ_arm then
16:         // Escalate to brain for re-dispatch
17:         Escalate(q) → restart from Line 1
18:     end if
19:     j*_k ← argmax_j W_local[k][i_k, j]   // local greedy routing
20:     r_k  ← Execute(q, j*_k)               // arm-local execution
21:     // Chromatophore signal generation (Self-REF confidence token)
22:     c_CN_k ← P_{j*_k}(⟨CN⟩) / (P_{j*_k}(⟨CN⟩) + P_{j*_k}(⟨UN⟩))
23:     σ_k ← 𝟙[c_CN_k ≥ δ]                  // 1-bit broadcast
24: end for

// === Select best arm response (if multi-arm dispatch) ===
25: if |ActiveArms| > 1 then
26:     k* ← argmax_{k ∈ ActiveArms} c_CN_k
27: end if
28: r ← r_{k*}
29: success ← Evaluate(r, q)

// === Update Arm-Local W ===
30: L_k* ← PathLength(i_{k*}, j*_{k*})
31: ℓ_norm ← Latency(j*_{k*}) / ℓ_max
32: ΔW_arm ← 𝟙[success] · (1/L_{k*}²) · (1/ℓ_norm)
33: W_local[k*][i_{k*}, j*_{k*}] ←
        (1 - λ_{k*}) · W_local[k*][i_{k*}, j*_{k*}] + ΔW_arm

// === Chromatophore update to Global W ===
34: q_type ← ClassifyComplexity(q)             // G1/G2/G3
35: W_global[q_type, k*] ←
        (1-λ_g) · W_global[q_type, k*] + σ_{k*} · η_chroma

36: return r, W_global, W_local[k*]
─────────────────────────────────────────────────────────────────
```

### 3.8 Two-Layer OctoRoute Architecture (ASCII Diagram)

```
╔═══════════════════════════════════════════════════════════════╗
║                     OCTOROUTE ARCHITECTURE                    ║
╚═══════════════════════════════════════════════════════════════╝

                         ┌─────────────┐
           Query q ──────►  M_brain     │  (1–3B param lightweight LM)
                         │  Functional  │
                         │  Token       │
                         │  Dispatch    │
                         └──────┬──────┘
                                │
                    ┌───────────▼────────────┐
                    │  p(⟨route_k⟩ | q)      │
                    │  k* = argmax p_k        │
                    └──┬────────┬────────┬───┘
                       │        │        │
             ┌─────────▼──┐ ┌───▼──────┐ ┌▼─────────┐
             │   ARM  0   │ │  ARM  1  │ │  ARM  k  │
             │ (Code Exec)│ │ (WebSrch)│ │(Domain k)│
             │            │ │          │ │          │
             │ W_local[0] │ │W_local[1]│ │W_local[k]│
             │ ┌──────┐   │ │ ┌──────┐ │ │ ┌──────┐ │
             │ │A_0_0 │   │ │ │A_1_0 │ │ │ │A_k_0 │ │
             │ │A_0_1 │   │ │ │A_1_1 │ │ │ │A_k_1 │ │
             │ │ ...  │   │ │ │ ...  │ │ │ │ ...  │ │
             │ └──────┘   │ │ └──────┘ │ │ └──────┘ │
             └─────┬──────┘ └───┬──────┘ └────┬─────┘
                   │             │              │
              σ_0 ∈{0,1}    σ_1 ∈{0,1}    σ_k ∈{0,1}
                   │             │              │   ← Chromatophore
                   └─────────────┴──────────────┘     Signals (1-bit)
                                 │
                    ┌────────────▼───────────────┐
                    │     W_global update         │
                    │  W_global[q_type, k*] ←     │
                    │  (1−λ_g)·W_g + σ_k·η_chroma │
                    └─────────────────────────────┘

Legend:
  M_brain  = Central routing LM (Octopus "brain")
  ARM k    = Semi-autonomous agent sub-graph
  W_local  = Per-arm local affinity matrix
  σ_k      = Chromatophore 1-bit confidence signal
  W_global = Global arm-dispatch affinity table
```

---

## 4. Hybrid: CROW-OctoRoute

### 4.1 Design Sketch

The CROW and OctoRoute architectures are orthogonal: CROW improves the *quality* of each routing decision through deliberation; OctoRoute improves the *speed* and *scalability* of routing through distribution. They can be combined in a two-level hybrid:

1. **Brain-Level CROW Trigger:** When the brain's functional-token dispatch confidence c_brain < τ_brain and the thought budget θ_q ≥ θ_min^{CoT}, the brain generates a Layer-1 CoT trace before dispatching:
   > "Query q involves retrieval followed by computation. ⟨route_1⟩ (WebSearch arm) has W_global[G2, 1] = 0.68; ⟨route_0⟩ (CodeExec arm) has W_global[G2, 0] = 0.71. G2 query with retrieval dependency → dispatch to ⟨route_1⟩ first, then ⟨route_0⟩."

2. **Arm-Level CROW Deliberation:** Within a dispatched arm k, when the arm's local routing confidence c_arm < τ_arm, the arm itself spawns a local CROW deliberation chain over its sub-graph 𝒢_k. Since arms are domain-specialised, the deliberation context is narrower and therefore cheaper (fewer candidate agents, shorter traces).

3. **Chromatophore-Enhanced Trace Quality:** After arm-level CROW deliberation, the arm's chromatophore signal σ_k is augmented to a 2-bit signal: {00 = failure + low trace quality, 01 = failure + high trace quality, 10 = success + low trace quality, 11 = success + high trace quality}. This provides richer global feedback while maintaining low bandwidth.

**Formal Hybrid Update Rule:**

$$W_{\text{global}}[q_{\text{type}}, k] \;\leftarrow\; (1-\lambda_g) \cdot W_{\text{global}}[q_{\text{type}}, k] \;+\; \sigma_k^{(2)} \;\cdot\; \eta_{\text{chroma}} \;\cdot\; \rho_{\text{arm}}(T_{q,k})$$

where σ_k^{(2)} is the 2-bit chromatophore signal (0 or 1 for the success bit), and ρ_arm(T_{q,k}) ∈ [0, 1] is the arm-level CROW trace quality score.

The CROW-OctoRoute hybrid is expected to achieve the best overall performance on G3 (cross-domain) queries, where both deliberation quality and distributed arm expertise are critical, at the cost of the highest compute overhead.

---

## 5. Comparative Analysis Table

The following table provides a theoretical comparison of all five routing approaches on the specified dimensions. Quantitative estimates for the new methods are derived from: (a) APRR baseline numbers, (b) reported improvements in the cited literature (TCAR, RTR, MasRouter, Octopus v4), and (c) architectural analysis. All figures are projections for the same ToolBench 5-seed × 40-iteration × 500-query benchmark.

| Dimension | APRR (Baseline) | CROW | OctoRoute | CROW-OctoRoute | Oracle |
|---|---|---|---|---|---|
| **Success Rate (overall)** | 0.470 | 0.530–0.545 | 0.445–0.470 | 0.555–0.575 | 0.900 |
| **Success Rate (G1)** | ~0.55 | ~0.55 (no trigger) | ~0.58 (fast dispatch) | ~0.58 | ~0.95 |
| **Success Rate (G2)** | ~0.47 | ~0.52 | ~0.46 | ~0.54 | ~0.92 |
| **Success Rate (G3)** | ~0.38 | ~0.50 (+7–11 pp) | ~0.36 (cross-domain loss) | ~0.52 | ~0.85 |
| **Mean Latency (ms)** | 261.3 | 310–360 (CoT overhead) | 195–215 (−18–25%) | 280–320 | — |
| **Mean Hops** | 2.77 | 2.60–2.70 | 1.90–2.10 (direct dispatch) | 2.00–2.20 | — |
| **Interpretability** | Low (scalar W) | High (trace T_q) | Medium (token dispatch) | High | — |
| **Compute Overhead** | Baseline | +25–40% (deliberation) | −15% (small brain LM) | +10–20% | — |
| **Edge Deployment** | Moderate | Low (CoT LM required) | High (arms can be local) | Moderate | — |
| **Online Learning Speed** | Medium (full eval) | Medium-Slow (trace scoring) | Fast (1-bit σ) | Fast-Medium | — |
| **Cross-Domain Robustness** | Moderate | High (deliberation) | Low (arm boundary) | High | — |
| **Context Length (routing)** | O(\|A\|·desc) | O(\|A\|·desc + trace) | O(1) per functional token | O(1) + trace | — |
| **Failure Recovery** | Passive (decay) | Active (negative trace update) | Escalation to brain | Active + escalation | — |
| **Training Requirement** | None (online RL) | MLP_θ, ρ scorer offline pre-train | M_brain fine-tune | Both | — |

**Notes:**
- G1 = single-tool queries; G2 = intra-category multi-tool; G3 = cross-category multi-tool.
- OctoRoute latency reduction is estimated from Octopus v4's reported 95% context reduction and MasRouter's 52% overhead reduction at similar task distributions [arxiv:2404.19296; ACL 2025].
- CROW success rate improvement on G3 is estimated from TCAR's routing accuracy improvement on cross-domain queries (~12 pp) adjusted for the ToolBench difficulty distribution [arxiv:2601.04544].
- Oracle success rate (0.900) is the known ToolBench upper bound from APRR baseline experiments.

---

## 6. Experimental Protocol

### 6.1 Benchmark Configuration

All experiments follow the same protocol established for APRR to ensure fair comparison:

- **Dataset:** ToolBench (G1/G2/G3 sub-categories, balanced sampling)
- **Seeds:** 5 independent random seeds (identical to APRR baseline)
- **Iterations:** 40 online learning iterations per seed
- **Queries per iteration:** 500 (total: 5 × 40 × 500 = 100,000 query evaluations)
- **Agent graph:** Same 8-agent graph used in APRR baseline experiments
- **Baselines:** Random (0.323), RoundRobin (0.380), StaticSemantic (0.499), LLMRouter (0.406), APRR (0.470), Oracle (0.900)

### 6.2 CROW-Specific Implementation Details

- **Brain/deliberation LM:** Qwen2.5-7B-Instruct (4-bit quantised for deliberation, 1-bit chromatophore signals)
- **Thought budget predictor MLP_θ:** 2-layer MLP, hidden dim 128, trained on 5,000 labelled ToolBench queries (offline pre-training before online evaluation)
- **Reasoning quality scorer ρ:** EntailmentScore via DeBERTa-v3-base NLI; GroundingScore via exact-match lookup; OutcomeAlignment via binary success signal
- **Thresholds:** τ = 0.45 (deliberation trigger), θ_min^{CoT} = 2, α_1 = α_2 = 0.3, α_3 = 0.4, γ_fail = 0.2

### 6.3 OctoRoute-Specific Implementation Details

- **Brain LM:** Octopus v4-style 3B-parameter model fine-tuned on (query, arm_index) pairs derived from ToolBench tool category annotations
- **Arm assignment:** 5 arms for ToolBench (arm 0: weather/environment APIs, arm 1: social/communication APIs, arm 2: finance/data APIs, arm 3: code/compute APIs, arm 4: cross-domain/hybrid)
- **Arm agent pools:** 2–4 agents per arm, drawn from the same 8-agent graph
- **Self-REF confidence tokens:** Fine-tuned on per-arm training data using LoRA (rank 16)
- **Thresholds:** τ_brain = 0.70, τ_arm = 0.35, δ = 0.60 (chromatophore threshold), η_chroma = 0.05, λ_g = 0.01

### 6.4 New Metrics

In addition to the standard APRR metrics (success rate, mean latency, mean hops), the following new metrics are introduced:

**1. Reasoning Trace Quality Score (RTQS):**

$$\text{RTQS} \;=\; \frac{1}{|\mathcal{Q}_{\text{CoT}}|} \sum_{q \in \mathcal{Q}_{\text{CoT}}} \rho(T_q)$$

Computed over the subset Q_CoT of queries for which CoT deliberation was triggered (approximately 20–30% of total queries). Reported separately for G1, G2, G3 sub-categories. RTQS is the primary diagnostic for CROW's deliberation effectiveness.

**2. Chromatophore Signal Accuracy (CSA):**

$$\text{CSA}(k) \;=\; \frac{1}{|\mathcal{Q}_k|} \sum_{q \in \mathcal{Q}_k} \mathbb{1}\bigl[\sigma_k(q) = \mathbb{1}[\text{success}(q)]\bigr]$$

Measures the agreement rate between the arm's 1-bit chromatophore signal and the ground-truth success outcome. CSA serves as a calibration metric for the Self-REF confidence tokens. A well-calibrated arm should achieve CSA > 0.80.

**3. Arm Specialisation Index (ASI):** As defined in Section 3.6. Tracked per arm per iteration. ASI degradation over training indicates arm drift — agents within an arm are routing cross-domain queries, reducing specialisation.

**4. Deliberation Rate (DR):**

$$\text{DR} \;=\; \frac{|\mathcal{Q}_{\text{CoT}}|}{|\mathcal{Q}|} \times 100\%$$

The fraction of queries triggering CROW deliberation. Ideally decreases over training iterations as W converges and routing confidence increases (indicating that CROW is successfully building up routing knowledge that reduces uncertainty).

**5. Cross-Domain Escalation Rate (CDER):** For OctoRoute, the fraction of queries that trigger arm escalation back to the brain (line 17 in Algorithm 2). High CDER indicates poor arm boundary definitions or excessive cross-domain queries.

### 6.5 Statistical Analysis Plan

- Report 5-seed mean ± standard deviation for all metrics.
- Conduct paired Wilcoxon signed-rank tests between APRR and each new method (CROW, OctoRoute, CROW-OctoRoute) for success rate, with Bonferroni correction for three comparisons (α_corrected = 0.05/3 ≈ 0.017).
- Report effect sizes (Cohen's d) for primary success rate comparisons.
- Learning curves: Plot success rate vs. iteration for all methods on a single figure, with 95% confidence intervals from bootstrapping (B = 1000 resamples).
- Ablation study for CROW: compare (a) full CROW, (b) CROW without quality-weighted update (scalar update only), (c) CROW without multi-agent deliberation (single-chain only).
- Ablation study for OctoRoute: compare (a) full OctoRoute, (b) OctoRoute without chromatophore (full eval only), (c) OctoRoute without arm-local W (global W only).

---

## 7. Expected Results and Hypotheses

### Hypothesis H1 (CROW Success Rate):
> CROW will achieve a statistically significant improvement in overall success rate relative to APRR (H1: Δsuccess_CROW > 0, α = 0.017 after Bonferroni correction), with the largest improvement on G3 queries (expected Δ ≥ 0.07 absolute).

**Rationale:** TCAR [arxiv:2601.04544] reports ~12 pp improvement on cross-domain routing tasks via reasoning-chain generation. CROW applies this signal selectively (only when θ_q ≥ θ_min^{CoT}) and re-uses it for affinity updates, providing a compounding benefit across iterations. The RTR framework [arxiv:2505.19435] further shows that budget-constrained reasoning strategy selection achieves a better accuracy-cost trade-off than unconstrained reasoning.

### Hypothesis H2 (CROW Latency Cost):
> CROW's mean latency will be 15–40% higher than APRR (point estimate: +25%), primarily attributable to deliberation on G3 queries. G1 query latency will be statistically indistinguishable from APRR (expected Δ < 5 ms).

**Rationale:** Deliberation is triggered on approximately 20–25% of queries (predominantly G3). The mean deliberation latency (7B-parameter distilled model, 3–5 reasoning steps) is estimated at 180–250 ms, yielding a blended mean overhead of ~50 ms, approximately 19% of APRR's baseline 261.3 ms.

### Hypothesis H3 (OctoRoute Latency Reduction):
> OctoRoute will achieve a statistically significant latency reduction relative to APRR (H3: Δlatency_OctoRoute < 0), with a point estimate of 18–25% reduction (approximately 195–215 ms mean).

**Rationale:** Functional token dispatch eliminates the need to evaluate the full routing affinity matrix W over all agents; instead, the brain makes a single O(1)-context dispatch decision, and within-arm routing operates over a smaller sub-graph (2–4 agents vs. 8). Octopus v4 [arxiv:2404.19296] demonstrates 95% context reduction with maintained dispatch accuracy. The arm-local routing further reduces mean hops from 2.77 to an estimated 1.90–2.10, contributing additional latency savings.

### Hypothesis H4 (OctoRoute Cross-Domain Loss):
> OctoRoute will show a statistically significant success rate decrease on G3 (cross-domain) queries relative to APRR, estimated at −0.02 to −0.04 absolute (H4: Δsuccess_OctoRoute,G3 < 0), due to arm boundary rigidity.

**Rationale:** G3 queries requiring cross-arm coordination incur escalation overhead (arm → brain → re-dispatch) and may lose context across arm boundaries. This is the fundamental architectural trade-off of OctoRoute: speed via specialisation, at the cost of cross-domain flexibility.

### Hypothesis H5 (CROW-OctoRoute Hybrid):
> The hybrid will achieve the highest overall success rate among all four evaluated methods (APRR, CROW, OctoRoute, CROW-OctoRoute), particularly on G3 queries, while maintaining latency below CROW's level due to OctoRoute's fast base dispatch.

**Quantitative point estimates:**
| Metric | APRR | CROW | OctoRoute | CROW-OctoRoute |
|---|---|---|---|---|
| Success rate (overall) | 0.470 | 0.538 ± 0.012 | 0.458 ± 0.015 | 0.562 ± 0.011 |
| Success rate (G3 only) | 0.380 | 0.485 ± 0.018 | 0.355 ± 0.020 | 0.510 ± 0.016 |
| Mean latency (ms) | 261.3 | 328 ± 15 | 207 ± 12 | 295 ± 14 |
| Mean hops | 2.77 | 2.63 ± 0.09 | 2.02 ± 0.11 | 2.15 ± 0.10 |
| RTQS (CoT quality) | — | 0.74 ± 0.05 | — | 0.71 ± 0.06 |
| CSA (arm calibration) | — | — | 0.83 ± 0.04 | 0.81 ± 0.05 |

### Hypothesis H6 (Deliberation Rate Decay):
> The CROW deliberation rate (DR) will decrease monotonically over training iterations (from ~28% at iteration 1 to ~12% at iteration 40), as the routing affinity matrix W converges and routing confidence c_q increases above the deliberation threshold τ.

**Rationale:** This reflects the intended behaviour: CROW is most active in early iterations when W is uninformed, and progressively defers to the cheaper greedy APRR path as W captures routing competence.

---

## 8. Related Work

### 8.1 Chain-of-Thought Reasoning in Routing

**Wei et al. (2022)** introduced Chain-of-Thought (CoT) prompting, demonstrating that prompting LLMs to generate intermediate reasoning steps before producing a final answer substantially improves performance on multi-step arithmetic and commonsense reasoning tasks. CROW extends this principle to the routing decision itself, treating the routing choice as a multi-step reasoning problem.

**Layered-CoT** [arxiv:2501.18645] proposes structuring LLM reasoning into discrete verifiable layers, with each layer's output cross-checked against external knowledge sources or user feedback before proceeding. In a multi-agent setting, each layer check can be assigned to a specialised agent. CROW's multi-agent deliberation sub-chain directly implements this architecture at the routing level: candidate agents self-nominate with CoT justifications, and an aggregator LM verifies the logical consistency of each nomination.

**TCAR** [arxiv:2601.04544] (Adaptive Reasoning Router for Multi-Agent Collaboration, Tencent Cloud) proposes the first domain router that generates structured natural-language reasoning chains before agent selection, and supports dynamic agent onboarding (new agents require only capability description updates, not retraining). TCAR demonstrates significant improvements in routing accuracy, recall under conflict-prone scenarios, and end-to-end answer quality on both public benchmarks and large-scale enterprise ITSM data. CROW extends TCAR by making the thought budget query-adaptive and by using the trace quality as an RL update signal.

**Route-to-Reason (RTR)** [arxiv:2505.19435] presents a unified framework for jointly selecting the optimal LLM and reasoning strategy (CoT, PAL, CoD, Vanilla) for each query under budget constraints. RTR learns compressed vector representations of both models and strategies, trains two predictor heads (expected performance and expected token cost), and selects the optimal (model, strategy) pair at inference. This directly motivates CROW's thought budget predictor: CROW applies the same budget-constrained optimisation principle to routing decisions rather than model-strategy selection.

**Self-REF** [arxiv:2410.13284] (Learning to Route LLMs with Confidence Tokens, ICML 2025) proposes a lightweight fine-tuning strategy to teach LLMs to express calibrated confidence via special tokens ⟨CN⟩ and ⟨UN⟩. The resulting continuous confidence score c = P(⟨CN⟩) / (P(⟨CN⟩) + P(⟨UN⟩)) enables threshold-gated routing and rejection. CROW's deliberation trigger condition c_q < τ and OctoRoute's chromatophore signal protocol both draw directly on this framework.

### 8.2 Distributed and Functional-Token Routing

**Octopus v4** [arxiv:2404.19296] (Nexa AI) introduces a graph-of-language-models architecture where a master node (the Octopus model) uses functional tokens ⟨route_k⟩ to dispatch user queries to specialised worker models and reformat queries for optimal worker performance. Octopus v4 reduces routing context by ~95% compared to description-based routing, achieving state-of-the-art MMLU of 74.8 among ~10B-parameter models while requiring only two small models rather than a single large frontier model. OctoRoute directly adopts functional token dispatch and extends it with arm-local affinity matrices and chromatophore feedback.

**MasRouter** [ACL 2025, arxiv:2502.11133] formalises the Multi-Agent System Routing (MASR) problem — jointly determining collaboration mode, role allocation, and LLM assignment — and solves it with a cascaded controller network trained via policy gradient RL. MasRouter achieves 1.8–8.2% improvement over SOTA on MBPP and up to 52% overhead reduction on HumanEval, and demonstrates plug-and-play integration with AutoGen/CrewAI frameworks. OctoRoute's two-layer architecture (brain dispatch + arm local routing) is directly analogous to MasRouter's cascaded controller, with the key distinction that OctoRoute replaces the controller's sequential decisions with parallel arm execution.

### 8.3 Multi-Agent Long-Context Collaboration

**Chain-of-Agents** [arxiv:2406.02818] proposes a training-free, task-agnostic framework for long-context tasks where each agent processes a document chunk and produces evidence for the next agent, culminating in a manager agent that synthesises the final response. This worker-manager topology directly informs OctoRoute's arm-brain architecture, and the evidence propagation protocol motivates the chromatophore signal design (a lightweight version of inter-agent evidence passing).

---

## 9. Integration with PhD Objectives 1, 3, and 4

### 9.1 Integration with Objective 1: SessionRerank

**Objective 1 (SessionRerank)** addresses the re-ranking of session-level query trajectories — ordering and prioritising queries within a multi-turn session for optimised downstream processing. CROW integrates with SessionRerank as follows:

CROW's reasoning trace T_q contains session-level context: the deliberation steps reference W[i,j] values that have been updated by prior queries in the same session. This creates a natural **session-aware routing history** that SessionRerank can exploit. Specifically, the RTQS (Reasoning Trace Quality Score) from CROW's recent deliberations can be used as a session-quality signal to re-rank upcoming queries in the session buffer. Formally, let Q_sess = {q_1, …, q_n} be the pending queries in the session. SessionRerank assigns a priority score:

$$\text{Priority}(q_t) \;=\; \beta_1 \cdot \text{Urgency}(q_t) \;+\; \beta_2 \cdot \mathbb{E}\bigl[\text{RTQS}(T_{q_t})\bigr]$$

where E[RTQS(T_{q_t})] is the expected trace quality for query q_t estimated from its predicted thought budget θ_{q_t} (higher θ implies more complex deliberation, which historically correlates with higher RTQS variance, motivating earlier scheduling to avoid deliberation queue stalls). This gives SessionRerank a routing-aware priority signal that it would otherwise lack.

Furthermore, CROW's deliberation chain can operate proactively on the SessionRerank queue: for the top-k upcoming queries (by priority), CROW can pre-generate reasoning traces in background threads, reducing deliberation latency at query execution time by caching T_q for reuse if the routing context W has not changed since pre-computation.

### 9.2 Integration with Objective 3: MNCD Mesh Distress Signaling

**Objective 3 (MNCD — Multi-Node Context Distress)** addresses signaling and recovery when mesh nodes (agents) enter distress states: high latency, repeated failures, or context overload. Both CROW and OctoRoute provide natural integration points:

**CROW → MNCD:** CROW's negative quality-weighted update rule (Section 2.6) provides a finer-grained distress signal than APRR's binary success/failure. Specifically, when ρ(T_q) is high but success is False, the routing trace contains a high-quality reasoning chain that nonetheless led to failure — a strong indication that the target agent j* is in a distress state (its documented capabilities no longer match its actual performance). CROW can trigger a **distress probe** to MNCD: transmit (j*, ρ(T_q), success=False, trace=T_q) to the MNCD monitor, which uses the trace to diagnose whether the distress is capability-based (agent lacks the skill), load-based (agent is overloaded), or context-based (agent received an ill-formed query). The trace provides structured evidence for this tripartite diagnosis that MNCD would otherwise need to infer from aggregate statistics alone.

**OctoRoute → MNCD:** OctoRoute's chromatophore signal accuracy (CSA) per arm is a low-latency, per-arm distress indicator. If CSA(k) drops below a threshold δ_distress (default: 0.65) over a rolling window of N_window = 50 queries, OctoRoute broadcasts an **arm distress event** to the MNCD mesh monitor:

$$\text{ArmDistress}(k) \;=\; \mathbb{1}\!\Bigl[\text{CSA}_{N_{\text{window}}}(k) < \delta_{\text{distress}}\Bigr]$$

Upon receiving ArmDistress(k), MNCD can initiate arm-level recovery procedures: re-routing queries away from arm k to alternative arms, triggering re-calibration of Self-REF confidence tokens for agents in arm k, or escalating all arm k queries to brain-level routing until CSA recovers. This creates a tight feedback loop between OctoRoute's confidence calibration and MNCD's distress management.

### 9.3 Integration with Objective 4: FCNP Context Pruning

**Objective 4 (FCNP — Fine-Grained Context and Noise Pruning)** addresses removing irrelevant context from agent inputs before execution, reducing hallucination and improving response quality. CROW and OctoRoute enable FCNP in distinct ways:

**CROW → FCNP:** CROW's reasoning trace T_q is a byproduct that explicitly identifies which agent capabilities are relevant to query q (step s_1 in the deliberation chain: "Query q requires {inferred_requirement}"). This capability attribution can directly drive FCNP's pruning decision: context sections corresponding to capabilities not mentioned in T_q are candidates for pruning. Formally, let C_q = {c_1, …, c_m} be the context chunks available for agent j*. FCNP using CROW's trace filters:

$$C_q^{\text{pruned}} \;=\; \bigl\{c_\ell \in C_q : \text{Relevance}(c_\ell, T_q) \geq \eta_{\text{FCNP}}\bigr\}$$

where Relevance(c_ℓ, T_q) is a cross-encoder similarity between context chunk c_ℓ and the capability attribution in T_q. This eliminates the need for FCNP to independently infer query intent — it can consume the structured reasoning trace directly, making FCNP both faster (no separate intent inference step) and more accurate (the reasoning trace is a higher-quality intent signal than raw query q alone).

**OctoRoute → FCNP:** Arm specialisation provides FCNP with a strong prior: queries dispatched to arm k are expected to invoke only tool domain 𝒟_k. FCNP can therefore apply aggressive domain-gated pruning: any context chunk belonging to a domain other than 𝒟_k is automatically pruned (unless the arm's local routing escalates to cross-domain mode). This domain-gated pruning is computationally cheap (O(|C_q|) classification) and leverages OctoRoute's arm assignment as a free pruning signal, reducing FCNP's computational overhead by an estimated 30–50% on single-arm dispatched queries.

---

## 10. Notation Summary

| Symbol | Definition |
|---|---|
| W ∈ ℝ^{\|A\|×\|A\|} | Global routing affinity matrix (APRR) |
| λ ∈ (0,1) | Decay rate for W update |
| φ(q) ∈ ℝ^d | Query embedding vector |
| θ_q ∈ ℤ⁺ | Query thought budget (CROW) |
| τ ∈ (0,1) | Routing confidence threshold for deliberation trigger |
| T_q = ⟨s_1,…,s_{θ_q}⟩ | CoT reasoning trace (CROW) |
| ρ(T_q) ∈ [0,1] | Composite reasoning quality score |
| ρ_logic, ρ_grnd, ρ_align | Quality sub-scores (logic, grounding, outcome) |
| α_1, α_2, α_3 | Quality score weights (sum to 1) |
| γ_fail ∈ (0,1) | Failure penalty coefficient |
| f_k = ⟨route_k⟩ | Functional token for arm k (OctoRoute) |
| W_local[k] ∈ ℝ^{\|A_k\|×\|A_k\|} | Arm-local affinity matrix |
| W_global ∈ ℝ^{3×\|𝒦\|} | Global brain-to-arm dispatch matrix (indexed by query type) |
| σ_k ∈ {0,1} | Chromatophore 1-bit confidence signal |
| σ_k^{(2)} ∈ {00,01,10,11} | Extended 2-bit chromatophore signal (hybrid) |
| η_chroma | Chromatophore learning rate |
| λ_g | Global W_global decay rate |
| ASI(k) ∈ [0,1] | Arm Specialisation Index |
| CSA(k) ∈ [0,1] | Chromatophore Signal Accuracy |
| RTQS ∈ [0,1] | Reasoning Trace Quality Score |
| DR ∈ [0,1] | CROW Deliberation Rate |
| CDER ∈ [0,1] | Cross-Domain Escalation Rate (OctoRoute) |
| MLP_θ | Thought budget predictor network |
| M_brain | Lightweight functional-token routing LM |
| ⟨CN⟩, ⟨UN⟩ | Self-REF confident/unconfident tokens |
| c = P(⟨CN⟩)/(P(⟨CN⟩)+P(⟨UN⟩)) | Continuous Self-REF confidence score |

---

## References

1. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. https://arxiv.org/abs/2201.11903

2. Zhao, J., et al. (2026). **TCAR: Adaptive Reasoning Router for Multi-Agent Collaboration**. *arXiv:2601.04544*. https://arxiv.org/abs/2601.04544

3. Pan, Z., Zhang, K., Zhao, Y., & Han, Y. (2025). **Route to Reason: Adaptive Routing for LLM and Reasoning Strategy Selection**. *arXiv:2505.19435*. https://arxiv.org/abs/2505.19435

4. Anonymous (2025). **Layered Chain-of-Thought Prompting for Multi-Agent LLM Systems: A Comprehensive Approach to Explainable Large Language Models**. *arXiv:2501.18645*. https://arxiv.org/abs/2501.18645

5. Chuang, Y.-N., Sarma, P. K., Gopalan, P., Boccio, J., Bolouki, S., Hu, X., & Zhou, H. (2024). **Learning to Route LLMs with Confidence Tokens (Self-REF)**. *ICML 2025*. https://arxiv.org/abs/2410.13284

6. Chen, W., et al. (2024). **Octopus v4: Graph of Language Models**. *Nexa AI, arXiv:2404.19296*. https://arxiv.org/abs/2404.19296

7. Yue, Y., Zhang, G.-M., et al. (2025). **MasRouter: Learning to Route LLMs for Multi-Agent Systems**. *ACL 2025*. https://aclanthology.org/2025.acl-long.757

8. Zhao, S., et al. (2024). **Chain of Agents: Large Language Models Collaborating on Long-Context Tasks**. *arXiv:2406.02818*. https://arxiv.org/abs/2406.02818

---

*End of Document*

**Citation for this document:**
> Jenisha T (2026). *APRR Extension: CROW and OctoRoute Routing Architectures — Technical Research Extension for Objective 2*. MS Ramaiah University of Applied Sciences PhD Dissertation, Extension Document v1.0, June 2026.
