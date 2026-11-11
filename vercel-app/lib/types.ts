// Shared types for the APRR benchmark dashboard.

export interface Stat {
  mean: number;
  std: number;
  ci95: number;
}

export interface RouterSummary {
  n: Stat;
  success_rate: Stat;
  success_rate_ci95: Stat;
  mean_latency_ms: Stat;
  p50_latency_ms: Stat;
  p95_latency_ms: Stat;
  mean_hops: Stat;
  median_hops: Stat;
  successful_mean_latency_ms: Stat;
}

export interface ConvergencePoint {
  iteration: number;
  success_rate: number;
  mean_latency_ms: number;
  mean_hops: number;
  success_rate_std: number;
  mean_latency_ms_std: number;
}

export interface MultiSeed {
  seeds: number[];
  n_queries: number;
  n_iterations: number;
  summary: Record<string, RouterSummary>;
  convergence_mean: Record<string, ConvergencePoint[]>;
}

export interface AblationRow {
  param: "alpha" | "beta" | "lam" | "gamma" | string;
  value: number;
  success_rate: number;
  latency_ms: number;
}

export type RouterName =
  | "Random"
  | "RoundRobin"
  | "StaticSemantic"
  | "LLMRouter"
  | "APRR"
  | "Oracle";

export const ROUTER_ORDER: RouterName[] = [
  "Random",
  "RoundRobin",
  "StaticSemantic",
  "LLMRouter",
  "APRR",
  "Oracle",
];

// Colour assignment: APRR = blue accent, Oracle (upper bound) = amber, baselines = a grey ramp.
// No red or green: those read as "faulty" and "correct". Validated for the dark chart surface.
export const ROUTER_COLORS: Record<string, string> = {
  Random: "#5b5b63",
  RoundRobin: "#71717a",
  StaticSemantic: "#a1a1aa",
  LLMRouter: "#d4d4d8",
  APRR: "#3987e5",
  Oracle: "#c98500",
};

// Line dash per router so identity never rests on shade alone.
export const ROUTER_DASH: Record<string, string | undefined> = {
  Random: "2 3",
  RoundRobin: "6 3",
  StaticSemantic: "10 3 2 3",
  LLMRouter: undefined,
  APRR: undefined,
  Oracle: "4 4",
};

// --- Objective 1 -> Objective 2 pipeline notebook results ---

export interface PipelineRouterResult {
  router: string;
  n_queries: number;
  success_rate: number;
  success_rate_first_half: number;
  success_rate_second_half: number;
  mean_routing_compute_ms: number;
}

export interface PipelineObj1Validation {
  n_queries: number;
  n_sessions: number;
  BM25_only: { "Hit@1": number; "Hit@5": number; "NDCG@5": number };
  BM25_plus_SessionRerank: { "Hit@1": number; "Hit@5": number; "NDCG@5": number };
}

export interface PipelineResults {
  dataset: string;
  objective1_repo: string;
  objective2_repo: string;
  agent_pool: string[];
  categories: string[];
  n_eval_queries: number;
  fusion_boost_fraction: number;
  fusion_boost_floor: number;
  objective1_validation: PipelineObj1Validation;
  results: PipelineRouterResult[];
}

// Without the O1 signal = grey; with the O1 fusion signal = blue accent; fixed baselines = dark grey.
export const PIPELINE_COLORS: Record<string, string> = {
  Random: "#5b5b63",
  StaticSemantic: "#5b5b63",
  "APRR-core (online, no Obj1 signal)": "#a1a1aa",
  "CTGR (online, no Obj1 signal)": "#a1a1aa",
  "FTDR (online, no Obj1 signal)": "#a1a1aa",
  "APRR-core + Obj1 fusion (online)": "#3987e5",
  "CTGR + Obj1 fusion (online)": "#3987e5",
  "FTDR + Obj1 fusion (online)": "#3987e5",
};
