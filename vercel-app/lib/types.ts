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

// Color assignment: APRR=red, Oracle=green, all baselines=blue family.
export const ROUTER_COLORS: Record<string, string> = {
  Random: "#1a73e8",
  RoundRobin: "#4f93f0",
  StaticSemantic: "#7fb1f5",
  LLMRouter: "#2b5fb0",
  APRR: "#d93025",
  Oracle: "#1e9e6a",
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

export const PIPELINE_COLORS: Record<string, string> = {
  Random: "#6b6b73",
  StaticSemantic: "#94A3B8",
  "APRR-core (online, no Obj1 signal)": "#0F766E",
  "CTGR (online, no Obj1 signal)": "#7C3AED",
  "FTDR (online, no Obj1 signal)": "#B45309",
  "APRR-core + Obj1 fusion (online)": "#0D9488",
  "CTGR + Obj1 fusion (online)": "#5B21B6",
  "FTDR + Obj1 fusion (online)": "#EA580C",
};
