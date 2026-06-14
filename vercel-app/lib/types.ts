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
