// AppWorld results of ACRS (aggregates only; no task text or API output is published).
import fs from "node:fs";
import path from "node:path";

export interface AwConsensus {
  voted_steps: number;
  certified: number;
  abstain_disagree: number;
  abstain_all_failed: number;
  candidate_executions: number;
  login_exempt: number;
}

export interface AwConfig {
  id: string;
  code: string;
  label: string;
  description: string;
  status: "done" | "pending";
  tgc_pct?: number;
  tgc_solved?: number;
  tgc_ci95?: [number, number];
  sgc_pct?: number;
  sgc_solved?: number;
  steps_per_task?: number;
  tokens_m?: number;
  prompt_tokens_mean_per_task?: number;
  history_tokens_per_step?: number;
  small_steps?: number;
  large_steps?: number;
  large_share_pct?: number;
  refetch_calls?: number;
  login_calls?: number;
  recoverability?: number;
  collateral_damage_tasks?: number;
  collateral_damage_pct?: number;
  error_step_rate_pct?: number;
  wall_min?: number;
  consensus?: AwConsensus;
  policy_denials?: number;
  provenance_denials?: number;
}

export interface AwPaired {
  a: string;
  b: string;
  a_only: number;
  b_only: number;
  p: number;
}

export interface AwWilcoxon {
  a: string;
  b: string;
  median_a: number;
  median_b: number;
  p: number;
}

export interface AwFinding {
  objective: string;
  rq: string;
  status: string;
  evidence: string;
  next: string;
}

export interface AwObjective {
  id: string;
  name: string;
  short: string;
  repo?: string;
  appworld: string;
}

export interface AwControl {
  id: string;
  scope: string;
  rule: string;
}

export interface AppWorldResults {
  title: string;
  updated: string;
  source: string;
  privacy: string;
  benchmark: {
    name: string;
    citation: string;
    apps: number;
    apis: number;
    split: string;
    tasks: number;
    scenarios: number;
    seeds: number;
    few_shot: number;
    grading: string;
  };
  models: { large: string; small: string; serving: string };
  objectives: AwObjective[];
  code_names: Record<string, string>;
  configs: AwConfig[];
  paired_mcnemar: AwPaired[];
  wilcoxon_prompt_tokens: AwWilcoxon[];
  findings: AwFinding[];
  caveats: string[];
  context: string;
  next_runs: string[];
  security_controls: AwControl[];
}

export function getAppWorld(): AppWorldResults {
  const raw = fs.readFileSync(
    path.join(process.cwd(), "public", "data", "appworld_dev.json"),
    "utf-8"
  );
  return JSON.parse(raw) as AppWorldResults;
}

export function cfg(data: AppWorldResults, id: string): AwConfig {
  const c = data.configs.find((x) => x.id === id);
  if (!c) throw new Error(`missing AppWorld config ${id}`);
  return c;
}

// Chart colours (dark surface). One accent for full ACRS, neutral greys for the other configurations,
// and two steps of the same blue for the model tiers. No red or green is used anywhere on the site.
export const AW = {
  accent: "#3987e5",
  accentSoft: "#9ec5f4",
  neutral: "#71717a",
  neutralSoft: "#a1a1aa",
  reference: "#c98500",
  track: "#26262c",
};
