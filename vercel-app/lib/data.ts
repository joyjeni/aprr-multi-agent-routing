import fs from "node:fs";
import path from "node:path";
import type { MultiSeed, AblationRow, PipelineResults } from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

export function getMultiSeed(): MultiSeed {
  const raw = fs.readFileSync(path.join(DATA_DIR, "multiseed.json"), "utf-8");
  return JSON.parse(raw) as MultiSeed;
}

export function getAblation(): AblationRow[] {
  const raw = fs.readFileSync(path.join(DATA_DIR, "ablation.json"), "utf-8");
  return JSON.parse(raw) as AblationRow[];
}

export function getPipelineResults(): PipelineResults {
  const raw = fs.readFileSync(
    path.join(DATA_DIR, "pipeline_results.json"),
    "utf-8"
  );
  return JSON.parse(raw) as PipelineResults;
}
