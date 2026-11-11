import { NextResponse } from "next/server";
import { getMultiSeed, getAblation, getPipelineResults } from "@/lib/data";
import { getAppWorld } from "@/lib/appworld";

export const dynamic = "force-static";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

// GET /api/results -> AppWorld aggregates (latest) plus the earlier ToolBench simulation payload.
export async function GET() {
  const appworld = getAppWorld();
  const multiseed = getMultiSeed();
  const ablation = getAblation();
  const pipeline = getPipelineResults();
  return NextResponse.json(
    {
      method: "APRR (ACRS objective O3: risk-weighted policy routing)",
      author: "Jenisha T",
      affiliation: "Ramaiah University of Applied Sciences",
      repository: "https://github.com/joyjeni/aprr-multi-agent-routing",
      appworld,
      earlier_toolbench_simulation: {
        headline: {
          latency_reduction_vs_static_semantic_pct: 35.7,
          hop_reduction_vs_static_semantic_pct: 23.9,
          note: "Simulated agents; APRR is on the non-oracle latency-success frontier together with LLM Router and Static Semantic.",
        },
        multiseed,
        ablation,
        pipeline,
      },
    },
    { headers: CORS }
  );
}

// POST /api/results -> echoes a submitted payload back (notebook integration hook).
export async function POST(request: Request) {
  let body: unknown = null;
  try {
    body = await request.json();
  } catch {
    body = null;
  }
  return NextResponse.json(
    { received: true, payload: body },
    { headers: CORS }
  );
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: CORS });
}
