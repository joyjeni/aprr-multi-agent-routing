import { NextResponse } from "next/server";
import { getMultiSeed, getAblation } from "@/lib/data";

export const dynamic = "force-static";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

// GET /api/results -> returns the full benchmark payload (multiseed + ablation).
export async function GET() {
  const multiseed = getMultiSeed();
  const ablation = getAblation();
  return NextResponse.json(
    {
      method: "APRR",
      title: "Adaptive Probabilistic Routing Reinforcement",
      author: "Jenisha T",
      affiliation: "MS Ramaiah University of Applied Sciences",
      repository: "https://github.com/joyjeni/aprr-multi-agent-routing",
      headline: {
        latency_reduction_pct: 35.7,
        hop_reduction_pct: 23.9,
        pareto_optimal: true,
      },
      multiseed,
      ablation,
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
