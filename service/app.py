#!/usr/bin/env python3
"""POST /route microservice for APRR core routing."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aprr.agents import AgentTopology  # noqa: E402
from aprr.router import APRRRouter, RouterConfig  # noqa: E402

TOPO = AgentTopology.default(seed=0)
ROUTER = APRRRouter(TOPO, RouterConfig(epsilon=0.0, seed=0))


def route(body: dict) -> dict:
    query = str(body.get("query", ""))
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    q_emb = rng.normal(size=TOPO.embeddings.shape[1])
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    path = ROUTER.route(q_emb, start=0, require_terminal=True)
    update = body.get("update") or {}
    if update:
        ROUTER.update_trail(
            path,
            bool(update.get("success", True)),
            float(update.get("latency_ms", 80)),
        )
    roles = [TOPO.agents[i].role for i in path]
    return {
        "service": "aprr",
        "path": path,
        "roles": roles,
        "formula": "P(a_j|a_i,q) ∝ W^α η^β ψ^γ",
        "W_mean": float(np.mean(ROUTER.W)),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path in {"/", "/health", "/route"}:
            self._send(200, {"ok": True, "service": "aprr", "n_agents": TOPO.n})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/route":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        self._send(200, route(body))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 43132
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"APRR microservice http://127.0.0.1:{port}/route", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
