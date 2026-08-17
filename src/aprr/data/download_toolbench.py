"""Download the real ToolBench benchmark eval splits from Hugging Face.

Uses the exact same dataset as Objective 1
(https://github.com/joyjeni/session-aware-toolbench-rerank) so both
objectives run on identical, verifiable, real data:
HF dataset repo ``tuandunghcmut/toolbench-v1``.

Only the ``benchmark/*.parquet`` eval splits are needed for Objective 2's
routing benchmark (candidate APIs + ground-truth relevant APIs per query);
the much larger multi-turn training shards used by Objective 1's
reranker are not required here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "tuandunghcmut/toolbench-v1"
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "toolbench_hf"


def main() -> str:
    ap = argparse.ArgumentParser(description="Download ToolBench eval splits from Hugging Face.")
    ap.add_argument("--out", default=None, help="Destination directory (default: <repo_root>/data/toolbench_hf)")
    args, _unknown = ap.parse_known_args()

    dest = Path(args.out) if args.out else _DEFAULT_ROOT
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[download] fetching {REPO_ID!r} -> {dest}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["benchmark/*.parquet", "README.md"],
    )
    print(f"[download] dataset at {path}")
    return path


if __name__ == "__main__":
    main()
