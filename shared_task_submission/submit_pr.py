#!/usr/bin/env python3
"""
Submit all data + scripts to evaleval/EEE_datastore as a community PR.

Usage:
  huggingface-cli login          # paste your write token
  python submit_pr.py            # dry-run by default
  python submit_pr.py --push     # actually create the PR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, CommitOperationAdd
except ImportError:
    print("pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

REPO_ID = "evaleval/EEE_datastore"
HERE = Path(__file__).resolve().parent

# What to upload and where it goes in the datastore
UPLOAD_MAP = [
    # (local_dir, repo_prefix, glob_pattern, exclude_patterns)
    (HERE / "data", "data", "**/*.json", {"__pycache__", ".DS_Store"}),
    (HERE / "scripts", "utils/eee_track1", "**/*.py", {"__pycache__", ".DS_Store", "raw"}),
    (HERE / "analysis", "utils/eee_track1/analysis", "**/*.py", {"__pycache__", ".DS_Store"}),
    (HERE / "analysis" / "output", "utils/eee_track1/analysis/output", "**/*.csv", set()),
    (HERE / "docs", "utils/eee_track1/docs", "*.*", set()),
]


def collect_operations() -> list[CommitOperationAdd]:
    ops: list[CommitOperationAdd] = []
    for local_dir, repo_prefix, pattern, excludes in UPLOAD_MAP:
        if not local_dir.exists():
            print(f"  SKIP (missing): {local_dir}")
            continue
        for path in sorted(local_dir.glob(pattern)):
            if not path.is_file():
                continue
            if any(ex in path.parts for ex in excludes):
                continue
            rel = path.relative_to(local_dir)
            repo_path = f"{repo_prefix}/{rel}"
            ops.append(CommitOperationAdd(
                path_or_fileobj=str(path),
                path_in_repo=repo_path,
            ))
    return ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Actually create the PR (default is dry-run)")
    args = parser.parse_args()

    print(f"Target: {REPO_ID}")
    print("Collecting files ...")
    ops = collect_operations()
    print(f"  {len(ops)} files to upload\n")

    # Summary by prefix
    from collections import Counter
    prefix_counts = Counter(op.path_in_repo.split("/")[0] for op in ops)
    for prefix, count in sorted(prefix_counts.items()):
        print(f"  {prefix}/  → {count} files")
    print()

    if not args.push:
        print("DRY-RUN — pass --push to create the PR.")
        print("First 10 files:")
        for op in ops[:10]:
            print(f"  {op.path_in_repo}")
        if len(ops) > 10:
            print(f"  ... and {len(ops) - 10} more")
        return

    api = HfApi()

    PR_TITLE = "Track 1: 5,206 records across 58 sources (6 leaderboards + 52 papers)"
    PR_BODY = """\
## Shared Task Submission — Track 1: Public Eval Data Parsing

**Submitter:** @steliospsychias

### What's included
- **5,206 JSON records** conforming to EEE schema v0.2.1
- **58 sources**: 6 leaderboards (HF Open LLM v2, AlpacaEval 2.0, Chatbot Arena, MT-Bench, WildBench, BigCodeBench) + 52 academic papers
- **~4,818 unique model IDs** across **33 benchmark dimensions**
- **154 cross-source collision pairs** across 9 benchmarks
- Extraction scripts (scrapers + paper extractor) under `utils/eee_track1/`
- Analysis scripts (collision detection, variance decomposition, power analysis) under `utils/eee_track1/analysis/`
- Full extraction methodology documentation under `utils/eee_track1/docs/`

### Validation
All 5,205 records validate against eval.schema.json (1 record is the schema-version test fixture).

### Key finding
Prompt template is documented in only 41.4% of sources — metadata coverage is the primary bottleneck for cross-source LLM evaluation comparisons.
"""

    print(f"Creating PR with {len(ops)} files ...")
    commit_info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message=PR_TITLE,
        commit_description=PR_BODY,
        create_pr=True,
    )
    print(f"\n✓ PR created: {commit_info.pr_url}")


if __name__ == "__main__":
    main()
