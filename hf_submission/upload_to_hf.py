"""
upload_to_hf.py — upload the Every Eval Ever dataset to Hugging Face Hub.

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   (or set HF_TOKEN env var)

Usage:
  python hf_submission/upload_to_hf.py [--repo-id YOUR_HF_ORG/every_eval_ever]
  python hf_submission/upload_to_hf.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print("huggingface_hub not installed — run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


DEFAULT_REPO_ID = "steliospsychias/every_eval_ever"
DATA_DIR = _ROOT / "data"
HF_SUBMISSION_DIR = _ROOT / "hf_submission"


def build_dataset_card() -> str:
    """read the README.md from hf_submission/ as the dataset card."""
    card_path = HF_SUBMISSION_DIR / "README.md"
    return card_path.read_text(encoding="utf-8") if card_path.exists() else ""


def count_records() -> dict[str, int]:
    """count records per leaderboard source."""
    counts: dict[str, int] = {}
    for source_dir in sorted(DATA_DIR.iterdir()):
        if source_dir.is_dir():
            n = sum(1 for _ in source_dir.rglob("*.json"))
            counts[source_dir.name] = n
    return counts


def upload(repo_id: str, dry_run: bool = False) -> None:
    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    print(f"target repo:  {repo_id}")
    print(f"data dir:     {DATA_DIR}")
    counts = count_records()
    total = sum(counts.values())
    print(f"records:      {total} total")
    for src, n in sorted(counts.items()):
        print(f"  {src}: {n}")
    print()

    if dry_run:
        print("dry-run mode — skipping upload")
        return

    # create or verify repo exists
    print("creating/verifying HF repo ...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )

    # upload dataset card
    card_content = build_dataset_card()
    if card_content:
        print("uploading dataset card ...")
        api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    # upload data directory
    print(f"uploading {total} records ...")
    upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="data",
        token=token,
        ignore_patterns=["*.DS_Store"],
    )

    print(f"\ndone — dataset available at: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="upload EEE dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repository ID")
    parser.add_argument("--dry-run", action="store_true", help="skip actual upload")
    args = parser.parse_args()

    upload(repo_id=args.repo_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
