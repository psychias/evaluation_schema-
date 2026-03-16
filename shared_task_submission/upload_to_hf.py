#!/usr/bin/env python3
"""
upload_to_hf.py — upload the Every Eval Ever dataset to Hugging Face Hub.

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login   (or set HF_TOKEN env var)

Usage:
  python upload_to_hf.py [--repo-id YOUR_HF_ORG/every_eval_ever]
  python upload_to_hf.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
DOCS_DIR = _HERE / "docs"
SCHEMA_DIR = _HERE / "schema"

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print("huggingface_hub not installed — run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


DEFAULT_REPO_ID = "steliospsychias/every_eval_ever"


def build_dataset_card() -> str:
    """Read the dataset card from docs/DATASET_CARD.md."""
    card_path = DOCS_DIR / "DATASET_CARD.md"
    if card_path.exists():
        return card_path.read_text(encoding="utf-8")
    # Fall back to README
    readme = _HERE / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""


def count_records() -> dict[str, int]:
    """Count records per source directory."""
    counts: dict[str, int] = {}
    if not DATA_DIR.exists():
        return counts
    for source_dir in sorted(DATA_DIR.iterdir()):
        if source_dir.is_dir() and not source_dir.name.startswith("."):
            n = sum(1 for _ in source_dir.rglob("*.json"))
            if n > 0:
                counts[source_dir.name] = n
    return counts


def upload(repo_id: str, dry_run: bool = False) -> None:
    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    print(f"Target repo:  {repo_id}")
    print(f"Data dir:     {DATA_DIR}")
    counts = count_records()
    total = sum(counts.values())
    print(f"Records:      {total} total across {len(counts)} sources")
    for src, n in sorted(counts.items()):
        print(f"  {src}: {n}")
    print()

    if dry_run:
        print("DRY-RUN mode — no upload performed.")
        print(f"Would upload {total} records to https://huggingface.co/datasets/{repo_id}")
        return

    # Create or verify repo exists
    print("Creating/verifying HF repo ...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )

    # Upload dataset card
    card_content = build_dataset_card()
    if card_content:
        print("Uploading dataset card ...")
        api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    # Upload schema
    schema_file = SCHEMA_DIR / "eval.schema.json"
    if schema_file.exists():
        print("Uploading schema ...")
        api.upload_file(
            path_or_fileobj=str(schema_file),
            path_in_repo="schema/eval.schema.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    # Upload data directory
    if DATA_DIR.exists():
        print(f"Uploading {total} records from data/ ...")
        upload_folder(
            folder_path=str(DATA_DIR),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="data",
            token=token,
            ignore_patterns=["*.DS_Store", "__pycache__"],
        )

    # Upload scripts
    scripts_dir = _HERE / "scripts"
    if scripts_dir.exists():
        print("Uploading extraction scripts ...")
        upload_folder(
            folder_path=str(scripts_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="scripts",
            token=token,
            ignore_patterns=["*.DS_Store", "__pycache__", "*.pyc", "raw/"],
        )

    # Upload converters
    converters_dir = _HERE / "converters"
    if converters_dir.exists():
        print("Uploading converters ...")
        upload_folder(
            folder_path=str(converters_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="converters",
            token=token,
            ignore_patterns=["*.DS_Store", "__pycache__", "*.pyc"],
        )

    # Upload analysis output
    analysis_dir = _HERE / "analysis"
    if analysis_dir.exists():
        print("Uploading analysis scripts & output ...")
        upload_folder(
            folder_path=str(analysis_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="analysis",
            token=token,
            ignore_patterns=["*.DS_Store", "__pycache__", "*.pyc"],
        )

    # Upload validation script
    validate_script = _HERE / "validate_submission.py"
    if validate_script.exists():
        print("Uploading validation script ...")
        api.upload_file(
            path_or_fileobj=str(validate_script),
            path_in_repo="validate_submission.py",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    print(f"\n✓ Done — dataset available at: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload EEE dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    upload(repo_id=args.repo_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
