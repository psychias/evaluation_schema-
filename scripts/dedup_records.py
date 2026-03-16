"""
dedup_records.py — Remove duplicate JSON records in data/papers_*/

When re-extraction runs without deleting the existing directory, it creates
duplicate records for the same model in the same paper. This script detects
and removes duplicates, keeping the record with the earlier timestamp (original).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"

def main():
    # Group records by paper + model_name
    # Key: (arxiv_id, model_name, developer)
    groups = defaultdict(list)

    for paper_dir in sorted(DATA_DIR.glob("papers_*")):
        arxiv_id = paper_dir.name.replace("papers_", "")
        for json_file in paper_dir.rglob("*.json"):
            try:
                record = json.loads(json_file.read_text())
            except Exception as e:
                print(f"  ERROR reading {json_file}: {e}")
                continue

            model_name = record.get("model_info", {}).get("name", "")
            model_dev = record.get("model_info", {}).get("developer", "unknown")
            key = (arxiv_id, model_name, model_dev)
            timestamp = float(record.get("retrieved_timestamp", "0"))
            groups[key].append((timestamp, json_file))

    deleted = 0
    for key, entries in groups.items():
        if len(entries) <= 1:
            continue
        # Sort by timestamp, keep the earliest (original)
        entries.sort(key=lambda x: x[0])
        original = entries[0]
        duplicates = entries[1:]

        arxiv_id, model_name, dev = key
        print(f"  DUPLICATE: {arxiv_id} / {model_name} ({dev}) - {len(duplicates)} copies to delete")

        for ts, dup_path in duplicates:
            try:
                dup_path.unlink()
                # Clean up empty parent dirs
                parent = dup_path.parent
                while parent != DATA_DIR and parent.exists():
                    try:
                        parent.rmdir()
                        parent = parent.parent
                    except OSError:
                        break
                print(f"    DELETED: {dup_path.name}")
                deleted += 1
            except Exception as e:
                print(f"    ERROR deleting {dup_path}: {e}")

    print(f"\nDeduplication complete. Deleted {deleted} duplicate records.")

if __name__ == "__main__":
    main()
