"""
fix_deepseekv2_records.py — Phase 5 fix.

papers_2405.04434 is actually the DeepSeek-V2 paper, NOT Qwen2.
Fix the source_metadata in those records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data" / "papers_2405.04434"

CORRECT_META = {
    "source_name": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
    "source_type": "documentation",
    "source_organization_name": "DeepSeek AI",
    "source_organization_url": "https://arxiv.org/abs/2405.04434",
    "evaluator_relationship": "first_party",
}

fixed = 0
for json_file in DATA_DIR.rglob("*.json"):
    record = json.loads(json_file.read_text())
    sm = record.get("source_metadata", {})
    if sm.get("source_name") != CORRECT_META["source_name"]:
        record["source_metadata"].update(CORRECT_META)
        # Also fix developer if "unknown" - DeepSeek-V2 models should be deepseek-ai
        model_dev = record.get("model_info", {}).get("developer", "unknown")
        if model_dev == "unknown":
            record["model_info"]["developer"] = "deepseek-ai"
            old_id = record["model_info"].get("id", "")
            if old_id.startswith("unknown/"):
                record["model_info"]["id"] = "deepseek-ai/" + old_id[len("unknown/"):]
        json_file.write_text(json.dumps(record, indent=2))
        print(f"  FIXED: {json_file.parent.name}")
        fixed += 1

print(f"\nFixed {fixed} records in papers_2405.04434")
