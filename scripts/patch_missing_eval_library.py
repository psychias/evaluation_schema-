"""
patch_missing_eval_library.py — Add missing eval_library field to existing JSON records.

The original leaderboard scrapers were run before eval_library was added to the schema.
This script patches the records in-place, adding the appropriate eval_library based on
the data source directory name.
"""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"

SOURCE_EVAL_LIBRARY = {
    "hfopenllm_v2": {"name": "lighteval", "version": "unknown"},
    "reward-bench": {"name": "reward-bench", "version": "unknown"},
    "helm_lite": {"name": "helm", "version": "unknown"},
    "helm_classic": {"name": "helm", "version": "unknown"},
    "helm_instruct": {"name": "helm", "version": "unknown"},
    "helm_capabilities": {"name": "helm", "version": "unknown"},
    "helm_mmlu": {"name": "helm", "version": "unknown"},
    "livecodebenchpro": {"name": "livecodebench", "version": "unknown"},
    "global-mmlu-lite": {"name": "unknown", "version": "unknown"},
    "test_eval": {"name": "unknown", "version": "unknown"},
}

total_patched = 0
for source_name, eval_lib in SOURCE_EVAL_LIBRARY.items():
    src_dir = DATA_DIR / source_name
    if not src_dir.exists():
        continue
    patched = 0
    for f in src_dir.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "eval_library" not in d:
            d["eval_library"] = eval_lib
            f.write_text(json.dumps(d, indent=2, ensure_ascii=False))
            patched += 1
    print(f"  {source_name}: patched {patched} records")
    total_patched += patched

print(f"\nTotal patched: {total_patched}")
