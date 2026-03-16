#!/usr/bin/env python3
"""Quick check of evaluator_relationship values across all JSON records."""
import json, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

rels = {}
for sub in sorted(os.listdir(DATA)):
    path = os.path.join(DATA, sub)
    if not os.path.isdir(path):
        continue
    for f in sorted(glob.glob(os.path.join(path, "**/*.json"), recursive=True)):
        try:
            rec = json.load(open(f))
            rel = rec.get("source_metadata", {}).get("evaluator_relationship", "MISSING")
            if sub not in rels:
                rels[sub] = set()
            rels[sub].add(rel)
        except Exception:
            pass

all_rels = set()
for v in rels.values():
    all_rels.update(v)
print("All evaluator_relationship values:", sorted(all_rels))
print()
for src in sorted(rels.keys()):
    print(f"  {src}: {sorted(rels[src])}")
print(f"\n  Total sources: {len(rels)}")
