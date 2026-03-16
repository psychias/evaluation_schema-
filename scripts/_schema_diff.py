#!/usr/bin/env python3
"""Compare local eval.schema.json with upstream GitHub version."""
import json, urllib.request, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
url = "https://raw.githubusercontent.com/evaleval/every_eval_ever/main/eval.schema.json"

with urllib.request.urlopen(url) as resp:
    upstream = json.loads(resp.read())

with open(ROOT / "eval.schema.json") as f:
    local = json.load(f)

print(f"Upstream version: {upstream.get('version', 'N/A')}")
print(f"Local    version: {local.get('version', 'N/A')}")
print()

if upstream == local:
    print("SCHEMAS IDENTICAL — no drift detected.")
    sys.exit(0)

def diff_dict(a, b, path=""):
    diffs = []
    all_keys = set(list(a.keys()) + list(b.keys()))
    for k in sorted(all_keys):
        p = f"{path}.{k}" if path else k
        if k not in a:
            diffs.append(f"  UPSTREAM ONLY: {p}")
        elif k not in b:
            diffs.append(f"  LOCAL ONLY:    {p}")
        elif a[k] != b[k]:
            if isinstance(a[k], dict) and isinstance(b[k], dict):
                diffs.extend(diff_dict(a[k], b[k], p))
            elif isinstance(a[k], list) and isinstance(b[k], list):
                diffs.append(f"  DIFFER at {p}: (list lengths {len(a[k])} vs {len(b[k])})")
            else:
                diffs.append(f"  DIFFER at {p}:")
                diffs.append(f"    upstream: {json.dumps(a[k])[:200]}")
                diffs.append(f"    local:    {json.dumps(b[k])[:200]}")
    return diffs

diffs = diff_dict(upstream, local)
print(f"SCHEMAS DIFFER — {len(diffs)} difference(s):")
for d in diffs:
    print(d)
sys.exit(1)
