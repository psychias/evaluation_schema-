#!/usr/bin/env python3
"""Check which sources use 'standard' placeholder for prompt_template."""
import json, glob, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

source_pt_values = defaultdict(set)

for sub in sorted(os.listdir(DATA)):
    path = os.path.join(DATA, sub)
    if not os.path.isdir(path):
        continue
    for f in sorted(glob.glob(os.path.join(path, "**/*.json"), recursive=True)):
        try:
            rec = json.load(open(f))
            for er in rec.get("evaluation_results", []):
                gc = er.get("generation_config", {})
                ad = gc.get("additional_details", {})
                pt = ad.get("prompt_template", "")
                if pt:
                    source_pt_values[sub].add(pt)
                else:
                    source_pt_values[sub].add("__MISSING__")
        except Exception:
            pass

# Classify sources
for src in sorted(source_pt_values.keys()):
    vals = source_pt_values[src]
    has_standard = "standard" in vals
    has_missing = "__MISSING__" in vals
    non_standard = vals - {"standard", "__MISSING__"}
    
    if non_standard:
        status = f"PARTIAL ({len(non_standard)} non-std templates)"
    elif has_standard and not has_missing:
        status = "ALL STANDARD (placeholder)"
    elif has_standard and has_missing:
        status = "STANDARD + MISSING"
    elif has_missing:
        status = "ALL MISSING"
    else:
        status = "UNKNOWN"
    
    print(f"  {src}: {status}  vals={sorted(vals)[:4]}")
