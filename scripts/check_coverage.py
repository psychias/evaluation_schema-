#!/usr/bin/env python3
"""Check record-level coverage percentages."""
import json, pathlib

recs = list(pathlib.Path("data").rglob("*.json"))
n_shot_present = 0
prompt_present = 0
temp_present = 0
harness_present = 0
total = 0

for p in recs:
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    el = d.get("eval_library", {})
    has_harness = el.get("name") not in (None, "", "N/A", "unknown")
    for r in d.get("evaluation_results", []):
        total += 1
        gc = r.get("generation_config", {})
        ad = gc.get("additional_details", {})
        if ad.get("n_shot") not in (None, "", "N/A", "unknown"):
            n_shot_present += 1
        if ad.get("prompt_template") not in (None, "", "N/A", "unknown", "standard"):
            prompt_present += 1
        if ad.get("temperature") not in (None, "", "N/A", "unknown"):
            temp_present += 1
        if has_harness:
            harness_present += 1

print(f"Total eval results: {total}")
print(f"n_shot: {n_shot_present}/{total} = {100*n_shot_present/total:.1f}%")
print(f"prompt_template: {prompt_present}/{total} = {100*prompt_present/total:.1f}%")
print(f"temperature: {temp_present}/{total} = {100*temp_present/total:.1f}%")
print(f"harness: {harness_present}/{total} = {100*harness_present/total:.1f}%")
