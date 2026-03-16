#!/usr/bin/env python3
"""Audit prompt_template coverage per source."""
import json
from pathlib import Path

data_dir = Path("data")

# Papers
paper_dirs = sorted([d for d in data_dir.iterdir() if d.name.startswith("papers_")])
print("=== Paper sources ===")
for pd in paper_dirs:
    total_evals = 0
    non_standard = 0
    for jf in pd.rglob("*.json"):
        try:
            rec = json.loads(jf.read_text())
            for er in rec.get("evaluation_results", []):
                total_evals += 1
                gc = er.get("generation_config", {})
                ad = gc.get("additional_details", {})
                pt = ad.get("prompt_template", "")
                if pt and pt != "standard":
                    non_standard += 1
        except Exception:
            pass
    pct = 100 * non_standard / total_evals if total_evals else 0
    print(f"  {pd.name}: {non_standard}/{total_evals} non-standard ({pct:.1f}%)")

# Leaderboards
print("\n=== Leaderboard sources ===")
lb_dirs = ["alpacaeval2", "bigcodebench", "chatbot_arena", "hfopenllm_v2", "mt_bench", "wildbench"]
for lb in lb_dirs:
    lbp = data_dir / lb
    if not lbp.exists():
        continue
    total_evals = 0
    has_pt = 0
    for jf in lbp.rglob("*.json"):
        try:
            rec = json.loads(jf.read_text())
            for er in rec.get("evaluation_results", []):
                total_evals += 1
                gc = er.get("generation_config", {})
                ad = gc.get("additional_details", {})
                pt = ad.get("prompt_template", "")
                if pt and pt != "standard":
                    has_pt += 1
        except Exception:
            pass
    pct = 100 * has_pt / total_evals if total_evals else 0
    print(f"  {lb}: {has_pt}/{total_evals} non-standard ({pct:.1f}%)")

# Summary: how many sources have ANY non-standard prompt_template
print("\n=== Source-level summary (has >=1 non-standard prompt_template) ===")
total_sources = 0
documented_sources = 0
for pd in paper_dirs:
    total_sources += 1
    has_any = False
    for jf in pd.rglob("*.json"):
        try:
            rec = json.loads(jf.read_text())
            for er in rec.get("evaluation_results", []):
                gc = er.get("generation_config", {})
                ad = gc.get("additional_details", {})
                pt = ad.get("prompt_template", "")
                if pt and pt != "standard":
                    has_any = True
                    break
        except Exception:
            pass
        if has_any:
            break
    if has_any:
        documented_sources += 1
        print(f"  YES: {pd.name}")
    else:
        print(f"  no:  {pd.name}")

for lb in lb_dirs:
    lbp = data_dir / lb
    if not lbp.exists():
        continue
    total_sources += 1
    has_any = False
    for jf in lbp.rglob("*.json"):
        try:
            rec = json.loads(jf.read_text())
            for er in rec.get("evaluation_results", []):
                gc = er.get("generation_config", {})
                ad = gc.get("additional_details", {})
                pt = ad.get("prompt_template", "")
                if pt and pt != "standard":
                    has_any = True
                    break
        except Exception:
            pass
        if has_any:
            break
    if has_any:
        documented_sources += 1
        print(f"  YES: {lb}")
    else:
        print(f"  no:  {lb}")

print(f"\nTotal: {documented_sources}/{total_sources} sources with non-standard prompt_template")
