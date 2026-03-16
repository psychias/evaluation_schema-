#!/usr/bin/env python3
"""
patch_docling_nshot.py — Patch n_shot values into paper records using
Docling-extracted eval details tables.

Currently targets: 2503.19786 (Gemma 3) using already-extracted
results/eval_details_table_18.csv (paper's Table 19).

For each evaluation result with n_shot='unknown', looks up the benchmark
name in the eval details table and fills in the n_shot value.

Also patches 'prompt_template' with 'cot' for benchmarks flagged as
chain-of-thought in the table (BBH, MATH, GSM8K, GPQA Diamond, MMLU-Pro).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ── Benchmark name mapping: EEE record names → Table row names ────────────
# Keys: lowercase normalized EEE evaluation_name
# Values: (n_shot_str, has_cot)
NSHOT_FROM_TABLE = {
    "mbpp":             ("3", False),
    "humaneval":        ("0", False),
    "hellaswag":        ("10", False),
    "boolq":            ("0", False),
    "piqa":             ("0", False),
    "siqa":             ("0", False),
    "triviaqa":         ("5", False),
    "naturalquestions": ("5", False),
    "natural questions": ("5", False),
    "nq":               ("5", False),
    "arc-c":            ("25", False),
    "arc-challenge":    ("25", False),
    "arc-e":            ("0", False),
    "arc-easy":         ("0", False),
    "winogrande":       ("5", False),
    "bbh":              ("few-shot", True),
    "drop":             ("1", False),
    "ageval":           ("3-5-shot", False),
    "agieval":          ("3-5-shot", False),
    "mmlu":             ("5", False),
    "math":             ("4", True),
    "gsm8k":            ("8", True),
    "gpqa":             ("5", True),
    "gpqa diamond":     ("5", True),
    "mmlu-pro":         ("5", True),
    "mgsm":             ("8", False),
    "flores":           ("1", False),
    "global-mmlu":      ("5", False),
    "global-mmlu-lite": ("5", False),
    "xquad":            ("5", False),
    "wmt24":            ("5", False),
    "wmt24++":          ("5", False),
    "ruler":            (None, False),   # not specified
    "mrcr":             ("few-shot", False),
    "ifeval":           (None, False),   # not in table
}


def normalize_bench_name(name: str) -> str:
    return name.lower().strip()


def patch_paper(arxiv_id: str, dry_run: bool = False) -> dict:
    folder = _ROOT / "data" / f"papers_{arxiv_id}"
    jsons = [j for j in folder.rglob("*.json") if j.is_file()]
    stats = {"files": len(jsons), "patched_nshot": 0, "patched_cot": 0, "skipped": 0}

    for jf in jsons:
        r = json.loads(jf.read_text())
        modified = False

        for ev in r.get("evaluation_results", []):
            bench = ev.get("evaluation_name", "")
            bench_norm = normalize_bench_name(bench)

            gc = ev.setdefault("generation_config", {})
            ad = gc.setdefault("additional_details", {})

            match = NSHOT_FROM_TABLE.get(bench_norm)
            if match is None:
                stats["skipped"] += 1
                continue

            nshot_val, has_cot = match

            # Patch n_shot if currently unknown
            if ad.get("n_shot") == "unknown" and nshot_val is not None:
                ad["n_shot"] = nshot_val
                modified = True
                stats["patched_nshot"] += 1
                print(f"  {jf.parent.name}/{jf.name[:8]}: {bench} → n_shot={nshot_val}")

            # Patch prompt_template with 'cot' if CoT-flagged and currently empty/standard
            if has_cot:
                pt = ad.get("prompt_template", "")
                if not pt or pt in ("standard", "unknown", ""):
                    ad["prompt_template"] = "cot"
                    modified = True
                    stats["patched_cot"] += 1
                    print(f"  {jf.parent.name}/{jf.name[:8]}: {bench} → prompt_template=cot")

        if modified and not dry_run:
            jf.write_text(json.dumps(r, indent=2, ensure_ascii=False))

    return stats


def main():
    import sys
    dry_run = "--dry-run" in sys.argv

    papers_to_patch = [
        "2503.19786",   # Gemma 3 — eval details table extracted via Docling
    ]

    for arxiv_id in papers_to_patch:
        folder = _ROOT / "data" / f"papers_{arxiv_id}"
        if not folder.exists():
            print(f"SKIP {arxiv_id}: no data folder")
            continue
        print(f"\nPatching {arxiv_id} {'(dry run)' if dry_run else ''}...")
        stats = patch_paper(arxiv_id, dry_run=dry_run)
        print(f"  → {stats['files']} files, "
              f"{stats['patched_nshot']} n_shot patches, "
              f"{stats['patched_cot']} cot patches, "
              f"{stats['skipped']} unmapped benchmarks skipped")

    if dry_run:
        print("\nDry run — no files written.")
    else:
        print("\nDone. Run analysis scripts to regenerate outputs.")


if __name__ == "__main__":
    main()
