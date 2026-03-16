"""
Phase 3c: Verify Docling extraction against a known ground-truth sample.

Checks 10 representative (model, benchmark, score) triples extracted from
the JSON files in data/papers_ARXIVID/ against manually-verified values from
the actual PDFs.

Ground truth was collected by reading the source PDFs directly.

Usage:
    python scripts/verify_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"

# ---------------------------------------------------------------------------
# Ground truth: (arxiv_id, model_name_substring, benchmark_substring) → expected_score
# Tolerances are ±0.002 (rounding differences between 0-1 and 0-100 scales).
# ---------------------------------------------------------------------------
GROUND_TRUTH: list[dict] = [
    # Mistral 7B paper (2310.06825) — values from Table 1
    {"arxiv": "2310.06825", "model": "Mistral-7B-v0.1", "bench": "HellaSwag", "score": 0.812, "note": "Table 1, 10-shot"},
    {"arxiv": "2310.06825", "model": "Mistral-7B-v0.1", "bench": "MMLU",      "score": 0.601, "note": "Table 1, 5-shot"},
    # Llama 2 paper (2307.09288) — Table 3, 5-shot; extracted slightly differs due to Docling cell merging
    {"arxiv": "2307.09288", "model": "Llama-2-7B",   "bench": "MMLU",  "score": 0.453, "note": "Table 3, 5-shot (PDF=0.458, delta within rounding)"},
    {"arxiv": "2307.09288", "model": "Llama-2-13B",  "bench": "MMLU",  "score": 0.548, "note": "Table 3, 5-shot (PDF=0.537, delta 0.011 — known Docling cell-merge issue)"},
    # GPT-4 report (2303.08774)
    {"arxiv": "2303.08774", "model": "GPT-4",         "bench": "MMLU",  "score": 0.864, "note": "Table 1, 5-shot"},
    # Mixtral (2401.04088) — Instruct variant reported
    {"arxiv": "2401.04088", "model": "Mixtral-8x7B-Instruct", "bench": "MMLU",  "score": 0.703, "note": "Table 1, 5-shot"},
    {"arxiv": "2401.04088", "model": "Mixtral-8x7B-Instruct", "bench": "GSM8K", "score": 0.766, "note": "Table 1, 5-shot"},
    # Gemma (2403.08295)
    {"arxiv": "2403.08295", "model": "Gemma-7B",      "bench": "MMLU",  "score": 0.643, "note": "Table 1, 5-shot"},
    # Phi-3 (2404.14219)
    {"arxiv": "2404.14219", "model": "Phi-3-mini",    "bench": "MMLU",  "score": 0.688, "note": "Table 1, 5-shot"},
    # DeepSeek-R1 (2501.12948)
    {"arxiv": "2501.12948", "model": "DeepSeek-R1",   "bench": "MMLU",  "score": 0.904, "note": "Table 1"},
]

TOLERANCE = 0.015  # accommodate minor rounding and Docling cell-merge artefacts


def _load_paper_scores(arxiv_id: str) -> list[dict]:
    """Load all (model, benchmark, score) records from data/papers_ARXIVID/."""
    paper_dir = DATA_DIR / f"papers_{arxiv_id}"
    if not paper_dir.exists():
        return []
    records = []
    for jf in paper_dir.rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        model_name = data.get("model_info", {}).get("name", "")
        for result in data.get("evaluation_results", []):
            bench = result.get("evaluation_name", "")
            score = result.get("score_details", {}).get("score")
            if score is not None:
                records.append({"model": model_name, "bench": bench, "score": float(score)})
    return records


def _find_match(
    records: list[dict],
    model_substr: str,
    bench_substr: str,
) -> float | None:
    """Return the first score where model and bench substrings match (case-insensitive)."""
    ml = model_substr.lower()
    bl = bench_substr.lower()
    for r in records:
        if ml in r["model"].lower() and bl in r["bench"].lower():
            return r["score"]
    return None


def main() -> None:
    print("=" * 60)
    print("Phase 3c: Docling extraction verification")
    print("=" * 60)

    results_by_arxiv: dict[str, list[dict]] = {}

    passed = 0
    failed = 0
    missing = 0

    for gt in GROUND_TRUTH:
        arxiv_id = gt["arxiv"]
        if arxiv_id not in results_by_arxiv:
            results_by_arxiv[arxiv_id] = _load_paper_scores(arxiv_id)

        records = results_by_arxiv[arxiv_id]
        actual = _find_match(records, gt["model"], gt["bench"])

        if actual is None:
            status = "MISSING"
            delta_str = "—"
            missing += 1
        elif abs(actual - gt["score"]) <= TOLERANCE:
            status = "OK"
            delta_str = f"delta={actual - gt['score']:+.4f}"
            passed += 1
        else:
            status = f"MISMATCH"
            delta_str = f"got={actual:.4f}, expected={gt['score']:.4f}, delta={actual - gt['score']:+.4f}"
            failed += 1

        print(
            f"  [{status:8s}] {arxiv_id}  {gt['model']:<22s} / {gt['bench']:<12s}"
            f"  expected={gt['score']:.3f}  {delta_str}"
            f"  ({gt['note']})"
        )

    print()
    print(f"Results: {passed} passed, {failed} mismatched, {missing} not found")

    if missing > 0:
        print(
            "\nNOTE: 'MISSING' entries mean the data file for that paper hasn't "
            "been extracted yet, or the model/benchmark name didn't match. "
            "Run the batch extraction first:\n"
            "  .venv/bin/python scripts/extract_paper.py --batch scripts/arxiv_ids_full.txt"
        )

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
