"""
collision_detection.py — find (model_id, benchmark) pairs reported by 2+ sources
with different scores OR with the same score but different evaluation methodology.

Deduplication rules (applied in order):
  1. Within-source deduplication: for each (source, model_id, benchmark),
     keep a single canonical row (median score; most-common n_shot/harness).
  2. Citation-duplicate exclusion: exclude source pairs where
         score_a == score_b  AND  n_shot_a == n_shot_b  AND  harness_a == harness_b
     These represent the same underlying evaluation run cited in multiple papers,
     not an independent re-evaluation.

Pairs that survive are genuine score collisions OR same-score pairs with
differing evaluation methodology (e.g. Case 2: HumanEval / Llama-2-7b with
score=0.122 in both papers but different harness frameworks).

Output: analysis_output/collision_pairs.csv
Columns: model_id, benchmark, source_a, source_b, score_a, score_b, delta,
         n_shot_a, n_shot_b, harness_a, harness_b, prompt_template_a, prompt_template_b
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "data"
OUT_DIR = _ROOT / "analysis_output"


def _get_gen_field(gen_cfg: dict | None, field: str) -> str:
    if gen_cfg is None:
        return ""
    details = gen_cfg.get("additional_details") or {}
    return str(details.get(field, ""))


def load_all_records() -> pd.DataFrame:
    """Load every JSON record into a flat DataFrame."""
    rows = []
    for source_dir in sorted(DATA_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for fpath in source_dir.rglob("*.json"):
            try:
                rec = json.loads(fpath.read_text())
            except Exception:
                continue
            model_id = rec.get("model_info", {}).get("id", "")
            harness_top = rec.get("eval_library", {}).get("name", "")
            for result in rec.get("evaluation_results", []):
                bench = result.get("evaluation_name", "")
                score = result.get("score_details", {}).get("score")
                gen_cfg = result.get("generation_config")
                n_shot = _get_gen_field(gen_cfg, "n_shot")
                prompt_template = _get_gen_field(gen_cfg, "prompt_template")
                # prefer per-result harness if present, fall back to top-level
                harness_result = _get_gen_field(gen_cfg, "harness")
                harness = harness_result if harness_result else harness_top
                if score is not None:
                    rows.append({
                        "model_id": model_id,
                        "benchmark": bench,
                        "source": source,
                        "score": float(score),
                        "n_shot": n_shot,
                        "harness": harness,
                        "prompt_template": prompt_template,
                    })
    return pd.DataFrame(rows)


def _most_common(series: pd.Series) -> str:
    counts = Counter(series.dropna().astype(str).tolist())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def deduplicate_within_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (source, model_id, benchmark) group that has multiple rows
    (arising from multiple JSON files per model in one source dir),
    collapse to a single canonical row:
      - score: median
      - n_shot, harness, prompt_template: most common value
    """
    result = []
    for (source, model_id, benchmark), grp in df.groupby(
        ["source", "model_id", "benchmark"], sort=False
    ):
        result.append({
            "source": source,
            "model_id": model_id,
            "benchmark": benchmark,
            "score": float(grp["score"].median()),
            "n_shot": _most_common(grp["n_shot"]),
            "harness": _most_common(grp["harness"]),
            "prompt_template": _most_common(grp["prompt_template"]),
        })
    return pd.DataFrame(result)


def is_citation_duplicate(row: dict) -> bool:
    """
    Return True when both sources report the same underlying evaluation run.

    Two criteria must both hold:
      1. Scores are identical (delta == 0).
      2. Both sources use the same harness framework.

    When criteria are met, the two paper records are citing the same run even if
    one uses 0-shot and the other uses 5-shot in its table header — the score
    identity combined with framework identity is sufficient evidence.

    Pairs that survive are either:
      • genuine score collisions (|delta| > 0), OR
      • same-score pairs with *different* harness frameworks — i.e., independent
        evaluations that happen to agree (e.g. Case 2: HumanEval / Llama-2-7b,
        score=0.122 in both papers but one uses meta-internal and the other
        uses lm-evaluation-harness).
    """
    if abs(row["delta"]) > 1e-9:
        return False  # genuine score difference — never a citation duplicate
    # Same score AND same harness → treat as citation duplicate regardless of n_shot
    return row["harness_a"] == row["harness_b"]


def detect_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Find (model_id, benchmark) pairs with entries in 2+ sources,
    excluding citation duplicates, then collapse to one canonical pair
    per (model_id, benchmark).

    Collapsing strategy (applied separately to genuine and methodology pairs):
      - Genuine (|delta| > 0):  keep the pair with the LARGEST |delta|.
        When the same (model, benchmark) appears in N sources the C(N,2)
        pairwise combinations all show the same phenomenon; the max-delta pair
        is the most informative representative.
      - Zero-delta (different harness/framework): keep the pair with the MOST
        different harness labels (prefer cross-organisation framework pairs,
        e.g. meta-internal vs lm-evaluation-harness over two lm-eval variants).
    """
    pairs = []
    grouped = df.groupby(["model_id", "benchmark"])
    for (model_id, benchmark), group in grouped:
        sources = sorted(group["source"].unique())
        if len(sources) < 2:
            continue
        # Build a lookup: source → canonical row
        src_rows = {
            src: group[group["source"] == src].iloc[0]
            for src in sources
        }
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sa, sb = sources[i], sources[j]
                row_a = src_rows[sa]
                row_b = src_rows[sb]
                score_a = float(row_a["score"])
                score_b = float(row_b["score"])
                delta = round(score_a - score_b, 6)
                candidate = {
                    "model_id": model_id,
                    "benchmark": benchmark,
                    "source_a": sa,
                    "source_b": sb,
                    "score_a": score_a,
                    "score_b": score_b,
                    "delta": delta,
                    "n_shot_a": str(row_a["n_shot"]),
                    "n_shot_b": str(row_b["n_shot"]),
                    "harness_a": str(row_a["harness"]),
                    "harness_b": str(row_b["harness"]),
                    "prompt_template_a": str(row_a["prompt_template"]),
                    "prompt_template_b": str(row_b["prompt_template"]),
                }
                if not is_citation_duplicate(candidate):
                    pairs.append(candidate)

    if not pairs:
        return pd.DataFrame()

    raw = pd.DataFrame(pairs)

    # ------------------------------------------------------------------
    # Per-(model_id, benchmark) deduplication for zero-delta pairs only:
    # When N sources all report the same score for the same (model, benchmark),
    # C(N,2) methodology pairs are generated — all describing the same
    # phenomenon. We collapse these to ONE representative pair per
    # (model_id, benchmark), choosing the pair with the most different
    # harness frameworks (proxy for maximum methodological independence).
    #
    # Genuine (|delta| > 0) pairs are kept in full: each source pair with
    # a different score represents a distinct evaluation discrepancy worth
    # reporting, and downstream OLS / power analyses need this granularity.
    # ------------------------------------------------------------------
    genuine_df = raw[raw["delta"].abs() > 1e-9].copy()
    methodology_df = raw[raw["delta"].abs() <= 1e-9].copy()

    kept = [genuine_df] if not genuine_df.empty else []

    for (model_id, benchmark), grp in methodology_df.groupby(["model_id", "benchmark"]):
        # Pick the pair with most different harness labels
        def harness_diff(row):
            a, b = str(row["harness_a"]), str(row["harness_b"])
            common = len(set(a.split("-")) & set(b.split("-")))
            return -common  # negate so that more-different → higher

        grp = grp.copy()
        grp["_hdiff"] = grp.apply(harness_diff, axis=1)
        best_meth = grp.loc[[grp["_hdiff"].idxmax()]].drop(columns=["_hdiff"])
        kept.append(best_meth)

    result = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if not result.empty:
        result = result.sort_values("delta", key=abs, ascending=False)
    return result


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading records...")
    df = load_all_records()
    print(f"  {len(df)} benchmark rows from {df['source'].nunique()} sources")

    print("Deduplicating within-source...")
    df_dedup = deduplicate_within_source(df)
    print(f"  {len(df_dedup)} rows after within-source deduplication")

    print("Detecting score collisions (excluding citation duplicates)...")
    collisions = detect_collisions(df_dedup)
    n_total = len(collisions)
    n_genuine = int((collisions["delta"].abs() > 1e-9).sum()) if not collisions.empty else 0
    n_methodology = n_total - n_genuine
    print(f"  {n_total} collision pairs found")
    print(f"    {n_genuine} with genuine score differences (|delta| > 0)")
    print(f"    {n_methodology} with same score but different methodology")

    out_path = OUT_DIR / "collision_pairs.csv"
    collisions.to_csv(out_path, index=False)
    print(f"  saved → {out_path}")

    if not collisions.empty:
        print("\nTop 15 by |delta|:")
        print(collisions.head(15)[
            ["model_id", "benchmark", "source_a", "source_b", "score_a", "score_b", "delta"]
        ].to_string(index=False))

        # Verify 3 mandatory case studies
        print("\n--- Mandatory case study verification ---")
        for label, mid, bench, sa, sb in [
            ("Case 1 (GSM8K/Mistral-7B-v0.1)",
             "mistralai/Mistral-7B-v0.1", "GSM8K",
             "papers_2312.11805", "papers_2403.05530"),
            ("Case 2 (HumanEval/Llama-2-7b)",
             "meta-llama/Llama-2-7b", "HumanEval",
             "papers_2307.09288", "papers_2309.10305"),
            ("Case 3 (HellaSwag/Mistral-7B-v0.1)",
             "mistralai/Mistral-7B-v0.1", "HellaSwag",
             "papers_2309.10305", "papers_2312.11805"),
        ]:
            hit = collisions[
                (collisions["model_id"] == mid)
                & (collisions["benchmark"] == bench)
                & (
                    ((collisions["source_a"] == sa) & (collisions["source_b"] == sb))
                    | ((collisions["source_a"] == sb) & (collisions["source_b"] == sa))
                )
            ]
            status = "FOUND" if not hit.empty else "MISSING"
            if not hit.empty:
                r = hit.iloc[0]
                print(f"  {status}: {label} delta={r['delta']:+.3f}")
            else:
                print(f"  {status}: {label}")


if __name__ == "__main__":
    main()
