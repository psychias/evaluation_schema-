# Data Corrections Report — Docling Migration + ID Fix (Phases 2–7)

Generated: 2026-03-14

## Summary

This report documents:
1. The migration of `scripts/extract_paper.py` from pdfplumber to Docling
2. The batch re-extraction of all 52+ source papers
3. **CRITICAL FIX**: Correction of 6 wrong arXiv IDs discovered by independent verification

---

## Phase 2: Docling Migration

### What Changed in `extract_paper.py`

| Component | Change |
|---|---|
| `pdfplumber` import | Removed entirely |
| `DoclingParser` class | Added — shared converter with per-path result caching |
| `TableExtractor` | Migrated to `DoclingParser`; fixed `export_to_dataframe(doc)` API; added header-row prepend so benchmark keywords are visible to `ResultsTableParser` |
| `ProseExtractor` | Replaced pdfplumber page loop with `_docling_parser.get_full_text()` |
| `LLMFallbackExtractor` | Replaced pdfplumber + pytesseract with `_docling_parser.get_full_text()` |
| `_detect_eval_library()` | Replaced pdfplumber 8-page loop with Docling full-text export |

### Critical Bug Fixed
The original Docling `TableExtractor` called `df.values.tolist()` which omits
column headers. Since benchmark names (e.g. "MMLU", "GSM8K") appear in column
headers, the `ResultsTableParser` density/keyword check was failing for most
tables. Fix: prepend `[str(c) for c in df.columns]` as the first row.

---

## Phase 3: Batch Re-Extraction

### Extraction Results

- **Papers processed:** 52 original + 17 Phase 7 additions (69 total)
- **Papers with results:** ~35
- **Total JSON files:** ~10,865
- **Schema failures:** 0

---

## Phase 3c: Verification Results (Sample)

| Paper | Model | Benchmark | Expected | Extracted | Status |
|---|---|---|---|---|---|
| 2310.06825 | Mistral-7B-v0.1 | HellaSwag | 0.812 | 0.812 | ✓ EXACT |
| 2310.06825 | Mistral-7B-v0.1 | MMLU | 0.601 | 0.601 | ✓ EXACT |
| 2307.09288 | Llama-2-7B | MMLU | 0.453 | 0.467 | ✓ within ±0.015 |
| 2307.09288 | Llama-2-13B | MMLU | 0.548 | 0.543 | ✓ within ±0.015 |
| 2303.08774 | GPT-4 | MMLU | 0.864 | 0.864 | ✓ EXACT |
| 2403.08295 | Gemma-7B | MMLU | 0.643 | 0.643 | ✓ EXACT |
| 2404.14219 | Phi-3-mini | MMLU | 0.688 | 0.690 | ✓ within ±0.015 |
| 2501.12948 | DeepSeek-R1 | MMLU | 0.904 | 0.907 | ✓ within ±0.015 |

**8/10 checks pass** (2 MISSING: Mixtral-Instruct not present in corrected paper source)

---

## CRITICAL: Wrong arXiv IDs (Fixed 2026-03-14)

Independent verification discovered that 6 paper folders used WRONG arXiv IDs.
These caused data contamination (scores attributed to the wrong papers).

| Wrong ID | What it actually was | Correct ID | Correct Paper |
|---|---|---|---|
| 2309.10305 | Baichuan 2: Open Large-scale Language Models | **2310.06825** | Mistral 7B |
| 2312.11805 | Gemini: A Family of Highly Capable Multimodal Models | **2401.04088** | Mixtral of Experts |
| 2402.01322 | Large fluctuations in NSPT computations... (physics) | **2402.00838** | OLMo: Accelerating the Science of Language Models |
| 2403.05530 | Gemini 1.5: Unlocking multimodal understanding | **2403.08295** | Gemma: Open Models Based on Gemini Research and Technology |
| 2403.07691 | ORPO: Monolithic Preference Optimization... | N/A | Claude 3 has no public arXiv paper |
| 2501.15451 | STATE ToxiCN: A Benchmark... (Chinese toxicity) | N/A | Not a valid paper source |

### What Was Fixed

1. **Deleted** all 6 wrong-ID data folders from `shared_task_submission/data/`
2. **Re-extracted** papers 2310.06825, 2401.04088, 2402.00838, 2403.08295
3. **Updated** `scripts/create_paper_records.py`: SRC and AID variables in all 4 affected functions
4. **Updated** `scripts/analysis/collision_detection.py`: case study source labels
5. **Updated** `scripts/figures/fig5_case_study.py`: source labels
6. **Updated** `scripts/figures/fig7_prompt_anatomy.py`: paper reference labels
7. **Updated** `scripts/analysis/reanalysis_r2.py`: source filter references
8. **Updated** `scripts/verify_sample.py`: ground truth paper IDs
9. **Updated** `scripts/arxiv_ids_full.txt`: correct IDs, removed wrong ones
10. **Regenerated** all analysis outputs and figures

### Impact on Analysis

After correcting the IDs, the dataset now has **233 collision pairs** (vs 154 previously).
The variance decomposition results changed significantly:

| Predictor | Old (wrong IDs) | New (correct IDs) |
|---|---|---|
| harness_differs partial R² | 0.049 | **0.249** |
| prompt_template_differs partial R² | **0.136** | 0.002 |
| n_shot_differs partial R² | 0.003 | 0.003 |

The new result shows `harness_differs` (different evaluation framework) is the dominant
predictor of score variance. The old result was an artifact of the wrong-ID data contamination.

---

## Phase 5: Leaderboard Scrapers

All 6 leaderboard scrapers re-run successfully (unchanged from previous report).

---

## Mandatory Collision Pairs (Verified)

| Case | Pair | Delta | Sources |
|---|---|---|---|
| 1 | GSM8K / Mistral-7B-v0.1 | −0.225 | papers_2401.04088 vs papers_2403.08295 |
| 2 | HumanEval / Llama-2-7B | 0.000 | papers_2307.09288 vs papers_2310.06825 |
| 3 | HellaSwag / Mistral-7B-v0.1 | −0.021 | papers_2310.06825 vs papers_2401.04088 |

All 3 mandatory collision pairs **CONFIRMED PRESENT** with expected delta values.

---

## Dataset Statistics (Post-Fix)

- **Total JSON records:** ~10,865
- **Total paper sources:** ~52 (correct IDs)
- **Collision pairs:** 233
- **Schema validation:** 0 failures

---

## Phase 8: LaTeX Re-Extraction + Value-Reuse Bug Fix (2026-03-15)

### Root Cause: Value-Reuse Bug (Systematic, All Papers Affected)

The pdfplumber-based pipeline cached ONE value per `(model_id, benchmark)` and reused it across ALL paper records. For example, Mistral-7B's GSM8K score was scraped once as `0.352` and written identically into every paper — so all 50+ papers that mentioned Mistral-7B showed GSM8K=0.352, regardless of what score each paper actually reported in its comparison table.

This broke the fundamental purpose of EEE: capturing how different papers report *different* scores for the same model.

### Fix: `scripts/extract_latex.py`

Built a new LaTeX-primary extraction engine:
1. Downloads `.tar.gz` LaTeX source from `https://arxiv.org/src/{arxiv_id}`
2. Expands all `\newcommand`/`\def` macros before parsing
3. Parses each `tabular` environment independently per paper
4. Extracts `(model, benchmark, score)` triples from each paper's own table independently
5. Falls back to Docling for papers with complex table formatting

### Parser Bugs Fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| `\%` not stripped in `clean_cell` | `44.4\%` → None in parse_score | Added `cell.replace(r"\%", "%")` |
| `~` (LaTeX non-breaking space) surviving | `Mistral~7B` as model name | Added `cell.replace("~", " ")` |
| Nested column spec `{lccc@{}}` leaking | First row cell was `lccccc@\nModel` | Added `_remove_brace_group()` |
| `is_numeric()` failing on `%` values | All data rows skipped as non-numeric | Added `rstrip("%")` in is_numeric |

### Extraction Results

```
Papers attempted:   65
Successfully extracted: 57  (55 LaTeX, 2 Docling-only)
Docling fallback:   10  (complex table structures)
Failed (0 triples):  8  (complex formatting or unavailable source)
JSON records written: 998
```

**Failed papers:** 2205.01068 (OPT), 2211.05100 (BLOOM), 2304.10457 (MPT-7B),
2404.10774 (JetMoE), 2407.12511 (Jamba 1.5 mini), 2411.14599 (Zamba2-7B),
2411.15138 (EXAONE 3.5), 2501.12599 (Kimi k1.5)

### Verified Corrections (Before → After)

| Model | Benchmark | Paper | Before | After |
|-------|-----------|-------|--------|-------|
| Mistral-7B | GSM8K | Mixtral (2401.04088) | 0.352 | 0.500 |
| Mistral-7B | GSM8K | Mistral (2310.06825) | 0.352 | 0.522 |
| Mistral-7B | MMLU | Mixtral (2401.04088) | 0.601 | 0.625 |
| Mistral-7B | MBPP | Mistral (2310.06825) | 0.413 | 0.475 |
| Mistral-7B | MBPP | Mixtral (2401.04088) | 0.413 | 0.502 |
| InternLM2-20B | MMLU | InternLM2 (2403.17297) | 0.643 | 0.676 |
| InternLM2-20B | HumanEval | InternLM2 (2403.17297) | 0.396 | 0.424 |

### Dataset Statistics (After Phase 8)

- **Total JSON records:** 998 (paper sources only; leaderboard data unchanged)
- **Total paper sources:** 57 (extracted)
- **Collision pairs:** 253
- **Unique models in collisions:** 103
- **Unique source pairs:** 52
- **MMLU n-shot p-value:** 0.011 (BH-FDR q=0.028)
- **MMLU prompt_template p-value:** <0.001 (BH-FDR q<0.001)
