# Every Eval Ever — Shared Task Submission

**Track 1: Public Eval Data Parsing**

Submission for the [Every Eval Ever Shared Task](https://evaleval.github.io/) at ACL 2026 EvalEval Workshop.

---

## Overview

| Stat | Value |
|---|---|
| **Total records** | 5,206 |
| **Unique model IDs** | ~4,818 |
| **Benchmark dimensions** | 33 |
| **Evaluation sources** | 58 (6 leaderboards + 11 papers) |
| **Cross-source collision pairs** | 154 |
| **Schema version** | EEE v0.2.1 |
| **License** | CC BY 4.0 (data) · MIT (code) |

## Sources Parsed

### Leaderboard Scrapers (6)

| # | Source | Script | Models | Benchmarks |
|---|---|---|---|---|
| 1 | HF Open LLM Leaderboard v2 | `scripts/hfopenllm_v2_scraper.py` | 4,496 | 6 |
| 2 | AlpacaEval 2.0 | `scripts/alpacaeval2_scraper.py` | 254 | 2 |
| 3 | Chatbot Arena | `scripts/chatbot_arena_scraper.py` | 39 | 1 |
| 4 | MT-Bench | `scripts/mtbench_scraper.py` | 44 | 1 |
| 5 | WildBench v2 | `scripts/wildbench_scraper.py` | 30 | 1 |
| 6 | BigCodeBench | `scripts/bigcodebench_scraper.py` | 22 | 2 |

### Academic Paper Extraction (11)

| # | Paper | arXiv ID | Models | Benchmarks |
|---|---|---|---|---|
| 1 | Falcon LLM | 2306.11644 | 10 | 4 |
| 2 | LLaMA 2 | 2307.09288 | 20 | 5 |
| 3 | Mistral 7B | 2309.10305 | 12 | 7 |
| 4 | Mixtral of Experts | 2312.11805 | 12 | 8 |
| 5 | OLMo | 2402.01322 | 13 | 6 |
| 6 | Gemma | 2403.05530 | 13 | 6 |
| 7 | InternLM2 | 2403.17297 | 13 | 5 |
| 8 | Phi-3 | 2404.14219 | 12 | 5 |
| 9 | Qwen2 | 2405.04434 | 15 | 6 |
| 10 | Llama 3.1 | 2407.21783 | 17 | 6 |
| 11 | DeepSeek-V3 | 2412.19437 | 16 | 6 |

## Repository Structure

```
shared_task_submission/
├── README.md                        # This file
├── validate_submission.py           # Self-contained validation script
├── upload_to_hf.py                  # Upload dataset to Hugging Face Hub
├── schema/
│   └── eval.schema.json             # EEE schema v0.2.1
├── data/                            # → Symlink to ../data/ (5,125 JSON records)
│   ├── alpacaeval2/
│   ├── bigcodebench/
│   ├── chatbot_arena/
│   ├── hfopenllm_v2/
│   ├── mt_bench/
│   ├── wildbench/
│   └── papers_{arxiv_id}/           # × 11 paper sources
├── scripts/
│   ├── base.py                      # Abstract base classes for scrapers
│   ├── alpacaeval2_scraper.py       # → Leaderboard scraper
│   ├── bigcodebench_scraper.py      # → Leaderboard scraper
│   ├── chatbot_arena_scraper.py     # → Leaderboard scraper
│   ├── hfopenllm_v2_scraper.py      # → Leaderboard scraper
│   ├── mtbench_scraper.py           # → Leaderboard scraper
│   ├── wildbench_scraper.py         # → Leaderboard scraper
│   ├── create_paper_records.py      # Paper table → JSON extraction
│   └── extract_paper.py            # Paper extraction utilities
├── converters/
│   ├── README.md                    # Converter documentation
│   ├── inspect/                     # Inspect AI → EEE converter
│   ├── helm/                        # HELM → EEE converter
│   └── lm_eval/                     # lm-eval-harness → EEE converter
├── analysis/
│   ├── collision_detection.py       # Cross-source collision pair detection
│   ├── coverage_audit.py            # Metadata coverage analysis
│   ├── variance_decomposition.py    # OLS variance attribution
│   ├── per_benchmark_ols.py         # Per-benchmark multivariable OLS
│   ├── rank_instability.py          # Kendall τ_b rank correlation
│   ├── power_simulation.py          # Bootstrap power analysis
│   └── output/                      # Pre-computed analysis CSVs
└── docs/
    ├── methodology_track1.txt       # Full extraction methodology (619 lines)
    └── DATASET_CARD.md              # HuggingFace dataset card
```

## Quick Start

### 1. Validate all records against the schema

```bash
python validate_submission.py
```

### 2. Run a specific scraper

```bash
# Example: scrape AlpacaEval 2.0
python scripts/alpacaeval2_scraper.py

# Example: extract paper tables
python scripts/create_paper_records.py
```

### 3. Convert evaluation logs from existing harnesses

```bash
# From lm-eval-harness
python -m converters.lm_eval --log_path /path/to/results.json

# From Inspect AI
python -m converters.inspect --log_path /path/to/eval.json

# From HELM
python -m converters.helm --log_path /path/to/helm_output/
```

### 4. Upload to Hugging Face Hub

```bash
pip install huggingface_hub
huggingface-cli login
python upload_to_hf.py --repo-id steliospsychias/every_eval_ever
# Or dry-run first:
python upload_to_hf.py --dry-run
```

## Schema Overview

Each JSON record conforms to EEE schema v0.2.1 with these required fields:

```json
{
  "schema_version": "0.2.1",
  "evaluation_id": "source/model_id/timestamp",
  "retrieved_timestamp": "unix_epoch",
  "source_metadata": {
    "source_type": "documentation | evaluation_run",
    "source_organization_name": "...",
    "evaluator_relationship": "first_party | third_party | collaborative | other"
  },
  "eval_library": { "name": "..." },
  "model_info": {
    "name": "...", "id": "...", "developer": "..."
  },
  "evaluation_results": [{
    "evaluation_name": "benchmark_name",
    "metric_config": { "lower_is_better": false, "score_type": "continuous" },
    "score_details": { "score": 0.0 }
  }]
}
```

Key methodology fields in `generation_config.additional_details`:
- **`n_shot`** — number of few-shot examples
- **`prompt_template`** — exact prompt format identifier (compared via string equality; "standard" = undocumented)
- **`temperature`** — sampling temperature

## Data Extraction Methodology

Full per-source documentation (619 lines) is in [`docs/methodology_track1.txt`](docs/methodology_track1.txt). Below is a summary of each pipeline stage and the decisions made at each step.

### Leaderboard Scrapers (6 sources)

Each scraper inherits from `BaseScraper` (`scripts/base.py`) and follows the same four-stage pipeline:

1. **Fetch** — HTTP GET from API or GitHub raw URLs, with rate limiting and retry logic. Multiple endpoints are tried in priority order; if all live sources fail, hardcoded fallback data from the original paper/leaderboard snapshot is used. Fallback records are timestamped to the original publication date so consumers can distinguish stale data from live-scraped data.

2. **Parse** — Source-specific parsing. Data formats vary widely:
   - HF Open LLM v2: single JSON endpoint from the HF Space API
   - AlpacaEval 2.0: CSV leaderboard + per-model JSON directories
   - Chatbot Arena: dated CSV snapshots + Gradio API fallback
   - MT-Bench: JSONL from the FastChat GitHub repo + hardcoded table
   - WildBench: JSON from GitHub (both list-of-dicts and dict-keyed layouts)
   - BigCodeBench: two JSON files (Complete + Instruct) from HuggingFace

3. **Convert** — Map source fields to EEE schema fields. Score normalization converts 0–100 percentage scales to 0–1 where applicable (see [Score Normalization](#score-normalization) below). Model IDs are normalized to HuggingFace `org/model` format using per-scraper lookup tables. Each scraper has its own `_normalise_model_id()` function.

4. **Validate & Write** — Every record is validated against `eval.schema.json` before writing. If `jsonschema` is not installed or the schema file is missing, records are written without validation (with a stderr warning). Records are saved as `data/{source}/{developer}/{model}/{uuid}.json`.

### Academic Paper Extraction (52 sources)

Paper extraction uses a semi-automated pipeline (`scripts/extract_paper.py` + `scripts/create_paper_records.py`):

1. **PDF download** — PDFs are fetched from arXiv and cached locally under `scripts/scrapers/raw/papers/`.

2. **Table extraction** — `pdfplumber` extracts all tables from each page. A two-tier heuristic identifies results tables:
   - Strong match: specific benchmark name found in column headers
   - Weak match: ≥2 generic metric terms + ≥25% numeric cell density
   - Both orientations (models-as-rows, models-as-columns) are tried with multi-row header collapsing.

3. **Extraction confidence tiering** — Every data point is tagged:
   - `high` — strong benchmark keyword + ≥50% numeric density
   - `medium` — strong keyword with 25–50% density, or keyword only in caption
   - `low` — accepted via keyword + numeric-count floor only
   - `llm` — extracted by the optional LLM fallback extractor

4. **Ablation filtering** — Tables with ≥40% ablation markers in the first column (`w/o`, `without`, `variant`, etc.) are rejected to avoid recording ablation variants as distinct models.

5. **Deduplication** — When the same (model, benchmark) appears in multiple tables, the first occurrence wins. Merge priority is `table > prose > LLM`.

6. **Metadata enrichment** — Each paper's methodology section is manually cross-referenced to fill n-shot, harness, and prompt template fields. Where a paper says only "standard" or provides no prompt details, `prompt_template` is recorded as `"standard"` and treated as undocumented for analysis purposes.

7. **LLM fallback** (optional, `--llm-fallback`) — When pdfplumber yields zero accepted tables, the full paper text is chunked and sent to GPT-4o-mini with structured JSON output. LLM-sourced scores are tagged with `_extraction_confidence='llm'` and should be cross-checked before use.

### Score Normalization

All scrapers apply normalization to map scores onto the scale declared by `metric_config.min_score` / `max_score`:

| Source | Raw scale | Normalized | Method |
|---|---|---|---|
| HF Open LLM v2 | 0–100 | 0–1 | ÷100 if `raw > 1.0` |
| AlpacaEval 2.0 | 0–100 | 0–1 | Unconditional ÷100 (`is_percentage=True`) |
| Papers (most benchmarks) | 0–100 | 0–1 | ÷100 if `raw > 1.0` |
| Papers (BLEU, ROUGE) | 0–100 | 0–100 | Kept on original scale |
| Papers (perplexity, WER) | unbounded | raw | No normalization; `max_score=None` |
| MT-Bench | 1–10 | 1–10 | Kept on original scale |
| Chatbot Arena | Elo ~800–1400 | raw | `max_score=3000` (conservative bound) |
| WildBench | −100–100 | raw | Kept on original scale |

**The ÷100 heuristic**: Several scrapers use `if raw_score > 1.0: score /= 100`. This is fragile — a genuine score of exactly 1.0 on a 0–100 scale (1% accuracy) would be silently left un-normalized. AlpacaEval avoids this by using an unconditional flag.

### Model ID Normalization

Models are normalized to HuggingFace `org/model` format:
- IDs already containing `/` are kept as-is
- IDs without `/` have the developer prefix prepended via lookup tables
- Each scraper maintains its own `_normalise_model_id()` and `_hf_ids` dict
- Developer inference for papers uses pattern matching sorted by key length (longest first) to prevent `"gpt"` → OpenAI from shadowing `"gpt-j"` → EleutherAI
- Ambiguous names (`alpaca`, `orca`, `hermes`, `vicuna`) produce a warning but are still recorded

---

## Source Data Issues and Inconsistencies

### 1. Metadata Coverage Gaps

| Field | Coverage | Detail |
|---|---|---|
| **Harness** | 58/58 (100%) | Universally documented |
| **N-shot** | 53/58 (91.4%) | Missing from 5 non-MCQ leaderboards (AlpacaEval, Chatbot Arena, MT-Bench, WildBench, BigCodeBench) |
| **Prompt template** | 24/58 (41.4%) | Only HF Open LLM v2 (100%) + 23 papers with non-standard templates; 29 papers record only `"standard"` (treated as undocumented) |
| **Temperature** | 1/58 (1.7%) | Only MT-Bench (0.7 per paper specification) |
| **Normalization mode** | 0/58 (0%) | Not a validated schema field; stored in `additional_details` for some papers |
| **Scoring mode** | 0/58 (0%) | Not a validated schema field |
| **Commit hash** | 0/58 (0%) | Not a validated schema field; harness version recorded as `"unknown"` for most sources |

The three highest-priority gaps (normalization mode, scoring mode, commit hash) are proposed as schema v0.3 extensions in the paper (§3).

### 2. Score Normalization Inconsistencies Across Sources

The database contains scores on **three different scales** depending on metric type:
- **0–1** (accuracy, pass@1, win rate): most MCQ benchmarks, HumanEval, AlpacaEval
- **0–100** (BLEU, ROUGE): kept on percentage scale, not divided
- **Unbounded** (perplexity, Elo, WER, WB-Score): raw values with no normalization

The `metric_config.min_score` / `max_score` fields disambiguate, but consumers must check these per-record before aggregation. Cross-benchmark score comparison without checking `metric_config` will produce nonsensical results.

### 3. PDF Table Extraction Artifacts

| Issue | Frequency | Impact |
|---|---|---|
| **`\multicolumn` corruption** | ~5–10% of paper tables | pdfplumber flattens spanning cells, shifting scores one+ columns left. Emits warning when >15% of rows have misaligned column counts. **Scores may be mapped to the wrong benchmark.** |
| **Figures/charts** | ~15–30% of result tables | Bar charts, radar plots, and scatter plots cannot be extracted via table parsing. Requires `--llm-fallback`. |
| **Prose-embedded results** | Common in short papers | Sentences like "achieves 87.3% on MMLU" are missed by the table-only pipeline. The prose extractor catches some but not all. |
| **Scanned PDFs** | Rare for arXiv | >50% non-text pages trigger a warning. Requires OCR via pytesseract. |

### 4. Cross-Source Score Disagreements (Collision Pairs)

154 collision pairs were detected where the same (model, benchmark) appears in two sources. Representative disagreements:

| Model | Benchmark | Source A | Source B | Δ | Likely cause |
|---|---|---|---|---|---|
| Mistral-7B-v0.1 | GSM8K | arXiv:2312 (0.352, 5-shot) | arXiv:2403 (0.577, 11-shot + CoT) | −0.225 | Different n-shot + chain-of-thought |
| Mistral-7B-v0.1 | HellaSwag | arXiv:2309 (0.812) | arXiv:2312 (0.833, continuation format) | −0.021 | Different prompt template (undocumented normalization likely) |
| Mistral-7B-v0.1 | MMLU | arXiv:2309 | arXiv:2403 | −0.018 | Same harness + n-shot, different prompt formatting across harness versions |
| Llama-2-7B | HumanEval | arXiv:2307 (0.122) | arXiv:2309 (0.122) | 0.000 | Identical methodology — positive control |

Near-zero deltas (5/9 benchmarks) reflect a **selection mechanism**: collision pairs with small gaps tend to come from papers sharing the same evaluation pipeline, not from genuinely independent re-evaluations.

### 5. Model Identity Ambiguities

- **Exact string matching** on HuggingFace model IDs is the only collision detection method. Model aliases (e.g., `meta-llama/Llama-2-7b-hf` vs. `meta-llama/Llama-2-7b`) and checkpoint variants are missed.
- **Mistral-7B-v0.1 over-representation**: appears in 37 of 154 collision pairs. A robustness check excluding Mistral-7B-v0.1 pairs confirms harness effects persist (MMLU $R^2_p$ = 0.189, HellaSwag $R^2_p$ = 0.224).
- **Developer attribution**: Community fine-tunes sharing prefixes with base models (`alpaca`, `orca`, `hermes`, `vicuna`) may be attributed to the wrong developer.
- **Proprietary models**: API-only models (GPT-4, Claude, Gemini) are assigned constructed IDs like `openai/gpt-4-0314` since no HuggingFace repo exists.

### 6. Leaderboard-Specific Issues

| Source | Issue |
|---|---|
| **HF Open LLM v2** | `lighteval` version not exposed by API — recorded as `"unknown"`. The ÷100 normalization edge case (score = 1.0 meaning 1%) is theoretically possible but unlikely in practice. |
| **AlpacaEval 2.0** | Standard error (`lc_win_rate_se`) is available in the CSV but not yet propagated to `score_details.uncertainty`. Generation config not exposed by leaderboard. |
| **Chatbot Arena** | Elo ratings are relative to the current model pool — not comparable across snapshots taken at different dates. Per-category breakdowns not captured. |
| **MT-Bench** | Only average turn score stored; per-category and per-turn breakdowns not captured. Live JSONL may lag behind newly added models. |
| **WildBench** | Generation config varies by model provider and is not exposed. Only WB-Score (aggregate) captured. |
| **BigCodeBench** | `calibrated-pass@1` not included. Some entries lack inference platform specification. |

### 7. Schema-Level Gaps

The EEE schema (v0.2.1) does not validate several factors known to affect scores:
- **Normalization mode** (none / byte_length / token_length / unconditional) — a multi-point driver of HELM vs. lm-eval-harness gaps
- **Scoring mode** (greedy_generation / log_likelihood / generation_execution)
- **Commit hash** — harness defaults change across releases
- **pass@k** — code benchmarks report different $k$ values
- **LLM-as-judge fields** — judge prompt, committee size, aggregation rule (relevant for MT-Bench, AlpacaEval, WildBench, Arena-Hard)

All can be stored in free-form `additional_details` but lack validation or cross-source interoperability. The shared task organizers' own illustrative schema examples contain values (`source_type: "evaluation_platform"`, non-string `additional_details` entries) that do not validate against the published schema — underscoring how easily undocumented divergence arises.

### 8. Deduplication and Merge Decisions

- **First-occurrence-wins** for paper extraction: when the same benchmark appears in multiple tables for one model, the first table's value is kept. Later tables (which may contain the corrected value) are discarded.
- **Confidence floor**: the minimum (most pessimistic) tier across all items determines the model record's confidence. A single `low`-confidence score downgrades the entire record.
- **Ablation filtering threshold (40%)**: a legitimate results table where ≥40% of row labels contain ablation markers would be incorrectly rejected; an ablation table with <40% markers would pass through undetected.

### 9. Reproducibility Caveats

- **Live source fragility**: all leaderboard scrapers depend on public APIs and GitHub URLs that can change without notice. The multi-source fallback ensures output is always produced, but fallback data is frozen at publication-date snapshots.
- **Non-deterministic ordering**: scrapers write one JSON file per model with a UUID filename. File ordering is not deterministic across runs, though content is identical.
- **Instance-level data**: none of the leaderboard scrapers produce per-sample predictions (`.jsonl` companions) because the leaderboards do not expose per-sample results in machine-readable format.

---

## Key Findings (from analysis)

- **154 collision pairs** across 9 benchmarks and 37 source pairs
- **Harness identity** is the most consistent predictor of cross-source score variance (MMLU: partial R² = 0.169, p = 0.012; HellaSwag: partial R² = 0.181, p = 0.034)
- **No effect survives BH-FDR correction** (q_min = 0.207) — confirming the need for larger collision sets
- **Metadata coverage** is the primary bottleneck: ~50% prompt template coverage needed for 80% power on MMLU and GSM8K
- **Selection mechanism**: near-zero deltas on 5/9 benchmarks reflect shared pipelines, not genuine reproducibility

## Dependencies

```
jsonschema>=4.23.0
requests>=2.32.0
beautifulsoup4>=4.12.0
pandas>=2.0.0
numpy>=1.26.0
tqdm>=4.66.0
scipy>=1.13.0
huggingface_hub>=0.20.0  # for upload only
```

## Citation

```bibtex
@inproceedings{psychias2026eee,
  title     = {When Same Benchmark $\neq$ Same Evaluation: Metadata Coverage
               as the Bottleneck for Cross-Source {LLM} Comparisons},
  author    = {Psychias, Stelios},
  booktitle = {Proceedings of the EvalEval Workshop at ACL 2026},
  year      = {2026},
}
```

## License

- **Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [MIT](https://opensource.org/licenses/MIT)
