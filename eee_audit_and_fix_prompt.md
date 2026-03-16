# eee-audit-and-fix

Autonomous prompt to audit all existing EEE extraction data against 12 quality rules, fix what's broken, fill what's missing, and re-extract only the papers that need it.

## Context

The data has already been extracted using LaTeX source (primary) and Docling (fallback). The extraction pipeline produces correct scores — verified on Gemma 3 (100% accurate) and Chinchilla (all numbers correct). But the audit found systematic issues with metadata, filtering, and completeness.

**Do NOT re-extract papers where the scores are correct.** Instead:
1. Audit every existing JSON record against the 12 rules below
2. Fix metadata in-place (developer, evaluator_relationship, n_shot, model names)
3. Delete records that violate Rule 3 (SOTA/human/random baselines)
4. For papers where tables or models are MISSING, re-extract only those gaps
5. Regenerate analysis with the corrected data

## Setup

1. **Read all in-scope files:**
   - All JSON files in `data/papers_*/` — this is what you're auditing
   - `schema/eval.schema.json` — the schema
   - `scripts/extract_latex.py` or `scripts/extract_paper.py` — the extraction code (for re-running on specific papers)
   - The 6 leaderboard scrapers
   - `docs/`

2. **Install dependencies:**
   ```bash
   pip install jsonschema requests pandas tabulate
   ```

---

## THE 12 EXTRACTION QUALITY RULES

### Rule 1: ALL benchmark tables extracted
A paper may have 5+ tables with scores. If the JSON only has benchmarks from one table, the paper needs re-extraction for the missing tables.

### Rule 2: ALL models from each table
A table with 4 model columns must produce 4 records. If a paper's JSON has fewer models than the table has columns, the paper needs partial re-extraction.

### Rule 3: No non-model records
Delete any record where `model_info.name` matches (case-insensitive): "SOTA", "supervised", "human", "random", "forecast", "oracle", "upper bound", "lower bound", "majority", "baseline". These are reference rows, not LLMs.

### Rule 4: No hallucinated model names
- "gpt-3.5-turbo" in any pre-2023 paper → rename to "GPT-3"
- Check that model names match what the paper actually says

### Rule 5: Developer must be populated
For well-known models, developer should NOT be "unknown":
```
Chinchilla, Gopher → "deepmind"
GPT-3, GPT-4 → "openai"
LLaMA, Llama-2, Llama-3 → "meta-llama"
Mistral, Mixtral → "mistralai"
Gemma → "google"
Falcon → "tiiuae"
MT-NLG → "nvidia"
Qwen → "Qwen"
DeepSeek → "deepseek-ai"
Phi → "microsoft"
OLMo → "allenai"
BLOOM → "bigscience"
Pythia → "EleutherAI"
MPT → "mosaicml"
Yi → "01-ai"
Command-R → "CohereForAI"
Jamba → "ai21labs"
Nemotron → "nvidia"
SOLAR → "upstage"
InternLM → "internlm"
Baichuan → "baichuan-inc"
StarCoder → "bigcode"
PaLM → "google"
```

### Rule 6: evaluator_relationship must be correct
- Paper's OWN model → `"first_party"`
- Competitor model evaluated BY the paper → `"third_party"`
- To determine: check if `model_info.developer` matches `source_metadata.source_organization_name`. If same org → first_party. If different → third_party.

### Rule 7: n_shot must be populated when available
If n_shot is "unknown", check the paper's LaTeX source for:
- Table caption: "Zero-shot", "5-shot", "Few-Shot"
- Column header annotations: "(5-shot)", "(0-shot)"
- Evaluation appendix tables listing n-shot per benchmark
Only leave as "unknown" if genuinely unstated.

### Rule 8: Multi-shot variants as separate entries
If a table has 0-shot, 5-shot, 64-shot rows for the same benchmark, each should be a separate entry in `evaluation_results[]` with different n_shot values.

### Rule 9: All benchmark types captured
Check that the JSON includes all standard benchmarks mentioned in the paper, not just one table's worth.

### Rule 10: Benchmark variants distinguished
"TriviaQA (unfiltered)" ≠ "TriviaQA (filtered)". "GSM8K (CoT)" ≠ "GSM8K".

### Rule 11: Dashes are missing values, not zero
No record should have score = 0.0 where the paper shows "–" or "—".

### Rule 12: eval_library populated
If the paper states the harness, it should be in `eval_library.name`.

---

## Phase 1: Automated Audit

Write `scripts/audit_records.py` that scans every JSON file in `data/` and reports violations.

### 1a. Rule 3 violations (delete these)

```python
import json, re
from pathlib import Path

NON_MODEL_PATTERNS = [
    r'\bSOTA\b', r'\bsota\b', r'\bsupervised\b', r'\bhuman\b', r'\bHuman\b',
    r'\brandom\b', r'\bRandom\b', r'\bforecast\b', r'\bForecast\b',
    r'\boracle\b', r'\bOracle\b', r'\bupper.?bound\b', r'\blower.?bound\b',
    r'\bmajority\b', r'\bbaseline\b', r'\bBaseline\b',
]

for json_path in Path("data").rglob("*.json"):
    rec = json.loads(json_path.read_text())
    name = rec.get("model_info", {}).get("name", "")
    if any(re.search(p, name) for p in NON_MODEL_PATTERNS):
        print(f"RULE 3 VIOLATION — delete: {json_path}  model={name}")
```

Action: Delete every file flagged. Log what was deleted.

### 1b. Rule 4 violations (fix model names)

```python
# Check for gpt-3.5-turbo in papers before 2023
for json_path in Path("data/papers_*").rglob("*.json"):
    rec = json.loads(json_path.read_text())
    arxiv_id = json_path.parts[1].replace("papers_", "")  # e.g. "2203.15556"
    year = int(arxiv_id[:2])
    name = rec["model_info"]["name"]
    if "gpt-3.5" in name.lower() or "gpt-4" in name.lower():
        if year <= 22:  # 2022 or earlier
            print(f"RULE 4 VIOLATION: {json_path}  '{name}' impossible in {arxiv_id}")
```

Action: Fix the model name, model ID, and developer in-place.

### 1c. Rule 5 violations (fix developer)

```python
DEVELOPER_MAP = {
    "chinchilla": "deepmind", "gopher": "deepmind",
    "gpt-3": "openai", "gpt-4": "openai", "gpt3": "openai",
    "llama": "meta-llama", "codellama": "meta-llama",
    "mistral": "mistralai", "mixtral": "mistralai",
    "gemma": "google", "palm": "google", "gemini": "google",
    "falcon": "tiiuae", "mt-nlg": "nvidia", "megatron": "nvidia",
    "qwen": "Qwen", "deepseek": "deepseek-ai",
    "phi": "microsoft", "orca": "microsoft",
    "olmo": "allenai", "bloom": "bigscience",
    "pythia": "EleutherAI", "mpt": "mosaicml", "dbrx": "databricks",
    "yi-": "01-ai", "command": "CohereForAI",
    "jamba": "ai21labs", "nemotron": "nvidia",
    "solar": "upstage", "internlm": "internlm",
    "baichuan": "baichuan-inc", "starcoder": "bigcode",
}

for json_path in Path("data/papers_*").rglob("*.json"):
    rec = json.loads(json_path.read_text())
    dev = rec["model_info"]["developer"]
    name = rec["model_info"]["name"].lower()
    if dev == "unknown":
        for pattern, correct_dev in DEVELOPER_MAP.items():
            if pattern in name:
                print(f"RULE 5 FIX: {json_path}  '{name}' → developer={correct_dev}")
                break
```

Action: Update developer in-place for all matches.

### 1d. Rule 6 violations (fix evaluator_relationship)

For each paper source, determine the publishing org. Then for each record:
- If model developer == paper org → `"first_party"`
- If model developer != paper org → `"third_party"`

Action: Fix in-place.

### 1e. Rule 7 violations (flag missing n_shot)

Count records with n_shot = "unknown". For papers with many such records, flag for manual LaTeX re-check to fill in n_shot from captions/appendix.

### 1f. Rule 11 violations (score = 0.0 suspiciously)

Flag any record where a standard benchmark (MMLU, GSM8K, HumanEval, etc.) has score = 0.0. Cross-check: is this a real zero or a dash that was misinterpreted?

### 1g. Rule 12 violations (eval_library missing)

Count records with `eval_library.name` = "unknown". Flag papers where the harness is stated in the text but not captured.

### Audit output

Save `results/audit_report.json`:
```json
{
  "total_records": 5205,
  "rule_3_violations": [{"path": "...", "model": "SOTA (open book)"}],
  "rule_4_violations": [{"path": "...", "old_name": "gpt-3.5-turbo", "fix": "GPT-3"}],
  "rule_5_fixes": [{"path": "...", "model": "Chinchilla", "old": "unknown", "new": "deepmind"}],
  "rule_6_fixes": [{"path": "...", "model": "GPT-3", "old": "first_party", "new": "third_party"}],
  "rule_7_unknown_nshot_count": 142,
  "rule_11_suspicious_zeros": [...],
  "rule_12_unknown_harness_count": 89
}
```

## Phase 2: Apply Automated Fixes

For rules 3, 4, 5, 6: apply fixes directly to the JSON files.

```python
# For each violation:
# - Rule 3: delete the file
# - Rule 4: edit model_info.name, model_info.id, model_info.developer
# - Rule 5: edit model_info.developer
# - Rule 6: edit source_metadata.evaluator_relationship
# Save the modified JSON back to the same path
```

After applying: re-run schema validation. Zero failures.

## Phase 3: Re-extract Papers with Missing Data

From the audit, identify papers where entire tables or models are missing. These need targeted re-extraction:

### 3a. Identify gaps

For each paper, compare what's in the JSON against what the paper should contain:
- Download the LaTeX source (or use cached copy)
- Count benchmark tables in the LaTeX
- Count models per table
- Compare against existing JSON records
- Flag papers where records are incomplete

### 3b. Re-extract only the gaps

For flagged papers:
1. Download LaTeX source (if not cached)
2. Parse the missing tables
3. Generate new JSON records for missing models/benchmarks
4. Apply all 12 rules during extraction
5. Validate against schema

Do NOT overwrite existing correct records. Only ADD missing ones.

### 3c. Known missing data to check for

The Chinchilla paper (2203.15556) is known to be incomplete:
- Missing: MMLU scores (Table 6), LAMBADA/RACE scores (Table 7), SIQA (Table 8), multi-shot QA variants (Table 9)
- These need re-extraction from LaTeX source

Check all other papers for similar gaps.

## Phase 4: Fix n_shot from LaTeX Source

For papers with many n_shot = "unknown" records:
1. Download LaTeX source
2. Find evaluation appendix/details table (many papers have one)
3. Find table captions mentioning n-shot
4. Update the n_shot field in existing records

This is a metadata-only fix — no scores change.

## Phase 5: Fix Leaderboard Scrapers

Add `eval_library` to all 6 scrapers:
- `hfopenllm_v2_scraper.py` → `"lighteval"`
- `mtbench_scraper.py` → `"fastchat"`
- `bigcodebench_scraper.py` → `"bigcodebench"`
- `wildbench_scraper.py` → `"wildbench"`
- `alpacaeval2_scraper.py` → `"alpacaeval"`
- `chatbot_arena_scraper.py` → `"chatbot_arena"`

Re-run scrapers. Validate.

Fix URL: `papers_2405.04434` → arXiv:2407.10671 (Qwen2).

## Phase 6: Validate Everything

1. Schema validation: zero failures
2. Rule 3 re-check: zero non-model records remain
3. Rule 5 re-check: count remaining "unknown" developers (should be near zero for known models)
4. Rule 6 re-check: no evaluator_relationship mismatches
5. Spot-check the 7 critical issues from the verification report
6. Spot-check Chinchilla paper completeness

## Phase 7: Regenerate Analysis and Figures

With corrected and completed data:
1. Collision detection
2. Score deltas
3. Variance decomposition (OLS)
4. Rank correlation (Kendall τ_b)
5. All figures (3–11)
6. Coverage statistics

Write `results/audit_fixes_report.md`:
- Records deleted (Rule 3)
- Records renamed (Rule 4)
- Developer fields fixed (Rule 5)
- evaluator_relationship fixed (Rule 6)
- n_shot filled in (Rule 7)
- Missing data re-extracted (which papers, which tables)
- New collision pair count
- Impact on analysis (did any R² values or delta distributions change?)

## Constraints

**CAN modify:**
- Existing JSON files in `data/` — metadata fixes in-place
- `scripts/` — leaderboard scrapers, extraction scripts
- `docs/` — update

**CANNOT modify:**
- `schema/eval.schema.json`
- `converters/`
- Correct scores in existing records (only fix metadata, delete bad records, add missing records)

## NEVER STOP

Work through all phases autonomously. Phase 1 and 2 are fast (minutes). Phase 3 may take longer for papers needing re-extraction. Do not pause to ask questions.

## Commit Strategy

- `audit: scan all records against 12 quality rules`
- `fix: delete non-model records (Rule 3)`
- `fix: correct model names, developers, evaluator_relationship (Rules 4-6)`
- `data: re-extract missing tables from incomplete papers`
- `fix: populate n_shot from LaTeX captions (Rule 7)`
- `fix: add eval_library to leaderboard scrapers`
- `analysis: regenerate figures with corrected data`
