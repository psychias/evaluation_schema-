# eee-docling-migration

Autonomous prompt for migrating the EEE paper extraction pipeline from pdfplumber to Docling, establishing correct data from source PDFs, and regenerating all outputs.

## Context

You are working on the codebase for an ACL 2026 paper: "You Can't Compare What You Don't Document: Metadata Gaps in LLM Benchmark Reporting." The project collects LLM evaluation results from 58 sources (6 leaderboards + 52 papers) into a standardised JSON schema (EEE v0.2.1), then runs collision detection and variance decomposition analysis.

The core problem: `scripts/extract_paper.py` uses **pdfplumber** for PDF table and text extraction. pdfplumber frequently produces incorrect results on academic PDFs — it mishandles multicolumn LaTeX layouts, merged cells, spanning headers, and sometimes shifts entire columns.

**CRITICAL ASSUMPTION: All values previously extracted from papers are wrong until proven otherwise.** The bad pdfplumber output has propagated everywhere:
- The generated JSON files in `data/papers_*/`
- `scripts/create_paper_records.py` (the hardcoded "ground truth" — was likely populated from pdfplumber output)
- The paper's own tables, case studies, and statistical analysis (Table 5, Figure 2, all deltas and R² values)
- `docs/DATASET_CARD.md` statistics

**Nothing in the existing repo can be used as a reference for correct values. The only source of truth is the actual arXiv PDFs themselves.**

Your job:
1. Replace pdfplumber with **Docling** (`docling` package by IBM)
2. Extract all paper data fresh from the source PDFs
3. Verify extraction correctness by reading the PDFs
4. Update everything downstream — `create_paper_records.py`, JSON files, analysis, figures
5. Expand paper coverage

## Setup

1. **Read all in-scope files first.** Read these for full context before writing any code:
   - `scripts/extract_paper.py` — the file you will primarily modify. ~1,870 lines. Contains `TableExtractor`, `ProseExtractor`, `LLMFallbackExtractor`, `ResultsTableParser`, `PaperConverter`, `PaperWriter`, and the `PaperExtractionPipeline` orchestrator.
   - `scripts/create_paper_records.py` — hardcoded records for paper sources. **These values are suspect.** You will replace them with PDF-verified values.
   - `scripts/base.py` — abstract base classes and schema validation helpers.
   - `schema/eval.schema.json` — the EEE JSON schema (v0.2.1). All output must validate against this.
   - `docs/methodology_track1.txt` — methodology documentation.
   - `docs/DATASET_CARD.md` — dataset card.
   - The 6 leaderboard scrapers in `scripts/` — read but do NOT modify (they don't use pdfplumber).
   - `converters/` — harness-specific converters. Read but do NOT modify.

2. **Install Docling:**
   ```bash
   pip install docling
   ```
   Docling depends on PyTorch and several vision models. GPU accelerates table structure recognition; CPU fallback works but is slower.

3. **Keep pdfplumber installed** for comparison during verification:
   ```bash
   pip install pdfplumber
   ```

4. **Create a git branch:**
   ```bash
   git checkout -b migration/docling-extraction
   ```

## Phase 1: Understand Docling's API

Before modifying the main codebase, write a small test script to understand Docling's table extraction capabilities.

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfium2DocumentBackend

pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = True
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: {
            "pipeline_options": pipeline_options,
            "backend": PyPdfium2DocumentBackend,
        }
    }
)

result = converter.convert("path/to/paper.pdf")

# Docling gives you structured tables:
for table in result.document.tables:
    df = table.export_to_dataframe()
    print(df.to_string())
```

Key Docling concepts:
- `DocumentConverter` is the main entry point
- `result.document.tables` gives structured `TableData` objects
- `.export_to_dataframe()` gives pandas output; `.export_to_dict()` gives raw cell data
- `result.document.export_to_markdown()` gives the full document as markdown
- Table cells carry row/column span information — the critical advantage over pdfplumber
- Docling's deep learning table structure model handles multirow headers, merged cells, and spanning columns

Test on 2–3 papers of varying complexity. Pick one with simple tables and one with complex multi-part tables. Print the extracted tables and visually verify them against the PDFs. Understand what Docling gets right and where it might struggle.

## Phase 2: Migrate extract_paper.py

You will replace the pdfplumber dependency with Docling in these specific classes:

### 2a. Replace `TableExtractor`

Current implementation uses `pdfplumber.open()` → `page.find_tables()` → `table_obj.extract()`.

New implementation:
- Use `DocumentConverter` to parse the full PDF once
- Extract all tables from `result.document.tables`
- Convert each Docling `TableData` to the same `list[list[str]]` format the rest of the pipeline expects
- Preserve the return type: `list[tuple[int, list[list[str]], str]]` — downstream code depends on this
- For `context_text`: use Docling's document structure to find caption/paragraph preceding each table
- Handle Docling's row/column spans by flattening into the rectangular grid format `ResultsTableParser` expects (repeat content across spanned cells)

### 2b. Replace `ProseExtractor`

Replace `pdfplumber.open()` page iteration with Docling's full-text export. The regex patterns (`_PATTERN_MODEL_FIRST`, `_PATTERN_BENCH_FIRST`) stay identical — only the text source changes.

### 2c. Replace `_detect_eval_library`

Replace pdfplumber text extraction with Docling text.

### 2d. Replace `LLMFallbackExtractor`'s PDF reading

Replace pdfplumber page reading with Docling text export.

### 2e. Cache the Docling parse

`DocumentConverter.convert()` is expensive. Parse each PDF once. Add a `DoclingParser` class that wraps the converter, call it once in `PaperExtractionPipeline.run()`, and pass the result to all extractors.

### 2f. Remove pdfplumber import

Replace with Docling imports. Remove the pdfplumber ImportError handling.

### 2g. Preserve all other logic

Do NOT change: `ResultsTableParser`, `PaperConverter`, `PaperWriter`, `CoverageStats`, `_BENCHMARK_KEYWORDS`, `_DEVELOPER_MAP`, score normalisation, the CLI in `main()`, or any helpers that don't touch pdfplumber.

## Phase 3: Extract Fresh from All Papers and Verify

### 3a. Download all 52 paper PDFs

Create `scripts/arxiv_ids_full.txt` with all 52 arXiv IDs from the paper's Table 4 (Appendix C):
```
2203.15556
2204.02311
2205.01068
2210.11416
2211.05100
2302.13971
2303.08774
2304.01373
2304.10457
2305.10403
2306.05685
2306.11644
2307.09288
2308.12950
2309.05463
2309.10305
2309.16609
2310.16944
2311.11045
2312.00752
2312.06550
2312.11805
2312.15166
2401.02385
2402.01322
2402.16819
2402.17834
2402.19173
2403.04652
2403.05530
2403.07691
2403.17297
2403.19887
2404.05892
2404.10774
2404.14219
2404.14619
2405.04324
2405.04434
2405.19327
2406.12793
2407.12511
2407.21783
2408.00118
2411.14599
2411.15138
2412.19437
2501.12948
2501.15451
2502.02737
```

### 3b. Run the migrated pipeline on every paper

```bash
python scripts/extract_paper.py --batch scripts/arxiv_ids_full.txt > extraction_full.log 2>&1
```

### 3c. Verify a sample against the actual PDFs

For at least **10 papers** spanning different table layouts and complexities, manually verify the extraction:

1. Downlaod the actual papers, and manually extract the information from them
2. Print each extracted table as a DataFrame
3. Read the actual values from the table
4. Compare against what the pipeline put in the JSON files
5. Pay close attention to:
   - Whether column headers and data columns are aligned correctly
   - Whether scores are in the right (model, benchmark) cell — column shifts are the most common error
   - Whether percentage vs. fraction normalisation is correct (e.g., "60.1" for MMLU → 0.601 in JSON)
   - Whether n-shot is captured correctly from column headers like "GSM8K (5-shot)" or "GSM8K (maj@8)"
   - Whether CoT (chain-of-thought) variants are distinguished from standard prompting
   - Whether model names are read correctly (merged cells, bold formatting, abbreviations)

Record every discrepancy. Fix the pipeline if systematic issues appear.

.

## Phase 4: Update create_paper_records.py

`scripts/create_paper_records.py` contains hardcoded values for ~10 paper sources. For every value in that file:

1. Find the corresponding (model, benchmark) pair in your fresh Docling extraction
2. If the value differs: replace it and add a comment `# corrected: was X.XXX`
3. Also verify and correct n_shot values, model names, and benchmark names

## Phase 5: Regenerate Leaderboard Data and Validate

The leaderboard scrapers don't use pdfplumber but should be re-run for consistency:

```bash
python scripts/hfopenllm_v2_scraper.py
python scripts/mtbench_scraper.py
python scripts/bigcodebench_scraper.py
python scripts/wildbench_scraper.py
python scripts/alpacaeval2_scraper.py
python scripts/chatbot_arena_scraper.py
```

Run schema validation on everything:
```bash
python validate_submission.py
```
Zero failures required.

## Phase 6: Regenerate Analysis and Figures

Re-run the full analysis pipeline with the corrected data:

1. Collision detection — find all (model, benchmark) pairs appearing in 2+ sources
2. Variance decomposition — OLS regressions per benchmark
3. Rank correlation — Kendall τ_b between source pairs
4. All figures (3–11 from the paper)

Write `data_corrections_report.md` documenting:
- How many (model, benchmark, score) values changed from the old extraction
- Summary statistics: mean |delta|, max |delta|, how many changed by >0.01
- New collision pair count and delta distributions
- New partial R² values
- Whether the paper's qualitative conclusions still hold
- Any collision pairs that appeared or disappeared due to corrected model/benchmark names

## Phase 7: Add More Papers

Expand coverage beyond the 52. Look for LLM papers from 2024–2026 that report benchmark tables and evaluate models already in the dataset. For each:
1. Run extraction
2. Verify against the PDF — do not trust extraction blindly
3. Update documentation
4. Re-run analysis

## Constraints

**CAN modify:**
- `scripts/extract_paper.py` — primary migration target
- `scripts/create_paper_records.py` — correct with verified values
- `docs/methodology_track1.txt`, `docs/DATASET_CARD.md` — update
- Create new scripts

**CANNOT modify:**
- `schema/eval.schema.json` — frozen at v0.2.1
- Leaderboard scrapers — no pdfplumber dependency
- `converters/` — different track
- Return types of `TableExtractor.extract()` or `ResultsTableParser.parse()`

## Error Handling

- **Docling crashes on a PDF:** Fall back to pdfplumber for that paper only. Log it.
- **Table extraction looks wrong:** Always compare against the PDF. Fix or flag for manual review.
- **Ambiguous table structure:** Record all variants with clear labels. Don't conflate GSM8K 5-shot with GSM8K CoT.

## NEVER STOP

Once Phase 2 begins, work autonomously through all phases. If you hit a blocker, try at least 3 approaches before moving on. The human may be away. Continue until complete or manually stopped.
do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.