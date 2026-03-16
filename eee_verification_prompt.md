# eee-verification-auditor

Autonomous prompt for a verification model to independently audit the EEE dataset by reading the original LaTeX source from arXiv.

## Context

A separate model has:
1. Migrated the paper extraction pipeline from pdfplumber to Docling
2. Re-extracted all 52 papers from source PDFs
3. Updated `create_paper_records.py` with corrected values
4. Regenerated all JSON records and analysis

Your job is to **independently verify** that the generated data is correct.

**Trust nothing except the original LaTeX source files from arXiv.** The authors typed exact numbers into LaTeX table environments — those numbers are the ground truth. No PDF parsing, no image OCR, no intermediary. You read the `.tex` file directly.

Do not trust:
- The generated JSON files (they're what you're auditing)
- `create_paper_records.py` (updated by the other model — may still be wrong)
- Any `verified_ground_truth.json` or `data_corrections_report.md` the other model produced
- Any programmatic PDF extraction output

## Setup

1. **Read for context:**
   - `schema/eval.schema.json` — the EEE JSON schema (v0.2.1)
   - `docs/DATASET_CARD.md` — dataset overview

2. **Install dependencies:**
   ```bash
   pip install jsonschema requests pandas
   ```

## Your Verification Method

arXiv stores the LaTeX source for nearly every paper. Download the source tarball, extract the `.tex` files, and parse the table environments directly.

### How to get LaTeX source from arXiv

```python
import requests
import tarfile
import io
import re
import os
from pathlib import Path

def download_arxiv_source(arxiv_id, dest_dir='/tmp/arxiv_src'):
    """Download and extract LaTeX source from arXiv."""
    out_dir = Path(dest_dir) / arxiv_id
    if out_dir.exists():
        return out_dir

    url = f'https://arxiv.org/src/{arxiv_id}'
    resp = requests.get(url, headers={'User-Agent': 'EEE-verification/1.0'})
    resp.raise_for_status()

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tar = tarfile.open(fileobj=io.BytesIO(resp.content))
        tar.extractall(path=out_dir)
        tar.close()
    except tarfile.ReadError:
        # Some papers are a single .tex file, not a tarball
        (out_dir / 'paper.tex').write_bytes(resp.content)

    return out_dir

def find_tex_files(src_dir):
    """Find all .tex files in the source directory."""
    return list(Path(src_dir).rglob('*.tex'))

def extract_tables_from_tex(tex_path):
    """Extract raw tabular environments from a .tex file."""
    content = tex_path.read_text(errors='replace')
    # Find all tabular/table environments
    # Handles: tabular, tabular*, table, table*
    pattern = r'\\begin\{(?:tabular\*?|table\*?)\}.*?\\end\{(?:tabular\*?|table\*?)\}'
    tables = re.findall(pattern, content, re.DOTALL)
    return tables, content
```

### How to parse a LaTeX table

LaTeX tables have unambiguous structure:
- `&` separates columns
- `\\` ends a row
- `\hline`, `\midrule`, `\toprule`, `\bottomrule` are visual separators (ignore)
- `\multicolumn{N}{align}{text}` spans N columns
- `\textbf{X}`, `\underline{X}`, `\emph{X}` are formatting (strip to get the value)
- `\rowcolor`, `\cellcolor` are visual (ignore)

```python
def parse_latex_table(table_str):
    """Parse a LaTeX tabular into a list of rows, each a list of cell strings."""
    # Remove comments
    lines = table_str.split('\n')
    lines = [re.sub(r'(?<!\\)%.*$', '', line) for line in lines]
    content = '\n'.join(lines)

    # Remove \begin{tabular}..., \end{tabular}, column spec
    content = re.sub(r'\\begin\{tabular\*?\}(\{[^}]*\})?(\{[^}]*\})?', '', content)
    content = re.sub(r'\\end\{tabular\*?\}', '', content)

    # Remove horizontal rules
    content = re.sub(r'\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]*\})', '', content)

    # Strip formatting commands but keep content
    content = re.sub(r'\\(?:textbf|textit|emph|underline|textsc|mathrm|text)\{([^}]*)\}', r'\1', content)
    content = re.sub(r'\\(?:bf|it|em|sc)\b\s*', '', content)
    content = re.sub(r'\{\\(?:bf|it|em|sc)\s+([^}]*)\}', r'\1', content)

    # Handle \multicolumn{n}{align}{text}
    def expand_multicolumn(match):
        n = int(match.group(1))
        text = match.group(3)
        # Return text once; downstream alignment handles span
        return text
    content = re.sub(r'\\multicolumn\{(\d+)\}\{([^}]*)\}\{([^}]*)\}', expand_multicolumn, content)

    # Split into rows by \\
    raw_rows = re.split(r'\\\\', content)

    rows = []
    for row in raw_rows:
        row = row.strip()
        if not row:
            continue
        # Split by & to get cells
        cells = [cell.strip() for cell in row.split('&')]
        # Skip rows that are only whitespace/separators
        if all(re.match(r'^[\s\-=_.]*$', c) for c in cells):
            continue
        rows.append(cells)

    return rows
```

### Reading the actual values

For each table:
1. The first non-separator row is typically the header (benchmark names)
2. Subsequent rows are model names + scores
3. Read the cell at (row, column) to get the score for (model, benchmark)

The values you read from the LaTeX source are **exactly what the authors typed**. There is no parsing ambiguity for the cell content itself — only for the table structure, which LaTeX makes explicit.

## Audit 1: Schema Compliance

Validate every JSON file against the schema:

```python
import json, pathlib
from jsonschema.validators import validator_for

schema = json.loads(pathlib.Path("schema/eval.schema.json").read_text())
validator = validator_for(schema)(schema)

passed = failed = 0
failures = []
for json_path in pathlib.Path("data").rglob("*.json"):
    try:
        validator.validate(json.loads(json_path.read_text()))
        passed += 1
    except Exception as exc:
        failed += 1
        failures.append((str(json_path), str(exc)[:200]))

print(f"Schema validation: {passed} passed, {failed} failed")
```

**Requirement:** Zero failures.

## Audit 2: Read LaTeX Source and Verify Scores

This is the core audit.

### 2a. Select papers to verify

Pick at least **10 papers** from the 52. Prioritise:
- Papers that evaluate competitor models (these create collision pairs)
- Papers with complex table layouts
- A mix of eras

Suggested starting set:
- 2307.09288 (Llama 2) — large comparison table, many baselines
- 2309.10305 (Mistral 7B) — evaluates several competitor models
- 2312.11805 (Mixtral) — multi-part results tables
- 2403.05530 (Gemma) — different n-shot configs across benchmarks
- 2403.17297 (InternLM2) — wide table with many benchmarks
- 2404.14219 (Phi-3) — comparison against many baselines
- 2405.04434 (Qwen2) — large table
- 2407.21783 (Llama 3.1) — comprehensive comparison
- 2402.01322 (OLMo) — medium complexity
- 2412.19437 (DeepSeek-V3) — recent paper

### 2b. For each paper

1. **Download the LaTeX source:**
   ```python
   src_dir = download_arxiv_source(arxiv_id)
   tex_files = find_tex_files(src_dir)
   ```

2. **Find the results tables.** Search the `.tex` files for tabular environments containing benchmark keywords (MMLU, GSM8K, HumanEval, HellaSwag, ARC, WinoGrande). Parse them.

3. **Read at least 5 specific (model, benchmark) cell values.** Prioritise:
   - Models appearing in multiple papers (Mistral-7B-v0.1, Llama-2-7B, Llama-2-13B, Llama-2-70B, Mixtral-8x7B)
   - Common benchmarks (MMLU, GSM8K, HumanEval, HellaSwag)

4. **For each value, record:**
   ```
   Paper: {arxiv_id}
   .tex file: {filename}
   Table header row (raw LaTeX): {the & separated header}
   Model row (raw LaTeX): {the & separated row}
   Model name: {from first cell}
   Benchmark: {from header, including n-shot annotation}
   Raw cell value: {the exact string the authors typed}
   Scale: percentage / fraction
   Normalised to [0,1]: {value / 100 if percentage}
   ```

5. **Find the corresponding JSON record** in `data/papers_{arxiv_id}/` and compare:
   - Does a record exist for this model?
   - Does `score_details.score` match (within ±0.002 for rounding)?

6. **Verdict:**
   ```
   PASS — LaTeX source: X.XXX, JSON: X.XXX
   FAIL — LaTeX source: X.XXX, JSON: Y.YYY (delta = Z.ZZZ)
   MISSING — value exists in LaTeX but no JSON record
   ```

### 2c. Target: minimum 50 individual value checks

10 papers × 5 values each = 50 minimum. Focus extra attention on models that create collision pairs across papers.

### 2d. Verify create_paper_records.py

For each function in that file (~10 functions), pick 2–3 hardcoded values and verify against the LaTeX source. ~25 additional checks.

### 2e. Edge cases

Some papers may not have LaTeX source on arXiv (rare). If a source tarball returns 404 or is not a valid tarball:
- Note it in your report
- Fall back to rendering the PDF page as an image and reading the table visually using pdf2image + poppler (`brew install poppler`)
- Flag this as lower-confidence verification

Some tables use macros defined elsewhere in the .tex file (e.g., `\modelname` or `\best{85.2}`). You need to:
- Search for `\newcommand` or `\def` definitions in the .tex files
- Expand macros before reading values
- Common patterns: `\best{X}` for bold-best, `\second{X}` for underline-second

## Audit 3: Internal Consistency

### 3a. Score ranges
- Standard benchmarks: scores in [0, 1]
- No NaN, no negative (unless perplexity)

### 3b. Duplicates
- No (model_id, benchmark) with conflicting scores within a single source

### 3c. Model ID consistency
- Same model should have identical ID string across all sources
- Flag malformed IDs

### 3d. Benchmark name consistency
- List all distinct benchmark names
- Flag extraction artefacts

### 3e. Developer attribution
- Flag any "unknown" developers

## Audit 4: Dataset Statistics

Compute and report (do not compare against any "expected" numbers):

```python
import json, pathlib
from collections import defaultdict

records = [json.loads(p.read_text()) for p in pathlib.Path("data").rglob("*.json")]
print(f"Total records: {len(records)}")

model_ids = {r["model_info"]["id"] for r in records}
print(f"Unique model IDs: {len(model_ids)}")

eval_index = defaultdict(set)
for r in records:
    source = r["source_metadata"]["source_name"]
    for ev in r["evaluation_results"]:
        eval_index[(r["model_info"]["id"], ev["evaluation_name"])].add(source)

collisions = {k: v for k, v in eval_index.items() if len(v) >= 2}
print(f"Collision pairs: {len(collisions)}")
print(f"Unique models in collisions: {len({k[0] for k in collisions})}")
print(f"Benchmarks with collisions: {sorted({k[1] for k in collisions})}")
```

## Audit 5: Cross-Source Delta Plausibility

For every collision pair, compute the score delta:

- Near-zero deltas between papers using the same harness and n-shot: expected
- Large deltas (>0.10) need an explanation (different n-shot, CoT vs. standard, different harness)
- **If you see a large unexplained delta: go to the LaTeX source of BOTH papers and read both values directly. Determine which (if either) JSON record is wrong.**

## Audit 6: Corrections Report Review

If the migration model produced `data_corrections_report.md`:
- For a sample of reported corrections, verify the new value against LaTeX source
- Flag any corrections that are wrong

## Output

Produce `verification_report.md` with:

1. **Executive summary**: PASS / CONDITIONAL PASS / FAIL, with pass count out of 50+ checks
2. **Schema compliance**: count
3. **Score verification table**: every value checked — raw LaTeX cell content, the JSON score, verdict. This is the heart of the report.
4. **create_paper_records.py verification**: sample check results
5. **Internal consistency**: issues found
6. **Dataset statistics**: computed values
7. **Cross-source delta plausibility**: flagged anomalies with LaTeX re-verification
8. **Corrections report review**: assessment
9. **Critical issues**: numbered must-fix list
10. **Recommendations**

## Severity Classification

- **CRITICAL**: Score differs from LaTeX source by >0.01. Schema failure. Systematic column shift.
- **HIGH**: Model ID inconsistency breaking collision detection. Score outside valid range.
- **MEDIUM**: Developer wrong. Non-collision score wrong. Benchmark name inconsistency.
- **LOW**: Minor rounding (≤0.002). Cosmetic issues.

## NEVER STOP

Work through all audits sequentially without pausing for input. If a LaTeX source download fails, fall back to PDF image reading for that paper and note it. Produce the full report before stopping.
do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
