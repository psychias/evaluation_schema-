import json
import csv
from pathlib import Path

OUT_DIR = Path("extraction_comparison/results")

def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []

pdfplumber_tables = load_json(OUT_DIR / "pdfplumber_all_tables.json")
docling_tables = load_json(OUT_DIR / "docling_all_tables.json")
latex_tables = load_json(OUT_DIR / "latex_all_tables.json")

BENCH_KW = ['MMLU', 'GSM8K', 'GSM', 'MATH', 'HumanEval', 'HellaSwag', 'ARC',
            'WinoGrande', 'MBPP', 'GPQA', 'AGIEval', 'BBH', 'BBEH', 'TriviaQA',
            'LiveCode', 'MGSM', 'DROP', 'IFEval', 'DocVQA', 'InfoVQA', 'TextVQA',
            'ChartQA', 'MMMU', 'VQA', 'RULER', 'MRCR', 'HiddenMath', 'ECLeKTic',
            'SimpleQA', 'BoolQ', 'PIQA', 'NaturalQ', 'WinoG', 'AGIE', 'Winogrande']

def has_bench(data):
    flat = str(data).lower()
    return any(k.lower() in flat for k in BENCH_KW)

pp_bench = [t for t in pdfplumber_tables if has_bench(t.get('data', []))]
dl_bench = [t for t in docling_tables if has_bench(t.get('data', []))]
lt_bench = [t for t in latex_tables if t.get('has_benchmark_keywords', False)]

lines = [
    "# Extraction Comparison: Gemma 3 Technical Report (arXiv:2503.19786)",
    "",
    "## Method Summary",
    "",
    f"| Method | Total tables found | Benchmark tables |",
    f"|--------|-------------------|-----------------|",
    f"| **pdfplumber** | {len(pdfplumber_tables)} | {len(pp_bench)} |",
    f"| **Docling** | {len(docling_tables)} | {len(dl_bench)} |",
    f"| **LaTeX source** | {len(latex_tables)} | {len(lt_bench)} |",
    "",
    "---",
    "",
    "## LaTeX Source Tables (ground truth)",
    "_These are parsed directly from the .tex files — exact numbers the authors typed._",
    "",
]

for t in lt_bench:
    lines.append(f"### LaTeX Table {t['index']} — `{t['file']}`  ({t['rows']} rows)")
    lines.append("```")
    for row in t['data']:
        lines.append(" | ".join(str(c)[:40] for c in row))
    lines.append("```")
    lines.append("")

lines += ["---", "", "## Docling Tables (ML layout detection)", ""]
for t in dl_bench:
    data = t.get('data', [])
    lines.append(f"### Docling Table {t.get('index','?')}  (page {t.get('page','?')}, {t.get('rows','?')} rows)")
    lines.append("```")
    if isinstance(data, list):
        for row in data[:15]:
            if isinstance(row, dict):
                lines.append(" | ".join(str(v)[:40] for v in row.values()))
            else:
                lines.append(str(row)[:120])
    lines.append("```")
    lines.append("")

lines += ["---", "", "## pdfplumber Tables", ""]
for t in pp_bench:
    lines.append(f"### pdfplumber Page {t['page']} Table {t['table_index_on_page']} (global #{t['global_index']}, {t['rows']} rows)")
    lines.append("```")
    for row in t['data']:
        lines.append(" | ".join(str(c)[:40] for c in row))
    lines.append("```")
    lines.append("")

# ----- Key value extraction for benchmark tables -----
lines += ["---", "", "## Key Score Comparison", "",
          "Comparing specific (model, benchmark) values across methods where the table appears in multiple methods.",
          ""]

# Focus on the core pre-training eval table (LaTeX Table 11 / Docling Table 10)
# LaTeX table 11: MMLU, MMLUpro, AGIE, MATH, GSM8K, GPQA, MBPP, HumanEval for Gemma 2 vs Gemma 3
# Find matching docling table
CORE_BENCHMARKS = ['MMLU', 'MATH', 'GSM8K', 'HumanEval', 'MBPP', 'GPQA']

# Build a simple lookup from docling data
docling_lookup = {}  # (model_col_header, benchmark) -> value
for t in docling_tables:
    data = t.get('data', [])
    if not data or not isinstance(data, list):
        continue
    if not isinstance(data[0], dict):
        continue
    flat = str(data).lower()
    if not any(k.lower() in flat for k in CORE_BENCHMARKS):
        continue
    cols = list(data[0].keys())
    for row in data:
        bench = list(row.values())[0] if row else ''
        for col, val in list(row.items())[1:]:
            docling_lookup[(str(col).strip(), str(bench).strip())] = str(val).strip()

# LaTeX lookup
latex_lookup = {}
for t in lt_bench:
    if t['index'] not in [11, 19]:  # pre-training and instruction-tuned eval tables
        continue
    rows = t['data']
    if len(rows) < 3:
        continue
    # First data row after headers has model names
    # Find header row: row with model names in first cell or empty first cell
    header_row = None
    for r in rows:
        if any(m in str(r) for m in ['2B', '4B', '9B', '12B', '27B', '1B']):
            header_row = r
            break
    if not header_row:
        continue
    for row in rows[rows.index(header_row)+1:]:
        bench = row[0].strip()
        for i, col in enumerate(header_row[1:], 1):
            if i < len(row):
                latex_lookup[(col.strip(), bench)] = row[i].strip()

lines.append("### Pre-training eval (LaTeX Table 11): Gemma 2 vs Gemma 3 — key scores")
lines.append("")
lines.append("| Benchmark | Gemma 2 9B (LaTeX) | Gemma 3 12B (LaTeX) | Gemma 3 27B (LaTeX) | Gemma 3 12B (Docling) | Gemma 3 27B (Docling) |")
lines.append("|-----------|-------------------|--------------------|--------------------|----------------------|----------------------|")

# Get LaTeX table 11 directly
lt11 = next((t for t in latex_tables if t['index'] == 11), None)
# Get Docling table 10 (pre-training scores)
dl10 = next((t for t in docling_tables if t.get('index') == 10), None)

if lt11 and dl10:
    lt_rows = {r[0].strip(): r for r in lt11['data'][2:]}  # skip 2 header rows
    dl_rows_raw = dl10.get('data', [])
    dl_rows = {}
    if dl_rows_raw and isinstance(dl_rows_raw[0], dict):
        for row in dl_rows_raw:
            vals = list(row.values())
            if vals:
                dl_rows[str(vals[0]).strip()] = row

    bench_map = {
        'MMLU': 'MMLU', 'MMLUpro': 'MMLUpro', 'AGIE': 'AGIE',
        'MATH': 'MATH', 'GSM8K': 'GSM8K', 'GPQA Diamond': 'GPQA Diamond',
        'MBPP': 'MBPP', 'HumanE': 'HumanE'
    }

    for bench_lt, bench_dl in bench_map.items():
        lt_row = lt_rows.get(bench_lt, [])
        # LaTeX row: [bench, 2B, 9B, 27B, '', 4B, 12B, 27B]
        g2_9b = lt_row[2] if len(lt_row) > 2 else '-'
        g3_12b_lt = lt_row[6] if len(lt_row) > 6 else '-'
        g3_27b_lt = lt_row[7] if len(lt_row) > 7 else '-'

        dl_row = dl_rows.get(bench_dl, {})
        dl_cols = list(dl_row.keys()) if dl_row else []
        g3_12b_dl = '-'
        g3_27b_dl = '-'
        for col in dl_cols:
            if '12B' in str(col) or '12b' in str(col).lower():
                g3_12b_dl = str(dl_row[col])
            if '27B' in str(col) or '27b' in str(col).lower():
                g3_27b_dl = str(dl_row[col])

        match_12 = '✓' if g3_12b_lt == g3_12b_dl else ('~' if g3_12b_lt == '-' or g3_12b_dl == '-' else '✗')
        match_27 = '✓' if g3_27b_lt == g3_27b_dl else ('~' if g3_27b_lt == '-' or g3_27b_dl == '-' else '✗')

        lines.append(f"| {bench_lt} | {g2_9b} | {g3_12b_lt} {match_12} | {g3_27b_lt} {match_27} | {g3_12b_dl} | {g3_27b_dl} |")

Path(OUT_DIR / "summary.md").write_text('\n'.join(lines))
print("Wrote summary.md")
print(f"\nLaTeX benchmark tables: {len(lt_bench)}")
print(f"Docling benchmark tables: {len(dl_bench)}")
print(f"pdfplumber benchmark tables: {len(pp_bench)}")
