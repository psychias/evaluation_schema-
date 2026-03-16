import re
import json
import csv
from pathlib import Path

SRC_DIR = Path("extraction_comparison/data/latex_src")
OUT_DIR = Path("extraction_comparison/results")
OUT_DIR.mkdir(exist_ok=True)

# Collect all \newcommand / \def macros
macros = {}
for tex_file in SRC_DIR.rglob("*.tex"):
    content = tex_file.read_text(errors='replace')
    for m in re.finditer(r'\\(?:newcommand|renewcommand)\*?\s*\{(\\[a-zA-Z]+)\}(?:\[\d+\])?\{([^}]*)\}', content):
        macros[m.group(1)] = m.group(2)
    for m in re.finditer(r'\\def\s*(\\[a-zA-Z]+)\s*\{([^}]*)\}', content):
        macros[m.group(1)] = m.group(2)
print(f"Found {len(macros)} macros")
for k, v in sorted(macros.items()):
    print(f"  {k} -> {v}")

def expand_macros(text, macros):
    for name, val in sorted(macros.items(), key=lambda x: -len(x[0])):
        safe_val = val  # capture in closure
        text = re.sub(re.escape(name) + r'(?![a-zA-Z])', lambda m, v=safe_val: v, text)
    return text

def clean_cell(cell):
    cell = re.sub(r'(?<!\\)%.*$', '', cell, flags=re.MULTILINE)
    # Strip bold/italic/formatting wrappers
    for cmd in ['textbf', 'textit', 'emph', 'underline', 'textsc', 'textrm',
                'text', 'mathrm', 'mathbf', 'mathit', 'ul', 'uline',
                'best', 'second', 'first', 'sota', 'ours']:
        cell = re.sub(r'\\' + cmd + r'\{([^{}]*)\}', r'\1', cell)
    cell = re.sub(r'\{\\(?:bf|it|em|sc)\s+([^}]*)\}', r'\1', cell)
    cell = re.sub(r'\\(?:bf|it|em|sc)\b\s*', '', cell)
    # multirow / multicolumn
    cell = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}', r'\1', cell)
    cell = re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}', r'\1', cell)
    # rule commands
    cell = re.sub(r'\\(?:hline|toprule|midrule|bottomrule|cmidrule(?:\([^)]*\))?\{[^}]*\}|addlinespace(?:\[[^\]]*\])?)\s*', '', cell)
    # color
    cell = re.sub(r'\\(?:rowcolor|cellcolor)\{[^}]*\}', '', cell)
    # math mode
    cell = re.sub(r'\$([^$]*)\$', r'\1', cell)
    # remaining commands with braces
    cell = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', cell)
    # remaining backslash commands
    cell = re.sub(r'\\[a-zA-Z]+', '', cell)
    # stray braces
    cell = re.sub(r'[{}]', '', cell)
    return cell.strip()

def parse_tabular(content):
    lines = content.split('\n')
    lines = [re.sub(r'(?<!\\)%.*$', '', l) for l in lines]
    content = '\n'.join(lines)
    # Remove begin/end
    content = re.sub(r'\\begin\{tabular[x*]?\}(?:\{[^}]*\}){1,3}', '', content)
    content = re.sub(r'\\end\{tabular[x*]?\}', '', content)
    content = re.sub(r'\\begin\{tabulary\}\{[^}]*\}\{[^}]*\}', '', content)
    content = re.sub(r'\\end\{tabulary\}', '', content)
    # Remove rule lines
    content = re.sub(r'\\(?:hline|toprule|midrule|bottomrule|cmidrule(?:\([^)]*\))?\{[^}]*\}|addlinespace(?:\[[^\]]*\])?)\s*', '\n', content)
    raw_rows = re.split(r'\\\\', content)
    rows = []
    for row in raw_rows:
        row = row.strip()
        if not row:
            continue
        cells = [clean_cell(c) for c in row.split('&')]
        if all(c == '' for c in cells):
            continue
        rows.append(cells)
    return rows

BENCH_KEYWORDS = [
    'MMLU', 'GSM', 'HumanEval', 'HellaSwag', 'ARC', 'WinoGrande', 'MATH',
    'MBPP', 'GPQA', 'AGIEval', 'BBH', 'BBEH', 'TriviaQA', 'PIQA', 'BoolQ',
    'LiveCode', 'MGSM', 'DROP', 'Flores', 'XQuAD', 'IFEval', 'WMT',
    'DocVQA', 'InfoVQA', 'TextVQA', 'ChartQA', 'MMMU', 'VQA', 'COCO',
    'AI2D', 'BLINK', 'RULER', 'MRCR', 'HiddenMath', 'ECLeKTic',
    'SimpleQA', 'FACTS', 'RealWorldQA', 'NaturalQ', 'Winogrande',
    'benchmark', 'accuracy', 'score', 'evaluation'
]

all_tables = []
table_idx = 0

for tex_file in sorted(SRC_DIR.rglob("*.tex")):
    content = tex_file.read_text(errors='replace')
    content_exp = expand_macros(content, macros)

    pattern = r'\\begin\{(tabular[x*]?|tabulary)\}(.*?)\\end\{\1\}'
    for m in re.finditer(pattern, content_exp, re.DOTALL):
        table_idx += 1
        raw_tex = m.group(0)
        rows = parse_tabular(raw_tex)
        if not rows:
            continue

        (OUT_DIR / f"latex_table{table_idx}_raw.tex").write_text(raw_tex)

        csv_path = OUT_DIR / f"latex_table{table_idx}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        flat = ' '.join(' '.join(r) for r in rows)
        has_bench = any(kw.lower() in flat.lower() for kw in BENCH_KEYWORDS)

        print(f"\nTable {table_idx} [{tex_file.name}] — {len(rows)} rows, bench={has_bench}")
        for row in rows[:5]:
            print("  ", [c[:35] for c in row])

        all_tables.append({
            "index": table_idx,
            "file": str(tex_file.relative_to(SRC_DIR)),
            "rows": len(rows),
            "has_benchmark_keywords": has_bench,
            "data": rows
        })

with open(OUT_DIR / "latex_all_tables.json", 'w') as f:
    json.dump(all_tables, f, indent=2)

bench_count = sum(1 for t in all_tables if t['has_benchmark_keywords'])
print(f"\n\nTotal tabular environments: {table_idx}")
print(f"Benchmark tables: {bench_count}")
