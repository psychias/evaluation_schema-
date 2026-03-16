#!/usr/bin/env python3.10
"""Comprehensive Docling pipeline test on Gemma 3 paper."""
import sys
import json
import csv
import re
from pathlib import Path

# -----------------------------------------------------------------------
# STEP 1: Parse with Docling and save full markdown
# -----------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Parse with Docling")
print("=" * 60)

from docling.document_converter import DocumentConverter

Path("results").mkdir(exist_ok=True)

converter = DocumentConverter()
result = converter.convert("data/pdfs/2503.19786.pdf")
doc = result.document

md = doc.export_to_markdown()
Path("results/docling_full_markdown.md").write_text(md)
print(f"Markdown: {len(md)} chars, {len(md.splitlines())} lines")

# -----------------------------------------------------------------------
# STEP 2: Explore document structure
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Explore document structure")
print("=" * 60)

structure_lines = []

structure_lines.append("=== Document attributes ===")
for attr in sorted(dir(doc)):
    if attr.startswith('_'):
        continue
    try:
        obj = getattr(doc, attr)
        if callable(obj):
            continue
        t = type(obj).__name__
        l = len(obj) if hasattr(obj, '__len__') else 'N/A'
        line = f"doc.{attr}: type={t}, len={l}"
        structure_lines.append(line)
        print(line)
    except Exception as e:
        line = f"doc.{attr}: ERROR {e}"
        structure_lines.append(line)
        print(line)

structure_lines.append("\n=== First items ===")
print("\n=== First items ===")
for attr in ['texts', 'tables', 'pictures', 'groups', 'key_value_items']:
    if hasattr(doc, attr):
        items = list(getattr(doc, attr))
        if items:
            first = items[0]
            block = [
                f"\ndoc.{attr}[0]:",
                f"  type: {type(first).__name__}",
                f"  dir: {[a for a in dir(first) if not a.startswith('_')]}",
                f"  str: {str(first)[:300]}",
            ]
            try:
                block.append(f"  repr: {repr(first)[:300]}")
            except:
                pass
            for line in block:
                structure_lines.append(line)
                print(line)

Path("results/docling_structure.txt").write_text("\n".join(structure_lines))
print("\nSaved: results/docling_structure.txt")

# -----------------------------------------------------------------------
# STEP 3: Extract and verify scores from tables
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Extract and verify scores from tables")
print("=" * 60)

import pandas as pd

tables_found = []
for i, table in enumerate(doc.tables):
    try:
        df = table.export_to_dataframe()
        df.to_csv(f"results/table_{i:02d}.csv", index=False)
        caption = ""
        for attr in ['caption', 'captions', 'text', 'label']:
            if hasattr(table, attr):
                v = getattr(table, attr)
                if v:
                    caption = str(v)[:200]
                    break
        tables_found.append({'index': i, 'shape': str(df.shape), 'caption': caption[:100]})
        print(f"Table {i:02d}: {df.shape} | caption: {caption[:80]}")
        print(df.head(3).to_string())
        print()
    except Exception as e:
        print(f"Table {i:02d}: ERROR {e}")

print(f"\nTotal tables: {len(tables_found)}")

# Search all tables for known values
search_values = {
    'MMLU-Pro': 67.5, 'GPQA Diamond': 42.4, 'MMLU': 78.6,
    'GSM8K': 82.6, 'HumanEval': 48.8, 'HellaSwag': 85.6, 'ARC-C': 70.6
}
verification = []
for i, table in enumerate(doc.tables):
    try:
        df = table.export_to_dataframe()
        df_str = df.to_string()
        for bench, expected in search_values.items():
            if bench.lower() in df_str.lower():
                for col in df.columns:
                    for idx in df.index:
                        cell = str(df.at[idx, col])
                        try:
                            v = float(cell.replace(',', ''))
                            if abs(v - expected) < 0.2:
                                verification.append({
                                    'table': i,
                                    'benchmark': bench,
                                    'expected': expected,
                                    'extracted': v,
                                    'pass': True
                                })
                                print(f"PASS: Table {i} {bench}: {v} (expected {expected})")
                        except:
                            pass
    except:
        pass

with open('results/score_verification.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['table', 'benchmark', 'expected', 'extracted', 'pass'])
    w.writeheader()
    w.writerows(verification)
print(f"Verified: {len(verification)} values")

# -----------------------------------------------------------------------
# STEP 4: Methodology extraction from text
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Methodology extraction from text")
print("=" * 60)

all_texts = []
if hasattr(doc, 'texts'):
    for t in doc.texts:
        label = str(getattr(t, 'label', ''))
        text = str(getattr(t, 'text', getattr(t, 'orig', '')))
        prov = getattr(t, 'prov', [])
        page = str(prov[0].page_no if prov else '')
        all_texts.append({'label': label, 'text': text, 'page': page})

print(f"Total text items: {len(all_texts)}")

# 4a: Section headings
headings = [t for t in all_texts if 'head' in t['label'].lower() or 'section' in t['label'].lower() or t['label'] in ['title', 'Title']]
print(f"\n=== HEADINGS ({len(headings)}) ===")
for h in headings:
    print(f"  [{h['label']}] p{h['page']}: {h['text'][:100]}")

heading_text = '\n'.join(f"[{h['label']}] p{h['page']}: {h['text']}" for h in headings)
Path('results/section_headings.txt').write_text(heading_text)

# 4b: Table captions
captions = [t for t in all_texts if 'caption' in t['label'].lower()]
print(f"\n=== TABLE CAPTIONS ({len(captions)}) ===")
for c in captions:
    print(f"  [{c['label']}] p{c['page']}: {c['text'][:150]}")

Path('results/table_captions.txt').write_text(
    '\n\n'.join(f"[{c['label']}] p{c['page']}:\n{c['text']}" for c in captions)
)

# 4c: Find Table 19 (eval details) in captions
print("\n=== LOOKING FOR TABLE 19 ===")
for c in captions:
    if '19' in c['text'] or 'eval' in c['text'].lower() or 'detail' in c['text'].lower() or 'config' in c['text'].lower() or 'n-shot' in c['text'].lower():
        print(f"  CANDIDATE: {c['text'][:200]}")

# -----------------------------------------------------------------------
# STEP 5: Search for methodology facts in full markdown
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Search for methodology facts in full markdown")
print("=" * 60)

md_text = Path('results/docling_full_markdown.md').read_text()

searches = {
    'harness': ['lm-evaluation-harness', 'lm_eval', 'LightEval', 'lighteval', 'HELM', 'harness'],
    'temperature': ['temperature', 'greedy', 'T=0', 'T = 0'],
    'n-shot': ['0-shot', '5-shot', '8-shot', 'few-shot', 'n-shot', 'shot'],
    'normalization': ['normalization', 'normaliz', 'char-len', 'byte', 'token norm'],
    'scoring': ['log-likelihood', 'generation', 'scoring', 'pass@1'],
    'prompt': ['prompt template', 'prompt format', 'chain-of-thought', 'CoT'],
}

findings = {}
for category, terms in searches.items():
    hits = []
    for term in terms:
        for m in re.finditer(re.escape(term), md_text, re.IGNORECASE):
            start = max(0, m.start() - 200)
            end = min(len(md_text), m.end() + 200)
            context = md_text[start:end].replace('\n', ' ')
            hits.append({'term': term, 'context': context})
    if hits:
        findings[category] = hits[:5]
        print(f"\n{category.upper()}: {len(hits)} hits")
        for h in hits[:3]:
            print(f"  [{h['term']}]: ...{h['context'][:200]}...")

with open('results/methodology_findings.txt', 'w') as f:
    for cat, hits in findings.items():
        f.write(f"\n{'=' * 40}\n{cat.upper()}\n{'=' * 40}\n")
        for h in hits:
            f.write(f"\n[{h['term']}]:\n...{h['context']}...\n")

# 4d: Find evaluation sections
print("\n=== EVALUATION SECTIONS ===")
eval_section_texts = []
in_eval_section = False
for t in all_texts:
    label = t['label'].lower()
    text = t['text']
    if any(kw in text.lower() for kw in ['evaluation', 'benchmark', 'experimental', 'pre-training eval', 'post-training eval']):
        if 'head' in label:
            in_eval_section = True
            eval_section_texts.append(f"\n## {text}\n")
            print(f"  SECTION: {text[:100]}")
    if in_eval_section and 'head' not in label:
        eval_section_texts.append(text)
        if len('\n'.join(eval_section_texts)) > 20000:
            break

Path('results/eval_sections.txt').write_text('\n'.join(eval_section_texts))
print(f"Eval sections text: {len(''.join(eval_section_texts))} chars")

# -----------------------------------------------------------------------
# STEP 6: Find Table 19 (eval details) as structured data
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Searching for eval details table")
print("=" * 60)

for i, table in enumerate(doc.tables):
    try:
        df = table.export_to_dataframe()
        df_str = df.to_string().lower()
        score = sum(1 for kw in ['shot', 'metric', 'normal', 'cot', 'benchmark', 'task'] if kw in df_str)
        if score >= 2:
            print(f"\nCandidate Table {i} (score={score}):")
            print(df.to_string())
            df.to_csv(f'results/eval_details_table_{i:02d}.csv', index=False)
    except Exception as e:
        print(f"Table {i}: {e}")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE")
print("=" * 60)
print("\nFiles in results/:")
for f in sorted(Path("results").iterdir()):
    print(f"  {f.name}: {f.stat().st_size} bytes")
