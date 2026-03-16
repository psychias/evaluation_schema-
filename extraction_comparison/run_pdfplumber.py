import pdfplumber
import json
import csv
from pathlib import Path

PDF_PATH = "extraction_comparison/data/2503.19786.pdf"
OUT_DIR = Path("extraction_comparison/results")
OUT_DIR.mkdir(exist_ok=True)

all_tables = []
table_idx = 0

with pdfplumber.open(PDF_PATH) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        tables = page.extract_tables()
        for t_idx, table in enumerate(tables):
            table_idx += 1
            rows = len(table)
            cols = len(table[0]) if table else 0
            print(f"Page {page_num}, table {t_idx+1}: {rows}r x {cols}c")

            csv_path = OUT_DIR / f"pdfplumber_page{page_num}_table{t_idx+1}.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in table:
                    writer.writerow([cell or '' for cell in row])

            all_tables.append({
                "page": page_num,
                "table_index_on_page": t_idx + 1,
                "global_index": table_idx,
                "rows": rows,
                "cols": cols,
                "data": [[cell or '' for cell in row] for row in table]
            })

with open(OUT_DIR / "pdfplumber_all_tables.json", 'w') as f:
    json.dump(all_tables, f, indent=2)

print(f"\nTotal tables found: {table_idx}")
