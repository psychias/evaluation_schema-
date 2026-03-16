from docling.document_converter import DocumentConverter
import json
import pandas as pd
from pathlib import Path

PDF_PATH = "extraction_comparison/data/2503.19786.pdf"
OUT_DIR = Path("extraction_comparison/results")
OUT_DIR.mkdir(exist_ok=True)

converter = DocumentConverter()
result = converter.convert(PDF_PATH)

all_tables = []
for t_idx, table in enumerate(result.document.tables):
    print(f"\n=== Table {t_idx+1} ===")
    try:
        df = table.export_to_dataframe()
        csv_path = OUT_DIR / f"docling_table{t_idx+1}.csv"
        df.to_csv(csv_path, index=False)
        print(df.to_string())

        page_info = None
        try:
            page_info = table.prov[0].page_no if hasattr(table, 'prov') and table.prov else None
        except:
            pass

        all_tables.append({
            "index": t_idx + 1,
            "page": page_info,
            "rows": len(df),
            "cols": len(df.columns),
            "data": df.fillna('').astype(str).to_dict(orient='records')
        })
    except Exception as e:
        print(f"  Error: {e}")
        all_tables.append({"index": t_idx+1, "error": str(e)})

with open(OUT_DIR / "docling_all_tables.json", 'w') as f:
    json.dump(all_tables, f, indent=2)

print(f"\nTotal tables: {len(all_tables)}")
