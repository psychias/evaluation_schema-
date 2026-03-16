"""
Phase 1 comparison: Docling vs pdfplumber on the same PDF.
Verifies Docling's API works and produces comparable table data.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pdfplumber
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

PDF_PATH = Path("scripts/scrapers/raw/papers/2309.10305.pdf")


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    # --- pdfplumber ---
    print("=" * 60)
    print("pdfplumber extraction")
    print("=" * 60)
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        pb_table_count = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            for table_obj in page.find_tables():
                raw = table_obj.extract()
                if not raw or len(raw) < 2:
                    continue
                pb_table_count += 1
                cleaned = [
                    [str(c).strip() if c else "" for c in row]
                    for row in raw
                ]
                print(f"\n--- pdfplumber Table {pb_table_count} (page {page_num}) ---")
                for row in cleaned[:5]:
                    print("  ", row[:8])
                if len(cleaned) > 5:
                    print(f"  ... ({len(cleaned)} rows total)")
        print(f"\nTotal pdfplumber tables: {pb_table_count}")

    # --- Docling ---
    print("\n" + "=" * 60)
    print("Docling extraction")
    print("=" * 60)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    result = converter.convert(str(PDF_PATH))

    print(f"\nDocling found {len(result.document.tables)} tables.\n")
    for i, table in enumerate(result.document.tables):
        try:
            df = table.export_to_dataframe()
            print(f"--- Docling Table {i} ({df.shape[0]} rows x {df.shape[1]} cols) ---")
            # Show first 5 rows, first 8 cols
            print(df.iloc[:5, :8].to_string())
            if df.shape[0] > 5:
                print(f"  ... ({df.shape[0]} rows total)")
            print()
        except Exception as exc:
            print(f"--- Docling Table {i}: export failed: {exc} ---\n")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"pdfplumber: {pb_table_count} tables")
    print(f"Docling:    {len(result.document.tables)} tables")

    # Verify key Docling API features
    print("\nDocling API verification:")
    print(f"  .tables count:             {len(result.document.tables)}")
    print(f"  .export_to_markdown():     {len(result.document.export_to_markdown())} chars")
    if result.document.tables:
        t0 = result.document.tables[0]
        df0 = t0.export_to_dataframe()
        print(f"  .export_to_dataframe():    {df0.shape}")
    print("\nDocling API is functional. Phase 1 complete.")


if __name__ == "__main__":
    main()
