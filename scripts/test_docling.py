"""
Phase 1: Test Docling's table extraction on Mixtral paper (2312.11805).

Verifies that Docling correctly extracts the known ground-truth values:
  - Mistral-7B-v0.1 on GSM8K:     0.352 (n_shot=5)
  - Mistral-7B-v0.1 on HellaSwag: 0.833 (n_shot=10)
  - Mistral-7B-v0.1 on MMLU:      0.601
"""

from __future__ import annotations

import sys
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

PDF_PATH = Path("scripts/scrapers/raw/papers/2309.10305.pdf")

GROUND_TRUTH = {
    ("Mistral 7B", "HellaSwag"): 0.812,
    ("Mistral 7B", "MMLU"): 0.601,
}

TOLERANCE = 0.002


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {PDF_PATH} with Docling...")

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

    # --- Explore tables ---
    print(f"\nFound {len(result.document.tables)} tables in the document.\n")

    for i, table in enumerate(result.document.tables):
        print(f"--- Table {i} ---")
        try:
            df = table.export_to_dataframe()
            print(df.to_string(index=False))
            print()
        except Exception as exc:
            print(f"  Could not export to DataFrame: {exc}")
            # Try dict export instead
            try:
                d = table.export_to_dict()
                print(f"  Dict keys: {list(d.keys())}")
                print()
            except Exception as exc2:
                print(f"  Dict export also failed: {exc2}\n")

    # --- Extract markdown for prose context ---
    print("\n--- Document Markdown (first 3000 chars) ---")
    md = result.document.export_to_markdown()
    print(md[:3000])
    print("...")

    # --- Check ground truth ---
    print("\n\n--- Ground Truth Verification ---")
    print("Searching all tables for known values...\n")

    found: dict[tuple[str, str], float | None] = {k: None for k in GROUND_TRUTH}

    for i, table in enumerate(result.document.tables):
        try:
            df = table.export_to_dataframe()
        except Exception:
            continue

        # Print tables that contain benchmark keywords
        text = df.to_string().lower()
        if any(kw in text for kw in ("gsm8k", "hellaswag", "mmlu")):
            print(f"Table {i} contains benchmark keywords:")
            print(df.to_string(index=False))
            print()

            # Try to find ground truth values
            for col in df.columns:
                col_lower = col.lower().strip()
                for row_idx in range(len(df)):
                    cell = str(df.iloc[row_idx, 0]).strip()
                    val_str = str(df.iloc[row_idx][col]).strip()
                    try:
                        val = float(val_str.rstrip("%").strip())
                    except ValueError:
                        continue

                    # Normalise percentage to 0-1 if needed
                    if val > 1.0:
                        val_norm = val / 100.0
                    else:
                        val_norm = val

                    for (model_gt, bench_gt), expected in GROUND_TRUTH.items():
                        if (model_gt.lower() in cell.lower() or
                                model_gt.lower().replace(" ", "-") in cell.lower()):
                            if bench_gt.lower() in col_lower:
                                found[(model_gt, bench_gt)] = val_norm

    print("\n--- Results ---")
    all_ok = True
    for (model, bench), expected in GROUND_TRUTH.items():
        actual = found[(model, bench)]
        if actual is None:
            status = "NOT FOUND"
            all_ok = False
        elif abs(actual - expected) <= TOLERANCE:
            status = "OK"
        else:
            status = f"MISMATCH (delta={actual - expected:+.4f})"
            all_ok = False
        print(f"  {model} / {bench}: expected={expected}, actual={actual} — {status}")

    print()
    if all_ok:
        print("All ground truth values verified. Proceed to Phase 2.")
    else:
        print("Some values missing or mismatched. Debug before proceeding.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
