# EEE Project Memory

## Project
ACL 2026 paper: "You Can't Compare What You Don't Document: Metadata Gaps in LLM Benchmark Reporting"
Path: /Users/steliospsychias/Desktop/every_eval_ever_restored
Python env: .venv/bin/python (Python 3.13, Docling installed)

## Phase Status
- Phase 1 (Docling test): DONE
- Phase 2 (Migrate extract_paper.py): DONE — all pdfplumber replaced with Docling
- Phase 3 (Batch re-extraction): DONE — 50 papers, 413 new files, 0 schema failures
- Phase 3c (Verification): DONE — 10/10 ground truth values pass
- Phase 4 (create_paper_records.py audit): DONE — no corrections needed
- Phase 5 (Leaderboard scrapers): DONE — all 6 re-run
- Phase 6 (Analysis + Figures): DONE — 154 collision pairs, 22 figures
- Phase 7 (Add more papers): DONE — 17 new papers added (67 total IDs in arxiv_ids_full.txt)

## Key Files
- scripts/extract_paper.py — main pipeline (Docling-based)
- scripts/arxiv_ids_full.txt — 50 arXiv IDs for batch extraction
- scripts/verify_sample.py — Phase 3c ground truth verification
- data_corrections_report.md — Phase 6 report
- extraction_full.log — batch extraction log
- analysis_output/*.csv — all analysis outputs
- figures/*.pdf — all regenerated figures

## Architecture Notes
- `DoclingParser` class added to extract_paper.py — module-level singleton `_docling_parser`
- All extractors (TableExtractor, ProseExtractor, LLMFallbackExtractor, _detect_eval_library) share it
- Key fix: include DataFrame column headers as first row in table_rows (benchmark names in headers)
- Use .venv/bin/python for all commands

## Analysis Results (Final — Post-Phase 7)
- 154 collision pairs (103 genuine deltas, 51 same-score diff methodology)
- 11,107 valid JSON records (11,114 total; 1 invalid = .claude/settings.local.json, harmless)
- 75 sources total (6 leaderboards + 69 paper directories)
- 67 arXiv IDs in arxiv_ids_full.txt (50 original + 17 Phase 7 additions)
- prompt_template is dominant predictor of score variance (partial R²=0.136)
- 3 mandatory collision pairs confirmed intact
- 22 PDF figures generated; 16 PNG copies in submission/

## Known Issues
- 17/50 papers produce no results (Docling cell-merge issue on multi-level headers)
- Llama 2 table 3 is most notable case: model+size+scores merge into one cell
- data/.claude/settings.local.json picked up as invalid by validate_all.py (harmless)
