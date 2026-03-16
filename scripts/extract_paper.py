"""
PDF and arXiv paper extraction pipeline for the EEE schema.

downloads PDFs from arXiv (or uses local files), extracts results tables
using Docling (IBM), and converts found benchmarks to EEE schema JSON files.

design follows SOLID principles:
  - PDFDownloader: fetch only
  - TableExtractor: table detection only
  - ResultsTableParser: parsing only
  - PaperConverter: schema conversion only
  - PaperWriter: file I/O only
  - PaperExtractionPipeline: orchestration only

usage:
    python scripts/extract_paper.py --arxiv_id 2407.21783
    python scripts/extract_paper.py --pdf path/to/paper.pdf
    python scripts/extract_paper.py --batch scripts/arxiv_ids.txt
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# add repo root and utils/ to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

import requests

from eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from helpers import get_developer
from eval_converters import SCHEMA_VERSION as _SCHEMA_VERSION

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
_TIMEOUT = 60  # seconds for HTTP
_PDF_DOWNLOAD_DIR = Path("scripts/scrapers/raw/papers")

# Specific benchmark names — a single match is sufficient evidence that a
# table contains evaluation results.
_BENCHMARK_KEYWORDS_STRONG: set[str] = {
    "mmlu", "humaneval", "human eval", "gsm8k", "gsm-8k",
    "bbh", "big-bench", "big bench", "hellaswag", "hella swag",
    "arc", "truthfulqa", "truthful qa", "winogrande", "wino grande",
    "piqa", "lambada", "pass@k", "pass@1", "mbpp", "human-eval",
    "math", "gpqa", "musr", "mmlu-pro", "mmlu pro", "ifeval", "if eval",
    "drop", "nq", "triviaqa", "trivia qa", "copa", "agieval",
    "boolq", "bool q", "swag", "race", "squad", "natural questions",
    "codexglue", "humaneval+", "bigcode", "swe-bench", "mmstar",
    "livecodebench", "aime", "agi eval", "c-eval",
}

# Generic metric terms — too common alone (any table with an 'accuracy'
# column would match).  Require ≥2 distinct weak matches to count as
# sufficient evidence, reducing false positives from non-eval tables.
_BENCHMARK_KEYWORDS_WEAK: set[str] = {
    "accuracy", "bleu", "rouge", "f1", "em", "exact match",
}

# Combined set for any code that still needs the full list (e.g. when
# identifying which *columns* are benchmark columns in a parsed table).
_BENCHMARK_KEYWORDS: set[str] = _BENCHMARK_KEYWORDS_STRONG | _BENCHMARK_KEYWORDS_WEAK

# Pre-built alternation string used by ProseExtractor regex patterns.
# Sorted longest-first so that "mmlu-pro" / "human eval" etc. match before
# the shorter "mmlu" / "eval" prefixes that would shadow them.
_BENCH_ALT_RE: str = "|".join(
    re.escape(kw)
    for kw in sorted(_BENCHMARK_KEYWORDS_STRONG, key=len, reverse=True)
)

# ---------------------------------------------------------------------------
# benchmark metadata — lower_is_better, score ranges, score types
# ---------------------------------------------------------------------------

# Benchmarks where a lower score is better.  Matched with substring search
# against the lowercased benchmark name extracted from the table.
_LOWER_IS_BETTER_PATTERNS: tuple[str, ...] = (
    "perplexity", "ppl", "wer", "cer",
    "word error", "char error",
    "bpw", "bpc", "bits per",
    "latency", "toxicity", "hallucination",
    "error rate", "false positive", "false negative",
)

# Benchmarks whose raw scores are genuinely unbounded (e.g. perplexity).
# For these we skip the ÷100 normalisation step and record the raw value.
_UNBOUNDED_SCORE_PATTERNS: tuple[str, ...] = (
    "perplexity", "ppl", "bpw", "bpc", "bits per",
    "wer", "cer", "word error", "char error",
    "latency",
)

# Benchmarks that are already reported on a 0–100 scale in most papers and
# should be kept on that scale rather than divided to 0–1.
# We record min_score=0, max_score=100 in the schema.
_SCALE_100_PATTERNS: tuple[str, ...] = (
    "bleu", "rouge",
)

# Eval-framework text signatures found in paper body → canonical lib name.
# Longer / more-specific strings must appear before shorter prefix strings.
_EVAL_FRAMEWORK_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("lm-evaluation-harness", "lm_eval"),
    ("lm_eval",               "lm_eval"),
    ("eleutherai/lm-eval",    "lm_eval"),
    ("inspect_ai",            "inspect_ai"),
    ("inspect ai",            "inspect_ai"),
    ("stanford-crfm/helm",    "helm"),
    ("helm",                  "helm"),
    ("openai evals",          "openai_evals"),
    ("big-bench",             "bigbench"),
    ("bigbench",              "bigbench"),
)

# regex matching common footnote / superscript markers attached to cell values
# e.g. "85.2†", "*73.1", "92‡", "81.0 a", "77.3^{1}"
_FOOTNOTE_RE = re.compile(r"[\*†‡§¶#^\d]+$|^[\*†‡§¶#]")

# developer lookup: lower-cased partial model name → developer org.
# Keys are matched via startswith / substring checks in _infer_developer.
# Longer / more-specific keys must appear before shorter prefix keys so
# that e.g. "codellama" is matched before the bare "llama" entry.
_DEVELOPER_MAP: dict[str, str] = {
    # ── OpenAI ──────────────────────────────────────────────────────────
    "gpt": "openai",
    "text-davinci": "openai",
    "text-curie": "openai",
    "davinci": "openai",
    "curie": "openai",
    "o1": "openai",
    "o3": "openai",
    "chatgpt": "openai",
    # ── Anthropic ───────────────────────────────────────────────────────
    "claude": "anthropic",
    # ── Google / DeepMind ────────────────────────────────────────────────
    "gemini": "google",
    "gemma": "google",
    "palm": "google",
    "t5": "google",
    "ul2": "google",
    "flan": "google",
    "chinchilla": "google-deepmind",
    "gopher": "google-deepmind",
    "sparrow": "google",
    # ── Meta ────────────────────────────────────────────────────────────
    "codellama": "meta-llama",    # must precede bare "llama"
    "llama": "meta-llama",
    "opt": "meta",
    # ── Mistral AI ──────────────────────────────────────────────────────
    "mistral": "mistralai",
    "mixtral": "mistralai",
    "open-mistral": "mistralai",
    "codestral": "mistralai",
    "mathstral": "mistralai",
    # ── Qwen (Alibaba) ──────────────────────────────────────────────────
    "qwen": "Qwen",
    # ── Microsoft ───────────────────────────────────────────────────────
    "phi": "microsoft",
    "orca": "microsoft",
    "wizardlm": "microsoft",
    # ── TII ─────────────────────────────────────────────────────────────
    "falcon": "tiiuae",
    # ── BigScience / HuggingFace ─────────────────────────────────────────
    "bloom": "bigscience",
    # ── EleutherAI ──────────────────────────────────────────────────────
    "pythia": "EleutherAI",
    "gpt-j": "EleutherAI",
    "gpt-neox": "EleutherAI",
    # ── MosaicML / Databricks ────────────────────────────────────────────
    "mpt": "mosaicml",
    "dbrx": "databricks",
    # ── DeepSeek ────────────────────────────────────────────────────────
    "deepseek": "deepseek-ai",
    # ── 01.AI ───────────────────────────────────────────────────────────
    "yi": "01-ai",
    # ── Allen AI ────────────────────────────────────────────────────────
    "olmo": "allenai",
    "tulu": "allenai",
    # ── xAI ─────────────────────────────────────────────────────────────
    "grok": "xai",
    # ── Stanford ────────────────────────────────────────────────────────
    "alpaca": "stanford",
    # ── LMSYS ───────────────────────────────────────────────────────────
    "vicuna": "lmsys",
    # ── BigCode ─────────────────────────────────────────────────────────
    "starcoder": "bigcode",
    "santacoder": "bigcode",
    # ── Salesforce ──────────────────────────────────────────────────────
    "codegen": "salesforce",
    "xgen": "salesforce",
    # ── Cohere ──────────────────────────────────────────────────────────
    "command-r": "CohereForAI",   # must precede bare "command"
    "command": "CohereForAI",
    "aya": "CohereForAI",
    # ── AI21 Labs ────────────────────────────────────────────────────────
    "jamba": "ai21labs",
    "jurassic": "ai21labs",
    # ── NVIDIA ──────────────────────────────────────────────────────────
    "nemotron": "nvidia",
    "megatron": "nvidia",
    # ── Upstage ─────────────────────────────────────────────────────────
    "solar": "upstage",
    # ── LG AI Research ──────────────────────────────────────────────────
    "exaone": "LGAI-RESEARCH",
    # ── THUDM (Tsinghua) ────────────────────────────────────────────────
    "chatglm": "THUDM",
    "glm": "THUDM",
    "codegeex": "THUDM",
    # ── Shanghai AI Lab ──────────────────────────────────────────────────
    "internlm": "internlm",
    "internvl": "OpenGVLab",
    # ── Baichuan ────────────────────────────────────────────────────────
    "baichuan": "baichuan-inc",
    # ── Together AI ──────────────────────────────────────────────────────
    "redpajama": "togethercomputer",
    # ── Writer ──────────────────────────────────────────────────────────
    "palmyra": "Writer",
    # ── Teknium / NousResearch ───────────────────────────────────────────
    "openhermes": "teknium",
    "nous-hermes": "NousResearch",
    "hermes": "NousResearch",
}

# simple in-process cache so repeated unknown model names don't all hit the
# HF Hub API during a single batch run.
_HF_AUTHOR_CACHE: dict[str, str] = {}

# Pre-sorted by key length descending so longer / more-specific patterns
# (e.g. "gpt-j", "command-r") match before their shorter prefixes ("gpt",
# "command").  Dict insertion order alone is NOT sufficient because shorter
# keys like "gpt" appear first in _DEVELOPER_MAP and would shadow "gpt-j"
# in a plain iteration, mis-attributing EleutherAI models to openai.
_DEVELOPER_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    sorted(_DEVELOPER_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)
)

# Patterns whose developer attribution is frequently wrong for community
# fine-tunes that borrow a well-known model name as a prefix.  When
# _infer_developer resolves via one of these, a stderr warning is emitted
# so the caller knows to verify before submission.  Examples:
#   'alpaca'  — Stanford's original, but ~100 community fine-tunes share the prefix
#   'orca'    — Microsoft's Orca, but many non-MSFT 'orca-*' fine-tunes exist
#   'hermes'  — NousResearch, but widely cloned with the same name prefix
#   'vicuna'  — LMSYS, but many third-party vicuna-based derivatives exist
_AMBIGUOUS_DEVELOPER_PATTERNS: frozenset[str] = frozenset({
    "alpaca",
    "orca",
    "hermes",
    "vicuna",
})


# ---------------------------------------------------------------------------
# S: PDF downloader — fetch only
# ---------------------------------------------------------------------------


class PDFDownloader:
    """download a PDF from arXiv or a local path."""

    def fetch(self, source: str) -> Path:
        """return a local Path to the PDF for *source*.

        *source* is either a local path or an arXiv ID (e.g. '2407.21783').
        """
        local = Path(source)
        if local.exists():
            return local

        # treat source as arXiv ID
        arxiv_id = source.strip()
        dest = _PDF_DOWNLOAD_DIR / f"{arxiv_id}.pdf"
        if dest.exists():
            return dest

        url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        print(f"  downloading {url} ...")
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "EEE-pipeline/1.0"})
        resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        print(f"  saved to {dest}")
        return dest


# ---------------------------------------------------------------------------
# S: table extractor — Docling table detection only
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Docling imports and shared parser
# ---------------------------------------------------------------------------
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
import pandas as pd


class DoclingParser:
    """Shared Docling PDF parser with per-path result caching.

    Parse each PDF once; all extractors reuse the cached result to avoid
    redundant deep-learning inference across TableExtractor, ProseExtractor,
    LLMFallbackExtractor, and _detect_eval_library.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    def parse(self, pdf_path: Path) -> Any:
        """Return the cached Docling ConversionResult for *pdf_path*."""
        key = str(pdf_path.resolve())
        if key not in self._cache:
            self._cache[key] = self._converter.convert(key)
        return self._cache[key]

    def get_full_text(self, pdf_path: Path) -> str:
        """Return the full document as markdown text."""
        return self.parse(pdf_path).document.export_to_markdown()


# Module-level shared parser instance — constructed lazily on first use.
_docling_parser = DoclingParser()


# ---------------------------------------------------------------------------
# S: table extractor — Docling table detection only
# ---------------------------------------------------------------------------


class TableExtractor:
    """extract raw tables from a PDF using Docling."""

    def extract(self, pdf_path: Path) -> list[tuple[int, list[list[str]], str]]:
        """Return list of (page_number, table_rows, context_text) from *pdf_path*.

        *context_text* is the markdown text preceding the table (caption/section header).
        """
        result = _docling_parser.parse(pdf_path)
        tables_out: list[tuple[int, list[list[str]], str]] = []
        doc = result.document
        markdown = doc.export_to_markdown()
        for table in doc.tables:
            try:
                df = table.export_to_dataframe(doc)
            except Exception:
                d = table.export_to_dict()
                rows = d.get("rows") or d.get("data") or []
                df = pd.DataFrame(rows)
            # Prepend column headers as the first row so benchmark keywords
            # (e.g. "MMLU", "GSM8K") are visible to ResultsTableParser.
            header = [str(c) for c in df.columns.tolist()]
            data_rows = df.astype(str).fillna("").values.tolist()
            table_rows = [header] + data_rows
            page_num = getattr(table, "page_num", 1)
            context_text = getattr(table, "caption", None)
            if not context_text:
                table_md = df.to_string(index=False)
                idx = markdown.find(table_md[:40]) if table_md else -1
                if idx > 0:
                    context_text = markdown[max(0, idx - 300):idx].strip()
                else:
                    context_text = ""
            tables_out.append((page_num, table_rows, context_text or ""))
        return tables_out


# ---------------------------------------------------------------------------
# S: prose extractor — regex-based extraction from running text and captions
# ---------------------------------------------------------------------------


class ProseExtractor:
    """Extract benchmark scores from running prose and figure captions.

    Handles two gaps that ``TableExtractor`` cannot cover without an LLM:

    * Inline sentences: ``"LLaMA-3 achieves 87.3 % on MMLU"``
    * Figure captions: ``"GPT-4 obtains 72.1 on HumanEval (Figure 3)"``

    Uses compiled regex patterns built from ``_BENCHMARK_KEYWORDS_STRONG``
    so there are zero new runtime dependencies.  Always runs before the LLM
    fallback so the LLM is reserved for genuinely unstructured cases.
    Results carry ``_extraction_confidence='prose'`` for clear provenance.

    Note: scores encoded *only* in bar-chart or radar-plot images are not
    accessible from text alone; those require ``--llm-fallback`` with a
    multimodal model.
    """

    # Pattern A: "MODEL achieves/scores/obtains/reaches/gets NUMBER on/for BENCHMARK"
    # MODEL captured as 1-4 hyphen-or-word tokens to avoid full-sentence matches.
    _PATTERN_MODEL_FIRST: re.Pattern[str] = re.compile(
        r"((?:[\w][\w\-\.]*(?:\s[\w\-\.]+){0,3}))\s+"
        r"(?:achieves?|scores?|obtains?|reaches?|reports?|gets?)\s+"
        r"(-?\d+(?:\.\d+)?)\s*%?\s+"
        r"(?:on|for)\s+("
        + _BENCH_ALT_RE
        + r")\b",
        re.IGNORECASE,
    )

    # Pattern B: "on/for BENCHMARK, MODEL achieves/scores NUMBER"
    _PATTERN_BENCH_FIRST: re.Pattern[str] = re.compile(
        r"(?:on|for)\s+("
        + _BENCH_ALT_RE
        + r")\b[^,\n]{0,20},?\s+"
        r"((?:[\w][\w\-\.]*(?:\s[\w\-\.]+){0,3}))\s+"
        r"(?:achieves?|scores?|obtains?|reaches?|gets?)\s+"
        r"(-?\d+(?:\.\d+)?)\s*%?",
        re.IGNORECASE,
    )

    # Stop-words that are never valid model names — filters false positives
    # in sentences like "the method achieves 87.3 on MMLU".
    _STOP_WORDS: frozenset[str] = frozenset({
        "the", "its", "our", "this", "that", "which", "model",
        "system", "method", "approach", "baseline", "both", "all",
        "each", "when", "then", "with", "where",
    })

    def extract(self, pdf_path: Path, arxiv_id: str) -> list[dict[str, Any]]:
        """Return ``{model, benchmark, score, _extraction_confidence}`` dicts.

        Reads the full document text via Docling's markdown export.
        """
        results: list[dict] = []
        seen: set[tuple[str, str]] = set()
        try:
            full_text = _docling_parser.get_full_text(pdf_path)
            if full_text.strip():
                self._scan_page(full_text, results, seen)
        except Exception as exc:  # noqa: BLE001
            print(f"  [prose] could not open PDF: {exc}", file=sys.stderr)
            return []
        if results:
            print(
                f"  [prose] {len(results)} data point(s) extracted from text "
                f"({len({r['model'] for r in results})} models, "
                f"{len({r['benchmark'] for r in results})} benchmarks)"
            )
        return results

    def _scan_page(
        self,
        text: str,
        results: list[dict],
        seen: set[tuple[str, str]],
    ) -> None:
        """Apply all patterns to *text*; append novel hits to *results*."""
        # Pattern A: model comes first
        for m in self._PATTERN_MODEL_FIRST.finditer(text):
            self._record(
                m.group(1), m.group(3), m.group(2), "prose", results, seen
            )
        # Pattern B: benchmark comes first
        for m in self._PATTERN_BENCH_FIRST.finditer(text):
            # group order is (benchmark, model, score)
            self._record(
                m.group(2), m.group(1), m.group(3), "prose", results, seen
            )

    def _record(
        self,
        model: str,
        benchmark: str,
        score_raw: str,
        confidence: str,
        results: list[dict],
        seen: set[tuple[str, str]],
    ) -> None:
        """Validate and append a (model, benchmark, score) hit."""
        model = model.strip()
        benchmark = benchmark.strip()
        # Reject implausibly short strings and pure-numeric captures
        if len(model) < 3 or model.isdigit():
            return
        if model.lower() in self._STOP_WORDS:
            return
        try:
            score = float(score_raw.rstrip("%"))
        except ValueError:
            return
        key = (model.lower(), benchmark.lower())
        if key in seen:
            return
        seen.add(key)
        results.append({
            "model": model,
            "benchmark": benchmark,
            "score": score,
            "_extraction_confidence": confidence,
        })


# ---------------------------------------------------------------------------
# S: results table parser — heuristic identification and parsing only
# ---------------------------------------------------------------------------


class ResultsTableParser:
    """identify and parse results tables from raw extracted table data."""

    def is_results_table(self, table: list[list[str]], context: str = "") -> bool:
        """Return True if *table* looks like a model evaluation results table.

        Three checks must all pass:

        1. **Benchmark keyword** — a known benchmark name appears somewhere in
           the table *or in the context text above it* (checked header-first,
           then full table, then context).
        2. **Numeric density** — at least 25 % of non-empty body cells, or at
           least 5 cells, contain a parseable score (int, float, or %).
        3. **Minimum size** — at least header row + 2 data rows, ensuring we
           don't flag single-row summary lines.

        Parameters
        ----------
        table:
            Normalised table rows (all cells are str, None already replaced).
        context:
            Plain text extracted from the region above the table on the same
            page (e.g. caption, section header).  When the table cells
            themselves carry no benchmark keyword, this text is the last
            resort before the table is rejected — it catches cases like a
            multi-part results block where the benchmark name only appears in
            the preceding paragraph.
        """
        # check 3: minimum size
        if len(table) < 3:
            return False

        # check 1: benchmark keyword presence — two-tier strategy:
        #   • one *strong* match (specific benchmark name) is sufficient;
        #   • generic metric terms (_BENCHMARK_KEYWORDS_WEAK) require ≥2
        #     distinct matches to reduce false positives from non-eval tables
        #     that happen to mention "accuracy" or "f1" (e.g. a confusion
        #     matrix or a hyperparameter ablation table).
        # Scan order: header row → second row → full table → context text.
        def _has_eval_signal(text: str) -> bool:
            if any(kw in text for kw in _BENCHMARK_KEYWORDS_STRONG):
                return True
            weak_hits = sum(1 for kw in _BENCHMARK_KEYWORDS_WEAK if kw in text)
            return weak_hits >= 2

        header_text = " ".join(cell.lower() for cell in table[0] if cell)
        has_benchmark = _has_eval_signal(header_text)
        if not has_benchmark and len(table) > 1:
            second_text = " ".join(c.lower() for c in table[1] if c)
            has_benchmark = _has_eval_signal(second_text)
        # slow path: scan the entire table (handles multi-row headers and
        # tables where benchmark names appear in the first column)
        if not has_benchmark:
            all_text = " ".join(
                cell.lower() for row in table for cell in row if cell
            )
            has_benchmark = _has_eval_signal(all_text)
        # last resort: check the caption / section header above the table.
        # Accept only a strong keyword here to avoid false positives from
        # sections like "Experimental Setup" that mention metric names in
        # passing (weak matches require 2 hits, which is strict enough).
        if not has_benchmark and context:
            has_benchmark = _has_eval_signal(context.lower())
        if not has_benchmark:
            return False

        # check 2: numeric density — also match percentages and negatives
        _numeric_re = re.compile(r"^-?\d+(\.\d+)?%?$")
        numeric_cells = 0
        non_empty_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if stripped:
                    non_empty_cells += 1
                    if _numeric_re.match(stripped):
                        numeric_cells += 1

        if non_empty_cells == 0:
            return False
        # require either an absolute floor of 5 numeric cells (small tables)
        # or 25 % density (larger tables)
        if not (numeric_cells >= 5 or (numeric_cells / non_empty_cells) >= 0.25):
            return False

        # check 4: reject ablation tables — first column dominated by ablation
        # marker language rather than distinct model names from different orgs.
        # Ablation tables pass checks 1–3 but generate EEE records where every
        # entry describes the same underlying model under ablation conditions,
        # making the schema data ambiguous to query.
        if self._is_ablation_table(table):
            return False

        return True

    @staticmethod
    def _is_ablation_table(table: list[list[str]]) -> bool:
        """Return True when the first column looks like an ablation study.

        Ablation tables compare variants of a single model (e.g. "w/o layer
        norm", "base + FT", "+ DPO") rather than multiple independent models.
        Heuristic: if ≥ 40 % of non-empty first-column cells contain an
        ablation marker token, classify as ablation.  The 40 % threshold
        tolerates one legitimate baseline row while catching tables that are
        predominantly ablation rows.
        """
        _ABLATION_MARKERS: frozenset[str] = frozenset({
            "w/o", "without", "ablat", "no ", "+ ", " + ",
            "only", "ours w", "base model", "full model",
            "remove", "drop ", "variant",
        })
        first_col = [
            row[0].lower().strip()
            for row in table[1:]
            if row and row[0].strip()
        ]
        if not first_col:
            return False
        ablation_count = sum(
            1 for v in first_col
            if any(marker in v for marker in _ABLATION_MARKERS)
        )
        return ablation_count / len(first_col) >= 0.40

    def extraction_confidence(self, table: list[list[str]], context: str = "") -> str:
        """Return a confidence tier for how reliably this table was extracted.

        Calibration
        -----------
        'high'   — ≥1 strong benchmark keyword in the table body AND numeric
                   density ≥ 50 %.  The table is unambiguously a results table.
        'medium' — ≥1 strong keyword with density 25–50 %, OR the table only
                   matched via the caption / context rather than its own cells.
        'low'    — Accepted on weak-keyword evidence (≥2 weak matches) or on
                   the numeric-count floor only (≥5 cells but < 25 % density).

        A 'low' result is not an error — the table still passed
        `is_results_table` and is included — but benchmark→score assignments
        should be treated with more scepticism and manually reviewed before
        submission.
        """
        all_text = " ".join(cell.lower() for row in table for cell in row if cell)
        context_lower = context.lower()

        strong_in_body = sum(
            1 for kw in _BENCHMARK_KEYWORDS_STRONG if kw in all_text
        )
        # context-only match: strong keyword found only in caption/header text,
        # not in the table cells themselves.
        strong_in_context_only = (
            strong_in_body == 0
            and any(kw in context_lower for kw in _BENCHMARK_KEYWORDS_STRONG)
        )

        # recompute numeric density (same logic as is_results_table)
        _numeric_re = re.compile(r"^-?\d+(\.\d+)?%?$")
        numeric_cells = 0
        non_empty_cells = 0
        for row in table[1:]:
            for cell in row:
                stripped = cell.strip()
                if stripped:
                    non_empty_cells += 1
                    if _numeric_re.match(stripped):
                        numeric_cells += 1
        density = (numeric_cells / non_empty_cells) if non_empty_cells > 0 else 0.0

        if strong_in_body >= 1 and density >= 0.50:
            return "high"
        if (strong_in_body >= 1 and density >= 0.25) or strong_in_context_only:
            return "medium"
        return "low"

    def parse(
        self, table: list[list[str]]
    ) -> list[dict[str, Any]]:
        """parse *table* into a list of {model, benchmark, score} dicts.

        handles both orientations (models-as-rows and models-as-columns).
        Also attempts to collapse multi-row headers into a single header row
        when the first row is mostly empty (common in LaTeX multicolumn tables).
        """
        results: list[dict[str, Any]] = []

        if not table or not table[0]:
            return results

        table = self._collapse_multirow_header(table)

        header = [_clean_cell(c) for c in table[0]]

        # determine orientation:
        # models-as-rows: first column contains model names, header has benchmarks
        # models-as-columns: first row is benchmarks, first column is metric names

        benchmarks_in_header = [
            h for h in header if any(kw in h.lower() for kw in _BENCHMARK_KEYWORDS)
        ]

        if benchmarks_in_header:
            results = self._parse_models_as_rows(header, table[1:])
        else:
            results = self._parse_models_as_columns(table)

        return results

    @staticmethod
    def _collapse_multirow_header(table: list[list[str]]) -> list[list[str]]:
        """merge row 0 and row 1 into a single header when row 0 is sparse.

        Many LaTeX-generated PDF tables have a top row with group labels and
        an immediately following subheader row with benchmark names.  When the
        first row has more than half its cells empty, we join the two rows
        cell-by-cell so downstream parsing sees a single enriched header.
        """
        if len(table) < 2:
            return table
        row0 = table[0]
        row1 = table[1]
        # count non-empty cells in row 0 (excluding the first/index column)
        non_empty = sum(1 for c in row0[1:] if c and c.strip())
        total = max(len(row0) - 1, 1)
        if non_empty / total > 0.5:
            # row 0 is already well-populated — no merge needed
            return table
        # merge: prefer the non-empty cell from either row; join when both present
        merged: list[str] = []
        for i in range(max(len(row0), len(row1))):
            c0 = row0[i].strip() if i < len(row0) else ""
            c1 = row1[i].strip() if i < len(row1) else ""
            if c0 and c1:
                merged.append(f"{c0} {c1}")
            else:
                merged.append(c0 or c1)
        return [merged] + table[2:]

    def _parse_models_as_rows(
        self, header: list[str], body: list[list[str]]
    ) -> list[dict]:
        """models are rows, benchmarks are columns."""
        results: list[dict] = []
        # first column is model name; rest are benchmark columns
        bench_names = [_clean_cell(h) for h in header[1:]]
        expected_cols = len(bench_names)
        total_data_rows = 0
        misaligned_rows = 0
        for row in body:
            if not row or not row[0]:
                continue
            model_name = _clean_cell(row[0])
            if not model_name or _is_separator(model_name):
                continue
            # A row should have 1 model-name cell + expected_cols score cells.
            # Fewer columns means pdfplumber collapsed a \multicolumn span,
            # silently shifting all subsequent scores one or more positions left.
            actual_data_cols = len(row) - 1
            total_data_rows += 1
            if actual_data_cols != expected_cols:
                misaligned_rows += 1
            for i, bench in enumerate(bench_names):
                if not bench:
                    continue
                cell_idx = i + 1
                if cell_idx >= len(row):
                    continue
                score = _parse_numeric(row[cell_idx])
                if score is None:
                    continue
                results.append(
                    {"model": model_name, "benchmark": bench, "score": score}
                )
        # Warn when misalignment is widespread — this means \multicolumn spanning
        # has corrupted the column→benchmark mapping for a significant fraction of
        # rows.  The threshold of 15 % is intentionally loose to avoid noise from
        # rare footnote rows that add an extra cell.
        if total_data_rows > 0 and misaligned_rows / total_data_rows > 0.15:
            print(
                f"  warning: {misaligned_rows}/{total_data_rows} data rows have a "
                f"column-count mismatch (expected {expected_cols} benchmark cols). "
                "Likely cause: \\multicolumn spanning — some scores may be "
                "assigned to the wrong benchmark. Review this table manually."
            )
        return results

    def _parse_models_as_columns(
        self, table: list[list[str]]
    ) -> list[dict]:
        """models are columns, benchmarks are rows."""
        results: list[dict] = []
        if len(table[0]) < 2:
            return results

        model_names = [_clean_cell(c) for c in table[0][1:]]

        for row in table[1:]:
            if not row or not row[0]:
                continue
            bench = _clean_cell(row[0])
            if not bench or _is_separator(bench):
                continue
            for i, model in enumerate(model_names):
                if not model:
                    continue
                cell_idx = i + 1
                if cell_idx >= len(row):
                    continue
                score = _parse_numeric(row[cell_idx])
                if score is None:
                    continue
                results.append(
                    {"model": model, "benchmark": bench, "score": score}
                )
        return results


# ---------------------------------------------------------------------------
# S: paper converter — schema conversion only
# ---------------------------------------------------------------------------


class PaperConverter:
    """convert extracted paper results to EEE schema dicts."""

    def convert(
        self,
        extracted: list[dict[str, Any]],
        arxiv_id: str,
        retrieved_timestamp: str,
        paper_title: str = "",
        eval_library: EvalLibrary | None = None,
    ) -> list[dict]:
        """group *extracted* by model and produce one EEE record per model."""
        # group results by model name
        by_model: dict[str, list[dict]] = {}
        for item in extracted:
            model = item["model"]
            by_model.setdefault(model, []).append(item)

        eval_name = _make_eval_name(arxiv_id)
        source_metadata = _make_source_metadata(arxiv_id, paper_title)
        lib = eval_library if eval_library is not None else EvalLibrary(name="unknown", version="unknown")

        records: list[dict] = []
        for model_name, items in by_model.items():
            try:
                record = self._convert_model(
                    model_name,
                    items,
                    eval_name,
                    arxiv_id,
                    retrieved_timestamp,
                    source_metadata,
                    lib,
                )
                records.append(record)
            except Exception as exc:
                print(f"  skipping model {model_name!r}: {exc}")

        return records

    def _convert_model(
        self,
        model_name: str,
        items: list[dict],
        eval_name: str,
        arxiv_id: str,
        retrieved_timestamp: str,
        source_metadata: SourceMetadata,
        eval_library: EvalLibrary,
    ) -> dict:
        """build one EvaluationLog dict for *model_name*."""
        developer = _infer_developer(model_name)
        if "/" in model_name:
            model_id = model_name
            developer = model_id.split("/")[0]
        else:
            model_id = f"{developer}/{model_name}"

        safe_id = model_id.replace("/", "_")
        evaluation_id = f"{eval_name}/{safe_id}/{retrieved_timestamp}"

        source_data = SourceDataUrl(
            dataset_name="arXiv paper",
            source_type="url",
            url=[f"https://arxiv.org/abs/{arxiv_id}"],
        )

        # Deduplicate by benchmark name: if the same benchmark appears in
        # multiple tables (e.g. main results + ablation), keep the first
        # occurrence — later tables often report a subset of the original.
        # build generation_config once; applied to every EvaluationResult below
        gen_config = GenerationConfig(
            additional_details={
                "note": "generation config not reported in source paper",
                "source": f"arXiv:{arxiv_id}",
                # Lowest confidence tier across all tables that contributed
                # scores for this model.  'low' means at least one table
                # was accepted on weak evidence; consumers should verify.
                "extraction_confidence": _aggregate_extraction_confidence(items),
            }
        )

        seen_benchmarks: dict[str, EvaluationResult] = {}
        for item in items:
            bench = item["benchmark"].strip()
            if bench in seen_benchmarks:
                continue  # skip duplicate; first table entry wins

            raw_score = float(item["score"])
            bench_lower = bench.lower()

            lower_is_better = any(p in bench_lower for p in _LOWER_IS_BETTER_PATTERNS)
            is_unbounded    = any(p in bench_lower for p in _UNBOUNDED_SCORE_PATTERNS)
            is_scale_100    = any(p in bench_lower for p in _SCALE_100_PATTERNS)

            if is_unbounded:
                # keep raw value; no meaningful upper bound
                score = round(raw_score, 4)
                min_score: float = 0.0
                max_score: float | None = None
            elif is_scale_100:
                # BLEU / ROUGE: keep on 0–100 scale
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 100.0
            elif raw_score > 1.0:
                # assume percentage — normalise to 0–1
                score = round(raw_score / 100.0, 4)
                min_score = 0.0
                max_score = 1.0
            else:
                score = round(raw_score, 4)
                min_score = 0.0
                max_score = 1.0

            seen_benchmarks[bench] = EvaluationResult(
                evaluation_name=bench,
                source_data=source_data,
                metric_config=MetricConfig(
                    evaluation_description=f"score on {bench} as reported in arXiv:{arxiv_id}",
                    lower_is_better=lower_is_better,
                    score_type=ScoreType.continuous,
                    min_score=min_score,
                    max_score=max_score,
                ),
                score_details=ScoreDetails(score=score),
                generation_config=gen_config,
            )

        eval_results: list[EvaluationResult] = list(seen_benchmarks.values())

        log = EvaluationLog(
            schema_version=_SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=ModelInfo(
                name=model_name,
                id=model_id,
                developer=developer,
                # inference_platform left None — unknown from paper tables
            ),
            evaluation_results=eval_results,
        )

        return log.model_dump(mode='json', exclude_none=True)


# ---------------------------------------------------------------------------
# S: paper writer — file I/O only
# ---------------------------------------------------------------------------


class PaperWriter:
    """write EEE schema dicts to disk under data/{eval_name}/."""

    def write(self, records: list[dict], eval_name: str) -> list[Path]:
        """write *records* to data/{eval_name}/... and return written paths."""
        paths: list[Path] = []
        for rec in records:
            try:
                model_id: str = rec["model_info"]["id"]
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                dev_safe = re.sub(r'[<>:"/\\|?*]', "_", developer)
                mod_safe = re.sub(r'[<>:"/\\|?*]', "_", model_name)

                out_dir = Path(f"data/{eval_name}") / dev_safe / mod_safe
                out_dir.mkdir(parents=True, exist_ok=True)

                path = out_dir / f"{uuid.uuid4()}.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)

                paths.append(path)
            except Exception as exc:
                mid = rec.get("model_info", {}).get("id", "?")
                print(f"  warning: could not write record for {mid}: {exc}")

        return paths


# ---------------------------------------------------------------------------
# S: LLM fallback extractor — prose and figure-caption extraction only
# ---------------------------------------------------------------------------


class LLMFallbackExtractor:
    """Extract benchmark results from paper text when table parsing yields nothing.

    Handles two gaps that table parsing cannot cover:
      - prose-embedded results (``"achieves 87.3 % on MMLU"``)
      - figure captions describing bar-chart scores

    Uses the OpenAI chat API with ``response_format=json_object`` to get
    structured output.  All responses are cached to disk so re-runs are
    fully deterministic and do not re-incur API costs.

    Requires ``openai`` package (``pip install openai``) and an
    ``OPENAI_API_KEY`` environment variable unless *api_key* is passed.
    """

    _SYSTEM = (
        "You are a precise data-extraction assistant for NLP research papers. "
        "Extract every LLM benchmark evaluation result from the provided text. "
        "Only include results with a clearly stated numeric score. "
        "Do not infer, estimate, or invent values. "
        "Output only valid JSON, nothing else."
    )

    _USER_TMPL = (
        "Paper arXiv:{arxiv_id} — chunk {chunk_idx}.\n\n"
        "Extract all (model_name, benchmark_name, score) triples. "
        "Include scores from tables, sentences, and figure captions. "
        "Do not include results without a numeric value.\n\n"
        "Respond with exactly this JSON schema "
        "(no extra keys, no markdown fences):\n"
        '{{ "results": [ {{"model": "<str>", "benchmark": "<str>", '
        '"score": <float>}} ] }}\n\n'
        "TEXT:\n{text}"
    )

    # Maximum characters per LLM request — well within 128 K-token context
    # while keeping per-call cost low on gpt-4o-mini.
    _MAX_CHARS_PER_CHUNK: int = 6_000

    _CACHE_DIR: Path = Path("scripts/scrapers/raw/llm_cache")

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._api_key = api_key   # falls back to OPENAI_API_KEY env var
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client: Any = None  # lazy-initialised on first use

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "openai package required for LLM fallback: pip install openai"
                ) from exc
            init_kwargs: dict[str, Any] = {}
            if self._api_key:
                init_kwargs["api_key"] = self._api_key
            self._client = openai.OpenAI(**init_kwargs)
        return self._client

    def _cache_path(self, arxiv_id: str, chunk_idx: int) -> Path:
        return self._CACHE_DIR / f"{arxiv_id}_chunk{chunk_idx:04d}.json"

    def _call_llm(self, arxiv_id: str, text: str, chunk_idx: int) -> list[dict]:
        """Call the LLM for one text chunk, returning from disk cache when available."""
        cache_file = self._cache_path(arxiv_id, chunk_idx)
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return cached.get("results", [])
            except Exception:
                pass  # corrupted cache — fall through to re-call

        client = self._get_client()
        prompt = self._USER_TMPL.format(
            arxiv_id=arxiv_id, chunk_idx=chunk_idx, text=text
        )
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw_json = response.choices[0].message.content or "{}"

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            parsed = {}

        # Validate structure and coerce types before caching.
        clean: list[dict] = []
        for item in parsed.get("results", []):
            try:
                score = float(item["score"])
                if score != score:  # NaN guard
                    continue
                clean.append({
                    "model": str(item["model"]).strip(),
                    "benchmark": str(item["benchmark"]).strip(),
                    "score": score,
                })
            except (KeyError, TypeError, ValueError):
                continue

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {"arxiv_id": arxiv_id, "chunk": chunk_idx, "results": clean},
                indent=2,
            ),
            encoding="utf-8",
        )
        return clean

    def _chunk_text(self, text: str) -> list[str]:
        """Split *text* into chunks at paragraph boundaries."""
        chunks: list[str] = []
        remaining = text
        while remaining:
            chunk = remaining[: self._MAX_CHARS_PER_CHUNK]
            # prefer splitting at a paragraph boundary in the second half
            boundary = chunk.rfind("\n\n", self._MAX_CHARS_PER_CHUNK // 2)
            if boundary > 0:
                chunk = chunk[:boundary]
            chunks.append(chunk)
            remaining = remaining[len(chunk):]
        return chunks

    def extract(self, pdf_path: Path, arxiv_id: str) -> list[dict[str, Any]]:
        """Extract benchmark results from prose and captions in *pdf_path*.

        Returns a list in the same format as ``ResultsTableParser.parse``
        (``{model, benchmark, score}``) with ``_extraction_confidence='llm'``
        to clearly distinguish these items from table-parsed results.

        Deduplication is applied: when the same (model, benchmark) pair appears
        in multiple chunks, the first occurrence wins, consistent with the
        table-parsing deduplication strategy.
        """
        pages: list[str] = []
        try:
            # Use the shared Docling parser (cached). Docling handles structured
            # PDFs with high fidelity; scanned PDFs can be enabled by setting
            # do_ocr=True in DoclingParser if needed.
            full_text = _docling_parser.get_full_text(pdf_path)
            if full_text.strip():
                pages = [full_text]
        except Exception as exc:
            print(f"  [llm-fallback] could not open PDF: {exc}", file=sys.stderr)
            return []

        full_text = "\n\n".join(pages)
        chunks = self._chunk_text(full_text)
        print(
            f"  [llm-fallback] {len(chunks)} chunk(s) → {self._model} "
            f"(cache: {self._CACHE_DIR})"
        )

        all_results: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for idx, chunk in enumerate(chunks):
            try:
                items = self._call_llm(arxiv_id, chunk, idx)
            except Exception as exc:
                print(f"  [llm-fallback] chunk {idx} failed: {exc}", file=sys.stderr)
                continue
            for item in items:
                key = (item["model"].lower(), item["benchmark"].lower())
                if key not in seen:
                    seen.add(key)
                    item["_extraction_confidence"] = "llm"
                    all_results.append(item)

        print(
            f"  [llm-fallback] extracted {len(all_results)} data point(s) "
            f"({len({it['model'] for it in all_results})} models, "
            f"{len({it['benchmark'] for it in all_results})} benchmarks)"
        )
        return all_results


# ---------------------------------------------------------------------------
# Coverage reporting — per-paper and batch stats
# ---------------------------------------------------------------------------


@dataclass
class CoverageStats:
    """Extraction quality metrics for a single paper.

    Accumulated inside ``PaperExtractionPipeline.run`` and returned to the
    caller so that ``main`` can aggregate stats across a batch run and write
    a machine-readable coverage report to disk.
    """

    arxiv_id: str
    tables_scanned: int = 0
    results_tables_accepted: int = 0
    density_rejected: int = 0
    table_data_points: int = 0
    prose_data_points: int = 0
    llm_data_points: int = 0
    unique_models: int = 0
    unique_benchmarks: int = 0
    files_written: int = 0
    source: str = "none"   # "table" | "llm" | "both" | "none"
    validation_passed: int = 0
    validation_failed: int = 0

    @property
    def total_data_points(self) -> int:
        return self.table_data_points + self.prose_data_points + self.llm_data_points


# ---------------------------------------------------------------------------
# D: pipeline orchestrator — depends on abstractions only
# ---------------------------------------------------------------------------


class PaperExtractionPipeline:
    """orchestrate download → extract → parse → convert → write for a paper."""

    def __init__(
        self,
        downloader: PDFDownloader,
        table_extractor: TableExtractor,
        table_parser: ResultsTableParser,
        converter: PaperConverter,
        writer: PaperWriter,
        llm_fallback: LLMFallbackExtractor | None = None,
        prose_extractor: ProseExtractor | None = None,
    ) -> None:
        self._downloader = downloader
        self._extractor = table_extractor
        self._parser = table_parser
        self._converter = converter
        self._writer = writer
        self._llm_fallback = llm_fallback
        self._prose_extractor = prose_extractor

    def run(self, source: str, *, force_llm: bool = False) -> CoverageStats:
        """process *source* (arXiv ID or PDF path) end-to-end.

        Parameters
        ----------
        source:
            arXiv paper ID (e.g. ``2407.21783``) or path to a local PDF.
        force_llm:
            When True, run the LLM fallback extractor unconditionally alongside
            table parsing.  When False (default), the LLM fallback is activated
            only when table parsing yields zero data points and an
            ``LLMFallbackExtractor`` was supplied at construction time.

        Returns
        -------
        CoverageStats
            Per-paper extraction quality metrics suitable for batch aggregation.
        """
        stats = CoverageStats(arxiv_id=source)
        retrieved_timestamp = str(time.time())
        arxiv_id = _normalise_arxiv_id(source)
        stats.arxiv_id = arxiv_id
        eval_name = _make_eval_name(arxiv_id)

        print(f"\n[{arxiv_id}] fetching paper title from arXiv...")
        paper_title = _fetch_arxiv_title(arxiv_id)
        if paper_title:
            print(f"  title: {paper_title}")
        else:
            print("  title not found — using arXiv ID as fallback")

        print(f"[{arxiv_id}] downloading PDF...")
        pdf_path = self._downloader.fetch(source)

        print(f"[{arxiv_id}] detecting eval framework...")
        eval_library = _detect_eval_library(pdf_path)

        print(f"[{arxiv_id}] extracting tables ({pdf_path.name})...")
        raw_tables = self._extractor.extract(pdf_path)
        stats.tables_scanned = len(raw_tables)
        print(f"  found {len(raw_tables)} tables across all pages")

        table_extracted: list[dict] = []
        _has_keyword_re = re.compile(
            "|".join(re.escape(kw) for kw in _BENCHMARK_KEYWORDS), re.IGNORECASE
        )

        for page_num, table, context_text in raw_tables:
            table_text = " ".join(cell for row in table for cell in row if cell)
            has_kw = bool(_has_keyword_re.search(table_text)) or bool(
                _has_keyword_re.search(context_text)
            )

            if self._parser.is_results_table(table, context_text):
                confidence = self._parser.extraction_confidence(table, context_text)
                items = self._parser.parse(table)
                for item in items:
                    item["_extraction_confidence"] = confidence
                table_extracted.extend(items)
                stats.results_tables_accepted += 1
                print(
                    f"  page {page_num}: results table accepted "
                    f"({len(table)} rows, {len(items)} data points, "
                    f"confidence={confidence})"
                )
            elif has_kw:
                stats.density_rejected += 1
                print(
                    f"  page {page_num}: table has benchmark keywords "
                    f"but failed density check ({len(table)} rows) — skipped"
                )

        stats.table_data_points = len(table_extracted)

        # Prose extraction: always runs; finds benchmark scores in running text
        # and figure captions via regex — no external dependencies required.
        prose_new: list[dict] = []
        if self._prose_extractor is not None:
            print(f"[{arxiv_id}] scanning prose and figure captions...")
            prose_raw = self._prose_extractor.extract(pdf_path, arxiv_id)
            table_key_set: set[tuple[str, str]] = {
                (it["model"].lower(), it["benchmark"].lower())
                for it in table_extracted
            }
            prose_new = [
                it for it in prose_raw
                if (it["model"].lower(), it["benchmark"].lower())
                not in table_key_set
            ]
            stats.prose_data_points = len(prose_new)

        # LLM fallback: run when neither table nor prose found results, OR forced.
        # This keeps LLM calls to genuinely hard cases (scanned PDFs, images).
        combined_so_far = table_extracted + prose_new
        llm_extracted: list[dict] = []
        if self._llm_fallback is not None and (force_llm or not combined_so_far):
            trigger = "forced" if force_llm else "no table/prose results"
            print(f"[{arxiv_id}] running LLM fallback extractor ({trigger})...")
            llm_extracted = self._llm_fallback.extract(pdf_path, arxiv_id)
            stats.llm_data_points = len(llm_extracted)

        # Merge: table > prose > LLM; each tier only fills gaps from the one above.
        combined_keys: set[tuple[str, str]] = {
            (it["model"].lower(), it["benchmark"].lower()) for it in combined_so_far
        }
        llm_new = [
            it for it in llm_extracted
            if (it["model"].lower(), it["benchmark"].lower()) not in combined_keys
        ]
        extracted = table_extracted + prose_new + llm_new

        # Determine source tier(s) for the coverage report.
        sources_used: list[str] = []
        if table_extracted:
            sources_used.append("table")
        if prose_new:
            sources_used.append("prose")
        if llm_new:
            sources_used.append("llm")
        stats.source = "+".join(sources_used) if sources_used else "none"

        stats.unique_models = len({item["model"] for item in extracted})
        stats.unique_benchmarks = len({item["benchmark"] for item in extracted})

        print(
            f"\n[{arxiv_id}] extraction summary:\n"
            f"  tables scanned:          {stats.tables_scanned}\n"
            f"  results tables accepted: {stats.results_tables_accepted}\n"
            f"  keyword-only rejects:    {stats.density_rejected}  "
            f"(potential false negatives — review manually)\n"
            f"  table data points:       {stats.table_data_points}\n"
            f"  prose data points (new): {stats.prose_data_points}\n"
            f"  llm data points (new):   {len(llm_new)}\n"
            f"  total data points:       {len(extracted)}\n"
            f"  unique models:           {stats.unique_models}\n"
            f"  unique benchmarks:       {stats.unique_benchmarks}\n"
            f"  source:                  {stats.source}"
        )

        if not extracted:
            print(f"[{arxiv_id}] no results found — skipping")
            return stats

        records = self._converter.convert(
            extracted, arxiv_id, retrieved_timestamp,
            paper_title=paper_title,
            eval_library=eval_library,
        )
        print(f"[{arxiv_id}] converted {len(records)} model records")

        unknown_devs = [
            r["model_info"]["id"]
            for r in records
            if r.get("model_info", {}).get("developer") == "unknown"
        ]
        if unknown_devs:
            print(
                f"  [developer-inference] {len(unknown_devs)} model(s) have "
                f"developer='unknown' \u2014 verify these before submission:\n"
                + "\n".join(f"    {mid}" for mid in unknown_devs)
            )

        paths = self._writer.write(records, eval_name)
        stats.files_written = len(paths)
        print(f"[{arxiv_id}] written {len(paths)} files to data/{eval_name}/")

        passed, failed = self._validate_written(paths, arxiv_id)
        stats.validation_passed = passed
        stats.validation_failed = failed

        return stats

    def _validate_written(self, paths: list[Path], arxiv_id: str) -> tuple[int, int]:
        """Validate each written JSON file against eval.schema.json.

        Returns
        -------
        tuple[int, int]
            ``(passed, failed)`` counts.  Failures are non-fatal so a batch
            run can continue; the caller should inspect warnings and fix
            source data before submitting a PR.
        """
        import json as _json
        from jsonschema.validators import validator_for

        schema_path = _ROOT / "eval.schema.json"
        if not schema_path.exists():
            print(f"  [validation] schema not found at {schema_path} — skipping")
            return 0, 0

        with schema_path.open() as fh:
            schema = _json.load(fh)
        validator_cls = validator_for(schema)
        validator = validator_cls(schema)

        passed = 0
        failed = 0
        for path in paths:
            try:
                with path.open() as fh:
                    instance = _json.load(fh)
                validator.validate(instance)
                passed += 1
            except Exception as exc:
                failed += 1
                print(f"  [validation] FAIL {path.name}: {exc}")

        status = "all passed" if failed == 0 else f"{failed} FAILED"
        print(
            f"[{arxiv_id}] schema validation: {passed} passed, {failed} failed — {status}"
        )
        return passed, failed


# ---------------------------------------------------------------------------
# pure helpers — no side effects
# ---------------------------------------------------------------------------


def _aggregate_extraction_confidence(items: list[dict]) -> str:
    """Return the lowest confidence tier across all extraction items for one model.

    Using the minimum (most pessimistic) tier ensures the field reflects the
    weakest evidence that contributed to this model's results, not an average
    that could mask a poorly-extracted table.  This matters for Track 1
    reviewers who need to know if *any* score came from a dubious table.
    """
    tier_rank = {"high": 2, "medium": 1, "low": 0}
    min_rank = 2
    for item in items:
        tier = item.get("_extraction_confidence", "low")
        min_rank = min(min_rank, tier_rank.get(tier, 0))
    return {2: "high", 1: "medium", 0: "low"}[min_rank]


def _normalise_arxiv_id(source: str) -> str:
    """extract the arXiv ID from *source* (path or ID string)."""
    p = Path(source)
    if p.exists() and p.suffix == ".pdf":
        return p.stem
    return source.strip()


def _make_eval_name(arxiv_id: str) -> str:
    """create a filesystem-safe eval_name from an arXiv ID."""
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", arxiv_id)
    return f"papers_{safe}"


def _make_source_metadata(arxiv_id: str, title: str = "") -> SourceMetadata:
    """build SourceMetadata for an academic paper."""
    return SourceMetadata(
        source_name=title if title else f"arXiv:{arxiv_id}",
        source_type="documentation",
        source_organization_name="arXiv",
        source_organization_url=f"https://arxiv.org/abs/{arxiv_id}",
        evaluator_relationship=EvaluatorRelationship.third_party,
    )


def _infer_developer(model_name: str, *, use_hf_api: bool = True) -> str:
    """infer the developer org from *model_name*.

    Resolution order:
    1. If the name is already in ``org/model`` HF format, return the org.
    2. Match against the local ``_DEVELOPER_MAP`` (startswith / substring).
    3. Try the HuggingFace Hub ``/api/models/{model_name}`` endpoint to read
       the ``author`` field.  Results are cached in ``_HF_AUTHOR_CACHE`` so
       repeated lookups within the same run incur only one HTTP request each.
    4. Delegate to the shared ``get_developer()`` helper as a final fallback.
    """
    if "/" in model_name:
        return model_name.split("/")[0]

    lower = model_name.lower()
    # Iterate the pre-sorted patterns tuple (longest key first) so that
    # "gpt-j" / "command-r" etc. are matched before the bare "gpt" / "command"
    # prefixes that would otherwise shadow them.
    for pattern, dev in _DEVELOPER_PATTERNS:
        if lower.startswith(pattern) or f"-{pattern}" in lower or f" {pattern}" in lower:
            if pattern in _AMBIGUOUS_DEVELOPER_PATTERNS:
                print(
                    f"  [developer-inference] '{model_name}' matched ambiguous "
                    f"pattern '{pattern}' \u2192 attributed to '{dev}'. "
                    f"Many community fine-tunes share this prefix; verify the "
                    f"attribution and use 'org/model' HF format if possible.",
                    file=sys.stderr,
                )
            return dev

    if use_hf_api:
        cached = _HF_AUTHOR_CACHE.get(model_name)
        if cached is not None:
            return cached
        try:
            resp = requests.get(
                f"https://huggingface.co/api/models/{model_name}",
                timeout=5,
                headers={"User-Agent": "EEE-pipeline/1.0"},
            )
            if resp.status_code == 200:
                author: str = resp.json().get("author", "") or ""
                if author:
                    _HF_AUTHOR_CACHE[model_name] = author
                    return author
        except Exception:  # noqa: BLE001  # network errors must not abort extraction
            pass

    result = get_developer(model_name)
    if result == "unknown":
        print(
            f"  [developer-inference] could not identify developer for {model_name!r} "
            f"\u2014 recorded as 'unknown'. Add a mapping to _DEVELOPER_MAP or use "
            f"HuggingFace 'org/model' format.",
            file=sys.stderr,
        )
    return result


def _clean_cell(value: str) -> str:
    """strip footnote/superscript markers and surrounding whitespace from *value*.

    Handles common decoration found in academic paper tables:
      - trailing markers: ``85.2†``, ``92*``, ``77.3‡``
      - leading markers:  ``*73.1``, ``†85.2``
      - inline superscript digits that pdfplumber sometimes preserves
        (e.g. ``LLaMA1 65B`` → ``LLaMA 65B`` is intentionally NOT stripped
         because superscripts on model names are less predictable; we only
         strip from pure numeric cells inside _parse_numeric).
    """
    return _FOOTNOTE_RE.sub("", value).strip()


def _parse_numeric(value: str) -> float | None:
    """parse *value* as a float; return None if not parseable.

    Handles formats commonly found in paper tables:
      - plain floats/ints:  ``85.2``, ``92``
      - percentage suffix:  ``85.2%``
      - footnote markers:   ``85.2†``, ``*92.0``, ``73.1‡``
      - parenthesised:      ``(85.2)``  (often used for std-dev or N/A rows)
      - dash / em-dash:     ``-``, ``—``  → None  (missing value sentinel)
      - bold LaTeX artefacts that pdfplumber sometimes leaves: ``\\textbf{85}
    """
    v = value.strip()
    # explicit missing-value sentinels
    if v in ("-", "—", "–", "n/a", "N/A", "na", "NA", ""):
        return None
    # strip footnote / superscript markers
    v = _FOOTNOTE_RE.sub("", v).strip()
    # unwrap parentheses  e.g. "(85.2)"
    if v.startswith("(") and v.endswith(")"):
        v = v[1:-1].strip()
    # strip trailing % and any remaining whitespace
    v = v.rstrip("%").strip()
    # strip common LaTeX bold/italic residue
    v = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", v)
    try:
        return float(v)
    except ValueError:
        return None


def _is_separator(text: str) -> bool:
    """return True if *text* looks like a table separator or section header.

    Matches:
      - rows of dashes / equals / underscores used as visual dividers
      - single-dot or ellipsis cells that pdfplumber extracts from ruling lines
    """
    stripped = text.strip()
    if not stripped:
        return True
    return bool(re.match(r"^[-=_.…\s]+$", stripped))


# ---------------------------------------------------------------------------
# helpers — arXiv metadata and eval-library detection
# ---------------------------------------------------------------------------


def _fetch_arxiv_title(arxiv_id: str) -> str:
    """fetch the paper title from the arXiv Atom API.

    Returns an empty string on any network or parse error so callers can
    safely fall back to the arXiv ID as a display name.
    """
    import xml.etree.ElementTree as ET

    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "EEE-pipeline/1.0"})
        if resp.status_code != 200:
            return ""
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return ""
        title_el = entry.find("atom:title", ns)
        if title_el is None or not title_el.text:
            return ""
        # arXiv Atom titles may contain internal newlines/extra spaces
        return " ".join(title_el.text.strip().split())
    except Exception:  # noqa: BLE001 — network errors must not abort extraction
        return ""


def _detect_eval_library(pdf_path: Path) -> EvalLibrary:
    """scan the first few pages of *pdf_path* for eval-framework signatures.

    Looks for known framework strings in the paper body (abstract + intro)
    and returns the matching EvalLibrary.  Falls back to name='unknown' if
    nothing is detected.
    """
    try:
        full_text = _docling_parser.get_full_text(pdf_path).lower()
    except Exception:  # noqa: BLE001
        return EvalLibrary(name="unknown", version="unknown")
    for signature, lib_name in _EVAL_FRAMEWORK_SIGNATURES:
        if signature.lower() in full_text:
            print(f"  detected eval framework: {lib_name}")
            return EvalLibrary(name=lib_name, version="unknown")
    return EvalLibrary(name="unknown", version="unknown")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _build_pipeline(
    llm_fallback: LLMFallbackExtractor | None = None,
) -> PaperExtractionPipeline:
    """construct the default pipeline with all default implementations."""
    return PaperExtractionPipeline(
        downloader=PDFDownloader(),
        table_extractor=TableExtractor(),
        table_parser=ResultsTableParser(),
        converter=PaperConverter(),
        writer=PaperWriter(),
        llm_fallback=llm_fallback,
        prose_extractor=ProseExtractor(),
    )


def main() -> None:
    """command-line entry point for the paper extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="extract LLM evaluation results from arXiv PDFs to EEE schema"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--arxiv_id",
        metavar="ID",
        help="arXiv paper ID (e.g. 2407.21783)",
    )
    src.add_argument(
        "--pdf",
        metavar="PATH",
        help="path to a local PDF file",
    )
    src.add_argument(
        "--batch",
        metavar="FILE",
        help="text file with one arXiv ID per line",
    )
    parser.add_argument(
        "--llm-fallback",
        action="store_true",
        default=False,
        help=(
            "activate the LLM fallback extractor. "
            "When table parsing yields no results the LLM extractor runs "
            "automatically; pass this flag to run it unconditionally alongside "
            "table parsing. Requires OPENAI_API_KEY in the environment."
        ),
    )
    parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        default="gpt-4o-mini",
        help="OpenAI model to use for LLM fallback (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    llm_fallback: LLMFallbackExtractor | None = None
    if args.llm_fallback:
        llm_fallback = LLMFallbackExtractor(model=args.llm_model)

    pipeline = _build_pipeline(llm_fallback=llm_fallback)
    all_stats: list[CoverageStats] = []

    if args.arxiv_id:
        stats = pipeline.run(args.arxiv_id, force_llm=bool(args.llm_fallback))
        all_stats.append(stats)
    elif args.pdf:
        stats = pipeline.run(args.pdf, force_llm=bool(args.llm_fallback))
        all_stats.append(stats)
    elif args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"batch file not found: {batch_file}", file=sys.stderr)
            sys.exit(1)
        ids = [
            line.strip()
            for line in batch_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"processing {len(ids)} arXiv papers from {args.batch}...")
        for arxiv_id in ids:
            try:
                stats = pipeline.run(arxiv_id, force_llm=bool(args.llm_fallback))
                all_stats.append(stats)
            except Exception as exc:
                print(f"  error processing {arxiv_id}: {exc}")
                all_stats.append(CoverageStats(arxiv_id=arxiv_id, source="none"))

        # Write machine-readable coverage report for batch runs.
        report_dir = Path("scripts/scrapers/raw")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"coverage_report_{int(time.time())}.json"
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "batch_file": str(args.batch),
            "total_papers": len(all_stats),
            "papers_with_results": sum(1 for s in all_stats if s.total_data_points > 0),
            "papers_with_no_results": sum(1 for s in all_stats if s.total_data_points == 0),
            "total_files_written": sum(s.files_written for s in all_stats),
            "total_validation_failed": sum(s.validation_failed for s in all_stats),
            # source is now a '+'-joined string of tiers used, e.g. "table+prose",
            # so we count substring membership rather than exact equality.
            "source_breakdown": {
                "table_only": sum(1 for s in all_stats if s.source == "table"),
                "prose_only": sum(1 for s in all_stats if s.source == "prose"),
                "llm_only": sum(1 for s in all_stats if s.source == "llm"),
                "has_table": sum(1 for s in all_stats if "table" in s.source),
                "has_prose": sum(1 for s in all_stats if "prose" in s.source),
                "has_llm": sum(1 for s in all_stats if "llm" in s.source),
                "none": sum(1 for s in all_stats if s.source == "none"),
            },
            "per_paper": [asdict(s) for s in all_stats],
        }
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\ncoverage report written to {report_path}")

    total = sum(s.files_written for s in all_stats)
    print(f"\ntotal files written: {total}")


if __name__ == "__main__":
    main()
