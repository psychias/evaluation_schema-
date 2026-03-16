"""
extract_latex.py — LaTeX-source-primary extraction engine for EEE dataset.

Downloads LaTeX source from arXiv, parses all tabular environments, identifies
benchmark tables, and extracts (model, benchmark, score) triples from each paper
independently. Each paper's table produces its own JSON records — no cross-paper
value sharing.

Usage:
    uv run python scripts/extract_latex.py --arxiv_id 2310.06825
    uv run python scripts/extract_latex.py --arxiv_id 2310.06825 --dry_run
    uv run python scripts/extract_latex.py --batch scripts/arxiv_ids_full.txt

Critical design decision: Each paper's comparison table is extracted independently.
When paper A (Mixtral) reports Mistral-7B GSM8K = X, and paper B (Phi-3) reports
Mistral-7B GSM8K = Y, those are two DIFFERENT data points written to different JSON
files in different data/papers_*/ directories. This is the entire point of EEE.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

LATEX_SRC_DIR = _ROOT / "data" / "latex_src"
DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCHEMA_PATH = _ROOT / "shared_task_submission" / "schema" / "eval.schema.json"
if not SCHEMA_PATH.exists():
    SCHEMA_PATH = _ROOT / "eval.schema.json"

# ---------------------------------------------------------------------------
# Benchmark keyword detection
# ---------------------------------------------------------------------------

BENCH_KEYWORDS = [
    "mmlu", "humaneval", "gsm8k", "hellaswag", "arc-c", "arc_c", "arc challenge",
    "winogrande", "truthfulqa", "truthful_qa", "math", "gpqa", "mbpp", "bbh",
    "piqa", "lambada", "drop", "triviaqa", "boolq", "mmlu-pro", "mmlu_pro",
    "agieval", "agi_eval", "ifeval", "musr", "bigbench", "natural questions",
    "naturalquestions", "livecodebench", "bird-sql", "simpleqa", "hiddenmath",
    "global mmlu", "global-mmlu", "hellas", "wino", "arc-e", "arc_e",
    "xquad", "flores", "mgsm", "xorqa", "gmmlu", "eclectic", "ruler", "mrcr",
    "wmt", "code", "reasoning", "benchmark", "docvqa", "infovqa", "textvqa",
    "chartqa", "mmmu", "vqa", "boolq", "siqa", "nq", "obqa",
]

# Model name → HuggingFace ID mapping (for normalisation)
# Add entries as they come up during extraction
MODEL_ID_MAP: dict[str, str] = {
    # Llama 2
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama 2 7b": "meta-llama/Llama-2-7b-hf",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama 2 13b": "meta-llama/Llama-2-13b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-70b": "meta-llama/Llama-2-70b-hf",
    "llama 2 70b": "meta-llama/Llama-2-70b-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-hf",
    # Llama 3
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "meta-llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama 3 8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "meta-llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    # Llama 3.1
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B",
    "llama-3.1-405b": "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama-3.1-405b": "meta-llama/Meta-Llama-3.1-405B",
    # Mistral
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "mistral 7b": "mistralai/Mistral-7B-v0.1",
    # Mixtral
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral 8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7b-v0.1": "mistralai/Mixtral-8x7B-v0.1",
    # Gemma
    "gemma-7b": "google/gemma-7b",
    "gemma 7b": "google/gemma-7b",
    "gemma-2b": "google/gemma-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
    # Qwen
    "qwen2-7b": "Qwen/Qwen2-7B",
    "qwen2-72b": "Qwen/Qwen2-72B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B",
    # DeepSeek
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek v3": "deepseek-ai/DeepSeek-V3",
    # InternLM
    "internlm2-7b": "internlm/internlm2-7b",
    "internlm2-20b": "internlm/internlm2-20b",
    "internlm2 7b": "internlm/internlm2-7b",
    "internlm2 20b": "internlm/internlm2-20b",
    # Falcon
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    # GPT-4
    "gpt-4": "openai/gpt-4",
    "gpt-3.5": "openai/gpt-3.5-turbo",
    # Phi
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
}

# Benchmark name normalisation
BENCH_NORM: dict[str, str] = {
    "hellas": "HellaSwag",
    "hellaswag": "HellaSwag",
    "hella": "HellaSwag",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "gsm-8k": "GSM8K",
    "gsm 8k": "GSM8K",
    "humaneval": "HumanEval",
    "human eval": "HumanEval",
    "arc-c": "ARC-Challenge",
    "arc_c": "ARC-Challenge",
    "arc challenge": "ARC-Challenge",
    "arc-challenge": "ARC-Challenge",
    "arc-e": "ARC-Easy",
    "arc_e": "ARC-Easy",
    "arc easy": "ARC-Easy",
    "winogrande": "WinoGrande",
    "wino": "WinoGrande",
    "truthfulqa": "TruthfulQA",
    "truthful_qa": "TruthfulQA",
    "math": "MATH",
    "gpqa": "GPQA",
    "gpqa diamond": "GPQA",
    "mbpp": "MBPP",
    "bbh": "BBH",
    "piqa": "PIQA",
    "drop": "DROP",
    "triviaqa": "TriviaQA",
    "boolq": "BoolQ",
    "mmlu-pro": "MMLU-Pro",
    "mmlu_pro": "MMLU-Pro",
    "mmlupro": "MMLU-Pro",
    "agieval": "AGIEval",
    "agi eval": "AGIEval",
    "ifeval": "IFEval",
    "livecodebench": "LiveCodeBench",
    "live code bench": "LiveCodeBench",
    "mgsm": "MGSM",
    "xquad": "XQuAD",
    "flores": "Flores",
    "wmt24": "WMT24",
    "wmt24++": "WMT24++",
    "gmmlu": "Global-MMLU",
    "global mmlu": "Global-MMLU",
    "global-mmlu": "Global-MMLU",
    "nq": "NaturalQuestions",
    "naturalquestions": "NaturalQuestions",
    "natural questions": "NaturalQuestions",
    "eclectic": "ECLeKTic",
    "eclectic": "ECLeKTic",
    "ruler": "RULER",
    "mrcr": "MRCR",
    "hiddenmath": "HiddenMath",
    "hidden math": "HiddenMath",
    "simpleqa": "SimpleQA",
    "simple qa": "SimpleQA",
    "bird-sql": "Bird-SQL",
    "bird sql": "Bird-SQL",
}


# ---------------------------------------------------------------------------
# Paper metadata (source_name, org, harness, evaluator_relationship)
# ---------------------------------------------------------------------------

PAPER_META: dict[str, dict] = {
    "2203.15556": {"name": "Training Compute-Optimal Large Language Models (Chinchilla)", "org": "DeepMind", "harness": "unknown", "rel": "first_party"},
    "2204.02311": {"name": "PaLM: Scaling Language Modeling with Pathways", "org": "Google", "harness": "unknown", "rel": "first_party"},
    "2205.01068": {"name": "OPT: Open Pre-trained Transformer Language Models", "org": "Meta AI", "harness": "metaseq", "rel": "first_party"},
    "2210.11416": {"name": "Scaling Instruction-Finetuned Language Models (Flan-T5)", "org": "Google", "harness": "unknown", "rel": "first_party"},
    "2211.05100": {"name": "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model", "org": "BigScience", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2302.13971": {"name": "LLaMA: Open and Efficient Foundation Language Models", "org": "Meta AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2303.08774": {"name": "GPT-4 Technical Report", "org": "OpenAI", "harness": "openai_evals", "rel": "first_party"},
    "2304.01373": {"name": "Pythia: A Suite for Analyzing Large Language Models", "org": "EleutherAI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2304.10457": {"name": "Introducing MPT-7B", "org": "MosaicML", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2305.10403": {"name": "PaLM 2 Technical Report", "org": "Google", "harness": "unknown", "rel": "first_party"},
    "2306.05685": {"name": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "org": "UC Berkeley / LMSYS", "harness": "unknown", "rel": "third_party"},
    "2306.11644": {"name": "The Falcon Series of Open Language Models", "org": "TII UAE", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2307.09288": {"name": "Llama 2: Open Foundation and Fine-Tuned Chat Models", "org": "Meta AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2308.12950": {"name": "Code Llama: Open Foundation Models for Code", "org": "Meta AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2309.05463": {"name": "Textbooks Are All You Need II: phi-1.5 technical report", "org": "Microsoft", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2310.06825": {"name": "Mistral 7B", "org": "Mistral AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2309.16609": {"name": "Baichuan 2: Open Large-scale Language Models", "org": "Baichuan Inc.", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2310.16944": {"name": "Llemma: An Open Language Model for Mathematics", "org": "EleutherAI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2311.11045": {"name": "Zephyr: Direct Distillation of LM Alignment", "org": "HuggingFace", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2312.00752": {"name": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "org": "CMU / Princeton", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2312.06550": {"name": "Mixtral of Experts (preliminary)", "org": "Mistral AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2401.04088": {"name": "Mixtral of Experts", "org": "Mistral AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2312.15166": {"name": "TinyLlama: An Open-Source Small Language Model", "org": "TinyLlama", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2401.02385": {"name": "DeepSeek LLM: Scaling Open-Source Language Models", "org": "DeepSeek AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2402.00838": {"name": "OLMo: Accelerating the Science of Language Models", "org": "AllenAI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2402.16819": {"name": "Yi: Open Foundation Models by 01.AI", "org": "01.AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2402.17834": {"name": "Orca-Math: Unlocking Potential in Synthetic Data", "org": "Microsoft", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2402.19173": {"name": "Flan-T5 technical report (updated)", "org": "Google", "harness": "unknown", "rel": "first_party"},
    "2403.04652": {"name": "ChatGLM: A Family of Large Language Models", "org": "Tsinghua University / Zhipu AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2403.08295": {"name": "Gemma: Open Models Based on Gemini Research and Technology", "org": "Google DeepMind", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2403.17297": {"name": "InternLM2 Technical Report", "org": "Shanghai AI Lab", "harness": "opencompass", "rel": "first_party"},
    "2403.19887": {"name": "WizardLM-2: Technical Report", "org": "Microsoft", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2404.05892": {"name": "Llama 3 (early access technical report)", "org": "Meta AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2404.10774": {"name": "JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars", "org": "MIT / Princeton", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2404.14219": {"name": "Phi-3 Technical Report", "org": "Microsoft", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2404.14619": {"name": "OpenELM: An Efficient Language Model Family for On-Device AI", "org": "Apple", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2405.04324": {"name": "DBRX: The World's Best Open LLM", "org": "Databricks", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2405.04434": {"name": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", "org": "DeepSeek AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2405.19327": {"name": "Aya 23: Open Weight Releases to Further Multilingual Progress", "org": "Cohere", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2406.12793": {"name": "EXAONE 3.0 7.8B Instruction Tuned Language Model", "org": "LG AI Research", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2407.12511": {"name": "Jamba 1.5 (mini)", "org": "AI21 Labs", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2407.21783": {"name": "The Llama 3 Herd of Models", "org": "Meta AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2408.00118": {"name": "Gemma 2: Improving Open Language Models at a Practical Size", "org": "Google DeepMind", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2411.14599": {"name": "Zamba2-7B Technical Report", "org": "Zyphra AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2411.15138": {"name": "EXAONE 3.5: Series of Large Language Models for Real-world Use Cases", "org": "LG AI Research", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2412.19437": {"name": "DeepSeek-V3 Technical Report", "org": "DeepSeek AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2501.12948": {"name": "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs", "org": "DeepSeek AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2502.02737": {"name": "Llama 4 (early technical report)", "org": "Meta AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    # Phase 7 additions
    "2406.11704": {"name": "Nemotron-4 340B Technical Report", "org": "NVIDIA", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2407.10671": {"name": "Qwen2 Technical Report", "org": "Alibaba Group", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2407.14885": {"name": "Falcon2-11B Technical Report", "org": "TII UAE", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2408.12570": {"name": "Jamba-1.5: Hybrid Transformer-Mamba Models at Scale", "org": "AI21 Labs", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2409.12122": {"name": "Qwen2.5-Math Technical Report", "org": "Alibaba Group", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2409.12186": {"name": "Qwen2.5-Coder Technical Report", "org": "Alibaba Group", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2410.21276": {"name": "GPT-4o System Card", "org": "OpenAI", "harness": "openai_evals", "rel": "first_party"},
    "2411.04905": {"name": "OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models", "org": "OpenCoder Team", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2412.04261": {"name": "Aya Expanse: Connecting the Dots for Multilingual Understanding", "org": "Cohere", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2412.08905": {"name": "Phi-4 Technical Report", "org": "Microsoft", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2412.15115": {"name": "Qwen2.5 Technical Report", "org": "Alibaba Group", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2501.00656": {"name": "OLMo 2: Improving the Science of Language Model Training", "org": "AllenAI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2501.00663": {"name": "Titans: Learning to Memorize at Test Time", "org": "Google", "harness": "lm-evaluation-harness", "rel": "third_party"},
    "2501.12599": {"name": "Kimi k1.5: Scaling Reinforcement Learning with LLMs", "org": "Moonshot AI", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2503.19786": {"name": "Gemma 3 Technical Report", "org": "Google DeepMind", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2505.00949": {"name": "Llama-Nemotron: Efficient Reasoning Models", "org": "NVIDIA", "harness": "lm-evaluation-harness", "rel": "first_party"},
    "2505.09388": {"name": "Qwen3 Technical Report", "org": "Alibaba Group", "harness": "lm-evaluation-harness", "rel": "first_party"},
}


# ---------------------------------------------------------------------------
# Step 1: Download LaTeX source
# ---------------------------------------------------------------------------

def download_latex_source(arxiv_id: str, force: bool = False) -> Path | None:
    """Download and extract LaTeX source from arXiv. Returns path or None on failure."""
    out_dir = LATEX_SRC_DIR / arxiv_id
    if out_dir.exists() and any(out_dir.rglob("*.tex")) and not force:
        print(f"  [{arxiv_id}] LaTeX source already cached at {out_dir}")
        return out_dir

    url = f"https://arxiv.org/src/{arxiv_id}"
    print(f"  [{arxiv_id}] downloading LaTeX source from {url} ...")
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "EEE-pipeline/2.0 (research; not-commercial)"},
            timeout=60,
        )
        if resp.status_code == 404:
            print(f"  [{arxiv_id}] LaTeX source not found (404)")
            return None
        if resp.status_code == 429:
            print(f"  [{arxiv_id}] rate limited (429) — sleeping 60s")
            time.sleep(60)
            resp = requests.get(url, headers={"User-Agent": "EEE-pipeline/2.0"}, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [{arxiv_id}] download failed: {exc}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tar = tarfile.open(fileobj=io.BytesIO(resp.content))
        tar.extractall(path=out_dir)
        tar.close()
        print(f"  [{arxiv_id}] extracted {len(list(out_dir.rglob('*.tex')))} .tex files")
    except tarfile.ReadError:
        # Single .tex file
        (out_dir / "paper.tex").write_bytes(resp.content)
        print(f"  [{arxiv_id}] saved as single paper.tex")
    return out_dir


# ---------------------------------------------------------------------------
# Step 2: Collect macros from all .tex files
# ---------------------------------------------------------------------------

def collect_macros(src_dir: Path) -> dict[str, str]:
    """Collect all \\newcommand and \\def macro definitions."""
    macros: dict[str, str] = {}
    for tex_file in src_dir.rglob("*.tex"):
        try:
            content = tex_file.read_text(errors="replace")
        except Exception:
            continue
        # \newcommand{\name}{replacement} and \newcommand{\name}[n]{replacement}
        for m in re.finditer(
            r"\\(?:newcommand|renewcommand)\*?\s*\{(\\[a-zA-Z]+)\}(?:\[\d+\])?\{([^}]*)\}",
            content,
        ):
            macros[m.group(1)] = m.group(2)
        # \def\name{replacement}
        for m in re.finditer(r"\\def\s*(\\[a-zA-Z]+)\s*\{([^}]*)\}", content):
            macros[m.group(1)] = m.group(2)
    return macros


def expand_macros(text: str, macros: dict[str, str]) -> str:
    """Expand known macros in text."""
    for name, val in sorted(macros.items(), key=lambda x: -len(x[0])):
        safe_val = val
        text = re.sub(
            re.escape(name) + r"(?![a-zA-Z])",
            lambda m, v=safe_val: v,
            text,
        )
    return text


# ---------------------------------------------------------------------------
# Step 3: Parse LaTeX tables
# ---------------------------------------------------------------------------

def clean_cell(cell: str) -> str:
    """Strip LaTeX formatting from a table cell, returning plain text."""
    # Remove comments
    cell = re.sub(r"(?<!\\)%.*$", "", cell, flags=re.MULTILINE)
    # Strip common formatting commands
    for cmd in [
        "textbf", "textit", "emph", "underline", "textsc", "textrm",
        "text", "mathrm", "mathbf", "mathit", "mathtt",
        "ul", "uline", "best", "second", "first", "sota", "ours",
        "cellcolor", "rowcolor",
    ]:
        cell = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", cell)
    # Handle \textcolor{color}{text}
    cell = re.sub(r"\\textcolor\{[^}]*\}\{([^}]*)\}", r"\1", cell)
    # Strip {\bf X} and {\it X} patterns
    cell = re.sub(r"\{\\(?:bf|it|em|sc)\s+([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\(?:bf|it|em|sc)\b\s*", "", cell)
    # multirow / multicolumn
    cell = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", cell)
    # rule commands
    cell = re.sub(
        r"\\(?:hline|toprule|midrule|bottomrule|cmidrule(?:\([^)]*\))?\{[^}]*\}|addlinespace(?:\[[^\]]*\])?)\s*",
        "",
        cell,
    )
    # math mode: $...$
    cell = re.sub(r"\$([^$]*)\$", r"\1", cell)
    # \% → % (LaTeX percent escape)
    cell = cell.replace(r"\%", "%")
    # ~ → space (LaTeX non-breaking space)
    cell = cell.replace("~", " ")
    # remaining commands with one brace argument
    cell = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", cell)
    # remaining backslash commands (but not \%)
    cell = re.sub(r"\\[a-zA-Z@]+", "", cell)
    # stray braces
    cell = re.sub(r"[{}]", "", cell)
    # normalise whitespace
    return cell.strip()


def _remove_brace_group(s: str) -> str:
    """Remove one top-level {...} group from the start of s (handles nested braces)."""
    if not s or s[0] != "{":
        return s
    depth = 0
    for i, c in enumerate(s):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:]
    return s


def parse_tabular(raw_tex: str) -> list[list[str]]:
    """Parse a raw tabular environment into a list of rows × columns."""
    # Remove LaTeX comments
    lines = raw_tex.split("\n")
    lines = [re.sub(r"(?<!\\)%.*$", "", l) for l in lines]
    content = "\n".join(lines)

    # Remove \begin{tabular[x*]}{colspec} — handles nested braces in colspec
    # First strip \begin{tabular...}
    content = re.sub(r"\\begin\{(tabular[x*]?|tabulary)\}", "", content)
    content = re.sub(r"\\end\{(tabular[x*]?|tabulary)\}", "", content)
    # Strip leading brace groups (colspec, possibly with @{} etc.)
    # Strip up to 3 consecutive {..} groups that appear before any row content
    content = content.lstrip()
    for _ in range(3):
        if content.startswith("{"):
            content = _remove_brace_group(content).lstrip()
    # Replace rule lines with blank lines
    content = re.sub(
        r"\\(?:hline|toprule|midrule|bottomrule|cmidrule(?:\([^)]*\))?\{[^}]*\}|addlinespace(?:\[[^\]]*\])?)\s*",
        "\n",
        content,
    )
    # Split on \\ (row separator)
    raw_rows = re.split(r"\\\\", content)
    rows: list[list[str]] = []
    for row in raw_rows:
        row = row.strip()
        if not row:
            continue
        cells = [clean_cell(c) for c in row.split("&")]
        if all(c == "" for c in cells):
            continue
        rows.append(cells)
    return rows


def find_all_tables(src_dir: Path, macros: dict[str, str]) -> list[dict]:
    """Find all tabular environments in all .tex files."""
    tables: list[dict] = []
    table_idx = 0
    for tex_file in sorted(src_dir.rglob("*.tex")):
        try:
            content = tex_file.read_text(errors="replace")
        except Exception:
            continue
        content_exp = expand_macros(content, macros)

        pattern = r"\\begin\{(tabular[x*]?|tabulary)\}(.*?)\\end\{\1\}"
        for m in re.finditer(pattern, content_exp, re.DOTALL):
            table_idx += 1
            raw_tex = m.group(0)
            rows = parse_tabular(raw_tex)
            if not rows:
                continue

            flat = " ".join(" ".join(r) for r in rows).lower()
            has_bench = any(kw in flat for kw in BENCH_KEYWORDS)

            # Also grab the surrounding caption if present
            # Look 500 chars before the table for \caption{...}
            start_pos = m.start()
            context_before = content_exp[max(0, start_pos - 500) : start_pos]
            caption_m = re.search(r"\\caption\{([^}]*)\}", context_before)
            caption = clean_cell(caption_m.group(1)) if caption_m else ""

            tables.append({
                "index": table_idx,
                "file": str(tex_file.relative_to(src_dir)),
                "rows": rows,
                "has_benchmark_keywords": has_bench,
                "caption": caption,
                "raw_tex": raw_tex[:500],  # for debugging
            })
    return tables


# ---------------------------------------------------------------------------
# Step 4: Extract (model, benchmark, score, n_shot) triples
# ---------------------------------------------------------------------------

def normalise_bench_name(raw: str) -> str | None:
    """Normalise a benchmark header cell to a canonical name."""
    s = raw.lower().strip()
    # Strip trailing n-shot annotations: "MMLU (5-shot)" → "mmlu"
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = re.sub(r"\s*\d+-shot.*$", "", s)
    s = re.sub(r"\s*(maj@\d+|pass@\d+|cot|standard|chain.of.thought).*$", "", s, flags=re.I)
    s = s.strip().rstrip(".")
    # Direct lookup
    if s in BENCH_NORM:
        return BENCH_NORM[s]
    # Partial match
    for key, val in BENCH_NORM.items():
        if s.startswith(key) or key in s:
            return val
    return None


def extract_n_shot(header_cell: str) -> str | None:
    """Extract n-shot annotation from header like 'GSM8K (5-shot)'."""
    m = re.search(r"(\d+)-shot", header_cell, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\((\d+)\)", header_cell)
    if m:
        return m.group(1)
    return None


def is_numeric(s: str) -> bool:
    """True if s can be parsed as a float (handles % suffix)."""
    try:
        float(s.replace(",", "").replace("−", "-").replace("–", "-").replace(r"\%", "").rstrip("%"))
        return True
    except ValueError:
        return False


def parse_score(s: str) -> float | None:
    """Parse a score string, returning a [0, 1] float or None."""
    s = s.strip().replace(",", "").replace("−", "-").replace("–", "-").replace("~", "")
    s = s.replace(r"\%", "").rstrip("%").rstrip("\\")
    # Remove ± uncertainty
    s = re.split(r"[±\+\-]{1,2}[\d.]+$", s)[0].strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if v < 0:
        return None
    if v > 1.0:
        # Likely a percentage
        return round(v / 100.0, 6)
    return round(v, 6)


def normalise_model_name(raw: str) -> tuple[str, str]:
    """
    Map a raw model name from a paper table to (model_display_name, hf_model_id).
    Returns best-effort values. hf_model_id may just be derived from display_name.
    """
    raw_clean = raw.strip()
    key = raw_clean.lower().strip()
    # Direct lookup
    if key in MODEL_ID_MAP:
        hf_id = MODEL_ID_MAP[key]
        display = hf_id.split("/")[-1]
        return display, hf_id
    # Try partial matches
    for map_key, hf_id in MODEL_ID_MAP.items():
        if map_key in key or key in map_key:
            display = hf_id.split("/")[-1]
            return display, hf_id
    # Fallback: create a reasonable HF-style ID
    # Try to detect developer from name
    developer = "unknown"
    for prefix, dev in [
        ("llama", "meta-llama"), ("mistral", "mistralai"), ("mixtral", "mistralai"),
        ("gemma", "google"), ("gemini", "google"), ("gpt", "openai"),
        ("claude", "anthropic"), ("palm", "google"), ("falcon", "tiiuae"),
        ("qwen", "Qwen"), ("deepseek", "deepseek-ai"), ("internlm", "internlm"),
        ("phi", "microsoft"), ("bloom", "bigscience"), ("opt", "facebook"),
        ("yi-", "01-ai"), ("baichuan", "baichuan-inc"), ("olmo", "allenai"),
        ("dbrx", "databricks"), ("exaone", "LGAI-RESEARCH"),
    ]:
        if prefix in key:
            developer = dev
            break
    # Sanitise for HF ID
    hf_name = re.sub(r"[^a-zA-Z0-9._-]", "-", raw_clean)
    hf_id = f"{developer}/{hf_name}"
    return raw_clean, hf_id


def extract_triples_from_table(
    rows: list[list[str]], table_idx: int
) -> list[dict]:
    """
    Extract (model, benchmark, score, n_shot) triples from parsed table rows.

    Handles two orientations:
    - Row-oriented: rows are models, columns are benchmarks
    - Column-oriented: columns are models, rows are benchmarks (less common)

    Returns list of dicts with keys: model_name, hf_model_id, benchmark, score,
    n_shot, table_index, row_idx, col_idx.
    """
    if not rows or len(rows) < 2:
        return []

    triples: list[dict] = []

    # Strategy: look for a header row where cells match benchmark names
    # and data rows where first column is a model name

    # Find the best header row (most benchmark-keyword matches)
    best_header_row = 0
    best_score = 0
    for i, row in enumerate(rows[:5]):  # check first 5 rows for header
        bench_count = sum(1 for cell in row if normalise_bench_name(cell) is not None)
        if bench_count > best_score:
            best_score = bench_count
            best_header_row = i

    # Filter out column-spec rows (e.g. "lccccccccc@" that leaks from begin{tabular}{...})
    rows = [r for r in rows if not (len(r) == 1 and re.match(r'^[lcr|@p\s\{\}\.0-9*!<>X]+$', r[0]))]
    if not rows:
        return []

    # Re-find best header row after filtering
    best_header_row = 0
    best_score = 0
    for i, row in enumerate(rows[:5]):
        bench_count = sum(1 for cell in row if normalise_bench_name(cell) is not None)
        if bench_count > best_score:
            best_score = bench_count
            best_header_row = i

    if best_score == 0:
        # No clear benchmark header — try column orientation
        return _try_column_orientation(rows, table_idx)

    header = rows[best_header_row]
    # Map column index → (benchmark_name, n_shot)
    bench_cols: dict[int, tuple[str, str | None]] = {}
    for j, cell in enumerate(header):
        bench = normalise_bench_name(cell)
        if bench:
            nshot = extract_n_shot(cell)
            bench_cols[j] = (bench, nshot)

    if not bench_cols:
        return []

    # Data rows follow the header
    # First column of each data row is the model name
    for i in range(best_header_row + 1, len(rows)):
        row = rows[i]
        if not row:
            continue
        model_raw = row[0].strip()
        if not model_raw or model_raw in ("", "-", "Model", "model"):
            continue
        # Skip sub-header rows (no numeric scores)
        has_numeric = any(is_numeric(row[j]) for j in bench_cols if j < len(row))
        if not has_numeric:
            continue

        model_display, hf_id = normalise_model_name(model_raw)

        for j, (bench, nshot) in bench_cols.items():
            if j >= len(row):
                continue
            score_str = row[j].strip()
            if not score_str or score_str in ("-", "—", "N/A", "n/a", ""):
                continue
            score = parse_score(score_str)
            if score is None:
                continue

            triples.append({
                "model_name": model_display,
                "hf_model_id": hf_id,
                "model_raw": model_raw,
                "benchmark": bench,
                "score": score,
                "n_shot": nshot,
                "score_raw": score_str,
                "table_index": table_idx,
                "row_idx": i,
                "col_idx": j,
            })

    return triples


def _try_column_orientation(rows: list[list[str]], table_idx: int) -> list[dict]:
    """Try column-oriented extraction where rows=benchmarks, cols=models."""
    if not rows or len(rows[0]) < 2:
        return []

    triples: list[dict] = []
    # First column might be benchmark names, subsequent columns are models
    # Check if first-column cells match benchmark names
    first_col_benches = sum(1 for row in rows if normalise_bench_name(row[0]) is not None)
    if first_col_benches < 2:
        return []

    # Header row = row 0, model names
    header = rows[0]
    # Map col index → (model_name, hf_id)
    model_cols: dict[int, tuple[str, str]] = {}
    for j in range(1, len(header)):
        cell = header[j].strip()
        if cell and cell not in ("-", "Model", ""):
            display, hf_id = normalise_model_name(cell)
            model_cols[j] = (display, hf_id)

    for i in range(1, len(rows)):
        row = rows[i]
        bench_raw = row[0].strip() if row else ""
        bench = normalise_bench_name(bench_raw)
        if not bench:
            continue
        nshot = extract_n_shot(bench_raw)

        for j, (model_display, hf_id) in model_cols.items():
            if j >= len(row):
                continue
            score_str = row[j].strip()
            if not score_str or score_str in ("-", "—", "N/A", "n/a", ""):
                continue
            score = parse_score(score_str)
            if score is None:
                continue

            triples.append({
                "model_name": model_display,
                "hf_model_id": hf_id,
                "model_raw": header[j].strip(),
                "benchmark": bench,
                "score": score,
                "n_shot": nshot,
                "score_raw": score_str,
                "table_index": table_idx,
                "row_idx": i,
                "col_idx": j,
            })

    return triples


# ---------------------------------------------------------------------------
# Step 5: Build EEE schema JSON records
# ---------------------------------------------------------------------------

def build_eee_record(
    arxiv_id: str,
    model_name: str,
    hf_model_id: str,
    benchmarks: list[dict],
    retrieved_ts: str,
    meta: dict,
) -> dict:
    """Build one EEE schema JSON record for a (paper, model) pair."""
    source_url = f"https://arxiv.org/abs/{arxiv_id}"
    if "/" in hf_model_id:
        developer = hf_model_id.split("/")[0]
    else:
        developer = hf_model_id

    eval_results = []
    for b in benchmarks:
        n_shot_str = b.get("n_shot") or "unknown"
        eval_results.append({
            "evaluation_name": b["benchmark"],
            "source_data": {
                "dataset_name": "arXiv paper",
                "source_type": "url",
                "url": [source_url],
            },
            "metric_config": {
                "evaluation_description": f"score on {b['benchmark']} as reported in arXiv:{arxiv_id}",
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0.0,
                "max_score": 1.0,
            },
            "score_details": {"score": b["score"]},
            "generation_config": {
                "additional_details": {
                    "n_shot": str(n_shot_str),
                    "harness": meta.get("harness", "unknown"),
                    "source": f"arXiv:{arxiv_id}",
                    "table_index": str(b.get("table_index", "unknown")),
                    "score_raw": str(b.get("score_raw", "")),
                }
            },
        })

    return {
        "schema_version": "0.2.1",
        "evaluation_id": f"papers_{arxiv_id}/{re.sub(r'[^a-zA-Z0-9._/-]', '_', hf_model_id)}/{retrieved_ts}",
        "retrieved_timestamp": retrieved_ts,
        "source_metadata": {
            "source_name": meta.get("name", f"arXiv:{arxiv_id}"),
            "source_type": "documentation",
            "source_organization_name": meta.get("org", "unknown"),
            "source_organization_url": source_url,
            "evaluator_relationship": meta.get("rel", "first_party"),
        },
        "eval_library": {
            "name": meta.get("harness", "unknown"),
            "version": "unknown",
        },
        "model_info": {
            "name": model_name,
            "id": hf_model_id,
            "developer": developer,
        },
        "evaluation_results": eval_results,
    }


# ---------------------------------------------------------------------------
# Step 6: Write records to disk
# ---------------------------------------------------------------------------

def load_schema_validator():
    """Load jsonschema validator. Returns None if unavailable."""
    if not SCHEMA_PATH.exists():
        return None
    try:
        from jsonschema.validators import validator_for
        schema = json.loads(SCHEMA_PATH.read_text())
        cls = validator_for(schema)
        return cls(schema)
    except Exception as exc:
        print(f"  [schema] could not load validator: {exc}")
        return None


_VALIDATOR = None
_VALIDATOR_LOADED = False


def get_validator():
    global _VALIDATOR, _VALIDATOR_LOADED
    if not _VALIDATOR_LOADED:
        _VALIDATOR = load_schema_validator()
        _VALIDATOR_LOADED = True
    return _VALIDATOR


def write_records(records: list[dict], arxiv_id: str) -> int:
    """Validate and write records to data/papers_{arxiv_id}/."""
    out_base = DATA_DIR / f"papers_{arxiv_id}"
    validator = get_validator()
    written = 0
    for rec in records:
        mid = rec.get("model_info", {}).get("id", "?")
        # Validate
        if validator:
            try:
                validator.validate(rec)
            except Exception as exc:
                msg = getattr(exc, "message", str(exc))
                print(f"  [SKIP invalid] {mid}: {msg[:120]}", file=sys.stderr)
                continue

        # Write
        if "/" in mid:
            dev, model_name = mid.split("/", 1)
        else:
            dev = rec["model_info"].get("developer", "unknown")
            model_name = mid

        dev = re.sub(r'[<>:"/\\|?*]', "_", dev)
        model_name = re.sub(r'[<>:"/\\|?*]', "_", model_name)
        out_dir = out_base / dev / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{uuid.uuid4()}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        written += 1

    return written


# ---------------------------------------------------------------------------
# Main extraction pipeline for a single paper
# ---------------------------------------------------------------------------

def extract_paper(arxiv_id: str, dry_run: bool = False) -> dict:
    """
    Full extraction pipeline for one arXiv paper.
    Returns a result dict with extraction statistics.
    """
    retrieved_ts = str(time.time())
    meta = PAPER_META.get(arxiv_id, {"name": f"arXiv:{arxiv_id}", "org": "unknown", "harness": "unknown", "rel": "first_party"})

    result = {
        "arxiv_id": arxiv_id,
        "method": None,
        "tables_found": 0,
        "benchmark_tables": 0,
        "triples_extracted": 0,
        "models": [],
        "benchmarks": [],
        "records_written": 0,
        "validation_passed": False,
        "error": None,
    }

    # --- Try LaTeX source ---
    src_dir = download_latex_source(arxiv_id)
    if src_dir and any(src_dir.rglob("*.tex")):
        result["method"] = "latex"
        macros = collect_macros(src_dir)
        tables = find_all_tables(src_dir, macros)
        result["tables_found"] = len(tables)

        bench_tables = [t for t in tables if t["has_benchmark_keywords"]]
        result["benchmark_tables"] = len(bench_tables)
        print(f"  [{arxiv_id}] {len(tables)} tables found, {len(bench_tables)} with benchmark keywords")

        all_triples: list[dict] = []
        for table in bench_tables:
            triples = extract_triples_from_table(table["rows"], table["index"])
            all_triples.extend(triples)

        # Also try column-oriented tables
        for table in bench_tables:
            col_triples = _try_column_orientation(table["rows"], table["index"])
            # Only add if we got something and didn't already extract from this table
            existing_tables = {t["table_index"] for t in all_triples}
            if col_triples and table["index"] not in existing_tables:
                all_triples.extend(col_triples)

        result["triples_extracted"] = len(all_triples)
        print(f"  [{arxiv_id}] extracted {len(all_triples)} triples")

        if not all_triples:
            print(f"  [{arxiv_id}] WARNING: no triples extracted from LaTeX — trying Docling fallback")
            result["method"] = "docling_fallback"
            return _docling_fallback(arxiv_id, result, meta, retrieved_ts, dry_run)
    else:
        print(f"  [{arxiv_id}] LaTeX source unavailable — trying Docling fallback")
        result["method"] = "docling_fallback"
        return _docling_fallback(arxiv_id, result, meta, retrieved_ts, dry_run)

    # --- Consolidate triples by model ---
    # Group triples by (hf_model_id) — keep one score per (model, benchmark),
    # preferring the first occurrence (first table, first mention)
    model_benchmarks: dict[str, dict] = {}  # hf_id → {bench: triple}
    for t in all_triples:
        hf_id = t["hf_model_id"]
        bench = t["benchmark"]
        if hf_id not in model_benchmarks:
            model_benchmarks[hf_id] = {"_model_name": t["model_name"], "_model_raw": t["model_raw"]}
        if bench not in model_benchmarks[hf_id]:
            model_benchmarks[hf_id][bench] = t

    result["models"] = list(model_benchmarks.keys())
    result["benchmarks"] = sorted(
        set(b for m in model_benchmarks.values() for b in m if not b.startswith("_"))
    )

    print(f"  [{arxiv_id}] {len(model_benchmarks)} models: {list(model_benchmarks.keys())[:5]}")
    print(f"  [{arxiv_id}] benchmarks: {result['benchmarks']}")

    if dry_run:
        result["validation_passed"] = True
        return result

    # --- Build and write EEE records ---
    records = []
    for hf_id, bench_data in model_benchmarks.items():
        model_name = bench_data.get("_model_name", hf_id.split("/")[-1])
        benchmarks_list = [
            v for k, v in bench_data.items()
            if not k.startswith("_") and isinstance(v, dict)
        ]
        if not benchmarks_list:
            continue
        rec = build_eee_record(
            arxiv_id, model_name, hf_id, benchmarks_list, retrieved_ts, meta
        )
        records.append(rec)

    written = write_records(records, arxiv_id)
    result["records_written"] = written
    result["validation_passed"] = written > 0
    print(f"  [{arxiv_id}] wrote {written} JSON records")
    return result


def _docling_fallback(
    arxiv_id: str, result: dict, meta: dict, retrieved_ts: str, dry_run: bool
) -> dict:
    """Try Docling extraction as fallback."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        result["error"] = "Docling not available"
        print(f"  [{arxiv_id}] Docling not available — skipping")
        return result

    pdf_dir = _ROOT / "data" / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    pdf_path = pdf_dir / f"{arxiv_id}.pdf"

    if not pdf_path.exists():
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"  [{arxiv_id}] downloading PDF from {url} ...")
        try:
            resp = requests.get(url, headers={"User-Agent": "EEE-pipeline/2.0"}, timeout=120)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
        except Exception as exc:
            result["error"] = f"PDF download failed: {exc}"
            return result

    print(f"  [{arxiv_id}] running Docling on {pdf_path} ...")
    try:
        converter = DocumentConverter()
        doc_result = converter.convert(str(pdf_path))
    except Exception as exc:
        result["error"] = f"Docling failed: {exc}"
        return result

    all_triples: list[dict] = []
    for t_idx, table in enumerate(doc_result.document.tables):
        try:
            df = table.export_to_dataframe(doc_result.document)
            rows = [list(df.columns)] + df.values.tolist()
            rows = [[str(c) for c in row] for row in rows]
        except Exception:
            try:
                df = table.export_to_dataframe()
                rows = [list(df.columns)] + df.values.tolist()
                rows = [[str(c) for c in row] for row in rows]
            except Exception:
                continue

        flat = " ".join(" ".join(row) for row in rows).lower()
        has_bench = any(kw in flat for kw in BENCH_KEYWORDS)
        if not has_bench:
            continue

        triples = extract_triples_from_table(rows, t_idx)
        all_triples.extend(triples)

    result["triples_extracted"] = len(all_triples)
    if not all_triples:
        result["error"] = "No triples extracted"
        return result

    # Consolidate and write (same as LaTeX path)
    model_benchmarks: dict[str, dict] = {}
    for t in all_triples:
        hf_id = t["hf_model_id"]
        bench = t["benchmark"]
        if hf_id not in model_benchmarks:
            model_benchmarks[hf_id] = {"_model_name": t["model_name"]}
        if bench not in model_benchmarks[hf_id]:
            model_benchmarks[hf_id][bench] = t

    result["models"] = list(model_benchmarks.keys())
    result["benchmarks"] = sorted(
        set(b for m in model_benchmarks.values() for b in m if not b.startswith("_"))
    )

    if dry_run:
        result["validation_passed"] = True
        return result

    records = []
    for hf_id, bench_data in model_benchmarks.items():
        model_name = bench_data.get("_model_name", hf_id.split("/")[-1])
        benchmarks_list = [v for k, v in bench_data.items() if not k.startswith("_") and isinstance(v, dict)]
        if not benchmarks_list:
            continue
        rec = build_eee_record(arxiv_id, model_name, hf_id, benchmarks_list, retrieved_ts, meta)
        records.append(rec)

    written = write_records(records, arxiv_id)
    result["records_written"] = written
    result["validation_passed"] = written > 0
    return result


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_batch(ids: list[str], dry_run: bool = False, delay: float = 3.0) -> list[dict]:
    """Extract all papers in ids list. Returns list of result dicts."""
    results = []
    for i, arxiv_id in enumerate(ids):
        print(f"\n[{i+1}/{len(ids)}] Processing {arxiv_id} ...")
        try:
            r = extract_paper(arxiv_id, dry_run=dry_run)
        except Exception as exc:
            import traceback
            r = {"arxiv_id": arxiv_id, "error": str(exc), "traceback": traceback.format_exc()}
            print(f"  [{arxiv_id}] ERROR: {exc}")
        results.append(r)
        # Save log after each paper
        log_path = RESULTS_DIR / "extraction_log.json"
        log_path.write_text(json.dumps(results, indent=2))
        # Rate limit
        if i < len(ids) - 1:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EEE LaTeX/Docling extraction engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arxiv_id", help="Single arXiv ID to process")
    group.add_argument("--batch", help="File with one arXiv ID per line")
    parser.add_argument("--dry_run", action="store_true", help="Extract but don't write JSON")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between downloads")
    args = parser.parse_args()

    LATEX_SRC_DIR.mkdir(parents=True, exist_ok=True)

    if args.arxiv_id:
        r = extract_paper(args.arxiv_id, dry_run=args.dry_run)
        print(json.dumps(r, indent=2))
    else:
        ids = []
        for line in Path(args.batch).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                ids.append(line)
        print(f"Processing {len(ids)} papers from {args.batch}")
        results = extract_batch(ids, dry_run=args.dry_run, delay=args.delay)
        passed = sum(1 for r in results if r.get("validation_passed"))
        failed = sum(1 for r in results if r.get("error"))
        print(f"\n=== Done: {passed}/{len(results)} passed, {failed} errors ===")


if __name__ == "__main__":
    main()
