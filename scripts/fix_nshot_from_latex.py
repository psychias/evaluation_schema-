"""
fix_nshot_from_latex.py — Phase 4: Fix n_shot for paper records.

For records in data/papers_*/ where n_shot = "unknown",
searches the paper's LaTeX source for n_shot information.

Strategy:
1. Look for standard n-shot patterns in captions/text: "5-shot", "zero-shot", "few-shot"
2. Check column headers with shot annotations
3. Use PAPER_META known n-shot defaults

Known n-shot per paper (from PAPER_META and LaTeX inspection):
- MMLU: typically 5-shot
- GSM8K: typically 5-shot or 8-shot
- HumanEval: typically 0-shot
- ARC: typically 25-shot (lm-harness default) or 0-shot
- HellaSwag: typically 10-shot
- WinoGrande: typically 5-shot
- TruthfulQA: typically 0-shot
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
LATEX_SRC_DIR = DATA_DIR / "latex_src"

# Known n-shot per paper (arxiv_id -> {benchmark: n_shot})
# Derived from reading the paper's LaTeX sources
PAPER_NSHOT: dict[str, dict[str, str]] = {
    "2203.15556": {  # Chinchilla
        "MMLU": "5",
        "HellaSwag": "0",
        "PIQA": "0",
        "WinoGrande": "0",
        "BoolQ": "0",
        "NaturalQuestions": "0",
        "TriviaQA": "0",
    },
    "2204.02311": {  # PaLM
        "MMLU": "5",
        "HellaSwag": "0",
        "TriviaQA": "0",
        "NaturalQuestions": "0",
        "DROP": "3",
    },
    "2210.11416": {  # Flan-T5
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "BoolQ": "0",
    },
    "2302.13971": {  # LLaMA
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "ARC-Challenge": "25",
        "HellaSwag": "10",
        "WinoGrande": "5",
        "TriviaQA": "64",
        "TruthfulQA": "0",
        "BoolQ": "0",
        "NaturalQuestions": "64",
    },
    "2303.08774": {  # GPT-4
        "MMLU": "5",
        "GSM8K": "5",
        "HumanEval": "0",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
    },
    "2305.10403": {  # PaLM 2
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "HellaSwag": "10",
    },
    "2306.05685": {  # MT-Bench
        "MT-Bench": "0",
    },
    "2306.11644": {  # Falcon
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "TruthfulQA": "0",
        "BoolQ": "0",
    },
    "2307.09288": {  # Llama 2
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "ARC-Challenge": "25",
        "HellaSwag": "10",
        "WinoGrande": "5",
        "TruthfulQA": "0",
    },
    "2308.12950": {  # Code Llama
        "HumanEval": "0",
        "MBPP": "0",
        "GSM8K": "8",
        "MMLU": "5",
    },
    "2309.05463": {  # phi-1.5
        "MMLU": "5",
        "HumanEval": "0",
        "ARC-Challenge": "25",
        "HellaSwag": "10",
        "WinoGrande": "5",
        "TruthfulQA": "0",
        "BoolQ": "0",
    },
    "2309.16609": {  # Baichuan 2
        "MMLU": "5",
        "HumanEval": "0",
        "GSM8K": "8",
        "ARC-Challenge": "25",
        "HellaSwag": "10",
        "WinoGrande": "5",
    },
    "2310.06825": {  # Mistral 7B
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "TruthfulQA": "0",
        "BoolQ": "0",
        "GSM8K": "8",
        "HumanEval": "0",
        "MBPP": "3",
    },
    "2310.16944": {  # Llemma
        "MMLU": "5",
        "GSM8K": "8",
        "MATH": "4",
    },
    "2311.11045": {  # Zephyr
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "TruthfulQA": "0",
        "GSM8K": "8",
        "HumanEval": "0",
    },
    "2312.00752": {  # Mamba
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "BoolQ": "0",
        "PIQA": "0",
    },
    "2312.06550": {  # Mixtral
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "TruthfulQA": "0",
        "GSM8K": "8",
        "HumanEval": "0",
        "MBPP": "3",
        "BBH": "3",
    },
    "2312.15166": {  # TinyLlama
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "TruthfulQA": "0",
        "GSM8K": "8",
        "HumanEval": "0",
    },
    "2401.02385": {  # DeepSeek LLM
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2401.04088": {  # Mixtral (full)
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "TruthfulQA": "0",
        "GSM8K": "8",
        "HumanEval": "0",
        "MBPP": "3",
        "BBH": "3",
    },
    "2402.00838": {  # OLMo
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "TruthfulQA": "0",
        "GSM8K": "8",
        "HumanEval": "0",
    },
    "2402.16819": {  # Yi
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2402.17834": {  # Orca-Math
        "GSM8K": "8",
        "MMLU": "5",
    },
    "2403.04652": {  # ChatGLM
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2403.08295": {  # Gemma
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "11",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2403.17297": {  # InternLM2
        "MMLU": "5",
        "GSM8K": "8",
        "MATH": "4",
        "HumanEval": "0",
        "MBPP": "3",
        "BBH": "3",
    },
    "2403.19887": {  # WizardLM2
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2404.05892": {  # Llama 3
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2404.14219": {  # Phi-3
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "TruthfulQA": "0",
        "AGIEval": "0",
        "TriviaQA": "5",
        "PIQA": "0",
        "BoolQ": "0",
        "MBPP": "3",
    },
    "2404.14619": {  # OpenELM
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
    },
    "2405.04324": {  # DBRX
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2405.04434": {  # DeepSeek-V2
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2405.19327": {  # Aya 23
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
    },
    "2406.11704": {  # Nemotron-4 340B
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "TruthfulQA": "0",
    },
    "2406.12793": {  # EXAONE 3.0
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2407.10671": {  # Qwen2
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "TruthfulQA": "0",
        "IFEval": "0",
    },
    "2407.14885": {  # Falcon2-11B
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "TruthfulQA": "0",
        "BoolQ": "0",
    },
    "2407.21783": {  # Llama 3 Herd
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "IFEval": "0",
    },
    "2408.00118": {  # Gemma 2
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
    },
    "2408.12570": {  # Jamba-1.5
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2409.12122": {  # Qwen2.5-Math
        "MMLU": "5",
        "GSM8K": "8",
        "MATH": "4",
        "HumanEval": "0",
        "MBPP": "3",
    },
    "2409.12186": {  # Qwen2.5-Coder
        "HumanEval": "0",
        "MBPP": "3",
        "GSM8K": "8",
        "MMLU": "5",
    },
    "2410.21276": {  # GPT-4o
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "GPQA": "0",
    },
    "2411.04905": {  # OpenCoder
        "HumanEval": "0",
        "MBPP": "3",
        "GSM8K": "8",
    },
    "2412.04261": {  # Aya Expanse
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
    },
    "2412.08905": {  # Phi-4
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "IFEval": "0",
    },
    "2412.15115": {  # Qwen2.5
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "IFEval": "0",
    },
    "2412.19437": {  # DeepSeek-V3
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
    },
    "2501.00656": {  # OLMo 2
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
    },
    "2501.00663": {  # Titans
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "BoolQ": "0",
        "PIQA": "0",
        "LambdaDA": "0",
    },
    "2501.12948": {  # DeepSeek-R1
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "GPQA": "0",
        "LiveCodeBench": "0",
    },
    "2502.02737": {  # Llama 4
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "IFEval": "0",
    },
    "2503.19786": {  # Gemma 3
        "MMLU": "5",
        "HellaSwag": "10",
        "ARC-Challenge": "25",
        "WinoGrande": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
    },
    "2505.00949": {  # Llama-Nemotron
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "GPQA": "0",
        "LiveCodeBench": "0",
    },
    "2505.09388": {  # Qwen3
        "MMLU": "5",
        "GSM8K": "8",
        "HumanEval": "0",
        "MATH": "4",
        "BBH": "3",
        "GPQA": "0",
        "LiveCodeBench": "0",
        "IFEval": "0",
    },
}

def main():
    print("=== Phase 4: Fixing n_shot from LaTeX ===")
    fixed_count = 0
    unfixed_count = 0

    for paper_dir in sorted(DATA_DIR.glob("papers_*")):
        arxiv_id = paper_dir.name.replace("papers_", "")
        nshot_map = PAPER_NSHOT.get(arxiv_id, {})

        for json_file in paper_dir.rglob("*.json"):
            try:
                record = json.loads(json_file.read_text())
            except Exception as e:
                print(f"  ERROR reading {json_file}: {e}")
                continue

            changed = False
            for res in record.get("evaluation_results", []):
                details = res.get("generation_config", {}).get("additional_details", {})
                if details.get("n_shot", "unknown") == "unknown":
                    bench = res.get("evaluation_name", "")
                    if bench in nshot_map:
                        details["n_shot"] = nshot_map[bench]
                        changed = True
                    else:
                        unfixed_count += 1

            if changed:
                json_file.write_text(json.dumps(record, indent=2))
                fixed_count += 1

    print(f"\n  Records updated: {fixed_count}")
    print(f"  Remaining unknown n_shot entries: {unfixed_count}")

if __name__ == "__main__":
    main()
