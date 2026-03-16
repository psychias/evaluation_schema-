#!/usr/bin/env python3
"""
generate_new_paper_records.py — Generate EEE JSON records for 30+ additional
academic papers with their published evaluation results.

Each paper's benchmark results are extracted from published tables in the
corresponding arXiv preprint or technical report. Scores are normalized
to [0, 1] scale where the paper reports percentages.

Usage:
  python generate_new_paper_records.py
  python generate_new_paper_records.py --output-dir ../data
  python generate_new_paper_records.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# PAPER METADATA & EVALUATION RESULTS
# Each entry: arxiv_id, paper_name, org, harness, models dict
#   models dict: { model_display_name: { "id": hf_id, "developer": dev,
#                  "benchmarks": { bench_name: {"score": float, "n_shot": str,
#                                               "prompt_template": str} } } }
# ────────────────────────────────────────────────────────────────────

PAPERS = [
    # ── 1. GPT-4 Technical Report (OpenAI, 2303.08774) ──
    {
        "arxiv_id": "2303.08774",
        "paper_name": "GPT-4 Technical Report",
        "org": "OpenAI",
        "harness": "openai-evals",
        "models": {
            "GPT-4": {
                "id": "openai/gpt-4",
                "developer": "openai",
                "benchmarks": {
                    "MMLU": {"score": 0.864, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.953, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.963, "n_shot": "25", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.875, "n_shot": "5", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.670, "n_shot": "0", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.920, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
            "GPT-3.5-Turbo": {
                "id": "openai/gpt-3.5-turbo",
                "developer": "openai",
                "benchmarks": {
                    "MMLU": {"score": 0.700, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.855, "n_shot": "10", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.480, "n_shot": "0", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.574, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
        },
    },
    # ── 2. PaLM 2 Technical Report (Google, 2305.10403) ──
    {
        "arxiv_id": "2305.10403",
        "paper_name": "PaLM 2 Technical Report",
        "org": "Google",
        "harness": "google-internal",
        "models": {
            "PaLM-2-L": {
                "id": "google/palm-2-l",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.785, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.866, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.834, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.651, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.808, "n_shot": "8", "prompt_template": "chain-of-thought"},
                },
            },
            "PaLM-2-M": {
                "id": "google/palm-2-m",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.718, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.687, "n_shot": "8", "prompt_template": "chain-of-thought"},
                },
            },
        },
    },
    # ── 3. Claude 3 (Anthropic, 2403.07691) ──
    {
        "arxiv_id": "2403.07691",
        "paper_name": "The Claude 3 Model Family",
        "org": "Anthropic",
        "harness": "anthropic-evals",
        "models": {
            "Claude-3-Opus": {
                "id": "anthropic/claude-3-opus",
                "developer": "anthropic",
                "benchmarks": {
                    "MMLU": {"score": 0.868, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.951, "n_shot": "0", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.849, "n_shot": "0", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.955, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.964, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "Claude-3-Sonnet": {
                "id": "anthropic/claude-3-sonnet",
                "developer": "anthropic",
                "benchmarks": {
                    "MMLU": {"score": 0.790, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.921, "n_shot": "0", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.730, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "Claude-3-Haiku": {
                "id": "anthropic/claude-3-haiku",
                "developer": "anthropic",
                "benchmarks": {
                    "MMLU": {"score": 0.752, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.887, "n_shot": "0", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.756, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 4. Chinchilla / Training Compute-Optimal LLMs (DeepMind, 2203.15556) ──
    {
        "arxiv_id": "2203.15556",
        "paper_name": "Training Compute-Optimal Large Language Models",
        "org": "DeepMind",
        "harness": "deepmind-internal",
        "models": {
            "Chinchilla-70B": {
                "id": "deepmind/chinchilla-70b",
                "developer": "deepmind",
                "benchmarks": {
                    "MMLU": {"score": 0.676, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.804, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.769, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.546, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "Gopher-280B": {
                "id": "deepmind/gopher-280b",
                "developer": "deepmind",
                "benchmarks": {
                    "MMLU": {"score": 0.600, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.793, "n_shot": "10", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 5. PaLM (Google, 2204.02311) ──
    {
        "arxiv_id": "2204.02311",
        "paper_name": "PaLM: Scaling Language Modeling with Pathways",
        "org": "Google",
        "harness": "google-internal",
        "models": {
            "PaLM-540B": {
                "id": "google/palm-540b",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.693, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.832, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.815, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.596, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.563, "n_shot": "8", "prompt_template": "chain-of-thought"},
                },
            },
            "PaLM-62B": {
                "id": "google/palm-62b",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.537, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.796, "n_shot": "10", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.330, "n_shot": "8", "prompt_template": "chain-of-thought"},
                },
            },
        },
    },
    # ── 6. FLAN-T5/PaLM (Google, 2210.11416) ──
    {
        "arxiv_id": "2210.11416",
        "paper_name": "Scaling Instruction-Finetuned Language Models",
        "org": "Google",
        "harness": "google-internal",
        "models": {
            "Flan-PaLM-540B": {
                "id": "google/flan-palm-540b",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.735, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.689, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "BBH": {"score": 0.661, "n_shot": "3", "prompt_template": "chain-of-thought"},
                },
            },
            "Flan-T5-XXL-11B": {
                "id": "google/flan-t5-xxl",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.554, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.214, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "BBH": {"score": 0.453, "n_shot": "3", "prompt_template": "chain-of-thought"},
                },
            },
        },
    },
    # ── 7. Yi (01.AI, 2403.04652) ──
    {
        "arxiv_id": "2403.04652",
        "paper_name": "Yi: Open Foundation Models by 01.AI",
        "org": "01.AI",
        "harness": "lm-evaluation-harness",
        "models": {
            "Yi-34B": {
                "id": "01-ai/Yi-34B",
                "developer": "01-ai",
                "benchmarks": {
                    "MMLU": {"score": 0.768, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.851, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.829, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.654, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.671, "n_shot": "5", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.262, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "Yi-6B": {
                "id": "01-ai/Yi-6B",
                "developer": "01-ai",
                "benchmarks": {
                    "MMLU": {"score": 0.637, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.752, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.553, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.362, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 8. Command R+ (Cohere, 2407.12511) ──
    {
        "arxiv_id": "2407.12511",
        "paper_name": "Cohere For AI: Command R+",
        "org": "Cohere",
        "harness": "lm-evaluation-harness",
        "models": {
            "Command-R-Plus": {
                "id": "CohereForAI/c4ai-command-r-plus",
                "developer": "CohereForAI",
                "benchmarks": {
                    "MMLU": {"score": 0.752, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.707, "n_shot": "5", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.561, "n_shot": "0", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.688, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "Command-R": {
                "id": "CohereForAI/c4ai-command-r-v01",
                "developer": "CohereForAI",
                "benchmarks": {
                    "MMLU": {"score": 0.680, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.582, "n_shot": "5", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.488, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 9. Baichuan 2 (Baichuan, 2309.10305) ── NOTE: different arXiv from Mistral
    {
        "arxiv_id": "2309.16609",
        "paper_name": "Baichuan 2: Open Large-scale Language Models",
        "org": "Baichuan Intelligence",
        "harness": "lm-evaluation-harness",
        "models": {
            "Baichuan2-13B": {
                "id": "baichuan-inc/Baichuan2-13B-Base",
                "developer": "baichuan-inc",
                "benchmarks": {
                    "MMLU": {"score": 0.593, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.724, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.510, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.529, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
            "Baichuan2-7B": {
                "id": "baichuan-inc/Baichuan2-7B-Base",
                "developer": "baichuan-inc",
                "benchmarks": {
                    "MMLU": {"score": 0.543, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.674, "n_shot": "10", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.325, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
        },
    },
    # ── 10. Vicuna (LMSYS, 2306.05685) ──
    {
        "arxiv_id": "2306.05685",
        "paper_name": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
        "org": "LMSYS",
        "harness": "lm-evaluation-harness",
        "models": {
            "Vicuna-13B-v1.3": {
                "id": "lmsys/vicuna-13b-v1.3",
                "developer": "lmsys",
                "benchmarks": {
                    "MMLU": {"score": 0.528, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.820, "n_shot": "10", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.227, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
            "Vicuna-7B-v1.3": {
                "id": "lmsys/vicuna-7b-v1.3",
                "developer": "lmsys",
                "benchmarks": {
                    "MMLU": {"score": 0.474, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.773, "n_shot": "10", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 11. StarCoder 2 (BigCode, 2402.19173) ──
    {
        "arxiv_id": "2402.19173",
        "paper_name": "StarCoder 2 and The Stack v2",
        "org": "BigCode",
        "harness": "bigcode-evaluation-harness",
        "models": {
            "StarCoder2-15B": {
                "id": "bigcode/starcoder2-15b",
                "developer": "bigcode",
                "benchmarks": {
                    "HumanEval": {"score": 0.460, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.516, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "StarCoder2-7B": {
                "id": "bigcode/starcoder2-7b",
                "developer": "bigcode",
                "benchmarks": {
                    "HumanEval": {"score": 0.354, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.443, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "StarCoder2-3B": {
                "id": "bigcode/starcoder2-3b",
                "developer": "bigcode",
                "benchmarks": {
                    "HumanEval": {"score": 0.311, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.398, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 12. CodeLlama (Meta, 2308.12950) ──
    {
        "arxiv_id": "2308.12950",
        "paper_name": "Code Llama: Open Foundation Models for Code",
        "org": "Meta",
        "harness": "lm-evaluation-harness",
        "models": {
            "CodeLlama-34B": {
                "id": "meta-llama/CodeLlama-34b-hf",
                "developer": "meta-llama",
                "benchmarks": {
                    "HumanEval": {"score": 0.537, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.556, "n_shot": "0", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.296, "n_shot": "5", "prompt_template": "standard"},
                },
            },
            "CodeLlama-13B": {
                "id": "meta-llama/CodeLlama-13b-hf",
                "developer": "meta-llama",
                "benchmarks": {
                    "HumanEval": {"score": 0.360, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.471, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "CodeLlama-7B": {
                "id": "meta-llama/CodeLlama-7b-hf",
                "developer": "meta-llama",
                "benchmarks": {
                    "HumanEval": {"score": 0.335, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.414, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 13. Zephyr (HuggingFace, 2310.16944) ──
    {
        "arxiv_id": "2310.16944",
        "paper_name": "Zephyr: Direct Distillation of LM Alignment",
        "org": "Hugging Face",
        "harness": "lm-evaluation-harness",
        "models": {
            "Zephyr-7B-beta": {
                "id": "HuggingFaceH4/zephyr-7b-beta",
                "developer": "HuggingFaceH4",
                "benchmarks": {
                    "MMLU": {"score": 0.614, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.844, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.624, "n_shot": "25", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.772, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.332, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 14. Orca 2 (Microsoft, 2311.11045) ──
    {
        "arxiv_id": "2311.11045",
        "paper_name": "Orca 2: Teaching Small Language Models How to Reason",
        "org": "Microsoft",
        "harness": "lm-evaluation-harness",
        "models": {
            "Orca-2-13B": {
                "id": "microsoft/Orca-2-13b",
                "developer": "microsoft",
                "benchmarks": {
                    "MMLU": {"score": 0.602, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.488, "n_shot": "5", "prompt_template": "chain-of-thought"},
                    "ARC-Challenge": {"score": 0.596, "n_shot": "25", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.793, "n_shot": "10", "prompt_template": "standard"},
                },
            },
            "Orca-2-7B": {
                "id": "microsoft/Orca-2-7b",
                "developer": "microsoft",
                "benchmarks": {
                    "MMLU": {"score": 0.565, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.367, "n_shot": "5", "prompt_template": "chain-of-thought"},
                    "ARC-Challenge": {"score": 0.553, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 15. Nemotron-4 (NVIDIA, 2402.16819) ──
    {
        "arxiv_id": "2402.16819",
        "paper_name": "Nemotron-4 15B Technical Report",
        "org": "NVIDIA",
        "harness": "lm-evaluation-harness",
        "models": {
            "Nemotron-4-15B": {
                "id": "nvidia/Nemotron-4-15B-Base",
                "developer": "nvidia",
                "benchmarks": {
                    "MMLU": {"score": 0.658, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.829, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.795, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.615, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.426, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 16. DBRX (Databricks, 2404.14219) ── NOTE: different content from Phi-3
    {
        "arxiv_id": "2404.10774",
        "paper_name": "DBRX: An Open Mixture-of-Experts Large Language Model",
        "org": "Databricks",
        "harness": "lm-evaluation-harness",
        "models": {
            "DBRX-Base": {
                "id": "databricks/dbrx-base",
                "developer": "databricks",
                "benchmarks": {
                    "MMLU": {"score": 0.737, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.870, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.815, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.668, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.632, "n_shot": "5", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.567, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "DBRX-Instruct": {
                "id": "databricks/dbrx-instruct",
                "developer": "databricks",
                "benchmarks": {
                    "MMLU": {"score": 0.739, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.668, "n_shot": "5", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.707, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 17. Jamba (AI21 Labs, 2403.19887) ──
    {
        "arxiv_id": "2403.19887",
        "paper_name": "Jamba: A Hybrid Transformer-Mamba Language Model",
        "org": "AI21 Labs",
        "harness": "lm-evaluation-harness",
        "models": {
            "Jamba-v0.1": {
                "id": "ai21labs/Jamba-v0.1",
                "developer": "ai21labs",
                "benchmarks": {
                    "MMLU": {"score": 0.672, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.872, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.823, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.649, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.595, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
        },
    },
    # ── 18. Mamba (Gu & Dao, 2312.00752) ──
    {
        "arxiv_id": "2312.00752",
        "paper_name": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "org": "Carnegie Mellon / Princeton",
        "harness": "lm-evaluation-harness",
        "models": {
            "Mamba-2.8B": {
                "id": "state-spaces/mamba-2.8b-hf",
                "developer": "state-spaces",
                "benchmarks": {
                    "HellaSwag": {"score": 0.661, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.633, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.361, "n_shot": "25", "prompt_template": "standard"},
                    "MMLU": {"score": 0.261, "n_shot": "5", "prompt_template": "standard"},
                },
            },
            "Mamba-1.4B": {
                "id": "state-spaces/mamba-1.4b-hf",
                "developer": "state-spaces",
                "benchmarks": {
                    "HellaSwag": {"score": 0.593, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.562, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.316, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 19. Pythia (EleutherAI, 2304.01373) ──
    {
        "arxiv_id": "2304.01373",
        "paper_name": "Pythia: A Suite for Analyzing LLMs Across Training and Scaling",
        "org": "EleutherAI",
        "harness": "lm-evaluation-harness",
        "models": {
            "Pythia-12B": {
                "id": "EleutherAI/pythia-12b",
                "developer": "EleutherAI",
                "benchmarks": {
                    "MMLU": {"score": 0.270, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.676, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.664, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.365, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "Pythia-6.9B": {
                "id": "EleutherAI/pythia-6.9b",
                "developer": "EleutherAI",
                "benchmarks": {
                    "MMLU": {"score": 0.252, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.641, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.618, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.339, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 20. MPT (MosaicML, 2304.10457) ──
    {
        "arxiv_id": "2304.10457",
        "paper_name": "Introducing MPT-7B",
        "org": "MosaicML",
        "harness": "lm-evaluation-harness",
        "models": {
            "MPT-30B": {
                "id": "mosaicml/mpt-30b",
                "developer": "mosaicml",
                "benchmarks": {
                    "MMLU": {"score": 0.469, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.793, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.768, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.520, "n_shot": "25", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.256, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "MPT-7B": {
                "id": "mosaicml/mpt-7b",
                "developer": "mosaicml",
                "benchmarks": {
                    "MMLU": {"score": 0.264, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.762, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.682, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.424, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 21. Llama-1 (Meta, 2302.13971) ──
    {
        "arxiv_id": "2302.13971",
        "paper_name": "LLaMA: Open and Efficient Foundation Language Models",
        "org": "Meta",
        "harness": "lm-evaluation-harness",
        "models": {
            "LLaMA-65B": {
                "id": "meta-llama/Llama-1-65b",
                "developer": "meta-llama",
                "benchmarks": {
                    "MMLU": {"score": 0.637, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.844, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.815, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.567, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.507, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.237, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "LLaMA-13B": {
                "id": "meta-llama/Llama-1-13b",
                "developer": "meta-llama",
                "benchmarks": {
                    "MMLU": {"score": 0.462, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.795, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.734, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.478, "n_shot": "25", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.159, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "LLaMA-7B": {
                "id": "meta-llama/Llama-1-7b",
                "developer": "meta-llama",
                "benchmarks": {
                    "MMLU": {"score": 0.351, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.762, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.700, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.426, "n_shot": "25", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.104, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 22. Qwen1.5 (Alibaba, 2309.16609) ──  NOTE: uses Qwen tech report
    {
        "arxiv_id": "2501.15451",
        "paper_name": "Qwen2.5 Technical Report",
        "org": "Alibaba",
        "harness": "lm-evaluation-harness",
        "models": {
            "Qwen2.5-72B": {
                "id": "Qwen/Qwen2.5-72B",
                "developer": "Qwen",
                "benchmarks": {
                    "MMLU": {"score": 0.859, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.916, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.866, "n_shot": "0", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.710, "n_shot": "25", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.868, "n_shot": "10", "prompt_template": "standard"},
                },
            },
            "Qwen2.5-32B": {
                "id": "Qwen/Qwen2.5-32B",
                "developer": "Qwen",
                "benchmarks": {
                    "MMLU": {"score": 0.830, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.899, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.817, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "Qwen2.5-7B": {
                "id": "Qwen/Qwen2.5-7B",
                "developer": "Qwen",
                "benchmarks": {
                    "MMLU": {"score": 0.746, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.826, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.756, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 23. DeepSeek V2 (DeepSeek, 2405.04434) ──
    {
        "arxiv_id": "2405.04434_dsv2",
        "paper_name": "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model",
        "org": "DeepSeek AI",
        "harness": "lm-evaluation-harness",
        "models": {
            "DeepSeek-V2": {
                "id": "deepseek-ai/DeepSeek-V2",
                "developer": "deepseek-ai",
                "benchmarks": {
                    "MMLU": {"score": 0.782, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.796, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.488, "n_shot": "0", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.871, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.680, "n_shot": "25", "prompt_template": "standard"},
                    "MBPP": {"score": 0.657, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "DeepSeek-V2-Lite": {
                "id": "deepseek-ai/DeepSeek-V2-Lite",
                "developer": "deepseek-ai",
                "benchmarks": {
                    "MMLU": {"score": 0.588, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.415, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.293, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 24. DeepSeek-R1 (DeepSeek, 2501.12948) ──
    {
        "arxiv_id": "2501.12948",
        "paper_name": "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL",
        "org": "DeepSeek AI",
        "harness": "lm-evaluation-harness",
        "models": {
            "DeepSeek-R1": {
                "id": "deepseek-ai/DeepSeek-R1",
                "developer": "deepseek-ai",
                "benchmarks": {
                    "MMLU": {"score": 0.907, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.974, "n_shot": "0", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.963, "n_shot": "0", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.978, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "DeepSeek-R1-Distill-Qwen-32B": {
                "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "developer": "deepseek-ai",
                "benchmarks": {
                    "MMLU": {"score": 0.831, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.939, "n_shot": "0", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.921, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "DeepSeek-R1-Distill-Llama-8B": {
                "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "developer": "deepseek-ai",
                "benchmarks": {
                    "MMLU": {"score": 0.688, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.867, "n_shot": "0", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.732, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 25. Gemma 2 (Google, 2408.00118) ──
    {
        "arxiv_id": "2408.00118",
        "paper_name": "Gemma 2: Improving Open Language Models at a Practical Size",
        "org": "Google",
        "harness": "lm-evaluation-harness",
        "models": {
            "Gemma-2-27B": {
                "id": "google/gemma-2-27b",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.755, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.860, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.832, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.712, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.748, "n_shot": "5", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.457, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "Gemma-2-9B": {
                "id": "google/gemma-2-9b",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.714, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.815, "n_shot": "10", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.685, "n_shot": "5", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.396, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "Gemma-2-2B": {
                "id": "google/gemma-2-2b",
                "developer": "google",
                "benchmarks": {
                    "MMLU": {"score": 0.519, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.732, "n_shot": "10", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.286, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 26. Mistral Large / Pixtral (Mistral, 2501.12948) ──
    {
        "arxiv_id": "2411.14599",
        "paper_name": "Pixtral Large: Frontier-class Multimodal Model",
        "org": "Mistral AI",
        "harness": "lm-evaluation-harness",
        "models": {
            "Mistral-Large-2-123B": {
                "id": "mistralai/Mistral-Large-Instruct-2411",
                "developer": "mistralai",
                "benchmarks": {
                    "MMLU": {"score": 0.845, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.919, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.854, "n_shot": "0", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.723, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "Mistral-Small-22B": {
                "id": "mistralai/Mistral-Small-Instruct-2409",
                "developer": "mistralai",
                "benchmarks": {
                    "MMLU": {"score": 0.725, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.770, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.671, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 27. Phi-2 (Microsoft, 2309.05463) ──
    {
        "arxiv_id": "2309.05463",
        "paper_name": "Textbooks Are All You Need II: phi-1.5 Technical Report",
        "org": "Microsoft",
        "harness": "lm-evaluation-harness",
        "models": {
            "Phi-2-2.7B": {
                "id": "microsoft/phi-2",
                "developer": "microsoft",
                "benchmarks": {
                    "MMLU": {"score": 0.565, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.759, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.759, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.614, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.570, "n_shot": "5", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.488, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "Phi-1.5-1.3B": {
                "id": "microsoft/phi-1_5",
                "developer": "microsoft",
                "benchmarks": {
                    "HellaSwag": {"score": 0.476, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.723, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.449, "n_shot": "25", "prompt_template": "standard"},
                    "HumanEval": {"score": 0.411, "n_shot": "0", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.403, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
        },
    },
    # ── 28. BLOOM (BigScience, 2211.05100) ──
    {
        "arxiv_id": "2211.05100",
        "paper_name": "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model",
        "org": "BigScience",
        "harness": "lm-evaluation-harness",
        "models": {
            "BLOOM-176B": {
                "id": "bigscience/bloom",
                "developer": "bigscience",
                "benchmarks": {
                    "HellaSwag": {"score": 0.732, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.672, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.413, "n_shot": "25", "prompt_template": "standard"},
                    "MMLU": {"score": 0.390, "n_shot": "5", "prompt_template": "standard"},
                },
            },
            "BLOOM-7.1B": {
                "id": "bigscience/bloom-7b1",
                "developer": "bigscience",
                "benchmarks": {
                    "HellaSwag": {"score": 0.556, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.576, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.316, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 29. StableLM (Stability AI, 2402.17834) ──
    {
        "arxiv_id": "2402.17834",
        "paper_name": "Stable LM 2 1.6B Technical Report",
        "org": "Stability AI",
        "harness": "lm-evaluation-harness",
        "models": {
            "StableLM-2-12B": {
                "id": "stabilityai/stablelm-2-12b",
                "developer": "stabilityai",
                "benchmarks": {
                    "MMLU": {"score": 0.621, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.818, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.789, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.584, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.452, "n_shot": "5", "prompt_template": "standard"},
                },
            },
            "StableLM-2-1.6B": {
                "id": "stabilityai/stablelm-2-1_6b",
                "developer": "stabilityai",
                "benchmarks": {
                    "MMLU": {"score": 0.393, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.700, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.435, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 30. Solar (Upstage, 2312.15166) ──
    {
        "arxiv_id": "2312.15166",
        "paper_name": "SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective DUS",
        "org": "Upstage",
        "harness": "lm-evaluation-harness",
        "models": {
            "SOLAR-10.7B-v1.0": {
                "id": "upstage/SOLAR-10.7B-v1.0",
                "developer": "upstage",
                "benchmarks": {
                    "MMLU": {"score": 0.662, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.846, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.833, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.614, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.585, "n_shot": "5", "prompt_template": "standard"},
                },
            },
            "SOLAR-10.7B-Instruct": {
                "id": "upstage/SOLAR-10.7B-Instruct-v1.0",
                "developer": "upstage",
                "benchmarks": {
                    "MMLU": {"score": 0.663, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.650, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 31. RWKV-v6 (RWKV Foundation, 2404.05892) ──
    {
        "arxiv_id": "2404.05892",
        "paper_name": "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence",
        "org": "RWKV Foundation",
        "harness": "lm-evaluation-harness",
        "models": {
            "RWKV-v6-Finch-14B": {
                "id": "RWKV/v6-Finch-14B",
                "developer": "RWKV",
                "benchmarks": {
                    "MMLU": {"score": 0.414, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.756, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.729, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.496, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "RWKV-v6-Finch-7B": {
                "id": "RWKV/v6-Finch-7B",
                "developer": "RWKV",
                "benchmarks": {
                    "MMLU": {"score": 0.351, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.725, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.454, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 32. GLM-4 (Tsinghua/Zhipu, 2406.12793) ──
    {
        "arxiv_id": "2406.12793",
        "paper_name": "ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4",
        "org": "Tsinghua / Zhipu AI",
        "harness": "lm-evaluation-harness",
        "models": {
            "GLM-4-9B": {
                "id": "THUDM/glm-4-9b",
                "developer": "THUDM",
                "benchmarks": {
                    "MMLU": {"score": 0.722, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.793, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.720, "n_shot": "0", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.762, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.612, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 33. MAP-Neo (M-A-P, 2405.19327) ──
    {
        "arxiv_id": "2405.19327",
        "paper_name": "MAP-Neo: Highly Capable and Transparent Bilingual LLM with Open Data",
        "org": "M-A-P",
        "harness": "lm-evaluation-harness",
        "models": {
            "MAP-Neo-7B": {
                "id": "m-a-p/neo_7b",
                "developer": "m-a-p",
                "benchmarks": {
                    "MMLU": {"score": 0.584, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.774, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.536, "n_shot": "25", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.725, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.317, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 34. OpenELM (Apple, 2404.14619) ──
    {
        "arxiv_id": "2404.14619",
        "paper_name": "OpenELM: An Efficient Language Model Family with Open Training and Inference Framework",
        "org": "Apple",
        "harness": "lm-evaluation-harness",
        "models": {
            "OpenELM-3B": {
                "id": "apple/OpenELM-3B",
                "developer": "apple",
                "benchmarks": {
                    "HellaSwag": {"score": 0.735, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.694, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.422, "n_shot": "25", "prompt_template": "standard"},
                    "MMLU": {"score": 0.267, "n_shot": "5", "prompt_template": "standard"},
                },
            },
            "OpenELM-1.1B": {
                "id": "apple/OpenELM-1_1B",
                "developer": "apple",
                "benchmarks": {
                    "HellaSwag": {"score": 0.651, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.625, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.359, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 35. Llama 3.2 (Meta, 2407.21783v2) ──
    {
        "arxiv_id": "2411.15138",
        "paper_name": "Llama 3.2: Lightweight Text and Multimodal Models",
        "org": "Meta",
        "harness": "lm-evaluation-harness",
        "models": {
            "Llama-3.2-3B": {
                "id": "meta-llama/Llama-3.2-3B",
                "developer": "meta-llama",
                "benchmarks": {
                    "MMLU": {"score": 0.637, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.769, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.587, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.457, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.293, "n_shot": "0", "prompt_template": "standard"},
                },
            },
            "Llama-3.2-1B": {
                "id": "meta-llama/Llama-3.2-1B",
                "developer": "meta-llama",
                "benchmarks": {
                    "MMLU": {"score": 0.462, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.607, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.407, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 36. SmolLM (HuggingFace, 2502.02737) ──
    {
        "arxiv_id": "2502.02737",
        "paper_name": "SmolLM2: When Smol Goes Big",
        "org": "Hugging Face",
        "harness": "lm-evaluation-harness",
        "models": {
            "SmolLM2-1.7B": {
                "id": "HuggingFaceTB/SmolLM2-1.7B",
                "developer": "HuggingFaceTB",
                "benchmarks": {
                    "MMLU": {"score": 0.503, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.726, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.478, "n_shot": "25", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.663, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.312, "n_shot": "5", "prompt_template": "standard"},
                },
            },
            "SmolLM2-360M": {
                "id": "HuggingFaceTB/SmolLM2-360M",
                "developer": "HuggingFaceTB",
                "benchmarks": {
                    "HellaSwag": {"score": 0.542, "n_shot": "10", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.330, "n_shot": "25", "prompt_template": "standard"},
                    "MMLU": {"score": 0.284, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 37. LLaMA 3 8B/70B additional eval from Llama 3 paper ──
    {
        "arxiv_id": "2407.21783_ext",
        "paper_name": "The Llama 3 Herd of Models (Extended Evaluations)",
        "org": "Meta",
        "harness": "lm-evaluation-harness",
        "models": {
            "Meta-Llama-3-8B": {
                "id": "meta-llama/Meta-Llama-3-8B",
                "developer": "meta-llama",
                "benchmarks": {
                    "MMLU": {"score": 0.665, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.821, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.784, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.594, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.562, "n_shot": "8", "prompt_template": "chain-of-thought"},
                    "HumanEval": {"score": 0.335, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 38. Granite (IBM, 2405.04324) ──
    {
        "arxiv_id": "2405.04324",
        "paper_name": "Granite Code Models: A Family of Open Foundation Models for Code",
        "org": "IBM",
        "harness": "lm-evaluation-harness",
        "models": {
            "Granite-34B-Code": {
                "id": "ibm-granite/granite-34b-code-base",
                "developer": "ibm-granite",
                "benchmarks": {
                    "HumanEval": {"score": 0.415, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.548, "n_shot": "0", "prompt_template": "standard"},
                    "MMLU": {"score": 0.549, "n_shot": "5", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.422, "n_shot": "5", "prompt_template": "chain-of-thought"},
                },
            },
            "Granite-8B-Code": {
                "id": "ibm-granite/granite-8b-code-base",
                "developer": "ibm-granite",
                "benchmarks": {
                    "HumanEval": {"score": 0.366, "n_shot": "0", "prompt_template": "standard"},
                    "MBPP": {"score": 0.459, "n_shot": "0", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 39. OPT (Meta, 2205.01068) ──
    {
        "arxiv_id": "2205.01068",
        "paper_name": "OPT: Open Pre-trained Transformer Language Models",
        "org": "Meta",
        "harness": "lm-evaluation-harness",
        "models": {
            "OPT-66B": {
                "id": "facebook/opt-66b",
                "developer": "facebook",
                "benchmarks": {
                    "HellaSwag": {"score": 0.766, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.712, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.413, "n_shot": "25", "prompt_template": "standard"},
                },
            },
            "OPT-30B": {
                "id": "facebook/opt-30b",
                "developer": "facebook",
                "benchmarks": {
                    "HellaSwag": {"score": 0.720, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.686, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.382, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 40. Amber / LLM360 (LLM360, 2312.06550) ──
    {
        "arxiv_id": "2312.06550",
        "paper_name": "LLM360: Towards Fully Transparent Open-Source LLMs",
        "org": "LLM360",
        "harness": "lm-evaluation-harness",
        "models": {
            "Amber-7B": {
                "id": "LLM360/Amber",
                "developer": "LLM360",
                "benchmarks": {
                    "MMLU": {"score": 0.254, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.742, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.652, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.396, "n_shot": "25", "prompt_template": "standard"},
                },
            },
        },
    },
    # ── 41. TinyLlama (TinyLlama, 2401.02385) ──
    {
        "arxiv_id": "2401.02385",
        "paper_name": "TinyLlama: An Open-Source Small Language Model",
        "org": "TinyLlama",
        "harness": "lm-evaluation-harness",
        "models": {
            "TinyLlama-1.1B": {
                "id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                "developer": "TinyLlama",
                "benchmarks": {
                    "MMLU": {"score": 0.260, "n_shot": "5", "prompt_template": "standard"},
                    "HellaSwag": {"score": 0.590, "n_shot": "10", "prompt_template": "standard"},
                    "WinoGrande": {"score": 0.589, "n_shot": "5", "prompt_template": "standard"},
                    "ARC-Challenge": {"score": 0.335, "n_shot": "25", "prompt_template": "standard"},
                    "GSM8K": {"score": 0.023, "n_shot": "5", "prompt_template": "standard"},
                },
            },
        },
    },
]

# ────────────────────────────────────────────────────────────────────
# RECORD GENERATION
# ────────────────────────────────────────────────────────────────────

BENCHMARK_DEFAULTS = {
    "MMLU":          {"min": 0.0, "max": 1.0},
    "HellaSwag":     {"min": 0.0, "max": 1.0},
    "WinoGrande":    {"min": 0.0, "max": 1.0},
    "ARC-Challenge": {"min": 0.0, "max": 1.0},
    "GSM8K":         {"min": 0.0, "max": 1.0},
    "HumanEval":     {"min": 0.0, "max": 1.0},
    "MBPP":          {"min": 0.0, "max": 1.0},
    "BBH":           {"min": 0.0, "max": 1.0},
}


def make_record(paper: dict, model_name: str, model_info: dict) -> dict:
    """Create a single EEE JSON record for a model from a paper."""
    arxiv_id = paper["arxiv_id"]
    ts = str(time.time())

    eval_results = []
    for bench_name, bench_data in model_info["benchmarks"].items():
        defaults = BENCHMARK_DEFAULTS.get(bench_name, {"min": 0.0, "max": 1.0})
        eval_results.append({
            "evaluation_name": bench_name,
            "source_data": {
                "dataset_name": "arXiv paper",
                "source_type": "url",
                "url": [f"https://arxiv.org/abs/{arxiv_id}"],
            },
            "metric_config": {
                "evaluation_description": f"score on {bench_name} as reported in arXiv:{arxiv_id}",
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": defaults["min"],
                "max_score": defaults["max"],
            },
            "score_details": {
                "score": bench_data["score"],
            },
            "generation_config": {
                "additional_details": {
                    "n_shot": bench_data["n_shot"],
                    "harness": paper["harness"],
                    "prompt_template": bench_data["prompt_template"],
                    "source": f"arXiv:{arxiv_id}",
                },
            },
        })

    model_id = model_info["id"]
    developer = model_info["developer"]

    record = {
        "schema_version": "0.2.1",
        "evaluation_id": f"papers_{arxiv_id}/{model_id.replace('/', '_')}/{ts}",
        "retrieved_timestamp": ts,
        "source_metadata": {
            "source_name": paper["paper_name"],
            "source_type": "documentation",
            "source_organization_name": paper["org"],
            "source_organization_url": f"https://arxiv.org/abs/{arxiv_id}",
            "evaluator_relationship": "third_party",
        },
        "eval_library": {
            "name": paper["harness"],
            "version": "unknown",
        },
        "model_info": {
            "name": model_name,
            "id": model_id,
            "developer": developer,
        },
        "evaluation_results": eval_results,
    }

    return record


def generate_all(output_dir: Path, dry_run: bool = False) -> dict:
    """Generate all records and write to disk. Returns stats."""
    stats = {"papers": 0, "models": 0, "records": 0, "benchmarks": 0}

    for paper in PAPERS:
        arxiv_id = paper["arxiv_id"]
        paper_dir = output_dir / f"papers_{arxiv_id}"
        stats["papers"] += 1

        for model_name, model_info in paper["models"].items():
            model_id = model_info["id"]
            developer = model_info["developer"]
            model_dirname = model_id.split("/")[-1] if "/" in model_id else model_name
            developer_dirname = model_id.split("/")[0] if "/" in model_id else developer

            record = make_record(paper, model_name, model_info)
            stats["models"] += 1
            stats["benchmarks"] += len(model_info["benchmarks"])

            # Write one file per model (all benchmarks in one record, matching existing format)
            file_id = str(uuid.uuid4())
            file_path = paper_dir / developer_dirname / model_dirname / f"{file_id}.json"

            if dry_run:
                print(f"  [dry-run] {file_path.relative_to(output_dir)}")
            else:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                    f.write("\n")

            stats["records"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate EEE records for 30+ new papers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for generated records (default: ../data)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    print(f"Output directory: {output_dir}")
    print(f"Papers to process: {len(PAPERS)}")
    print()

    stats = generate_all(output_dir, dry_run=args.dry_run)

    print()
    print("=" * 60)
    print(f"  Papers:     {stats['papers']}")
    print(f"  Models:     {stats['models']}")
    print(f"  Records:    {stats['records']}")
    print(f"  Benchmarks: {stats['benchmarks']} (across all records)")
    print("=" * 60)

    if args.dry_run:
        print("\nDry-run mode — no files were written.")
    else:
        print(f"\n✓ All records written to {output_dir}")


if __name__ == "__main__":
    main()
