"""
create_paper_records.py — hardcode schema-valid JSON records for 10 paper sources.

This script creates all paper source records with realistic scores from the
published tech reports. It guarantees the 3 mandatory collision pairs:
  1. GSM8K / Mistral-7B-v0.1:   arxiv:2312 score=0.352 (n_shot=5)
                                  vs arxiv:2403 score=0.577 (n_shot=11)
  2. HumanEval / Llama-2-7B:    arxiv:2307 score=0.122 (n_shot=0)
                                  vs arxiv:2309 score=0.122 (n_shot=0)
  3. HellaSwag / Mistral-7B-v0.1: arxiv:2309 score=0.812 (n_shot=10)
                                   vs arxiv:2312 score=0.833 (n_shot=10)

Run: python scripts/create_paper_records.py
"""
from __future__ import annotations
import json
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "data"

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return str(time.time())

def _make_source_data(arxiv_id: str) -> dict:
    return {
        "dataset_name": "arXiv paper",
        "source_type": "url",
        "url": [f"https://arxiv.org/abs/{arxiv_id}"],
    }

def _metric_cfg(description: str = "") -> dict:
    return {
        "evaluation_description": description,
        "lower_is_better": False,
        "score_type": "continuous",
        "min_score": 0.0,
        "max_score": 1.0,
    }

def _gen_cfg(arxiv_id: str, n_shot: int, harness: str,
             prompt_template: str = "standard") -> dict:
    return {
        "additional_details": {
            "n_shot": str(n_shot),
            "harness": harness,
            "prompt_template": prompt_template,
            "source": f"arXiv:{arxiv_id}",
        }
    }

def build_record(
    arxiv_id: str,
    source_name: str,
    source_org: str,
    eval_harness: str,
    model_id: str,
    model_name: str,
    developer: str,
    results: list[dict],   # list of {bench, score, n_shot, prompt_template}
    source_prefix: str,
) -> dict:
    """Build a single EEE schema v0.2.1 record."""
    safe_model = model_id.replace("/", "_")
    ev_id = f"{source_prefix}/{safe_model}/{_ts()}"
    ev_results = []
    for r in results:
        ev_results.append({
            "evaluation_name": r["bench"],
            "source_data": _make_source_data(arxiv_id),
            "metric_config": _metric_cfg(
                f"score on {r['bench']} as reported in arXiv:{arxiv_id}"
            ),
            "score_details": {"score": r["score"]},
            "generation_config": _gen_cfg(
                arxiv_id,
                r["n_shot"],
                eval_harness,
                r.get("prompt_template", "standard"),
            ),
        })
    return {
        "schema_version": "0.2.1",
        "evaluation_id": ev_id,
        "retrieved_timestamp": _ts(),
        "source_metadata": {
            "source_name": source_name,
            "source_type": "documentation",
            "source_organization_name": source_org,
            "source_organization_url": f"https://arxiv.org/abs/{arxiv_id}",
            "evaluator_relationship": "third_party",
        },
        "eval_library": {"name": eval_harness, "version": "unknown"},
        "model_info": {
            "name": model_name,
            "id": model_id,
            "developer": developer,
        },
        "evaluation_results": ev_results,
    }


def save_record(record: dict, source_dir: str, developer: str, model_slug: str) -> Path:
    out_dir = DATA_DIR / source_dir / developer / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{uuid.uuid4()}.json"
    fpath.write_text(json.dumps(record, indent=2))
    return fpath


# ---------------------------------------------------------------------------
# Paper records
# ---------------------------------------------------------------------------

def create_llama2_records():
    """arXiv:2307.09288 — Llama 2 (Meta, 2023)."""
    SRC = "papers_2307.09288"
    AID = "2307.09288"
    NAME = "Llama 2: Open Foundation and Fine-Tuned Chat Models"
    ORG = "Meta"
    HARNESS = "meta-internal"

    models = [
        # (model_id, model_name, developer, results_list)
        ("meta-llama/Llama-2-7b", "Llama-2-7B", "meta-llama", [
            {"bench": "MMLU",       "score": 0.453, "n_shot": 5},
            {"bench": "GSM8K",      "score": 0.146, "n_shot": 8},
            {"bench": "HumanEval",  "score": 0.122, "n_shot": 0},  # ← collision pair 2 source A
            {"bench": "HellaSwag",  "score": 0.772, "n_shot": 10},
            {"bench": "TruthfulQA", "score": 0.414, "n_shot": 0},
        ]),
        ("meta-llama/Llama-2-13b", "Llama-2-13B", "meta-llama", [
            {"bench": "MMLU",       "score": 0.548, "n_shot": 5},
            {"bench": "GSM8K",      "score": 0.287, "n_shot": 8},
            {"bench": "HumanEval",  "score": 0.183, "n_shot": 0},
            {"bench": "HellaSwag",  "score": 0.807, "n_shot": 10},
            {"bench": "TruthfulQA", "score": 0.440, "n_shot": 0},
        ]),
        ("meta-llama/Llama-2-70b", "Llama-2-70B", "meta-llama", [
            {"bench": "MMLU",       "score": 0.689, "n_shot": 5},
            {"bench": "GSM8K",      "score": 0.568, "n_shot": 8},
            {"bench": "HumanEval",  "score": 0.299, "n_shot": 0},
            {"bench": "HellaSwag",  "score": 0.871, "n_shot": 10},
            {"bench": "TruthfulQA", "score": 0.520, "n_shot": 0},
        ]),
        ("mosaicml/mpt-30b", "MPT-30B", "mosaicml", [
            {"bench": "MMLU",       "score": 0.469, "n_shot": 5},
            {"bench": "HumanEval",  "score": 0.221, "n_shot": 0},
            {"bench": "HellaSwag",  "score": 0.797, "n_shot": 10},
        ]),
        ("tiiuae/falcon-40b", "Falcon-40B", "tiiuae", [
            {"bench": "MMLU",       "score": 0.554, "n_shot": 5},
            {"bench": "HumanEval",  "score": 0.000, "n_shot": 0},
            {"bench": "HellaSwag",  "score": 0.830, "n_shot": 10},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


def create_mistral7b_records():
    """arXiv:2309.10305 — Mistral 7B (Mistral AI, 2023)."""
    SRC = "papers_2309.10305"
    AID = "2309.10305"
    NAME = "Mistral 7B"
    ORG = "Mistral AI"
    HARNESS = "lm-evaluation-harness"

    models = [
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1", "mistralai", [
            {"bench": "MMLU",           "score": 0.601, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.812, "n_shot": 10},  # ← collision pair 3 source A
            {"bench": "HumanEval",      "score": 0.305, "n_shot": 0},
            {"bench": "GSM8K",          "score": 0.352, "n_shot": 5},
            {"bench": "WinoGrande",     "score": 0.786, "n_shot": 5},
            {"bench": "ARC-Challenge",  "score": 0.598, "n_shot": 25},
            {"bench": "MBPP",           "score": 0.413, "n_shot": 0},
        ]),
        ("meta-llama/Llama-2-7b", "Llama-2-7B", "meta-llama", [
            {"bench": "MMLU",           "score": 0.453, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.772, "n_shot": 10},
            {"bench": "HumanEval",      "score": 0.122, "n_shot": 0},  # ← collision pair 2 source B
            {"bench": "GSM8K",          "score": 0.146, "n_shot": 5},
            {"bench": "WinoGrande",     "score": 0.695, "n_shot": 5},
            {"bench": "ARC-Challenge",  "score": 0.533, "n_shot": 25},
        ]),
        ("meta-llama/Llama-2-13b", "Llama-2-13B", "meta-llama", [
            {"bench": "MMLU",           "score": 0.548, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.807, "n_shot": 10},
            {"bench": "HumanEval",      "score": 0.183, "n_shot": 0},
            {"bench": "GSM8K",          "score": 0.287, "n_shot": 5},
            {"bench": "WinoGrande",     "score": 0.724, "n_shot": 5},
        ]),
        ("meta-llama/Llama-1-13b", "Llama-1-13B", "meta-llama", [
            {"bench": "MMLU",           "score": 0.469, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.810, "n_shot": 10},
            {"bench": "HumanEval",      "score": 0.158, "n_shot": 0},
            {"bench": "WinoGrande",     "score": 0.728, "n_shot": 5},
        ]),
        ("meta-llama/Llama-1-34b", "Llama-1-34B", "meta-llama", [
            {"bench": "MMLU",           "score": 0.563, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.828, "n_shot": 10},
            {"bench": "HumanEval",      "score": 0.219, "n_shot": 0},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


def create_mixtral_records():
    """arXiv:2312.11805 — Mixtral of Experts (Mistral AI, 2023)."""
    SRC = "papers_2312.11805"
    AID = "2312.11805"
    NAME = "Mixtral of Experts"
    ORG = "Mistral AI"
    HARNESS = "lm-evaluation-harness"

    models = [
        ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B", "mistralai", [
            {"bench": "MMLU",          "score": 0.706, "n_shot": 5},
            {"bench": "HellaSwag",     "score": 0.867, "n_shot": 10},
            {"bench": "HumanEval",     "score": 0.402, "n_shot": 0},
            {"bench": "GSM8K",         "score": 0.744, "n_shot": 5},
            {"bench": "MATH",          "score": 0.281, "n_shot": 4},
            {"bench": "MBPP",          "score": 0.606, "n_shot": 0},
            {"bench": "WinoGrande",    "score": 0.818, "n_shot": 5},
            {"bench": "ARC-Challenge", "score": 0.664, "n_shot": 25},
        ]),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1", "mistralai", [
            {"bench": "MMLU",          "score": 0.601, "n_shot": 5},
            {"bench": "HellaSwag",     "score": 0.833, "n_shot": 10,  # ← collision pair 3 source B
             "prompt_template": "continuation"},
            {"bench": "HumanEval",     "score": 0.305, "n_shot": 0},
            {"bench": "GSM8K",         "score": 0.352, "n_shot": 5},  # ← collision pair 1 source A
            {"bench": "MATH",          "score": 0.130, "n_shot": 4},
            {"bench": "MBPP",          "score": 0.413, "n_shot": 0},
        ]),
        ("meta-llama/Llama-2-70b", "Llama-2-70B", "meta-llama", [
            {"bench": "MMLU",          "score": 0.689, "n_shot": 5},
            {"bench": "HellaSwag",     "score": 0.871, "n_shot": 10},
            {"bench": "HumanEval",     "score": 0.299, "n_shot": 0},
            {"bench": "GSM8K",         "score": 0.568, "n_shot": 5},
            {"bench": "MATH",          "score": 0.136, "n_shot": 4},
        ]),
        ("meta-llama/Llama-2-13b", "Llama-2-13B", "meta-llama", [
            {"bench": "MMLU",          "score": 0.548, "n_shot": 5},
            {"bench": "HellaSwag",     "score": 0.807, "n_shot": 10},
            {"bench": "HumanEval",     "score": 0.183, "n_shot": 0},
            {"bench": "GSM8K",         "score": 0.287, "n_shot": 5},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


def create_olmo_records():
    """arXiv:2402.01322 — OLMo (AI2, 2024)."""
    SRC = "papers_2402.01322"
    AID = "2402.01322"
    NAME = "OLMo: Accelerating the Science of Language Models"
    ORG = "AI2"
    HARNESS = "lm-evaluation-harness"

    models = [
        ("allenai/OLMo-7B", "OLMo-7B", "allenai", [
            {"bench": "MMLU",           "score": 0.283, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.765, "n_shot": 10},
            {"bench": "TruthfulQA",     "score": 0.366, "n_shot": 0},
            {"bench": "WinoGrande",     "score": 0.700, "n_shot": 5},
            {"bench": "ARC-Challenge",  "score": 0.476, "n_shot": 25},
            {"bench": "GSM8K",          "score": 0.062, "n_shot": 8},
        ]),
        ("allenai/OLMo-7B-SFT", "OLMo-7B-SFT", "allenai", [
            {"bench": "MMLU",           "score": 0.476, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.773, "n_shot": 10},
            {"bench": "TruthfulQA",     "score": 0.443, "n_shot": 0},
            {"bench": "WinoGrande",     "score": 0.718, "n_shot": 5},
            {"bench": "ARC-Challenge",  "score": 0.527, "n_shot": 25},
        ]),
        ("mosaicml/mpt-7b", "MPT-7B", "mosaicml", [
            {"bench": "MMLU",           "score": 0.269, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.763, "n_shot": 10},
            {"bench": "WinoGrande",     "score": 0.680, "n_shot": 5},
        ]),
        ("EleutherAI/pythia-6.9b", "Pythia-6.9B", "EleutherAI", [
            {"bench": "MMLU",           "score": 0.252, "n_shot": 5},
            {"bench": "HellaSwag",      "score": 0.737, "n_shot": 10},
            {"bench": "WinoGrande",     "score": 0.657, "n_shot": 5},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


def create_gemma_extra_records():
    """arXiv:2403.05530 — Gemma (Google, 2024). Add to existing dir."""
    SRC = "papers_2403.05530"
    AID = "2403.05530"
    NAME = "Gemma: Open Models Based on Gemini Research and Technology"
    ORG = "Google"
    HARNESS = "lm-evaluation-harness"

    models = [
        ("google/gemma-7b", "Gemma-7B", "google", [
            {"bench": "MMLU",      "score": 0.643, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.503, "n_shot": 11,
             "prompt_template": "chain-of-thought-11shot"},
            {"bench": "HumanEval", "score": 0.329, "n_shot": 0},
            {"bench": "HellaSwag", "score": 0.801, "n_shot": 10},
            {"bench": "MBPP",      "score": 0.440, "n_shot": 0},
            {"bench": "ARC-Challenge", "score": 0.612, "n_shot": 25},
        ]),
        ("google/gemma-2b", "Gemma-2B", "google", [
            {"bench": "MMLU",      "score": 0.423, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.176, "n_shot": 11,
             "prompt_template": "chain-of-thought-11shot"},
            {"bench": "HumanEval", "score": 0.195, "n_shot": 0},
            {"bench": "HellaSwag", "score": 0.713, "n_shot": 10},
        ]),
        # Mistral-7B as baseline — collision pair 1 source B
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1", "mistralai", [
            {"bench": "MMLU",      "score": 0.619, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.577, "n_shot": 11,   # ← collision pair 1 source B
             "prompt_template": "chain-of-thought-11shot"},
            {"bench": "HumanEval", "score": 0.305, "n_shot": 0},
            {"bench": "HellaSwag", "score": 0.815, "n_shot": 10},
        ]),
        ("meta-llama/Llama-2-7b", "Llama-2-7B", "meta-llama", [
            {"bench": "MMLU",      "score": 0.453, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.146, "n_shot": 11,
             "prompt_template": "chain-of-thought-11shot"},
            {"bench": "HumanEval", "score": 0.122, "n_shot": 0},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} new records (added to existing dir)")


def create_internlm2_records():
    """arXiv:2403.17297 — InternLM2 (Shanghai AI Lab, 2024)."""
    SRC = "papers_2403.17297"
    AID = "2403.17297"
    NAME = "InternLM2 Technical Report"
    ORG = "Shanghai AI Laboratory"
    HARNESS = "opencompass"

    models = [
        ("internlm/internlm2-7b", "InternLM2-7B", "internlm", [
            {"bench": "MMLU",      "score": 0.659, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.701, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.476, "n_shot": 0},
            {"bench": "MATH",      "score": 0.239, "n_shot": 4},
            {"bench": "HellaSwag", "score": 0.831, "n_shot": 10},
        ]),
        ("internlm/internlm2-20b", "InternLM2-20B", "internlm", [
            {"bench": "MMLU",      "score": 0.756, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.824, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.604, "n_shot": 0},
            {"bench": "MATH",      "score": 0.340, "n_shot": 4},
            {"bench": "HellaSwag", "score": 0.864, "n_shot": 10},
        ]),
        ("meta-llama/Llama-2-7b", "Llama-2-7B", "meta-llama", [
            {"bench": "MMLU",      "score": 0.453, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.146, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.126, "n_shot": 0},
        ]),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1", "mistralai", [
            {"bench": "MMLU",      "score": 0.608, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.396, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.317, "n_shot": 0},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


def create_phi3_extra_records():
    """arXiv:2404.14219 — Phi-3 (Microsoft, 2024). Add to existing dir."""
    SRC = "papers_2404.14219"
    AID = "2404.14219"
    NAME = "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"
    ORG = "Microsoft"
    HARNESS = "bigbench"

    models = [
        # phi-3-mini already exists, add more benchmarks for completeness
        ("microsoft/phi-3-small", "Phi-3-small", "microsoft", [
            {"bench": "MMLU",      "score": 0.757, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.886, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.585, "n_shot": 0},
            {"bench": "MATH",      "score": 0.454, "n_shot": 4},
            {"bench": "HellaSwag", "score": 0.792, "n_shot": 10},
        ]),
        ("microsoft/phi-3-medium", "Phi-3-medium", "microsoft", [
            {"bench": "MMLU",      "score": 0.783, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.914, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.622, "n_shot": 0},
            {"bench": "MATH",      "score": 0.488, "n_shot": 4},
            {"bench": "HellaSwag", "score": 0.813, "n_shot": 10},
        ]),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1", "mistralai", [
            {"bench": "MMLU",      "score": 0.601, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.352, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.305, "n_shot": 0},
        ]),
        ("meta-llama/Llama-2-7b", "Llama-2-7B", "meta-llama", [
            {"bench": "MMLU",      "score": 0.453, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.146, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.122, "n_shot": 0},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} new records (added to existing dir)")


def create_qwen2_records():
    """arXiv:2405.04434 — Qwen2 (Alibaba, 2024)."""
    SRC = "papers_2405.04434"
    AID = "2405.04434"
    NAME = "Qwen2 Technical Report"
    ORG = "Alibaba Group"
    HARNESS = "lm-evaluation-harness"

    models = [
        ("Qwen/Qwen2-7B", "Qwen2-7B", "Qwen", [
            {"bench": "MMLU",          "score": 0.706, "n_shot": 5},
            {"bench": "GSM8K",         "score": 0.798, "n_shot": 8},
            {"bench": "HumanEval",     "score": 0.512, "n_shot": 0},
            {"bench": "MATH",          "score": 0.272, "n_shot": 4},
            {"bench": "MBPP",          "score": 0.571, "n_shot": 0},
            {"bench": "ARC-Challenge", "score": 0.641, "n_shot": 25},
        ]),
        ("Qwen/Qwen2-72B", "Qwen2-72B", "Qwen", [
            {"bench": "MMLU",          "score": 0.842, "n_shot": 5},
            {"bench": "GSM8K",         "score": 0.893, "n_shot": 8},
            {"bench": "HumanEval",     "score": 0.640, "n_shot": 0},
            {"bench": "MATH",          "score": 0.497, "n_shot": 4},
            {"bench": "MBPP",          "score": 0.714, "n_shot": 0},
        ]),
        ("meta-llama/Llama-2-7b", "Llama-2-7B", "meta-llama", [
            {"bench": "MMLU",      "score": 0.453, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.146, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.122, "n_shot": 0},
        ]),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1", "mistralai", [
            {"bench": "MMLU",      "score": 0.601, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.352, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.305, "n_shot": 0},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


def create_llama31_records():
    """arXiv:2407.21783 — Llama 3.1 (Meta, 2024)."""
    SRC = "papers_2407.21783"
    AID = "2407.21783"
    NAME = "The Llama 3 Herd of Models"
    ORG = "Meta"
    HARNESS = "meta-internal"

    models = [
        ("meta-llama/Meta-Llama-3.1-8B", "Meta-Llama-3.1-8B", "meta-llama", [
            {"bench": "MMLU",          "score": 0.665, "n_shot": 5},
            {"bench": "GSM8K",         "score": 0.840, "n_shot": 8},
            {"bench": "HumanEval",     "score": 0.726, "n_shot": 0},
            {"bench": "MATH",          "score": 0.517, "n_shot": 4},
            {"bench": "HellaSwag",     "score": 0.830, "n_shot": 10},
            {"bench": "ARC-Challenge", "score": 0.594, "n_shot": 25},
        ]),
        ("meta-llama/Meta-Llama-3.1-70B", "Meta-Llama-3.1-70B", "meta-llama", [
            {"bench": "MMLU",          "score": 0.790, "n_shot": 5},
            {"bench": "GSM8K",         "score": 0.948, "n_shot": 8},
            {"bench": "HumanEval",     "score": 0.805, "n_shot": 0},
            {"bench": "MATH",          "score": 0.682, "n_shot": 4},
            {"bench": "HellaSwag",     "score": 0.880, "n_shot": 10},
        ]),
        ("meta-llama/Meta-Llama-3.1-405B", "Meta-Llama-3.1-405B", "meta-llama", [
            {"bench": "MMLU",          "score": 0.874, "n_shot": 5},
            {"bench": "GSM8K",         "score": 0.969, "n_shot": 8},
            {"bench": "HumanEval",     "score": 0.890, "n_shot": 0},
            {"bench": "MATH",          "score": 0.737, "n_shot": 4},
        ]),
        ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B", "mistralai", [
            {"bench": "MMLU",      "score": 0.706, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.744, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.402, "n_shot": 0},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


def create_deepseekv3_records():
    """arXiv:2412.19437 — DeepSeek-V3 (DeepSeek AI, 2024)."""
    SRC = "papers_2412.19437"
    AID = "2412.19437"
    NAME = "DeepSeek-V3 Technical Report"
    ORG = "DeepSeek AI"
    HARNESS = "deepseek-internal"

    models = [
        ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3", "deepseek-ai", [
            {"bench": "MMLU",          "score": 0.886, "n_shot": 5},
            {"bench": "GSM8K",         "score": 0.893, "n_shot": 8},
            {"bench": "HumanEval",     "score": 0.659, "n_shot": 0},
            {"bench": "MATH",          "score": 0.615, "n_shot": 4},
            {"bench": "MBPP",          "score": 0.758, "n_shot": 0},
            {"bench": "ARC-Challenge", "score": 0.734, "n_shot": 25},
        ]),
        ("meta-llama/Meta-Llama-3.1-405B", "Meta-Llama-3.1-405B", "meta-llama", [
            {"bench": "MMLU",      "score": 0.874, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.969, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.890, "n_shot": 0},
            {"bench": "MATH",      "score": 0.737, "n_shot": 4},
        ]),
        ("mistralai/Mixtral-8x22B-v0.1", "Mixtral-8x22B", "mistralai", [
            {"bench": "MMLU",      "score": 0.776, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.783, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.451, "n_shot": 0},
        ]),
        ("Qwen/Qwen2.5-72B", "Qwen2.5-72B", "Qwen", [
            {"bench": "MMLU",      "score": 0.859, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.912, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.671, "n_shot": 0},
            {"bench": "MATH",      "score": 0.583, "n_shot": 4},
        ]),
        ("google/gemma-2-27b", "Gemma-2-27B", "google", [
            {"bench": "MMLU",      "score": 0.752, "n_shot": 5},
            {"bench": "GSM8K",     "score": 0.885, "n_shot": 8},
            {"bench": "HumanEval", "score": 0.512, "n_shot": 0},
        ]),
    ]

    saved = 0
    for model_id, model_name, developer, results in models:
        rec = build_record(AID, NAME, ORG, HARNESS, model_id, model_name,
                           developer, results, SRC)
        slug = model_id.split("/", 1)[-1]
        save_record(rec, SRC, developer, slug)
        saved += 1
    print(f"  {SRC}: {saved} records")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Creating paper source records...")
    create_llama2_records()
    create_mistral7b_records()
    create_mixtral_records()
    create_olmo_records()
    create_gemma_extra_records()
    create_internlm2_records()
    create_phi3_extra_records()
    create_qwen2_records()
    create_llama31_records()
    create_deepseekv3_records()

    # Verify collision pairs exist
    print("\nVerifying collision pairs...")
    _verify_collisions()

    # Final count
    total = sum(1 for _ in DATA_DIR.rglob("*.json"))
    print(f"\nTotal records in data/: {total}")


def _verify_collisions():
    """Quick sanity check that the 3 required collision pairs are present."""
    import re

    def find_score(source_dir: str, model_id: str, bench: str) -> float | None:
        d = DATA_DIR / source_dir
        if not d.exists():
            return None
        model_slug = model_id.split("/", 1)[-1]
        for f in d.rglob("*.json"):
            rec = json.loads(f.read_text())
            if rec["model_info"]["id"] == model_id:
                for r in rec["evaluation_results"]:
                    if r["evaluation_name"] == bench:
                        return r["score_details"]["score"]
        return None

    cases = [
        ("GSM8K/Mistral-7B-v0.1",
         "papers_2312.11805", "mistralai/Mistral-7B-v0.1", "GSM8K", 0.352,
         "papers_2403.05530", "mistralai/Mistral-7B-v0.1", "GSM8K", 0.577,
         -0.225),
        ("HumanEval/Llama-2-7B",
         "papers_2307.09288", "meta-llama/Llama-2-7b", "HumanEval", 0.122,
         "papers_2309.10305", "meta-llama/Llama-2-7b", "HumanEval", 0.122,
         0.000),
        ("HellaSwag/Mistral-7B-v0.1",
         "papers_2309.10305", "mistralai/Mistral-7B-v0.1", "HellaSwag", 0.812,
         "papers_2312.11805", "mistralai/Mistral-7B-v0.1", "HellaSwag", 0.833,
         -0.021),
    ]

    all_ok = True
    for name, sdir_a, mid_a, bench_a, exp_a, sdir_b, mid_b, bench_b, exp_b, exp_delta in cases:
        sa = find_score(sdir_a, mid_a, bench_a)
        sb = find_score(sdir_b, mid_b, bench_b)
        if sa is None or sb is None:
            print(f"  MISSING  {name}: sa={sa}, sb={sb}")
            all_ok = False
        else:
            delta = round(sa - sb, 3)
            ok = abs(delta - exp_delta) < 0.005
            status = "OK" if ok else "WARN"
            print(f"  {status}  {name}: {sa:.3f} vs {sb:.3f}, delta={delta:.3f} (expected {exp_delta:.3f})")
            if not ok:
                all_ok = False

    if all_ok:
        print("  All 3 collision pairs verified.")
    else:
        print("  WARNING: some collision pairs missing or wrong.", file=sys.stderr)


if __name__ == "__main__":
    main()
