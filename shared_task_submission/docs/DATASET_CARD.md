---
license: cc-by-4.0
task_categories:
  - text-generation
language:
  - en
tags:
  - evaluation
  - benchmarks
  - llm
  - meta-evaluation
  - reproducibility
  - leaderboard
  - every-eval-ever
  - acl-2026
  - evaleval
size_categories:
  - 1K<n<10K
pretty_name: Every Eval Ever
---

# Every Eval Ever — Unified LLM Evaluation Database

## Dataset Summary

**Every Eval Ever (EEE)** is a unified, schema-validated collection of LLM
evaluation results aggregated from 17 public sources (6 leaderboards +
11 research papers), standardized into a common JSON schema (v0.2.1).

The dataset enables cross-source comparison of evaluation results,
collision pair detection (same model + benchmark in different sources),
and methodology attribution analysis.

| Stat | Value |
|---|---|
| Total records | 10,975 |
| Unique model IDs | ~4,600 |
| Benchmark dimensions | 22 |
| Evaluation sources | 60 |
| Cross-source collision pairs | 253 |
| Schema version | EEE v0.2.1 |
| License | CC BY 4.0 |

## Dataset Sources

### Leaderboards (6)

| Source | Organisation | Models | Benchmarks |
|---|---|---|---|
| HF Open LLM Leaderboard v2 | Hugging Face | 4,496 | 6 |
| AlpacaEval 2.0 | Stanford | 254 | 2 |
| LMSYS Chatbot Arena | LMSYS | 39 | 1 |
| MT-Bench | LMSYS | 44 | 1 |
| WildBench v2 | AI2 | 30 | 1 |
| BigCodeBench | BigCode | 22 | 2 |

### Research Papers (11)

| Paper | arXiv | Models |
|---|---|---|
| Falcon LLM | 2306.11644 | 10 |
| LLaMA 2 | 2307.09288 | 20 |
| Mistral 7B | 2309.10305 | 12 |
| Mixtral of Experts | 2312.11805 | 12 |
| OLMo | 2402.01322 | 13 |
| Gemma | 2403.05530 | 13 |
| InternLM2 | 2403.17297 | 13 |
| Phi-3 | 2404.14219 | 12 |
| Qwen2 | 2405.04434 | 15 |
| Llama 3.1 | 2407.21783 | 17 |
| DeepSeek-V3 | 2412.19437 | 16 |

## Dataset Structure

```
data/
├── alpacaeval2/
├── bigcodebench/
├── chatbot_arena/
├── hfopenllm_v2/      # 4,496 model directories
│   └── {developer}/
│       └── {model}/
│           └── {uuid}.json
├── mt_bench/
├── wildbench/
└── papers_{arxiv_id}/  # × 11 paper sources
```

Each JSON file conforms to the EEE schema v0.2.1:

```json
{
  "schema_version": "0.2.1",
  "evaluation_id": "hfopenllm_v2/meta-llama_Llama-3.1-8B-Instruct/1740000000",
  "retrieved_timestamp": "1740000000",
  "source_metadata": {
    "source_name": "HuggingFace Open LLM Leaderboard v2",
    "source_type": "documentation",
    "source_organization_name": "Hugging Face",
    "evaluator_relationship": "third_party"
  },
  "eval_library": {"name": "lighteval", "version": "unknown"},
  "model_info": {
    "name": "meta-llama/Llama-3.1-8B-Instruct",
    "id": "meta-llama/Llama-3.1-8B-Instruct",
    "developer": "meta-llama"
  },
  "evaluation_results": [{
    "evaluation_name": "IFEval",
    "metric_config": {
      "lower_is_better": false,
      "score_type": "continuous",
      "min_score": 0.0,
      "max_score": 1.0
    },
    "score_details": {"score": 0.7354},
    "generation_config": {
      "additional_details": {
        "n_shot": "0",
        "prompt_template": "lighteval_default"
      }
    }
  }]
}
```

## Key Methodology Fields

| Field | Coverage | Description |
|---|---|---|
| `eval_library.name` (harness) | 100% | Which evaluation framework was used |
| `n_shot` | 70.6% | Number of few-shot examples |
| `prompt_template` | 17.6% | Exact prompt format identifier |
| `temperature` | 5.9% | Sampling temperature |

## Key Findings

- **253 collision pairs**: same (model, benchmark) evaluated by 2+ independent sources
- **Metadata coverage is the bottleneck**: prompt template documented in only 41% of sources
- **Prompt template** is the dominant predictor of MMLU cross-source variance
  (partial R² = 0.298, p < 0.001, BH-FDR q < 0.001)
- **N-shot count** is a significant secondary MMLU predictor (partial R² = 0.074, p = 0.011, q = 0.028)
- Both MMLU effects survive BH-FDR correction at q < 0.05; harness effects for HumanEval and GSM8K
  are not significant after correcting base/instruct model confusion in source data
- **50% prompt template coverage** (~7–9 additional documented sources) would enable
  80% power for MMLU methodology attribution

## Usage

```python
from datasets import load_dataset
ds = load_dataset("steliospsychias/every_eval_ever")
```

Or load individual records:
```python
import json, pathlib
records = [
    json.loads(p.read_text())
    for p in pathlib.Path("data").rglob("*.json")
]
print(f"Loaded {len(records)} evaluation records")
```

## Validation

```bash
python validate_submission.py
```

All records are validated against the JSON schema before inclusion.

## Citation

```bibtex
@inproceedings{psychias2026eee,
  title     = {When Same Benchmark $\neq$ Same Evaluation: Metadata Coverage
               as the Bottleneck for Cross-Source {LLM} Comparisons},
  author    = {Psychias, Stelios},
  booktitle = {Proceedings of the EvalEval Workshop at ACL 2026},
  year      = {2026},
}
```

## License

- **Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [MIT](https://opensource.org/licenses/MIT)
