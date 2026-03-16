# Every Eval Ever — Hugging Face Dataset Card

## Dataset Summary

**Every Eval Ever** is a unified, schema-validated collection of LLM evaluation
results aggregated from multiple public leaderboards and academic papers.

| Stat | Value |
|---|---|
| Total records | 4,902 |
| Unique model IDs | 4,591 |
| Benchmark dimensions | 22 |
| Evaluation sources | 17 |
| Score collision pairs | 94 |
| Schema version | EEE v0.2.1 |
| License | CC BY 4.0 |

## Dataset Sources

| Source | Organisation | Models | Benchmarks |
|---|---|---|---|
| HF Open LLM Leaderboard v2 | Hugging Face | 4,496 | 6 |
| LMSYS Chatbot Arena | LMSYS | 39 | 1 |
| AlpacaEval 2.0 | Stanford | 40 | 2 |
| MT-Bench | LMSYS | 44 | 1 |
| WildBench v2 | AI2 | 30 | 1 |
| BigCodeBench | BigCode | 22 | 2 |
| LLaMA 3.1 Tech. Report | Meta | 17 | 6 |
| DeepSeek-V3 Tech. Report | DeepSeek AI | 16 | 6 |
| Gemma Tech. Report | Google | 13 | 6 |
| Qwen2 Tech. Report | Alibaba | 15 | 6 |
| Mistral 7B | Mistral AI | 12 | 7 |
| InternLM2 Tech. Report | SJTU AI Lab | 13 | 5 |
| Phi-3 Tech. Report | Microsoft | 12 | 5 |
| Mixtral of Experts | Mistral AI | 12 | 8 |
| LLaMA 2 Tech. Report | Meta | 20 | 5 |
| OLMo Tech. Report | AI2 | 13 | 6 |
| Falcon LLM | TII | 10 | 4 |

## Dataset Structure

Each record is a JSON file conforming to the [EEE schema v0.2.1](https://github.com/evaleval/every_eval_ever).

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
  "evaluation_results": [
    {
      "evaluation_name": "IFEval",
      "source_data": {"dataset_name": "IFEval", "source_type": "hf_dataset", "hf_repo": "google/IFEval"},
      "metric_config": {"lower_is_better": false, "score_type": "continuous", "min_score": 0.0, "max_score": 1.0},
      "score_details": {"score": 0.7354}
    }
  ]
}
```

## Key Findings

- **94 score collision pairs**: same (model, benchmark) in 2+ independent sources with different scores (after excluding citation duplicates where all of score, harness and n-shot are identical)
- **Prompt-template** is the dominant driver of score deltas (partial R² = 0.157, p < 0.001); n-shot also significant (partial R² = 0.058, p = 0.001)
- **Rank order preserved** despite absolute score differences (mean Kendall τ = 0.965)
- **Statistical power**: aggregating k=20 sources yields only 18.8% power to detect observed deltas (k for 80% power not reached)

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
```

## Citation

```bibtex
@inproceedings{psychias2026eee,
  title     = {Every Eval Ever: A Unified, Schema-Validated Dataset of LLM Evaluation Results},
  author    = {Psychias, Stelios},
  booktitle = {ACL 2026 Workshop on Evaluating Evaluations},
  year      = {2026},
}
```

## Methodology

Data is collected programmatically from public leaderboard APIs and published
CSV/JSON files. Where live sources are unavailable, frozen snapshots with
explicit timestamps are used. All records are validated against the official
JSON Schema before storage. See `scripts/scrapers/` for full collection code.

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Code: [MIT](https://opensource.org/licenses/MIT).
