"""Quick audit: count records in different ways to reconcile the paper's 14,213 figure."""
import json, glob, os

# 1. JSON files per source
json_count = {}
eval_row_count = {}
model_count = {}

for f in glob.glob("data/**/*.json", recursive=True):
    parts = f.split(os.sep)
    source = parts[1]
    json_count[source] = json_count.get(source, 0) + 1
    try:
        d = json.load(open(f))
        n_results = len(d.get("evaluation_results", []))
        eval_row_count[source] = eval_row_count.get(source, 0) + n_results
        model_id = d.get("model_info", {}).get("id", "") or d.get("model_info", {}).get("name", "")
        if model_id:
            model_count.setdefault(source, set()).add(model_id)
    except:
        continue

original_sources = sorted([s for s in json_count if s not in ("global-mmlu-lite", "reward-bench")])
new_sources = sorted([s for s in json_count if s in ("global-mmlu-lite", "reward-bench")])

print("=" * 70)
print(f"{'Source':<25} {'JSON files':>10} {'Eval rows':>10} {'Models':>10}")
print("=" * 70)
total_json = total_eval = total_models = 0
for s in original_sources:
    n_json = json_count[s]
    n_eval = eval_row_count.get(s, 0)
    n_mod = len(model_count.get(s, set()))
    print(f"{s:<25} {n_json:>10} {n_eval:>10} {n_mod:>10}")
    total_json += n_json
    total_eval += n_eval
    total_models += n_mod
print("-" * 70)
print(f"{'17 original TOTAL':<25} {total_json:>10} {total_eval:>10}")
print()

for s in new_sources:
    print(f"{s:<25} {json_count[s]:>10} {eval_row_count.get(s, 0):>10} {len(model_count.get(s, set())):>10}")

print(f"\n{'ALL 19 TOTAL':<25} {sum(json_count.values()):>10} {sum(eval_row_count.values()):>10}")
