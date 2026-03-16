"""Patch HF Open LLM v2 records with known n_shot and prompt_template values."""
import json
import glob

NSHOT_MAP = {
    "IFEval": "0",
    "BBH": "3",
    "MATH Level 5": "4",
    "GPQA": "0",
    "MUSR": "0",
    "MMLU-PRO": "5",
}

PROMPT_MAP = {
    "IFEval": "lighteval/ifeval",
    "BBH": "lighteval/bbh",
    "MATH Level 5": "lighteval/math_hard",
    "GPQA": "lighteval/gpqa",
    "MUSR": "lighteval/musr",
    "MMLU-PRO": "lighteval/mmlu_pro",
}

files = glob.glob("data/hfopenllm_v2/**/*.json", recursive=True)
patched = 0
for fpath in files:
    with open(fpath) as f:
        d = json.load(f)
    changed = False
    for r in d.get("evaluation_results", []):
        bench = r.get("evaluation_name", "")
        nshot = NSHOT_MAP.get(bench)
        prompt = PROMPT_MAP.get(bench, "lighteval")
        if nshot is not None:
            if r.get("generation_config") is None:
                r["generation_config"] = {}
            gc = r["generation_config"]
            if gc.get("additional_details") is None:
                gc["additional_details"] = {}
            gc["additional_details"]["n_shot"] = nshot
            gc["additional_details"]["prompt_template"] = prompt
            changed = True
    if changed:
        with open(fpath, "w") as f:
            json.dump(d, f, indent=2)
        patched += 1

print(f"Patched {patched}/{len(files)} HF records")
