"""
apply_audit_fixes.py — Phase 2 of EEE audit.

Applies fixes from audit_report.json:
  Rule 3  — Delete non-model records
  Rule 4  — Fix wrong model names (gpt-3.5-turbo in pre-2023 papers -> remove record)
  Rule 5  — Fix developer = "unknown" for known models
  Rule 6  — Fix evaluator_relationship
  Rule 12 — Fix eval_library.name

Also deletes additional non-model records found in Rule 11 suspicious zeros
(models like "# few-shot", "Performance gain est.", "Avg. Contam. %").
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"

# Import PAPER_META
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_latex",
    str(_ROOT / "scripts" / "extract_latex.py"),
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
PAPER_META = _mod.PAPER_META

# Load audit report
audit_report_path = RESULTS_DIR / "audit_report.json"
audit_report = json.loads(audit_report_path.read_text())
violations = audit_report["violations"]

# Track changes
changes_log = {
    "deleted_rule3": [],
    "deleted_rule4": [],
    "deleted_rule11_non_model": [],
    "fixed_rule5_developer": [],
    "fixed_rule6_eval_rel": [],
    "fixed_rule12_eval_library": [],
    "errors": [],
}

# Additional non-model patterns to detect in rule11 suspicious zeros
# (metadata/performance rows that aren't actual model evaluations)
RULE11_NON_MODEL_NAMES = {
    "avg. contam. %", "# few-shot", "performance gain est.", "performance gain",
}

def delete_record(json_path: Path, reason: str, log_key: str) -> bool:
    """Delete JSON file and its parent directory if empty."""
    if not json_path.exists():
        print(f"  SKIP (not found): {json_path}")
        return False
    try:
        json_path.unlink()
        # Clean up parent directories if empty
        parent = json_path.parent
        while parent != DATA_DIR and parent.exists():
            try:
                parent.rmdir()  # Only succeeds if empty
                parent = parent.parent
            except OSError:
                break  # Directory not empty
        changes_log[log_key].append(str(json_path))
        print(f"  DELETED ({reason}): {json_path.name}")
        return True
    except Exception as e:
        changes_log["errors"].append(f"delete {json_path}: {e}")
        print(f"  ERROR deleting {json_path}: {e}")
        return False

def fix_json_record(json_path: Path, fix_fn, log_key: str, description: str) -> bool:
    """Read JSON, apply fix_fn, write back."""
    if not json_path.exists():
        print(f"  SKIP (not found): {json_path}")
        return False
    try:
        record = json.loads(json_path.read_text())
        changed, record = fix_fn(record)
        if changed:
            json_path.write_text(json.dumps(record, indent=2))
            changes_log[log_key].append(str(json_path))
            print(f"  FIXED ({description}): {json_path.parent.name}")
            return True
        else:
            print(f"  SKIP (no change needed): {json_path.parent.name}")
            return False
    except Exception as e:
        changes_log["errors"].append(f"fix {json_path}: {e}")
        print(f"  ERROR fixing {json_path}: {e}")
        return False

# =============================================================================
# Phase 2a: Rule 3 — Delete non-model records
# =============================================================================
print("=== Phase 2a: Rule 3 — Deleting non-model records ===")
for v in violations["rule3_non_model"]:
    p = Path(v["path"])
    delete_record(p, f"non-model: {v['model_name']}", "deleted_rule3")

# =============================================================================
# Phase 2b: Rule 4 — Delete gpt-3.5-turbo/gpt-4 in pre-2023 papers
# (gpt-3.5-turbo didn't exist in 2022, so these are anachronistic)
# =============================================================================
print("\n=== Phase 2b: Rule 4 — Deleting wrong model names ===")
for v in violations["rule4_wrong_model_name"]:
    p = Path(v["path"])
    delete_record(p, f"anachronistic model {v['model_name']} in {v['arxiv_id']}", "deleted_rule4")

# =============================================================================
# Phase 2c: Rule 11 — Delete additional non-model records (metadata rows)
# =============================================================================
print("\n=== Phase 2c: Rule 11 — Deleting non-model rows from suspicious zeros ===")
for v in violations["rule11_suspicious_zeros"]:
    model_name_lower = v["model_name"].strip().lower()
    # Clean up any LaTeX artifacts
    model_name_clean = model_name_lower.replace("\\", "").replace("#", "#").strip()
    if any(nm in model_name_clean for nm in RULE11_NON_MODEL_NAMES):
        p = Path(v["path"])
        delete_record(p, f"metadata row: {v['model_name']}", "deleted_rule11_non_model")

# =============================================================================
# Phase 2d: Rule 5 — Fix developer = "unknown" for known models
# =============================================================================
print("\n=== Phase 2d: Rule 5 — Fixing unknown developers ===")

DEVELOPER_MAP = [
    (["chinchilla", "gopher"], "deepmind"),
    (["gpt-3", "gpt3", "gpt-4", "gpt4", "chatgpt", "davinci", "text-davinci", "code-davinci"], "openai"),
    (["llama", "codellama", "code-llama"], "meta-llama"),
    (["mistral", "mixtral"], "mistralai"),
    (["gemma", "palm", "gemini", "bard"], "google"),
    (["falcon"], "tiiuae"),
    (["mt-nlg", "megatron", "nemotron"], "nvidia"),
    (["qwen"], "Qwen"),
    (["deepseek"], "deepseek-ai"),
    (["phi-", "phi ", "orca-math", "wizardlm"], "microsoft"),
    (["olmo"], "allenai"),
    (["bloom"], "bigscience"),
    (["pythia"], "EleutherAI"),
    (["mpt-", "mpt "], "mosaicml"),
    (["dbrx"], "databricks"),
    (["yi-", "yi "], "01-ai"),
    (["command", "aya"], "CohereForAI"),
    (["jamba"], "ai21labs"),
    (["solar"], "upstage"),
    (["internlm"], "internlm"),
    (["baichuan"], "baichuan-inc"),
    (["starcoder"], "bigcode"),
    (["claude"], "anthropic"),
    (["titans"], "google"),
]

def infer_developer(name: str) -> str | None:
    n = name.lower()
    for keywords, dev in DEVELOPER_MAP:
        for kw in keywords:
            if kw in n:
                return dev
    return None

def make_rule5_fix(inferred_dev: str):
    def fix_fn(record: dict):
        changed = False
        model_info = record.get("model_info", {})
        if model_info.get("developer", "unknown") == "unknown":
            model_info["developer"] = inferred_dev
            # Also update model_info.id if it starts with "unknown/"
            old_id = model_info.get("id", "")
            if old_id.startswith("unknown/"):
                model_suffix = old_id[len("unknown/"):]
                model_info["id"] = f"{inferred_dev}/{model_suffix}"
            changed = True
        record["model_info"] = model_info
        return changed, record
    return fix_fn

for v in violations["rule5_unknown_developer"]:
    p = Path(v["path"])
    if not p.exists():
        print(f"  SKIP (already deleted): {p.name}")
        continue
    inferred = v["inferred_developer"]
    fix_fn = make_rule5_fix(inferred)
    fix_json_record(p, fix_fn, "fixed_rule5_developer", f"developer={inferred}")

# =============================================================================
# Phase 2e: Rule 6 — Fix evaluator_relationship (already 0 violations, but run anyway)
# =============================================================================
print("\n=== Phase 2e: Rule 6 — evaluator_relationship (0 violations) ===")
for v in violations.get("rule6_wrong_eval_rel", []):
    p = Path(v["path"])
    expected_rel = v["expected_rel"]
    def make_rule6_fix(exp_rel):
        def fix_fn(record):
            changed = False
            sm = record.get("source_metadata", {})
            if sm.get("evaluator_relationship") != exp_rel:
                sm["evaluator_relationship"] = exp_rel
                record["source_metadata"] = sm
                changed = True
            return changed, record
        return fix_fn
    fix_json_record(p, make_rule6_fix(expected_rel), "fixed_rule6_eval_rel", f"rel={expected_rel}")

# =============================================================================
# Phase 2f: Rule 12 — Fix eval_library.name (already 0 violations)
# =============================================================================
print("\n=== Phase 2f: Rule 12 — eval_library (0 violations) ===")
for v in violations.get("rule12_eval_library_unknown", []):
    p = Path(v["path"])
    harness = v["paper_harness"]
    def make_rule12_fix(h):
        def fix_fn(record):
            changed = False
            el = record.get("eval_library", {})
            if el.get("name", "unknown") == "unknown" and h != "unknown":
                el["name"] = h
                record["eval_library"] = el
                changed = True
            return changed, record
        return fix_fn
    fix_json_record(p, make_rule12_fix(harness), "fixed_rule12_eval_library", f"harness={harness}")

# =============================================================================
# Save changes log
# =============================================================================
log_path = RESULTS_DIR / "audit_fixes_log.json"
log_path.write_text(json.dumps(changes_log, indent=2))

print("\n=== Phase 2 Summary ===")
print(f"  Deleted Rule 3 (non-model): {len(changes_log['deleted_rule3'])}")
print(f"  Deleted Rule 4 (wrong model): {len(changes_log['deleted_rule4'])}")
print(f"  Deleted Rule 11 (non-model metadata rows): {len(changes_log['deleted_rule11_non_model'])}")
print(f"  Fixed Rule 5 (developer): {len(changes_log['fixed_rule5_developer'])}")
print(f"  Fixed Rule 6 (eval_rel): {len(changes_log['fixed_rule6_eval_rel'])}")
print(f"  Fixed Rule 12 (eval_library): {len(changes_log['fixed_rule12_eval_library'])}")
print(f"  Errors: {len(changes_log['errors'])}")
if changes_log["errors"]:
    for e in changes_log["errors"]:
        print(f"    {e}")
print(f"\nLog saved to {log_path}")
