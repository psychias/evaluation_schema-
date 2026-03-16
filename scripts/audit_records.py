"""
audit_records.py — Phase 1 & 2 of EEE audit.

Scans all JSON records in data/papers_*/ against audit rules, saves
results/audit_report.json, then applies fixes in-place.

Rules implemented:
  Rule 3  — Non-model records (delete)
  Rule 4  — Wrong model names (gpt-3.5-turbo/gpt-4 in pre-2023 papers)
  Rule 5  — Developer = "unknown" for known models
  Rule 6  — evaluator_relationship fix (first_party vs third_party)
  Rule 11 — Suspicious zeros
  Rule 12 — eval_library.name fix using PAPER_META
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Import PAPER_META from extract_latex.py
# ---------------------------------------------------------------------------
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_latex",
    str(_ROOT / "scripts" / "extract_latex.py"),
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
PAPER_META = _mod.PAPER_META

# ---------------------------------------------------------------------------
# Rule 3 — Non-model name patterns
# ---------------------------------------------------------------------------
NON_MODEL_PATTERNS = [
    r"^sota$", r"^sota\b", r"^supervised\b", r"^supervised[- ]sota",
    r"^human\b", r"^human[- ]eval", r"^random\b", r"^random[- ]baseline",
    r"^forecast\b", r"^oracle\b",
    r"^upper[- ]bound", r"^lower[- ]bound",
    r"^majority\b", r"^majority[- ]baseline",
    r"^baseline\b",
]

def is_non_model(name: str) -> bool:
    n = name.strip().lower()
    for pat in NON_MODEL_PATTERNS:
        if re.match(pat, n):
            return True
    return False

# ---------------------------------------------------------------------------
# Rule 5 — Developer inference from model name
# ---------------------------------------------------------------------------
DEVELOPER_MAP = [
    (["chinchilla", "gopher"], "deepmind"),
    (["gpt-3", "gpt3", "gpt-4", "gpt4", "chatgpt", "davinci", "text-davinci"], "openai"),
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
    (["mpt-"], "mosaicml"),
    (["dbrx"], "databricks"),
    (["yi-", "yi "], "01-ai"),
    (["command", "aya"], "CohereForAI"),
    (["jamba"], "ai21labs"),
    (["solar"], "upstage"),
    (["internlm"], "internlm"),
    (["baichuan"], "baichuan-inc"),
    (["starcoder"], "bigcode"),
    (["claude"], "anthropic"),
    (["titan"], "google"),
]

def infer_developer(name: str) -> str | None:
    n = name.lower()
    for keywords, dev in DEVELOPER_MAP:
        for kw in keywords:
            if kw in n:
                return dev
    return None

# ---------------------------------------------------------------------------
# Rule 6 — evaluator_relationship
# ---------------------------------------------------------------------------
def infer_eval_rel(model_developer: str, source_org: str) -> str:
    """first_party if developer matches source org, else third_party."""
    if not model_developer or model_developer == "unknown":
        return "third_party"

    md = model_developer.lower()
    so = source_org.lower()

    # Direct match
    if md == so:
        return "first_party"

    # Known org equivalences
    org_groups = [
        {"google", "deepmind", "google deepmind", "google brain"},
        {"meta", "meta ai", "meta-llama", "facebook"},
        {"microsoft", "msft"},
        {"openai"},
        {"mistralai", "mistral ai"},
        {"allenai", "allen ai", "allen institute"},
        {"nvidia"},
        {"databricks"},
        {"mosaicml"},
        {"ai21labs", "ai21"},
        {"deepseek-ai", "deepseek ai"},
        {"tiiuae", "tii uae", "tii"},
        {"bigscience"},
        {"eleutherai", "eleuther ai"},
        {"huggingface", "hf"},
        {"01-ai", "01.ai"},
        {"qwen", "alibaba group", "alibaba"},
        {"cohere", "cohereforai"},
        {"anthropic"},
        {"upstage"},
        {"internlm", "shanghai ai lab"},
        {"baichuan-inc", "baichuan inc"},
        {"bigcode"},
    ]

    for group in org_groups:
        if any(md in g for g in group) and any(so in g for g in group):
            return "first_party"

    # Check substring match with org groups
    for group in org_groups:
        md_match = any(g in md or md in g for g in group)
        so_match = any(g in so or so in g for g in group)
        if md_match and so_match:
            return "first_party"

    return "third_party"

# ---------------------------------------------------------------------------
# Rule 11 — Suspicious zeros
# ---------------------------------------------------------------------------
SUSPICIOUS_ZERO_BENCHMARKS = {"MMLU", "GSM8K", "HumanEval", "HellaSwag", "ARC-Challenge"}

def check_suspicious_zeros(record: dict) -> list[str]:
    issues = []
    for res in record.get("evaluation_results", []):
        bench = res.get("evaluation_name", "")
        score = res.get("score_details", {}).get("score", None)
        if bench in SUSPICIOUS_ZERO_BENCHMARKS and score == 0.0:
            issues.append(f"score=0.0 on {bench}")
    return issues

# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def get_arxiv_id_from_path(path: Path) -> str:
    """Extract arxiv_id from path like data/papers_2203.15556/..."""
    for part in path.parts:
        if part.startswith("papers_"):
            return part.replace("papers_", "")
    return "unknown"

def scan_all_records():
    """Scan all JSON records and return audit findings."""
    violations = {
        "rule3_non_model": [],
        "rule4_wrong_model_name": [],
        "rule5_unknown_developer": [],
        "rule6_wrong_eval_rel": [],
        "rule11_suspicious_zeros": [],
        "rule12_eval_library_unknown": [],
    }

    all_records = []

    for paper_dir in sorted(DATA_DIR.glob("papers_*")):
        arxiv_id = paper_dir.name.replace("papers_", "")
        paper_meta = PAPER_META.get(arxiv_id, {})

        for json_file in paper_dir.rglob("*.json"):
            try:
                record = json.loads(json_file.read_text())
            except Exception as e:
                print(f"  ERROR reading {json_file}: {e}")
                continue

            model_name = record.get("model_info", {}).get("name", "")
            model_dev = record.get("model_info", {}).get("developer", "unknown")
            source_org = record.get("source_metadata", {}).get("source_organization_name", "")
            eval_rel = record.get("source_metadata", {}).get("evaluator_relationship", "")
            eval_lib = record.get("eval_library", {}).get("name", "unknown")

            entry = {
                "path": str(json_file),
                "arxiv_id": arxiv_id,
                "model_name": model_name,
                "model_developer": model_dev,
                "source_org": source_org,
                "evaluator_relationship": eval_rel,
                "eval_library": eval_lib,
            }
            all_records.append(entry)

            # Rule 3 — Non-model records
            if is_non_model(model_name):
                violations["rule3_non_model"].append({
                    "path": str(json_file),
                    "model_name": model_name,
                    "arxiv_id": arxiv_id,
                })

            # Rule 4 — Wrong model names (gpt-3.5-turbo or gpt-4 in 2022 or earlier papers)
            year_prefix = arxiv_id[:2] if len(arxiv_id) >= 2 else "99"
            try:
                year = int(year_prefix)
            except:
                year = 99
            if year <= 22:
                model_lower = model_name.lower()
                if "gpt-3.5" in model_lower or "gpt-4" in model_lower:
                    violations["rule4_wrong_model_name"].append({
                        "path": str(json_file),
                        "model_name": model_name,
                        "arxiv_id": arxiv_id,
                    })

            # Rule 5 — Developer = unknown for known models
            if model_dev == "unknown":
                inferred = infer_developer(model_name)
                if inferred:
                    violations["rule5_unknown_developer"].append({
                        "path": str(json_file),
                        "model_name": model_name,
                        "arxiv_id": arxiv_id,
                        "inferred_developer": inferred,
                    })

            # Rule 6 — evaluator_relationship
            expected_rel = infer_eval_rel(model_dev, source_org)
            # Also check using PAPER_META's rel
            paper_rel = paper_meta.get("rel", None)
            if paper_rel and eval_rel != paper_rel:
                violations["rule6_wrong_eval_rel"].append({
                    "path": str(json_file),
                    "model_name": model_name,
                    "arxiv_id": arxiv_id,
                    "current_rel": eval_rel,
                    "expected_rel": paper_rel,
                })

            # Rule 11 — Suspicious zeros
            zero_issues = check_suspicious_zeros(record)
            if zero_issues:
                violations["rule11_suspicious_zeros"].append({
                    "path": str(json_file),
                    "model_name": model_name,
                    "arxiv_id": arxiv_id,
                    "issues": zero_issues,
                })

            # Rule 12 — eval_library.name = unknown when harness is known
            if eval_lib == "unknown" and paper_meta.get("harness", "unknown") != "unknown":
                violations["rule12_eval_library_unknown"].append({
                    "path": str(json_file),
                    "model_name": model_name,
                    "arxiv_id": arxiv_id,
                    "paper_harness": paper_meta["harness"],
                })

    return violations, all_records

def main():
    print("=== Phase 1: Scanning all records ===")
    violations, all_records = scan_all_records()

    report = {
        "total_records": len(all_records),
        "violations": violations,
        "summary": {
            "rule3_non_model": len(violations["rule3_non_model"]),
            "rule4_wrong_model_name": len(violations["rule4_wrong_model_name"]),
            "rule5_unknown_developer": len(violations["rule5_unknown_developer"]),
            "rule6_wrong_eval_rel": len(violations["rule6_wrong_eval_rel"]),
            "rule11_suspicious_zeros": len(violations["rule11_suspicious_zeros"]),
            "rule12_eval_library_unknown": len(violations["rule12_eval_library_unknown"]),
        }
    }

    out_path = RESULTS_DIR / "audit_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nAudit report saved to {out_path}")
    print(f"\nTotal records scanned: {len(all_records)}")
    print(f"Summary:")
    for rule, count in report["summary"].items():
        print(f"  {rule}: {count}")

    return violations

if __name__ == "__main__":
    main()
