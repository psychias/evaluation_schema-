#!/usr/bin/env python3
"""
validate_submission.py — validate all EEE JSON records against the schema.

Usage:
  python validate_submission.py                     # validate data/ directory
  python validate_submission.py --data-dir ../data  # custom data directory
  python validate_submission.py --verbose            # show per-file results
  python validate_submission.py --summary            # show only summary (default)

Exits with code 0 if all records are valid, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from jsonschema import validate, ValidationError, Draft7Validator
except ImportError:
    print("jsonschema not installed — run: pip install jsonschema>=4.23.0", file=sys.stderr)
    sys.exit(1)

_HERE = Path(__file__).resolve().parent
DEFAULT_SCHEMA = _HERE / "schema" / "eval.schema.json"
DEFAULT_DATA = _HERE / "data"


def load_schema(schema_path: Path) -> dict:
    """Load and return the JSON schema."""
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def validate_records(
    data_dir: Path,
    schema: dict,
    verbose: bool = False,
) -> tuple[int, int, list[str]]:
    """Validate all JSON files under data_dir.

    Returns (n_valid, n_invalid, error_messages).
    """
    validator = Draft7Validator(schema)
    n_valid = 0
    n_invalid = 0
    errors: list[str] = []

    json_files = sorted(data_dir.rglob("*.json"))
    if not json_files:
        print(f"WARNING: No JSON files found under {data_dir}", file=sys.stderr)
        return 0, 0, ["No JSON files found"]

    for path in json_files:
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except json.JSONDecodeError as e:
            n_invalid += 1
            msg = f"INVALID (JSON parse error): {path.relative_to(data_dir)} — {e}"
            errors.append(msg)
            if verbose:
                print(f"  ✗ {msg}")
            continue

        errs = list(validator.iter_errors(record))
        if errs:
            n_invalid += 1
            for err in errs:
                msg = f"INVALID: {path.relative_to(data_dir)} — {err.message} (at {'.'.join(str(p) for p in err.absolute_path)})"
                errors.append(msg)
                if verbose:
                    print(f"  ✗ {msg}")
        else:
            n_valid += 1
            if verbose:
                print(f"  ✓ {path.relative_to(data_dir)}")

    return n_valid, n_invalid, errors


def count_sources(data_dir: Path) -> dict[str, int]:
    """Count records per top-level source directory."""
    counts: dict[str, int] = {}
    for source_dir in sorted(data_dir.iterdir()):
        if source_dir.is_dir() and not source_dir.name.startswith("."):
            n = sum(1 for _ in source_dir.rglob("*.json"))
            if n > 0:
                counts[source_dir.name] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EEE submission records")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Path to data directory (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"Path to JSON schema (default: {DEFAULT_SCHEMA})",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-file results")
    parser.add_argument("--summary", action="store_true", help="Print summary only (default)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    schema_path = args.schema.resolve()

    if not schema_path.exists():
        print(f"ERROR: Schema not found at {schema_path}", file=sys.stderr)
        sys.exit(1)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Schema:     {schema_path}")
    print(f"Data dir:   {data_dir}")
    print()

    schema = load_schema(schema_path)
    print(f"Schema version: {schema.get('version', 'unknown')}")

    # Count sources
    counts = count_sources(data_dir)
    total = sum(counts.values())
    print(f"Sources:    {len(counts)}")
    print(f"Records:    {total}")
    for src, n in counts.items():
        print(f"  {src}: {n}")
    print()

    # Validate
    print("Validating all records ...")
    n_valid, n_invalid, errors = validate_records(data_dir, schema, verbose=args.verbose)
    print()

    # Summary
    print("=" * 60)
    print(f"  VALID:   {n_valid:>6}")
    print(f"  INVALID: {n_invalid:>6}")
    print(f"  TOTAL:   {n_valid + n_invalid:>6}")
    print("=" * 60)

    if n_invalid > 0:
        print(f"\n{n_invalid} validation error(s):")
        for err in errors[:20]:
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)
    else:
        print("\n✓ All records are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
