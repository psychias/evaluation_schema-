"""Abstract base classes and protocol interfaces for the EEE scraper pipeline."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_ROOT = Path(__file__).resolve().parent.parent.parent
_schema_validator: Any = None
_schema_validator_loaded: bool = False


def _get_schema_validator() -> Any:
    """Cached jsonschema validator; returns None if unavailable."""
    global _schema_validator, _schema_validator_loaded
    if _schema_validator_loaded:
        return _schema_validator
    _schema_validator_loaded = True

    schema_path = _ROOT / "eval.schema.json"
    if not schema_path.exists():
        print(
            f"  [validation] eval.schema.json not found at {schema_path} "
            "— pre-write validation disabled",
            file=sys.stderr,
        )
        return None

    try:
        from jsonschema.validators import validator_for  # type: ignore[import-untyped]

        with schema_path.open(encoding="utf-8") as fh:
            schema = json.load(fh)
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        _schema_validator = validator_cls(schema)
    except Exception as exc:  # noqa: BLE001
        print(
            f"  [validation] could not load schema validator: {exc} "
            "— pre-write validation disabled",
            file=sys.stderr,
        )
    return _schema_validator

# ---------------------------------------------------------------------------
# protocol interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class IFetcher(Protocol):
    def fetch(self, source: str) -> Any: ...


@runtime_checkable
class IConverter(Protocol):
    def convert(self, raw: Any) -> list[dict]: ...


@runtime_checkable
class IValidator(Protocol):
    def validate(self, records: list[dict]) -> list[dict]: ...


@runtime_checkable
class IWriter(Protocol):
    def write(self, records: list[dict]) -> list[Path]: ...


# ---------------------------------------------------------------------------
# pipeline orchestrator
# ---------------------------------------------------------------------------


class NoopValidator:
    """Pass-through validator — accepts every record unchanged."""

    def validate(self, records: list[dict]) -> list[dict]:
        return records


class EEEPipeline:
    """Wires fetcher → converter → validator → writer."""

    def __init__(
        self,
        fetcher: IFetcher,
        converter: IConverter,
        writer: IWriter,
        validator: IValidator | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._converter = converter
        self._validator = validator if validator is not None else NoopValidator()
        self._writer = writer

    def run(self, source: str) -> list[Path]:
        raw = self._fetcher.fetch(source)
        records = self._converter.convert(raw)
        valid = self._validator.validate(records)
        return self._writer.write(valid)


# ---------------------------------------------------------------------------
# abstract base classes
# ---------------------------------------------------------------------------


class BaseLeaderboardScraper(ABC):
    """Base class for leaderboard scrapers. Subclass and implement fetch_raw + convert."""

    # subclasses set these class-level attributes
    eval_name: str = ""
    source_name: str = ""
    source_organization: str = ""
    output_dir: str = ""

    @abstractmethod
    def fetch_raw(self) -> Any:
        """fetch raw data from the leaderboard source."""
        ...

    @abstractmethod
    def convert(self, raw: Any, retrieved_timestamp: str) -> list[dict]:
        """convert raw leaderboard data to EEE schema dicts."""
        ...

    def save_raw(self, raw: Any, raw_dir: Path) -> None:
        """save raw data to *raw_dir* for reproducibility."""
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{self.eval_name}_raw.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, ensure_ascii=False)
        print(f"  raw data saved → {path}")

    def run(self, save_raw: bool = True) -> int:
        """run fetch → convert → write; return count of written files."""
        import time

        raw_dir = Path("scripts/scrapers/raw")
        retrieved_timestamp = str(time.time())

        print(f"[{self.eval_name}] fetching raw data...")
        raw = self.fetch_raw()

        if save_raw:
            self.save_raw(raw, raw_dir)

        print(f"[{self.eval_name}] converting to EEE schema...")
        records = self.convert(raw, retrieved_timestamp)

        print(f"[{self.eval_name}] writing {len(records)} records...")
        count = self._write_records(records)
        self._write_methodology(retrieved_timestamp)
        print(f"[{self.eval_name}] done — {count} files written.")
        return count

    def _write_records(self, records: list[dict]) -> int:
        """Validate against schema and write to data/{eval_name}/. Invalid records are skipped."""
        import uuid

        validator = _get_schema_validator()
        if validator is None:
            print(
                f"  [validation] WARNING: schema validator unavailable — "
                f"{len(records)} record(s) will be written without pre-write "
                f"schema validation. Ensure jsonschema is installed and "
                f"eval.schema.json is present before submitting a PR.",
                file=sys.stderr,
            )
        count = 0
        validation_failures = 0
        for rec in records:
            mid = rec.get("model_info", {}).get("id", "?")

            # leaderboard scrapers must emit source_type='documentation'
            src_type = rec.get("source_metadata", {}).get("source_type")
            if src_type and src_type != "documentation":
                print(
                    f"  [source_type] WARNING {mid}: source_type='{src_type}' — "
                    f"leaderboard scrapers must use 'documentation'. "
                    f"Record will still be validated; fix before submitting a PR.",
                    file=sys.stderr,
                )

            try:
                # validate before touching the filesystem
                if validator is not None:
                    validator.validate(rec)
            except Exception as exc:  # noqa: BLE001
                # exc from jsonschema has a .message attribute; fall back to str()
                msg = getattr(exc, "message", str(exc))
                print(
                    f"  [validation] SKIP {mid}: {msg}",
                    file=sys.stderr,
                )
                validation_failures += 1
                continue

            try:
                model_id: str = rec["model_info"]["id"]
                # model_id is e.g. "meta-llama/Llama-3.1-8B"
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                developer = _sanitize(developer)
                model_name = _sanitize(model_name)

                out_dir = Path(self.output_dir) / developer / model_name
                out_dir.mkdir(parents=True, exist_ok=True)

                path = out_dir / f"{uuid.uuid4()}.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)

                count += 1
            except Exception as exc:
                print(f"  warning: could not write record for {mid}: {exc}")

        if validation_failures:
            print(
                f"  [validation] {validation_failures}/{len(records)} record(s) "
                f"skipped due to schema validation errors — review stderr above "
                f"and fix source data before submitting a PR.",
                file=sys.stderr,
            )
        return count


    def _write_methodology(self, retrieved_timestamp: str) -> None:
        """Auto-generate methodology.txt for the shared task deliverables."""
        import datetime

        out_dir = Path("scripts/scrapers")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.eval_name}_methodology.txt"
        dt = datetime.datetime.utcfromtimestamp(float(retrieved_timestamp)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        content = (
            f"Methodology: {self.eval_name}\n"
            f"{'=' * (13 + len(self.eval_name))}\n\n"
            f"Source name:         {self.source_name}\n"
            f"Source organisation: {self.source_organization}\n"
            f"Schema version:      0.2.1\n"
            f"Retrieved at:        {dt} (UTC)\n"
            f"Script:              scripts/scrapers/{self.eval_name}_scraper.py\n\n"
            f"Extraction approach\n"
            f"-------------------\n"
            f"Data is fetched programmatically from the {self.source_name} public\n"
            f"API or web interface at the URL(s) documented in the scraper source.\n"
            f"Each entry is converted to the EEE eval.schema.json (v0.2.1) format\n"
            f"and validated against the schema before being written to disk.\n"
            f"source_type is set to 'documentation' for all records, reflecting\n"
            f"that scores originate from a published leaderboard (not a local run).\n\n"
            f"Known issues / manual decisions\n"
            f"--------------------------------\n"
            f"(none documented — update this section with any anomalies encountered\n"
            f" during extraction or conversion before submitting a PR)\n"
        )
        path.write_text(content, encoding="utf-8")
        print(f"  methodology written → {path}")


class BasePaperExtractor(ABC):
    """Base class for paper extractors."""

    @abstractmethod
    def extract(self, source: str) -> list[dict]: ...


class BaseProprietaryConverter(ABC):
    """Base class for Track 2 proprietary evaluation converters (first-party evals)."""

    # subclasses must set these
    eval_name: str = ""
    source_name: str = ""
    source_organization: str = ""
    output_dir: str = ""

    evaluator_relationship: str = "first_party"

    @abstractmethod
    def load_raw(self, source_path: Path) -> Any: ...

    @abstractmethod
    def convert(self, raw: Any, retrieved_timestamp: str) -> list[dict]: ...

    def run(self, source_path: Path, *, dry_run: bool = False) -> int:
        """Load → convert → write; return count of files written."""
        import time

        retrieved_timestamp = str(time.time())

        print(f"[{self.eval_name}] loading raw data from {source_path}...")
        raw = self.load_raw(source_path)

        print(f"[{self.eval_name}] converting to EEE schema...")
        records = self.convert(raw, retrieved_timestamp)
        print(f"[{self.eval_name}] converted {len(records)} records")

        if dry_run:
            print(f"[{self.eval_name}] dry-run mode — skipping disk writes")
            return len(records)

        count = self._write_records(records)
        print(f"[{self.eval_name}] done — {count} files written to {self.output_dir}/")
        return count

    def _write_records(self, records: list[dict]) -> int:
        """Validate and write records. Invalid records are skipped."""
        import uuid

        validator = _get_schema_validator()
        count = 0
        for rec in records:
            mid = rec.get("model_info", {}).get("id", "?")
            try:
                if validator is not None:
                    validator.validate(rec)
            except Exception as exc:  # noqa: BLE001
                msg = getattr(exc, "message", str(exc))
                print(
                    f"  [validation] SKIP {mid}: {msg}",
                    file=sys.stderr,
                )
                continue

            try:
                model_id: str = rec["model_info"]["id"]
                if "/" in model_id:
                    developer, model_name = model_id.split("/", 1)
                else:
                    developer = rec["model_info"].get("developer", "unknown")
                    model_name = model_id

                out_dir = Path(self.output_dir) / _sanitize(developer) / _sanitize(model_name)
                out_dir.mkdir(parents=True, exist_ok=True)

                path = out_dir / f"{uuid.uuid4()}.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=2, ensure_ascii=False)

                count += 1
            except Exception as exc:
                print(f"  warning: could not write record for {mid}: {exc}")

        return count


def _sanitize(name: str) -> str:
    """Sanitize for use as a filesystem path component."""
    import re

    return re.sub(r'[<>:"/\\|?*]', "_", name)
