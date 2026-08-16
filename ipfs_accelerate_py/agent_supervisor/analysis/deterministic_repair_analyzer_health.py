"""DCR-012 exact, content-addressed analyzer-health receipts.

This is an adapter over an indexer's already-produced tracked-path and parser
records.  It does not run a parser, invoke a provider, or weaken index limits.
The adapter exists because the current index script is independently dirty:
callers hand it the exact current forest identity and complete path accounting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Final

ANALYZER_HEALTH_INTERFACE: Final[str] = "AnalyzerHealth@1"
REPOSITORY_INDEX_INTERFACE: Final[str] = "RepositoryIndex@1"
DETERMINISTIC_ANALYZER_HEALTH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-analyzer-health@1"
)


class ParserDisposition(StrEnum):
    """The one required disposition for every tracked path."""

    PARSED = "parsed"
    TYPED_UNSUPPORTED = "typed_unsupported"
    EXCLUDED_BY_REVIEWED_POLICY = "excluded_by_reviewed_policy"
    PARSER_FAILURE = "parser_failure"


class MandatoryParserState(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _identity(value: Any, prefix: str) -> str:
    return f"{prefix}:sha256:{hashlib.sha256(_canonical(value)).hexdigest()}"


def _path(value: object) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or ".." in value.split("/"):
        raise ValueError("paths must be non-empty relative paths")
    return value


@dataclass(frozen=True)
class TrackedPathParserRecord:
    path: str
    disposition: ParserDisposition
    parser_id: str = ""
    source_digest: str = ""
    reason_code: str = ""
    reviewed_policy_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        if not isinstance(self.disposition, ParserDisposition):
            raise ValueError("disposition must be ParserDisposition")
        if self.disposition is ParserDisposition.PARSED and not self.parser_id:
            raise ValueError("parsed records require parser_id")
        if (
            self.disposition is ParserDisposition.EXCLUDED_BY_REVIEWED_POLICY
            and not self.reviewed_policy_id
        ):
            raise ValueError("reviewed exclusions require reviewed_policy_id")
        if self.disposition is not ParserDisposition.PARSED and not self.reason_code:
            raise ValueError("non-parsed records require reason_code")

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "disposition": self.disposition.value,
            "parser_id": self.parser_id,
            "source_digest": self.source_digest,
            "reason_code": self.reason_code,
            "reviewed_policy_id": self.reviewed_policy_id,
        }


@dataclass(frozen=True)
class RepositoryIndex:
    """Exact tracked-path accounting bound to a current forest identity."""

    forest_identity: str
    tracked_paths: tuple[str, ...]
    records: tuple[TrackedPathParserRecord, ...]

    INTERFACE: ClassVar[str] = REPOSITORY_INDEX_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.forest_identity, str) or not self.forest_identity:
            raise ValueError("forest_identity is required")
        tracked = tuple(sorted(_path(path) for path in self.tracked_paths))
        if len(tracked) != len(set(tracked)):
            raise ValueError("tracked paths must be unique")
        if not all(isinstance(record, TrackedPathParserRecord) for record in self.records):
            raise ValueError("records must be TrackedPathParserRecord values")
        records = tuple(sorted(self.records, key=lambda record: record.path))
        row_paths = tuple(record.path for record in records)
        if len(row_paths) != len(set(row_paths)):
            raise ValueError("each tracked path must have exactly one parser record")
        if set(row_paths) != set(tracked):
            raise ValueError("parser records must account for every tracked path exactly once")
        object.__setattr__(self, "tracked_paths", tracked)
        object.__setattr__(self, "records", records)

    @property
    def receipt_id(self) -> str:
        return _identity(self.to_dict(include_receipt=False), "repository-index")

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "interface": self.INTERFACE,
            "forest_identity": self.forest_identity,
            "tracked_paths": list(self.tracked_paths),
            "records": [record.to_dict() for record in self.records],
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class AnalyzerHealth:
    """DCR-012 non-authoritative health receipt for one complete index."""

    repository_index: RepositoryIndex
    mandatory_parser_states: Mapping[str, MandatoryParserState]
    stored_baseline_parser_failures: int = 0

    INTERFACE: ClassVar[str] = ANALYZER_HEALTH_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.repository_index, RepositoryIndex):
            raise ValueError("repository_index must be RepositoryIndex")
        if (
            isinstance(self.stored_baseline_parser_failures, bool)
            or self.stored_baseline_parser_failures < 0
        ):
            raise ValueError("stored baseline parser failures must be non-negative")
        states: dict[str, MandatoryParserState] = {}
        for parser_id, state in self.mandatory_parser_states.items():
            if not isinstance(parser_id, str) or not parser_id:
                raise ValueError("mandatory parser ids must be non-empty")
            states[parser_id] = (
                state if isinstance(state, MandatoryParserState) else MandatoryParserState(state)
            )
        object.__setattr__(self, "mandatory_parser_states", dict(sorted(states.items())))

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        for record in self.repository_index.records:
            if record.disposition is ParserDisposition.PARSER_FAILURE:
                blockers.append(f"parser_failure:{record.path}")
            elif record.disposition is ParserDisposition.TYPED_UNSUPPORTED:
                blockers.append(f"typed_unsupported:{record.path}")
        for parser_id, state in self.mandatory_parser_states.items():
            if state is not MandatoryParserState.AVAILABLE:
                blockers.append(f"mandatory_parser_{state.value}:{parser_id}")
        if self.stored_baseline_parser_failures == 22:
            blockers.append("stale_22_failure_baseline")
        return tuple(sorted(blockers))

    @property
    def safe_for_completion(self) -> bool:
        return not self.blockers

    @property
    def receipt_id(self) -> str:
        return _identity(self.to_dict(include_receipt=False), "analyzer-health")

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DETERMINISTIC_ANALYZER_HEALTH_SCHEMA,
            "interface": self.INTERFACE,
            "repository_index": self.repository_index.to_dict(),
            "mandatory_parser_states": {
                key: value.value for key, value in self.mandatory_parser_states.items()
            },
            "stored_baseline_parser_failures": self.stored_baseline_parser_failures,
            "blockers": list(self.blockers),
            "safe_for_completion": self.safe_for_completion,
            "completion_authoritative": False,
            "provider_or_llm_invoked": False,
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


def build_analyzer_health(
    *,
    forest_identity: str,
    tracked_paths: Sequence[str],
    records: Sequence[TrackedPathParserRecord],
    mandatory_parser_states: Mapping[str, MandatoryParserState | str],
    stored_baseline_parser_failures: int = 0,
) -> AnalyzerHealth:
    """Build a byte-stable receipt; callers must provide the current forest ID."""

    index = RepositoryIndex(forest_identity, tuple(tracked_paths), tuple(records))
    return AnalyzerHealth(index, mandatory_parser_states, stored_baseline_parser_failures)


__all__ = [
    "ANALYZER_HEALTH_INTERFACE",
    "DETERMINISTIC_ANALYZER_HEALTH_SCHEMA",
    "MandatoryParserState",
    "ParserDisposition",
    "REPOSITORY_INDEX_INTERFACE",
    "AnalyzerHealth",
    "RepositoryIndex",
    "TrackedPathParserRecord",
    "build_analyzer_health",
]
