"""Whole-tree polyglot AST parse-health recovery (SCA-166 / SCAEV166HEALTH).

Every parser-eligible path must resolve to a successful AST record or a typed
bounded failure.  JS/TS/JSX/TSX/CJS/MJS authority is required to come from a
real parser (the TypeScript compiler API via :class:`PolyglotASTProvider`),
never a regex/heuristic stand-in.  Reviewed per-language thresholds either
pass or the resulting health report remains a completion blocker.

Source bodies are accepted only as transient canary inputs.  They are never
retained on health objects, never serialized into reports, and therefore never
enter model context through this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .analyzer_health import AnalyzerHealthStatus
from .polyglot_ast_provider import (
    POLYGLOT_AST_PROVIDER_SCHEMA,
    TYPESCRIPT_EXTRACTOR_VERSION,
    PolyglotASTProvider,
    PolyglotASTProviderError,
    language_for_path,
)


POLYGLOT_AST_HEALTH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/polyglot-ast-health@1"
)
POLYGLOT_AST_HEALTH_EVIDENCE = "SCAEV166HEALTH"
POLYGLOT_AST_HEALTH_INTERFACE = "PolyglotASTHealth@1"

# Languages whose successful parse must bind the TypeScript compiler API.
_JS_TS_FAMILY = frozenset(
    {"javascript", "jsx", "typescript", "tsx", "cjs", "mjs"}
)
_JS_TS_CANONICAL = frozenset({"javascript", "jsx", "typescript", "tsx"})

_REAL_JS_TS_PRODUCERS = frozenset(
    {
        "typescript-compiler-api",
        TYPESCRIPT_EXTRACTOR_VERSION,
    }
)
_FORBIDDEN_AUTHORITY_MARKERS = frozenset(
    {
        "regex",
        "regexp",
        "heuristic",
        "approximate",
        "token-scan",
        "token_scan",
        "line-scan-only",
        "fake-parser",
        "fabricated",
    }
)
_FORBIDDEN_BODY_KEYS = frozenset(
    {
        "body",
        "bytes",
        "contents",
        "file_contents",
        "source",
        "source_body",
        "source_code",
        "source_contents",
        "source_text",
        "ast",
        "ast_body",
    }
)

# Path / status fields accepted from RepositoryIndex / coverage ledgers.
_SUCCESS_STATUSES = frozenset({"indexed", "cache_hit", "success", "parsed"})
_FAILURE_STATUSES = frozenset(
    {"parse_failure", "failure", "failed", "error", "bounded_failure"}
)
_ELIGIBLE_STATUSES = _SUCCESS_STATUSES | _FAILURE_STATUSES
_NON_ELIGIBLE_STATUSES = frozenset(
    {"not_applicable", "unsupported", "deleted", "excluded", "n/a", ""}
)

_DEFAULT_MAX_FAILURE_SAMPLES = 8
_DEFAULT_MAX_CLUSTERS = 64
_DEFAULT_MAX_PATH_SAMPLES = 32


class PathParseOutcome(str, Enum):
    """Closed disposition vocabulary for one parser-eligible path."""

    SUCCESS = "success"
    BOUNDED_FAILURE = "bounded_failure"
    NOT_ELIGIBLE = "not_eligible"


class ParserAuthorityKind(str, Enum):
    """Who claimed the right to produce AST facts for a language family."""

    REAL_TYPESCRIPT_COMPILER = "real_typescript_compiler"
    PYTHON_AST = "python_ast"
    STDLIB_JSON = "stdlib_json"
    REGEX_FORBIDDEN = "regex_forbidden"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class PolyglotASTHealthError(ValueError):
    """Invalid health input, authority claim, or report serialization."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "polyglot_ast_health_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


@dataclass(frozen=True)
class LanguageHealthThresholds:
    """Reviewed per-language parse-health budget.

    Non-zero failures that fit the budget remain a completion-unsafe
    ``partial`` signal.  Exceeding the budget is ``unhealthy``.  Neither
    state is converted into fabricated success.
    """

    max_parser_failures: int = 10
    max_parser_failure_ratio: float = 0.01
    min_success_ratio: float = 0.0
    require_real_js_ts_parser: bool = True
    require_canaries: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_parser_failures, bool) or self.max_parser_failures < 0:
            raise PolyglotASTHealthError(
                "max_parser_failures must be a non-negative integer",
                reason_code="invalid_threshold",
            )
        for name in ("max_parser_failure_ratio", "min_success_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PolyglotASTHealthError(
                    f"{name} must be between 0 and 1",
                    reason_code="invalid_threshold",
                )
            object.__setattr__(self, name, value)

    @classmethod
    def from_value(
        cls, value: "LanguageHealthThresholds | Mapping[str, Any] | None"
    ) -> "LanguageHealthThresholds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "language thresholds must be LanguageHealthThresholds or a mapping"
            )
        known = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in value if key not in known)
        if unknown:
            raise PolyglotASTHealthError(
                f"unknown language health thresholds: {', '.join(unknown)}",
                reason_code="unknown_threshold_fields",
            )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Reviewed defaults applied when a language has no explicit override.
DEFAULT_LANGUAGE_THRESHOLDS: Mapping[str, LanguageHealthThresholds] = {
    "python": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.05
    ),
    "json": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.05
    ),
    "json-schema": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.05
    ),
    "openapi-json": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.05
    ),
    "javascript": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.01
    ),
    "jsx": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.01
    ),
    "typescript": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.01
    ),
    "tsx": LanguageHealthThresholds(
        max_parser_failures=10, max_parser_failure_ratio=0.01
    ),
}


@dataclass(frozen=True)
class PathDispositionRecord:
    """Body-free disposition for one tracked path."""

    path: str
    language: str
    outcome: PathParseOutcome
    reason_code: str
    parser_status: str = ""
    parser_identity: str = ""
    parser_authority: ParserAuthorityKind = ParserAuthorityKind.UNKNOWN
    disposition_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(self.path or "").strip())
        object.__setattr__(self, "language", _normalize_language(self.language))
        object.__setattr__(
            self,
            "outcome",
            PathParseOutcome(str(getattr(self.outcome, "value", self.outcome))),
        )
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "").strip() or "unspecified"
        )
        object.__setattr__(
            self, "parser_status", str(self.parser_status or "").strip()
        )
        object.__setattr__(
            self, "parser_identity", str(self.parser_identity or "").strip()
        )
        object.__setattr__(
            self,
            "parser_authority",
            ParserAuthorityKind(
                str(getattr(self.parser_authority, "value", self.parser_authority))
            ),
        )
        if not self.disposition_id:
            object.__setattr__(
                self,
                "disposition_id",
                _disposition_id(
                    self.path,
                    self.language,
                    self.outcome.value,
                    self.reason_code,
                    self.parser_identity,
                ),
            )
        if self.outcome is PathParseOutcome.BOUNDED_FAILURE and not self.reason_code:
            raise PolyglotASTHealthError(
                f"bounded failure requires a typed reason: {self.path}",
                reason_code="missing_failure_reason",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition_id": self.disposition_id,
            "path": self.path,
            "language": self.language,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "parser_status": self.parser_status,
            "parser_identity": self.parser_identity,
            "parser_authority": self.parser_authority.value,
        }


@dataclass(frozen=True)
class FailureCluster:
    """Aggregate of bounded failures sharing language/reason/parser identity."""

    language: str
    reason_code: str
    parser_identity: str
    count: int
    sample_disposition_ids: tuple[str, ...] = ()
    sample_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", _normalize_language(self.language))
        object.__setattr__(
            self, "reason_code", str(self.reason_code or "unspecified").strip()
        )
        object.__setattr__(
            self, "parser_identity", str(self.parser_identity or "").strip()
        )
        object.__setattr__(self, "count", int(self.count))
        if self.count < 1:
            raise PolyglotASTHealthError(
                "failure cluster count must be at least 1",
                reason_code="invalid_cluster",
            )
        object.__setattr__(
            self,
            "sample_disposition_ids",
            tuple(str(item) for item in self.sample_disposition_ids if str(item)),
        )
        object.__setattr__(
            self,
            "sample_paths",
            tuple(str(item) for item in self.sample_paths if str(item)),
        )

    @property
    def cluster_id(self) -> str:
        return _identity(
            "failure-cluster",
            {
                "language": self.language,
                "reason_code": self.reason_code,
                "parser_identity": self.parser_identity,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "language": self.language,
            "reason_code": self.reason_code,
            "parser_identity": self.parser_identity,
            "count": self.count,
            "sample_disposition_ids": list(self.sample_disposition_ids),
            "sample_paths": list(self.sample_paths),
        }


@dataclass(frozen=True)
class LanguageHealthReport:
    """Per-language threshold evaluation."""

    language: str
    eligible_count: int
    success_count: int
    failure_count: int
    failure_ratio: float
    success_ratio: float
    status: AnalyzerHealthStatus
    reasons: tuple[str, ...]
    authority: ParserAuthorityKind
    thresholds: LanguageHealthThresholds

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.status is AnalyzerHealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "eligible_count": self.eligible_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "failure_ratio": self.failure_ratio,
            "success_ratio": self.success_ratio,
            "status": self.status.value,
            "safe_for_completion_reasoning": self.safe_for_completion_reasoning,
            "reasons": list(self.reasons),
            "authority": self.authority.value,
            "thresholds": self.thresholds.to_dict(),
        }


@dataclass(frozen=True)
class PolyglotCanaryResult:
    """One deterministic language canary without residual source text."""

    fixture_id: str
    language: str
    passed: bool
    producer: str = ""
    producer_version: str = ""
    compiler_name: str = ""
    compiler_version: str = ""
    authority: ParserAuthorityKind = ParserAuthorityKind.UNKNOWN
    reason_code: str = ""
    symbol_count: int = 0
    parse_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "language": self.language,
            "passed": self.passed,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "compiler_name": self.compiler_name,
            "compiler_version": self.compiler_version,
            "authority": self.authority.value,
            "reason_code": self.reason_code,
            "symbol_count": self.symbol_count,
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class PolyglotCanaryReport:
    results: tuple[PolyglotCanaryResult, ...]
    provider_schema: str = POLYGLOT_AST_PROVIDER_SCHEMA

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(item.passed for item in self.results)

    @property
    def fixture_count(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_schema": self.provider_schema,
            "passed": self.passed,
            "fixture_count": self.fixture_count,
            "failed_fixture_ids": [
                item.fixture_id for item in self.results if not item.passed
            ],
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(frozen=True)
class ParserAuthorityRepair:
    """Receipt for recovering a real JS/TS parser adapter configuration."""

    repaired: bool
    typescript_path: str
    producer: str
    producer_version: str
    compiler_name: str
    compiler_version: str
    authority: ParserAuthorityKind
    reason_code: str = ""
    candidate_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repaired": self.repaired,
            "typescript_path": self.typescript_path,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "compiler_name": self.compiler_name,
            "compiler_version": self.compiler_version,
            "authority": self.authority.value,
            "reason_code": self.reason_code,
            "candidate_paths": list(self.candidate_paths),
        }


@dataclass(frozen=True)
class PolyglotASTHealthReport:
    """Content-addressed whole-tree polyglot AST health receipt."""

    status: AnalyzerHealthStatus
    reasons: tuple[str, ...]
    dispositions: tuple[PathDispositionRecord, ...]
    clusters: tuple[FailureCluster, ...]
    language_health: tuple[LanguageHealthReport, ...]
    canaries: PolyglotCanaryReport
    authority_repair: ParserAuthorityRepair | None
    thresholds: Mapping[str, LanguageHealthThresholds]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence_id: str = POLYGLOT_AST_HEALTH_EVIDENCE
    schema: str = POLYGLOT_AST_HEALTH_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            AnalyzerHealthStatus(str(getattr(self.status, "value", self.status))),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(dict.fromkeys(str(item) for item in self.reasons if str(item))),
        )
        object.__setattr__(self, "dispositions", tuple(self.dispositions))
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "language_health", tuple(self.language_health))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(
            self,
            "thresholds",
            {
                str(key): LanguageHealthThresholds.from_value(value)
                for key, value in dict(self.thresholds).items()
            },
        )
        payload = self.to_dict()
        if report_contains_source_body(payload):
            raise PolyglotASTHealthError(
                "polyglot AST health report embeds a source body",
                reason_code="source_body_forbidden",
            )

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.status is AnalyzerHealthStatus.HEALTHY

    @property
    def healthy(self) -> bool:
        return self.status is AnalyzerHealthStatus.HEALTHY

    @property
    def completion_blocker(self) -> bool:
        return not self.safe_for_completion_reasoning

    def content_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict(include_identity=False))

    def content_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.content_bytes()).hexdigest()

    def content_identity(self) -> dict[str, Any]:
        """Return a content-addressed identity for the body-free report."""

        digest = self.content_digest()
        identity: dict[str, Any] = {
            "profile": "strict-dag-json-v1",
            "digest": digest,
            "byte_length": len(self.content_bytes()),
            "validated": False,
            "cid": "",
        }
        try:
            from .content_identity_bridge import identify_strict_artifact

            bound = identify_strict_artifact(self.to_dict(include_identity=False))
            if hasattr(bound, "to_dict"):
                payload = bound.to_dict()
            elif isinstance(bound, Mapping):
                payload = dict(bound)
            else:
                payload = {}
            if payload:
                identity.update(
                    {
                        "digest": str(payload.get("digest") or digest),
                        "cid": str(payload.get("cid") or ""),
                        "byte_length": int(
                            payload.get("byte_length") or identity["byte_length"]
                        ),
                        "validated": bool(payload.get("validated", False)),
                        "multibase": payload.get("multibase"),
                        "multicodec": payload.get("multicodec"),
                        "multihash": payload.get("multihash"),
                        "cid_version": payload.get("cid_version"),
                        "reason_codes": list(payload.get("reason_codes") or []),
                    }
                )
        except Exception as exc:  # identity is best-effort; digest always present
            identity["identity_error"] = f"{type(exc).__name__}: {exc}"
        return identity

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "interface": POLYGLOT_AST_HEALTH_INTERFACE,
            "evidence_id": self.evidence_id,
            "status": self.status.value,
            "healthy": self.healthy,
            "safe_for_completion_reasoning": self.safe_for_completion_reasoning,
            "completion_blocker": self.completion_blocker,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "thresholds": {
                language: thresholds.to_dict()
                for language, thresholds in sorted(self.thresholds.items())
            },
            "language_health": [item.to_dict() for item in self.language_health],
            "clusters": [item.to_dict() for item in self.clusters],
            "canaries": self.canaries.to_dict(),
            "authority_repair": (
                self.authority_repair.to_dict()
                if self.authority_repair is not None
                else None
            ),
            # Compact disposition ledger: IDs only at the report root keep the
            # receipt bounded; full rows are available via `dispositions`.
            "disposition_ids": [item.disposition_id for item in self.dispositions],
            "dispositions": [item.to_dict() for item in self.dispositions],
            "provider_schema": POLYGLOT_AST_PROVIDER_SCHEMA,
        }
        if include_identity:
            payload["content_identity"] = self.content_identity()
        return payload


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity(prefix: str, value: Any) -> str:
    return (
        f"{prefix}:sha256:"
        + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    )


def _disposition_id(
    path: str,
    language: str,
    outcome: str,
    reason_code: str,
    parser_identity: str,
) -> str:
    return _identity(
        "path-disposition",
        {
            "path": path,
            "language": language,
            "outcome": outcome,
            "reason_code": reason_code,
            "parser_identity": parser_identity,
        },
    )


def _normalize_language(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    if not raw:
        return ""
    if raw.startswith("."):
        raw = raw[1:]
    # ASTBlobRecord may bind "typescript@typescript-5.9.3".
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    aliases = {
        "cjs": "javascript",
        "mjs": "javascript",
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "json_schema": "json-schema",
        "openapi": "openapi-json",
    }
    return aliases.get(raw, raw)


def typed_reason_code(parser_reason: Any, *, fallback: str = "parse_failure") -> str:
    """Extract a stable machine reason from a free-form parser_reason string."""

    text = str(parser_reason or "").strip()
    if not text:
        return fallback
    # Common shapes: "compiler_unavailable: ...", "JSONDecodeError at line ..."
    head = text.split(":", 1)[0].strip()
    head = head.split(" at line", 1)[0].strip()
    head = head.split(" ", 1)[0].strip() if " " in head and head[0].isupper() else head
    normalized = head.casefold().replace(" ", "_")
    if not normalized:
        return fallback
    return normalized[:128]


def report_contains_source_body(value: Any) -> bool:
    """Return True when a report tree embeds a forbidden source/AST body key."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_BODY_KEYS:
                return True
            if report_contains_source_body(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(report_contains_source_body(item) for item in value)
    return False


def classify_parser_authority(
    *,
    language: str = "",
    producer: str = "",
    producer_version: str = "",
    parser_identity: str = "",
    parser_reason: str = "",
    compiler_name: str = "",
) -> ParserAuthorityKind:
    """Classify parser authority without inspecting source bodies."""

    language_name = _normalize_language(language)
    blob = " ".join(
        str(item or "")
        for item in (
            producer,
            producer_version,
            parser_identity,
            parser_reason,
            compiler_name,
        )
    ).casefold()
    if any(marker in blob for marker in _FORBIDDEN_AUTHORITY_MARKERS):
        return ParserAuthorityKind.REGEX_FORBIDDEN

    producer_name = str(producer or "").strip()
    if language_name in _JS_TS_CANONICAL or language_name in _JS_TS_FAMILY:
        if producer_name in _REAL_JS_TS_PRODUCERS or (
            "typescript-compiler-api" in blob
            and "typescript-ast-extractor@" in blob
        ):
            return ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
        if compiler_name.strip().casefold() == "typescript" and (
            "typescript" in blob
        ):
            return ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
        reason = typed_reason_code(parser_reason, fallback="")
        if reason in {
            "compiler_unavailable",
            "node_unavailable",
            "extractor_unavailable",
            "process_failed",
            "process_timeout",
        }:
            return ParserAuthorityKind.UNAVAILABLE
        if producer_name:
            return ParserAuthorityKind.UNKNOWN
        return ParserAuthorityKind.UNAVAILABLE

    if language_name == "python":
        if "regex" in blob:
            return ParserAuthorityKind.REGEX_FORBIDDEN
        return ParserAuthorityKind.PYTHON_AST
    if language_name in {"json", "json-schema", "openapi-json"}:
        return ParserAuthorityKind.STDLIB_JSON
    return ParserAuthorityKind.UNKNOWN


def js_ts_uses_real_parser(
    *,
    producer: str = "",
    producer_version: str = "",
    compiler_name: str = "",
    parser_identity: str = "",
    authority: ParserAuthorityKind | str | None = None,
) -> bool:
    """True only when JS/TS family authority is the TypeScript compiler API."""

    if authority is not None:
        kind = ParserAuthorityKind(str(getattr(authority, "value", authority)))
        if kind is ParserAuthorityKind.REGEX_FORBIDDEN:
            return False
        if kind is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER:
            return True
    classified = classify_parser_authority(
        language="typescript",
        producer=producer,
        producer_version=producer_version,
        parser_identity=parser_identity,
        compiler_name=compiler_name,
    )
    return classified is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER


def _row_mapping(row: Mapping[str, Any] | PathDispositionRecord) -> Mapping[str, Any]:
    if isinstance(row, PathDispositionRecord):
        return row.to_dict()
    if not isinstance(row, Mapping):
        raise PolyglotASTHealthError(
            "coverage rows must be mappings",
            reason_code="invalid_row",
        )
    return row


def classify_path_disposition(
    row: Mapping[str, Any] | PathDispositionRecord,
    *,
    producer: str = "",
    producer_version: str = "",
    compiler_name: str = "",
) -> PathDispositionRecord:
    """Map one coverage/index row to success, bounded failure, or not eligible."""

    payload = _row_mapping(row)
    path = str(payload.get("path") or "").strip()
    if not path:
        raise PolyglotASTHealthError(
            "coverage row is missing path",
            reason_code="missing_path",
        )
    language = _normalize_language(
        payload.get("language") or _language_from_path(path)
    )
    status = str(
        payload.get("parser_status")
        or payload.get("status")
        or payload.get("outcome")
        or ""
    ).strip().casefold()
    parser_identity = str(payload.get("parser_identity") or "").strip()
    parser_reason = str(
        payload.get("parser_reason")
        or payload.get("reason_code")
        or payload.get("parse_error")
        or ""
    )
    row_producer = str(payload.get("producer") or producer or "")
    row_producer_version = str(
        payload.get("producer_version") or producer_version or ""
    )
    row_compiler = str(payload.get("compiler_name") or compiler_name or "")

    if status in _NON_ELIGIBLE_STATUSES and status not in _ELIGIBLE_STATUSES:
        return PathDispositionRecord(
            path=path,
            language=language,
            outcome=PathParseOutcome.NOT_ELIGIBLE,
            reason_code=typed_reason_code(
                parser_reason, fallback=status or "not_eligible"
            ),
            parser_status=status or "not_applicable",
            parser_identity=parser_identity,
            parser_authority=classify_parser_authority(
                language=language,
                producer=row_producer,
                producer_version=row_producer_version,
                parser_identity=parser_identity,
                parser_reason=parser_reason,
                compiler_name=row_compiler,
            ),
        )

    authority = classify_parser_authority(
        language=language,
        producer=row_producer,
        producer_version=row_producer_version,
        parser_identity=parser_identity,
        parser_reason=parser_reason,
        compiler_name=row_compiler,
    )
    if status in _FAILURE_STATUSES or (
        not status and parser_reason and status not in _SUCCESS_STATUSES
    ):
        reason = typed_reason_code(parser_reason, fallback="parse_failure")
        if authority is ParserAuthorityKind.REGEX_FORBIDDEN:
            reason = "regex_authority_forbidden"
        return PathDispositionRecord(
            path=path,
            language=language,
            outcome=PathParseOutcome.BOUNDED_FAILURE,
            reason_code=reason,
            parser_status=status or "parse_failure",
            parser_identity=parser_identity,
            parser_authority=authority,
        )
    if status in _SUCCESS_STATUSES or (
        not status and not parser_reason and language
    ):
        if (
            language in _JS_TS_CANONICAL
            and authority is ParserAuthorityKind.REGEX_FORBIDDEN
        ):
            return PathDispositionRecord(
                path=path,
                language=language,
                outcome=PathParseOutcome.BOUNDED_FAILURE,
                reason_code="regex_authority_forbidden",
                parser_status=status or "parse_failure",
                parser_identity=parser_identity,
                parser_authority=authority,
            )
        return PathDispositionRecord(
            path=path,
            language=language,
            outcome=PathParseOutcome.SUCCESS,
            reason_code="indexed",
            parser_status=status or "indexed",
            parser_identity=parser_identity,
            parser_authority=authority,
        )
    # Unknown status with a language still needs a typed disposition.
    if language:
        return PathDispositionRecord(
            path=path,
            language=language,
            outcome=PathParseOutcome.BOUNDED_FAILURE,
            reason_code=typed_reason_code(
                parser_reason or status, fallback="untyped_parser_status"
            ),
            parser_status=status or "parse_failure",
            parser_identity=parser_identity,
            parser_authority=authority,
        )
    return PathDispositionRecord(
        path=path,
        language=language,
        outcome=PathParseOutcome.NOT_ELIGIBLE,
        reason_code=typed_reason_code(parser_reason, fallback="not_eligible"),
        parser_status=status or "not_applicable",
        parser_identity=parser_identity,
        parser_authority=authority,
    )


def _language_from_path(path: str) -> str:
    try:
        return language_for_path(path)
    except PolyglotASTProviderError:
        return ""


def classify_path_dispositions(
    rows: Iterable[Mapping[str, Any] | PathDispositionRecord],
) -> tuple[PathDispositionRecord, ...]:
    """Classify every input row; eligible paths never remain silent."""

    records = [classify_path_disposition(row) for row in rows]
    return tuple(records)


def cluster_failures(
    dispositions: Sequence[PathDispositionRecord],
    *,
    max_samples: int = _DEFAULT_MAX_FAILURE_SAMPLES,
    max_clusters: int = _DEFAULT_MAX_CLUSTERS,
) -> tuple[FailureCluster, ...]:
    """Cluster bounded failures by language / reason / parser identity."""

    if max_samples < 0 or max_clusters < 0:
        raise PolyglotASTHealthError(
            "cluster sample limits must be non-negative",
            reason_code="invalid_cluster_limits",
        )
    buckets: dict[tuple[str, str, str], list[PathDispositionRecord]] = {}
    for item in dispositions:
        if item.outcome is not PathParseOutcome.BOUNDED_FAILURE:
            continue
        key = (item.language, item.reason_code, item.parser_identity)
        buckets.setdefault(key, []).append(item)

    clusters: list[FailureCluster] = []
    for (language, reason_code, parser_identity), members in sorted(
        buckets.items(),
        key=lambda pair: (-len(pair[1]), pair[0][0], pair[0][1], pair[0][2]),
    ):
        if len(clusters) >= max_clusters:
            break
        samples = members[:max_samples]
        clusters.append(
            FailureCluster(
                language=language,
                reason_code=reason_code,
                parser_identity=parser_identity,
                count=len(members),
                sample_disposition_ids=tuple(
                    item.disposition_id for item in samples
                ),
                sample_paths=tuple(item.path for item in samples),
            )
        )
    return tuple(clusters)


def evaluate_language_health(
    dispositions: Sequence[PathDispositionRecord],
    *,
    language: str,
    thresholds: LanguageHealthThresholds | Mapping[str, Any] | None = None,
    canaries_passed: bool = True,
    observed_authority: ParserAuthorityKind | None = None,
) -> LanguageHealthReport:
    """Evaluate one language against reviewed thresholds."""

    language_name = _normalize_language(language)
    policy = LanguageHealthThresholds.from_value(thresholds)
    eligible = [
        item
        for item in dispositions
        if item.language == language_name
        and item.outcome
        in {PathParseOutcome.SUCCESS, PathParseOutcome.BOUNDED_FAILURE}
    ]
    success = sum(
        item.outcome is PathParseOutcome.SUCCESS for item in eligible
    )
    failure = len(eligible) - success
    eligible_count = len(eligible)
    failure_ratio = (
        failure / eligible_count if eligible_count else (1.0 if failure else 0.0)
    )
    success_ratio = (
        success / eligible_count if eligible_count else (1.0 if not failure else 0.0)
    )

    if observed_authority is not None:
        authority = ParserAuthorityKind(
            str(getattr(observed_authority, "value", observed_authority))
        )
    elif eligible:
        # Prefer success authorities, then any non-unknown failure authority.
        success_authorities = [
            item.parser_authority
            for item in eligible
            if item.outcome is PathParseOutcome.SUCCESS
        ]
        if success_authorities:
            authority = success_authorities[0]
        else:
            authority = eligible[0].parser_authority
    else:
        authority = ParserAuthorityKind.UNKNOWN

    unhealthy: list[str] = []
    partial: list[str] = []

    if eligible_count and failure:
        if (
            failure > policy.max_parser_failures
            or failure_ratio > policy.max_parser_failure_ratio
        ):
            unhealthy.append("parser_failure_budget_exceeded")
        else:
            partial.append("parser_failures_within_budget")
    if eligible_count and success_ratio < policy.min_success_ratio:
        unhealthy.append("success_ratio_below_budget")

    if language_name in _JS_TS_CANONICAL and policy.require_real_js_ts_parser:
        if authority is ParserAuthorityKind.REGEX_FORBIDDEN:
            unhealthy.append("regex_authority_forbidden")
        elif (
            eligible_count
            and success > 0
            and authority is not ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
        ):
            unhealthy.append("js_ts_real_parser_required")
        elif (
            eligible_count
            and success == 0
            and authority is ParserAuthorityKind.UNAVAILABLE
        ):
            # Failures are typed, but promotion is still blocked until a real
            # parser can serve the family.
            unhealthy.append("js_ts_parser_unavailable")

    if policy.require_canaries and not canaries_passed:
        unhealthy.append("canary_failure")

    if unhealthy:
        status = AnalyzerHealthStatus.UNHEALTHY
        reasons = tuple(dict.fromkeys(unhealthy + partial))
    elif partial:
        status = AnalyzerHealthStatus.PARTIAL
        reasons = tuple(dict.fromkeys(partial))
    else:
        status = AnalyzerHealthStatus.HEALTHY
        reasons = ()

    return LanguageHealthReport(
        language=language_name,
        eligible_count=eligible_count,
        success_count=success,
        failure_count=failure,
        failure_ratio=failure_ratio,
        success_ratio=success_ratio,
        status=status,
        reasons=reasons,
        authority=authority,
        thresholds=policy,
    )


# Deterministic in-memory canary sources.  They never leave this module in a
# serialized report; only metadata about the run is retained.
_CANARY_FIXTURES: tuple[tuple[str, str, str], ...] = (
    (
        "python-protocol",
        "python",
        "class Runner:\n    def run(self, request):\n        return request\n",
    ),
    (
        "json-schema",
        "json",
        '{"$id":"canary","type":"object","properties":{"id":{"type":"string"}}}',
    ),
    (
        "javascript-module",
        "javascript",
        "export function run(input) { return input; }\n",
    ),
    (
        "typescript-service",
        "typescript",
        "export function run(input: string): string { return input; }\n",
    ),
    (
        "tsx-component",
        "tsx",
        "export const View = (props: {n: number}) => props.n;\n",
    ),
    (
        "jsx-component",
        "jsx",
        "export const View = (props) => props.value;\n",
    ),
    (
        "cjs-export",
        "cjs",
        "function run(input) { return input; }\nmodule.exports = { run };\n",
    ),
    (
        "mjs-export",
        "mjs",
        "export function run(input) { return input; }\n",
    ),
)


def run_polyglot_ast_canaries(
    provider: PolyglotASTProvider | None = None,
    *,
    fixtures: Sequence[tuple[str, str, str]] | None = None,
) -> PolyglotCanaryReport:
    """Run deterministic language canaries through a real polyglot provider.

    Source text is fed to the provider and discarded; only body-free metadata
    is retained on the report.
    """

    active = provider or PolyglotASTProvider()
    results: list[PolyglotCanaryResult] = []
    for fixture_id, language, source in fixtures or _CANARY_FIXTURES:
        language_name = _normalize_language(language)
        try:
            extraction = active.extract_with_metadata(
                source,
                language,
                blob_identity=f"blob:canary:{fixture_id}",
            )
            # Immediately drop the source reference from this frame's outer
            # visibility by never storing it on the result.
            authority = classify_parser_authority(
                language=language_name,
                producer=extraction.producer,
                producer_version=extraction.producer_version,
                compiler_name=extraction.compiler_name,
            )
            parse_error = str(extraction.record.parse_error or "")
            js_ts = language_name in _JS_TS_CANONICAL
            real_ok = (
                (not js_ts)
                or authority is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
            )
            passed = not parse_error and real_ok
            reason = ""
            if parse_error:
                reason = typed_reason_code(parse_error)
            elif not real_ok:
                reason = "js_ts_real_parser_required"
            results.append(
                PolyglotCanaryResult(
                    fixture_id=fixture_id,
                    language=language_name,
                    passed=passed,
                    producer=extraction.producer,
                    producer_version=extraction.producer_version,
                    compiler_name=extraction.compiler_name,
                    compiler_version=extraction.compiler_version,
                    authority=authority,
                    reason_code=reason,
                    symbol_count=len(extraction.record.qualified_symbols),
                    parse_error=parse_error[:256],
                )
            )
        except PolyglotASTProviderError as exc:
            authority = classify_parser_authority(
                language=language_name,
                parser_reason=exc.reason_code,
            )
            results.append(
                PolyglotCanaryResult(
                    fixture_id=fixture_id,
                    language=language_name,
                    passed=False,
                    authority=authority,
                    reason_code=str(exc.reason_code),
                    parse_error=str(exc)[:256],
                )
            )
        except Exception as exc:  # canaries classify exceptions, never leak source
            results.append(
                PolyglotCanaryResult(
                    fixture_id=fixture_id,
                    language=language_name,
                    passed=False,
                    authority=ParserAuthorityKind.UNKNOWN,
                    reason_code="canary_exception",
                    parse_error=f"{type(exc).__name__}: {exc}"[:256],
                )
            )
    report = PolyglotCanaryReport(tuple(results))
    if report_contains_source_body(report.to_dict()):
        raise PolyglotASTHealthError(
            "canary report embeds a source body",
            reason_code="source_body_forbidden",
        )
    return report


def discover_typescript_path(
    *,
    explicit: str | os.PathLike[str] | None = None,
    search_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Locate a local TypeScript compiler API package without starting Node."""

    candidates: list[str] = []
    if explicit:
        candidates.append(str(Path(explicit)))
    env_path = os.environ.get("TYPESCRIPT_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    roots: list[Path] = []
    if search_roots is not None:
        roots.extend(Path(item) for item in search_roots)
    else:
        cwd = Path.cwd()
        roots.extend(
            [
                cwd,
                cwd / "swissknife",
                Path("/home/barberb/lift_coding/swissknife"),
            ]
        )
        # Walk a few parents so worktree checkouts still find the host tree.
        here = Path(__file__).resolve()
        roots.extend(list(here.parents)[:8])

    for root in roots:
        for relative in (
            Path("node_modules/typescript"),
            Path("swissknife/node_modules/typescript"),
        ):
            candidates.append(str((root / relative).resolve()))

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
        path = Path(normalized)
        package_json = path / "package.json" if path.is_dir() else None
        if path.is_dir() and package_json is not None and package_json.is_file():
            return str(path), tuple(unique)
        if path.is_file() and path.name.endswith(".js"):
            # require.resolve('typescript') may point at lib/typescript.js
            parent = path.parent
            if parent.name == "lib":
                parent = parent.parent
            if (parent / "package.json").is_file():
                return str(parent), tuple(unique)
    return "", tuple(unique)


def repair_polyglot_parser_authority(
    provider: PolyglotASTProvider | None = None,
    *,
    typescript_path: str | os.PathLike[str] | None = None,
    search_roots: Sequence[str | os.PathLike[str]] | None = None,
    probe_source: str = "export function canary(): number { return 1; }\n",
) -> tuple[PolyglotASTProvider, ParserAuthorityRepair]:
    """Bind a real TypeScript compiler API path onto a polyglot provider.

    This repairs adapter *configuration* only: it never fabricates AST success
    for inventory paths and never converts a parse failure into a success row.
    """

    discovered, candidates = discover_typescript_path(
        explicit=typescript_path, search_roots=search_roots
    )
    base = provider or PolyglotASTProvider()
    if not discovered:
        repair = ParserAuthorityRepair(
            repaired=False,
            typescript_path="",
            producer="",
            producer_version="",
            compiler_name="",
            compiler_version="",
            authority=ParserAuthorityKind.UNAVAILABLE,
            reason_code="compiler_unavailable",
            candidate_paths=candidates,
        )
        return base, repair

    repaired_provider = PolyglotASTProvider(
        limits=base.limits,
        node_executable=base.node_executable,
        extractor_path=base.extractor_path,
        typescript_path=discovered,
        expected_typescript_version=base.expected_typescript_version,
        process_runner=getattr(base, "_process_runner", None),
    )
    try:
        extraction = repaired_provider.extract_with_metadata(
            probe_source,
            "typescript",
            blob_identity="blob:authority-repair-probe",
        )
    except PolyglotASTProviderError as exc:
        return repaired_provider, ParserAuthorityRepair(
            repaired=False,
            typescript_path=discovered,
            producer="",
            producer_version="",
            compiler_name="",
            compiler_version="",
            authority=classify_parser_authority(
                language="typescript", parser_reason=exc.reason_code
            ),
            reason_code=str(exc.reason_code),
            candidate_paths=candidates,
        )
    authority = classify_parser_authority(
        language="typescript",
        producer=extraction.producer,
        producer_version=extraction.producer_version,
        compiler_name=extraction.compiler_name,
    )
    repaired = (
        authority is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
        and not extraction.record.parse_error
    )
    return repaired_provider, ParserAuthorityRepair(
        repaired=repaired,
        typescript_path=discovered,
        producer=extraction.producer,
        producer_version=extraction.producer_version,
        compiler_name=extraction.compiler_name,
        compiler_version=extraction.compiler_version,
        authority=authority,
        reason_code="" if repaired else typed_reason_code(
            extraction.record.parse_error, fallback="authority_probe_failed"
        ),
        candidate_paths=candidates,
    )


def _threshold_map(
    thresholds: Mapping[str, LanguageHealthThresholds | Mapping[str, Any]]
    | LanguageHealthThresholds
    | None,
) -> dict[str, LanguageHealthThresholds]:
    if thresholds is None:
        return {
            language: LanguageHealthThresholds.from_value(value)
            for language, value in DEFAULT_LANGUAGE_THRESHOLDS.items()
        }
    if isinstance(thresholds, LanguageHealthThresholds):
        return {
            language: thresholds
            for language in DEFAULT_LANGUAGE_THRESHOLDS
        }
    return {
        _normalize_language(language) or str(language): LanguageHealthThresholds.from_value(
            value
        )
        for language, value in thresholds.items()
    }


def assess_polyglot_ast_health(
    rows: Iterable[Mapping[str, Any] | PathDispositionRecord],
    *,
    thresholds: Mapping[str, LanguageHealthThresholds | Mapping[str, Any]]
    | LanguageHealthThresholds
    | None = None,
    provider: PolyglotASTProvider | None = None,
    run_canaries: bool = True,
    repair_authority: bool = True,
    search_roots: Sequence[str | os.PathLike[str]] | None = None,
    max_disposition_samples: int = 0,
) -> PolyglotASTHealthReport:
    """Classify path outcomes, cluster failures, canary, and gate completion.

    When ``max_disposition_samples`` is 0, every disposition is retained.  Set
    a positive bound only for extremely large ledgers where the caller already
    has the full path inventory elsewhere; the metrics still use the full set.
    """

    dispositions = classify_path_dispositions(rows)
    eligible = [
        item
        for item in dispositions
        if item.outcome
        in {PathParseOutcome.SUCCESS, PathParseOutcome.BOUNDED_FAILURE}
    ]
    if any(
        item.outcome is PathParseOutcome.BOUNDED_FAILURE
        and item.reason_code in {"", "unspecified"}
        for item in eligible
    ):
        # Defensive: classify_path_disposition always assigns a reason.
        raise PolyglotASTHealthError(
            "eligible path missing typed bounded failure reason",
            reason_code="missing_failure_reason",
        )

    authority_repair: ParserAuthorityRepair | None = None
    active_provider = provider
    if repair_authority:
        active_provider, authority_repair = repair_polyglot_parser_authority(
            provider, search_roots=search_roots
        )

    if run_canaries:
        canaries = run_polyglot_ast_canaries(active_provider)
    else:
        canaries = PolyglotCanaryReport(())

    policy = _threshold_map(thresholds)
    languages = sorted(
        {
            item.language
            for item in eligible
            if item.language
        }
        | set(policy)
    )
    # Canary pass is evaluated per language where fixtures exist.
    canary_by_language: dict[str, bool] = {}
    for result in canaries.results:
        canary_by_language[result.language] = canary_by_language.get(
            result.language, True
        ) and result.passed
    # cjs/mjs normalize to javascript.
    if "javascript" in canary_by_language:
        canary_by_language.setdefault(
            "javascript", canary_by_language["javascript"]
        )

    observed_js_ts_authority = ParserAuthorityKind.UNAVAILABLE
    if authority_repair and authority_repair.repaired:
        observed_js_ts_authority = ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
    else:
        for result in canaries.results:
            if result.language in _JS_TS_CANONICAL and result.passed:
                observed_js_ts_authority = ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
                break
        else:
            for item in eligible:
                if (
                    item.language in _JS_TS_CANONICAL
                    and item.outcome is PathParseOutcome.SUCCESS
                    and item.parser_authority
                    is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
                ):
                    observed_js_ts_authority = (
                        ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
                    )
                    break

    language_reports: list[LanguageHealthReport] = []
    for language in languages:
        language_eligible = [
            item for item in eligible if item.language == language
        ]
        if not language_eligible and language not in {
            item.language for item in eligible
        }:
            # Skip default-threshold languages with zero inventory presence.
            if language in DEFAULT_LANGUAGE_THRESHOLDS and language not in {
                item.language for item in dispositions if item.language == language
            }:
                continue
        language_threshold = policy.get(language) or LanguageHealthThresholds()
        canary_ok = True
        if language_threshold.require_canaries and run_canaries:
            if language in _JS_TS_CANONICAL:
                canary_ok = all(
                    canary_by_language.get(name, False)
                    for name in (language,)
                    if any(r.language == name for r in canaries.results)
                )
                # Family-wide: if any JS/TS canary ran, require that language's
                # fixture (or the shared javascript fixture for cjs/mjs).
                related = [
                    item
                    for item in canaries.results
                    if item.language == language
                    or (
                        language == "javascript"
                        and item.language == "javascript"
                    )
                ]
                canary_ok = bool(related) and all(item.passed for item in related)
            else:
                related = [
                    item
                    for item in canaries.results
                    if item.language == language
                ]
                canary_ok = (not related) or all(item.passed for item in related)
        authority = (
            observed_js_ts_authority
            if language in _JS_TS_CANONICAL
            else None
        )
        language_reports.append(
            evaluate_language_health(
                dispositions,
                language=language,
                thresholds=language_threshold,
                canaries_passed=canary_ok,
                observed_authority=authority,
            )
        )

    # Drop languages that never appeared in the inventory.
    language_reports = [
        item
        for item in language_reports
        if item.eligible_count > 0
        or any(d.language == item.language for d in dispositions)
    ]
    language_reports = [
        item for item in language_reports if item.eligible_count > 0
    ]

    clusters = cluster_failures(dispositions)
    unhealthy_reasons: list[str] = []
    partial_reasons: list[str] = []
    for report in language_reports:
        prefix = f"language:{report.language}:"
        if report.status is AnalyzerHealthStatus.UNHEALTHY:
            unhealthy_reasons.extend(prefix + reason for reason in report.reasons)
        elif report.status is AnalyzerHealthStatus.PARTIAL:
            partial_reasons.extend(prefix + reason for reason in report.reasons)

    if run_canaries and not canaries.passed:
        unhealthy_reasons.append("canary_failure")
    if any(
        item.parser_authority is ParserAuthorityKind.REGEX_FORBIDDEN
        for item in eligible
    ):
        unhealthy_reasons.append("regex_authority_forbidden")

    # Completeness: every eligible path must be success or bounded failure.
    incomplete = [
        item
        for item in dispositions
        if item.language
        and item.outcome is PathParseOutcome.NOT_ELIGIBLE
        and item.parser_status in _ELIGIBLE_STATUSES
    ]
    if incomplete:
        unhealthy_reasons.append("eligible_path_missing_disposition")

    if unhealthy_reasons:
        status = AnalyzerHealthStatus.UNHEALTHY
        reasons = tuple(dict.fromkeys(unhealthy_reasons + partial_reasons))
    elif partial_reasons:
        status = AnalyzerHealthStatus.PARTIAL
        reasons = tuple(dict.fromkeys(partial_reasons))
    else:
        status = AnalyzerHealthStatus.HEALTHY
        reasons = ()

    retained = dispositions
    if max_disposition_samples > 0:
        # Prefer retaining failures, then successes, for audit samples.
        failures = [
            item
            for item in dispositions
            if item.outcome is PathParseOutcome.BOUNDED_FAILURE
        ]
        successes = [
            item
            for item in dispositions
            if item.outcome is PathParseOutcome.SUCCESS
        ]
        other = [
            item
            for item in dispositions
            if item.outcome is PathParseOutcome.NOT_ELIGIBLE
        ]
        retained_list = (
            failures[:max_disposition_samples]
            + successes[
                : max(0, max_disposition_samples - min(len(failures), max_disposition_samples))
            ]
        )
        if len(retained_list) < max_disposition_samples:
            retained_list.extend(
                other[: max_disposition_samples - len(retained_list)]
            )
        retained = tuple(retained_list)

    metrics = {
        "tracked_row_count": len(dispositions),
        "eligible_path_count": len(eligible),
        "success_count": sum(
            item.outcome is PathParseOutcome.SUCCESS for item in eligible
        ),
        "bounded_failure_count": sum(
            item.outcome is PathParseOutcome.BOUNDED_FAILURE for item in eligible
        ),
        "not_eligible_count": sum(
            item.outcome is PathParseOutcome.NOT_ELIGIBLE for item in dispositions
        ),
        "cluster_count": len(clusters),
        "language_count": len(language_reports),
        "canaries_passed": canaries.passed if run_canaries else None,
        "authority_repaired": bool(
            authority_repair.repaired if authority_repair else False
        ),
        "js_ts_real_parser": (
            observed_js_ts_authority
            is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
        ),
        "disposition_sample_count": len(retained),
        "disposition_complete": max_disposition_samples <= 0
        or len(retained) >= len(dispositions),
    }
    return PolyglotASTHealthReport(
        status=status,
        reasons=reasons,
        dispositions=retained,
        clusters=clusters,
        language_health=tuple(language_reports),
        canaries=canaries,
        authority_repair=authority_repair,
        thresholds=policy,
        metrics=metrics,
    )


def _normalize_ledger_row(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize coverage, index, or snapshot disposition rows."""

    row = dict(item)
    if "parser_status" not in row and "status" not in row:
        # Snapshot dispositions use `kind` / `reason_code` without parser fields.
        kind = str(row.get("kind") or row.get("disposition_kind") or "").casefold()
        reason = str(row.get("reason_code") or row.get("reason") or "")
        if kind in {"semantic_ast", "structured_data"}:
            # Eligible-for-parser inventory without a parse outcome is incomplete
            # until a later index row supplies success/failure.
            row.setdefault("parser_status", "parse_failure")
            row.setdefault(
                "parser_reason",
                reason or "parser_outcome_missing",
            )
        elif kind in {
            "text_reference",
            "binary_artifact",
            "dependency_artifact",
            "unsupported",
            "excluded",
            "parse_failure",
        }:
            if kind == "parse_failure":
                row.setdefault("parser_status", "parse_failure")
                row.setdefault("parser_reason", reason or "parse_failure")
            elif kind == "unsupported":
                row.setdefault("parser_status", "unsupported")
                row.setdefault("parser_reason", reason or "unsupported")
            else:
                row.setdefault("parser_status", "not_applicable")
                row.setdefault("parser_reason", reason or kind or "not_applicable")
        elif kind:
            row.setdefault("parser_status", "not_applicable")
            row.setdefault("parser_reason", reason or kind)
    if not row.get("language") and row.get("path"):
        row["language"] = _language_from_path(str(row.get("path") or ""))
    return row


def load_coverage_rows(path: str | os.PathLike[str]) -> tuple[dict[str, Any], ...]:
    """Load body-free coverage/index/snapshot rows from a JSON ledger."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("rows")
        if rows is None:
            rows = payload.get("dispositions")
        if rows is None:
            raise PolyglotASTHealthError(
                "coverage document is missing rows or dispositions",
                reason_code="missing_rows",
            )
    else:
        raise PolyglotASTHealthError(
            "coverage document must be an object or array",
            reason_code="invalid_coverage_document",
        )
    if not isinstance(rows, list):
        raise PolyglotASTHealthError(
            "coverage rows must be an array",
            reason_code="invalid_rows",
        )
    return tuple(
        _normalize_ledger_row(item) for item in rows if isinstance(item, Mapping)
    )


def write_polyglot_ast_health_report(
    report: PolyglotASTHealthReport,
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically write a content-addressed health report (no source bodies)."""

    target = Path(path)
    payload = report.to_dict(include_identity=True)
    if report_contains_source_body(payload):
        raise PolyglotASTHealthError(
            "refusing to write report that embeds a source body",
            reason_code="source_body_forbidden",
        )
    encoded = _canonical_json_bytes(payload) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload["content_identity"]


def build_health_report_from_coverage(
    coverage_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    provider: PolyglotASTProvider | None = None,
    repair_authority: bool = True,
    run_canaries: bool = True,
    max_disposition_samples: int = _DEFAULT_MAX_PATH_SAMPLES,
    search_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> PolyglotASTHealthReport:
    """Assess a coverage/index ledger and optionally persist the receipt."""

    rows = load_coverage_rows(coverage_path)
    report = assess_polyglot_ast_health(
        rows,
        provider=provider,
        repair_authority=repair_authority,
        run_canaries=run_canaries,
        max_disposition_samples=max_disposition_samples,
        search_roots=search_roots,
    )
    if output_path is not None:
        write_polyglot_ast_health_report(report, output_path)
    return report


__all__ = [
    "DEFAULT_LANGUAGE_THRESHOLDS",
    "POLYGLOT_AST_HEALTH_EVIDENCE",
    "POLYGLOT_AST_HEALTH_INTERFACE",
    "POLYGLOT_AST_HEALTH_SCHEMA",
    "FailureCluster",
    "LanguageHealthReport",
    "LanguageHealthThresholds",
    "ParserAuthorityKind",
    "ParserAuthorityRepair",
    "PathDispositionRecord",
    "PathParseOutcome",
    "PolyglotASTHealthError",
    "PolyglotASTHealthReport",
    "PolyglotCanaryReport",
    "PolyglotCanaryResult",
    "assess_polyglot_ast_health",
    "build_health_report_from_coverage",
    "classify_parser_authority",
    "classify_path_disposition",
    "classify_path_dispositions",
    "cluster_failures",
    "discover_typescript_path",
    "evaluate_language_health",
    "js_ts_uses_real_parser",
    "load_coverage_rows",
    "repair_polyglot_parser_authority",
    "report_contains_source_body",
    "run_polyglot_ast_canaries",
    "typed_reason_code",
    "write_polyglot_ast_health_report",
]
