"""Bounded, trust-aware proof context capsules for Codex and Leanstral.

Capsules are projections, never graph dumps.  Selection starts at exact
identifiers in the queryable evidence store, follows a small allowlisted graph
neighborhood, and then copies only reviewed public fields into the prompt.
Every budget is applied while constructing the capsule, before ``to_prompt``
can expose it to a model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifact_store import (
    MAX_GRAPH_QUERY_HOPS,
    MAX_QUERY_ROWS,
    query_code_evidence_neighborhood,
)


PROOF_CONTEXT_CAPSULE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.proof-context-capsule@1"
)
PROOF_CONTEXT_QUERY_SCHEMA = "ipfs_accelerate_py.agent_supervisor.proof-context-query@1"
PROOF_CONTEXT_LIMITS_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.proof-context-limits@1"
)

DEFAULT_MAX_CONTEXT_ROWS = 96
DEFAULT_MAX_CONTEXT_BYTES = 48 * 1024
DEFAULT_MAX_CONTEXT_TOKENS = 12_000
DEFAULT_MAX_GRAPH_HOPS = 2
DEFAULT_MAX_SOURCE_EXCERPTS = 8
DEFAULT_MAX_SOURCE_EXCERPT_BYTES = 4 * 1024
DEFAULT_MAX_SOURCE_BYTES = 16 * 1024
DEFAULT_MAX_PROOF_TRANSCRIPTS = 4
DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES = 2 * 1024
DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES_TOTAL = 6 * 1024


class ProofContextError(ValueError):
    """Base error for invalid or unsafe proof-context requests."""


class ProofContextBudgetError(ProofContextError):
    """Raised when even the mandatory capsule envelope cannot fit its budget."""


class ProofContextTarget(str, Enum):
    CODEX = "codex"
    LEANSTRAL = "leanstral"


class ContextTrust(str, Enum):
    TRUSTED_FACT = "trusted_fact"
    UNTRUSTED_SUGGESTION = "untrusted_suggestion"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    REQUIRED_FALLBACK_CHECK = "required_fallback_check"


def estimate_context_tokens(text: str) -> int:
    """Return the stable tokenizer-independent estimate used by budgets.

    Four UTF-8 bytes per token is the conventional prompt planning estimate.
    Callers with an exact deployed tokenizer may inject a stricter counter into
    :class:`ProofContextBuilder`.
    """

    size = len(text.encode("utf-8"))
    return (size + 3) // 4


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _exact(values: Sequence[str] | str, *, label: str) -> tuple[str, ...]:
    result = _strings(values)
    if any(item in {"*", "%"} for item in result):
        raise ProofContextError(f"{label} selectors must be exact")
    return result


@dataclass(frozen=True)
class ProofContextLimits:
    """Hard pre-assembly limits for one model capsule."""

    max_rows: int = DEFAULT_MAX_CONTEXT_ROWS
    max_bytes: int = DEFAULT_MAX_CONTEXT_BYTES
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_graph_hops: int = DEFAULT_MAX_GRAPH_HOPS
    max_source_excerpts: int = DEFAULT_MAX_SOURCE_EXCERPTS
    max_source_excerpt_bytes: int = DEFAULT_MAX_SOURCE_EXCERPT_BYTES
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES
    max_proof_transcripts: int = DEFAULT_MAX_PROOF_TRANSCRIPTS
    max_proof_transcript_bytes: int = DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES
    max_proof_transcript_bytes_total: int = DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES_TOTAL

    def __post_init__(self) -> None:
        for name in (
            "max_rows",
            "max_bytes",
            "max_tokens",
            "max_source_excerpt_bytes",
            "max_source_bytes",
            "max_proof_transcript_bytes",
            "max_proof_transcript_bytes_total",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ProofContextError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("max_source_excerpts", "max_proof_transcripts"):
            value = int(getattr(self, name))
            if value < 0:
                raise ProofContextError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        hops = int(self.max_graph_hops)
        if hops < 0 or hops > MAX_GRAPH_QUERY_HOPS:
            raise ProofContextError(
                f"max_graph_hops must be between 0 and {MAX_GRAPH_QUERY_HOPS}"
            )
        object.__setattr__(self, "max_graph_hops", hops)
        if self.max_rows > MAX_QUERY_ROWS:
            object.__setattr__(self, "max_rows", MAX_QUERY_ROWS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_CONTEXT_LIMITS_SCHEMA,
            "max_rows": self.max_rows,
            "max_bytes": self.max_bytes,
            "max_tokens": self.max_tokens,
            "max_graph_hops": self.max_graph_hops,
            "max_source_excerpts": self.max_source_excerpts,
            "max_source_excerpt_bytes": self.max_source_excerpt_bytes,
            "max_source_bytes": self.max_source_bytes,
            "max_proof_transcripts": self.max_proof_transcripts,
            "max_proof_transcript_bytes": self.max_proof_transcript_bytes,
            "max_proof_transcript_bytes_total": (self.max_proof_transcript_bytes_total),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofContextLimits":
        schema = str(payload.get("schema") or PROOF_CONTEXT_LIMITS_SCHEMA)
        if schema != PROOF_CONTEXT_LIMITS_SCHEMA:
            raise ProofContextError(f"unsupported context limits schema: {schema}")
        return cls(
            max_rows=int(payload.get("max_rows", DEFAULT_MAX_CONTEXT_ROWS)),
            max_bytes=int(payload.get("max_bytes", DEFAULT_MAX_CONTEXT_BYTES)),
            max_tokens=int(payload.get("max_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            max_graph_hops=int(payload.get("max_graph_hops", DEFAULT_MAX_GRAPH_HOPS)),
            max_source_excerpts=int(
                payload.get("max_source_excerpts", DEFAULT_MAX_SOURCE_EXCERPTS)
            ),
            max_source_excerpt_bytes=int(
                payload.get(
                    "max_source_excerpt_bytes",
                    DEFAULT_MAX_SOURCE_EXCERPT_BYTES,
                )
            ),
            max_source_bytes=int(
                payload.get("max_source_bytes", DEFAULT_MAX_SOURCE_BYTES)
            ),
            max_proof_transcripts=int(
                payload.get("max_proof_transcripts", DEFAULT_MAX_PROOF_TRANSCRIPTS)
            ),
            max_proof_transcript_bytes=int(
                payload.get(
                    "max_proof_transcript_bytes",
                    DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES,
                )
            ),
            max_proof_transcript_bytes_total=int(
                payload.get(
                    "max_proof_transcript_bytes_total",
                    DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES_TOTAL,
                )
            ),
        )


ContextBudget = ProofContextLimits
ProofContextBudget = ProofContextLimits


@dataclass(frozen=True)
class ProofContextQuery:
    """Exact selectors for a task-local proof neighborhood."""

    task_id: str
    symbols: tuple[str, ...] = ()
    dependency_task_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        task_id = str(self.task_id or "").strip()
        if not task_id or task_id in {"*", "%"}:
            raise ProofContextError("task_id must be one exact identifier")
        object.__setattr__(self, "task_id", task_id)
        for name, label in (
            ("symbols", "symbol"),
            ("dependency_task_ids", "dependency task"),
            ("obligation_ids", "obligation"),
            ("receipt_ids", "receipt"),
            ("contradiction_ids", "contradiction"),
        ):
            object.__setattr__(self, name, _exact(getattr(self, name), label=label))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_CONTEXT_QUERY_SCHEMA,
            "task_id": self.task_id,
            "symbols": list(self.symbols),
            "dependency_task_ids": list(self.dependency_task_ids),
            "obligation_ids": list(self.obligation_ids),
            "receipt_ids": list(self.receipt_ids),
            "contradiction_ids": list(self.contradiction_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofContextQuery":
        schema = str(payload.get("schema") or PROOF_CONTEXT_QUERY_SCHEMA)
        if schema != PROOF_CONTEXT_QUERY_SCHEMA:
            raise ProofContextError(f"unsupported context query schema: {schema}")
        return cls(
            task_id=str(payload.get("task_id") or ""),
            symbols=tuple(payload.get("symbols") or ()),
            dependency_task_ids=tuple(payload.get("dependency_task_ids") or ()),
            obligation_ids=tuple(payload.get("obligation_ids") or ()),
            receipt_ids=tuple(payload.get("receipt_ids") or ()),
            contradiction_ids=tuple(payload.get("contradiction_ids") or ()),
        )


@dataclass(frozen=True)
class ContextEntry:
    """One compact public claim with an explicit trust class."""

    trust: ContextTrust
    kind: str
    record_id: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trust = (
            self.trust
            if isinstance(self.trust, ContextTrust)
            else ContextTrust(str(self.trust))
        )
        object.__setattr__(self, "trust", trust)
        for name in ("kind", "record_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProofContextError(f"context entry {name} is required")
            object.__setattr__(self, name, value)
        if not isinstance(self.fields, Mapping):
            raise ProofContextError("context entry fields must be a mapping")
        object.__setattr__(self, "fields", _public_mapping(self.fields))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust": self.trust.value,
            "kind": self.kind,
            "record_id": self.record_id,
            "fields": dict(self.fields),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextEntry":
        return cls(
            trust=ContextTrust(str(payload.get("trust") or "")),
            kind=str(payload.get("kind") or ""),
            record_id=str(payload.get("record_id") or ""),
            fields=payload.get("fields") or {},
        )


@dataclass(frozen=True)
class SourceExcerpt:
    symbol: str
    path: str
    text: str
    start_line: int = 1
    end_line: int = 1

    def __post_init__(self) -> None:
        symbol = str(self.symbol or "").strip()
        if not symbol:
            raise ProofContextError("source excerpt symbol is required")
        path = str(self.path or "").strip().replace("\\", "/")
        if path:
            normalized = PurePosixPath(path)
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ProofContextError(
                    "source excerpt path must be repository-relative"
                )
            path = normalized.as_posix()
        text = str(self.text)
        start = max(1, int(self.start_line))
        end = max(start, int(self.end_line))
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "start_line", start)
        object.__setattr__(self, "end_line", end)

    @property
    def byte_count(self) -> int:
        return len(self.text.encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceExcerpt":
        result = cls(
            symbol=str(payload.get("symbol") or ""),
            path=str(payload.get("path") or ""),
            text=str(payload.get("text") or ""),
            start_line=int(payload.get("start_line") or 1),
            end_line=int(payload.get("end_line") or payload.get("start_line") or 1),
        )
        if "byte_count" in payload and int(payload["byte_count"]) != result.byte_count:
            raise ProofContextError("source excerpt byte_count does not match text")
        return result


@dataclass(frozen=True)
class ProofTranscriptExcerpt:
    receipt_id: str
    obligation_id: str
    text: str

    def __post_init__(self) -> None:
        receipt_id = str(self.receipt_id or "").strip()
        if not receipt_id:
            raise ProofContextError("proof transcript receipt_id is required")
        object.__setattr__(self, "receipt_id", receipt_id)
        object.__setattr__(self, "obligation_id", str(self.obligation_id or "").strip())
        object.__setattr__(self, "text", str(self.text))

    @property
    def byte_count(self) -> int:
        return len(self.text.encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "obligation_id": self.obligation_id,
            "text": self.text,
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofTranscriptExcerpt":
        result = cls(
            receipt_id=str(payload.get("receipt_id") or ""),
            obligation_id=str(payload.get("obligation_id") or ""),
            text=str(payload.get("text") or ""),
        )
        if not result.receipt_id:
            raise ProofContextError("proof transcript receipt_id is required")
        if "byte_count" in payload and int(payload["byte_count"]) != result.byte_count:
            raise ProofContextError("proof transcript byte_count does not match text")
        return result


@dataclass(frozen=True)
class ProofContextUsage:
    rows: int = 0
    bytes: int = 0
    tokens: int = 0
    graph_hops: int = 0
    source_excerpts: int = 0
    source_bytes: int = 0
    proof_transcripts: int = 0
    proof_transcript_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "rows",
            "bytes",
            "tokens",
            "graph_hops",
            "source_excerpts",
            "source_bytes",
            "proof_transcripts",
            "proof_transcript_bytes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ProofContextError("context usage values must be non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "bytes": self.bytes,
            "tokens": self.tokens,
            "graph_hops": self.graph_hops,
            "source_excerpts": self.source_excerpts,
            "source_bytes": self.source_bytes,
            "proof_transcripts": self.proof_transcripts,
            "proof_transcript_bytes": self.proof_transcript_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofContextUsage":
        values = {
            name: int(payload.get(name) or 0)
            for name in (
                "rows",
                "bytes",
                "tokens",
                "graph_hops",
                "source_excerpts",
                "source_bytes",
                "proof_transcripts",
                "proof_transcript_bytes",
            )
        }
        if any(value < 0 for value in values.values()):
            raise ProofContextError("context usage values must be non-negative")
        return cls(**values)


@dataclass(frozen=True)
class ProofContextCapsule:
    """Final immutable prompt artifact with explicit trust partitions."""

    target: ProofContextTarget
    query: ProofContextQuery
    limits: ProofContextLimits
    trusted_facts: tuple[ContextEntry, ...] = ()
    untrusted_suggestions: tuple[ContextEntry, ...] = ()
    unsupported_semantics: tuple[ContextEntry, ...] = ()
    required_fallback_checks: tuple[str, ...] = ()
    source_excerpts: tuple[SourceExcerpt, ...] = ()
    proof_transcripts: tuple[ProofTranscriptExcerpt, ...] = ()
    usage: ProofContextUsage = field(default_factory=ProofContextUsage)
    truncated: bool = False
    omitted: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = (
            self.target
            if isinstance(self.target, ProofContextTarget)
            else ProofContextTarget(str(self.target))
        )
        object.__setattr__(self, "target", target)
        if not isinstance(self.query, ProofContextQuery):
            raise ProofContextError("capsule query must be a ProofContextQuery")
        if not isinstance(self.limits, ProofContextLimits):
            raise ProofContextError("capsule limits must be ProofContextLimits")
        for name, item_type in (
            ("trusted_facts", ContextEntry),
            ("untrusted_suggestions", ContextEntry),
            ("unsupported_semantics", ContextEntry),
            ("source_excerpts", SourceExcerpt),
            ("proof_transcripts", ProofTranscriptExcerpt),
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(item, item_type) for item in values):
                raise ProofContextError(f"capsule {name} has an invalid entry")
            object.__setattr__(self, name, values)
        if any(
            item.trust is not ContextTrust.TRUSTED_FACT for item in self.trusted_facts
        ):
            raise ProofContextError("trusted_facts contains a non-trusted entry")
        if any(
            item.trust is not ContextTrust.UNTRUSTED_SUGGESTION
            for item in self.untrusted_suggestions
        ):
            raise ProofContextError(
                "untrusted_suggestions contains a differently classified entry"
            )
        if any(
            item.trust is not ContextTrust.UNSUPPORTED_SEMANTICS
            for item in self.unsupported_semantics
        ):
            raise ProofContextError(
                "unsupported_semantics contains a differently classified entry"
            )
        object.__setattr__(
            self,
            "required_fallback_checks",
            _strings(self.required_fallback_checks),
        )
        if not isinstance(self.usage, ProofContextUsage):
            raise ProofContextError("capsule usage must be ProofContextUsage")
        omitted = {
            str(name): int(value)
            for name, value in self.omitted.items()
            if int(value) >= 0
        }
        object.__setattr__(self, "omitted", omitted)

    @property
    def capsule_id(self) -> str:
        payload = self._dict(include_identity=False)
        payload["usage"] = {
            **self.usage.to_dict(),
            # Size counters describe this serialization and are not semantic
            # inputs to its own content identity.
            "bytes": 0,
            "tokens": 0,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return f"proof-context:sha256:{digest}"

    def _dict(self, *, include_identity: bool) -> dict[str, Any]:
        result = {
            "schema": PROOF_CONTEXT_CAPSULE_SCHEMA,
            "target": self.target.value,
            "query": self.query.to_dict(),
            "limits": self.limits.to_dict(),
            "trusted_facts": [item.to_dict() for item in self.trusted_facts],
            "untrusted_suggestions": [
                item.to_dict() for item in self.untrusted_suggestions
            ],
            "unsupported_semantics": [
                item.to_dict() for item in self.unsupported_semantics
            ],
            "required_fallback_checks": list(self.required_fallback_checks),
            "source_excerpts": [item.to_dict() for item in self.source_excerpts],
            "proof_transcripts": [item.to_dict() for item in self.proof_transcripts],
            "usage": self.usage.to_dict(),
            "truncated": self.truncated,
            "omitted": dict(sorted(self.omitted.items())),
        }
        if include_identity:
            result["capsule_id"] = self.capsule_id
        return result

    def to_dict(self) -> dict[str, Any]:
        return self._dict(include_identity=True)

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_prompt(self) -> str:
        """Return the already-budgeted prompt; no late graph data is loaded."""

        return self.to_json()

    def render_for_codex(self) -> str:
        return self.to_prompt()

    def render_for_leanstral(self) -> str:
        return self.to_prompt()

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> "ProofContextCapsule":
        schema = str(payload.get("schema") or "")
        if schema != PROOF_CONTEXT_CAPSULE_SCHEMA:
            raise ProofContextError(f"unsupported proof context schema: {schema}")
        query = payload.get("query")
        limits = payload.get("limits")
        usage = payload.get("usage")
        if not all(isinstance(value, Mapping) for value in (query, limits, usage)):
            raise ProofContextError("capsule query, limits, and usage must be mappings")
        result = cls(
            target=ProofContextTarget(str(payload.get("target") or "")),
            query=ProofContextQuery.from_dict(query),
            limits=ProofContextLimits.from_dict(limits),
            trusted_facts=tuple(
                ContextEntry.from_dict(item)
                for item in payload.get("trusted_facts") or ()
            ),
            untrusted_suggestions=tuple(
                ContextEntry.from_dict(item)
                for item in payload.get("untrusted_suggestions") or ()
            ),
            unsupported_semantics=tuple(
                ContextEntry.from_dict(item)
                for item in payload.get("unsupported_semantics") or ()
            ),
            required_fallback_checks=tuple(
                payload.get("required_fallback_checks") or ()
            ),
            source_excerpts=tuple(
                SourceExcerpt.from_dict(item)
                for item in payload.get("source_excerpts") or ()
            ),
            proof_transcripts=tuple(
                ProofTranscriptExcerpt.from_dict(item)
                for item in payload.get("proof_transcripts") or ()
            ),
            usage=ProofContextUsage.from_dict(usage),
            truncated=bool(payload.get("truncated")),
            omitted=payload.get("omitted") or {},
        )
        claimed_id = str(payload.get("capsule_id") or "")
        if claimed_id and claimed_id != result.capsule_id:
            raise ProofContextError("capsule identity does not match payload")
        text = result.to_json()
        actual_bytes = len(text.encode("utf-8"))
        actual_tokens = int(token_counter(text))
        if actual_bytes != result.usage.bytes:
            raise ProofContextError("capsule byte usage does not match payload")
        if actual_tokens != result.usage.tokens:
            raise ProofContextError("capsule token usage does not match payload")
        if actual_bytes > result.limits.max_bytes:
            raise ProofContextBudgetError("capsule exceeds its byte limit")
        if actual_tokens > result.limits.max_tokens:
            raise ProofContextBudgetError("capsule exceeds its token limit")
        if result.usage.rows > result.limits.max_rows:
            raise ProofContextBudgetError("capsule exceeds its row limit")
        if len(result.source_excerpts) > result.limits.max_source_excerpts:
            raise ProofContextBudgetError("capsule exceeds its source excerpt limit")
        if result.usage.source_bytes > result.limits.max_source_bytes:
            raise ProofContextBudgetError("capsule exceeds its source byte limit")
        if len(result.proof_transcripts) > result.limits.max_proof_transcripts:
            raise ProofContextBudgetError("capsule exceeds its proof transcript limit")
        if (
            result.usage.proof_transcript_bytes
            > result.limits.max_proof_transcript_bytes_total
        ):
            raise ProofContextBudgetError(
                "capsule exceeds its proof transcript byte limit"
            )
        return result

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> "ProofContextCapsule":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProofContextError("proof context JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ProofContextError("proof context JSON must contain an object")
        return cls.from_dict(payload, token_counter=token_counter)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


_FORBIDDEN_KEY_PARTS = (
    "witness",
    "private_premise",
    "secret",
    "credential",
    "full_ast",
    "repository_ast",
)


def _public_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return _public_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_public_value(item) for item in value[:64]]
    return str(value)


def _public_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key in sorted(value, key=str):
        key = str(raw_key)
        lowered = key.lower()
        if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
            continue
        if "transcript" in lowered or lowered in {"stdout", "stderr", "proof_log"}:
            continue
        result[key] = _public_value(value[raw_key])
    return result


_PUBLIC_FIELDS: Mapping[str, tuple[str, ...]] = {
    "task": (
        "task_id",
        "canonical_task_id",
        "canonical_task_cid",
        "title",
        "status",
        "priority",
        "goal_id",
        "subgoal_id",
        "depends_on",
        "acceptance",
        "acceptance_criteria",
        "validation",
        "validation_commands",
        "promotion_policy",
    ),
    "symbol": (
        "qualified_symbol",
        "symbol",
        "path",
        "file_path",
        "source_hash",
        "source_sha256",
        "symbol_hash",
        "line_start",
        "line_end",
        "start_line",
        "end_line",
    ),
    "obligation": (
        "obligation_id",
        "task_id",
        "statement",
        "canonical_statement",
        "template_id",
        "template_version",
        "template_semantic_hash",
        "assumptions",
        "premise_ids",
        "required_assurance",
        "supported_backends",
        "status",
        "support_status",
        "unsupported_semantics",
        "fallback_checks",
        "fallback_tests",
        "required_tests",
        "promotion_policy",
        "counterexample",
        "counterexamples",
        "unsat_core",
    ),
    "proof": (
        "receipt_id",
        "proof_id",
        "attempt_id",
        "task_id",
        "obligation_id",
        "repository_tree_id",
        "verdict",
        "status",
        "solver_verdict",
        "kernel_accepted",
        "authoritative_assurance",
        "assurance",
        "freshness",
        "counterexample",
        "counterexamples",
        "unsat_core",
        "diagnostics",
    ),
    "validation": (
        "validation_receipt_id",
        "receipt_id",
        "task_id",
        "obligation_ids",
        "command",
        "validation_command",
        "status",
        "passed",
        "freshness",
        "contradiction_id",
        "contradiction",
        "failure_reason",
    ),
    "merge": (
        "merge_receipt_id",
        "receipt_cid",
        "task_id",
        "repository_tree_id",
        "status",
        "completion_status",
        "freshness",
        "contradiction_id",
        "contradiction",
    ),
    "enrichment": (
        "enrichment_id",
        "id",
        "source",
        "summary",
        "claim",
        "suggestion",
        "reason",
        "target",
        "targets",
        "contradiction_id",
    ),
    "evidence": (
        "evidence_id",
        "content_id",
        "task_id",
        "obligation_id",
        "kind",
        "status",
        "freshness",
        "contradiction_id",
        "contradiction",
        "summary",
    ),
}


def _node_record(node: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = node.get("payload")
    if not isinstance(payload, Mapping):
        return {}
    record = payload.get("record")
    return record if isinstance(record, Mapping) else {}


def _entry_from_node(node: Mapping[str, Any]) -> ContextEntry:
    kind = str(node.get("node_kind") or "")
    record = _node_record(node)
    fields = {
        name: _public_value(record[name])
        for name in _PUBLIC_FIELDS.get(kind, ())
        if name in record
    }
    # Indexed identities remain available even for reference-only records.
    for name in ("task_id", "symbol", "obligation_id", "assurance", "freshness"):
        value = node.get(name)
        if value not in (None, "") and name not in fields:
            fields[name] = _public_value(value)
    trust = (
        ContextTrust.UNTRUSTED_SUGGESTION
        if not bool(node.get("authoritative")) or kind == "enrichment"
        else ContextTrust.TRUSTED_FACT
    )
    return ContextEntry(
        trust=trust,
        kind=kind,
        record_id=str(node.get("record_key") or node.get("node_id") or ""),
        fields=fields,
    )


def _unsupported_entries(
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[ContextEntry, ...]:
    entries: dict[tuple[str, str], ContextEntry] = {}
    for node in nodes:
        record = _node_record(node)
        status = str(
            record.get("support_status")
            or record.get("translation_status")
            or record.get("status")
            or ""
        ).lower()
        semantics = record.get("unsupported_semantics")
        if not semantics and status not in {"unsupported", "ambiguous"}:
            continue
        text_values = _strings(semantics) or (
            str(record.get("unsupported_reason") or record.get("reason") or status),
        )
        fields = {
            "obligation_id": str(
                node.get("obligation_id") or record.get("obligation_id") or ""
            ),
            "reasons": list(text_values),
            "support_status": status or "unsupported",
        }
        entry = ContextEntry(
            trust=ContextTrust.UNSUPPORTED_SEMANTICS,
            kind="unsupported_semantics",
            record_id=str(node.get("record_key") or node.get("node_id") or ""),
            fields=fields,
        )
        entries[(entry.kind, entry.record_id)] = entry
    return tuple(entries[key] for key in sorted(entries))


def _fallback_checks(
    nodes: Sequence[Mapping[str, Any]],
    template_records: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    checks: set[str] = set()
    for record in [*(_node_record(node) for node in nodes), *template_records]:
        for name in (
            "fallback_checks",
            "fallback_tests",
            "required_tests",
            "validation_commands",
        ):
            checks.update(_strings(record.get(name)))
    return tuple(sorted(checks))


def _template_records(
    templates: Any, nodes: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    if templates is None:
        return ()
    values = getattr(templates, "templates", templates)
    if isinstance(values, Mapping):
        iterable: Iterable[Any] = values.values()
    else:
        iterable = values
    requested = {
        str(_node_record(node).get("template_id") or "").strip()
        for node in nodes
        if str(_node_record(node).get("template_id") or "").strip()
    }
    if not requested:
        return ()
    records: list[Mapping[str, Any]] = []
    for value in iterable:
        to_dict = getattr(value, "to_dict", None)
        record = to_dict() if callable(to_dict) else value
        if not isinstance(record, Mapping):
            continue
        template_id = str(record.get("template_id") or "").strip()
        if template_id not in requested:
            continue
        public = {
            name: _public_value(record[name])
            for name in (
                "template_id",
                "version",
                "canonical_statement",
                "semantic_hash",
                "supported_backends",
                "assumptions",
                "fallback_tests",
                "supported_code_shapes",
            )
            if name in record
        }
        records.append(public)
    return tuple(
        sorted(
            records,
            key=lambda item: (str(item.get("template_id")), str(item.get("version"))),
        )
    )


def _truncate_utf8(text: str, maximum: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum:
        return text, False
    marker = "\n…[truncated]"
    marker_bytes = marker.encode("utf-8")
    if maximum <= len(marker_bytes):
        return encoded[:maximum].decode("utf-8", errors="ignore"), True
    prefix = encoded[: maximum - len(marker_bytes)].decode("utf-8", errors="ignore")
    return prefix + marker, True


def _source_path(record: Mapping[str, Any]) -> str:
    raw = str(record.get("path") or record.get("file_path") or "").replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        return ""
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        return ""
    return path.as_posix()


class ProofContextBuilder:
    """Build capsules from the DuckDB query plane and optional source root."""

    def __init__(
        self,
        graph_path: Path | str,
        *,
        source_root: Path | str | None = None,
        templates: Any = None,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.source_root = (
            Path(source_root).resolve() if source_root is not None else None
        )
        self.templates = templates
        if not callable(token_counter):
            raise ProofContextError("token_counter must be callable")
        self.token_counter = token_counter

    def _measure(self, capsule: ProofContextCapsule) -> tuple[int, int]:
        text = capsule.to_json()
        return len(text.encode("utf-8")), int(self.token_counter(text))

    def _with_usage(
        self, capsule: ProofContextCapsule, *, rows: int
    ) -> ProofContextCapsule:
        result = capsule
        # The decimal size fields affect serialization length.  Iterate to the
        # small fixed point so reported usage equals the actual prompt.
        for _ in range(8):
            byte_count, token_count = self._measure(result)
            usage = ProofContextUsage(
                rows=rows,
                bytes=byte_count,
                tokens=token_count,
                graph_hops=result.limits.max_graph_hops,
                source_excerpts=len(result.source_excerpts),
                source_bytes=sum(item.byte_count for item in result.source_excerpts),
                proof_transcripts=len(result.proof_transcripts),
                proof_transcript_bytes=sum(
                    item.byte_count for item in result.proof_transcripts
                ),
            )
            updated = replace(result, usage=usage)
            if updated.usage == result.usage:
                return updated
            result = updated
        return result

    def _fits(self, capsule: ProofContextCapsule, *, rows: int) -> bool:
        measured = self._with_usage(capsule, rows=rows)
        return (
            measured.usage.bytes <= measured.limits.max_bytes
            and measured.usage.tokens <= measured.limits.max_tokens
        )

    def _automatic_sources(
        self, nodes: Sequence[Mapping[str, Any]], query: ProofContextQuery
    ) -> dict[str, SourceExcerpt]:
        if self.source_root is None:
            return {}
        result: dict[str, SourceExcerpt] = {}
        selected_symbols = set(query.symbols)
        for node in nodes:
            if str(node.get("node_kind")) != "symbol":
                continue
            symbol = str(node.get("symbol") or "")
            if selected_symbols and symbol not in selected_symbols:
                continue
            record = _node_record(node)
            relative = _source_path(record)
            if not symbol or not relative:
                continue
            candidate = (self.source_root / relative).resolve()
            try:
                candidate.relative_to(self.source_root)
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(
                1,
                int(record.get("line_start") or record.get("start_line") or 1),
            )
            end = int(record.get("line_end") or record.get("end_line") or start)
            end = max(start, min(end, len(lines)))
            text = "\n".join(lines[start - 1 : end])
            result[symbol] = SourceExcerpt(symbol, relative, text, start, end)
        return result

    def build(
        self,
        query: ProofContextQuery | None = None,
        *,
        task_id: str = "",
        symbols: Sequence[str] | str = (),
        dependency_task_ids: Sequence[str] | str = (),
        obligation_ids: Sequence[str] | str = (),
        receipt_ids: Sequence[str] | str = (),
        contradiction_ids: Sequence[str] | str = (),
        target: ProofContextTarget | str = ProofContextTarget.CODEX,
        limits: ProofContextLimits | None = None,
        source_excerpts: Mapping[str, Any] | Sequence[SourceExcerpt] | None = None,
    ) -> ProofContextCapsule:
        if query is None:
            query = ProofContextQuery(
                task_id=task_id,
                symbols=tuple(_strings(symbols)),
                dependency_task_ids=tuple(_strings(dependency_task_ids)),
                obligation_ids=tuple(_strings(obligation_ids)),
                receipt_ids=tuple(_strings(receipt_ids)),
                contradiction_ids=tuple(_strings(contradiction_ids)),
            )
        elif task_id or any(
            (
                symbols,
                dependency_task_ids,
                obligation_ids,
                receipt_ids,
                contradiction_ids,
            )
        ):
            raise ProofContextError("pass either query or selector keywords, not both")
        resolved_target = (
            target
            if isinstance(target, ProofContextTarget)
            else ProofContextTarget(str(target))
        )
        budget = limits or ProofContextLimits()
        neighborhood = query_code_evidence_neighborhood(
            self.graph_path,
            task_id=query.task_id,
            symbols=query.symbols,
            dependency_task_ids=query.dependency_task_ids,
            obligation_ids=query.obligation_ids,
            receipt_ids=query.receipt_ids,
            contradiction_ids=query.contradiction_ids,
            max_hops=budget.max_graph_hops,
            limit=budget.max_rows,
        )
        nodes = tuple(neighborhood["nodes"])
        if not any(
            node.get("node_kind") == "task" and node.get("task_id") == query.task_id
            for node in nodes
        ):
            raise ProofContextError(f"exact task was not found: {query.task_id}")
        selector_indexes = {
            "symbols": {
                str(node.get("symbol") or "")
                for node in nodes
                if node.get("node_kind") == "symbol"
            },
            "dependency_task_ids": {
                str(node.get("task_id") or "")
                for node in nodes
                if node.get("node_kind") == "task"
            },
            "obligation_ids": {
                str(node.get("obligation_id") or "")
                for node in nodes
                if node.get("node_kind") == "obligation"
            },
            "receipt_ids": {
                str(node.get("record_key") or "")
                for node in nodes
                if node.get("node_kind") in {"proof", "validation", "merge"}
            },
            "contradiction_ids": {
                value
                for node in nodes
                for value in (
                    str(node.get("record_key") or ""),
                    str(_node_record(node).get("contradiction_id") or ""),
                    str(_node_record(node).get("source_receipt_id") or ""),
                )
                if value
            },
        }
        for name in (
            "symbols",
            "dependency_task_ids",
            "obligation_ids",
            "receipt_ids",
            "contradiction_ids",
        ):
            missing = sorted(set(getattr(query, name)) - selector_indexes[name])
            if missing:
                raise ProofContextError(
                    f"exact {name} selectors were not found: {', '.join(missing)}"
                )

        mandatory_record_ids = {
            str(node.get("record_key") or "")
            for node in nodes
            if (
                node.get("node_kind") == "task"
                and (
                    node.get("task_id") == query.task_id
                    or node.get("task_id") in query.dependency_task_ids
                )
            )
            or (
                node.get("node_kind") == "symbol"
                and node.get("symbol") in query.symbols
            )
            or (
                node.get("node_kind") == "obligation"
                and node.get("obligation_id") in query.obligation_ids
            )
            or str(node.get("record_key") or "") in query.receipt_ids
            or any(
                value in query.contradiction_ids
                for value in (
                    str(node.get("record_key") or ""),
                    str(_node_record(node).get("contradiction_id") or ""),
                    str(_node_record(node).get("source_receipt_id") or ""),
                )
            )
        }

        templates = _template_records(self.templates, nodes)
        trusted: list[ContextEntry] = []
        suggestions: list[ContextEntry] = []
        for node in sorted(
            nodes,
            key=lambda item: (
                0 if item.get("task_id") == query.task_id else 1,
                {
                    "task": 0,
                    "symbol": 1,
                    "obligation": 2,
                    "proof": 3,
                    "validation": 4,
                    "merge": 5,
                    "evidence": 6,
                    "enrichment": 7,
                }.get(str(item.get("node_kind")), 9),
                str(item.get("record_key")),
            ),
        ):
            entry = _entry_from_node(node)
            (
                suggestions
                if entry.trust is ContextTrust.UNTRUSTED_SUGGESTION
                else trusted
            ).append(entry)
        for template in templates:
            trusted.append(
                ContextEntry(
                    trust=ContextTrust.TRUSTED_FACT,
                    kind="invariant_template",
                    record_id=(
                        f"{template.get('template_id', '')}@{template.get('version', '')}"
                    ),
                    fields=template,
                )
            )
        unsupported = list(_unsupported_entries(nodes))
        fallback_checks = list(_fallback_checks(nodes, templates))

        # Normalize only exact selected symbol excerpts.  A mapping's key is
        # always the symbol selector; arbitrary path keys cannot inject text.
        source_map = self._automatic_sources(nodes, query)
        if source_excerpts is not None:
            supplied: Iterable[SourceExcerpt]
            if isinstance(source_excerpts, Mapping):
                normalized: list[SourceExcerpt] = []
                symbol_records = {
                    str(node.get("symbol")): _node_record(node)
                    for node in nodes
                    if node.get("node_kind") == "symbol"
                }
                for symbol, value in source_excerpts.items():
                    symbol_text = str(symbol)
                    if symbol_text not in symbol_records:
                        continue
                    if isinstance(value, SourceExcerpt):
                        excerpt = value
                    elif isinstance(value, Mapping):
                        excerpt = SourceExcerpt(
                            symbol=symbol_text,
                            path=str(
                                value.get("path")
                                or _source_path(symbol_records[symbol_text])
                            ),
                            text=str(value.get("text") or ""),
                            start_line=int(value.get("start_line") or 1),
                            end_line=int(
                                value.get("end_line") or value.get("start_line") or 1
                            ),
                        )
                    else:
                        excerpt = SourceExcerpt(
                            symbol=symbol_text,
                            path=_source_path(symbol_records[symbol_text]),
                            text=str(value),
                        )
                    normalized.append(excerpt)
                supplied = normalized
            else:
                supplied = source_excerpts
            selected_node_symbols = {
                str(node.get("symbol"))
                for node in nodes
                if node.get("node_kind") == "symbol"
            }
            for excerpt in supplied:
                if (
                    isinstance(excerpt, SourceExcerpt)
                    and excerpt.symbol in selected_node_symbols
                ):
                    source_map[excerpt.symbol] = excerpt

        excerpts: list[SourceExcerpt] = []
        source_bytes = 0
        source_omitted = 0
        for symbol in sorted(source_map):
            if len(excerpts) >= budget.max_source_excerpts:
                source_omitted += 1
                continue
            excerpt = source_map[symbol]
            allowance = min(
                budget.max_source_excerpt_bytes,
                budget.max_source_bytes - source_bytes,
            )
            if allowance <= 0:
                source_omitted += 1
                continue
            text, was_truncated = _truncate_utf8(excerpt.text, allowance)
            candidate = replace(
                excerpt,
                text=text,
                end_line=min(
                    excerpt.end_line,
                    excerpt.start_line + text.count("\n"),
                ),
            )
            excerpts.append(candidate)
            source_bytes += candidate.byte_count
            source_omitted += int(was_truncated)

        transcripts: list[ProofTranscriptExcerpt] = []
        transcript_bytes = 0
        transcript_omitted = 0
        for node in nodes:
            if str(node.get("node_kind")) != "proof":
                continue
            record = _node_record(node)
            raw = next(
                (
                    record.get(name)
                    for name in (
                        "proof_transcript",
                        "kernel_transcript",
                        "solver_transcript",
                        "transcript",
                    )
                    if record.get(name) not in (None, "")
                ),
                None,
            )
            if raw is None:
                continue
            if len(transcripts) >= budget.max_proof_transcripts:
                transcript_omitted += 1
                continue
            allowance = min(
                budget.max_proof_transcript_bytes,
                budget.max_proof_transcript_bytes_total - transcript_bytes,
            )
            if allowance <= 0:
                transcript_omitted += 1
                continue
            text, was_truncated = _truncate_utf8(str(raw), allowance)
            item = ProofTranscriptExcerpt(
                receipt_id=str(
                    record.get("receipt_id") or node.get("record_key") or ""
                ),
                obligation_id=str(
                    record.get("obligation_id") or node.get("obligation_id") or ""
                ),
                text=text,
            )
            transcripts.append(item)
            transcript_bytes += item.byte_count
            transcript_omitted += int(was_truncated)

        omitted = {
            "graph_rows": int(bool(neighborhood.get("truncated"))),
            "trusted_facts": 0,
            "untrusted_suggestions": 0,
            "unsupported_semantics": 0,
            "fallback_checks": 0,
            "source_excerpts": source_omitted,
            "proof_transcripts": transcript_omitted,
        }
        capsule = ProofContextCapsule(
            target=resolved_target,
            query=query,
            limits=budget,
            truncated=bool(neighborhood.get("truncated")),
            omitted=omitted,
        )
        if not self._fits(capsule, rows=int(neighborhood["row_count"])):
            raise ProofContextBudgetError(
                "context byte/token limits are too small for the mandatory envelope"
            )

        # Required fallbacks and unsupported semantics are packed before
        # descriptive graph facts because they preserve fail-closed behavior.
        def try_replace(field_name: str, values: Sequence[Any]) -> None:
            nonlocal capsule
            for value in values:
                current = getattr(capsule, field_name)
                candidate = replace(
                    capsule,
                    **{field_name: (*current, value)},
                )
                if self._fits(candidate, rows=int(neighborhood["row_count"])):
                    capsule = candidate
                else:
                    omitted[field_name] += 1
                    capsule = replace(capsule, truncated=True)

        try_replace("required_fallback_checks", fallback_checks)
        try_replace("unsupported_semantics", unsupported)
        try_replace("trusted_facts", trusted)
        try_replace("untrusted_suggestions", suggestions)
        try_replace("source_excerpts", excerpts)
        try_replace("proof_transcripts", transcripts)
        capsule = replace(capsule, omitted=omitted)
        retained_record_ids = {
            item.record_id
            for item in (
                *capsule.trusted_facts,
                *capsule.untrusted_suggestions,
            )
        }
        if omitted["fallback_checks"] or omitted["unsupported_semantics"]:
            raise ProofContextBudgetError(
                "context limits cannot contain all required fail-closed checks"
            )
        if not mandatory_record_ids.issubset(retained_record_ids):
            raise ProofContextBudgetError(
                "context limits cannot contain every exact selected record"
            )
        capsule = self._with_usage(capsule, rows=int(neighborhood["row_count"]))
        # Adding final omission counters can cross a boundary by a few bytes.
        # Remove lowest-authority/lowest-priority entries until it fits.
        for field_name in (
            "untrusted_suggestions",
            "proof_transcripts",
            "source_excerpts",
            "trusted_facts",
        ):
            while not self._fits(
                capsule, rows=int(neighborhood["row_count"])
            ) and getattr(capsule, field_name):
                current = getattr(capsule, field_name)
                capsule = replace(
                    capsule,
                    **{field_name: current[:-1]},
                    truncated=True,
                )
                omitted[field_name] += 1
                capsule = replace(capsule, omitted=omitted)
        capsule = self._with_usage(capsule, rows=int(neighborhood["row_count"]))
        final_record_ids = {
            item.record_id
            for item in (
                *capsule.trusted_facts,
                *capsule.untrusted_suggestions,
            )
        }
        if not mandatory_record_ids.issubset(final_record_ids):
            raise ProofContextBudgetError(
                "context limits cannot contain every exact selected record"
            )
        if (
            capsule.usage.bytes > budget.max_bytes
            or capsule.usage.tokens > budget.max_tokens
        ):
            raise ProofContextBudgetError(
                "context limits cannot contain mandatory fail-closed metadata"
            )
        return capsule


def build_proof_context_capsule(
    graph_path: Path | str,
    query: ProofContextQuery | None = None,
    **kwargs: Any,
) -> ProofContextCapsule:
    """Convenience entry point for one bounded capsule."""

    builder_keys = {"source_root", "templates", "token_counter"}
    builder_arguments = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in builder_keys
    }
    return ProofContextBuilder(graph_path, **builder_arguments).build(query, **kwargs)


generate_proof_context_capsule = build_proof_context_capsule
ContextCapsule = ProofContextCapsule


__all__ = [
    "ContextBudget",
    "ContextCapsule",
    "ContextEntry",
    "ContextTrust",
    "DEFAULT_MAX_CONTEXT_BYTES",
    "DEFAULT_MAX_CONTEXT_ROWS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_MAX_GRAPH_HOPS",
    "DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES",
    "DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES_TOTAL",
    "DEFAULT_MAX_PROOF_TRANSCRIPTS",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DEFAULT_MAX_SOURCE_EXCERPT_BYTES",
    "DEFAULT_MAX_SOURCE_EXCERPTS",
    "PROOF_CONTEXT_CAPSULE_SCHEMA",
    "PROOF_CONTEXT_LIMITS_SCHEMA",
    "PROOF_CONTEXT_QUERY_SCHEMA",
    "ProofContextBudget",
    "ProofContextBudgetError",
    "ProofContextBuilder",
    "ProofContextCapsule",
    "ProofContextError",
    "ProofContextLimits",
    "ProofContextQuery",
    "ProofContextTarget",
    "ProofContextUsage",
    "ProofTranscriptExcerpt",
    "SourceExcerpt",
    "build_proof_context_capsule",
    "estimate_context_tokens",
    "generate_proof_context_capsule",
]
