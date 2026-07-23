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
PROOF_PLANNING_CONTEXT_CAPSULE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.proof-planning-context-capsule@1"
)
PROOF_PLANNING_CONTEXT_LIMITS_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.proof-planning-context-limits@1"
)
LEANSTRAL_FIXED_THEOREM_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.leanstral-fixed-theorem@1"
)
LEANSTRAL_PROOF_CONTEXT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.leanstral-proof-context@1"
)
LEANSTRAL_PROOF_OUTPUT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.leanstral-proof-proposal@1"
)
LEANSTRAL_PROMPT_LIMITS_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.leanstral-prompt-limits@1"
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
DEFAULT_MAX_PLANNING_CANDIDATES = 8
DEFAULT_MAX_PLANNING_OBLIGATIONS = 32
DEFAULT_MAX_REJECTED_ALTERNATIVES = 8
DEFAULT_MAX_REJECTION_RATIONALE_BYTES = 1024
DEFAULT_MAX_PLANNING_DEPENDENCIES = 32
DEFAULT_MAX_PLANNING_RESOURCE_CLASSES = 16
DEFAULT_MAX_PLANNING_CONTEXT_BYTES = 16 * 1024
DEFAULT_MAX_PLANNING_CONTEXT_TOKENS = 4 * 1024
DEFAULT_MAX_LEANSTRAL_PREMISES = 64
DEFAULT_MAX_LEANSTRAL_TRUSTED_RECEIPTS = 16
DEFAULT_MAX_LEANSTRAL_FAILURES = 12
DEFAULT_MAX_LEANSTRAL_FAILURE_BYTES = 768
DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFTS = 4
DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFT_BYTES = 2 * 1024


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


def _ordered_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


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

    def to_planning_context(
        self,
        *,
        candidates: Sequence[Any] = (),
        evaluation: Any = None,
        available_resource_classes: Sequence[str] | str = (),
        proof_critical_path: Sequence[str] | str = (),
        limits: "ProofPlanningContextLimits | None" = None,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> "ProofPlanningContextCapsule":
        """Project this graph capsule into the smaller planner/router contract.

        The projection deliberately does not retain source excerpts, proof
        transcripts, arbitrary graph fields, or model suggestions.  It is
        therefore safe to pass to a planning router without making the
        router's prompt grow with the repository evidence graph.
        """

        return build_proof_planning_context_capsule(
            self,
            candidates=candidates,
            evaluation=evaluation,
            available_resource_classes=available_resource_classes,
            proof_critical_path=proof_critical_path,
            limits=limits,
            token_counter=token_counter,
        )

    def for_router(self, **kwargs: Any) -> "ProofPlanningContextCapsule":
        """Compatibility spelling for :meth:`to_planning_context`."""

        return self.to_planning_context(**kwargs)

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


@dataclass(frozen=True)
class FixedTheoremIdentity:
    """Supervisor-owned Lean theorem fields that a model cannot rewrite.

    ``equivalence_key`` intentionally omits task and obligation identifiers.
    Two tasks may therefore reuse a model draft when (and only when) their
    theorem statement, template, source scope, and allowed premise identities
    are identical.  The reusable draft remains untrusted model output.
    """

    theorem_id: str
    obligation_id: str
    conclusion: str
    template_id: str
    source_scope: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    allowed_premise_ids: tuple[str, ...] = ()
    declaration_name: str = ""
    template_version: str = ""
    template_semantic_hash: str = ""
    repository_tree_id: str = ""
    canonical_source_digest: str = ""
    schema: str = LEANSTRAL_FIXED_THEOREM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEANSTRAL_FIXED_THEOREM_SCHEMA:
            raise ProofContextError("unsupported fixed theorem schema")
        for name in ("theorem_id", "obligation_id", "conclusion", "template_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProofContextError(f"fixed theorem {name} is required")
            object.__setattr__(self, name, value)
        scope = _ordered_strings(self.source_scope)
        if not scope:
            raise ProofContextError("fixed theorem source_scope is required")
        object.__setattr__(self, "source_scope", scope)
        object.__setattr__(self, "assumptions", _ordered_strings(self.assumptions))
        object.__setattr__(
            self,
            "allowed_premise_ids",
            _ordered_strings(self.allowed_premise_ids),
        )
        for name in (
            "declaration_name",
            "template_version",
            "template_semantic_hash",
            "repository_tree_id",
            "canonical_source_digest",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())

    @property
    def equivalence_key(self) -> str:
        semantic_fields = {
            "assumptions": list(self.assumptions),
            "conclusion": self.conclusion,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_semantic_hash": self.template_semantic_hash,
            "source_scope": list(self.source_scope),
            "allowed_premise_ids": list(self.allowed_premise_ids),
            "canonical_source_digest": self.canonical_source_digest,
        }
        return (
            "lean-theorem:sha256:"
            + hashlib.sha256(
                _canonical_json(semantic_fields).encode("utf-8")
            ).hexdigest()
        )

    @property
    def identity_digest(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(
                _canonical_json(self.to_dict(include_equivalence=False)).encode("utf-8")
            ).hexdigest()
        )

    def to_dict(self, *, include_equivalence: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "theorem_id": self.theorem_id,
            "obligation_id": self.obligation_id,
            "declaration_name": self.declaration_name,
            "assumptions": list(self.assumptions),
            "conclusion": self.conclusion,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_semantic_hash": self.template_semantic_hash,
            "source_scope": list(self.source_scope),
            "allowed_premise_ids": list(self.allowed_premise_ids),
            "repository_tree_id": self.repository_tree_id,
            "canonical_source_digest": self.canonical_source_digest,
        }
        if include_equivalence:
            result["identity_digest"] = self.identity_digest
            result["equivalence_key"] = self.equivalence_key
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixedTheoremIdentity":
        if not isinstance(payload, Mapping):
            raise ProofContextError("fixed theorem must be a mapping")
        template = payload.get("template")
        template_fields = template if isinstance(template, Mapping) else {}
        raw_scope = payload.get("source_scope")
        if isinstance(raw_scope, Mapping):
            raw_scope = (
                raw_scope.get("scope_ids")
                or raw_scope.get("symbols")
                or raw_scope.get("paths")
                or ()
            )
        result = cls(
            schema=str(payload.get("schema") or LEANSTRAL_FIXED_THEOREM_SCHEMA),
            theorem_id=str(
                payload.get("theorem_id")
                or payload.get("identity")
                or payload.get("obligation_id")
                or ""
            ),
            obligation_id=str(payload.get("obligation_id") or ""),
            declaration_name=str(
                payload.get("declaration_name") or payload.get("name") or ""
            ),
            assumptions=_ordered_strings(payload.get("assumptions")),
            conclusion=str(
                payload.get("conclusion")
                or payload.get("canonical_statement")
                or payload.get("statement")
                or ""
            ),
            template_id=str(
                payload.get("template_id")
                or template_fields.get("template_id")
                or (template if isinstance(template, str) else "")
                or ""
            ),
            template_version=str(
                payload.get("template_version")
                or template_fields.get("version")
                or template_fields.get("template_version")
                or ""
            ),
            template_semantic_hash=str(
                payload.get("template_semantic_hash")
                or payload.get("semantic_hash")
                or template_fields.get("semantic_hash")
                or template_fields.get("template_semantic_hash")
                or ""
            ),
            source_scope=_ordered_strings(
                raw_scope
                or payload.get("source_scope_ids")
                or payload.get("ast_scope_ids")
                or ()
            ),
            allowed_premise_ids=_ordered_strings(
                payload.get("allowed_premise_ids") or payload.get("premise_ids") or ()
            ),
            repository_tree_id=str(
                payload.get("repository_tree_id") or payload.get("tree_id") or ""
            ),
            canonical_source_digest=str(
                payload.get("canonical_source_digest")
                or payload.get("source_digest")
                or ""
            ),
        )
        claimed_identity = str(payload.get("identity_digest") or "")
        if claimed_identity and claimed_identity != result.identity_digest:
            raise ProofContextError("fixed theorem identity digest does not match")
        claimed_equivalence = str(payload.get("equivalence_key") or "")
        if claimed_equivalence and claimed_equivalence != result.equivalence_key:
            raise ProofContextError("fixed theorem equivalence key does not match")
        return result

    @classmethod
    def from_capsule(
        cls,
        capsule: "ProofContextCapsule",
        *,
        obligation_id: str = "",
        **overrides: Any,
    ) -> "FixedTheoremIdentity":
        """Derive the immutable theorem fields from one exact obligation."""

        if not isinstance(capsule, ProofContextCapsule):
            raise ProofContextError("capsule must be a ProofContextCapsule")
        selected = str(
            obligation_id
            or (
                capsule.query.obligation_ids[0]
                if len(capsule.query.obligation_ids) == 1
                else ""
            )
        ).strip()
        if not selected:
            raise ProofContextError("one exact obligation_id is required")
        records = [
            entry.fields
            for entry in capsule.trusted_facts
            if entry.kind == "obligation"
            and str(entry.fields.get("obligation_id") or entry.record_id) == selected
        ]
        if len(records) != 1:
            raise ProofContextError("fixed theorem obligation is absent or ambiguous")
        values = dict(records[0])
        values.update(overrides)
        values.setdefault("theorem_id", selected)
        values.setdefault("obligation_id", selected)
        values.setdefault("source_scope", capsule.query.symbols)
        return cls.from_dict(values)


FixedTheorem = FixedTheoremIdentity


@dataclass(frozen=True)
class LeanstralAllowedPremise:
    premise_id: str
    statement: str
    source_scope: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("premise_id", "statement"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProofContextError(f"allowed premise {name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_scope", _ordered_strings(self.source_scope))

    def to_dict(self) -> dict[str, Any]:
        return {
            "premise_id": self.premise_id,
            "statement": self.statement,
            "source_scope": list(self.source_scope),
        }

    @classmethod
    def from_value(cls, value: Any) -> "LeanstralAllowedPremise":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProofContextError("allowed premise must be a mapping")
        return cls(
            premise_id=str(
                value.get("premise_id")
                or value.get("obligation_id")
                or value.get("id")
                or ""
            ),
            statement=str(
                value.get("statement")
                or value.get("canonical_statement")
                or value.get("declaration")
                or ""
            ),
            source_scope=_ordered_strings(
                value.get("source_scope")
                or value.get("source_scope_ids")
                or value.get("ast_scope_ids")
                or ()
            ),
        )


AllowedPremise = LeanstralAllowedPremise


@dataclass(frozen=True)
class LeanstralTrustedReceipt:
    receipt_id: str
    obligation_id: str
    assurance: str
    verdict: str
    repository_tree_id: str = ""

    def __post_init__(self) -> None:
        for name in ("receipt_id", "obligation_id", "assurance", "verdict"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProofContextError(f"trusted receipt {name} is required")
            object.__setattr__(self, name, value)
        if self.assurance.casefold() in {"", "unverified", "candidate"}:
            raise ProofContextError("trusted receipt must contain checked evidence")
        if self.verdict.casefold() not in {"proved", "accepted", "valid", "passed"}:
            raise ProofContextError("trusted receipt must have an accepted verdict")
        object.__setattr__(
            self, "repository_tree_id", str(self.repository_tree_id or "").strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "obligation_id": self.obligation_id,
            "assurance": self.assurance,
            "verdict": self.verdict,
            "repository_tree_id": self.repository_tree_id,
            "trust": ContextTrust.TRUSTED_FACT.value,
            "checked_evidence": True,
        }

    @classmethod
    def from_value(cls, value: Any) -> "LeanstralTrustedReceipt":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProofContextError("trusted receipt must be a mapping")
        return cls(
            receipt_id=str(
                value.get("receipt_id")
                or value.get("proof_id")
                or value.get("record_id")
                or ""
            ),
            obligation_id=str(value.get("obligation_id") or ""),
            assurance=str(
                value.get("authoritative_assurance") or value.get("assurance") or ""
            ),
            verdict=str(
                value.get("authoritative_verdict")
                or value.get("verdict")
                or value.get("status")
                or ""
            ),
            repository_tree_id=str(
                value.get("repository_tree_id") or value.get("tree_id") or ""
            ),
        )


TrustedPriorReceipt = LeanstralTrustedReceipt


@dataclass(frozen=True)
class LeanstralCompactFailure:
    failure_id: str
    summary: str
    failure_code: str = ""
    obligation_id: str = ""

    def __post_init__(self) -> None:
        for name in ("failure_id", "summary"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProofContextError(f"compact failure {name} is required")
            object.__setattr__(self, name, value)
        for name in ("failure_code", "obligation_id"):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())

    def compact(self, maximum_bytes: int) -> "LeanstralCompactFailure":
        text, _ = _truncate_utf8(self.summary, maximum_bytes)
        return replace(self, summary=text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "failure_code": self.failure_code,
            "obligation_id": self.obligation_id,
            "summary": self.summary,
        }

    @classmethod
    def from_value(cls, value: Any) -> "LeanstralCompactFailure":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProofContextError("compact failure must be a mapping")
        return cls(
            failure_id=str(
                value.get("failure_id")
                or value.get("contradiction_id")
                or value.get("receipt_id")
                or value.get("record_id")
                or ""
            ),
            failure_code=str(value.get("failure_code") or value.get("status") or ""),
            obligation_id=str(value.get("obligation_id") or ""),
            summary=str(
                value.get("summary")
                or value.get("failure_reason")
                or value.get("contradiction")
                or value.get("diagnostics")
                or value.get("counterexample")
                or ""
            ),
        )


CompactProofFailure = LeanstralCompactFailure


@dataclass(frozen=True)
class LeanstralReusableDraft:
    artifact_id: str
    equivalence_key: str
    proposal_kind: str
    proof_text: str = ""
    decomposition: tuple[Mapping[str, Any], ...] = ()
    source_theorem_id: str = ""

    def __post_init__(self) -> None:
        for name in ("artifact_id", "equivalence_key"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ProofContextError(f"reusable draft {name} is required")
            object.__setattr__(self, name, value)
        kind = str(self.proposal_kind or "").strip().casefold()
        if kind not in {"proof", "decomposition"}:
            raise ProofContextError(
                "reusable draft proposal_kind must be proof or decomposition"
            )
        proof_text = str(self.proof_text or "").strip()
        decomposition = tuple(
            _public_mapping(item)
            for item in self.decomposition
            if isinstance(item, Mapping)
        )
        if kind == "proof" and not proof_text:
            raise ProofContextError("reusable proof draft needs proof_text")
        if kind == "decomposition" and not decomposition:
            raise ProofContextError("reusable decomposition draft needs subgoals")
        object.__setattr__(self, "proposal_kind", kind)
        object.__setattr__(self, "proof_text", proof_text)
        object.__setattr__(self, "decomposition", decomposition)
        object.__setattr__(
            self, "source_theorem_id", str(self.source_theorem_id or "").strip()
        )

    def compact(self, maximum_bytes: int) -> "LeanstralReusableDraft":
        if self.proposal_kind == "proof":
            proof, _ = _truncate_utf8(self.proof_text, maximum_bytes)
            return replace(self, proof_text=proof)
        packed: list[Mapping[str, Any]] = []
        used = 0
        for item in self.decomposition:
            encoded = len(_canonical_json(item).encode("utf-8"))
            if used + encoded > maximum_bytes:
                break
            packed.append(item)
            used += encoded
        if not packed:
            raise ProofContextBudgetError(
                "reusable draft byte limit cannot contain one decomposition item"
            )
        return replace(self, decomposition=tuple(packed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "equivalence_key": self.equivalence_key,
            "source_theorem_id": self.source_theorem_id,
            "proposal_kind": self.proposal_kind,
            "proof_text": self.proof_text,
            "decomposition": [dict(item) for item in self.decomposition],
            "trust": ContextTrust.UNTRUSTED_SUGGESTION.value,
            "checked_evidence": False,
            "reusable_as_evidence": False,
        }

    @classmethod
    def from_value(cls, value: Any) -> "LeanstralReusableDraft":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProofContextError("reusable draft must be a mapping")
        if bool(value.get("verified")) or bool(value.get("kernel_checked")):
            raise ProofContextError(
                "checked proof results belong in trusted receipts, not reusable drafts"
            )
        assurance = str(value.get("assurance") or "unverified").casefold()
        if assurance not in {"", "unverified", "candidate"}:
            raise ProofContextError(
                "checked proof results belong in trusted receipts, not reusable drafts"
            )
        decomposition = value.get("decomposition") or ()
        if isinstance(decomposition, Mapping):
            decomposition = (decomposition,)
        kind = str(value.get("proposal_kind") or "").casefold()
        if not kind:
            kind = "decomposition" if decomposition else "proof"
        return cls(
            artifact_id=str(value.get("artifact_id") or value.get("record_id") or ""),
            equivalence_key=str(
                value.get("theorem_equivalence_key")
                or value.get("equivalence_key")
                or ""
            ),
            source_theorem_id=str(
                value.get("theorem_id") or value.get("source_theorem_id") or ""
            ),
            proposal_kind=kind,
            proof_text=str(value.get("proof_text") or value.get("draft_text") or ""),
            decomposition=tuple(decomposition),
        )


ReusableProofDraft = LeanstralReusableDraft


@dataclass(frozen=True)
class LeanstralPromptLimits:
    max_bytes: int = DEFAULT_MAX_CONTEXT_BYTES
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_premises: int = DEFAULT_MAX_LEANSTRAL_PREMISES
    max_trusted_receipts: int = DEFAULT_MAX_LEANSTRAL_TRUSTED_RECEIPTS
    max_failures: int = DEFAULT_MAX_LEANSTRAL_FAILURES
    max_failure_bytes: int = DEFAULT_MAX_LEANSTRAL_FAILURE_BYTES
    max_reusable_drafts: int = DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFTS
    max_reusable_draft_bytes: int = DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFT_BYTES

    def __post_init__(self) -> None:
        for name in (
            "max_bytes",
            "max_tokens",
            "max_premises",
            "max_failure_bytes",
            "max_reusable_draft_bytes",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ProofContextError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("max_trusted_receipts", "max_failures", "max_reusable_drafts"):
            value = int(getattr(self, name))
            if value < 0:
                raise ProofContextError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEANSTRAL_PROMPT_LIMITS_SCHEMA,
            "max_bytes": self.max_bytes,
            "max_tokens": self.max_tokens,
            "max_premises": self.max_premises,
            "max_trusted_receipts": self.max_trusted_receipts,
            "max_failures": self.max_failures,
            "max_failure_bytes": self.max_failure_bytes,
            "max_reusable_drafts": self.max_reusable_drafts,
            "max_reusable_draft_bytes": self.max_reusable_draft_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeanstralPromptLimits":
        schema = str(value.get("schema") or LEANSTRAL_PROMPT_LIMITS_SCHEMA)
        if schema != LEANSTRAL_PROMPT_LIMITS_SCHEMA:
            raise ProofContextError("unsupported Leanstral prompt limits schema")
        defaults = cls()
        return cls(
            **{
                name: int(value.get(name, getattr(defaults, name)))
                for name in (
                    "max_bytes",
                    "max_tokens",
                    "max_premises",
                    "max_trusted_receipts",
                    "max_failures",
                    "max_failure_bytes",
                    "max_reusable_drafts",
                    "max_reusable_draft_bytes",
                )
            }
        )


_LEANSTRAL_OUTPUT_SCHEMA: Mapping[str, Any] = {
    "schema": LEANSTRAL_PROOF_OUTPUT_SCHEMA,
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "theorem_id", "proposal_kind"],
    "properties": {
        "schema": {"const": LEANSTRAL_PROOF_OUTPUT_SCHEMA},
        "theorem_id": {"type": "string", "description": "copy exactly"},
        "proposal_kind": {"enum": ["proof", "decomposition"]},
        "proof_text": {"type": "string"},
        "decomposition": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subgoal_id", "statement"],
                "properties": {
                    "subgoal_id": {"type": "string"},
                    "statement": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    "oneOf": [
        {"required": ["proof_text"]},
        {"required": ["decomposition"]},
    ],
}


@dataclass(frozen=True)
class LeanstralProofContext:
    """A compact fixed-theorem prompt projected from a bounded capsule."""

    capsule_id: str
    theorem: FixedTheoremIdentity
    allowed_premises: tuple[LeanstralAllowedPremise, ...]
    trusted_prior_receipts: tuple[LeanstralTrustedReceipt, ...] = ()
    compact_failures: tuple[LeanstralCompactFailure, ...] = ()
    reusable_untrusted_drafts: tuple[LeanstralReusableDraft, ...] = ()
    limits: LeanstralPromptLimits = field(default_factory=LeanstralPromptLimits)
    omitted: Mapping[str, int] = field(default_factory=dict)
    token_counter: Callable[[str], int] = field(
        default=estimate_context_tokens,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        capsule_id = str(self.capsule_id or "").strip()
        if not capsule_id:
            raise ProofContextError("Leanstral context capsule_id is required")
        object.__setattr__(self, "capsule_id", capsule_id)
        if not isinstance(self.theorem, FixedTheoremIdentity):
            raise ProofContextError("Leanstral context theorem must be fixed")
        if not isinstance(self.limits, LeanstralPromptLimits):
            raise ProofContextError("Leanstral context limits are invalid")
        for name, kind in (
            ("allowed_premises", LeanstralAllowedPremise),
            ("trusted_prior_receipts", LeanstralTrustedReceipt),
            ("compact_failures", LeanstralCompactFailure),
            ("reusable_untrusted_drafts", LeanstralReusableDraft),
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(item, kind) for item in values):
                raise ProofContextError(f"Leanstral context {name} is invalid")
            object.__setattr__(self, name, values)
        premise_ids = tuple(item.premise_id for item in self.allowed_premises)
        if len(premise_ids) != len(set(premise_ids)):
            raise ProofContextError("allowed premise identities must be unique")
        if premise_ids != self.theorem.allowed_premise_ids:
            raise ProofContextError(
                "allowed premises must exactly match the fixed theorem premise identities"
            )
        if len(premise_ids) > self.limits.max_premises:
            raise ProofContextBudgetError("fixed theorem exceeds the premise budget")
        if any(
            item.equivalence_key != self.theorem.equivalence_key
            for item in self.reusable_untrusted_drafts
        ):
            raise ProofContextError("reusable draft is not theorem-equivalent")
        object.__setattr__(
            self,
            "omitted",
            {
                str(key): int(value)
                for key, value in self.omitted.items()
                if int(value) >= 0
            },
        )
        if not callable(self.token_counter):
            raise ProofContextError("Leanstral token_counter must be callable")
        if self.prompt_bytes > self.limits.max_bytes:
            raise ProofContextBudgetError("Leanstral prompt exceeds its byte budget")
        if self.prompt_tokens > self.limits.max_tokens:
            raise ProofContextBudgetError("Leanstral prompt exceeds its token budget")

    @property
    def context_id(self) -> str:
        payload = self.to_dict()
        return (
            "leanstral-context:sha256:"
            + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEANSTRAL_PROOF_CONTEXT_SCHEMA,
            "instruction": (
                "Prove the fixed Lean theorem or decompose it into proof subgoals. "
                "Use only allowed_premises and trusted_prior_receipts as checked "
                "evidence. Compact failures and reusable drafts are diagnostic "
                "hints, not checked evidence. Return only JSON matching output_schema."
            ),
            "capsule_id": self.capsule_id,
            "fixed_theorem": self.theorem.to_dict(),
            "allowed_premises": [item.to_dict() for item in self.allowed_premises],
            "trusted_prior_receipts": [
                item.to_dict() for item in self.trusted_prior_receipts
            ],
            "compact_failures": [item.to_dict() for item in self.compact_failures],
            "reusable_untrusted_drafts": [
                item.to_dict() for item in self.reusable_untrusted_drafts
            ],
            "immutable_constraints": {
                "may_propose": ["proof_text", "decomposition"],
                "must_not_change": [
                    "theorem_id",
                    "obligation_id",
                    "assumptions",
                    "conclusion",
                    "template_id",
                    "template_version",
                    "template_semantic_hash",
                    "source_scope",
                    "allowed_premise_ids",
                    "canonical_source_digest",
                ],
                "no_new_premises": True,
                "no_source_scope_expansion": True,
                "response_must_match_output_schema": True,
            },
            "output_schema": json.loads(_canonical_json(_LEANSTRAL_OUTPUT_SCHEMA)),
            "limits": self.limits.to_dict(),
            "truncated": any(self.omitted.values()),
            "omitted": dict(sorted(self.omitted.items())),
        }

    def to_prompt(self) -> str:
        return _canonical_json(self.to_dict())

    render_for_leanstral = to_prompt

    @property
    def prompt_bytes(self) -> int:
        return len(self.to_prompt().encode("utf-8"))

    @property
    def prompt_tokens(self) -> int:
        return int(self.token_counter(self.to_prompt()))


FixedTheoremProofContext = LeanstralProofContext
LeanstralProofContextCapsule = LeanstralProofContext


def _derive_leanstral_premises(
    capsule: ProofContextCapsule,
    theorem: FixedTheoremIdentity,
) -> tuple[LeanstralAllowedPremise, ...]:
    records: dict[str, LeanstralAllowedPremise] = {}
    for entry in capsule.trusted_facts:
        if entry.kind != "obligation":
            continue
        premise_id = str(
            entry.fields.get("obligation_id") or entry.record_id or ""
        ).strip()
        if premise_id not in theorem.allowed_premise_ids:
            continue
        try:
            records[premise_id] = LeanstralAllowedPremise.from_value(entry.fields)
        except ProofContextError:
            continue
    missing = sorted(set(theorem.allowed_premise_ids) - set(records))
    if missing:
        raise ProofContextError(
            "allowed premise statements are absent from the capsule: "
            + ", ".join(missing)
        )
    return tuple(records[premise_id] for premise_id in theorem.allowed_premise_ids)


def _derive_leanstral_receipts(
    capsule: ProofContextCapsule,
) -> tuple[LeanstralTrustedReceipt, ...]:
    result: list[LeanstralTrustedReceipt] = []
    for entry in capsule.trusted_facts:
        if entry.kind not in {"proof", "proof_receipt", "formal_proof_receipt"}:
            continue
        value = dict(entry.fields)
        value.setdefault("record_id", entry.record_id)
        try:
            result.append(LeanstralTrustedReceipt.from_value(value))
        except ProofContextError:
            # Authoritative graph membership is not enough: only an explicitly
            # accepted, checked receipt is rendered as trusted proof evidence.
            continue
    return tuple(sorted(result, key=lambda item: item.receipt_id))


def _derive_leanstral_failures(
    capsule: ProofContextCapsule,
) -> tuple[LeanstralCompactFailure, ...]:
    result: list[LeanstralCompactFailure] = []
    for entry in (*capsule.trusted_facts, *capsule.untrusted_suggestions):
        fields = entry.fields
        status = str(fields.get("status") or fields.get("verdict") or "").casefold()
        has_failure = status in {
            "failed",
            "failure",
            "rejected",
            "invalid",
            "timed_out",
            "inconclusive",
        } or any(
            fields.get(name)
            for name in (
                "failure_reason",
                "contradiction",
                "diagnostics",
                "counterexample",
            )
        )
        if not has_failure:
            continue
        value = dict(fields)
        value.setdefault("record_id", entry.record_id)
        try:
            result.append(LeanstralCompactFailure.from_value(value))
        except ProofContextError:
            continue
    return tuple(sorted(result, key=lambda item: item.failure_id))


def _derive_leanstral_drafts(
    capsule: ProofContextCapsule,
) -> tuple[LeanstralReusableDraft, ...]:
    result: list[LeanstralReusableDraft] = []
    for entry in capsule.untrusted_suggestions:
        fields = dict(entry.fields)
        if not (
            entry.kind in {"llm_output", "model_draft", "leanstral_draft"}
            or fields.get("artifact_kind") == "llm_output"
            or fields.get("stage") == "model_draft"
        ):
            continue
        fields.setdefault("record_id", entry.record_id)
        try:
            result.append(LeanstralReusableDraft.from_value(fields))
        except ProofContextError:
            continue
    return tuple(sorted(result, key=lambda item: item.artifact_id))


def build_leanstral_proof_context(
    capsule: ProofContextCapsule,
    theorem: FixedTheoremIdentity | Mapping[str, Any],
    *,
    allowed_premises: Sequence[LeanstralAllowedPremise | Mapping[str, Any]] = (),
    trusted_prior_receipts: Sequence[LeanstralTrustedReceipt | Mapping[str, Any]] = (),
    compact_failures: Sequence[LeanstralCompactFailure | Mapping[str, Any]] = (),
    reusable_drafts: Sequence[LeanstralReusableDraft | Mapping[str, Any]] = (),
    limits: LeanstralPromptLimits | Mapping[str, Any] | None = None,
    token_counter: Callable[[str], int] = estimate_context_tokens,
) -> LeanstralProofContext:
    """Project a bounded capsule into one immutable Leanstral solve prompt."""

    if not isinstance(capsule, ProofContextCapsule):
        raise ProofContextError("capsule must be a ProofContextCapsule")
    if capsule.target is not ProofContextTarget.LEANSTRAL:
        raise ProofContextError("Leanstral prompts require a Leanstral context capsule")
    fixed = (
        theorem
        if isinstance(theorem, FixedTheoremIdentity)
        else FixedTheoremIdentity.from_dict(theorem)
    )
    if (
        capsule.query.obligation_ids
        and fixed.obligation_id not in capsule.query.obligation_ids
    ):
        raise ProofContextError("fixed theorem obligation is outside the capsule")
    if capsule.query.symbols and not set(fixed.source_scope).issubset(
        capsule.query.symbols
    ):
        raise ProofContextError("fixed theorem source scope is outside the capsule")
    obligation_records = [
        entry.fields
        for entry in capsule.trusted_facts
        if entry.kind == "obligation"
        and str(entry.fields.get("obligation_id") or entry.record_id)
        == fixed.obligation_id
    ]
    if len(obligation_records) != 1:
        raise ProofContextError(
            "fixed theorem must bind one trusted capsule obligation"
        )
    canonical_obligation = obligation_records[0]
    comparisons = (
        ("conclusion", fixed.conclusion, canonical_obligation.get("conclusion")),
        ("template_id", fixed.template_id, canonical_obligation.get("template_id")),
        (
            "template_version",
            fixed.template_version,
            canonical_obligation.get("template_version"),
        ),
        (
            "template_semantic_hash",
            fixed.template_semantic_hash,
            canonical_obligation.get("template_semantic_hash"),
        ),
        (
            "canonical_source_digest",
            fixed.canonical_source_digest,
            canonical_obligation.get("canonical_source_digest"),
        ),
    )
    for name, supplied, canonical in comparisons:
        if canonical not in (None, "") and str(canonical).strip() != supplied:
            raise ProofContextError(
                f"fixed theorem {name} does not match the capsule obligation"
            )
    canonical_assumptions = canonical_obligation.get("assumptions")
    if (
        canonical_assumptions not in (None, ())
        and _ordered_strings(canonical_assumptions) != fixed.assumptions
    ):
        raise ProofContextError(
            "fixed theorem assumptions do not match the capsule obligation"
        )
    canonical_premises = canonical_obligation.get(
        "allowed_premise_ids", canonical_obligation.get("premise_ids")
    )
    if (
        canonical_premises not in (None, ())
        and _ordered_strings(canonical_premises) != fixed.allowed_premise_ids
    ):
        raise ProofContextError(
            "fixed theorem premises do not match the capsule obligation"
        )
    prompt_limits = (
        limits
        if isinstance(limits, LeanstralPromptLimits)
        else (
            LeanstralPromptLimits.from_dict(limits)
            if isinstance(limits, Mapping)
            else LeanstralPromptLimits(
                max_bytes=capsule.limits.max_bytes,
                max_tokens=capsule.limits.max_tokens,
            )
        )
    )
    canonical_premises_by_id = {
        item.premise_id: item for item in _derive_leanstral_premises(capsule, fixed)
    }
    premises = (
        tuple(LeanstralAllowedPremise.from_value(item) for item in allowed_premises)
        if allowed_premises
        else tuple(
            canonical_premises_by_id[premise_id]
            for premise_id in fixed.allowed_premise_ids
        )
    )
    if len(premises) > prompt_limits.max_premises:
        raise ProofContextBudgetError("fixed theorem exceeds the premise budget")
    if tuple(item.premise_id for item in premises) != fixed.allowed_premise_ids:
        raise ProofContextError(
            "allowed premises must exactly match the fixed theorem premise identities"
        )
    for premise in premises:
        canonical = canonical_premises_by_id[premise.premise_id]
        if premise != canonical:
            raise ProofContextError(
                "allowed premise content does not match the trusted capsule record"
            )

    receipts = (
        tuple(
            LeanstralTrustedReceipt.from_value(item) for item in trusted_prior_receipts
        )
        if trusted_prior_receipts
        else _derive_leanstral_receipts(capsule)
    )
    allowed_receipt_obligations = {
        fixed.obligation_id,
        *fixed.allowed_premise_ids,
    }
    for receipt in receipts:
        if receipt.obligation_id not in allowed_receipt_obligations:
            raise ProofContextError(
                "trusted receipt obligation is outside the fixed theorem"
            )
        if (
            fixed.repository_tree_id
            and receipt.repository_tree_id
            and receipt.repository_tree_id != fixed.repository_tree_id
        ):
            raise ProofContextError(
                "trusted receipt repository tree does not match the fixed theorem"
            )
    failures = (
        tuple(LeanstralCompactFailure.from_value(item) for item in compact_failures)
        if compact_failures
        else _derive_leanstral_failures(capsule)
    )
    failures = tuple(
        item.compact(prompt_limits.max_failure_bytes)
        for item in failures
        if not item.obligation_id or item.obligation_id in allowed_receipt_obligations
    )
    drafts = [LeanstralReusableDraft.from_value(item) for item in reusable_drafts]
    drafts.extend(_derive_leanstral_drafts(capsule))
    unique_drafts: dict[str, LeanstralReusableDraft] = {}
    for item in drafts:
        if item.equivalence_key == fixed.equivalence_key:
            unique_drafts[item.artifact_id] = item.compact(
                prompt_limits.max_reusable_draft_bytes
            )

    # Begin with every optional record omitted.  Each successful atomic pack
    # adds the record and decrements its omission count together, so a prompt
    # sitting exactly on a byte boundary can never be invalidated by a later
    # bookkeeping update.
    omitted = {
        "trusted_prior_receipts": len(receipts),
        "compact_failures": len(failures),
        "reusable_untrusted_drafts": len(unique_drafts),
    }
    receipts = receipts[: prompt_limits.max_trusted_receipts]
    failures = failures[: prompt_limits.max_failures]
    drafts_tuple = tuple(unique_drafts[key] for key in sorted(unique_drafts))[
        : prompt_limits.max_reusable_drafts
    ]
    result = LeanstralProofContext(
        capsule_id=capsule.capsule_id,
        theorem=fixed,
        allowed_premises=premises,
        limits=prompt_limits,
        omitted=omitted,
        token_counter=token_counter,
    )
    # Optional context is packed in trust/usefulness order and never allowed to
    # evict the theorem or its complete premise allowlist.
    for name, values in (
        ("trusted_prior_receipts", receipts),
        ("compact_failures", failures),
        ("reusable_untrusted_drafts", drafts_tuple),
    ):
        for value in values:
            candidate_values = (*getattr(result, name), value)
            candidate_omitted = {
                **result.omitted,
                name: max(0, result.omitted.get(name, 0) - 1),
            }
            try:
                result = replace(
                    result,
                    **{name: candidate_values},
                    omitted=candidate_omitted,
                )
            except ProofContextBudgetError:
                continue
    return result


generate_leanstral_proof_context = build_leanstral_proof_context
build_fixed_theorem_leanstral_context = build_leanstral_proof_context


def generate_fixed_theorem_leanstral_prompt(
    capsule: ProofContextCapsule,
    theorem: FixedTheoremIdentity | Mapping[str, Any],
    **kwargs: Any,
) -> str:
    """Build and render one bounded fixed-theorem prompt."""

    return build_leanstral_proof_context(capsule, theorem, **kwargs).to_prompt()


@dataclass(frozen=True)
class ProofPlanningContextLimits:
    """Hard limits for the proof projection supplied to a plan router."""

    max_candidates: int = DEFAULT_MAX_PLANNING_CANDIDATES
    max_obligations: int = DEFAULT_MAX_PLANNING_OBLIGATIONS
    max_rejected_alternatives: int = DEFAULT_MAX_REJECTED_ALTERNATIVES
    max_rationale_bytes: int = DEFAULT_MAX_REJECTION_RATIONALE_BYTES
    max_dependencies: int = DEFAULT_MAX_PLANNING_DEPENDENCIES
    max_resource_classes: int = DEFAULT_MAX_PLANNING_RESOURCE_CLASSES
    max_bytes: int = DEFAULT_MAX_PLANNING_CONTEXT_BYTES
    max_tokens: int = DEFAULT_MAX_PLANNING_CONTEXT_TOKENS

    def __post_init__(self) -> None:
        for name in (
            "max_candidates",
            "max_obligations",
            "max_rejected_alternatives",
            "max_dependencies",
            "max_resource_classes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ProofContextError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("max_rationale_bytes", "max_bytes", "max_tokens"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ProofContextError(f"{name} must be positive")
            if name == "max_rationale_bytes" and value < 5:
                raise ProofContextError(
                    "max_rationale_bytes must fit a JSON rationale envelope"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_PLANNING_CONTEXT_LIMITS_SCHEMA,
            "max_candidates": self.max_candidates,
            "max_obligations": self.max_obligations,
            "max_rejected_alternatives": self.max_rejected_alternatives,
            "max_rationale_bytes": self.max_rationale_bytes,
            "max_dependencies": self.max_dependencies,
            "max_resource_classes": self.max_resource_classes,
            "max_bytes": self.max_bytes,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofPlanningContextLimits":
        schema = str(payload.get("schema") or PROOF_PLANNING_CONTEXT_LIMITS_SCHEMA)
        if schema != PROOF_PLANNING_CONTEXT_LIMITS_SCHEMA:
            raise ProofContextError(
                f"unsupported proof planning context limits schema: {schema}"
            )
        return cls(
            max_candidates=int(
                payload.get("max_candidates", DEFAULT_MAX_PLANNING_CANDIDATES)
            ),
            max_obligations=int(
                payload.get("max_obligations", DEFAULT_MAX_PLANNING_OBLIGATIONS)
            ),
            max_rejected_alternatives=int(
                payload.get(
                    "max_rejected_alternatives",
                    DEFAULT_MAX_REJECTED_ALTERNATIVES,
                )
            ),
            max_rationale_bytes=int(
                payload.get(
                    "max_rationale_bytes",
                    DEFAULT_MAX_REJECTION_RATIONALE_BYTES,
                )
            ),
            max_dependencies=int(
                payload.get(
                    "max_dependencies",
                    DEFAULT_MAX_PLANNING_DEPENDENCIES,
                )
            ),
            max_resource_classes=int(
                payload.get(
                    "max_resource_classes",
                    DEFAULT_MAX_PLANNING_RESOURCE_CLASSES,
                )
            ),
            max_bytes=int(payload.get("max_bytes", DEFAULT_MAX_PLANNING_CONTEXT_BYTES)),
            max_tokens=int(
                payload.get("max_tokens", DEFAULT_MAX_PLANNING_CONTEXT_TOKENS)
            ),
        )


def _finite_non_negative_number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProofContextError(
            f"{label} must be a finite non-negative number"
        ) from exc
    if result < 0 or result == float("inf") or result != result:
        raise ProofContextError(f"{label} must be a finite non-negative number")
    return result


def _unit_interval_number(value: Any, *, label: str) -> float:
    result = _finite_non_negative_number(value, label=label)
    if result > 1:
        raise ProofContextError(f"{label} must be between 0 and 1")
    return result


def _planning_mapping(value: Any, *, default_key: str = "value") -> dict[str, Any]:
    """Normalize a declared delta while rejecting arbitrary object graphs."""

    if isinstance(value, Mapping):
        return _public_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            str(index): _public_value(item) for index, item in enumerate(value[:64])
        }
    if value in (None, ""):
        return {}
    return {default_key: _public_value(value)}


def _planning_declarations(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        # Mapping inputs are accepted at the adapter boundary, but the router
        # receives only stable declaration identifiers.
        return _strings(value.keys())
    return _strings(value)


@dataclass(frozen=True)
class PlanningObligationContext:
    """Minimal verifier-derived state needed to schedule one obligation."""

    obligation_id: str
    status: str = "unknown"
    support_status: str = "unknown"
    required_assurance: str = ""
    freshness: str = ""
    dependencies: tuple[str, ...] = ()
    reusable_receipt_ids: tuple[str, ...] = ()
    unsupported_semantics: tuple[str, ...] = ()
    fallback_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        obligation_id = str(self.obligation_id or "").strip()
        if not obligation_id:
            raise ProofContextError("planning obligation_id is required")
        object.__setattr__(self, "obligation_id", obligation_id)
        for name in (
            "status",
            "support_status",
            "required_assurance",
            "freshness",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in (
            "dependencies",
            "reusable_receipt_ids",
            "unsupported_semantics",
            "fallback_checks",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "status": self.status,
            "support_status": self.support_status,
            "required_assurance": self.required_assurance,
            "freshness": self.freshness,
            "dependencies": list(self.dependencies),
            "reusable_receipt_ids": list(self.reusable_receipt_ids),
            "unsupported_semantics": list(self.unsupported_semantics),
            "fallback_checks": list(self.fallback_checks),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningObligationContext":
        return cls(
            obligation_id=str(payload.get("obligation_id") or ""),
            status=str(payload.get("status") or "unknown"),
            support_status=str(payload.get("support_status") or "unknown"),
            required_assurance=str(payload.get("required_assurance") or ""),
            freshness=str(payload.get("freshness") or ""),
            dependencies=tuple(payload.get("dependencies") or ()),
            reusable_receipt_ids=tuple(payload.get("reusable_receipt_ids") or ()),
            unsupported_semantics=tuple(payload.get("unsupported_semantics") or ()),
            fallback_checks=tuple(payload.get("fallback_checks") or ()),
        )


@dataclass(frozen=True)
class PlanningCandidateContext:
    """Allowlisted proof-aware candidate declaration exposed to the router."""

    candidate_id: str
    obligation_impact: tuple[str, ...]
    required_assurance: str
    proof_cost: float
    cache_likelihood: float
    dependencies: tuple[str, ...]
    expected_evidence_delta: tuple[str, ...]
    resource_classes: tuple[str, ...] = ()
    proof_critical_path: float = 0.0
    downstream_unlock_value: float = 0.0
    risk: float = 0.0
    freshness: float = 1.0

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id or "").strip()
        if not candidate_id:
            raise ProofContextError("planning candidate_id is required")
        object.__setattr__(self, "candidate_id", candidate_id)
        impact = _planning_declarations(self.obligation_impact)
        if not impact:
            raise ProofContextError("planning candidate obligation_impact is required")
        object.__setattr__(self, "obligation_impact", impact)
        assurance = str(self.required_assurance or "").strip()
        if not assurance:
            raise ProofContextError("planning candidate required_assurance is required")
        object.__setattr__(self, "required_assurance", assurance)
        object.__setattr__(
            self,
            "proof_cost",
            _finite_non_negative_number(self.proof_cost, label="proof_cost"),
        )
        object.__setattr__(
            self,
            "cache_likelihood",
            _unit_interval_number(self.cache_likelihood, label="cache_likelihood"),
        )
        object.__setattr__(
            self,
            "risk",
            _unit_interval_number(self.risk, label="risk"),
        )
        object.__setattr__(
            self,
            "freshness",
            _unit_interval_number(self.freshness, label="freshness"),
        )
        object.__setattr__(
            self,
            "proof_critical_path",
            _finite_non_negative_number(
                self.proof_critical_path,
                label="proof_critical_path",
            ),
        )
        object.__setattr__(
            self,
            "downstream_unlock_value",
            _finite_non_negative_number(
                self.downstream_unlock_value,
                label="downstream_unlock_value",
            ),
        )
        object.__setattr__(self, "dependencies", _strings(self.dependencies))
        delta = _planning_declarations(self.expected_evidence_delta)
        if not delta:
            raise ProofContextError(
                "planning candidate expected_evidence_delta is required"
            )
        object.__setattr__(self, "expected_evidence_delta", delta)
        object.__setattr__(self, "resource_classes", _strings(self.resource_classes))

    @property
    def branch_id(self) -> str:
        """Compatibility with general plan-branch evaluators."""

        return self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "obligation_impact": list(self.obligation_impact),
            "required_assurance": self.required_assurance,
            "proof_cost": self.proof_cost,
            "cache_likelihood": self.cache_likelihood,
            "dependencies": list(self.dependencies),
            "expected_evidence_delta": list(self.expected_evidence_delta),
            "resource_classes": list(self.resource_classes),
            "proof_critical_path": self.proof_critical_path,
            "downstream_unlock_value": self.downstream_unlock_value,
            "risk": self.risk,
            "freshness": self.freshness,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningCandidateContext":
        nested_candidate = payload.get("candidate")
        if isinstance(nested_candidate, Mapping):
            payload = nested_candidate
        return cls(
            candidate_id=str(
                payload.get("candidate_id")
                or payload.get("branch_id")
                or payload.get("plan_id")
                or ""
            ),
            obligation_impact=_planning_declarations(payload.get("obligation_impact")),
            required_assurance=str(payload.get("required_assurance") or ""),
            proof_cost=payload.get(
                "proof_cost", payload.get("estimated_proof_cost", 0)
            ),
            cache_likelihood=payload.get("cache_likelihood", 0),
            dependencies=tuple(payload.get("dependencies") or ()),
            expected_evidence_delta=_planning_declarations(
                payload.get("expected_evidence_delta")
            ),
            resource_classes=tuple(
                payload.get("resource_classes")
                or payload.get("available_resource_classes")
                or ()
            ),
            proof_critical_path=payload.get("proof_critical_path", 0),
            downstream_unlock_value=payload.get("downstream_unlock_value", 0),
            risk=payload.get("risk", 0),
            freshness=payload.get("freshness", 1),
        )


@dataclass(frozen=True)
class RejectedPlanAlternative:
    """A rejected candidate identity with its bounded durable rationale."""

    candidate_id: str
    rationale: tuple[str, ...]
    score_millionths: int | None = None

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id or "").strip()
        if not candidate_id:
            raise ProofContextError("rejected alternative candidate_id is required")
        rationale = _strings(self.rationale)
        if not rationale:
            raise ProofContextError("rejected alternative rationale is required")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "rationale", rationale)
        if self.score_millionths is not None:
            object.__setattr__(self, "score_millionths", int(self.score_millionths))

    @property
    def reason(self) -> str:
        return "; ".join(self.rationale)

    @property
    def rationale_bytes(self) -> int:
        return len(_canonical_json(list(self.rationale)).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "rationale": list(self.rationale),
        }
        if self.score_millionths is not None:
            result["score_millionths"] = self.score_millionths
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RejectedPlanAlternative":
        rationale = payload.get("rationale")
        if rationale is None:
            rationale = payload.get("reason")
        return cls(
            candidate_id=str(
                payload.get("candidate_id") or payload.get("branch_id") or ""
            ),
            rationale=_strings(rationale),
            score_millionths=(
                int(payload["score_millionths"])
                if payload.get("score_millionths") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ProofPlanningContextUsage:
    bytes: int = 0
    tokens: int = 0
    candidates: int = 0
    obligations: int = 0
    rejected_alternatives: int = 0
    rationale_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "bytes",
            "tokens",
            "candidates",
            "obligations",
            "rejected_alternatives",
            "rationale_bytes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ProofContextError(
                    "proof planning context usage values must be non-negative"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, int]:
        return {
            "bytes": self.bytes,
            "tokens": self.tokens,
            "candidates": self.candidates,
            "obligations": self.obligations,
            "rejected_alternatives": self.rejected_alternatives,
            "rationale_bytes": self.rationale_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofPlanningContextUsage":
        return cls(
            **{
                name: int(payload.get(name) or 0)
                for name in (
                    "bytes",
                    "tokens",
                    "candidates",
                    "obligations",
                    "rejected_alternatives",
                    "rationale_bytes",
                )
            }
        )


@dataclass(frozen=True)
class ProofPlanningContextCapsule:
    """Small immutable proof projection and the only router-facing payload."""

    task_id: str
    source_capsule_id: str
    limits: ProofPlanningContextLimits
    obligations: tuple[PlanningObligationContext, ...] = ()
    candidates: tuple[PlanningCandidateContext, ...] = ()
    selected_candidate_id: str = ""
    rejected_alternatives: tuple[RejectedPlanAlternative, ...] = ()
    proof_critical_path: tuple[str, ...] = ()
    available_resource_classes: tuple[str, ...] = ()
    required_fallback_checks: tuple[str, ...] = ()
    source_truncated: bool = False
    usage: ProofPlanningContextUsage = field(default_factory=ProofPlanningContextUsage)
    truncated: bool = False
    omitted: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_id = str(self.task_id or "").strip()
        source_capsule_id = str(self.source_capsule_id or "").strip()
        if not task_id:
            raise ProofContextError("proof planning context task_id is required")
        if not source_capsule_id:
            raise ProofContextError(
                "proof planning context source_capsule_id is required"
            )
        if not isinstance(self.limits, ProofPlanningContextLimits):
            raise ProofContextError(
                "proof planning context limits must be ProofPlanningContextLimits"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "source_capsule_id", source_capsule_id)
        for name, item_type in (
            ("obligations", PlanningObligationContext),
            ("candidates", PlanningCandidateContext),
            ("rejected_alternatives", RejectedPlanAlternative),
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(item, item_type) for item in values):
                raise ProofContextError(
                    f"proof planning context {name} has an invalid entry"
                )
            object.__setattr__(self, name, values)
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ProofContextError("proof planning candidate ids must be unique")
        rejected_ids = [item.candidate_id for item in self.rejected_alternatives]
        if len(rejected_ids) != len(set(rejected_ids)):
            raise ProofContextError(
                "rejected proof planning candidate ids must be unique"
            )
        selected = str(self.selected_candidate_id or "").strip()
        if selected and selected not in candidate_ids:
            raise ProofContextError(
                "selected_candidate_id must identify a retained candidate"
            )
        object.__setattr__(self, "selected_candidate_id", selected)
        for name in (
            "proof_critical_path",
            "available_resource_classes",
            "required_fallback_checks",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        if not isinstance(self.usage, ProofPlanningContextUsage):
            raise ProofContextError(
                "proof planning context usage must be ProofPlanningContextUsage"
            )
        object.__setattr__(
            self,
            "omitted",
            {
                str(name): int(value)
                for name, value in self.omitted.items()
                if int(value) >= 0
            },
        )

    @property
    def capsule_id(self) -> str:
        payload = self._dict(include_identity=False)
        payload["usage"] = {
            **self.usage.to_dict(),
            "bytes": 0,
            "tokens": 0,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return f"proof-planning-context:sha256:{digest}"

    def _dict(self, *, include_identity: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": PROOF_PLANNING_CONTEXT_CAPSULE_SCHEMA,
            "task_id": self.task_id,
            "source_capsule_id": self.source_capsule_id,
            "limits": self.limits.to_dict(),
            "obligations": [item.to_dict() for item in self.obligations],
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "rejected_alternatives": [
                item.to_dict() for item in self.rejected_alternatives
            ],
            "proof_critical_path": list(self.proof_critical_path),
            "available_resource_classes": list(self.available_resource_classes),
            "required_fallback_checks": list(self.required_fallback_checks),
            "source_truncated": self.source_truncated,
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
        """Return the already-bounded router payload without loading more data."""

        return self.to_json()

    render_for_router = to_prompt
    render_for_codex = to_prompt

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> "ProofPlanningContextCapsule":
        schema = str(payload.get("schema") or "")
        if schema != PROOF_PLANNING_CONTEXT_CAPSULE_SCHEMA:
            raise ProofContextError(
                f"unsupported proof planning context schema: {schema}"
            )
        limits_payload = payload.get("limits")
        usage_payload = payload.get("usage")
        if not isinstance(limits_payload, Mapping) or not isinstance(
            usage_payload, Mapping
        ):
            raise ProofContextError(
                "proof planning context limits and usage must be mappings"
            )
        result = cls(
            task_id=str(payload.get("task_id") or ""),
            source_capsule_id=str(payload.get("source_capsule_id") or ""),
            limits=ProofPlanningContextLimits.from_dict(limits_payload),
            obligations=tuple(
                PlanningObligationContext.from_dict(item)
                for item in payload.get("obligations") or ()
            ),
            candidates=tuple(
                PlanningCandidateContext.from_dict(item)
                for item in payload.get("candidates") or ()
            ),
            selected_candidate_id=str(payload.get("selected_candidate_id") or ""),
            rejected_alternatives=tuple(
                RejectedPlanAlternative.from_dict(item)
                for item in payload.get("rejected_alternatives") or ()
            ),
            proof_critical_path=tuple(payload.get("proof_critical_path") or ()),
            available_resource_classes=tuple(
                payload.get("available_resource_classes") or ()
            ),
            required_fallback_checks=tuple(
                payload.get("required_fallback_checks") or ()
            ),
            source_truncated=bool(payload.get("source_truncated")),
            usage=ProofPlanningContextUsage.from_dict(usage_payload),
            truncated=bool(payload.get("truncated")),
            omitted=payload.get("omitted") or {},
        )
        claimed_id = str(payload.get("capsule_id") or "")
        if claimed_id and claimed_id != result.capsule_id:
            raise ProofContextError(
                "proof planning context identity does not match payload"
            )
        text = result.to_json()
        byte_count = len(text.encode("utf-8"))
        token_count = int(token_counter(text))
        if byte_count != result.usage.bytes:
            raise ProofContextError(
                "proof planning context byte usage does not match payload"
            )
        if token_count != result.usage.tokens:
            raise ProofContextError(
                "proof planning context token usage does not match payload"
            )
        result._validate_limits()
        return result

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> "ProofPlanningContextCapsule":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProofContextError("proof planning context JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ProofContextError(
                "proof planning context JSON must contain an object"
            )
        return cls.from_dict(payload, token_counter=token_counter)

    def _validate_limits(self) -> None:
        if len(self.candidates) > self.limits.max_candidates:
            raise ProofContextBudgetError(
                "proof planning context exceeds its candidate limit"
            )
        if len(self.obligations) > self.limits.max_obligations:
            raise ProofContextBudgetError(
                "proof planning context exceeds its obligation limit"
            )
        if len(self.rejected_alternatives) > self.limits.max_rejected_alternatives:
            raise ProofContextBudgetError(
                "proof planning context exceeds its rejected alternative limit"
            )
        if any(
            item.rationale_bytes > self.limits.max_rationale_bytes
            for item in self.rejected_alternatives
        ):
            raise ProofContextBudgetError(
                "proof planning context exceeds its rationale limit"
            )
        if len(self.proof_critical_path) > self.limits.max_dependencies:
            raise ProofContextBudgetError(
                "proof planning context exceeds its dependency limit"
            )
        if len(self.available_resource_classes) > self.limits.max_resource_classes:
            raise ProofContextBudgetError(
                "proof planning context exceeds its resource class limit"
            )
        if self.usage.bytes > self.limits.max_bytes:
            raise ProofContextBudgetError(
                "proof planning context exceeds its byte limit"
            )
        if self.usage.tokens > self.limits.max_tokens:
            raise ProofContextBudgetError(
                "proof planning context exceeds its token limit"
            )


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
        "conclusion",
        "template_id",
        "template_version",
        "template_semantic_hash",
        "assumptions",
        "premise_ids",
        "allowed_premise_ids",
        "ast_scope_ids",
        "source_scope",
        "source_scope_ids",
        "repository_tree_id",
        "canonical_source_digest",
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
        "authoritative_verdict",
        "kernel_accepted",
        "authoritative_assurance",
        "assurance",
        "freshness",
        "counterexample",
        "counterexamples",
        "unsat_core",
        "diagnostics",
        "failure_code",
        "failure_reason",
        "artifact_kind",
        "stage",
        "artifact_id",
        "theorem_id",
        "theorem_equivalence_key",
        "proposal_kind",
        "proof_text",
        "draft_text",
        "decomposition",
        "verified",
        "kernel_checked",
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
        "artifact_kind",
        "stage",
        "artifact_id",
        "theorem_id",
        "theorem_equivalence_key",
        "proposal_kind",
        "proof_text",
        "draft_text",
        "decomposition",
        "verified",
        "kernel_checked",
        "assurance",
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


def _object_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise ProofContextError(
        f"proof planning value must be mapping-like, got {type(value).__name__}"
    )


def _candidate_context(value: Any) -> PlanningCandidateContext:
    if isinstance(value, PlanningCandidateContext):
        return value
    nested = getattr(value, "candidate", None)
    if nested is not None:
        return _candidate_context(nested)
    return PlanningCandidateContext.from_dict(_object_mapping(value))


def _planning_obligations(
    capsule: ProofContextCapsule,
) -> tuple[PlanningObligationContext, ...]:
    records: dict[str, dict[str, Any]] = {}
    proof_receipts: dict[str, set[str]] = {}
    proof_freshness: dict[str, str] = {}
    proof_status: dict[str, str] = {}
    for entry in capsule.trusted_facts:
        fields = entry.fields
        obligation_id = str(fields.get("obligation_id") or "").strip()
        normalized_kind = entry.kind.strip().casefold().replace("-", "_")
        if (
            normalized_kind
            in {
                "obligation",
                "proof_obligation",
                "code_proof_obligation",
            }
            and obligation_id
        ):
            records[obligation_id] = {
                "obligation_id": obligation_id,
                "status": str(fields.get("status") or "unknown"),
                "support_status": str(fields.get("support_status") or "unknown"),
                "required_assurance": str(fields.get("required_assurance") or ""),
                "freshness": str(fields.get("freshness") or ""),
                "dependencies": _strings(fields.get("premise_ids")),
                "unsupported_semantics": _strings(fields.get("unsupported_semantics")),
                "fallback_checks": _strings(
                    (
                        *_strings(fields.get("fallback_checks")),
                        *_strings(fields.get("fallback_tests")),
                        *_strings(fields.get("required_tests")),
                    )
                ),
            }
        elif (
            normalized_kind
            in {
                "proof",
                "proof_receipt",
                "formal_proof_receipt",
                "receipt",
            }
            and obligation_id
        ):
            record = records.setdefault(
                obligation_id,
                {
                    "obligation_id": obligation_id,
                    "status": "unknown",
                    "support_status": "supported",
                    "required_assurance": str(
                        fields.get("required_assurance")
                        or fields.get("authoritative_assurance")
                        or fields.get("assurance")
                        or ""
                    ),
                    "freshness": str(fields.get("freshness") or ""),
                    "dependencies": (),
                    "unsupported_semantics": (),
                    "fallback_checks": (),
                },
            )
            if not record["required_assurance"]:
                record["required_assurance"] = str(
                    fields.get("required_assurance")
                    or fields.get("authoritative_assurance")
                    or fields.get("assurance")
                    or ""
                )
            verdict = str(fields.get("verdict") or fields.get("status") or "").lower()
            freshness = str(fields.get("freshness") or "").lower()
            receipt_id = str(
                fields.get("receipt_id") or fields.get("proof_id") or entry.record_id
            ).strip()
            if verdict in {"proved", "passed", "verified", "valid"} and freshness in {
                "",
                "current",
                "fresh",
                "valid",
            }:
                proof_receipts.setdefault(obligation_id, set()).add(receipt_id)
            if freshness:
                proof_freshness[obligation_id] = freshness
            if verdict:
                proof_status[obligation_id] = verdict
    for entry in capsule.unsupported_semantics:
        obligation_id = str(entry.fields.get("obligation_id") or "").strip()
        if not obligation_id:
            continue
        record = records.setdefault(
            obligation_id,
            {
                "obligation_id": obligation_id,
                "status": "unknown",
                "support_status": "unsupported",
                "required_assurance": "",
                "freshness": "",
                "dependencies": (),
                "unsupported_semantics": (),
                "fallback_checks": (),
            },
        )
        record["support_status"] = str(
            entry.fields.get("support_status") or "unsupported"
        )
        record["unsupported_semantics"] = _strings(
            (
                *_strings(record.get("unsupported_semantics")),
                *_strings(entry.fields.get("reasons")),
            )
        )
    results: list[PlanningObligationContext] = []
    for obligation_id in sorted(records):
        record = records[obligation_id]
        status = proof_status.get(obligation_id) or str(record["status"])
        freshness = proof_freshness.get(obligation_id) or str(record["freshness"])
        results.append(
            PlanningObligationContext(
                obligation_id=obligation_id,
                status=status,
                support_status=str(record["support_status"]),
                required_assurance=str(record["required_assurance"]),
                freshness=freshness,
                dependencies=tuple(record["dependencies"]),
                reusable_receipt_ids=tuple(
                    sorted(proof_receipts.get(obligation_id, ()))
                ),
                unsupported_semantics=tuple(record["unsupported_semantics"]),
                fallback_checks=tuple(record["fallback_checks"]),
            )
        )
    return tuple(results)


def _evaluation_value(evaluation: Any, name: str, default: Any) -> Any:
    if evaluation is None:
        return default
    if isinstance(evaluation, Mapping):
        return evaluation.get(name, default)
    return getattr(evaluation, name, default)


def _candidate_identity(value: Any) -> str:
    if isinstance(value, PlanningCandidateContext):
        return value.candidate_id
    nested_value = getattr(value, "candidate", None)
    if nested_value is not None:
        return _candidate_identity(nested_value)
    payload = _object_mapping(value)
    nested_candidate = payload.get("candidate")
    if isinstance(nested_candidate, Mapping):
        payload = nested_candidate
    nested = payload.get("branch")
    if isinstance(nested, Mapping):
        payload = nested
    return str(
        payload.get("candidate_id")
        or payload.get("branch_id")
        or payload.get("plan_id")
        or ""
    ).strip()


def _with_planning_usage(
    capsule: ProofPlanningContextCapsule,
    *,
    token_counter: Callable[[str], int],
) -> ProofPlanningContextCapsule:
    result = capsule
    for _ in range(8):
        text = result.to_json()
        usage = ProofPlanningContextUsage(
            bytes=len(text.encode("utf-8")),
            tokens=int(token_counter(text)),
            candidates=len(result.candidates),
            obligations=len(result.obligations),
            rejected_alternatives=len(result.rejected_alternatives),
            rationale_bytes=sum(
                item.rationale_bytes for item in result.rejected_alternatives
            ),
        )
        updated = replace(result, usage=usage)
        if updated.usage == result.usage:
            return updated
        result = updated
    return result


def build_proof_planning_context_capsule(
    proof_context: ProofContextCapsule,
    *,
    candidates: Sequence[Any] = (),
    evaluation: Any = None,
    available_resource_classes: Sequence[str] | str = (),
    proof_critical_path: Sequence[str] | str = (),
    limits: ProofPlanningContextLimits | None = None,
    token_counter: Callable[[str], int] = estimate_context_tokens,
) -> ProofPlanningContextCapsule:
    """Build the bounded proof-only projection passed to a planning router.

    ``evaluation`` may be a :class:`PlanEvaluation`-like object or mapping.
    When present, its selected and rejected candidates are used to retain the
    chosen identity, score, and every rationale that fits the explicit
    rejection budget.  No source, transcript, arbitrary evidence field, or
    untrusted graph suggestion crosses this boundary.
    """

    if not isinstance(proof_context, ProofContextCapsule):
        raise ProofContextError("proof_context must be a ProofContextCapsule")
    if not callable(token_counter):
        raise ProofContextError("token_counter must be callable")
    budget = limits or ProofPlanningContextLimits()
    normalized_candidates = [_candidate_context(item) for item in candidates]
    selected_value = _evaluation_value(evaluation, "selected", None)
    rejected_values = tuple(_evaluation_value(evaluation, "rejected", ()) or ())
    if not normalized_candidates and selected_value is not None:
        normalized_candidates = [
            _candidate_context(item) for item in (selected_value, *rejected_values)
        ]
    selected_id = (
        _candidate_identity(selected_value) if selected_value is not None else ""
    )
    candidate_by_id: dict[str, PlanningCandidateContext] = {}
    for candidate in normalized_candidates:
        if candidate.candidate_id in candidate_by_id:
            raise ProofContextError(
                f"duplicate proof planning candidate_id: {candidate.candidate_id}"
            )
        candidate_by_id[candidate.candidate_id] = candidate
    if selected_id and selected_id not in candidate_by_id:
        candidate = _candidate_context(selected_value)
        candidate_by_id[candidate.candidate_id] = candidate

    # The selected alternative gets first claim on the finite candidate
    # budget; remaining candidates are ordered canonically.
    candidate_order = sorted(
        candidate_by_id.values(),
        key=lambda item: (
            0 if item.candidate_id == selected_id else 1,
            item.candidate_id,
        ),
    )
    candidate_omitted = max(0, len(candidate_order) - budget.max_candidates)
    candidate_order = candidate_order[: budget.max_candidates]
    candidate_dependency_omitted = sum(
        max(0, len(item.dependencies) - budget.max_dependencies)
        for item in candidate_order
    )
    candidate_resource_omitted = sum(
        max(0, len(item.resource_classes) - budget.max_resource_classes)
        for item in candidate_order
    )
    candidate_order = [
        replace(
            item,
            dependencies=item.dependencies[: budget.max_dependencies],
            resource_classes=item.resource_classes[: budget.max_resource_classes],
        )
        for item in candidate_order
    ]
    if selected_id and all(
        item.candidate_id != selected_id for item in candidate_order
    ):
        raise ProofContextBudgetError(
            "proof planning limits cannot contain the selected candidate"
        )

    obligations = list(_planning_obligations(proof_context))
    obligation_omitted = max(0, len(obligations) - budget.max_obligations)
    obligations = obligations[: budget.max_obligations]
    obligations = [
        replace(
            item,
            dependencies=item.dependencies[: budget.max_dependencies],
        )
        for item in obligations
    ]

    scores = _evaluation_value(evaluation, "scores", {}) or {}
    rationales = _evaluation_value(evaluation, "rationales", {}) or {}
    if not isinstance(scores, Mapping):
        scores = {}
    if not isinstance(rationales, Mapping):
        rationales = {}
    rejected: list[RejectedPlanAlternative] = []
    rejected_omitted = 0
    minimum_rationale_bytes = len(_canonical_json(["x"]).encode("utf-8"))
    if rejected_values and budget.max_rationale_bytes < minimum_rationale_bytes:
        raise ProofContextBudgetError(
            "proof planning rationale limit is too small to retain a rejection reason"
        )
    for value in rejected_values:
        candidate_id = _candidate_identity(value)
        if not candidate_id:
            raise ProofContextError(
                "evaluated rejected candidate has no stable identity"
            )
        rationale = _strings(rationales.get(candidate_id))
        value_payload = _object_mapping(value)
        if not rationale:
            rationale = _strings(
                value_payload.get("rationale") or value_payload.get("reason")
            )
        if not rationale:
            raise ProofContextError(
                f"rejected candidate {candidate_id!r} has no rationale"
            )
        joined_rationale = "; ".join(rationale)
        rationale_text, was_truncated = _truncate_utf8(
            joined_rationale,
            max(1, budget.max_rationale_bytes - 4),
        )
        # JSON quoting can expand quotes and backslashes.  Reduce the public
        # reason until the serialized rationale, not merely its raw text,
        # satisfies the per-alternative limit.
        while True:
            bounded_rationale = (rationale_text or "x",)
            measured_rationale = RejectedPlanAlternative(
                candidate_id=candidate_id,
                rationale=bounded_rationale,
            ).rationale_bytes
            if measured_rationale <= budget.max_rationale_bytes:
                break
            raw = rationale_text.encode("utf-8")
            reduction = max(1, measured_rationale - budget.max_rationale_bytes)
            shortened = raw[:-reduction].decode("utf-8", errors="ignore")
            if not shortened or shortened == rationale_text:
                rationale_text = "x"
            else:
                rationale_text = shortened
            was_truncated = True
        if len(rejected) >= budget.max_rejected_alternatives:
            rejected_omitted += 1
            continue
        rejected.append(
            RejectedPlanAlternative(
                candidate_id=candidate_id,
                rationale=bounded_rationale,
                score_millionths=(
                    int(scores[candidate_id])
                    if candidate_id in scores
                    else (
                        int(value_payload["score_millionths"])
                        if value_payload.get("score_millionths") is not None
                        else None
                    )
                ),
            )
        )
        rejected_omitted += int(was_truncated)

    critical_path = _strings(proof_critical_path)
    if not critical_path:
        critical_path = tuple(
            item.obligation_id
            for item in obligations
            if item.status.lower() not in {"proved", "passed", "verified", "valid"}
        )
    path_omitted = max(0, len(critical_path) - budget.max_dependencies)
    critical_path = critical_path[: budget.max_dependencies]
    resource_classes = _strings(available_resource_classes)[
        : budget.max_resource_classes
    ]
    resource_omitted = max(
        0,
        len(_strings(available_resource_classes)) - len(resource_classes),
    )
    omitted = {
        "obligations": obligation_omitted,
        "candidates": candidate_omitted,
        "rejected_alternatives": rejected_omitted,
        "proof_critical_path": path_omitted,
        "available_resource_classes": resource_omitted,
        "candidate_dependencies": candidate_dependency_omitted,
        "candidate_resource_classes": candidate_resource_omitted,
        "fallback_checks": 0,
    }
    result = ProofPlanningContextCapsule(
        task_id=proof_context.query.task_id,
        source_capsule_id=proof_context.capsule_id,
        limits=budget,
        proof_critical_path=critical_path,
        available_resource_classes=resource_classes,
        source_truncated=proof_context.truncated,
        truncated=proof_context.truncated or any(omitted.values()),
        omitted=omitted,
    )

    def fits(value: ProofPlanningContextCapsule) -> bool:
        measured = _with_planning_usage(value, token_counter=token_counter)
        return (
            measured.usage.bytes <= budget.max_bytes
            and measured.usage.tokens <= budget.max_tokens
        )

    if not fits(result):
        raise ProofContextBudgetError(
            "proof planning byte/token limits are too small for the mandatory envelope"
        )

    # Fallback requirements and the selected candidate are fail-closed inputs,
    # so they precede optional obligation/candidate/rejection detail.
    for check in proof_context.required_fallback_checks:
        candidate = replace(
            result,
            required_fallback_checks=(
                *result.required_fallback_checks,
                check,
            ),
        )
        if not fits(candidate):
            raise ProofContextBudgetError(
                "proof planning limits cannot contain required fallback checks"
            )
        result = candidate
    for obligation in obligations:
        candidate = replace(result, obligations=(*result.obligations, obligation))
        if fits(candidate):
            result = candidate
        else:
            omitted["obligations"] += 1
            result = replace(result, truncated=True, omitted=omitted)
    for plan_candidate in candidate_order:
        candidate = replace(
            result,
            candidates=(*result.candidates, plan_candidate),
            selected_candidate_id=(
                selected_id
                if plan_candidate.candidate_id == selected_id
                else result.selected_candidate_id
            ),
        )
        if fits(candidate):
            result = candidate
        elif plan_candidate.candidate_id == selected_id:
            raise ProofContextBudgetError(
                "proof planning limits cannot contain the selected candidate"
            )
        else:
            omitted["candidates"] += 1
            result = replace(result, truncated=True, omitted=omitted)
    for alternative in rejected:
        candidate = replace(
            result,
            rejected_alternatives=(
                *result.rejected_alternatives,
                alternative,
            ),
        )
        if fits(candidate):
            result = candidate
        else:
            omitted["rejected_alternatives"] += 1
            result = replace(result, truncated=True, omitted=omitted)

    result = replace(result, omitted=omitted)
    result = _with_planning_usage(result, token_counter=token_counter)
    result._validate_limits()
    if selected_id and result.selected_candidate_id != selected_id:
        raise ProofContextBudgetError(
            "proof planning limits cannot retain the selected candidate"
        )
    return result


build_planning_proof_context_capsule = build_proof_planning_context_capsule
generate_proof_planning_context_capsule = build_proof_planning_context_capsule
PlanningProofContext = ProofPlanningContextCapsule
PlanningProofContextCapsule = ProofPlanningContextCapsule


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
    "AllowedPremise",
    "CompactProofFailure",
    "DEFAULT_MAX_CONTEXT_BYTES",
    "DEFAULT_MAX_CONTEXT_ROWS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_MAX_GRAPH_HOPS",
    "DEFAULT_MAX_LEANSTRAL_FAILURES",
    "DEFAULT_MAX_LEANSTRAL_FAILURE_BYTES",
    "DEFAULT_MAX_LEANSTRAL_PREMISES",
    "DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFTS",
    "DEFAULT_MAX_LEANSTRAL_REUSABLE_DRAFT_BYTES",
    "DEFAULT_MAX_LEANSTRAL_TRUSTED_RECEIPTS",
    "DEFAULT_MAX_PLANNING_CANDIDATES",
    "DEFAULT_MAX_PLANNING_CONTEXT_BYTES",
    "DEFAULT_MAX_PLANNING_CONTEXT_TOKENS",
    "DEFAULT_MAX_PLANNING_DEPENDENCIES",
    "DEFAULT_MAX_PLANNING_OBLIGATIONS",
    "DEFAULT_MAX_PLANNING_RESOURCE_CLASSES",
    "DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES",
    "DEFAULT_MAX_PROOF_TRANSCRIPT_BYTES_TOTAL",
    "DEFAULT_MAX_PROOF_TRANSCRIPTS",
    "DEFAULT_MAX_REJECTED_ALTERNATIVES",
    "DEFAULT_MAX_REJECTION_RATIONALE_BYTES",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DEFAULT_MAX_SOURCE_EXCERPT_BYTES",
    "DEFAULT_MAX_SOURCE_EXCERPTS",
    "PROOF_CONTEXT_CAPSULE_SCHEMA",
    "PROOF_CONTEXT_LIMITS_SCHEMA",
    "PROOF_CONTEXT_QUERY_SCHEMA",
    "PROOF_PLANNING_CONTEXT_CAPSULE_SCHEMA",
    "PROOF_PLANNING_CONTEXT_LIMITS_SCHEMA",
    "LEANSTRAL_FIXED_THEOREM_SCHEMA",
    "LEANSTRAL_PROMPT_LIMITS_SCHEMA",
    "LEANSTRAL_PROOF_CONTEXT_SCHEMA",
    "LEANSTRAL_PROOF_OUTPUT_SCHEMA",
    "FixedTheorem",
    "FixedTheoremIdentity",
    "FixedTheoremProofContext",
    "LeanstralAllowedPremise",
    "LeanstralCompactFailure",
    "LeanstralPromptLimits",
    "LeanstralProofContext",
    "LeanstralProofContextCapsule",
    "LeanstralReusableDraft",
    "LeanstralTrustedReceipt",
    "PlanningCandidateContext",
    "PlanningObligationContext",
    "PlanningProofContext",
    "PlanningProofContextCapsule",
    "ProofContextBudget",
    "ProofContextBudgetError",
    "ProofContextBuilder",
    "ProofContextCapsule",
    "ProofContextError",
    "ProofContextLimits",
    "ProofContextQuery",
    "ProofContextTarget",
    "ProofContextUsage",
    "ProofPlanningContextCapsule",
    "ProofPlanningContextLimits",
    "ProofPlanningContextUsage",
    "ProofTranscriptExcerpt",
    "RejectedPlanAlternative",
    "ReusableProofDraft",
    "SourceExcerpt",
    "TrustedPriorReceipt",
    "build_planning_proof_context_capsule",
    "build_proof_context_capsule",
    "build_leanstral_proof_context",
    "build_fixed_theorem_leanstral_context",
    "build_proof_planning_context_capsule",
    "estimate_context_tokens",
    "generate_proof_context_capsule",
    "generate_leanstral_proof_context",
    "generate_fixed_theorem_leanstral_prompt",
    "generate_proof_planning_context_capsule",
]
