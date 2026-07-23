"""Bounded, proof-carrying formal-plan context for implementation models.

This module is the dispatch boundary between a canonical, checked
``FormalWorkPlan`` and a language model.  A capsule contains one task
transition and only its bounded graph neighbourhood.  The supervisor-owned
theorem, acceptance policy, evidence, and content identities are carried as
immutable response bindings; model output can propose implementation work but
cannot replace those authoritative inputs.

The builder deliberately operates on the compiler and validator products
rather than reconstructing task semantics from prose.  All row, hop, byte,
token, and source-excerpt limits are applied while the capsule is built.  The
model invocation helper revalidates those limits immediately before calling a
provider.
"""

from __future__ import annotations

import json
import posixpath
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from .formal_plan_compiler import (
    CompilationStatus,
    PlanCompilationResult,
    PlanGraphProjection,
)
from .formal_plan_validator import (
    PlanValidationResult,
)
from .formal_planning_contracts import FormalWorkPlan, PlanTask
from .formal_verification_contracts import canonical_json, content_identity
from .proof_context import estimate_context_tokens


FORMAL_PLAN_CONTEXT_VERSION: Final = 1
FORMAL_PLAN_CONTEXT_CAPSULE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-context@1"
)
FORMAL_PLAN_CONTEXT_QUERY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-context-query@1"
)
FORMAL_PLAN_CONTEXT_LIMITS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-context-limits@1"
)
FORMAL_PLAN_GRAPH_SLICE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-graph-slice@1"
)
FORMAL_PLAN_MODEL_RESPONSE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-model-response@1"
)
FORMAL_PLAN_CONTEXT_MEASUREMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-plan-context-measurement@1"
)
IMPLEMENTATION_OUTCOME_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/implementation-outcome@1"
)

DEFAULT_MAX_FORMAL_PLAN_ROWS: Final = 96
DEFAULT_MAX_FORMAL_PLAN_GRAPH_HOPS: Final = 2
DEFAULT_MAX_FORMAL_PLAN_CONTEXT_BYTES: Final = 48 * 1024
DEFAULT_MAX_FORMAL_PLAN_CONTEXT_TOKENS: Final = 12_000
DEFAULT_MAX_FORMAL_PLAN_SOURCE_EXCERPTS: Final = 8
DEFAULT_MAX_FORMAL_PLAN_SOURCE_EXCERPT_BYTES: Final = 4 * 1024
DEFAULT_MAX_FORMAL_PLAN_SOURCE_BYTES: Final = 16 * 1024
DEFAULT_MAX_FORMAL_PLAN_AST_SYMBOLS: Final = 32
DEFAULT_MAX_FORMAL_PLAN_EVIDENCE: Final = 24
DEFAULT_MAX_FORMAL_PLAN_COUNTEREXAMPLES: Final = 8
DEFAULT_MAX_FORMAL_PLAN_OBLIGATIONS: Final = 32
DEFAULT_MAX_FORMAL_PLAN_ALLOWED_PATHS: Final = 32
DEFAULT_MAX_FORMAL_PLAN_TESTS: Final = 32


class FormalPlanContextError(ValueError):
    """Invalid or non-canonical formal-plan context input."""


class FormalPlanContextBudgetError(FormalPlanContextError):
    """A required capsule cannot fit within the configured limits."""


class FormalPlanResponseError(FormalPlanContextError):
    """A model response does not bind the immutable capsule inputs."""


class FormalPlanContextTarget(str, Enum):
    CODEX = "codex"
    LEANSTRAL = "leanstral"


class ImplementationOutcomeStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    NOT_RUN = "not_run"


def _text(value: Any, *, label: str, required: bool = False) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise FormalPlanContextError(f"{label} is required")
    return result


def _positive(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise FormalPlanContextError(f"{label} must be a {qualifier} integer")
    return value


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise FormalPlanContextError("expected a string or sequence of strings")
    return tuple(
        sorted({_text(item, label="value", required=True) for item in values})
    )


def _ordered_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray)
    ):
        raise FormalPlanContextError("expected a string or sequence of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, label="value", required=True)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        decoded = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise FormalPlanContextError("record must contain canonical JSON values") from exc
    if not isinstance(decoded, dict):
        raise FormalPlanContextError("record must be a mapping")
    return decoded


def _canonical_records(values: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(_canonical_mapping(item) for item in values)


def _truncate_utf8(text: str, maximum: int) -> tuple[str, bool]:
    raw = str(text).encode("utf-8")
    if len(raw) <= maximum:
        return str(text), False
    if maximum <= 0:
        return "", bool(raw)
    suffix = b"..."
    if maximum <= len(suffix):
        return raw[:maximum].decode("utf-8", errors="ignore"), True
    prefix = raw[: maximum - len(suffix)].decode("utf-8", errors="ignore")
    return prefix + suffix.decode("ascii"), True


def _path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    if raw.startswith("/") or "\x00" in raw:
        raise FormalPlanContextError("allowed paths must be repository-relative")
    normalized = posixpath.normpath(raw)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise FormalPlanContextError("allowed paths must remain inside the repository")
    return PurePosixPath(normalized).as_posix()


_PRIVATE_KEY_PARTS: Final = (
    "credential",
    "private_witness",
    "secret",
    "token_value",
    "repository_ast",
    "full_ast",
)


def _public_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)[:256]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return _public_mapping(value, depth=depth + 1)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_public_value(item, depth=depth + 1) for item in value[:64]]
    if hasattr(value, "to_dict"):
        return _public_value(value.to_dict(), depth=depth + 1)
    return str(value)


def _public_mapping(value: Mapping[str, Any], *, depth: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key in sorted(value, key=str):
        key = str(raw_key)
        lowered = key.lower()
        if any(part in lowered for part in _PRIVATE_KEY_PARTS):
            continue
        if lowered in {"stdout", "stderr", "transcript", "proof_log"}:
            continue
        result[key] = _public_value(value[raw_key], depth=depth)
    return _canonical_mapping(result)


@dataclass(frozen=True)
class FormalPlanContextLimits:
    """All limits applied before a capsule can be rendered or dispatched."""

    max_rows: int = DEFAULT_MAX_FORMAL_PLAN_ROWS
    max_graph_hops: int = DEFAULT_MAX_FORMAL_PLAN_GRAPH_HOPS
    max_bytes: int = DEFAULT_MAX_FORMAL_PLAN_CONTEXT_BYTES
    max_tokens: int = DEFAULT_MAX_FORMAL_PLAN_CONTEXT_TOKENS
    max_source_excerpts: int = DEFAULT_MAX_FORMAL_PLAN_SOURCE_EXCERPTS
    max_source_excerpt_bytes: int = DEFAULT_MAX_FORMAL_PLAN_SOURCE_EXCERPT_BYTES
    max_source_bytes: int = DEFAULT_MAX_FORMAL_PLAN_SOURCE_BYTES
    max_ast_symbols: int = DEFAULT_MAX_FORMAL_PLAN_AST_SYMBOLS
    max_trusted_evidence: int = DEFAULT_MAX_FORMAL_PLAN_EVIDENCE
    max_counterexamples: int = DEFAULT_MAX_FORMAL_PLAN_COUNTEREXAMPLES
    max_unresolved_obligations: int = DEFAULT_MAX_FORMAL_PLAN_OBLIGATIONS
    max_allowed_paths: int = DEFAULT_MAX_FORMAL_PLAN_ALLOWED_PATHS
    max_tests: int = DEFAULT_MAX_FORMAL_PLAN_TESTS

    def __post_init__(self) -> None:
        for name in (
            "max_rows",
            "max_bytes",
            "max_tokens",
            "max_source_excerpt_bytes",
            "max_source_bytes",
        ):
            object.__setattr__(
                self, name, _positive(getattr(self, name), label=name)
            )
        for name in (
            "max_graph_hops",
            "max_source_excerpts",
            "max_ast_symbols",
            "max_trusted_evidence",
            "max_counterexamples",
            "max_unresolved_obligations",
            "max_allowed_paths",
            "max_tests",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), label=name, allow_zero=True),
            )
        if self.max_source_excerpt_bytes > self.max_source_bytes:
            raise FormalPlanContextError(
                "max_source_excerpt_bytes cannot exceed max_source_bytes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_CONTEXT_LIMITS_SCHEMA,
            **{
                name: getattr(self, name)
                for name in (
                    "max_rows",
                    "max_graph_hops",
                    "max_bytes",
                    "max_tokens",
                    "max_source_excerpts",
                    "max_source_excerpt_bytes",
                    "max_source_bytes",
                    "max_ast_symbols",
                    "max_trusted_evidence",
                    "max_counterexamples",
                    "max_unresolved_obligations",
                    "max_allowed_paths",
                    "max_tests",
                )
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalPlanContextLimits":
        schema = payload.get("schema", FORMAL_PLAN_CONTEXT_LIMITS_SCHEMA)
        if schema != FORMAL_PLAN_CONTEXT_LIMITS_SCHEMA:
            raise FormalPlanContextError(f"unsupported limits schema: {schema}")
        defaults = cls()
        return cls(
            **{
                name: int(payload.get(name, getattr(defaults, name)))
                for name in (
                    "max_rows",
                    "max_graph_hops",
                    "max_bytes",
                    "max_tokens",
                    "max_source_excerpts",
                    "max_source_excerpt_bytes",
                    "max_source_bytes",
                    "max_ast_symbols",
                    "max_trusted_evidence",
                    "max_counterexamples",
                    "max_unresolved_obligations",
                    "max_allowed_paths",
                    "max_tests",
                )
            }
        )


@dataclass(frozen=True)
class FormalPlanContextQuery:
    """Exact selector for one canonical task transition."""

    task_id: str
    transition_event_id: str = ""
    ast_symbol_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id", _text(self.task_id, label="task_id", required=True)
        )
        object.__setattr__(
            self,
            "transition_event_id",
            _text(self.transition_event_id, label="transition_event_id"),
        )
        object.__setattr__(self, "ast_symbol_ids", _strings(self.ast_symbol_ids))
        object.__setattr__(self, "evidence_ids", _strings(self.evidence_ids))
        for value in (
            self.task_id,
            self.transition_event_id,
            *self.ast_symbol_ids,
            *self.evidence_ids,
        ):
            if any(marker in value for marker in ("*", "?", "[", "]")):
                raise FormalPlanContextError(
                    "formal-plan context selectors must be exact"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_CONTEXT_QUERY_SCHEMA,
            "task_id": self.task_id,
            "transition_event_id": self.transition_event_id,
            "ast_symbol_ids": list(self.ast_symbol_ids),
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalPlanContextQuery":
        schema = payload.get("schema", FORMAL_PLAN_CONTEXT_QUERY_SCHEMA)
        if schema != FORMAL_PLAN_CONTEXT_QUERY_SCHEMA:
            raise FormalPlanContextError(f"unsupported query schema: {schema}")
        return cls(
            task_id=payload.get("task_id", ""),
            transition_event_id=payload.get("transition_event_id", ""),
            ast_symbol_ids=tuple(payload.get("ast_symbol_ids") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
        )


@dataclass(frozen=True)
class FormalPlanSourceExcerpt:
    symbol_id: str
    path: str
    text: str
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "symbol_id", _text(self.symbol_id, label="symbol_id", required=True)
        )
        object.__setattr__(self, "path", _path(self.path))
        object.__setattr__(self, "text", str(self.text))
        if not isinstance(self.truncated, bool):
            raise FormalPlanContextError("source excerpt truncated must be boolean")

    @property
    def byte_count(self) -> int:
        return len(self.text.encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "path": self.path,
            "text": self.text,
            "byte_count": self.byte_count,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalPlanSourceExcerpt":
        result = cls(
            symbol_id=payload.get("symbol_id", ""),
            path=payload.get("path", ""),
            text=payload.get("text", ""),
            truncated=bool(payload.get("truncated", False)),
        )
        if "byte_count" in payload and int(payload["byte_count"]) != result.byte_count:
            raise FormalPlanContextError("source excerpt byte count does not match")
        return result


@dataclass(frozen=True)
class FormalPlanGraphSlice:
    nodes: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()
    max_hops: int = 0
    truncated: bool = False
    omitted_rows: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _canonical_records(self.nodes))
        object.__setattr__(self, "edges", _canonical_records(self.edges))
        object.__setattr__(
            self, "max_hops", _positive(self.max_hops, label="max_hops", allow_zero=True)
        )
        object.__setattr__(
            self,
            "omitted_rows",
            _positive(self.omitted_rows, label="omitted_rows", allow_zero=True),
        )
        if not isinstance(self.truncated, bool):
            raise FormalPlanContextError("graph slice truncated must be boolean")

    @property
    def row_count(self) -> int:
        return len(self.nodes) + len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_GRAPH_SLICE_SCHEMA,
            "nodes": [dict(item) for item in self.nodes],
            "edges": [dict(item) for item in self.edges],
            "row_count": self.row_count,
            "max_hops": self.max_hops,
            "truncated": self.truncated,
            "omitted_rows": self.omitted_rows,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalPlanGraphSlice":
        schema = payload.get("schema", FORMAL_PLAN_GRAPH_SLICE_SCHEMA)
        if schema != FORMAL_PLAN_GRAPH_SLICE_SCHEMA:
            raise FormalPlanContextError(f"unsupported graph slice schema: {schema}")
        result = cls(
            nodes=tuple(payload.get("nodes") or ()),
            edges=tuple(payload.get("edges") or ()),
            max_hops=int(payload.get("max_hops") or 0),
            truncated=bool(payload.get("truncated", False)),
            omitted_rows=int(payload.get("omitted_rows") or 0),
        )
        if "row_count" in payload and int(payload["row_count"]) != result.row_count:
            raise FormalPlanContextError("graph row count does not match payload")
        return result


@dataclass(frozen=True)
class FormalTaskTransition:
    task_id: str
    task_cid: str
    event_id: str
    event_kind: str
    actor_ids: tuple[str, ...]
    from_state: Any
    to_state: Any
    dependency_task_ids: tuple[str, ...] = ()
    terminal_states: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("task_id", "task_cid", "event_id", "event_kind"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), label=name, required=True),
            )
        object.__setattr__(self, "actor_ids", _strings(self.actor_ids))
        object.__setattr__(
            self, "dependency_task_ids", _strings(self.dependency_task_ids)
        )
        object.__setattr__(self, "terminal_states", _strings(self.terminal_states))
        object.__setattr__(self, "from_state", _public_value(self.from_state))
        object.__setattr__(self, "to_state", _public_value(self.to_state))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "actor_ids": list(self.actor_ids),
            "from_state": self.from_state,
            "to_state": self.to_state,
            "dependency_task_ids": list(self.dependency_task_ids),
            "terminal_states": list(self.terminal_states),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalTaskTransition":
        return cls(
            task_id=payload.get("task_id", ""),
            task_cid=payload.get("task_cid", ""),
            event_id=payload.get("event_id", ""),
            event_kind=payload.get("event_kind", ""),
            actor_ids=tuple(payload.get("actor_ids") or ()),
            from_state=payload.get("from_state"),
            to_state=payload.get("to_state"),
            dependency_task_ids=tuple(payload.get("dependency_task_ids") or ()),
            terminal_states=tuple(payload.get("terminal_states") or ()),
        )


@dataclass(frozen=True)
class FormalPlanContextUsage:
    rows: int = 0
    graph_hops: int = 0
    bytes: int = 0
    tokens: int = 0
    source_excerpts: int = 0
    source_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "rows",
            "graph_hops",
            "bytes",
            "tokens",
            "source_excerpts",
            "source_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), label=name, allow_zero=True),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "rows",
                "graph_hops",
                "bytes",
                "tokens",
                "source_excerpts",
                "source_bytes",
            )
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalPlanContextUsage":
        return cls(
            **{
                name: int(payload.get(name) or 0)
                for name in (
                    "rows",
                    "graph_hops",
                    "bytes",
                    "tokens",
                    "source_excerpts",
                    "source_bytes",
                )
            }
        )


@dataclass(frozen=True)
class FormalPlanResponseBinding:
    capsule_cid: str
    plan_cid: str
    task_cid: str
    theorem_cid: str
    acceptance_policy_cid: str
    authoritative_evidence_cid: str

    def __post_init__(self) -> None:
        for name in (
            "capsule_cid",
            "plan_cid",
            "task_cid",
            "theorem_cid",
            "acceptance_policy_cid",
            "authoritative_evidence_cid",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), label=name, required=True),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in (
                "capsule_cid",
                "plan_cid",
                "task_cid",
                "theorem_cid",
                "acceptance_policy_cid",
                "authoritative_evidence_cid",
            )
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalPlanResponseBinding":
        return cls(
            **{
                name: payload.get(name, "")
                for name in (
                    "capsule_cid",
                    "plan_cid",
                    "task_cid",
                    "theorem_cid",
                    "acceptance_policy_cid",
                    "authoritative_evidence_cid",
                )
            }
        )


@dataclass(frozen=True)
class FormalPlanContextCapsule:
    """Immutable, already-budgeted model input for one checked transition."""

    target: FormalPlanContextTarget
    query: FormalPlanContextQuery
    limits: FormalPlanContextLimits
    plan_cid: str
    task_cid: str
    validation_cid: str
    repository_tree_cid: str
    transition: FormalTaskTransition
    theorem: Mapping[str, Any]
    acceptance_policy: Mapping[str, Any]
    assumptions: tuple[Mapping[str, Any], ...] = ()
    required_preconditions: tuple[Mapping[str, Any], ...] = ()
    required_effects: tuple[Mapping[str, Any], ...] = ()
    relevant_ast_symbols: tuple[Mapping[str, Any], ...] = ()
    trusted_evidence: tuple[Mapping[str, Any], ...] = ()
    counterexamples: tuple[Mapping[str, Any], ...] = ()
    allowed_paths: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    unresolved_obligations: tuple[Mapping[str, Any], ...] = ()
    graph_slice: FormalPlanGraphSlice = field(default_factory=FormalPlanGraphSlice)
    source_excerpts: tuple[FormalPlanSourceExcerpt, ...] = ()
    usage: FormalPlanContextUsage = field(default_factory=FormalPlanContextUsage)
    truncated: bool = False
    omitted: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = (
            self.target
            if isinstance(self.target, FormalPlanContextTarget)
            else FormalPlanContextTarget(str(self.target))
        )
        object.__setattr__(self, "target", target)
        if not isinstance(self.query, FormalPlanContextQuery):
            raise FormalPlanContextError("query must be FormalPlanContextQuery")
        if not isinstance(self.limits, FormalPlanContextLimits):
            raise FormalPlanContextError("limits must be FormalPlanContextLimits")
        if not isinstance(self.transition, FormalTaskTransition):
            raise FormalPlanContextError("transition must be FormalTaskTransition")
        if not isinstance(self.graph_slice, FormalPlanGraphSlice):
            raise FormalPlanContextError("graph_slice must be FormalPlanGraphSlice")
        if not isinstance(self.usage, FormalPlanContextUsage):
            raise FormalPlanContextError("usage must be FormalPlanContextUsage")
        for name in (
            "plan_cid",
            "task_cid",
            "validation_cid",
            "repository_tree_cid",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    label=name,
                    required=name != "repository_tree_cid",
                ),
            )
        if self.task_cid != self.transition.task_cid:
            raise FormalPlanContextError("transition does not bind the capsule task")
        object.__setattr__(self, "theorem", _canonical_mapping(self.theorem))
        object.__setattr__(
            self, "acceptance_policy", _canonical_mapping(self.acceptance_policy)
        )
        for name in (
            "assumptions",
            "required_preconditions",
            "required_effects",
            "relevant_ast_symbols",
            "trusted_evidence",
            "counterexamples",
            "unresolved_obligations",
        ):
            object.__setattr__(
                self, name, _canonical_records(tuple(getattr(self, name)))
            )
        object.__setattr__(
            self, "allowed_paths", tuple(sorted({_path(item) for item in self.allowed_paths}))
        )
        object.__setattr__(self, "tests", _strings(self.tests))
        if not all(
            isinstance(item, FormalPlanSourceExcerpt) for item in self.source_excerpts
        ):
            raise FormalPlanContextError("source_excerpts contains an invalid record")
        object.__setattr__(self, "source_excerpts", tuple(self.source_excerpts))
        object.__setattr__(
            self,
            "omitted",
            {
                str(key): _positive(value, label=f"omitted.{key}", allow_zero=True)
                for key, value in sorted(self.omitted.items())
            },
        )
        if not isinstance(self.truncated, bool):
            raise FormalPlanContextError("truncated must be boolean")

    @property
    def theorem_cid(self) -> str:
        return content_identity(self.theorem)

    @property
    def acceptance_policy_cid(self) -> str:
        return content_identity(self.acceptance_policy)

    @property
    def authoritative_evidence_cid(self) -> str:
        return content_identity([dict(item) for item in self.trusted_evidence])

    @property
    def capsule_cid(self) -> str:
        payload = self._dict(include_identity=False)
        payload["usage"] = {
            **self.usage.to_dict(),
            "bytes": 0,
            "tokens": 0,
        }
        return content_identity(payload)

    @property
    def capsule_id(self) -> str:
        return self.capsule_cid

    @property
    def bindings(self) -> FormalPlanResponseBinding:
        return FormalPlanResponseBinding(
            capsule_cid=self.capsule_cid,
            plan_cid=self.plan_cid,
            task_cid=self.task_cid,
            theorem_cid=self.theorem_cid,
            acceptance_policy_cid=self.acceptance_policy_cid,
            authoritative_evidence_cid=self.authoritative_evidence_cid,
        )

    def _dict(self, *, include_identity: bool) -> dict[str, Any]:
        # During identity calculation the self-referential capsule binding is
        # represented by an empty value.  The rendered contract receives the
        # resulting CID.  All other bindings participate directly in identity.
        contract_bindings = {
            "capsule_cid": self.capsule_cid if include_identity else "",
            "plan_cid": self.plan_cid,
            "task_cid": self.task_cid,
            "theorem_cid": self.theorem_cid,
            "acceptance_policy_cid": self.acceptance_policy_cid,
            "authoritative_evidence_cid": self.authoritative_evidence_cid,
        }
        result: dict[str, Any] = {
            "schema": FORMAL_PLAN_CONTEXT_CAPSULE_SCHEMA,
            "version": FORMAL_PLAN_CONTEXT_VERSION,
            "target": self.target.value,
            "query": self.query.to_dict(),
            "limits": self.limits.to_dict(),
            "plan_cid": self.plan_cid,
            "task_cid": self.task_cid,
            "validation_cid": self.validation_cid,
            "repository_tree_cid": self.repository_tree_cid,
            "transition": self.transition.to_dict(),
            "theorem": dict(self.theorem),
            "acceptance_policy": dict(self.acceptance_policy),
            "assumptions": [dict(item) for item in self.assumptions],
            "required_preconditions": [
                dict(item) for item in self.required_preconditions
            ],
            "required_effects": [dict(item) for item in self.required_effects],
            "relevant_ast_symbols": [
                dict(item) for item in self.relevant_ast_symbols
            ],
            "trusted_evidence": [dict(item) for item in self.trusted_evidence],
            "counterexamples": [dict(item) for item in self.counterexamples],
            "allowed_paths": list(self.allowed_paths),
            "tests": list(self.tests),
            "unresolved_obligations": [
                dict(item) for item in self.unresolved_obligations
            ],
            "graph_slice": self.graph_slice.to_dict(),
            "source_excerpts": [item.to_dict() for item in self.source_excerpts],
            "usage": self.usage.to_dict(),
            "truncated": self.truncated,
            "omitted": dict(self.omitted),
            "model_contract": {
                "response_schema": FORMAL_PLAN_MODEL_RESPONSE_SCHEMA,
                "bindings": contract_bindings,
                "immutable_fields": [
                    "plan_cid",
                    "task_cid",
                    "theorem",
                    "acceptance_policy",
                    "trusted_evidence",
                ],
                "instruction": (
                    "Propose implementation work only. Echo every binding exactly; "
                    "do not restate or modify authoritative fields."
                ),
            },
        }
        if include_identity:
            result["capsule_cid"] = self.capsule_cid
        return result

    def to_dict(self) -> dict[str, Any]:
        return self._dict(include_identity=True)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_prompt(self) -> str:
        """Return only the bounded serialization; no late graph read occurs."""

        self.validate_limits()
        return self.to_json()

    def render_for_codex(self) -> str:
        return self.to_prompt()

    def render_for_leanstral(self) -> str:
        return self.to_prompt()

    def validate_limits(
        self, *, token_counter: Callable[[str], int] = estimate_context_tokens
    ) -> None:
        text = self.to_json()
        actual_bytes = len(text.encode("utf-8"))
        actual_tokens = int(token_counter(text))
        expected_source_bytes = sum(item.byte_count for item in self.source_excerpts)
        checks = (
            (self.usage.rows == self.graph_slice.row_count, "row usage is stale"),
            (
                self.usage.graph_hops == self.graph_slice.max_hops,
                "graph-hop usage is stale",
            ),
            (self.usage.bytes == actual_bytes, "byte usage is stale"),
            (self.usage.tokens == actual_tokens, "token usage is stale"),
            (
                self.usage.source_excerpts == len(self.source_excerpts),
                "source-excerpt usage is stale",
            ),
            (
                self.usage.source_bytes == expected_source_bytes,
                "source-byte usage is stale",
            ),
            (self.usage.rows <= self.limits.max_rows, "row limit exceeded"),
            (
                self.usage.graph_hops <= self.limits.max_graph_hops,
                "graph-hop limit exceeded",
            ),
            (actual_bytes <= self.limits.max_bytes, "byte limit exceeded"),
            (actual_tokens <= self.limits.max_tokens, "token limit exceeded"),
            (
                len(self.source_excerpts) <= self.limits.max_source_excerpts,
                "source-excerpt limit exceeded",
            ),
            (
                expected_source_bytes <= self.limits.max_source_bytes,
                "source-byte limit exceeded",
            ),
            (
                all(
                    item.byte_count <= self.limits.max_source_excerpt_bytes
                    for item in self.source_excerpts
                ),
                "per-source-excerpt limit exceeded",
            ),
            (
                len(self.relevant_ast_symbols) <= self.limits.max_ast_symbols,
                "AST-symbol limit exceeded",
            ),
            (
                len(self.trusted_evidence) <= self.limits.max_trusted_evidence,
                "trusted-evidence limit exceeded",
            ),
            (
                len(self.counterexamples) <= self.limits.max_counterexamples,
                "counterexample limit exceeded",
            ),
            (
                len(self.unresolved_obligations)
                <= self.limits.max_unresolved_obligations,
                "obligation limit exceeded",
            ),
            (
                len(self.allowed_paths) <= self.limits.max_allowed_paths,
                "allowed-path limit exceeded",
            ),
            (len(self.tests) <= self.limits.max_tests, "test limit exceeded"),
        )
        for passed, message in checks:
            if not passed:
                raise FormalPlanContextBudgetError(message)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> "FormalPlanContextCapsule":
        if payload.get("schema") != FORMAL_PLAN_CONTEXT_CAPSULE_SCHEMA:
            raise FormalPlanContextError("unsupported formal-plan context schema")
        required_mappings = ("query", "limits", "transition", "graph_slice", "usage")
        if not all(isinstance(payload.get(name), Mapping) for name in required_mappings):
            raise FormalPlanContextError("capsule structural records must be mappings")
        result = cls(
            target=payload.get("target", ""),
            query=FormalPlanContextQuery.from_dict(payload["query"]),
            limits=FormalPlanContextLimits.from_dict(payload["limits"]),
            plan_cid=payload.get("plan_cid", ""),
            task_cid=payload.get("task_cid", ""),
            validation_cid=payload.get("validation_cid", ""),
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            transition=FormalTaskTransition.from_dict(payload["transition"]),
            theorem=payload.get("theorem") or {},
            acceptance_policy=payload.get("acceptance_policy") or {},
            assumptions=tuple(payload.get("assumptions") or ()),
            required_preconditions=tuple(
                payload.get("required_preconditions") or ()
            ),
            required_effects=tuple(payload.get("required_effects") or ()),
            relevant_ast_symbols=tuple(payload.get("relevant_ast_symbols") or ()),
            trusted_evidence=tuple(payload.get("trusted_evidence") or ()),
            counterexamples=tuple(payload.get("counterexamples") or ()),
            allowed_paths=tuple(payload.get("allowed_paths") or ()),
            tests=tuple(payload.get("tests") or ()),
            unresolved_obligations=tuple(
                payload.get("unresolved_obligations") or ()
            ),
            graph_slice=FormalPlanGraphSlice.from_dict(payload["graph_slice"]),
            source_excerpts=tuple(
                FormalPlanSourceExcerpt.from_dict(item)
                for item in (payload.get("source_excerpts") or ())
            ),
            usage=FormalPlanContextUsage.from_dict(payload["usage"]),
            truncated=bool(payload.get("truncated", False)),
            omitted=payload.get("omitted") or {},
        )
        claimed = payload.get("capsule_cid") or payload.get("capsule_id")
        if claimed and claimed != result.capsule_cid:
            raise FormalPlanContextError("capsule identity does not match payload")
        contract = payload.get("model_contract")
        if contract is not None:
            if not isinstance(contract, Mapping):
                raise FormalPlanContextError("model_contract must be a mapping")
            bindings = contract.get("bindings")
            if not isinstance(bindings, Mapping) or dict(bindings) != result.bindings.to_dict():
                raise FormalPlanContextError("model contract bindings do not match capsule")
        result.validate_limits(token_counter=token_counter)
        return result

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> "FormalPlanContextCapsule":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FormalPlanContextError("formal-plan context JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise FormalPlanContextError("formal-plan context JSON must be an object")
        return cls.from_dict(payload, token_counter=token_counter)


def _safe_graph_node(node: Mapping[str, Any]) -> dict[str, Any]:
    attributes = node.get("attributes")
    public_attributes: dict[str, Any] = {}
    if isinstance(attributes, Mapping):
        for key in ("content_id", "source_cid", "operator", "predicate"):
            if key in attributes:
                public_attributes[key] = _public_value(attributes[key])
    return {
        "node_id": _text(node.get("node_id"), label="node_id", required=True),
        "record_id": _text(node.get("record_id"), label="record_id", required=True),
        "kind": _text(node.get("kind"), label="kind", required=True),
        "attributes": public_attributes,
    }


def _safe_graph_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": _text(edge.get("edge_id"), label="edge_id", required=True),
        "kind": _text(edge.get("kind"), label="kind", required=True),
        "source": _text(edge.get("source"), label="source", required=True),
        "target": _text(edge.get("target"), label="target", required=True),
    }


def _graph_records(
    graph: PlanGraphProjection | Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(graph, PlanGraphProjection):
        raw_nodes, raw_edges = graph.nodes, graph.edges
    elif isinstance(graph, Mapping):
        raw_nodes, raw_edges = graph.get("nodes") or (), graph.get("edges") or ()
    else:
        raise FormalPlanContextError("graph must be a PlanGraphProjection or mapping")
    return (
        sorted((_safe_graph_node(item) for item in raw_nodes), key=lambda item: item["node_id"]),
        sorted((_safe_graph_edge(item) for item in raw_edges), key=lambda item: item["edge_id"]),
    )


def query_formal_plan_graph(
    graph: PlanGraphProjection | Mapping[str, Any],
    query: FormalPlanContextQuery | str,
    *,
    limits: FormalPlanContextLimits | None = None,
    token_counter: Callable[[str], int] = estimate_context_tokens,
) -> FormalPlanGraphSlice:
    """Select an exact, bounded undirected neighbourhood from a plan graph."""

    budget = limits or FormalPlanContextLimits()
    selector = (
        query if isinstance(query, FormalPlanContextQuery) else FormalPlanContextQuery(query)
    )
    nodes, edges = _graph_records(graph)
    by_id = {item["node_id"]: item for item in nodes}
    seed_ids = {
        item["node_id"]
        for item in nodes
        if selector.task_id
        in {
            item["record_id"],
            str(item["attributes"].get("source_cid") or ""),
            str(item["attributes"].get("content_id") or ""),
        }
    }
    if not seed_ids:
        raise FormalPlanContextError(
            f"task {selector.task_id!r} is absent from the formal-plan graph"
        )
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    for edge in edges:
        if edge["source"] in adjacency and edge["target"] in adjacency:
            adjacency[edge["source"]].append(edge["target"])
            adjacency[edge["target"]].append(edge["source"])
    distance = {node_id: 0 for node_id in seed_ids}
    queue = deque(sorted(seed_ids))
    while queue:
        node_id = queue.popleft()
        if distance[node_id] >= budget.max_graph_hops:
            continue
        for neighbour in sorted(adjacency[node_id]):
            if neighbour not in distance:
                distance[neighbour] = distance[node_id] + 1
                queue.append(neighbour)
    ordered_node_ids = sorted(
        distance, key=lambda node_id: (distance[node_id], node_id)
    )
    selected_nodes: list[dict[str, Any]] = []
    for node_id in ordered_node_ids:
        if len(selected_nodes) >= budget.max_rows:
            break
        selected_nodes.append(by_id[node_id])
    selected_ids = {item["node_id"] for item in selected_nodes}
    candidate_edges = [
        item
        for item in edges
        if item["source"] in selected_ids and item["target"] in selected_ids
    ]
    selected_edges = candidate_edges[: max(0, budget.max_rows - len(selected_nodes))]
    reachable_rows = len(ordered_node_ids) + len(candidate_edges)
    result = FormalPlanGraphSlice(
        nodes=tuple(selected_nodes),
        edges=tuple(selected_edges),
        max_hops=max((distance[item["node_id"]] for item in selected_nodes), default=0),
        truncated=reachable_rows > len(selected_nodes) + len(selected_edges),
        omitted_rows=max(0, reachable_rows - len(selected_nodes) - len(selected_edges)),
    )

    # A graph query is independently safe even when used without the capsule
    # builder.  Remove farthest optional rows until its serialized projection
    # satisfies the same byte/token ceiling.
    while True:
        text = canonical_json(result.to_dict())
        if (
            len(text.encode("utf-8")) <= budget.max_bytes
            and int(token_counter(text)) <= budget.max_tokens
        ):
            return result
        if result.edges:
            result = replace(
                result,
                edges=result.edges[:-1],
                truncated=True,
                omitted_rows=result.omitted_rows + 1,
            )
            continue
        removable = [
            item for item in result.nodes if item["node_id"] not in seed_ids
        ]
        if removable:
            remove_id = removable[-1]["node_id"]
            remaining = tuple(
                item for item in result.nodes if item["node_id"] != remove_id
            )
            result = replace(
                result,
                nodes=remaining,
                max_hops=max(
                    (distance[item["node_id"]] for item in remaining), default=0
                ),
                truncated=True,
                omitted_rows=result.omitted_rows + 1,
            )
            continue
        raise FormalPlanContextBudgetError(
            "byte/token limits are too small for the mandatory graph seed"
        )


def _task_for(plan: FormalWorkPlan, task_id: str) -> PlanTask:
    matches = []
    for task in plan.tasks:
        aliases = {task.task_id, *_strings(task.metadata.get("source_cids"))}
        if task_id in aliases:
            matches.append(task)
    if len(matches) != 1:
        noun = "absent" if not matches else "ambiguous"
        raise FormalPlanContextError(f"selected task {task_id!r} is {noun}")
    return matches[0]


def _transition_for(
    plan: FormalWorkPlan, task: PlanTask, event_id: str
) -> FormalTaskTransition:
    events = [item for item in plan.events if item.task_id == task.task_id]
    if event_id:
        candidates = [item for item in events if item.event_id == event_id]
        if len(candidates) != 1:
            raise FormalPlanContextError(
                f"transition event {event_id!r} is not owned by the selected task"
            )
        event = candidates[0]
    else:
        completion = [item for item in events if item.kind.value == "completed"]
        candidates = completion or sorted(events, key=lambda item: item.logical_time)
        if not candidates:
            raise FormalPlanContextError("selected task has no formal transition event")
        event = candidates[-1]
    effects = [
        item
        for item in plan.effects
        if item.task_id == task.task_id and item.event_id == event.event_id
    ]
    fluent_initial = {
        item.fluent_id: item.initial_value for item in plan.fluents
    }
    from_state: Any = None
    to_state: Any = event.kind.value
    state_effect = next(
        (item for item in effects if item.fluent_id.endswith(":state")), None
    )
    if state_effect is not None:
        from_state = fluent_initial.get(state_effect.fluent_id)
        to_state = state_effect.value
    return FormalTaskTransition(
        task_id=task.task_id,
        task_cid=task.task_id,
        event_id=event.event_id,
        event_kind=event.kind.value,
        actor_ids=task.actor_ids,
        from_state=from_state,
        to_state=to_state,
        dependency_task_ids=task.depends_on,
        terminal_states=task.terminal_states,
    )


def _formula_records(compilation: PlanCompilationResult) -> dict[str, dict[str, Any]]:
    return {
        item.formula_id: _public_mapping(item.to_record())
        for item in compilation.formulas
    }


def _relevant_theorem(
    compilation: PlanCompilationResult,
    task: PlanTask,
    preconditions: Sequence[Mapping[str, Any]],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if override is not None:
        result = _public_mapping(override)
        if not result:
            raise FormalPlanContextError("theorem override cannot be empty")
        return result
    formulas = _formula_records(compilation)
    formula_ids = {
        str(item.get("formula_id") or "") for item in preconditions
    }
    plan = compilation.plan
    assert plan is not None
    for constraint in plan.temporal_constraints:
        if task.task_id in constraint.subject_ids or any(
            item in constraint.subject_ids for item in task.depends_on
        ):
            formula_ids.add(constraint.formula_id)
    goal = next(item for item in plan.goals if item.goal_id == task.goal_id)
    formula_ids.add(goal.satisfaction_formula_id)
    selected = [
        formulas[formula_id]
        for formula_id in sorted(formula_ids)
        if formula_id and formula_id in formulas
    ]
    return {
        "kind": "fixed_task_transition_theorem",
        "task_cid": task.task_id,
        "formula_ids": sorted(
            formula_id for formula_id in formula_ids if formula_id
        ),
        "formulas": selected,
        "claim": (
            "The selected transition may be implemented only while all listed "
            "preconditions and acceptance obligations remain unchanged."
        ),
    }


def _ast_projection(
    task: PlanTask,
    records: Sequence[Mapping[str, Any]],
    requested: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    scope_ids = set(_strings(task.metadata.get("changed_ast_scope_ids"))) | set(requested)
    result: list[dict[str, Any]] = []
    covered: set[str] = set()
    task_aliases = {task.task_id, *_strings(task.metadata.get("source_cids"))}
    for raw in records:
        symbol_id = _text(
            raw.get("symbol_cid")
            or raw.get("scope_cid")
            or raw.get("ast_cid")
            or raw.get("content_id")
            or raw.get("id"),
            label="symbol_id",
        )
        task_ref = _text(raw.get("task_cid") or raw.get("task_id"), label="task_ref")
        if symbol_id not in scope_ids and task_ref not in task_aliases:
            continue
        record = {
            "symbol_id": symbol_id,
            "qualified_symbol": _text(
                raw.get("qualified_symbol")
                or raw.get("qualified_name")
                or raw.get("symbol")
                or raw.get("symbol_name"),
                label="qualified_symbol",
            ),
            "path": _path(raw.get("path") or raw.get("file_path"))
            if raw.get("path") or raw.get("file_path")
            else "",
            "tree_cid": _text(
                raw.get("tree_cid") or raw.get("repository_tree_id"),
                label="tree_cid",
            ),
        }
        result.append(_canonical_mapping(record))
        covered.add(symbol_id)
    for symbol_id in sorted(scope_ids - covered):
        result.append(
            {
                "symbol_id": symbol_id,
                "qualified_symbol": "",
                "path": "",
                "tree_cid": "",
            }
        )
    unique = {
        (item["symbol_id"], item["qualified_symbol"], item["path"]): item
        for item in result
    }
    return tuple(unique[key] for key in sorted(unique))


def _source_projection(
    excerpts: Mapping[str, Any] | Sequence[FormalPlanSourceExcerpt],
    symbols: Sequence[Mapping[str, Any]],
    limits: FormalPlanContextLimits,
) -> tuple[FormalPlanSourceExcerpt, ...]:
    aliases: dict[str, tuple[str, str]] = {}
    for symbol in symbols:
        symbol_id = str(symbol.get("symbol_id") or "")
        qualified = str(symbol.get("qualified_symbol") or "")
        path = str(symbol.get("path") or "")
        aliases[symbol_id] = (symbol_id, path)
        if qualified:
            aliases[qualified] = (symbol_id, path)
    candidates: list[FormalPlanSourceExcerpt] = []
    if isinstance(excerpts, Mapping):
        for key in sorted(excerpts):
            if key not in aliases:
                continue
            symbol_id, default_path = aliases[key]
            value = excerpts[key]
            if isinstance(value, Mapping):
                text = str(value.get("text") or value.get("source") or "")
                path = str(value.get("path") or default_path)
            else:
                text, path = str(value), default_path
            bounded, was_truncated = _truncate_utf8(
                text, limits.max_source_excerpt_bytes
            )
            candidates.append(
                FormalPlanSourceExcerpt(
                    symbol_id=symbol_id,
                    path=path,
                    text=bounded,
                    truncated=was_truncated,
                )
            )
    else:
        for value in excerpts:
            if not isinstance(value, FormalPlanSourceExcerpt):
                raise FormalPlanContextError(
                    "source excerpt sequences must contain FormalPlanSourceExcerpt"
                )
            if value.symbol_id not in aliases:
                continue
            bounded, was_truncated = _truncate_utf8(
                value.text, limits.max_source_excerpt_bytes
            )
            candidates.append(
                replace(value, text=bounded, truncated=value.truncated or was_truncated)
            )
    result: list[FormalPlanSourceExcerpt] = []
    used = 0
    for item in candidates:
        if len(result) >= limits.max_source_excerpts:
            break
        if used + item.byte_count > limits.max_source_bytes:
            remaining = limits.max_source_bytes - used
            if remaining <= 0:
                break
            bounded, was_truncated = _truncate_utf8(item.text, remaining)
            item = replace(item, text=bounded, truncated=True)
        result.append(item)
        used += item.byte_count
    return tuple(result)


def _validation_evidence(validation: PlanValidationResult) -> list[dict[str, Any]]:
    result = [
        {
            "evidence_id": validation.validation_id,
            "kind": "formal_plan_validation",
            "status": validation.status.value,
            "outcome": validation.outcome.value,
            "consistency_level": validation.consistency_level.value,
            "plan_cid": validation.plan_id,
            "bounds_cid": validation.bounds.bounds_id,
            "checks_performed": [item.value for item in validation.checks_performed],
            "authoritative_for": "plan_consistency_only",
        }
    ]
    for item in validation.evidence:
        if item.accepted:
            record = item.to_dict()
            record["authoritative_for"] = (
                "exact_obligation"
                if item.reconstructed
                else "plan_check_only"
            )
            result.append(_public_mapping(record))
    return result


def _acceptance_policy(
    plan: FormalWorkPlan,
    task: PlanTask,
    tests: Sequence[str],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if override is not None:
        result = _public_mapping(override)
        if not result:
            raise FormalPlanContextError("acceptance policy override cannot be empty")
        return result
    requirement_ids = set(task.evidence_requirement_ids)
    goal = next(item for item in plan.goals if item.goal_id == task.goal_id)
    requirement_ids.update(goal.evidence_requirement_ids)
    requirements = [
        _public_mapping(item.to_record())
        for item in plan.evidence_requirements
        if item.requirement_id in requirement_ids
    ]
    return {
        "task_cid": task.task_id,
        "goal_cid": task.goal_id,
        "policy_cids": list(_strings(task.metadata.get("policy_ids"))),
        "requirements": requirements,
        "required_tests": list(_strings(tests)),
        "rule": (
            "Only supervisor validation may satisfy or waive a requirement; "
            "model output is never acceptance evidence."
        ),
    }


def _obligations(
    plan: FormalWorkPlan,
    task: PlanTask,
    validation: PlanValidationResult,
    explicit: Sequence[Mapping[str, Any] | str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    requirement_ids = set(task.evidence_requirement_ids)
    for item in plan.evidence_requirements:
        if item.requirement_id in requirement_ids:
            result.append(
                {
                    "obligation_id": item.requirement_id,
                    "kind": item.kind.value,
                    "criterion": str(item.metadata.get("criterion") or ""),
                    "minimum_code_assurance": item.minimum_code_assurance.value,
                    "fallback_tests": list(item.fallback_check_ids),
                    "status": "unresolved_until_post_implementation_validation",
                }
            )
    for finding in validation.findings:
        result.append(
            {
                "obligation_id": finding.finding_id,
                "kind": "validation_finding",
                "criterion": finding.message,
                "status": finding.disposition.value,
                "subject_ids": list(finding.subject_ids),
            }
        )
    for abstraction_id in plan.abstraction_ids:
        result.append(
            {
                "obligation_id": abstraction_id,
                "kind": "compiler_abstraction",
                "criterion": "Preserve and resolve the recorded compiler abstraction.",
                "status": "unresolved",
            }
        )
    for item in explicit:
        if isinstance(item, Mapping):
            record = _public_mapping(item)
            record.setdefault(
                "obligation_id", content_identity(record)
            )
        else:
            statement = _text(item, label="unresolved obligation", required=True)
            record = {
                "obligation_id": content_identity({"statement": statement}),
                "kind": "explicit",
                "criterion": statement,
                "status": "unresolved",
            }
        result.append(record)
    unique = {
        str(item.get("obligation_id") or content_identity(item)): item for item in result
    }
    return [unique[key] for key in sorted(unique)]


def _counterexamples(validation: PlanValidationResult) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if validation.countermodel is not None:
        result.append(
            {
                "counterexample_id": validation.countermodel.countermodel_id,
                "kind": "bounded_countermodel",
                "value": validation.countermodel.to_dict(),
            }
        )
    for finding in validation.findings:
        result.append(
            {
                "counterexample_id": finding.finding_id,
                "kind": finding.disposition.value,
                "check": finding.check.value,
                "message": finding.message,
                "subject_ids": list(finding.subject_ids),
                "logical_time": finding.logical_time,
            }
        )
    return result


def _with_usage(
    capsule: FormalPlanContextCapsule,
    *,
    token_counter: Callable[[str], int],
) -> FormalPlanContextCapsule:
    result = capsule
    for _ in range(8):
        text = result.to_json()
        usage = FormalPlanContextUsage(
            rows=result.graph_slice.row_count,
            graph_hops=result.graph_slice.max_hops,
            bytes=len(text.encode("utf-8")),
            tokens=int(token_counter(text)),
            source_excerpts=len(result.source_excerpts),
            source_bytes=sum(item.byte_count for item in result.source_excerpts),
        )
        candidate = replace(result, usage=usage)
        if candidate.usage == result.usage:
            return candidate
        result = candidate
    return result


class FormalPlanContextBuilder:
    """Build one formal-plan capsule from compiler and validator artifacts."""

    def __init__(
        self,
        compilation: PlanCompilationResult,
        validation: PlanValidationResult | Mapping[str, Any],
        *,
        token_counter: Callable[[str], int] = estimate_context_tokens,
    ) -> None:
        if not isinstance(compilation, PlanCompilationResult):
            raise FormalPlanContextError(
                "compilation must be a PlanCompilationResult"
            )
        if compilation.status is not CompilationStatus.COMPILED or compilation.plan is None:
            raise FormalPlanContextError(
                "a successfully compiled formal plan is required"
            )
        if isinstance(validation, Mapping):
            validation = PlanValidationResult.from_dict(validation)
        if not isinstance(validation, PlanValidationResult):
            raise FormalPlanContextError(
                "validation must be a PlanValidationResult"
            )
        if validation.plan_id != compilation.plan_id:
            raise FormalPlanContextError(
                "formal-plan validation does not bind the compiled plan"
            )
        self.compilation = compilation
        self.validation = validation
        self.token_counter = token_counter

    def build(
        self,
        query: FormalPlanContextQuery | None = None,
        *,
        task_id: str = "",
        transition_event_id: str = "",
        target: FormalPlanContextTarget | str = FormalPlanContextTarget.CODEX,
        limits: FormalPlanContextLimits | None = None,
        ast_records: Sequence[Mapping[str, Any]] = (),
        trusted_evidence: Sequence[Mapping[str, Any]] = (),
        source_excerpts: Mapping[str, Any]
        | Sequence[FormalPlanSourceExcerpt] = (),
        allowed_paths: Sequence[str] = (),
        tests: Sequence[str] = (),
        unresolved_obligations: Sequence[Mapping[str, Any] | str] = (),
        theorem: Mapping[str, Any] | None = None,
        acceptance_policy: Mapping[str, Any] | None = None,
    ) -> FormalPlanContextCapsule:
        budget = limits or FormalPlanContextLimits()
        if query is None:
            query = FormalPlanContextQuery(
                task_id=_text(task_id, label="task_id", required=True),
                transition_event_id=transition_event_id,
            )
        elif task_id and task_id != query.task_id:
            raise FormalPlanContextError("task_id conflicts with the context query")
        plan = self.compilation.plan
        assert plan is not None
        task = _task_for(plan, query.task_id)
        transition = _transition_for(
            plan, task, query.transition_event_id or transition_event_id
        )
        preconditions = [
            _public_mapping(item.to_record())
            for item in plan.preconditions
            if item.precondition_id in task.precondition_ids
        ]
        effects = [
            _public_mapping(item.to_record())
            for item in plan.effects
            if item.effect_id in task.effect_ids
        ]
        symbols_all = list(
            _ast_projection(task, ast_records, query.ast_symbol_ids)
        )
        symbols = symbols_all[: budget.max_ast_symbols]

        derived_tests: set[str] = set(_strings(tests))
        for requirement in plan.evidence_requirements:
            if requirement.requirement_id in task.evidence_requirement_ids:
                derived_tests.update(requirement.fallback_check_ids)
        selected_tests = tuple(sorted(derived_tests))[: budget.max_tests]
        selected_paths = {
            _path(item) for item in allowed_paths if str(item or "").strip()
        }
        selected_paths.update(
            item["path"] for item in symbols if item.get("path")
        )
        paths_all = tuple(sorted(selected_paths))
        paths = paths_all[: budget.max_allowed_paths]

        evidence_all = _validation_evidence(self.validation)
        evidence_all.extend(_public_mapping(item) for item in trusted_evidence)
        if query.evidence_ids:
            requested_evidence = set(query.evidence_ids)
            selected_evidence: list[dict[str, Any]] = []
            found_evidence: set[str] = set()
            for index, item in enumerate(evidence_all):
                evidence_id = str(
                    item.get("evidence_id")
                    or item.get("receipt_id")
                    or content_identity(item)
                )
                # The formal validation summary is mandatory.  Other evidence
                # is included only when its exact identity was requested.
                if index == 0 or evidence_id in requested_evidence:
                    selected_evidence.append(item)
                if evidence_id in requested_evidence:
                    found_evidence.add(evidence_id)
            missing = sorted(requested_evidence - found_evidence)
            if missing:
                raise FormalPlanContextError(
                    "requested trusted evidence is absent: " + ", ".join(missing)
                )
            evidence_all = selected_evidence
        evidence_unique = {
            str(
                item.get("evidence_id")
                or item.get("receipt_id")
                or content_identity(item)
            ): item
            for item in evidence_all
        }
        evidence = [
            evidence_unique[key] for key in sorted(evidence_unique)
        ][: budget.max_trusted_evidence]
        counterexamples_all = _counterexamples(self.validation)
        counterexamples = counterexamples_all[: budget.max_counterexamples]
        obligations_all = _obligations(
            plan, task, self.validation, unresolved_obligations
        )
        obligations = obligations_all[: budget.max_unresolved_obligations]
        excerpts = _source_projection(source_excerpts, symbols, budget)
        graph_slice = query_formal_plan_graph(
            self.compilation.graph_projection,
            FormalPlanContextQuery(
                task_id=task.task_id,
                transition_event_id=transition.event_id,
                ast_symbol_ids=query.ast_symbol_ids,
                evidence_ids=query.evidence_ids,
            ),
            limits=budget,
            token_counter=self.token_counter,
        )
        assumptions = tuple(
            _public_mapping(item.to_dict()) for item in self.validation.assumptions
        )
        symbol_aliases = {
            alias
            for item in symbols
            for alias in (
                str(item.get("symbol_id") or ""),
                str(item.get("qualified_symbol") or ""),
            )
            if alias
        }
        if isinstance(source_excerpts, Mapping):
            relevant_source_count = sum(
                1 for key in source_excerpts if str(key) in symbol_aliases
            )
        else:
            relevant_source_count = sum(
                1
                for item in source_excerpts
                if isinstance(item, FormalPlanSourceExcerpt)
                and item.symbol_id in symbol_aliases
            )
        omitted = {
            "ast_symbols": max(0, len(symbols_all) - len(symbols)),
            "trusted_evidence": max(0, len(evidence_all) - len(evidence)),
            "counterexamples": max(
                0, len(counterexamples_all) - len(counterexamples)
            ),
            "unresolved_obligations": max(
                0, len(obligations_all) - len(obligations)
            ),
            "allowed_paths": max(0, len(paths_all) - len(paths)),
            "tests": max(0, len(derived_tests) - len(selected_tests)),
            "source_excerpts": max(
                0,
                relevant_source_count - len(excerpts),
            ),
            "graph_rows": graph_slice.omitted_rows,
        }
        capsule = FormalPlanContextCapsule(
            target=(
                target
                if isinstance(target, FormalPlanContextTarget)
                else FormalPlanContextTarget(str(target))
            ),
            query=FormalPlanContextQuery(
                task_id=task.task_id,
                transition_event_id=transition.event_id,
                ast_symbol_ids=query.ast_symbol_ids,
                evidence_ids=query.evidence_ids,
            ),
            limits=budget,
            plan_cid=plan.plan_id,
            task_cid=task.task_id,
            validation_cid=self.validation.validation_id,
            repository_tree_cid=plan.repository_tree_id,
            transition=transition,
            theorem=_relevant_theorem(
                self.compilation, task, preconditions, theorem
            ),
            acceptance_policy=_acceptance_policy(
                plan, task, selected_tests, acceptance_policy
            ),
            assumptions=assumptions,
            required_preconditions=tuple(preconditions),
            required_effects=tuple(effects),
            relevant_ast_symbols=tuple(symbols),
            trusted_evidence=tuple(evidence),
            counterexamples=tuple(counterexamples),
            allowed_paths=paths,
            tests=selected_tests,
            unresolved_obligations=tuple(obligations),
            graph_slice=graph_slice,
            source_excerpts=excerpts,
            truncated=graph_slice.truncated or any(omitted.values()),
            omitted=omitted,
        )

        def measured(value: FormalPlanContextCapsule) -> FormalPlanContextCapsule:
            return _with_usage(value, token_counter=self.token_counter)

        capsule = measured(capsule)
        # Optional detail is removed in a deterministic order.  Supervisor
        # semantics, theorem, policy, transition, preconditions/effects, tests,
        # and unresolved obligations are never silently dropped.
        while (
            capsule.usage.bytes > budget.max_bytes
            or capsule.usage.tokens > budget.max_tokens
        ):
            if capsule.source_excerpts:
                omitted["source_excerpts"] += 1
                capsule = replace(
                    capsule,
                    source_excerpts=capsule.source_excerpts[:-1],
                    truncated=True,
                    omitted=omitted,
                )
            elif capsule.graph_slice.edges:
                omitted["graph_rows"] += 1
                capsule = replace(
                    capsule,
                    graph_slice=replace(
                        capsule.graph_slice,
                        edges=capsule.graph_slice.edges[:-1],
                        truncated=True,
                        omitted_rows=capsule.graph_slice.omitted_rows + 1,
                    ),
                    truncated=True,
                    omitted=omitted,
                )
            elif len(capsule.graph_slice.nodes) > 1:
                omitted["graph_rows"] += 1
                capsule = replace(
                    capsule,
                    graph_slice=replace(
                        capsule.graph_slice,
                        nodes=capsule.graph_slice.nodes[:-1],
                        truncated=True,
                        omitted_rows=capsule.graph_slice.omitted_rows + 1,
                        max_hops=(
                            capsule.graph_slice.max_hops
                            if len(capsule.graph_slice.nodes) > 2
                            else 0
                        ),
                    ),
                    truncated=True,
                    omitted=omitted,
                )
            elif len(capsule.trusted_evidence) > 1:
                omitted["trusted_evidence"] += 1
                capsule = replace(
                    capsule,
                    trusted_evidence=capsule.trusted_evidence[:-1],
                    truncated=True,
                    omitted=omitted,
                )
            elif capsule.counterexamples:
                omitted["counterexamples"] += 1
                capsule = replace(
                    capsule,
                    counterexamples=capsule.counterexamples[:-1],
                    truncated=True,
                    omitted=omitted,
                )
            else:
                raise FormalPlanContextBudgetError(
                    "byte/token limits are too small for the mandatory formal-plan envelope"
                )
            capsule = measured(capsule)
        capsule.validate_limits(token_counter=self.token_counter)
        return capsule


@dataclass(frozen=True)
class FormalPlanModelResponse:
    """Validated, non-authoritative implementation proposal from a model."""

    bindings: FormalPlanResponseBinding
    proposed_steps: tuple[Mapping[str, Any], ...] = ()
    changed_paths: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    unresolved_obligations: tuple[str, ...] = ()
    proof_draft: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, FormalPlanResponseBinding):
            raise FormalPlanResponseError("response bindings are required")
        object.__setattr__(
            self, "proposed_steps", _canonical_records(self.proposed_steps)
        )
        object.__setattr__(
            self, "changed_paths", tuple(sorted({_path(item) for item in self.changed_paths}))
        )
        object.__setattr__(self, "tests", _strings(self.tests))
        object.__setattr__(
            self, "unresolved_obligations", _strings(self.unresolved_obligations)
        )
        object.__setattr__(self, "proof_draft", str(self.proof_draft or ""))
        object.__setattr__(self, "notes", str(self.notes or ""))

    @property
    def response_cid(self) -> str:
        return content_identity(self.to_dict())

    @property
    def response_id(self) -> str:
        return self.response_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_MODEL_RESPONSE_SCHEMA,
            "bindings": self.bindings.to_dict(),
            "proposal": {
                "steps": [dict(item) for item in self.proposed_steps],
                "changed_paths": list(self.changed_paths),
                "tests": list(self.tests),
                "unresolved_obligations": list(self.unresolved_obligations),
                "proof_draft": self.proof_draft,
                "notes": self.notes,
            },
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        capsule: FormalPlanContextCapsule,
        payload: Mapping[str, Any],
    ) -> "FormalPlanModelResponse":
        """Deserialize through the authoritative capsule validation gate."""

        return validate_formal_plan_model_response(capsule, payload)

    @classmethod
    def from_json(
        cls,
        capsule: FormalPlanContextCapsule,
        text: str,
    ) -> "FormalPlanModelResponse":
        return validate_formal_plan_model_response(capsule, text)


_PROTECTED_RESPONSE_KEYS: Final = frozenset(
    {
        "plan_cid",
        "task_cid",
        "capsule_cid",
        "theorem",
        "theorem_cid",
        "acceptance_policy",
        "acceptance_policy_cid",
        "authoritative_evidence",
        "authoritative_evidence_cid",
        "trusted_evidence",
        "validation_cid",
    }
)


def _find_protected_key(value: Any, *, in_bindings: bool = False) -> str:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            text = str(key)
            if text in _PROTECTED_RESPONSE_KEYS and not in_bindings:
                return text
            found = _find_protected_key(
                nested, in_bindings=in_bindings or text == "bindings"
            )
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            found = _find_protected_key(nested, in_bindings=in_bindings)
            if found:
                return found
    return ""


def validate_formal_plan_model_response(
    capsule: FormalPlanContextCapsule,
    response: str | Mapping[str, Any],
) -> FormalPlanModelResponse:
    """Parse a proposal and enforce exact capsule/plan/task/trust bindings."""

    capsule.validate_limits()
    if isinstance(response, str):
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise FormalPlanResponseError("model response is not valid JSON") from exc
    else:
        payload = response
    if not isinstance(payload, Mapping):
        raise FormalPlanResponseError("model response must be a JSON object")
    if payload.get("schema") != FORMAL_PLAN_MODEL_RESPONSE_SCHEMA:
        raise FormalPlanResponseError("model response schema is missing or unsupported")
    bindings_value = payload.get("bindings")
    proposal = payload.get("proposal")
    if not isinstance(bindings_value, Mapping) or not isinstance(proposal, Mapping):
        raise FormalPlanResponseError("model response requires bindings and proposal")
    expected_binding_keys = set(capsule.bindings.to_dict())
    supplied_binding_keys = {str(key) for key in bindings_value}
    if supplied_binding_keys != expected_binding_keys:
        unexpected = sorted(supplied_binding_keys - expected_binding_keys)
        missing = sorted(expected_binding_keys - supplied_binding_keys)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        raise FormalPlanResponseError(
            "model response binding fields are not exact: " + "; ".join(detail)
        )
    found = _find_protected_key(proposal)
    if found:
        raise FormalPlanResponseError(
            f"model proposal attempts to alter protected field {found!r}"
        )
    bindings = FormalPlanResponseBinding.from_dict(bindings_value)
    expected = capsule.bindings
    if bindings != expected:
        mismatches = [
            name
            for name in expected.to_dict()
            if getattr(bindings, name) != getattr(expected, name)
        ]
        raise FormalPlanResponseError(
            "model response binding mismatch: " + ", ".join(mismatches)
        )
    steps_value = proposal.get("steps") or ()
    if not isinstance(steps_value, Sequence) or isinstance(
        steps_value, (str, bytes, bytearray)
    ):
        raise FormalPlanResponseError("proposal steps must be a sequence")
    steps = tuple(
        _public_mapping(item) if isinstance(item, Mapping) else {"instruction": str(item)}
        for item in steps_value
    )
    result = FormalPlanModelResponse(
        bindings=bindings,
        proposed_steps=steps,
        changed_paths=tuple(proposal.get("changed_paths") or ()),
        tests=tuple(proposal.get("tests") or ()),
        unresolved_obligations=tuple(
            proposal.get("unresolved_obligations") or ()
        ),
        proof_draft=str(proposal.get("proof_draft") or ""),
        notes=str(proposal.get("notes") or ""),
    )
    for changed_path in result.changed_paths:
        if not any(
            changed_path == allowed
            or changed_path.startswith(allowed.rstrip("/") + "/")
            for allowed in capsule.allowed_paths
        ):
            raise FormalPlanResponseError(
                f"model response changed path is outside the capsule: {changed_path}"
            )
    return result


def invoke_formal_plan_model(
    capsule: FormalPlanContextCapsule,
    generate: Callable[[str], str | Mapping[str, Any]],
) -> FormalPlanModelResponse:
    """Validate budgets, invoke one model, then validate its bound response."""

    capsule.validate_limits()
    raw = generate(capsule.to_prompt())
    return validate_formal_plan_model_response(capsule, raw)


@dataclass(frozen=True)
class ImplementationOutcome:
    status: ImplementationOutcomeStatus = ImplementationOutcomeStatus.NOT_RUN
    accepted: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    changed_paths: tuple[str, ...] = ()
    obligations_resolved: int = 0
    obligations_remaining: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            self.status
            if isinstance(self.status, ImplementationOutcomeStatus)
            else ImplementationOutcomeStatus(str(self.status)),
        )
        if not isinstance(self.accepted, bool):
            raise FormalPlanContextError("outcome accepted must be boolean")
        for name in (
            "tests_passed",
            "tests_failed",
            "obligations_resolved",
            "obligations_remaining",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), label=name, allow_zero=True),
            )
        object.__setattr__(
            self, "changed_paths", tuple(sorted({_path(item) for item in self.changed_paths}))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IMPLEMENTATION_OUTCOME_SCHEMA,
            "status": self.status.value,
            "accepted": self.accepted,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "changed_paths": list(self.changed_paths),
            "obligations_resolved": self.obligations_resolved,
            "obligations_remaining": self.obligations_remaining,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationOutcome":
        schema = payload.get("schema", IMPLEMENTATION_OUTCOME_SCHEMA)
        if schema != IMPLEMENTATION_OUTCOME_SCHEMA:
            raise FormalPlanContextError(f"unsupported outcome schema: {schema}")
        accepted = payload.get("accepted", False)
        if not isinstance(accepted, bool):
            raise FormalPlanContextError("outcome accepted must be boolean")
        return cls(
            status=payload.get("status", ImplementationOutcomeStatus.NOT_RUN),
            accepted=accepted,
            tests_passed=int(payload.get("tests_passed") or 0),
            tests_failed=int(payload.get("tests_failed") or 0),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            obligations_resolved=int(payload.get("obligations_resolved") or 0),
            obligations_remaining=int(payload.get("obligations_remaining") or 0),
        )


@dataclass(frozen=True)
class FormalPlanContextMeasurement:
    capsule_cid: str
    capsule_bytes: int
    capsule_tokens: int
    unbounded_prompt_bytes: int
    unbounded_prompt_tokens: int
    bytes_saved: int
    tokens_saved: int
    byte_ratio_millionths: int
    token_ratio_millionths: int
    capsule_outcome: ImplementationOutcome
    unbounded_prompt_outcome: ImplementationOutcome

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capsule_cid",
            _text(self.capsule_cid, label="capsule_cid", required=True),
        )
        for name in (
            "capsule_bytes",
            "capsule_tokens",
            "unbounded_prompt_bytes",
            "unbounded_prompt_tokens",
            "byte_ratio_millionths",
            "token_ratio_millionths",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), label=name, allow_zero=True),
            )
        for name in ("bytes_saved", "tokens_saved"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise FormalPlanContextError(f"{name} must be an integer")
        if not isinstance(self.capsule_outcome, ImplementationOutcome) or not isinstance(
            self.unbounded_prompt_outcome, ImplementationOutcome
        ):
            raise FormalPlanContextError("measurement outcomes are required")

    @property
    def outcome_comparison(self) -> dict[str, int | bool]:
        return {
            "capsule_accepted": self.capsule_outcome.accepted,
            "unbounded_prompt_accepted": self.unbounded_prompt_outcome.accepted,
            "accepted_delta": int(self.capsule_outcome.accepted)
            - int(self.unbounded_prompt_outcome.accepted),
            "tests_passed_delta": self.capsule_outcome.tests_passed
            - self.unbounded_prompt_outcome.tests_passed,
            "tests_failed_delta": self.capsule_outcome.tests_failed
            - self.unbounded_prompt_outcome.tests_failed,
            "obligations_resolved_delta": self.capsule_outcome.obligations_resolved
            - self.unbounded_prompt_outcome.obligations_resolved,
            "obligations_remaining_delta": self.capsule_outcome.obligations_remaining
            - self.unbounded_prompt_outcome.obligations_remaining,
        }

    @property
    def measurement_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_PLAN_CONTEXT_MEASUREMENT_SCHEMA,
            "capsule_cid": self.capsule_cid,
            "capsule_bytes": self.capsule_bytes,
            "capsule_tokens": self.capsule_tokens,
            "unbounded_prompt_bytes": self.unbounded_prompt_bytes,
            "unbounded_prompt_tokens": self.unbounded_prompt_tokens,
            "bytes_saved": self.bytes_saved,
            "tokens_saved": self.tokens_saved,
            "byte_ratio_millionths": self.byte_ratio_millionths,
            "token_ratio_millionths": self.token_ratio_millionths,
            "capsule_outcome": self.capsule_outcome.to_dict(),
            "unbounded_prompt_outcome": self.unbounded_prompt_outcome.to_dict(),
            "outcome_comparison": self.outcome_comparison,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "FormalPlanContextMeasurement":
        schema = payload.get("schema", FORMAL_PLAN_CONTEXT_MEASUREMENT_SCHEMA)
        if schema != FORMAL_PLAN_CONTEXT_MEASUREMENT_SCHEMA:
            raise FormalPlanContextError(
                f"unsupported formal-plan measurement schema: {schema}"
            )
        capsule_outcome = payload.get("capsule_outcome")
        baseline_outcome = payload.get("unbounded_prompt_outcome")
        if not isinstance(capsule_outcome, Mapping) or not isinstance(
            baseline_outcome, Mapping
        ):
            raise FormalPlanContextError(
                "formal-plan measurement outcomes must be mappings"
            )
        result = cls(
            capsule_cid=payload.get("capsule_cid", ""),
            capsule_bytes=int(payload.get("capsule_bytes") or 0),
            capsule_tokens=int(payload.get("capsule_tokens") or 0),
            unbounded_prompt_bytes=int(
                payload.get("unbounded_prompt_bytes") or 0
            ),
            unbounded_prompt_tokens=int(
                payload.get("unbounded_prompt_tokens") or 0
            ),
            bytes_saved=int(payload.get("bytes_saved") or 0),
            tokens_saved=int(payload.get("tokens_saved") or 0),
            byte_ratio_millionths=int(
                payload.get("byte_ratio_millionths") or 0
            ),
            token_ratio_millionths=int(
                payload.get("token_ratio_millionths") or 0
            ),
            capsule_outcome=ImplementationOutcome.from_dict(capsule_outcome),
            unbounded_prompt_outcome=ImplementationOutcome.from_dict(
                baseline_outcome
            ),
        )
        comparison = payload.get("outcome_comparison")
        if comparison is not None and comparison != result.outcome_comparison:
            raise FormalPlanContextError(
                "formal-plan measurement outcome comparison does not match"
            )
        claimed = payload.get("measurement_cid") or payload.get("measurement_id")
        if claimed and claimed != result.measurement_cid:
            raise FormalPlanContextError(
                "formal-plan measurement identity does not match payload"
            )
        return result

    @classmethod
    def from_json(cls, text: str) -> "FormalPlanContextMeasurement":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FormalPlanContextError(
                "formal-plan measurement JSON is malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise FormalPlanContextError(
                "formal-plan measurement JSON must be an object"
            )
        return cls.from_dict(payload)


def measure_formal_plan_context(
    capsule: FormalPlanContextCapsule,
    unbounded_planning_prompt: str,
    *,
    capsule_outcome: ImplementationOutcome | Mapping[str, Any] | None = None,
    unbounded_prompt_outcome: ImplementationOutcome
    | Mapping[str, Any]
    | None = None,
    token_counter: Callable[[str], int] = estimate_context_tokens,
) -> FormalPlanContextMeasurement:
    """Compare bounded capsule size and implementation outcomes to a baseline."""

    capsule.validate_limits(token_counter=token_counter)
    baseline = str(unbounded_planning_prompt)
    if not baseline:
        raise FormalPlanContextError("unbounded planning prompt baseline is required")

    def outcome(
        value: ImplementationOutcome | Mapping[str, Any] | None,
    ) -> ImplementationOutcome:
        if value is None:
            return ImplementationOutcome()
        return (
            value
            if isinstance(value, ImplementationOutcome)
            else ImplementationOutcome.from_dict(value)
        )

    baseline_bytes = len(baseline.encode("utf-8"))
    baseline_tokens = int(token_counter(baseline))
    return FormalPlanContextMeasurement(
        capsule_cid=capsule.capsule_cid,
        capsule_bytes=capsule.usage.bytes,
        capsule_tokens=capsule.usage.tokens,
        unbounded_prompt_bytes=baseline_bytes,
        unbounded_prompt_tokens=baseline_tokens,
        bytes_saved=baseline_bytes - capsule.usage.bytes,
        tokens_saved=baseline_tokens - capsule.usage.tokens,
        byte_ratio_millionths=(
            capsule.usage.bytes * 1_000_000 // baseline_bytes
            if baseline_bytes
            else 0
        ),
        token_ratio_millionths=(
            capsule.usage.tokens * 1_000_000 // baseline_tokens
            if baseline_tokens
            else 0
        ),
        capsule_outcome=outcome(capsule_outcome),
        unbounded_prompt_outcome=outcome(unbounded_prompt_outcome),
    )


def build_formal_plan_context_capsule(
    compilation: PlanCompilationResult,
    validation: PlanValidationResult | Mapping[str, Any],
    query: FormalPlanContextQuery | None = None,
    **kwargs: Any,
) -> FormalPlanContextCapsule:
    """Convenience entry point for a single proof-carrying capsule."""

    token_counter = kwargs.pop("token_counter", estimate_context_tokens)
    return FormalPlanContextBuilder(
        compilation, validation, token_counter=token_counter
    ).build(query, **kwargs)


generate_formal_plan_context_capsule = build_formal_plan_context_capsule
build_formal_plan_capsule = build_formal_plan_context_capsule
validate_model_response = validate_formal_plan_model_response
compare_formal_plan_context = measure_formal_plan_context
FormalPlanCapsule = FormalPlanContextCapsule
FormalPlanCapsuleLimits = FormalPlanContextLimits
FormalPlanCapsuleQuery = FormalPlanContextQuery
ModelResponseBinding = FormalPlanResponseBinding


__all__ = [
    "DEFAULT_MAX_FORMAL_PLAN_ALLOWED_PATHS",
    "DEFAULT_MAX_FORMAL_PLAN_AST_SYMBOLS",
    "DEFAULT_MAX_FORMAL_PLAN_CONTEXT_BYTES",
    "DEFAULT_MAX_FORMAL_PLAN_CONTEXT_TOKENS",
    "DEFAULT_MAX_FORMAL_PLAN_COUNTEREXAMPLES",
    "DEFAULT_MAX_FORMAL_PLAN_EVIDENCE",
    "DEFAULT_MAX_FORMAL_PLAN_GRAPH_HOPS",
    "DEFAULT_MAX_FORMAL_PLAN_OBLIGATIONS",
    "DEFAULT_MAX_FORMAL_PLAN_ROWS",
    "DEFAULT_MAX_FORMAL_PLAN_SOURCE_BYTES",
    "DEFAULT_MAX_FORMAL_PLAN_SOURCE_EXCERPT_BYTES",
    "DEFAULT_MAX_FORMAL_PLAN_SOURCE_EXCERPTS",
    "DEFAULT_MAX_FORMAL_PLAN_TESTS",
    "FORMAL_PLAN_CONTEXT_CAPSULE_SCHEMA",
    "FORMAL_PLAN_CONTEXT_LIMITS_SCHEMA",
    "FORMAL_PLAN_CONTEXT_MEASUREMENT_SCHEMA",
    "FORMAL_PLAN_CONTEXT_QUERY_SCHEMA",
    "FORMAL_PLAN_CONTEXT_VERSION",
    "FORMAL_PLAN_GRAPH_SLICE_SCHEMA",
    "FORMAL_PLAN_MODEL_RESPONSE_SCHEMA",
    "FormalPlanCapsule",
    "FormalPlanCapsuleLimits",
    "FormalPlanCapsuleQuery",
    "FormalPlanContextBudgetError",
    "FormalPlanContextBuilder",
    "FormalPlanContextCapsule",
    "FormalPlanContextError",
    "FormalPlanContextLimits",
    "FormalPlanContextMeasurement",
    "FormalPlanContextQuery",
    "FormalPlanContextTarget",
    "FormalPlanContextUsage",
    "FormalPlanGraphSlice",
    "FormalPlanModelResponse",
    "FormalPlanResponseBinding",
    "FormalPlanResponseError",
    "FormalPlanSourceExcerpt",
    "FormalTaskTransition",
    "ImplementationOutcome",
    "ImplementationOutcomeStatus",
    "ModelResponseBinding",
    "build_formal_plan_capsule",
    "build_formal_plan_context_capsule",
    "compare_formal_plan_context",
    "generate_formal_plan_context_capsule",
    "invoke_formal_plan_model",
    "measure_formal_plan_context",
    "query_formal_plan_graph",
    "validate_formal_plan_model_response",
    "validate_model_response",
]
