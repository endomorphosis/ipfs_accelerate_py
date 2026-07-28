"""Canonical, prompt-safe counterexamples for formal supervisor work.

Provider output is deliberately not a persistence or model-context contract.
This module is the boundary which turns solver models, contradiction reports,
model-checker traces, protocol attacks, hypertraces, kernel failures, and
runtime temporal violations into one small content-addressed representation.

Every public entry point follows the same order:

1. discard private and non-context channels;
2. canonicalize and minimize the typed witness;
3. enforce structural and byte limits;
4. compute semantic identity and deduplicate; and only then
5. persist or assemble a model-facing capsule.

In particular, secret values are never hashed into public identities.  This
avoids turning a counterexample store into an equality oracle for credentials
or hidden witnesses.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


FORMAL_COUNTEREXAMPLE_VERSION: Final = 1
FORMAL_COUNTEREXAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-counterexample@1"
)
COUNTEREXAMPLE_GRAPH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-knowledge-graph@1"
)
COUNTEREXAMPLE_CAPSULE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-context-capsule@1"
)
COUNTEREXAMPLE_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/counterexample-store-record@1"
)

DEFAULT_MAX_COUNTEREXAMPLE_BYTES: Final = 32 * 1024
DEFAULT_MAX_PAYLOAD_BYTES: Final = 12 * 1024
DEFAULT_MAX_CAPSULE_BYTES: Final = 16 * 1024
DEFAULT_MAX_TEXT_CHARS: Final = 512
DEFAULT_MAX_COLLECTION_ITEMS: Final = 64
DEFAULT_MAX_NESTING_DEPTH: Final = 6
DEFAULT_MAX_TRACE_STEPS: Final = 16
DEFAULT_MAX_CAPSULE_COUNTEREXAMPLES: Final = 8
DEFAULT_MAX_GRAPH_NODES: Final = 96
DEFAULT_MAX_GRAPH_EDGES: Final = 128
ABSOLUTE_MAX_COUNTEREXAMPLE_BYTES: Final = 64 * 1024
REDACTED: Final = "<redacted>"
OMITTED: Final = "<omitted>"


class CounterexampleValidationError(ContractValidationError):
    """Raised when a counterexample contract is malformed or unsafe."""


class CounterexampleBudgetError(CounterexampleValidationError):
    """Raised when even the minimal safe representation cannot fit a limit."""


class CounterexampleKind(str, Enum):
    """Formal and runtime failure families represented by the canonical IR."""

    SMT_MODEL = "smt_model"
    SMT_UNSAT_CORE = "smt_unsat_core"
    UNSAT_CORE = "smt_unsat_core"  # compatibility spelling
    DCEC_CONTRADICTION = "dcec_contradiction"
    TDFOL_CONTRADICTION = "tdfol_contradiction"
    TLA_TRACE = "tla_trace"
    PROTOCOL_ATTACK = "protocol_attack"
    HYPERTRACE = "hypertrace"
    KERNEL_ERROR = "kernel_error"
    RUNTIME_MTL_VIOLATION = "runtime_mtl_violation"
    GENERIC_FAILURE = "generic_failure"


class CounterexampleNodeKind(str, Enum):
    COUNTEREXAMPLE = "counterexample"
    PLAN = "plan"
    TASK = "task"
    TREE = "tree"
    AST_SCOPE = "ast_scope"
    ASSUMPTION = "assumption"
    OBLIGATION = "obligation"
    PROVIDER = "provider"
    RECEIPT = "receipt"
    EVIDENCE = "evidence"
    POLICY = "policy"


class CounterexampleEdgeKind(str, Enum):
    COUNTEREXAMPLE_TO = "counterexample_to"
    AFFECTS = "affects"
    OBSERVED_ON = "observed_on"
    SCOPED_TO = "scoped_to"
    USES_ASSUMPTION = "uses_assumption"
    PRODUCED_BY = "produced_by"
    RECORDED_BY = "recorded_by"
    INVALIDATES = "invalidates"
    GOVERNED_BY = "governed_by"


class RepairClass(str, Enum):
    """Deterministic repair families understood by the following replan stage."""

    ADD_DEPENDENCY = "add_or_correct_dependency"
    SPLIT_TASK = "split_non_atomic_task"
    TIGHTEN_AUTHORITY = "tighten_authority_or_fencing"
    ADD_OBLIGATION = "add_obligation_or_fallback_test"
    CONSTRAIN_SCOPE = "constrain_ast_scope_or_model_bound"
    ADD_PREMISE = "add_premise_or_evidence_dependency"
    ADJUST_RESOURCES = "adjust_portfolio_or_resource_bound"
    HUMAN_REVIEW = "request_scoped_human_review"


class ConfidentialityDisposition(str, Enum):
    PUBLIC_REDACTED = "public_redacted"


_PRIVATE_KEY_RE = re.compile(
    r"(?:^|[_\-.])(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|credential|authorization|cookie|"
    r"private[_-]?key|private[_-]?premise|private[_-]?input|"
    r"hidden[_-]?witness|private[_-]?witness|witness)(?:$|[_\-.])",
    re.IGNORECASE,
)
_FORBIDDEN_CHANNEL_RE = re.compile(
    r"^(?:raw|raw_data|raw_output|provider_output|prover_output|stdout|stderr|"
    r"transcript|full_trace|full_model|source|source_code|source_text|"
    r"source_excerpt|file_content|repository_source|proof_text|command_output|"
    r"(?:[a-z0-9]+_)+source(?:_code|_text|_excerpt|_content)?)$",
    re.IGNORECASE,
)
_INLINE_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
        r"refresh[_-]?token|authorization)\s*[:=]\s*\S+"
    ),
    re.compile(r"-----BEGIN [A-Z0-9 ]*(?:PRIVATE KEY|CREDENTIAL)[A-Z0-9 ]*-----"),
    re.compile(r"(?i)\b(?:https?|ssh)://[^/\s:@]+:[^/\s@]+@"),
)
_VOLATILE_KEYS = frozenset(
    {
        "timestamp",
        "observed_at",
        "created_at",
        "duration",
        "duration_ms",
        "elapsed",
        "elapsed_ms",
        "memory_bytes",
        "pid",
        "host",
    }
)
_TRACE_KEYS = ("trace", "states", "steps", "events", "attack_trace")
_CORE_KEYS = (
    "unsat_core",
    "unsat-core",
    "unsatisfiable_core",
    "core",
    "conflicting_assumptions",
)


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({item.value for item in kind}))
        raise CounterexampleValidationError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _text(
    value: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        result = str(value).strip()
    result, _ = _redact_inline(result)
    if len(result) > maximum:
        result = result[: max(0, maximum - 1)] + "…"
    if required and not result:
        raise CounterexampleValidationError(f"{name} is required")
    return result


def _ids(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        raw_values: Iterable[Any] = ()
    elif isinstance(values, str):
        raw_values = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        raw_values = values
    else:
        raise CounterexampleValidationError(f"{name} must be a sequence")
    return tuple(
        sorted(
            {
                item
                for item in (
                    _text(value, name, maximum=256) for value in raw_values
                )
                if item and item not in {REDACTED, OMITTED}
            }
        )
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CounterexampleValidationError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CounterexampleValidationError(f"{name} keys must be strings")
    return dict(value)


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise CounterexampleValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _claimed_identity(payload: Mapping[str, Any], actual: str, *names: str) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed and claimed != actual:
            raise CounterexampleValidationError(
                "counterexample content identity does not match"
            )


def _redact_inline(value: str) -> tuple[str, int]:
    result = value
    count = 0
    for pattern in _INLINE_SECRET_PATTERNS:
        result, matches = pattern.subn(REDACTED, result)
        count += matches
    return result, count


@dataclass
class _SanitizationBudget:
    remaining_items: int
    maximum_text: int
    maximum_depth: int
    dropped_fields: int = 0
    redacted_values: int = 0
    truncated: bool = False


def _safe_value(
    value: Any,
    budget: _SanitizationBudget,
    *,
    depth: int = 0,
    drop_volatile: bool = False,
) -> Any:
    """Return bounded public JSON without inspecting private field values."""

    if budget.remaining_items <= 0:
        budget.truncated = True
        return OMITTED
    budget.remaining_items -= 1
    if depth >= budget.maximum_depth:
        budget.truncated = True
        return OMITTED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            budget.truncated = True
            return "<non-finite-number>"
        return format(value, ".17g")
    if isinstance(value, str):
        result, redactions = _redact_inline(value)
        budget.redacted_values += redactions
        if len(result) > budget.maximum_text:
            budget.truncated = True
            result = result[: max(0, budget.maximum_text - 1)] + "…"
        return result
    if isinstance(value, (bytes, bytearray, memoryview)):
        budget.dropped_fields += 1
        return OMITTED
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        keys = sorted(value, key=lambda item: str(item))
        if len(keys) > budget.remaining_items:
            budget.truncated = True
        for raw_key in keys:
            key = str(raw_key).strip()
            normalized_key = key.lower().replace("-", "_")
            if (
                _PRIVATE_KEY_RE.search(normalized_key)
                or _FORBIDDEN_CHANNEL_RE.match(normalized_key)
            ):
                # Do not read the value: public digests must not become secret
                # equality oracles.
                budget.dropped_fields += 1
                continue
            if drop_volatile and normalized_key in _VOLATILE_KEYS:
                continue
            if budget.remaining_items <= 0:
                budget.truncated = True
                break
            result[key] = _safe_value(
                value[raw_key],
                budget,
                depth=depth + 1,
                drop_volatile=drop_volatile,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        items = list(value)
        if len(items) > budget.remaining_items:
            budget.truncated = True
        result = []
        for item in items:
            if budget.remaining_items <= 0:
                budget.truncated = True
                break
            result.append(
                _safe_value(
                    item,
                    budget,
                    depth=depth + 1,
                    drop_volatile=drop_volatile,
                )
            )
        return result
    if isinstance(value, (set, frozenset)):
        items = sorted(value, key=repr)
        normalized = [
            _safe_value(
                item,
                budget,
                depth=depth + 1,
                drop_volatile=drop_volatile,
            )
            for item in items[: max(0, budget.remaining_items)]
        ]
        if len(items) > len(normalized):
            budget.truncated = True
        return sorted(normalized, key=canonical_json_bytes)
    budget.dropped_fields += 1
    return OMITTED


def _bounded_public(
    value: Any,
    limits: "CounterexampleLimits",
    *,
    maximum_bytes: int | None = None,
    drop_volatile: bool = False,
) -> tuple[Any, "_SanitizationBudget"]:
    budget = _SanitizationBudget(
        remaining_items=limits.max_collection_items,
        maximum_text=limits.max_text_chars,
        maximum_depth=limits.max_nesting_depth,
    )
    result = _safe_value(value, budget, drop_volatile=drop_volatile)
    encoded = canonical_json_bytes(result)
    byte_limit = maximum_bytes or limits.max_payload_bytes
    if len(encoded) > byte_limit:
        # The digest is over an already-public value, never over discarded
        # private channels.
        result = {
            "omitted": "<payload-exceeded-byte-limit>",
            "public_digest": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        }
        budget.truncated = True
    return result, budget


def _canonical_set(values: Any, limits: "CounterexampleLimits") -> list[Any]:
    if values is None:
        items: list[Any] = []
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        items = list(values)
    else:
        items = [values]
    normalized, _ = _bounded_public(items, limits)
    if not isinstance(normalized, list):
        return [normalized] if normalized else []
    unique = {canonical_json_bytes(item): item for item in normalized}
    return [unique[key] for key in sorted(unique)]


def _without_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_volatile(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower().replace("-", "_") not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_without_volatile(item) for item in value]
    return value


def _contains_unsafe_key(value: Any) -> bool:
    """Inspect field names only; values are intentionally never reflected."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if _PRIVATE_KEY_RE.search(normalized) or _FORBIDDEN_CHANNEL_RE.match(
                normalized
            ):
                return True
            if _contains_unsafe_key(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return any(_contains_unsafe_key(item) for item in value)
    return False


def _contains_inline_secret(value: Any) -> bool:
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _INLINE_SECRET_PATTERNS)
    if isinstance(value, Mapping):
        return any(_contains_inline_secret(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return any(_contains_inline_secret(item) for item in value)
    return False


def _count_unsafe_fields(value: Any) -> int:
    """Count unsafe field names without reading their associated values."""

    if isinstance(value, Mapping):
        count = 0
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if _PRIVATE_KEY_RE.search(normalized) or _FORBIDDEN_CHANNEL_RE.match(
                normalized
            ):
                count += 1
                continue
            count += _count_unsafe_fields(item)
        return count
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return sum(_count_unsafe_fields(item) for item in value)
    return 0


def _minimal_trace(value: Any, limits: "CounterexampleLimits") -> tuple[list[Any], _SanitizationBudget]:
    if isinstance(value, Mapping):
        for key in _TRACE_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray, memoryview)
            ):
                value = candidate
                break
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        value = [value] if value not in (None, "") else []
    raw_steps = list(value)
    if len(raw_steps) > limits.max_trace_steps * 2:
        raw_steps = [
            raw_steps[0],
            *raw_steps[-(limits.max_trace_steps * 2 - 1) :],
        ]
        pretruncated = True
    else:
        pretruncated = False
    public, budget = _bounded_public(raw_steps, limits, drop_volatile=True)
    budget.truncated = budget.truncated or pretruncated
    steps = public if isinstance(public, list) else [public]
    result: list[Any] = []
    previous: bytes | None = None
    for step in steps:
        semantic = canonical_json_bytes(_without_volatile(step))
        if semantic == previous:
            continue
        previous = semantic
        result.append(step)
    if len(result) > limits.max_trace_steps:
        # Retain both the initial condition and the suffix ending at the
        # violation.  Middle state explosion is not actionable prompt context.
        keep_tail = max(1, limits.max_trace_steps - 1)
        result = [result[0], *result[-keep_tail:]]
        result = result[: limits.max_trace_steps]
        budget.truncated = True
    return result, budget


@dataclass(frozen=True)
class CounterexampleLimits:
    """Hard limits applied before persistence and prompt construction."""

    max_counterexample_bytes: int = DEFAULT_MAX_COUNTEREXAMPLE_BYTES
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_capsule_bytes: int = DEFAULT_MAX_CAPSULE_BYTES
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    max_collection_items: int = DEFAULT_MAX_COLLECTION_ITEMS
    max_nesting_depth: int = DEFAULT_MAX_NESTING_DEPTH
    max_trace_steps: int = DEFAULT_MAX_TRACE_STEPS
    max_capsule_counterexamples: int = DEFAULT_MAX_CAPSULE_COUNTEREXAMPLES
    max_graph_nodes: int = DEFAULT_MAX_GRAPH_NODES
    max_graph_edges: int = DEFAULT_MAX_GRAPH_EDGES

    def __post_init__(self) -> None:
        minima = {
            "max_counterexample_bytes": 1024,
            "max_payload_bytes": 256,
            "max_capsule_bytes": 1024,
            "max_text_chars": 32,
            "max_collection_items": 4,
            "max_nesting_depth": 2,
            "max_trace_steps": 2,
            "max_capsule_counterexamples": 1,
            "max_graph_nodes": 1,
            "max_graph_edges": 1,
        }
        for name, minimum in minima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise CounterexampleValidationError(
                    f"{name} must be an integer of at least {minimum}"
                )
        if self.max_payload_bytes >= self.max_counterexample_bytes:
            raise CounterexampleValidationError(
                "max_payload_bytes must be below max_counterexample_bytes"
            )
        if self.max_payload_bytes > DEFAULT_MAX_PAYLOAD_BYTES:
            raise CounterexampleValidationError(
                "max_payload_bytes exceeds the canonical public payload limit"
            )
        if self.max_counterexample_bytes > ABSOLUTE_MAX_COUNTEREXAMPLE_BYTES:
            raise CounterexampleValidationError(
                "max_counterexample_bytes exceeds the absolute public limit"
            )


@dataclass(frozen=True)
class RedactionReport:
    disposition: ConfidentialityDisposition = (
        ConfidentialityDisposition.PUBLIC_REDACTED
    )
    dropped_fields: int = 0
    redacted_values: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ConfidentialityDisposition, "disposition"),
        )
        for name in ("dropped_fields", "redacted_values"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CounterexampleValidationError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "dropped_fields": self.dropped_fields,
            "redacted_values": self.redacted_values,
            "contains_private_material": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RedactionReport":
        if payload.get("contains_private_material") not in (None, False):
            raise CounterexampleValidationError(
                "counterexample cannot claim private material"
            )
        return cls(
            disposition=payload.get(
                "disposition", ConfidentialityDisposition.PUBLIC_REDACTED
            ),
            dropped_fields=payload.get("dropped_fields", 0),
            redacted_values=payload.get("redacted_values", 0),
        )


@dataclass(frozen=True)
class CounterexampleBindings:
    """Content identities connecting a failure to its evidence neighborhood."""

    plan_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    tree_ids: tuple[str, ...] = ()
    ast_scope_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    invalidated_evidence_ids: tuple[str, ...] = ()
    policy_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "plan_ids",
            "task_ids",
            "tree_ids",
            "ast_scope_ids",
            "assumption_ids",
            "obligation_ids",
            "provider_ids",
            "receipt_ids",
            "invalidated_evidence_ids",
            "policy_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_ids": list(self.plan_ids),
            "task_ids": list(self.task_ids),
            "tree_ids": list(self.tree_ids),
            "ast_scope_ids": list(self.ast_scope_ids),
            "assumption_ids": list(self.assumption_ids),
            "obligation_ids": list(self.obligation_ids),
            "provider_ids": list(self.provider_ids),
            "receipt_ids": list(self.receipt_ids),
            "invalidated_evidence_ids": list(self.invalidated_evidence_ids),
            "policy_ids": list(self.policy_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CounterexampleBindings":
        value = payload or {}

        def many(plural: str, *singulars: str) -> Any:
            if plural in value:
                return value.get(plural)
            return tuple(value.get(name) for name in singulars if value.get(name))

        assumptions = value.get("assumption_ids")
        if assumptions is None:
            assumptions = value.get("premise_ids")
        if assumptions is None:
            assumptions = many(
                "_missing_assumption_ids", "assumption_id", "premise_id"
            )

        return cls(
            plan_ids=many("plan_ids", "plan_id", "plan_cid"),
            task_ids=many("task_ids", "task_id", "task_cid"),
            tree_ids=many(
                "tree_ids", "tree_id", "tree_cid", "repository_tree_id"
            ),
            ast_scope_ids=many(
                "ast_scope_ids", "ast_scope_id", "scope_id", "symbol_cid"
            ),
            assumption_ids=assumptions,
            obligation_ids=many(
                "obligation_ids", "obligation_id", "property_id"
            ),
            provider_ids=many("provider_ids", "provider_id", "prover_id"),
            receipt_ids=many("receipt_ids", "receipt_id"),
            invalidated_evidence_ids=many(
                "invalidated_evidence_ids",
                "invalidated_evidence_id",
                "evidence_id",
            ),
            policy_ids=many("policy_ids", "policy_id"),
        )

    def merged(self, other: "CounterexampleBindings") -> "CounterexampleBindings":
        if not isinstance(other, CounterexampleBindings):
            raise CounterexampleValidationError(
                "bindings can only be merged with bindings"
            )
        return CounterexampleBindings(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    @property
    def semantic_dict(self) -> dict[str, Any]:
        """Bindings which define the failure, excluding producer lineage."""

        return {
            "plan_ids": list(self.plan_ids),
            "task_ids": list(self.task_ids),
            "tree_ids": list(self.tree_ids),
            "ast_scope_ids": list(self.ast_scope_ids),
            "assumption_ids": list(self.assumption_ids),
            "obligation_ids": list(self.obligation_ids),
            "policy_ids": list(self.policy_ids),
        }


@dataclass(frozen=True)
class FormalCounterexample(CanonicalContract):
    """One minimized, redacted, content-addressed proof failure."""

    SCHEMA = FORMAL_COUNTEREXAMPLE_SCHEMA

    kind: CounterexampleKind
    property_class: str
    violated_property: str
    summary: str
    payload: Mapping[str, Any]
    bindings: CounterexampleBindings = field(default_factory=CounterexampleBindings)
    assumption_ids: tuple[str, ...] = ()
    finite_bounds: Mapping[str, Any] = field(default_factory=dict)
    observation_policy_id: str = ""
    repair_classes: tuple[RepairClass, ...] = ()
    redaction: RedactionReport = field(default_factory=RedactionReport)
    minimized: bool = True
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, CounterexampleKind, "kind")
        )
        object.__setattr__(
            self,
            "property_class",
            _text(self.property_class, "property_class", required=True, maximum=128),
        )
        object.__setattr__(
            self,
            "violated_property",
            _text(
                self.violated_property,
                "violated_property",
                required=True,
                maximum=256,
            ),
        )
        object.__setattr__(
            self, "summary", _text(self.summary, "summary", required=True)
        )
        if not isinstance(self.bindings, CounterexampleBindings):
            raise CounterexampleValidationError(
                "bindings must be CounterexampleBindings"
            )
        if not isinstance(self.redaction, RedactionReport):
            raise CounterexampleValidationError(
                "redaction must be a RedactionReport"
            )
        object.__setattr__(self, "assumption_ids", _ids(self.assumption_ids, "assumption_ids"))
        object.__setattr__(
            self,
            "observation_policy_id",
            _text(
                self.observation_policy_id,
                "observation_policy_id",
                maximum=256,
            ),
        )
        repairs = tuple(
            sorted(
                {
                    _enum(item, RepairClass, "repair_classes")
                    for item in self.repair_classes
                },
                key=lambda item: item.value,
            )
        )
        object.__setattr__(self, "repair_classes", repairs)
        if self.minimized is not True:
            raise CounterexampleValidationError(
                "formal counterexamples must be minimized"
            )
        if not isinstance(self.truncated, bool):
            raise CounterexampleValidationError("truncated must be boolean")
        payload = _mapping(self.payload, "payload")
        bounds = _mapping(self.finite_bounds, "finite_bounds")
        if (
            _contains_unsafe_key(payload)
            or _contains_unsafe_key(bounds)
            or _contains_inline_secret(payload)
            or _contains_inline_secret(bounds)
        ):
            raise CounterexampleValidationError(
                "counterexample contains a private or unbounded channel"
            )
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "finite_bounds", bounds)
        if len(canonical_json_bytes(payload)) > DEFAULT_MAX_PAYLOAD_BYTES:
            raise CounterexampleValidationError(
                "counterexample payload exceeds the canonical public limit"
            )
        if len(self.canonical_bytes()) > ABSOLUTE_MAX_COUNTEREXAMPLE_BYTES:
            raise CounterexampleValidationError(
                "counterexample exceeds the absolute public limit"
            )

    @property
    def semantic_id(self) -> str:
        return content_identity(
            {
                "kind": self.kind.value,
                "property_class": self.property_class,
                "violated_property": self.violated_property,
                "payload": self.payload,
                "bindings": self.bindings.semantic_dict,
                "assumption_ids": list(self.assumption_ids),
                "finite_bounds": self.finite_bounds,
                "observation_policy_id": self.observation_policy_id,
            }
        )

    @property
    def counterexample_id(self) -> str:
        return self.semantic_id

    @property
    def byte_size(self) -> int:
        return len(self.canonical_bytes())

    @property
    def contains_private_material(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "counterexample_version": FORMAL_COUNTEREXAMPLE_VERSION,
            "counterexample_id": self.semantic_id,
            "kind": self.kind,
            "property_class": self.property_class,
            "violated_property": self.violated_property,
            "summary": self.summary,
            "payload": self.payload,
            "bindings": self.bindings.to_dict(),
            "assumption_ids": self.assumption_ids,
            "finite_bounds": self.finite_bounds,
            "observation_policy_id": self.observation_policy_id,
            "repair_classes": self.repair_classes,
            "redaction": self.redaction.to_dict(),
            "minimized": True,
            "truncated": self.truncated,
            "contains_private_material": False,
            "contains_raw_prover_output": False,
            "contains_source": False,
        }

    def to_capsule_dict(self) -> dict[str, Any]:
        """Return the only counterexample projection allowed in model context."""

        return {
            "counterexample_id": self.semantic_id,
            "kind": self.kind.value,
            "property_class": self.property_class,
            "violated_property": self.violated_property,
            "summary": self.summary,
            "payload": self.payload,
            "bindings": self.bindings.to_dict(),
            "assumption_ids": list(self.assumption_ids),
            "finite_bounds": dict(self.finite_bounds),
            "observation_policy_id": self.observation_policy_id,
            "repair_classes": [item.value for item in self.repair_classes],
            "minimized": True,
            "redacted": True,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalCounterexample":
        _schema(payload, cls.SCHEMA)
        redaction = payload.get("redaction")
        bindings = payload.get("bindings")
        result = cls(
            kind=payload.get("kind", CounterexampleKind.GENERIC_FAILURE),
            property_class=payload.get("property_class", ""),
            violated_property=payload.get("violated_property", ""),
            summary=payload.get("summary", ""),
            payload=payload.get("payload") or {},
            bindings=CounterexampleBindings.from_dict(
                bindings if isinstance(bindings, Mapping) else None
            ),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            finite_bounds=payload.get("finite_bounds") or {},
            observation_policy_id=payload.get("observation_policy_id", ""),
            repair_classes=tuple(payload.get("repair_classes") or ()),
            redaction=RedactionReport.from_dict(
                redaction if isinstance(redaction, Mapping) else {}
            ),
            minimized=payload.get("minimized", False),
            truncated=payload.get("truncated", False),
        )
        if payload.get("contains_private_material") not in (None, False):
            raise CounterexampleValidationError(
                "counterexample claims to contain private material"
            )
        if payload.get("contains_raw_prover_output") not in (None, False):
            raise CounterexampleValidationError(
                "counterexample claims to contain raw prover output"
            )
        if payload.get("contains_source") not in (None, False):
            raise CounterexampleValidationError(
                "counterexample claims to contain source"
            )
        _claimed_identity(
            payload,
            result.semantic_id,
            "counterexample_id",
        )
        _claimed_identity(payload, result.content_id, "content_id")
        return result


def _merge_mapping_bindings(
    raw: Mapping[str, Any], explicit: CounterexampleBindings | Mapping[str, Any] | None
) -> CounterexampleBindings:
    discovered = CounterexampleBindings.from_dict(raw)
    nested = raw.get("bindings")
    if isinstance(nested, Mapping):
        discovered = discovered.merged(CounterexampleBindings.from_dict(nested))
    partition = raw.get("partition")
    if isinstance(partition, Mapping):
        discovered = discovered.merged(CounterexampleBindings.from_dict(partition))
    if explicit is None:
        return discovered
    supplied = (
        explicit
        if isinstance(explicit, CounterexampleBindings)
        else CounterexampleBindings.from_dict(explicit)
    )
    return discovered.merged(supplied)


def _infer_kind(raw: Mapping[str, Any]) -> CounterexampleKind:
    explicit = raw.get("counterexample_kind", raw.get("kind"))
    if explicit:
        normalized = str(getattr(explicit, "value", explicit)).strip().lower()
        aliases = {
            "model": CounterexampleKind.SMT_MODEL,
            "smt": CounterexampleKind.SMT_MODEL,
            "unsat": CounterexampleKind.SMT_UNSAT_CORE,
            "unsat_core": CounterexampleKind.SMT_UNSAT_CORE,
            "contradiction": CounterexampleKind.TDFOL_CONTRADICTION,
            "runtime_violation": CounterexampleKind.RUNTIME_MTL_VIOLATION,
            "runtime_temporal_violation": CounterexampleKind.RUNTIME_MTL_VIOLATION,
            "attack": CounterexampleKind.PROTOCOL_ATTACK,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return CounterexampleKind(normalized)
        except ValueError:
            pass
    schema = str(raw.get("schema") or "").lower()
    provider = " ".join(
        str(raw.get(name) or "").lower()
        for name in ("provider_id", "prover_id", "engine", "logic", "property_class")
    )
    keys = {str(key).lower().replace("-", "_") for key in raw}
    if "runtime-temporal-counterexample" in schema or "runtime_mtl" in provider:
        return CounterexampleKind.RUNTIME_MTL_VIOLATION
    if "hypertrace" in schema or "hyper" in provider or "trace_refs" in keys:
        return CounterexampleKind.HYPERTRACE
    if keys & {key.replace("-", "_") for key in _CORE_KEYS}:
        return CounterexampleKind.SMT_UNSAT_CORE
    if "dcec" in provider:
        return CounterexampleKind.DCEC_CONTRADICTION
    if "tdfol" in provider:
        return CounterexampleKind.TDFOL_CONTRADICTION
    if (
        any(item in provider for item in ("tla", "tlc", "apalache"))
        or "tla" in schema
    ):
        return CounterexampleKind.TLA_TRACE
    if (
        any(item in provider for item in ("tamarin", "proverif", "protocol"))
        or "attack_trace" in keys
    ):
        return CounterexampleKind.PROTOCOL_ATTACK
    if (
        "kernel" in provider
        or "kernel" in schema
        or "failure_code" in keys
        or "kernel_id" in keys
    ):
        return CounterexampleKind.KERNEL_ERROR
    if keys & {"model", "assignment", "assignments"}:
        return CounterexampleKind.SMT_MODEL
    if keys & {"contradiction", "conflict", "conflicting_formulas"}:
        return CounterexampleKind.TDFOL_CONTRADICTION
    if keys & {"events", "states", "steps", "trace"}:
        return CounterexampleKind.TLA_TRACE
    return CounterexampleKind.GENERIC_FAILURE


def _first(raw: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in raw and raw.get(name) not in (None, "", (), [], {}):
            return raw.get(name)
    return default


def _counterexample_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Select a conclusive retained attempt without copying raw channels."""

    result = dict(raw)
    plan = raw.get("plan")
    if isinstance(plan, Mapping):
        if plan.get("plan_id") or plan.get("content_id"):
            result.setdefault(
                "plan_id", plan.get("plan_id") or plan.get("content_id")
            )
        obligation = plan.get("obligation")
        if isinstance(obligation, Mapping):
            for key in (
                "obligation_id",
                "property_kind",
                "premise_ids",
            ):
                if key in obligation:
                    result.setdefault(key, obligation[key])

    attempts = raw.get("attempts")
    selected_attempt: Mapping[str, Any] | None = None
    if isinstance(attempts, Sequence) and not isinstance(
        attempts, (str, bytes, bytearray, memoryview)
    ):
        claimed = str(raw.get("counterexample_attempt_id") or "")
        candidates: list[tuple[Mapping[str, Any], str]] = []
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            attempt_id = str(
                attempt.get("attempt_id")
                or attempt.get("content_id")
                or ""
            )
            outcome = str(
                attempt.get("effective_outcome")
                or attempt.get("reported_outcome")
                or ""
            ).lower()
            if outcome == "counterexample" and attempt.get("conclusive") is True:
                candidates.append((attempt, attempt_id))
            if claimed and attempt_id == claimed:
                selected_attempt = attempt
                result.setdefault("receipt_id", claimed)
                break
        if selected_attempt is None and len(candidates) == 1:
            selected_attempt, attempt_id = candidates[0]
            receipt_id = claimed or attempt_id
            if receipt_id:
                result.setdefault("receipt_id", receipt_id)

    if selected_attempt is not None:
        for key in ("prover_id", "provider_id"):
            if selected_attempt.get(key):
                result.setdefault("provider_id", selected_attempt[key])
        evidence = selected_attempt.get("evidence")
        if isinstance(evidence, Mapping):
            for key, item in evidence.items():
                result.setdefault(str(key), item)
    else:
        evidence = raw.get("evidence")
        if isinstance(evidence, Mapping):
            for key, item in evidence.items():
                result.setdefault(str(key), item)

    nested = result.get("counterexample")
    if isinstance(nested, Mapping):
        for key, item in nested.items():
            result.setdefault(str(key), item)
    elif isinstance(nested, Sequence) and not isinstance(
        nested, (str, bytes, bytearray, memoryview)
    ):
        result.setdefault("trace", nested)
    return result


def _property_class(kind: CounterexampleKind, raw: Mapping[str, Any]) -> str:
    supplied = raw.get("property_class")
    property_kind = raw.get("property_kind")
    known_classes = {
        "finite_constraint",
        "state_machine",
        "authorization",
        "protocol",
        "hyperproperty",
        "runtime_trace",
        "kernel_check",
        "typed_planning",
        "temporal_deontic",
        "first_order_theorem",
    }
    if not supplied and str(property_kind or "").strip().lower() in known_classes:
        supplied = property_kind
    if supplied:
        return _text(supplied, "property_class", required=True, maximum=128)
    return {
        CounterexampleKind.SMT_MODEL: "finite_constraint",
        CounterexampleKind.SMT_UNSAT_CORE: "finite_constraint",
        CounterexampleKind.DCEC_CONTRADICTION: "typed_planning",
        CounterexampleKind.TDFOL_CONTRADICTION: "temporal_deontic",
        CounterexampleKind.TLA_TRACE: "state_machine",
        CounterexampleKind.PROTOCOL_ATTACK: "protocol",
        CounterexampleKind.HYPERTRACE: "hyperproperty",
        CounterexampleKind.KERNEL_ERROR: "kernel_check",
        CounterexampleKind.RUNTIME_MTL_VIOLATION: "runtime_trace",
        CounterexampleKind.GENERIC_FAILURE: "formal_verification",
    }[kind]


def _default_summary(
    kind: CounterexampleKind, violated_property: str, payload: Mapping[str, Any]
) -> str:
    subject = violated_property or "the selected obligation"
    if kind is CounterexampleKind.SMT_MODEL:
        return f"SMT model falsifies {subject}."
    if kind is CounterexampleKind.SMT_UNSAT_CORE:
        return f"Minimal assumption core is inconsistent with {subject}."
    if kind is CounterexampleKind.DCEC_CONTRADICTION:
        return f"DCEC norms or events contradict {subject}."
    if kind is CounterexampleKind.TDFOL_CONTRADICTION:
        return f"TDFOL premises contradict {subject}."
    if kind is CounterexampleKind.TLA_TRACE:
        return f"Bounded state trace violates {subject}."
    if kind is CounterexampleKind.PROTOCOL_ATTACK:
        return f"Protocol attack reaches a state forbidden by {subject}."
    if kind is CounterexampleKind.HYPERTRACE:
        return f"Two-trace observation differs under {subject}."
    if kind is CounterexampleKind.RUNTIME_MTL_VIOLATION:
        return f"Observed runtime events violate {subject}."
    if kind is CounterexampleKind.KERNEL_ERROR:
        code = _text(payload.get("failure_code"), "failure_code", maximum=96)
        suffix = f" ({code})" if code else ""
        return f"Independent kernel rejected {subject}{suffix}."
    return f"Formal verification failed for {subject}."


def _repair_classes(kind: CounterexampleKind) -> tuple[RepairClass, ...]:
    return {
        CounterexampleKind.SMT_MODEL: (
            RepairClass.ADD_PREMISE,
            RepairClass.CONSTRAIN_SCOPE,
        ),
        CounterexampleKind.SMT_UNSAT_CORE: (
            RepairClass.ADD_PREMISE,
            RepairClass.SPLIT_TASK,
        ),
        CounterexampleKind.DCEC_CONTRADICTION: (
            RepairClass.ADD_DEPENDENCY,
            RepairClass.TIGHTEN_AUTHORITY,
        ),
        CounterexampleKind.TDFOL_CONTRADICTION: (
            RepairClass.ADD_DEPENDENCY,
            RepairClass.ADD_PREMISE,
        ),
        CounterexampleKind.TLA_TRACE: (
            RepairClass.SPLIT_TASK,
            RepairClass.TIGHTEN_AUTHORITY,
        ),
        CounterexampleKind.PROTOCOL_ATTACK: (
            RepairClass.TIGHTEN_AUTHORITY,
            RepairClass.ADD_OBLIGATION,
        ),
        CounterexampleKind.HYPERTRACE: (
            RepairClass.CONSTRAIN_SCOPE,
            RepairClass.ADD_OBLIGATION,
        ),
        CounterexampleKind.KERNEL_ERROR: (
            RepairClass.ADD_PREMISE,
            RepairClass.ADD_OBLIGATION,
        ),
        CounterexampleKind.RUNTIME_MTL_VIOLATION: (
            RepairClass.ADD_DEPENDENCY,
            RepairClass.ADD_OBLIGATION,
        ),
        CounterexampleKind.GENERIC_FAILURE: (RepairClass.HUMAN_REVIEW,),
    }[kind]


_KERNEL_ACTIONS: Mapping[str, str] = {
    "binding_mismatch": "Rebuild the proof for the exact obligation and tree.",
    "corrupt_evidence": "Discard the artifact and rerun independent checking.",
    "digest_mismatch": "Recreate the reconstruction from canonical inputs.",
    "environment_mismatch": "Rerun with the pinned kernel environment.",
    "forbidden_declaration": "Remove declarations from the candidate proof term.",
    "forbidden_import": "Use only imports fixed by the reviewed theorem template.",
    "incomplete_proof": "Replace admitted or incomplete proof terms.",
    "kernel_unavailable": "Restore the configured independent kernel.",
    "malformed_reconstruction": "Regenerate the typed reconstruction packet.",
    "statement_mismatch": "Bind the proof to the fixed theorem statement.",
    "timeout": "Reduce the proof or use the configured bounded fallback.",
    "unsupported_kernel": "Select a supported kernel or scoped fallback.",
}


def _normalize_payload(
    kind: CounterexampleKind,
    raw: Mapping[str, Any],
    limits: CounterexampleLimits,
) -> tuple[dict[str, Any], _SanitizationBudget]:
    budget = _SanitizationBudget(
        remaining_items=limits.max_collection_items,
        maximum_text=limits.max_text_chars,
        maximum_depth=limits.max_nesting_depth,
    )

    if kind is CounterexampleKind.SMT_MODEL:
        model = _first(raw, ("model", "assignments", "assignment", "counterexample"), {})
        public, budget = _bounded_public(model, limits, drop_volatile=True)
        payload = {"assignments": public}
    elif kind is CounterexampleKind.SMT_UNSAT_CORE:
        core = _first(raw, _CORE_KEYS, ())
        payload = {"core": _canonical_set(core, limits)}
        _, budget = _bounded_public(core, limits)
    elif kind in {
        CounterexampleKind.DCEC_CONTRADICTION,
        CounterexampleKind.TDFOL_CONTRADICTION,
    }:
        contradiction = _first(
            raw,
            ("contradiction", "conflict", "conflicting_formulas", "diagnostic"),
            {},
        )
        public, budget = _bounded_public(
            contradiction, limits, drop_volatile=True
        )
        premises = _canonical_set(
            _first(raw, ("conflicting_assumptions", "premises", "assumptions"), ()),
            limits,
        )
        payload = {"contradiction": public, "premises": premises}
    elif kind in {
        CounterexampleKind.TLA_TRACE,
        CounterexampleKind.PROTOCOL_ATTACK,
        CounterexampleKind.RUNTIME_MTL_VIOLATION,
    }:
        trace = _first(raw, _TRACE_KEYS, ())
        steps, budget = _minimal_trace(trace, limits)
        payload = {"steps": steps}
        if kind is CounterexampleKind.TLA_TRACE:
            for key in ("invariant", "action", "state", "model_id"):
                if raw.get(key) not in (None, ""):
                    value, extra = _bounded_public(raw[key], limits)
                    payload[key] = value
                    budget.dropped_fields += extra.dropped_fields
                    budget.redacted_values += extra.redacted_values
                    budget.truncated = budget.truncated or extra.truncated
        elif kind is CounterexampleKind.PROTOCOL_ATTACK:
            for key in ("query", "claim", "role", "session"):
                if raw.get(key) not in (None, ""):
                    value, extra = _bounded_public(raw[key], limits)
                    payload[key] = value
                    budget.dropped_fields += extra.dropped_fields
                    budget.redacted_values += extra.redacted_values
                    budget.truncated = budget.truncated or extra.truncated
        else:
            partition = raw.get("partition")
            if isinstance(partition, Mapping):
                value, extra = _bounded_public(partition, limits)
                payload["partition"] = value
                budget.dropped_fields += extra.dropped_fields
                budget.redacted_values += extra.redacted_values
                budget.truncated = budget.truncated or extra.truncated
            trigger = raw.get("trigger_event_id")
            if trigger:
                payload["trigger_event_id"] = _text(
                    trigger, "trigger_event_id", maximum=256
                )
    elif kind is CounterexampleKind.HYPERTRACE:
        allowed = {
            key: raw[key]
            for key in (
                "model_id",
                "model_identity",
                "observation_policy_id",
                "observation_policy_name",
                "observed_fields",
                "trace_refs",
                "differences",
            )
            if key in raw
        }
        # Some adapters wrap their public counterexample.
        nested = raw.get("counterexample")
        if isinstance(nested, Mapping):
            for key in (
                "model_id",
                "model_identity",
                "observation_policy_id",
                "observation_policy_name",
                "observed_fields",
                "trace_refs",
                "differences",
            ):
                if key in nested and key not in allowed:
                    allowed[key] = nested[key]
        public, budget = _bounded_public(allowed, limits, drop_volatile=True)
        payload = public if isinstance(public, dict) else {"hypertrace": public}
    elif kind is CounterexampleKind.KERNEL_ERROR:
        code = _text(
            _first(raw, ("failure_code", "code", "status"), "kernel_rejected"),
            "failure_code",
            required=True,
            maximum=96,
        ).lower()
        payload = {
            "failure_code": code,
            "action": _KERNEL_ACTIONS.get(
                code, "Rebuild and independently check the canonical proof artifact."
            ),
        }
        for key in (
            "kernel_id",
            "kernel_version",
            "target",
            "theorem_id",
            "obligation_id",
            "artifact_id",
        ):
            if raw.get(key):
                payload[key] = _text(raw[key], key, maximum=256)
        # Raw exception/reason text is intentionally excluded.  Kernel codes
        # select reviewed actionable diagnostics instead.
        _, budget = _bounded_public(
            {
                key: value
                for key, value in raw.items()
                if key in payload
            },
            limits,
        )
        for key in raw:
            if _PRIVATE_KEY_RE.search(str(key)) or _FORBIDDEN_CHANNEL_RE.match(str(key)):
                budget.dropped_fields += 1
    else:
        code = _text(
            _first(raw, ("failure_code", "code", "status"), "verification_failed"),
            "failure_code",
            maximum=96,
        )
        payload = {
            "failure_code": code,
            "action": "Inspect the bound obligation and rerun its scoped verifier.",
        }
        _, budget = _bounded_public(raw, limits)

    bounded, extra = _bounded_public(
        payload, limits, maximum_bytes=limits.max_payload_bytes
    )
    budget.dropped_fields += extra.dropped_fields
    budget.redacted_values += extra.redacted_values
    budget.truncated = budget.truncated or extra.truncated
    if not isinstance(bounded, dict):
        bounded = {"value": bounded}
    return bounded, budget


def _extract_bounds(raw: Mapping[str, Any], limits: CounterexampleLimits) -> dict[str, Any]:
    value = _first(raw, ("finite_bounds", "bounds", "model_bounds"), {})
    if not isinstance(value, Mapping):
        return {}
    public, _ = _bounded_public(value, limits, maximum_bytes=2048)
    return public if isinstance(public, dict) else {}


def _fit_counterexample(
    value: FormalCounterexample, limits: CounterexampleLimits
) -> FormalCounterexample:
    if value.byte_size <= limits.max_counterexample_bytes:
        return value
    encoded = canonical_json_bytes(value.payload)
    compact = replace(
        value,
        payload={
            "omitted": "<counterexample-exceeded-byte-limit>",
            "public_digest": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        },
        truncated=True,
    )
    if compact.byte_size > limits.max_counterexample_bytes:
        compact = replace(
            compact,
            summary=_default_summary(
                compact.kind, compact.violated_property, compact.payload
            ),
            repair_classes=(),
        )
    if compact.byte_size > limits.max_counterexample_bytes:
        raise CounterexampleBudgetError(
            "minimal formal counterexample exceeds persistence byte limit"
        )
    return compact


def normalize_counterexample(
    value: Any,
    *,
    kind: CounterexampleKind | str | None = None,
    bindings: CounterexampleBindings | Mapping[str, Any] | None = None,
    property_class: str = "",
    violated_property: str = "",
    summary: str = "",
    assumption_ids: Iterable[str] = (),
    finite_bounds: Mapping[str, Any] | None = None,
    observation_policy_id: str = "",
    repair_classes: Iterable[RepairClass | str] | None = None,
    limits: CounterexampleLimits | None = None,
) -> FormalCounterexample:
    """Normalize a provider result or typed failure into the canonical IR."""

    active_limits = limits or CounterexampleLimits()
    if isinstance(value, FormalCounterexample):
        return _fit_counterexample(value, active_limits)
    if isinstance(value, Mapping):
        raw = _counterexample_view(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if not isinstance(converted, Mapping):
            raise CounterexampleValidationError("to_dict() must return an object")
        plan = getattr(value, "plan", None)
        plan_id = getattr(plan, "plan_id", "") if plan is not None else ""
        if plan_id and "plan_id" not in converted:
            converted = {**converted, "plan_id": str(plan_id)}
        raw = _counterexample_view(converted)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        raw = {"counterexample": list(value)}
    else:
        raise CounterexampleValidationError(
            "counterexample input must be an object, sequence, or typed contract"
        )

    selected_kind = (
        _enum(kind, CounterexampleKind, "kind") if kind is not None else _infer_kind(raw)
    )
    selected_bindings = _merge_mapping_bindings(raw, bindings)
    selected_property_class = (
        _text(
            property_class,
            "property_class",
            required=True,
            maximum=128,
        )
        if property_class
        else _property_class(selected_kind, raw)
    )
    selected_property = _text(
        violated_property
        or _first(
            raw,
            (
                "violated_property",
                "property_id",
                "invariant",
                "query",
                "obligation_id",
                "theorem_id",
            ),
            "unknown-obligation",
        ),
        "violated_property",
        required=True,
        maximum=256,
    )
    payload, sanitation = _normalize_payload(selected_kind, raw, active_limits)
    sanitation.dropped_fields = max(
        sanitation.dropped_fields, _count_unsafe_fields(raw)
    )
    selected_assumptions = _ids(
        ((assumption_ids,) if isinstance(assumption_ids, str) else tuple(assumption_ids))
        + selected_bindings.assumption_ids
        + (
            (raw.get("assumption_ids"),)
            if isinstance(raw.get("assumption_ids"), str)
            else tuple(raw.get("assumption_ids") or ())
        )
        + (
            (raw.get("premise_ids"),)
            if isinstance(raw.get("premise_ids"), str)
            else tuple(raw.get("premise_ids") or ())
        ),
        "assumption_ids",
    )
    selected_bounds = (
        _extract_bounds(raw, active_limits)
        if finite_bounds is None
        else _extract_bounds({"finite_bounds": finite_bounds}, active_limits)
    )
    selected_observation_policy = _text(
        observation_policy_id
        or raw.get("observation_policy_id")
        or (
            payload.get("observation_policy_id")
            if selected_kind is CounterexampleKind.HYPERTRACE
            else ""
        ),
        "observation_policy_id",
        maximum=256,
    )
    selected_repairs = (
        tuple(repair_classes)
        if repair_classes is not None
        else _repair_classes(selected_kind)
    )
    selected_summary = (
        _text(summary, "summary", required=True)
        if summary
        else _default_summary(selected_kind, selected_property, payload)
    )
    result = FormalCounterexample(
        kind=selected_kind,
        property_class=selected_property_class,
        violated_property=selected_property,
        summary=selected_summary,
        payload=payload,
        bindings=selected_bindings,
        assumption_ids=selected_assumptions,
        finite_bounds=selected_bounds,
        observation_policy_id=selected_observation_policy,
        repair_classes=tuple(selected_repairs),
        redaction=RedactionReport(
            dropped_fields=sanitation.dropped_fields,
            redacted_values=sanitation.redacted_values,
        ),
        minimized=True,
        truncated=sanitation.truncated,
    )
    return _fit_counterexample(result, active_limits)


class FormalCounterexampleNormalizer:
    """Reusable normalizer carrying one reviewed limit policy."""

    def __init__(self, limits: CounterexampleLimits | None = None) -> None:
        self.limits = limits or CounterexampleLimits()

    def normalize(self, value: Any, **kwargs: Any) -> FormalCounterexample:
        return normalize_counterexample(value, limits=self.limits, **kwargs)

    def normalize_many(
        self, values: Iterable[Any], **kwargs: Any
    ) -> tuple[FormalCounterexample, ...]:
        return deduplicate_counterexamples(
            self.normalize(value, **kwargs) for value in values
        )


def _normalizer_for(kind: CounterexampleKind):
    def normalize(value: Any, **kwargs: Any) -> FormalCounterexample:
        return normalize_counterexample(value, kind=kind, **kwargs)

    return normalize


normalize_smt_model = _normalizer_for(CounterexampleKind.SMT_MODEL)
normalize_unsat_core = _normalizer_for(CounterexampleKind.SMT_UNSAT_CORE)
normalize_dcec_contradiction = _normalizer_for(
    CounterexampleKind.DCEC_CONTRADICTION
)
normalize_tdfol_contradiction = _normalizer_for(
    CounterexampleKind.TDFOL_CONTRADICTION
)
normalize_tla_trace = _normalizer_for(CounterexampleKind.TLA_TRACE)
normalize_protocol_attack = _normalizer_for(CounterexampleKind.PROTOCOL_ATTACK)
normalize_hypertrace = _normalizer_for(CounterexampleKind.HYPERTRACE)
normalize_kernel_error = _normalizer_for(CounterexampleKind.KERNEL_ERROR)
normalize_runtime_mtl_violation = _normalizer_for(
    CounterexampleKind.RUNTIME_MTL_VIOLATION
)
normalize_proof_failure = normalize_counterexample


def deduplicate_counterexamples(
    values: Iterable[FormalCounterexample],
) -> tuple[FormalCounterexample, ...]:
    """Semantically deduplicate failures while retaining all public lineage."""

    by_identity: dict[str, FormalCounterexample] = {}
    for value in values:
        if not isinstance(value, FormalCounterexample):
            raise CounterexampleValidationError(
                "deduplication requires normalized FormalCounterexample values"
            )
        existing = by_identity.get(value.semantic_id)
        if existing is None:
            by_identity[value.semantic_id] = value
            continue
        merged_bindings = existing.bindings.merged(value.bindings)
        by_identity[value.semantic_id] = replace(
            existing,
            bindings=merged_bindings,
            redaction=RedactionReport(
                dropped_fields=max(
                    existing.redaction.dropped_fields,
                    value.redaction.dropped_fields,
                ),
                redacted_values=max(
                    existing.redaction.redacted_values,
                    value.redaction.redacted_values,
                ),
            ),
            truncated=existing.truncated or value.truncated,
        )
    return tuple(by_identity[key] for key in sorted(by_identity))


@dataclass(frozen=True)
class CounterexampleGraphNode:
    node_id: str
    kind: CounterexampleNodeKind

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", _text(self.node_id, "node_id", required=True, maximum=512)
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, CounterexampleNodeKind, "node kind")
        )

    def to_dict(self) -> dict[str, str]:
        return {"node_id": self.node_id, "kind": self.kind.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterexampleGraphNode":
        return cls(payload.get("node_id", ""), payload.get("kind", ""))


@dataclass(frozen=True)
class CounterexampleGraphEdge:
    source_id: str
    target_id: str
    kind: CounterexampleEdgeKind

    def __post_init__(self) -> None:
        for name in ("source_id", "target_id"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=True, maximum=512),
            )
        object.__setattr__(
            self, "kind", _enum(self.kind, CounterexampleEdgeKind, "edge kind")
        )
        if self.source_id == self.target_id:
            raise CounterexampleValidationError("graph self edges are not allowed")

    @property
    def edge_id(self) -> str:
        return content_identity(
            {
                "source_id": self.source_id,
                "target_id": self.target_id,
                "kind": self.kind.value,
            }
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterexampleGraphEdge":
        result = cls(
            payload.get("source_id", ""),
            payload.get("target_id", ""),
            payload.get("kind", ""),
        )
        claimed = payload.get("edge_id")
        if claimed and claimed != result.edge_id:
            raise CounterexampleValidationError("graph edge identity does not match")
        return result


_BINDING_GRAPH_SPECS = (
    ("plan_ids", CounterexampleNodeKind.PLAN, CounterexampleEdgeKind.COUNTEREXAMPLE_TO),
    ("task_ids", CounterexampleNodeKind.TASK, CounterexampleEdgeKind.AFFECTS),
    ("tree_ids", CounterexampleNodeKind.TREE, CounterexampleEdgeKind.OBSERVED_ON),
    ("ast_scope_ids", CounterexampleNodeKind.AST_SCOPE, CounterexampleEdgeKind.SCOPED_TO),
    (
        "assumption_ids",
        CounterexampleNodeKind.ASSUMPTION,
        CounterexampleEdgeKind.USES_ASSUMPTION,
    ),
    (
        "obligation_ids",
        CounterexampleNodeKind.OBLIGATION,
        CounterexampleEdgeKind.COUNTEREXAMPLE_TO,
    ),
    (
        "provider_ids",
        CounterexampleNodeKind.PROVIDER,
        CounterexampleEdgeKind.PRODUCED_BY,
    ),
    (
        "receipt_ids",
        CounterexampleNodeKind.RECEIPT,
        CounterexampleEdgeKind.RECORDED_BY,
    ),
    (
        "invalidated_evidence_ids",
        CounterexampleNodeKind.EVIDENCE,
        CounterexampleEdgeKind.INVALIDATES,
    ),
    (
        "policy_ids",
        CounterexampleNodeKind.POLICY,
        CounterexampleEdgeKind.GOVERNED_BY,
    ),
)


@dataclass(frozen=True)
class CounterexampleKnowledgeGraph(CanonicalContract):
    """Deterministic graph projection for counterexample lineage and impact."""

    SCHEMA = COUNTEREXAMPLE_GRAPH_SCHEMA

    counterexamples: tuple[FormalCounterexample, ...] = ()
    nodes: tuple[CounterexampleGraphNode, ...] = ()
    edges: tuple[CounterexampleGraphEdge, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(item, FormalCounterexample)
            for item in self.counterexamples
        ):
            raise CounterexampleValidationError(
                "graph counterexamples must be normalized"
            )
        if any(not isinstance(item, CounterexampleGraphNode) for item in self.nodes):
            raise CounterexampleValidationError("graph nodes are malformed")
        if any(not isinstance(item, CounterexampleGraphEdge) for item in self.edges):
            raise CounterexampleValidationError("graph edges are malformed")
        examples = deduplicate_counterexamples(self.counterexamples)
        nodes = {item.node_id: item for item in self.nodes}
        edges = {item.edge_id: item for item in self.edges}
        for counterexample in examples:
            source = counterexample.semantic_id
            nodes[source] = CounterexampleGraphNode(
                source, CounterexampleNodeKind.COUNTEREXAMPLE
            )
            for field_name, node_kind, edge_kind in _BINDING_GRAPH_SPECS:
                for target in getattr(counterexample.bindings, field_name):
                    existing = nodes.get(target)
                    if existing is not None and existing.kind is not node_kind:
                        raise CounterexampleValidationError(
                            f"graph node {target!r} has conflicting kinds"
                        )
                    nodes[target] = CounterexampleGraphNode(target, node_kind)
                    edge = CounterexampleGraphEdge(source, target, edge_kind)
                    edges[edge.edge_id] = edge
        node_ids = set(nodes)
        if any(
            edge.source_id not in node_ids or edge.target_id not in node_ids
            for edge in edges.values()
        ):
            raise CounterexampleValidationError(
                "every graph edge endpoint must have a node"
            )
        object.__setattr__(self, "counterexamples", examples)
        object.__setattr__(
            self, "nodes", tuple(nodes[key] for key in sorted(nodes))
        )
        object.__setattr__(
            self, "edges", tuple(edges[key] for key in sorted(edges))
        )

    @classmethod
    def from_counterexamples(
        cls, values: Iterable[FormalCounterexample]
    ) -> "CounterexampleKnowledgeGraph":
        return cls(tuple(values))

    def add(
        self, value: FormalCounterexample
    ) -> "CounterexampleKnowledgeGraph":
        return CounterexampleKnowledgeGraph(
            self.counterexamples + (value,), self.nodes, self.edges
        )

    def neighbors(self, node_id: str) -> tuple[CounterexampleGraphNode, ...]:
        targets = {
            edge.target_id
            for edge in self.edges
            if edge.source_id == node_id
        } | {
            edge.source_id
            for edge in self.edges
            if edge.target_id == node_id
        }
        by_id = {node.node_id: node for node in self.nodes}
        return tuple(by_id[item] for item in sorted(targets))

    def counterexamples_for(
        self, *node_ids: str
    ) -> tuple[FormalCounterexample, ...]:
        selected = {item for item in node_ids if item}
        if not selected:
            return self.counterexamples
        counterexample_ids = {
            edge.source_id
            for edge in self.edges
            if edge.target_id in selected
        }
        return tuple(
            item
            for item in self.counterexamples
            if item.semantic_id in counterexample_ids
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "counterexample_version": FORMAL_COUNTEREXAMPLE_VERSION,
            "counterexamples": tuple(item.to_record() for item in self.counterexamples),
            "nodes": tuple(item.to_dict() for item in self.nodes),
            "edges": tuple(item.to_dict() for item in self.edges),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterexampleKnowledgeGraph":
        _schema(payload, cls.SCHEMA)
        result = cls(
            counterexamples=tuple(
                FormalCounterexample.from_dict(item)
                for item in payload.get("counterexamples") or ()
                if isinstance(item, Mapping)
            ),
            nodes=tuple(
                CounterexampleGraphNode.from_dict(item)
                for item in payload.get("nodes") or ()
                if isinstance(item, Mapping)
            ),
            edges=tuple(
                CounterexampleGraphEdge.from_dict(item)
                for item in payload.get("edges") or ()
                if isinstance(item, Mapping)
            ),
        )
        _claimed_identity(payload, result.content_id, "content_id")
        return result


def build_counterexample_graph(
    values: Iterable[FormalCounterexample],
) -> CounterexampleKnowledgeGraph:
    return CounterexampleKnowledgeGraph.from_counterexamples(values)


@dataclass(frozen=True)
class CounterexampleCapsuleUsage:
    counterexamples: int
    graph_nodes: int
    graph_edges: int
    encoded_bytes: int
    omitted_counterexamples: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CounterexampleValidationError(
                    f"capsule usage {name} must be a non-negative integer"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "counterexamples": self.counterexamples,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "encoded_bytes": self.encoded_bytes,
            "omitted_counterexamples": self.omitted_counterexamples,
        }


@dataclass(frozen=True)
class CounterexampleContextCapsule(CanonicalContract):
    """A finite graph neighborhood safe to include in a model prompt."""

    SCHEMA = COUNTEREXAMPLE_CAPSULE_SCHEMA

    target_ids: tuple[str, ...]
    counterexamples: tuple[Mapping[str, Any], ...]
    nodes: tuple[CounterexampleGraphNode, ...]
    edges: tuple[CounterexampleGraphEdge, ...]
    usage: CounterexampleCapsuleUsage
    byte_limit: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", _ids(self.target_ids, "target_ids"))
        if any(not isinstance(item, Mapping) for item in self.counterexamples):
            raise CounterexampleValidationError(
                "capsule counterexamples must be public projections"
            )
        if any(not isinstance(item, CounterexampleGraphNode) for item in self.nodes):
            raise CounterexampleValidationError("capsule nodes are malformed")
        if any(not isinstance(item, CounterexampleGraphEdge) for item in self.edges):
            raise CounterexampleValidationError("capsule edges are malformed")
        if not isinstance(self.usage, CounterexampleCapsuleUsage):
            raise CounterexampleValidationError("capsule usage is malformed")
        if (
            isinstance(self.byte_limit, bool)
            or not isinstance(self.byte_limit, int)
            or self.byte_limit < 1024
        ):
            raise CounterexampleValidationError(
                "capsule byte_limit must be at least 1024"
            )
        node_ids = {item.node_id for item in self.nodes}
        if any(
            edge.source_id not in node_ids or edge.target_id not in node_ids
            for edge in self.edges
        ):
            raise CounterexampleValidationError(
                "every capsule edge endpoint must have a node"
            )
        if (
            self.usage.counterexamples != len(self.counterexamples)
            or self.usage.graph_nodes != len(self.nodes)
            or self.usage.graph_edges != len(self.edges)
        ):
            raise CounterexampleValidationError(
                "capsule usage does not match retained contents"
            )
        serialized = json.dumps(
            list(self.counterexamples), sort_keys=True, separators=(",", ":")
        ).lower()
        forbidden = (
            "hidden_witness",
            "private_witness",
            "private_inputs",
            "api_key",
            "access_token",
            "raw_output",
            "prover_output",
            "source_excerpt",
            "source_code",
        )
        if any(marker in serialized for marker in forbidden):
            raise CounterexampleValidationError(
                "unsafe field entered a counterexample capsule"
            )

    @property
    def byte_size(self) -> int:
        return len(self.canonical_bytes())

    def _payload(self) -> dict[str, Any]:
        return {
            "counterexample_version": FORMAL_COUNTEREXAMPLE_VERSION,
            "target_ids": self.target_ids,
            "counterexamples": self.counterexamples,
            "nodes": tuple(item.to_dict() for item in self.nodes),
            "edges": tuple(item.to_dict() for item in self.edges),
            "usage": self.usage.to_dict(),
            "limits": {"max_bytes": self.byte_limit},
            "minimized": True,
            "redacted": True,
            "contains_private_material": False,
            "contains_raw_prover_output": False,
            "contains_source": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterexampleContextCapsule":
        _schema(payload, cls.SCHEMA)
        usage = payload.get("usage")
        limits = payload.get("limits")
        if not isinstance(usage, Mapping) or not isinstance(limits, Mapping):
            raise CounterexampleValidationError(
                "capsule usage and limits are required"
            )
        result = cls(
            target_ids=tuple(payload.get("target_ids") or ()),
            counterexamples=tuple(
                dict(item)
                for item in payload.get("counterexamples") or ()
                if isinstance(item, Mapping)
            ),
            nodes=tuple(
                CounterexampleGraphNode.from_dict(item)
                for item in payload.get("nodes") or ()
                if isinstance(item, Mapping)
            ),
            edges=tuple(
                CounterexampleGraphEdge.from_dict(item)
                for item in payload.get("edges") or ()
                if isinstance(item, Mapping)
            ),
            usage=CounterexampleCapsuleUsage(
                counterexamples=usage.get("counterexamples", 0),
                graph_nodes=usage.get("graph_nodes", 0),
                graph_edges=usage.get("graph_edges", 0),
                encoded_bytes=usage.get("encoded_bytes", 0),
                omitted_counterexamples=usage.get("omitted_counterexamples", 0),
            ),
            byte_limit=limits.get("max_bytes", 0),
        )
        if result.byte_size > result.byte_limit:
            raise CounterexampleBudgetError("decoded capsule exceeds its byte limit")
        if result.usage.encoded_bytes != result.byte_size:
            raise CounterexampleValidationError(
                "capsule encoded byte measurement does not match"
            )
        _claimed_identity(payload, result.content_id, "content_id")
        return result


def _capsule_with_usage(
    *,
    targets: tuple[str, ...],
    entries: list[Mapping[str, Any]],
    nodes: list[CounterexampleGraphNode],
    edges: list[CounterexampleGraphEdge],
    byte_limit: int,
    omitted: int,
) -> CounterexampleContextCapsule:
    encoded_bytes = 0
    capsule: CounterexampleContextCapsule | None = None
    for _ in range(5):
        capsule = CounterexampleContextCapsule(
            target_ids=targets,
            counterexamples=tuple(entries),
            nodes=tuple(nodes),
            edges=tuple(edges),
            usage=CounterexampleCapsuleUsage(
                len(entries), len(nodes), len(edges), encoded_bytes, omitted
            ),
            byte_limit=byte_limit,
        )
        measured = capsule.byte_size
        if measured == encoded_bytes:
            return capsule
        encoded_bytes = measured
    assert capsule is not None
    return capsule


def build_counterexample_context_capsule(
    source: CounterexampleKnowledgeGraph | Iterable[FormalCounterexample],
    *,
    target_ids: Iterable[str] = (),
    limits: CounterexampleLimits | None = None,
) -> CounterexampleContextCapsule:
    """Select, minimize, and byte-bound a model-facing graph neighborhood."""

    active_limits = limits or CounterexampleLimits()
    graph = (
        source
        if isinstance(source, CounterexampleKnowledgeGraph)
        else CounterexampleKnowledgeGraph.from_counterexamples(source)
    )
    targets = _ids(tuple(target_ids), "target_ids")
    selected = list(graph.counterexamples_for(*targets))
    total_selected = len(selected)
    selected = selected[: active_limits.max_capsule_counterexamples]
    entries: list[Mapping[str, Any]] = [
        item.to_capsule_dict() for item in selected
    ]
    selected_ids = {item.semantic_id for item in selected}
    edges = [
        edge
        for edge in graph.edges
        if edge.source_id in selected_ids
        and (not targets or edge.target_id in set(targets))
    ][: active_limits.max_graph_edges]
    endpoint_ids = selected_ids | {
        value
        for edge in edges
        for value in (edge.source_id, edge.target_id)
    }
    nodes = [
        node for node in graph.nodes if node.node_id in endpoint_ids
    ][: active_limits.max_graph_nodes]
    omitted = total_selected - len(entries)

    while True:
        capsule = _capsule_with_usage(
            targets=targets,
            entries=entries,
            nodes=nodes,
            edges=edges,
            byte_limit=active_limits.max_capsule_bytes,
            omitted=omitted,
        )
        if capsule.byte_size <= active_limits.max_capsule_bytes:
            return capsule
        if edges:
            edges.pop()
            continue
        if nodes:
            nodes.pop()
            continue
        if len(entries) > 1:
            entries.pop()
            omitted += 1
            continue
        if (
            entries
            and "payload" in entries[0]
            and not (
                isinstance(entries[0].get("payload"), Mapping)
                and entries[0]["payload"].get("omitted")
                == "<capsule-byte-limit>"
            )
        ):
            compact = dict(entries[0])
            payload_bytes = canonical_json_bytes(compact["payload"])
            compact["payload"] = {
                "omitted": "<capsule-byte-limit>",
                "public_digest": (
                    "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
                ),
            }
            compact["truncated"] = True
            entries[0] = compact
            continue
        if entries:
            entries.clear()
            omitted += 1
            continue
        raise CounterexampleBudgetError(
            "counterexample capsule metadata exceeds its byte limit"
        )


build_counterexample_capsule = build_counterexample_context_capsule
assemble_counterexample_context = build_counterexample_context_capsule


def _bounded_jsonl_lines(stream: Any, maximum_chars: int) -> Iterable[str]:
    """Yield bounded complete lines and drain oversized records in chunks."""

    while True:
        line = stream.readline(maximum_chars + 1)
        if not line:
            return
        if len(line) <= maximum_chars and line.endswith("\n"):
            yield line
            continue
        if len(line) <= maximum_chars:
            # A final line without a newline is still a complete JSON record.
            yield line
            return
        while line and not line.endswith("\n"):
            line = stream.readline(maximum_chars + 1)


class CounterexampleStore:
    """Append-only JSONL store which accepts only normalized public records."""

    def __init__(
        self,
        path: Path | str,
        *,
        limits: CounterexampleLimits | None = None,
    ) -> None:
        self.path = Path(path)
        self.limits = limits or CounterexampleLimits()

    def _record(self, value: FormalCounterexample) -> dict[str, Any]:
        result = {
            "schema": COUNTEREXAMPLE_STORE_SCHEMA,
            "counterexample_version": FORMAL_COUNTEREXAMPLE_VERSION,
            "counterexample_id": value.semantic_id,
            "counterexample": value.to_record(),
        }
        if len(canonical_json_bytes(result)) > self.limits.max_counterexample_bytes:
            raise CounterexampleBudgetError(
                "counterexample store record exceeds persistence byte limit"
            )
        return result

    def persist(
        self,
        value: Any,
        **normalization: Any,
    ) -> tuple[FormalCounterexample, bool]:
        """Normalize and append once; return ``(counterexample, inserted)``."""

        counterexample = normalize_counterexample(
            value, limits=self.limits, **normalization
        )
        record = self._record(counterexample)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.seek(0)
                for line in _bounded_jsonl_lines(
                    stream, self.limits.max_counterexample_bytes
                ):
                    if counterexample.semantic_id not in line:
                        continue
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(existing, Mapping)
                        and existing.get("counterexample_id")
                        == counterexample.semantic_id
                    ):
                        return counterexample, False
                stream.seek(0, os.SEEK_END)
                stream.write(
                    canonical_json_bytes(record).decode("utf-8") + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return counterexample, True

    def load(
        self, *, maximum_records: int = 4096
    ) -> tuple[FormalCounterexample, ...]:
        if (
            isinstance(maximum_records, bool)
            or not isinstance(maximum_records, int)
            or maximum_records <= 0
        ):
            raise CounterexampleValidationError(
                "maximum_records must be a positive integer"
            )
        if not self.path.exists() or self.path.is_dir():
            return ()
        values: list[FormalCounterexample] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line in _bounded_jsonl_lines(
                stream, self.limits.max_counterexample_bytes
            ):
                if len(values) >= maximum_records:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, Mapping):
                    continue
                payload = record.get("counterexample")
                if not isinstance(payload, Mapping):
                    continue
                try:
                    values.append(FormalCounterexample.from_dict(payload))
                except CounterexampleValidationError:
                    continue
        return deduplicate_counterexamples(values)

    def persist_many(
        self, values: Iterable[Any], **normalization: Any
    ) -> tuple[tuple[FormalCounterexample, ...], int]:
        """Persist a stream and return normalized values plus insertion count."""

        retained: list[FormalCounterexample] = []
        inserted = 0
        for value in values:
            counterexample, was_inserted = self.persist(
                value, **normalization
            )
            retained.append(counterexample)
            inserted += int(was_inserted)
        return deduplicate_counterexamples(retained), inserted

    def load_graph(
        self, *, maximum_records: int = 4096
    ) -> CounterexampleKnowledgeGraph:
        """Reconstruct the canonical graph projection from durable bindings."""

        return CounterexampleKnowledgeGraph.from_counterexamples(
            self.load(maximum_records=maximum_records)
        )


def persist_counterexample(
    path: Path | str,
    value: Any,
    *,
    limits: CounterexampleLimits | None = None,
    **normalization: Any,
) -> tuple[FormalCounterexample, bool]:
    return CounterexampleStore(path, limits=limits).persist(
        value, **normalization
    )


def load_counterexamples(
    path: Path | str,
    *,
    limits: CounterexampleLimits | None = None,
    maximum_records: int = 4096,
) -> tuple[FormalCounterexample, ...]:
    return CounterexampleStore(path, limits=limits).load(
        maximum_records=maximum_records
    )


# Friendly architectural names.
FormalCounterexampleGraph = CounterexampleKnowledgeGraph
FormalCounterexampleContextCapsule = CounterexampleContextCapsule
CounterexampleGraph = CounterexampleKnowledgeGraph
CounterexampleCapsule = CounterexampleContextCapsule


__all__ = [
    "ABSOLUTE_MAX_COUNTEREXAMPLE_BYTES",
    "COUNTEREXAMPLE_CAPSULE_SCHEMA",
    "COUNTEREXAMPLE_GRAPH_SCHEMA",
    "COUNTEREXAMPLE_STORE_SCHEMA",
    "DEFAULT_MAX_CAPSULE_BYTES",
    "DEFAULT_MAX_CAPSULE_COUNTEREXAMPLES",
    "DEFAULT_MAX_COLLECTION_ITEMS",
    "DEFAULT_MAX_COUNTEREXAMPLE_BYTES",
    "DEFAULT_MAX_GRAPH_EDGES",
    "DEFAULT_MAX_GRAPH_NODES",
    "DEFAULT_MAX_NESTING_DEPTH",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "DEFAULT_MAX_TEXT_CHARS",
    "DEFAULT_MAX_TRACE_STEPS",
    "FORMAL_COUNTEREXAMPLE_SCHEMA",
    "FORMAL_COUNTEREXAMPLE_VERSION",
    "ConfidentialityDisposition",
    "CounterexampleBindings",
    "CounterexampleBudgetError",
    "CounterexampleCapsule",
    "CounterexampleCapsuleUsage",
    "CounterexampleContextCapsule",
    "CounterexampleEdgeKind",
    "CounterexampleGraph",
    "CounterexampleGraphEdge",
    "CounterexampleGraphNode",
    "CounterexampleKnowledgeGraph",
    "CounterexampleKind",
    "CounterexampleLimits",
    "CounterexampleNodeKind",
    "CounterexampleStore",
    "CounterexampleValidationError",
    "FormalCounterexample",
    "FormalCounterexampleContextCapsule",
    "FormalCounterexampleGraph",
    "FormalCounterexampleNormalizer",
    "RedactionReport",
    "RepairClass",
    "assemble_counterexample_context",
    "build_counterexample_capsule",
    "build_counterexample_context_capsule",
    "build_counterexample_graph",
    "deduplicate_counterexamples",
    "load_counterexamples",
    "normalize_counterexample",
    "normalize_dcec_contradiction",
    "normalize_hypertrace",
    "normalize_kernel_error",
    "normalize_proof_failure",
    "normalize_protocol_attack",
    "normalize_runtime_mtl_violation",
    "normalize_smt_model",
    "normalize_tdfol_contradiction",
    "normalize_tla_trace",
    "normalize_unsat_core",
    "persist_counterexample",
]
