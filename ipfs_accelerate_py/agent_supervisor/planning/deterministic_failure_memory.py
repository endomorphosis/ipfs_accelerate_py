"""DCR-063: typed failure memory and non-thrashing replan decisions.

Interfaces
----------
* ``FailureMemory@1`` — durable, content-addressed memory of typed repair
  failures (stale, conflict, validation, proof, resource, capability).
* ``ReplanDecision@1`` — fail-closed decision that either authorizes one
  deterministic retry/rescue or emits no work.

Normative rules (fail-closed)
-----------------------------
* Replaying unchanged inputs emits no duplicate work.
* Retry is allowed only on typed new evidence **or** a strictly decreasing
  :class:`RetryMeasure`.
* Retry/rescue cannot route to a provider/model node.
* A refuted candidate CID cannot be selected again.
* Counterexamples and policy bindings are never erased or relaxed.
* Runtime model calls remain 0; write authority is never granted.

Predicted symbols: :class:`FailureMemory`, :class:`RetryMeasure`,
:class:`FailureAttempt`, :class:`FailureMemoryReceipt`,
:func:`decide_replan`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    canonical_json,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interfaces / evidence / schemas
# ---------------------------------------------------------------------------

FAILURE_MEMORY_INTERFACE: Final[str] = "FailureMemory@1"
REPLAN_DECISION_INTERFACE: Final[str] = "ReplanDecision@1"
DCR_REPLAN_EVIDENCE: Final[str] = "dcr/replan@1"
DETERMINISTIC_FAILURE_MEMORY_VERSION: Final[int] = 1

FAILURE_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-failure-memory@1"
)
FAILURE_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-failure-attempt@1"
)
RETRY_MEASURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-retry-measure@1"
)
REPLAN_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-replan-decision@1"
)
FAILURE_MEMORY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-failure-memory-receipt@1"
)
DEFAULT_REPLAN_FIXTURES_REL: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/replan-fixtures.json"
)

MAX_ATTEMPTS: Final[int] = 4_096
MAX_REFUTED: Final[int] = 4_096
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_MEASURE_COMPONENT: Final[int] = 1_000_000

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@=-]{0,255}$")

_PROVIDER_ROUTE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "provider",
        "model",
        "llm",
        "language_model",
        "codex",
        "grok",
        "openai",
        "anthropic",
        "chat_completion",
        "completion_provider",
    }
)


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class FailureMemoryError(ContractValidationError):
    """Malformed failure-memory input or closed-boundary violation."""


class FailureClass(str, Enum):
    """Closed failure taxonomy persisted across restart."""

    STALE = "stale"
    CONFLICT = "conflict"
    VALIDATION = "validation"
    PROOF = "proof"
    RESOURCE = "resource"
    CAPABILITY = "capability"


class ReplanDisposition(str, Enum):
    """Closed outcomes for one replan decision."""

    RETRY_NEW_EVIDENCE = "retry_new_evidence"
    RETRY_STRICTLY_DECREASING_MEASURE = "retry_strictly_decreasing_measure"
    NO_DUPLICATE_WORK = "no_duplicate_work"
    REFUTED_CANDIDATE = "refuted_candidate"
    PROVIDER_ROUTE_FORBIDDEN = "provider_route_forbidden"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    MEMORY_BOUND_REACHED = "memory_bound_reached"
    ABSTAIN = "abstain"


class AttemptRouteKind(str, Enum):
    """Closed route vocabulary; provider/model routes never authorize retry."""

    DETERMINISTIC_OPERATOR = "deterministic_operator"
    DETERMINISTIC_REPLAN = "deterministic_replan"
    RESCUE_OPERATOR = "rescue_operator"
    ANALYSIS_PROBE = "analysis_probe"
    ABSTAIN = "abstain"
    PROVIDER = "provider"
    MODEL = "model"
    LLM = "llm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        if required:
            raise FailureMemoryError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise FailureMemoryError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise FailureMemoryError(f"{name} is required")
    if "\x00" in text:
        raise FailureMemoryError(f"{name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise FailureMemoryError(f"{name} exceeds the {limit}-byte bound")
    return text


def _identifier(value: Any, name: str, *, allow_empty: bool = False) -> str:
    text = _text(value, name, required=not allow_empty)
    if not text and allow_empty:
        return ""
    if not _IDENTIFIER.fullmatch(text):
        raise FailureMemoryError(f"{name} must be a bounded typed identifier")
    return text


def _identifiers(
    values: Iterable[Any],
    name: str,
    *,
    allow_empty: bool = True,
    limit: int = MAX_REFUTED,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise FailureMemoryError(f"{name} must be an array")
    result = tuple(sorted({_identifier(item, name) for item in values}))
    if len(result) > limit:
        raise FailureMemoryError(f"{name} exceeds the {limit}-identifier bound")
    if not result and not allow_empty:
        raise FailureMemoryError(f"{name} must not be empty")
    return result


def _non_negative(value: Any, name: str, *, maximum: int = MAX_MEASURE_COMPONENT) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FailureMemoryError(f"{name} must be a non-negative integer")
    if value > maximum:
        raise FailureMemoryError(f"{name} exceeds the {maximum} bound")
    return value


def _positive(value: Any, name: str, *, maximum: int = MAX_ATTEMPTS) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FailureMemoryError(f"{name} must be a positive integer")
    if value > maximum:
        raise FailureMemoryError(f"{name} exceeds the {maximum} bound")
    return value


def _route_kind(value: Any) -> AttemptRouteKind:
    text = _text(value, "route_kind").lower().replace("-", "_")
    try:
        return AttemptRouteKind(text)
    except ValueError as exc:
        # Normalize common aliases without inventing open vocabulary.
        aliases = {
            "provider_route": AttemptRouteKind.PROVIDER,
            "model_route": AttemptRouteKind.MODEL,
            "llm_route": AttemptRouteKind.LLM,
            "provider_node": AttemptRouteKind.PROVIDER,
            "model_node": AttemptRouteKind.MODEL,
            "language_model": AttemptRouteKind.MODEL,
            "deterministic": AttemptRouteKind.DETERMINISTIC_OPERATOR,
            "operator": AttemptRouteKind.DETERMINISTIC_OPERATOR,
            "rescue": AttemptRouteKind.RESCUE_OPERATOR,
            "probe": AttemptRouteKind.ANALYSIS_PROBE,
            "replan": AttemptRouteKind.DETERMINISTIC_REPLAN,
        }
        if text in aliases:
            return aliases[text]
        raise FailureMemoryError(f"unsupported route_kind: {text}") from exc


def is_provider_or_model_route(route_kind: AttemptRouteKind | str) -> bool:
    """Return True when a route would require a provider/model node."""

    kind = (
        route_kind
        if isinstance(route_kind, AttemptRouteKind)
        else _route_kind(route_kind)
    )
    if kind in {
        AttemptRouteKind.PROVIDER,
        AttemptRouteKind.MODEL,
        AttemptRouteKind.LLM,
    }:
        return True
    token = kind.value
    return any(marker in token for marker in _PROVIDER_ROUTE_MARKERS)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryMeasure:
    """Lexicographic progress measure for non-thrashing retries.

    A retry without new evidence is authorized only when the proposed measure
    is **strictly less** than the prior measure in dictionary order.  No
    component may increase while claiming progress.
    """

    open_counterexamples: int = 0
    validation_findings: int = 0
    remaining_candidates: int = 0
    resource_debt: int = 0
    capability_gaps: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _non_negative(getattr(self, name), name),
            )

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.open_counterexamples,
            self.validation_findings,
            self.remaining_candidates,
            self.resource_debt,
            self.capability_gaps,
        )

    def strictly_decreases(self, prior: "RetryMeasure") -> bool:
        if not isinstance(prior, RetryMeasure):
            raise FailureMemoryError("prior measure must be RetryMeasure")
        return self.as_tuple() < prior.as_tuple()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RETRY_MEASURE_SCHEMA,
            "open_counterexamples": self.open_counterexamples,
            "validation_findings": self.validation_findings,
            "remaining_candidates": self.remaining_candidates,
            "resource_debt": self.resource_debt,
            "capability_gaps": self.capability_gaps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "RetryMeasure":
        value = dict(payload or {})
        value.pop("schema", None)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise FailureMemoryError(
                "unknown retry measure fields: " + ", ".join(unknown)
            )
        return cls(**{name: value.get(name, 0) for name in allowed})


@dataclass(frozen=True)
class FailureAttempt:
    """One typed failure observation bound to a prior candidate.

    Delivery noise (timestamps, transport ids) is excluded from
    :attr:`attempt_key` so replaying the same semantic inputs collapses to the
    same attempt identity.
    """

    failure_class: FailureClass
    prior_candidate_cid: str
    evidence_cid: str
    measure: RetryMeasure
    route_kind: AttemptRouteKind = AttemptRouteKind.DETERMINISTIC_OPERATOR
    scope_id: str = "scope:default"
    plan_id: str = "plan:default"
    counterexample_ids: tuple[str, ...] = ()
    refuted: bool = True
    operator_kind: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_class", FailureClass(self.failure_class))
        object.__setattr__(
            self,
            "prior_candidate_cid",
            _identifier(self.prior_candidate_cid, "prior_candidate_cid"),
        )
        object.__setattr__(
            self, "evidence_cid", _identifier(self.evidence_cid, "evidence_cid")
        )
        measure = (
            self.measure
            if isinstance(self.measure, RetryMeasure)
            else RetryMeasure.from_dict(self.measure)
        )
        object.__setattr__(self, "measure", measure)
        object.__setattr__(self, "route_kind", _route_kind(self.route_kind))
        object.__setattr__(
            self, "scope_id", _identifier(self.scope_id, "scope_id")
        )
        object.__setattr__(
            self, "plan_id", _identifier(self.plan_id, "plan_id")
        )
        object.__setattr__(
            self,
            "counterexample_ids",
            _identifiers(self.counterexample_ids, "counterexample_ids"),
        )
        if not isinstance(self.refuted, bool):
            raise FailureMemoryError("refuted must be boolean")
        operator = _text(self.operator_kind, "operator_kind", required=False)
        if operator and not _IDENTIFIER.fullmatch(operator):
            raise FailureMemoryError("operator_kind must be a bounded typed identifier")
        object.__setattr__(self, "operator_kind", operator)

    @property
    def attempt_key(self) -> str:
        """Semantic attempt identity independent of delivery noise.

        Counterexample identifiers are retained on the attempt for audit but
        intentionally excluded from the key so accumulating open witnesses
        cannot thrash identity or rewrite durable memory slots.
        """

        return content_identity(
            {
                "schema": FAILURE_ATTEMPT_SCHEMA,
                "failure_class": self.failure_class.value,
                "prior_candidate_cid": self.prior_candidate_cid,
                "evidence_cid": self.evidence_cid,
                "measure": self.measure.to_dict(),
                "route_kind": self.route_kind.value,
                "scope_id": self.scope_id,
                "plan_id": self.plan_id,
                "refuted": self.refuted,
                "operator_kind": self.operator_kind,
            }
        )

    @property
    def is_provider_route(self) -> bool:
        return is_provider_or_model_route(self.route_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FAILURE_ATTEMPT_SCHEMA,
            "attempt_key": self.attempt_key,
            "failure_class": self.failure_class.value,
            "prior_candidate_cid": self.prior_candidate_cid,
            "evidence_cid": self.evidence_cid,
            "measure": self.measure.to_dict(),
            "route_kind": self.route_kind.value,
            "scope_id": self.scope_id,
            "plan_id": self.plan_id,
            "counterexample_ids": list(self.counterexample_ids),
            "refuted": self.refuted,
            "operator_kind": self.operator_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureAttempt":
        if not isinstance(payload, Mapping):
            raise FailureMemoryError("failure attempt must be an object")
        allowed = {
            "schema",
            "attempt_key",
            "failure_class",
            "prior_candidate_cid",
            "evidence_cid",
            "measure",
            "route_kind",
            "scope_id",
            "plan_id",
            "counterexample_ids",
            "refuted",
            "operator_kind",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise FailureMemoryError(
                "unknown failure attempt fields: " + ", ".join(unknown)
            )
        result = cls(
            failure_class=payload.get("failure_class", FailureClass.VALIDATION),
            prior_candidate_cid=payload.get("prior_candidate_cid", ""),
            evidence_cid=payload.get("evidence_cid", ""),
            measure=payload.get("measure") or {},
            route_kind=payload.get(
                "route_kind", AttemptRouteKind.DETERMINISTIC_OPERATOR
            ),
            scope_id=payload.get("scope_id", "scope:default"),
            plan_id=payload.get("plan_id", "plan:default"),
            counterexample_ids=tuple(payload.get("counterexample_ids") or ()),
            refuted=bool(payload.get("refuted", True)),
            operator_kind=str(payload.get("operator_kind") or ""),
        )
        claimed = str(payload.get("attempt_key") or "")
        if claimed and claimed != result.attempt_key:
            raise FailureMemoryError(
                "failure attempt identity does not match content"
            )
        return result


@dataclass(frozen=True)
class FailureMemoryPolicy:
    """Finite bounds for durable failure learning."""

    max_attempts: int = 1_024
    max_refuted_candidates: int = 1_024
    max_retries_per_attempt_key: int = 8
    max_retries_per_scope: int = 64

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _positive(getattr(self, name), name)
            )
        if self.max_refuted_candidates > self.max_attempts:
            # Refuted set may equal attempts; never exceed memory capacity.
            pass

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "FailureMemoryPolicy":
        value = dict(payload or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise FailureMemoryError(
                "unknown failure memory policy fields: " + ", ".join(unknown)
            )
        defaults = cls()
        return cls(
            **{
                name: value.get(name, getattr(defaults, name))
                for name in allowed
            }
        )


@dataclass(frozen=True)
class ReplanDecision:
    """Fail-closed decision for one retry/rescue evaluation."""

    SCHEMA: ClassVar[str] = REPLAN_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = REPLAN_DECISION_INTERFACE

    disposition: ReplanDisposition
    attempt_key: str
    failure_class: FailureClass
    prior_candidate_cid: str
    evidence_cid: str
    measure: RetryMeasure
    should_replan: bool
    emits_work: bool
    allows_provider_route: bool
    selected_candidate_cid: str = ""
    refuted_candidate_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    runtime_model_calls: int = 0
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", ReplanDisposition(self.disposition)
        )
        object.__setattr__(
            self, "attempt_key", _identifier(self.attempt_key, "attempt_key")
        )
        object.__setattr__(
            self, "failure_class", FailureClass(self.failure_class)
        )
        object.__setattr__(
            self,
            "prior_candidate_cid",
            _identifier(self.prior_candidate_cid, "prior_candidate_cid"),
        )
        object.__setattr__(
            self, "evidence_cid", _identifier(self.evidence_cid, "evidence_cid")
        )
        measure = (
            self.measure
            if isinstance(self.measure, RetryMeasure)
            else RetryMeasure.from_dict(self.measure)
        )
        object.__setattr__(self, "measure", measure)
        for name in ("should_replan", "emits_work", "allows_provider_route", "grants_write_authority"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise FailureMemoryError(f"{name} must be boolean")
        # Hard invariants for the non-thrashing contract.
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "allows_provider_route", False)
        if self.should_replan and not self.emits_work:
            raise FailureMemoryError(
                "should_replan requires emits_work"
            )
        if not self.should_replan and self.emits_work:
            raise FailureMemoryError(
                "emits_work requires should_replan"
            )
        if self.disposition in {
            ReplanDisposition.NO_DUPLICATE_WORK,
            ReplanDisposition.REFUTED_CANDIDATE,
            ReplanDisposition.PROVIDER_ROUTE_FORBIDDEN,
            ReplanDisposition.RETRY_BUDGET_EXHAUSTED,
            ReplanDisposition.MEMORY_BOUND_REACHED,
            ReplanDisposition.ABSTAIN,
        } and self.should_replan:
            raise FailureMemoryError(
                f"{self.disposition.value} cannot authorize replan work"
            )
        if self.disposition in {
            ReplanDisposition.RETRY_NEW_EVIDENCE,
            ReplanDisposition.RETRY_STRICTLY_DECREASING_MEASURE,
        } and not self.should_replan:
            raise FailureMemoryError(
                f"{self.disposition.value} must authorize replan work"
            )
        selected = _identifier(
            self.selected_candidate_cid,
            "selected_candidate_cid",
            allow_empty=True,
        )
        object.__setattr__(self, "selected_candidate_cid", selected)
        refuted = _identifiers(
            self.refuted_candidate_cids, "refuted_candidate_cids"
        )
        object.__setattr__(self, "refuted_candidate_cids", refuted)
        if selected and selected in refuted:
            raise FailureMemoryError(
                "selected candidate cannot be a previously refuted candidate"
            )
        reasons = _identifiers(self.reason_codes, "reason_codes", allow_empty=True)
        object.__setattr__(self, "reason_codes", reasons)

    @property
    def decision_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def evidence_subset(self) -> dict[str, Any]:
        """Compact evidence projection required by DCR-063."""

        return {
            "evidence_id": DCR_REPLAN_EVIDENCE,
            "attempt_key": self.attempt_key,
            "failure_class": self.failure_class.value,
            "prior_candidate": self.prior_candidate_cid,
            "new_evidence": self.evidence_cid,
            "measure": self.measure.to_dict(),
            "disposition": self.disposition.value,
            "should_replan": self.should_replan,
            "emits_work": self.emits_work,
            "allows_provider_route": self.allows_provider_route,
            "selected_candidate_cid": self.selected_candidate_cid,
            "refuted_candidate_cids": list(self.refuted_candidate_cids),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
        }

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": REPLAN_DECISION_SCHEMA,
            "interface": REPLAN_DECISION_INTERFACE,
            "disposition": self.disposition.value,
            "attempt_key": self.attempt_key,
            "failure_class": self.failure_class.value,
            "prior_candidate_cid": self.prior_candidate_cid,
            "evidence_cid": self.evidence_cid,
            "measure": self.measure.to_dict(),
            "should_replan": self.should_replan,
            "emits_work": self.emits_work,
            "allows_provider_route": False,
            "selected_candidate_cid": self.selected_candidate_cid,
            "refuted_candidate_cids": list(self.refuted_candidate_cids),
            "reason_codes": list(self.reason_codes),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
        }
        if include_identity:
            payload["decision_id"] = self.decision_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplanDecision":
        if not isinstance(payload, Mapping):
            raise FailureMemoryError("replan decision must be an object")
        allowed = {
            "schema",
            "interface",
            "decision_id",
            "disposition",
            "attempt_key",
            "failure_class",
            "prior_candidate_cid",
            "evidence_cid",
            "measure",
            "should_replan",
            "emits_work",
            "allows_provider_route",
            "selected_candidate_cid",
            "refuted_candidate_cids",
            "reason_codes",
            "runtime_model_calls",
            "grants_write_authority",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise FailureMemoryError(
                "unknown replan decision fields: " + ", ".join(unknown)
            )
        if payload.get("schema") not in {None, "", REPLAN_DECISION_SCHEMA}:
            raise FailureMemoryError("unsupported replan decision schema")
        if payload.get("interface") not in {None, "", REPLAN_DECISION_INTERFACE}:
            raise FailureMemoryError("unsupported replan decision interface")
        result = cls(
            disposition=payload.get("disposition", ReplanDisposition.ABSTAIN),
            attempt_key=payload.get("attempt_key", ""),
            failure_class=payload.get("failure_class", FailureClass.VALIDATION),
            prior_candidate_cid=payload.get("prior_candidate_cid", ""),
            evidence_cid=payload.get("evidence_cid", ""),
            measure=payload.get("measure") or {},
            should_replan=bool(payload.get("should_replan", False)),
            emits_work=bool(payload.get("emits_work", False)),
            allows_provider_route=False,
            selected_candidate_cid=str(payload.get("selected_candidate_cid") or ""),
            refuted_candidate_cids=tuple(
                payload.get("refuted_candidate_cids") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = str(payload.get("decision_id") or "")
        if claimed and claimed != result.decision_id:
            raise FailureMemoryError(
                "replan decision identity does not match content"
            )
        return result


@dataclass(frozen=True)
class FailureMemoryReceipt:
    """Tamper-evident receipt for one recorded attempt and decision."""

    SCHEMA: ClassVar[str] = FAILURE_MEMORY_RECEIPT_SCHEMA

    attempt: FailureAttempt
    decision: ReplanDecision
    memory_state_id: str
    observed_at_milliseconds: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, FailureAttempt):
            if not isinstance(self.attempt, Mapping):
                raise FailureMemoryError("attempt must be FailureAttempt")
            object.__setattr__(
                self, "attempt", FailureAttempt.from_dict(self.attempt)
            )
        if not isinstance(self.decision, ReplanDecision):
            if not isinstance(self.decision, Mapping):
                raise FailureMemoryError("decision must be ReplanDecision")
            object.__setattr__(
                self, "decision", ReplanDecision.from_dict(self.decision)
            )
        object.__setattr__(
            self,
            "memory_state_id",
            _identifier(self.memory_state_id, "memory_state_id"),
        )
        object.__setattr__(
            self,
            "observed_at_milliseconds",
            _positive(self.observed_at_milliseconds, "observed_at_milliseconds", maximum=10**15),
        )
        if self.decision.attempt_key != self.attempt.attempt_key:
            raise FailureMemoryError(
                "receipt decision attempt_key must match the attempt"
            )
        if self.decision.prior_candidate_cid != self.attempt.prior_candidate_cid:
            raise FailureMemoryError(
                "receipt decision prior candidate must match the attempt"
            )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def evidence_subset(self) -> dict[str, Any]:
        subset = self.decision.evidence_subset()
        subset["receipt_id"] = self.receipt_id
        subset["memory_state_id"] = self.memory_state_id
        return subset

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": FAILURE_MEMORY_RECEIPT_SCHEMA,
            "attempt": self.attempt.to_dict(),
            "decision": self.decision.to_dict(),
            "memory_state_id": self.memory_state_id,
            "observed_at_milliseconds": self.observed_at_milliseconds,
            "runtime_model_calls": 0,
            "grants_write_authority": False,
        }
        if include_identity:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureMemoryReceipt":
        if not isinstance(payload, Mapping):
            raise FailureMemoryError("failure memory receipt must be an object")
        allowed = {
            "schema",
            "receipt_id",
            "attempt",
            "decision",
            "memory_state_id",
            "observed_at_milliseconds",
            "runtime_model_calls",
            "grants_write_authority",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise FailureMemoryError(
                "unknown failure memory receipt fields: " + ", ".join(unknown)
            )
        if payload.get("schema") not in {None, "", FAILURE_MEMORY_RECEIPT_SCHEMA}:
            raise FailureMemoryError("unsupported failure memory receipt schema")
        result = cls(
            attempt=payload.get("attempt") or {},
            decision=payload.get("decision") or {},
            memory_state_id=payload.get("memory_state_id", ""),
            observed_at_milliseconds=payload.get("observed_at_milliseconds", 1),
        )
        claimed = str(payload.get("receipt_id") or "")
        if claimed and claimed != result.receipt_id:
            raise FailureMemoryError(
                "failure memory receipt identity does not match content"
            )
        return result


@dataclass(frozen=True)
class FailureMemorySnapshot:
    """Canonical durable projection of failure memory state."""

    policy: FailureMemoryPolicy
    attempts: tuple[FailureAttempt, ...]
    refuted_candidate_cids: tuple[str, ...]
    attempt_key_retry_counts: Mapping[str, int] = field(default_factory=dict)
    scope_retry_counts: Mapping[str, int] = field(default_factory=dict)
    evidence_history_by_scope: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    last_measure_by_scope: Mapping[str, RetryMeasure] = field(default_factory=dict)
    counterexample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy, FailureMemoryPolicy):
            if not isinstance(self.policy, Mapping):
                raise FailureMemoryError("snapshot policy is invalid")
            object.__setattr__(
                self, "policy", FailureMemoryPolicy.from_dict(self.policy)
            )
        attempts = tuple(
            item if isinstance(item, FailureAttempt) else FailureAttempt.from_dict(item)
            for item in self.attempts
        )
        keys = [item.attempt_key for item in attempts]
        if len(keys) != len(set(keys)):
            raise FailureMemoryError("failure memory contains duplicate attempts")
        if len(attempts) > self.policy.max_attempts:
            raise FailureMemoryError("failure memory exceeds its attempt bound")
        object.__setattr__(
            self,
            "attempts",
            tuple(sorted(attempts, key=lambda item: item.attempt_key)),
        )
        refuted = _identifiers(
            self.refuted_candidate_cids,
            "refuted_candidate_cids",
            limit=self.policy.max_refuted_candidates,
        )
        object.__setattr__(self, "refuted_candidate_cids", refuted)
        retry_counts = {
            _identifier(key, "attempt_key"): _non_negative(value, "retry_count")
            for key, value in dict(self.attempt_key_retry_counts).items()
        }
        object.__setattr__(
            self, "attempt_key_retry_counts", MappingProxyType(retry_counts)
        )
        scope_counts = {
            _identifier(key, "scope_id"): _non_negative(value, "scope_retry_count")
            for key, value in dict(self.scope_retry_counts).items()
        }
        object.__setattr__(
            self, "scope_retry_counts", MappingProxyType(scope_counts)
        )
        evidence_history = {
            _identifier(key, "scope_id"): _identifiers(values, "evidence_history")
            for key, values in dict(self.evidence_history_by_scope).items()
        }
        object.__setattr__(
            self,
            "evidence_history_by_scope",
            MappingProxyType(evidence_history),
        )
        last_measures = {}
        for key, value in dict(self.last_measure_by_scope).items():
            scope = _identifier(key, "scope_id")
            measure = (
                value
                if isinstance(value, RetryMeasure)
                else RetryMeasure.from_dict(value)
            )
            last_measures[scope] = measure
        object.__setattr__(
            self, "last_measure_by_scope", MappingProxyType(last_measures)
        )
        object.__setattr__(
            self,
            "counterexample_ids",
            _identifiers(self.counterexample_ids, "counterexample_ids"),
        )

    @property
    def state_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": FAILURE_MEMORY_SCHEMA,
            "interface": FAILURE_MEMORY_INTERFACE,
            "memory_version": DETERMINISTIC_FAILURE_MEMORY_VERSION,
            "policy": self.policy.to_dict(),
            "attempts": [item.to_dict() for item in self.attempts],
            "refuted_candidate_cids": list(self.refuted_candidate_cids),
            "attempt_key_retry_counts": dict(self.attempt_key_retry_counts),
            "scope_retry_counts": dict(self.scope_retry_counts),
            "evidence_history_by_scope": {
                key: list(values)
                for key, values in self.evidence_history_by_scope.items()
            },
            "last_measure_by_scope": {
                key: value.to_dict()
                for key, value in self.last_measure_by_scope.items()
            },
            "counterexample_ids": list(self.counterexample_ids),
        }
        if include_identity:
            payload["state_id"] = self.state_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureMemorySnapshot":
        if not isinstance(payload, Mapping):
            raise FailureMemoryError("failure memory snapshot must be an object")
        expected = {
            "schema",
            "interface",
            "memory_version",
            "state_id",
            "policy",
            "attempts",
            "refuted_candidate_cids",
            "attempt_key_retry_counts",
            "scope_retry_counts",
            "evidence_history_by_scope",
            "last_measure_by_scope",
            "counterexample_ids",
        }
        if set(payload) != expected:
            raise FailureMemoryError(
                "failure memory snapshot must use the closed schema"
            )
        if (
            payload.get("schema") != FAILURE_MEMORY_SCHEMA
            or payload.get("interface") != FAILURE_MEMORY_INTERFACE
            or payload.get("memory_version") != DETERMINISTIC_FAILURE_MEMORY_VERSION
        ):
            raise FailureMemoryError(
                "failure memory snapshot version is unsupported"
            )
        original = dict(payload)
        original.pop("state_id", None)
        if payload.get("state_id") != content_identity(original):
            raise FailureMemoryError(
                "failure memory state identity does not match content"
            )
        return cls(
            policy=FailureMemoryPolicy.from_dict(payload.get("policy") or {}),
            attempts=tuple(payload.get("attempts") or ()),
            refuted_candidate_cids=tuple(
                payload.get("refuted_candidate_cids") or ()
            ),
            attempt_key_retry_counts=dict(
                payload.get("attempt_key_retry_counts") or {}
            ),
            scope_retry_counts=dict(payload.get("scope_retry_counts") or {}),
            evidence_history_by_scope={
                key: tuple(values)
                for key, values in dict(
                    payload.get("evidence_history_by_scope") or {}
                ).items()
            },
            last_measure_by_scope=dict(
                payload.get("last_measure_by_scope") or {}
            ),
            counterexample_ids=tuple(payload.get("counterexample_ids") or ()),
        )


class FailureMemory:
    """Bounded durable index of typed repair failures.

    Persistence is optional and atomic.  Counterexample identities accumulate
    and are never erased by retry decisions.
    """

    INTERFACE: ClassVar[str] = FAILURE_MEMORY_INTERFACE

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        policy: FailureMemoryPolicy | None = None,
    ) -> None:
        self.path = self._resolve_path(path)
        self.policy = policy or FailureMemoryPolicy()
        self._attempts: dict[str, FailureAttempt] = {}
        self._refuted: set[str] = set()
        self._attempt_retries: dict[str, int] = {}
        self._scope_retries: dict[str, int] = {}
        self._evidence_history: dict[str, set[str]] = {}
        self._last_measure: dict[str, RetryMeasure] = {}
        self._counterexamples: set[str] = set()
        if self.path is not None and self.path.exists():
            snapshot = self._load_path(self.path)
            if policy is not None and snapshot.policy != policy:
                raise FailureMemoryError(
                    "persisted failure-memory policy does not match requested policy"
                )
            self._load_snapshot(snapshot)

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        candidate = Path(path)
        if candidate.exists() and candidate.is_dir():
            candidate = candidate / "deterministic_failure_memory.json"
        elif not candidate.suffix:
            candidate = candidate / "deterministic_failure_memory.json"
        if candidate.is_symlink():
            raise FailureMemoryError("failure-memory state cannot be a symlink")
        return candidate

    def _load_snapshot(self, snapshot: FailureMemorySnapshot) -> None:
        self.policy = snapshot.policy
        self._attempts = {item.attempt_key: item for item in snapshot.attempts}
        self._refuted = set(snapshot.refuted_candidate_cids)
        self._attempt_retries = dict(snapshot.attempt_key_retry_counts)
        self._scope_retries = dict(snapshot.scope_retry_counts)
        self._evidence_history = {
            key: set(values)
            for key, values in snapshot.evidence_history_by_scope.items()
        }
        self._last_measure = dict(snapshot.last_measure_by_scope)
        self._counterexamples = set(snapshot.counterexample_ids)

    def snapshot(self) -> FailureMemorySnapshot:
        return FailureMemorySnapshot(
            policy=self.policy,
            attempts=tuple(self._attempts.values()),
            refuted_candidate_cids=tuple(sorted(self._refuted)),
            attempt_key_retry_counts=dict(self._attempt_retries),
            scope_retry_counts=dict(self._scope_retries),
            evidence_history_by_scope={
                key: tuple(sorted(values))
                for key, values in self._evidence_history.items()
            },
            last_measure_by_scope=dict(self._last_measure),
            counterexample_ids=tuple(sorted(self._counterexamples)),
        )

    @property
    def state_id(self) -> str:
        return self.snapshot().state_id

    @property
    def refuted_candidate_cids(self) -> tuple[str, ...]:
        return tuple(sorted(self._refuted))

    @property
    def counterexample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._counterexamples))

    def is_refuted(self, candidate_cid: str) -> bool:
        return _identifier(candidate_cid, "candidate_cid") in self._refuted

    def filter_admissible_candidates(
        self, candidate_cids: Sequence[str]
    ) -> tuple[str, ...]:
        """Return candidates that are not already refuted, in stable order."""

        ordered = _identifiers(candidate_cids, "candidate_cids", allow_empty=True)
        return tuple(item for item in ordered if item not in self._refuted)

    def record_attempt(
        self,
        attempt: FailureAttempt | Mapping[str, Any],
        *,
        observed_at_milliseconds: int = 1,
        proposed_candidate_cid: str = "",
        proposed_route_kind: AttemptRouteKind | str = AttemptRouteKind.DETERMINISTIC_REPLAN,
    ) -> FailureMemoryReceipt:
        """Record a failure and decide whether retry/rescue may emit work."""

        value = (
            attempt
            if isinstance(attempt, FailureAttempt)
            else FailureAttempt.from_dict(attempt)
        )
        decision = decide_replan(
            value,
            memory=self,
            proposed_candidate_cid=proposed_candidate_cid,
            proposed_route_kind=proposed_route_kind,
        )
        # Always retain the failure observation and its counterexamples.
        self._counterexamples.update(value.counterexample_ids)
        history = self._evidence_history.setdefault(value.scope_id, set())
        history.add(value.evidence_cid)
        if value.refuted:
            if (
                value.prior_candidate_cid not in self._refuted
                and len(self._refuted) >= self.policy.max_refuted_candidates
            ):
                # Bound reached: keep prior set; still return the decision.
                pass
            else:
                self._refuted.add(value.prior_candidate_cid)
        existing = self._attempts.get(value.attempt_key)
        if existing is None:
            if len(self._attempts) >= self.policy.max_attempts:
                if decision.should_replan:
                    decision = replace(
                        decision,
                        disposition=ReplanDisposition.MEMORY_BOUND_REACHED,
                        should_replan=False,
                        emits_work=False,
                        selected_candidate_cid="",
                        reason_codes=tuple(
                            sorted(
                                {
                                    *decision.reason_codes,
                                    "memory_bound_reached",
                                }
                            )
                        ),
                    )
            else:
                self._attempts[value.attempt_key] = value
        # Existing attempts keep their exact identity.  Counterexamples are
        # retained in the memory-level set and must never be erased or used to
        # rewrite a durable attempt key.
        if decision.should_replan:
            self._attempt_retries[value.attempt_key] = (
                self._attempt_retries.get(value.attempt_key, 0) + 1
            )
            self._scope_retries[value.scope_id] = (
                self._scope_retries.get(value.scope_id, 0) + 1
            )
            self._last_measure[value.scope_id] = value.measure
        elif value.scope_id not in self._last_measure:
            self._last_measure[value.scope_id] = value.measure
        self.persist()
        return FailureMemoryReceipt(
            attempt=self._attempts.get(value.attempt_key, value),
            decision=decision,
            memory_state_id=self.state_id,
            observed_at_milliseconds=observed_at_milliseconds,
        )

    observe = record_attempt
    record = record_attempt

    def decide(
        self,
        attempt: FailureAttempt | Mapping[str, Any],
        *,
        proposed_candidate_cid: str = "",
        proposed_route_kind: AttemptRouteKind | str = AttemptRouteKind.DETERMINISTIC_REPLAN,
    ) -> ReplanDecision:
        """Pure decision against current memory without mutating state."""

        return decide_replan(
            attempt,
            memory=self,
            proposed_candidate_cid=proposed_candidate_cid,
            proposed_route_kind=proposed_route_kind,
        )

    def persist(self) -> Path | None:
        if self.path is None:
            return None
        path = self.path
        if path.exists() and path.is_symlink():
            raise FailureMemoryError("failure-memory state cannot be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (canonical_json(self.snapshot().to_dict()) + "\n").encode("utf-8")
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FailureMemory":
        return cls(path)

    @staticmethod
    def _load_path(path: Path) -> FailureMemorySnapshot:
        if path.is_symlink() or not path.is_file():
            raise FailureMemoryError(
                "failure-memory state is unavailable or unsafe"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FailureMemoryError(
                "failure-memory state is unavailable or malformed"
            ) from exc
        return FailureMemorySnapshot.from_dict(payload)


def decide_replan(
    attempt: FailureAttempt | Mapping[str, Any],
    *,
    memory: FailureMemory | FailureMemorySnapshot | None = None,
    proposed_candidate_cid: str = "",
    proposed_route_kind: AttemptRouteKind | str = AttemptRouteKind.DETERMINISTIC_REPLAN,
) -> ReplanDecision:
    """Decide whether a retry/rescue may emit deterministic work.

    Acceptance rules:
    * unchanged inputs → ``NO_DUPLICATE_WORK`` (no work emitted);
    * provider/model route → ``PROVIDER_ROUTE_FORBIDDEN``;
    * refuted proposed candidate → ``REFUTED_CANDIDATE``;
    * otherwise retry only on new evidence or strictly decreasing measure.
    """

    value = (
        attempt
        if isinstance(attempt, FailureAttempt)
        else FailureAttempt.from_dict(attempt)
    )
    if isinstance(memory, FailureMemory):
        snapshot = memory.snapshot()
    elif isinstance(memory, FailureMemorySnapshot):
        snapshot = memory
    elif memory is None:
        snapshot = FailureMemorySnapshot(
            policy=FailureMemoryPolicy(),
            attempts=(),
            refuted_candidate_cids=(),
        )
    else:
        raise FailureMemoryError(
            "memory must be FailureMemory, FailureMemorySnapshot, or None"
        )

    route = _route_kind(proposed_route_kind)
    proposed = _identifier(
        proposed_candidate_cid, "proposed_candidate_cid", allow_empty=True
    )
    refuted = set(snapshot.refuted_candidate_cids)
    if value.refuted:
        refuted.add(value.prior_candidate_cid)
    refuted_tuple = tuple(sorted(refuted))

    def _decision(
        disposition: ReplanDisposition,
        *,
        should_replan: bool,
        selected: str = "",
        reasons: Sequence[str] = (),
    ) -> ReplanDecision:
        return ReplanDecision(
            disposition=disposition,
            attempt_key=value.attempt_key,
            failure_class=value.failure_class,
            prior_candidate_cid=value.prior_candidate_cid,
            evidence_cid=value.evidence_cid,
            measure=value.measure,
            should_replan=should_replan,
            emits_work=should_replan,
            allows_provider_route=False,
            selected_candidate_cid=selected if should_replan else "",
            refuted_candidate_cids=refuted_tuple,
            reason_codes=tuple(sorted({str(item) for item in reasons if str(item)})),
        )

    # Retry/rescue cannot route to a provider/model — including the failed
    # attempt itself and any proposed rescue route.
    if value.is_provider_route or is_provider_or_model_route(route):
        return _decision(
            ReplanDisposition.PROVIDER_ROUTE_FORBIDDEN,
            should_replan=False,
            reasons=("provider_or_model_route_forbidden", route.value),
        )

    if proposed and proposed in refuted:
        return _decision(
            ReplanDisposition.REFUTED_CANDIDATE,
            should_replan=False,
            reasons=("refuted_candidate", proposed),
        )
    if proposed and proposed == value.prior_candidate_cid and value.refuted:
        return _decision(
            ReplanDisposition.REFUTED_CANDIDATE,
            should_replan=False,
            reasons=("repeat_refuted_prior_candidate", proposed),
        )

    attempt_retries = snapshot.attempt_key_retry_counts.get(value.attempt_key, 0)
    scope_retries = snapshot.scope_retry_counts.get(value.scope_id, 0)
    if (
        attempt_retries >= snapshot.policy.max_retries_per_attempt_key
        or scope_retries >= snapshot.policy.max_retries_per_scope
    ):
        return _decision(
            ReplanDisposition.RETRY_BUDGET_EXHAUSTED,
            should_replan=False,
            reasons=("retry_budget_exhausted",),
        )

    known_attempt = any(
        item.attempt_key == value.attempt_key for item in snapshot.attempts
    )
    evidence_history = set(
        snapshot.evidence_history_by_scope.get(value.scope_id, ())
    )
    prior_measure = snapshot.last_measure_by_scope.get(value.scope_id)

    # Exact semantic replay of a previously recorded attempt: no new work.
    if known_attempt and value.evidence_cid in evidence_history:
        if prior_measure is not None and value.measure.strictly_decreases(prior_measure):
            return _decision(
                ReplanDisposition.RETRY_STRICTLY_DECREASING_MEASURE,
                should_replan=True,
                selected=proposed,
                reasons=("strictly_decreasing_measure",),
            )
        return _decision(
            ReplanDisposition.NO_DUPLICATE_WORK,
            should_replan=False,
            reasons=("unchanged_inputs", "no_duplicate_work"),
        )

    # New typed evidence in this scope authorizes one deterministic retry.
    if value.evidence_cid not in evidence_history:
        return _decision(
            ReplanDisposition.RETRY_NEW_EVIDENCE,
            should_replan=True,
            selected=proposed,
            reasons=("typed_new_evidence", value.failure_class.value),
        )

    # Same evidence family but strictly better measure.
    if prior_measure is not None and value.measure.strictly_decreases(prior_measure):
        return _decision(
            ReplanDisposition.RETRY_STRICTLY_DECREASING_MEASURE,
            should_replan=True,
            selected=proposed,
            reasons=("strictly_decreasing_measure",),
        )

    if known_attempt:
        return _decision(
            ReplanDisposition.NO_DUPLICATE_WORK,
            should_replan=False,
            reasons=("unchanged_inputs", "no_duplicate_work"),
        )

    # First observation of this attempt key under already-seen evidence without
    # measure progress: still abstain rather than thrash.
    if prior_measure is not None and not value.measure.strictly_decreases(prior_measure):
        return _decision(
            ReplanDisposition.NO_DUPLICATE_WORK,
            should_replan=False,
            reasons=("no_measure_progress", "no_duplicate_work"),
        )

    return _decision(
        ReplanDisposition.RETRY_NEW_EVIDENCE,
        should_replan=True,
        selected=proposed,
        reasons=("first_observation", value.failure_class.value),
    )


def materialize_replan_fixtures(
    *,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a compact recipe-style fixture for DCR-063 validation."""

    memory = FailureMemory(
        policy=FailureMemoryPolicy(
            max_attempts=32,
            max_refuted_candidates=32,
            max_retries_per_attempt_key=4,
            max_retries_per_scope=8,
        )
    )
    base_measure = RetryMeasure(
        open_counterexamples=2,
        validation_findings=3,
        remaining_candidates=4,
        resource_debt=100,
        capability_gaps=1,
    )
    first = FailureAttempt(
        failure_class=FailureClass.VALIDATION,
        prior_candidate_cid="candidate:op-a",
        evidence_cid="evidence:validation-v1",
        measure=base_measure,
        route_kind=AttemptRouteKind.DETERMINISTIC_OPERATOR,
        scope_id="scope:dcr063",
        plan_id="plan:dcr063",
        counterexample_ids=("cex:validation-1",),
        operator_kind="add_registration",
    )
    receipt_new = memory.record_attempt(
        first,
        observed_at_milliseconds=100,
        proposed_candidate_cid="candidate:op-b",
        proposed_route_kind=AttemptRouteKind.DETERMINISTIC_REPLAN,
    )
    receipt_replay = memory.record_attempt(
        first,
        observed_at_milliseconds=101,
        proposed_candidate_cid="candidate:op-b",
        proposed_route_kind=AttemptRouteKind.DETERMINISTIC_REPLAN,
    )
    receipt_refuted = decide_replan(
        first,
        memory=memory,
        proposed_candidate_cid="candidate:op-a",
        proposed_route_kind=AttemptRouteKind.DETERMINISTIC_REPLAN,
    )
    receipt_provider = decide_replan(
        replace(first, route_kind=AttemptRouteKind.PROVIDER),
        memory=memory,
        proposed_candidate_cid="candidate:op-c",
        proposed_route_kind=AttemptRouteKind.MODEL,
    )
    improved = replace(
        first,
        evidence_cid="evidence:validation-v1",
        measure=RetryMeasure(
            open_counterexamples=1,
            validation_findings=2,
            remaining_candidates=3,
            resource_debt=50,
            capability_gaps=0,
        ),
        prior_candidate_cid="candidate:op-b",
    )
    receipt_progress = memory.record_attempt(
        improved,
        observed_at_milliseconds=102,
        proposed_candidate_cid="candidate:op-c",
        proposed_route_kind=AttemptRouteKind.RESCUE_OPERATOR,
    )
    payload = {
        "artifact_schema": FAILURE_MEMORY_SCHEMA,
        "evidence_id": DCR_REPLAN_EVIDENCE,
        "interfaces": {
            "failure_memory": FAILURE_MEMORY_INTERFACE,
            "replan_decision": REPLAN_DECISION_INTERFACE,
        },
        "version": DETERMINISTIC_FAILURE_MEMORY_VERSION,
        "runtime_model_calls": 0,
        "grants_write_authority": False,
        "memory": memory.snapshot().to_dict(),
        "receipts": {
            "new_evidence": receipt_new.to_dict(),
            "replay_no_duplicate": receipt_replay.to_dict(),
            "progress": receipt_progress.to_dict(),
        },
        "decisions": {
            "refuted_candidate": receipt_refuted.to_dict(),
            "provider_route_forbidden": receipt_provider.to_dict(),
        },
        "evidence_subset": {
            "new_evidence": receipt_new.evidence_subset(),
            "replay_no_duplicate": receipt_replay.evidence_subset(),
            "progress": receipt_progress.evidence_subset(),
            "refuted_candidate": receipt_refuted.evidence_subset(),
            "provider_route_forbidden": receipt_provider.evidence_subset(),
        },
    }
    if destination is None:
        root = Path(repo_root) if repo_root else Path.cwd()
        destination = root / DEFAULT_REPLAN_FIXTURES_REL
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "AttemptRouteKind",
    "DCR_REPLAN_EVIDENCE",
    "DEFAULT_REPLAN_FIXTURES_REL",
    "DETERMINISTIC_FAILURE_MEMORY_VERSION",
    "FAILURE_ATTEMPT_SCHEMA",
    "FAILURE_MEMORY_INTERFACE",
    "FAILURE_MEMORY_RECEIPT_SCHEMA",
    "FAILURE_MEMORY_SCHEMA",
    "FailureAttempt",
    "FailureClass",
    "FailureMemory",
    "FailureMemoryError",
    "FailureMemoryPolicy",
    "FailureMemoryReceipt",
    "FailureMemorySnapshot",
    "REPLAN_DECISION_INTERFACE",
    "REPLAN_DECISION_SCHEMA",
    "RETRY_MEASURE_SCHEMA",
    "ReplanDecision",
    "ReplanDisposition",
    "RetryMeasure",
    "decide_replan",
    "is_provider_or_model_route",
    "materialize_replan_fixtures",
]
