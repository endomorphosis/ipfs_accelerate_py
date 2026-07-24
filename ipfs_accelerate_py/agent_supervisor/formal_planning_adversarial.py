"""Fail-closed admission checks for formal-planning and prover evidence.

Formal-looking output is attacker-controlled until every semantic and
execution binding has been checked.  This module is the common admission
boundary for the G12 prover matrix.  It deliberately does not execute a
prover.  Instead, it checks the evidence returned by a prover against the
canonical request which authorized that execution.

The boundary has three important properties:

* all plan, formula, model, premise, tool, cache, receipt, and trace identities
  are compared structurally rather than inferred from provider prose;
* property-specific evidence is not treated as a numeric assurance ladder
  (for example, a runtime trace cannot prove a kernel theorem); and
* duplicate validation work uses the durable fenced single-flight channel
  shared by :mod:`prover_evidence_store`.

The records intentionally contain identities and bounded public diagnostics
only.  Raw prover output, witnesses, model prompts, and hypertraces must remain
in their access-controlled origin stores.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    _canonical_value,
    canonical_json,
)
from .multi_prover_router import PropertyKind
from .prover_evidence_store import (
    ProverEvidenceStore,
    build_prover_evidence_key,
)


FORMAL_PLANNING_ADVERSARIAL_VERSION: Final = 1
PLAN_TRUST_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/plan-trust-binding@1"
)
PROVER_BOUNDARY_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prover-boundary-evidence@1"
)
ADVERSARIAL_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-planning-adversarial-policy@1"
)
ADVERSARIAL_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-planning-admission@1"
)
MAX_PUBLIC_FINDING_BYTES: Final = 384
DEFAULT_CACHE_MAX_AGE_MS: Final = 24 * 60 * 60 * 1000


class AdversarialValidationError(ContractValidationError):
    """An adversarial boundary record is malformed."""


class EvidenceSource(str, Enum):
    """Origin boundary of one result.

    ``CACHE`` is not an authority.  A cache envelope must contain and
    successfully revalidate its typed origin evidence.
    """

    MODEL_TEXT = "model_text"
    NATIVE_HEURISTIC = "native_heuristic"
    SOLVER = "solver"
    MODEL_CHECKER = "model_checker"
    AUTHORIZATION_ENGINE = "authorization_engine"
    PROTOCOL_ENGINE = "protocol_engine"
    HYPERPROPERTY_ENGINE = "hyperproperty_engine"
    RUNTIME_MONITOR = "runtime_monitor"
    KERNEL = "kernel"
    ZKP = "zkp"
    CACHE = "cache"


class EvidenceExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    INCOMPLETE = "incomplete"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"


class EvidenceConclusion(str, Enum):
    HOLDS = "holds"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class EvidenceClass(str, Enum):
    """Property-specific evidence class.

    These values are intentionally not ordered.  Policy must name acceptable
    classes for a property instead of comparing strings or ranks.
    """

    UNVERIFIED = "unverified"
    SOLVER_CANDIDATE = "solver_candidate"
    BOUNDED_MODEL_CHECKED = "bounded_model_checked"
    AUTHORIZATION_CHECKED = "authorization_checked"
    PROTOCOL_CHECKED = "protocol_checked"
    HYPERPROPERTY_CHECKED = "hyperproperty_checked"
    RUNTIME_CHECKED = "runtime_checked"
    KERNEL_VERIFIED = "kernel_verified"
    ATTESTED = "attested"


class AdmissionDisposition(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class BoundaryKind(str, Enum):
    PLAN = "plan"
    ACTOR_AUTHORITY = "actor_authority"
    TEMPORAL_BOUNDS = "temporal_bounds"
    TASK_DEPENDENCIES = "task_dependencies"
    FORMULA = "formula"
    MODEL = "model"
    PREMISES = "premises"
    TOOLCHAIN = "toolchain"
    CACHE = "cache"
    RECEIPT = "receipt"
    TRACE = "trace"
    ASSURANCE = "assurance"
    CAPABILITY = "capability"
    CONFORMANCE = "conformance"
    EXPLORATION = "exploration"
    PROTOCOL = "protocol"
    HYPERTRACE = "hypertrace"
    MONITOR = "monitor"
    SINGLE_FLIGHT = "single_flight"


class FindingCode(str, Enum):
    BINDING_MISMATCH = "binding_mismatch"
    ACTOR_NOT_AUTHORIZED = "actor_not_authorized"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_VERSION_UNVERIFIED = "tool_version_unverified"
    EXECUTION_NOT_SUCCESSFUL = "execution_not_successful"
    CONFORMANCE_NOT_PASSED = "conformance_not_passed"
    RECEIPT_NOT_VERIFIED = "receipt_not_verified"
    SOURCE_NOT_AUTHORITATIVE = "source_not_authoritative"
    SOLVER_DISAGREEMENT = "solver_disagreement"
    BOUNDS_MISSING = "bounds_missing"
    EXPLORATION_INCOMPLETE = "exploration_incomplete"
    PROTOCOL_FALSE_POSITIVE = "protocol_false_positive"
    HYPERTRACE_LEAKAGE = "hypertrace_leakage"
    MONITOR_GAP = "monitor_gap"
    MODEL_TEXT_UNTRUSTED = "model_text_untrusted"
    HEURISTIC_PROOF_UNTRUSTED = "heuristic_proof_untrusted"
    SIMULATED_ZKP = "simulated_zkp"
    ATTESTATION_SUBJECT_UNTRUSTED = "attestation_subject_untrusted"
    CACHE_STALE = "cache_stale"
    CACHE_INVALIDATED = "cache_invalidated"
    CACHE_POISONED = "cache_poisoned"
    CACHE_ORIGIN_MISSING = "cache_origin_missing"
    FORGED_ASSURANCE = "forged_assurance"
    PROPERTY_CLASS_MISMATCH = "property_class_mismatch"
    EVIDENCE_CLASS_NOT_ALLOWED = "evidence_class_not_allowed"
    ASSURANCE_INSUFFICIENT = "assurance_insufficient"
    VALIDATION_CANCELLED = "validation_cancelled"
    SINGLE_FLIGHT_FAILED = "single_flight_failed"


_BOUNDARY_FOR_FIELD: Final[Mapping[str, BoundaryKind]] = {
    "plan_id": BoundaryKind.PLAN,
    "task_id": BoundaryKind.PLAN,
    "repository_tree_id": BoundaryKind.PLAN,
    "policy_id": BoundaryKind.PLAN,
    "lane_id": BoundaryKind.PLAN,
    "actor_id": BoundaryKind.ACTOR_AUTHORITY,
    "authority_ids": BoundaryKind.ACTOR_AUTHORITY,
    "temporal_bounds": BoundaryKind.TEMPORAL_BOUNDS,
    "dependency_ids": BoundaryKind.TASK_DEPENDENCIES,
    "formula_id": BoundaryKind.FORMULA,
    "normalized_model_id": BoundaryKind.MODEL,
    "premise_ids": BoundaryKind.PREMISES,
    "tool_versions": BoundaryKind.TOOLCHAIN,
    "executable_digests": BoundaryKind.TOOLCHAIN,
    "conformance_fixture_set_id": BoundaryKind.CONFORMANCE,
    "cache_key_id": BoundaryKind.CACHE,
    "receipt_id": BoundaryKind.RECEIPT,
    "trace_id": BoundaryKind.TRACE,
}

_PRIVATE_MARKERS: Final = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "secret",
        "token",
        "witness",
    }
)

_BOUNDED_SOURCES: Final = frozenset(
    {
        EvidenceSource.MODEL_CHECKER,
        EvidenceSource.PROTOCOL_ENGINE,
        EvidenceSource.HYPERPROPERTY_ENGINE,
    }
)


def _json(value: Any, name: str) -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    try:
        return _canonical_value(value)
    except (TypeError, ValueError, ContractValidationError) as exc:
        raise AdversarialValidationError(
            f"{name} must contain canonical JSON values"
        ) from exc


def _mapping(value: Any, name: str, *, nonempty: bool = False) -> dict[str, Any]:
    result = _json(value, name)
    if not isinstance(result, dict):
        raise AdversarialValidationError(f"{name} must be an object")
    if nonempty and not result:
        raise AdversarialValidationError(f"{name} must not be empty")
    return result


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise AdversarialValidationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise AdversarialValidationError(f"{name} must not be empty")
    return result


def _strings(
    value: Iterable[Any] | None,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        items: Iterable[Any] = ()
    elif isinstance(value, (str, bytes, bytearray)):
        raise AdversarialValidationError(f"{name} must be a sequence")
    else:
        items = value
    result = tuple(sorted({_text(item, name) for item in items}))
    if required and not result:
        raise AdversarialValidationError(f"{name} must not be empty")
    return result


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise AdversarialValidationError(f"{name} is unsupported") from exc


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdversarialValidationError(f"{name} must be a non-negative integer")
    return value


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_json(_canonical_value(value)).encode("utf-8")
        ).hexdigest()
    )


def _public_message(value: str) -> str:
    encoded = value.encode("utf-8", errors="replace")[:MAX_PUBLIC_FINDING_BYTES]
    return encoded.decode("utf-8", errors="ignore")


def _has_explicit_bounds(value: Mapping[str, Any]) -> bool:
    if not value:
        return False

    def valid(item: Any) -> bool:
        if item is None or item is False:
            return False
        if isinstance(item, str):
            return bool(item.strip()) and item.strip().lower() not in {
                "none",
                "unbounded",
                "unknown",
            }
        if isinstance(item, int):
            return not isinstance(item, bool) and item > 0
        if isinstance(item, Mapping):
            return bool(item) and all(valid(child) for child in item.values())
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return bool(item) and all(valid(child) for child in item)
        return True

    return all(valid(item) for item in value.values())


def _contains_private_material(
    value: Any,
    *,
    forbidden_values: frozenset[str],
) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if any(marker in normalized for marker in _PRIVATE_MARKERS):
                if item not in (None, "", "[REDACTED]", "<redacted>", "redacted"):
                    return True
            if _contains_private_material(item, forbidden_values=forbidden_values):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(
            _contains_private_material(item, forbidden_values=forbidden_values)
            for item in value
        )
    return isinstance(value, str) and value in forbidden_values


@dataclass(frozen=True)
class PlanTrustBinding:
    """Canonical identity of every input capable of changing a plan result."""

    plan_id: str
    task_id: str
    repository_tree_id: str
    policy_id: str
    lane_id: str
    actor_id: str
    authority_ids: tuple[str, ...]
    temporal_bounds: Mapping[str, Any]
    dependency_ids: tuple[str, ...]
    formula_id: str
    normalized_model_id: str
    premise_ids: tuple[str, ...]
    tool_versions: Mapping[str, Any]
    executable_digests: Mapping[str, Any]
    conformance_fixture_set_id: str
    cache_key_id: str
    receipt_id: str
    trace_id: str

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "task_id",
            "repository_tree_id",
            "policy_id",
            "lane_id",
            "actor_id",
            "formula_id",
            "normalized_model_id",
            "conformance_fixture_set_id",
            "cache_key_id",
            "receipt_id",
            "trace_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "authority_ids",
            _strings(self.authority_ids, "authority_ids", required=True),
        )
        object.__setattr__(
            self, "dependency_ids", _strings(self.dependency_ids, "dependency_ids")
        )
        object.__setattr__(
            self, "premise_ids", _strings(self.premise_ids, "premise_ids")
        )
        object.__setattr__(
            self, "temporal_bounds", _mapping(self.temporal_bounds, "temporal_bounds")
        )
        object.__setattr__(
            self,
            "tool_versions",
            _mapping(self.tool_versions, "tool_versions", nonempty=True),
        )
        object.__setattr__(
            self,
            "executable_digests",
            _mapping(self.executable_digests, "executable_digests", nonempty=True),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PLAN_TRUST_BINDING_SCHEMA,
            "version": FORMAL_PLANNING_ADVERSARIAL_VERSION,
            **{
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.__dict__.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @property
    def content_id(self) -> str:
        return "plan-trust-binding:" + _digest(self._payload())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanTrustBinding":
        if not isinstance(value, Mapping):
            raise AdversarialValidationError("plan trust binding must be an object")
        if value.get("schema") not in (None, "", PLAN_TRUST_BINDING_SCHEMA):
            raise AdversarialValidationError("unsupported plan trust binding schema")
        result = cls(
            plan_id=value.get("plan_id", ""),
            task_id=value.get("task_id", ""),
            repository_tree_id=value.get("repository_tree_id", ""),
            policy_id=value.get("policy_id", ""),
            lane_id=value.get("lane_id", ""),
            actor_id=value.get("actor_id", ""),
            authority_ids=tuple(value.get("authority_ids") or ()),
            temporal_bounds=value.get("temporal_bounds") or {},
            dependency_ids=tuple(value.get("dependency_ids") or ()),
            formula_id=value.get("formula_id", ""),
            normalized_model_id=value.get("normalized_model_id", ""),
            premise_ids=tuple(value.get("premise_ids") or ()),
            tool_versions=value.get("tool_versions") or {},
            executable_digests=value.get("executable_digests") or {},
            conformance_fixture_set_id=value.get("conformance_fixture_set_id", ""),
            cache_key_id=value.get("cache_key_id", ""),
            receipt_id=value.get("receipt_id", ""),
            trace_id=value.get("trace_id", ""),
        )
        claimed = value.get("content_id")
        if claimed and claimed != result.content_id:
            raise AdversarialValidationError(
                "plan trust binding content identity mismatch"
            )
        return result


@dataclass(frozen=True)
class ProverBoundaryEvidence:
    """Typed public projection returned by one prover-matrix lane."""

    property_class: PropertyKind | str
    source: EvidenceSource
    status: EvidenceExecutionStatus
    conclusion: EvidenceConclusion
    plan_id: str
    task_id: str
    repository_tree_id: str
    policy_id: str
    lane_id: str
    actor_id: str
    authority_ids: tuple[str, ...]
    temporal_bounds: Mapping[str, Any]
    dependency_ids: tuple[str, ...]
    formula_id: str
    normalized_model_id: str
    premise_ids: tuple[str, ...]
    tool_versions: Mapping[str, Any]
    executable_digests: Mapping[str, Any]
    conformance_fixture_set_id: str
    cache_key_id: str
    receipt_id: str
    trace_id: str
    claimed_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    actor_authorized: bool = True
    tool_available: bool = True
    executable_versions_verified: bool = True
    conformance_passed: bool = True
    authoritative: bool = True
    bounded: bool = False
    exploration_complete: bool = True
    solver_verdicts: Mapping[str, Any] = field(default_factory=dict)
    negative_fixtures_passed: bool = True
    hypertrace: Mapping[str, Any] = field(default_factory=dict)
    hypertrace_redacted: bool = True
    hypertrace_isolated: bool = True
    monitor_coverage_complete: bool = True
    monitor_gaps: tuple[str, ...] = ()
    receipt_digest_verified: bool = True
    receipt_bindings_verified: bool = True
    simulated: bool = False
    attested_receipt_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    cache_created_at_ms: int = 0
    cache_expires_at_ms: int = 0
    cache_invalidated: bool = False
    cache_digest_verified: bool = True
    origin: "ProverBoundaryEvidence | None" = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "property_class",
            _enum(self.property_class, PropertyKind, "property_class"),
        )
        object.__setattr__(self, "source", _enum(self.source, EvidenceSource, "source"))
        object.__setattr__(
            self,
            "status",
            _enum(self.status, EvidenceExecutionStatus, "status"),
        )
        object.__setattr__(
            self,
            "conclusion",
            _enum(self.conclusion, EvidenceConclusion, "conclusion"),
        )
        object.__setattr__(
            self,
            "claimed_assurance",
            _enum(self.claimed_assurance, AssuranceLevel, "claimed_assurance"),
        )
        object.__setattr__(
            self,
            "attested_receipt_assurance",
            _enum(
                self.attested_receipt_assurance,
                AssuranceLevel,
                "attested_receipt_assurance",
            ),
        )
        for name in (
            "plan_id",
            "task_id",
            "repository_tree_id",
            "policy_id",
            "lane_id",
            "actor_id",
            "formula_id",
            "normalized_model_id",
            "conformance_fixture_set_id",
            "cache_key_id",
            "receipt_id",
            "trace_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("authority_ids", "dependency_ids", "premise_ids", "monitor_gaps"):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    name,
                    required=name == "authority_ids",
                ),
            )
        for name, nonempty in (
            ("temporal_bounds", False),
            ("tool_versions", True),
            ("executable_digests", True),
            ("solver_verdicts", False),
            ("hypertrace", False),
        ):
            object.__setattr__(
                self, name, _mapping(getattr(self, name), name, nonempty=nonempty)
            )
        for name in (
            "actor_authorized",
            "tool_available",
            "executable_versions_verified",
            "conformance_passed",
            "authoritative",
            "bounded",
            "exploration_complete",
            "negative_fixtures_passed",
            "hypertrace_redacted",
            "hypertrace_isolated",
            "monitor_coverage_complete",
            "receipt_digest_verified",
            "receipt_bindings_verified",
            "simulated",
            "cache_invalidated",
            "cache_digest_verified",
        ):
            if not isinstance(getattr(self, name), bool):
                raise AdversarialValidationError(f"{name} must be boolean")
        object.__setattr__(
            self,
            "cache_created_at_ms",
            _nonnegative(self.cache_created_at_ms, "cache_created_at_ms"),
        )
        object.__setattr__(
            self,
            "cache_expires_at_ms",
            _nonnegative(self.cache_expires_at_ms, "cache_expires_at_ms"),
        )
        if self.origin is not None and not isinstance(
            self.origin, ProverBoundaryEvidence
        ):
            raise AdversarialValidationError(
                "origin must be typed prover boundary evidence"
            )
        if self.source is not EvidenceSource.CACHE and self.origin is not None:
            raise AdversarialValidationError(
                "only cache evidence may contain origin evidence"
            )
        if self.source is EvidenceSource.CACHE and self.origin is self:
            raise AdversarialValidationError("cache evidence cannot contain itself")

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": PROVER_BOUNDARY_EVIDENCE_SCHEMA,
            "version": FORMAL_PLANNING_ADVERSARIAL_VERSION,
        }
        for name, value in self.__dict__.items():
            if isinstance(value, Enum):
                payload[name] = value.value
            elif isinstance(value, tuple):
                payload[name] = list(value)
            elif isinstance(value, ProverBoundaryEvidence):
                payload[name] = value.to_dict()
            else:
                payload[name] = value
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @property
    def content_id(self) -> str:
        return "prover-boundary-evidence:" + _digest(self._payload())

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        _depth: int = 0,
    ) -> "ProverBoundaryEvidence":
        if _depth > 1:
            raise AdversarialValidationError("nested cache origins are not allowed")
        if not isinstance(value, Mapping):
            raise AdversarialValidationError(
                "prover boundary evidence must be an object"
            )
        if value.get("schema") not in (None, "", PROVER_BOUNDARY_EVIDENCE_SCHEMA):
            raise AdversarialValidationError(
                "unsupported prover boundary evidence schema"
            )
        origin_value = value.get("origin")
        origin = None
        if origin_value is not None:
            if not isinstance(origin_value, Mapping):
                raise AdversarialValidationError("cache origin must be an object")
            origin = cls.from_dict(origin_value, _depth=_depth + 1)
        result = cls(
            property_class=value.get("property_class", ""),
            source=value.get("source", ""),
            status=value.get("status", ""),
            conclusion=value.get("conclusion", EvidenceConclusion.UNKNOWN),
            plan_id=value.get("plan_id", ""),
            task_id=value.get("task_id", ""),
            repository_tree_id=value.get("repository_tree_id", ""),
            policy_id=value.get("policy_id", ""),
            lane_id=value.get("lane_id", ""),
            actor_id=value.get("actor_id", ""),
            authority_ids=tuple(value.get("authority_ids") or ()),
            temporal_bounds=value.get("temporal_bounds") or {},
            dependency_ids=tuple(value.get("dependency_ids") or ()),
            formula_id=value.get("formula_id", ""),
            normalized_model_id=value.get("normalized_model_id", ""),
            premise_ids=tuple(value.get("premise_ids") or ()),
            tool_versions=value.get("tool_versions") or {},
            executable_digests=value.get("executable_digests") or {},
            conformance_fixture_set_id=value.get("conformance_fixture_set_id", ""),
            cache_key_id=value.get("cache_key_id", ""),
            receipt_id=value.get("receipt_id", ""),
            trace_id=value.get("trace_id", ""),
            claimed_assurance=value.get("claimed_assurance", AssuranceLevel.UNVERIFIED),
            actor_authorized=value.get("actor_authorized", True),
            tool_available=value.get("tool_available", True),
            executable_versions_verified=value.get(
                "executable_versions_verified", True
            ),
            conformance_passed=value.get("conformance_passed", True),
            authoritative=value.get("authoritative", True),
            bounded=value.get("bounded", False),
            exploration_complete=value.get("exploration_complete", True),
            solver_verdicts=value.get("solver_verdicts") or {},
            negative_fixtures_passed=value.get("negative_fixtures_passed", True),
            hypertrace=value.get("hypertrace") or {},
            hypertrace_redacted=value.get("hypertrace_redacted", True),
            hypertrace_isolated=value.get("hypertrace_isolated", True),
            monitor_coverage_complete=value.get("monitor_coverage_complete", True),
            monitor_gaps=tuple(value.get("monitor_gaps") or ()),
            receipt_digest_verified=value.get("receipt_digest_verified", True),
            receipt_bindings_verified=value.get("receipt_bindings_verified", True),
            simulated=value.get("simulated", False),
            attested_receipt_assurance=value.get(
                "attested_receipt_assurance", AssuranceLevel.UNVERIFIED
            ),
            cache_created_at_ms=value.get("cache_created_at_ms", 0),
            cache_expires_at_ms=value.get("cache_expires_at_ms", 0),
            cache_invalidated=value.get("cache_invalidated", False),
            cache_digest_verified=value.get("cache_digest_verified", True),
            origin=origin,
        )
        claimed = value.get("content_id")
        if claimed and claimed != result.content_id:
            raise AdversarialValidationError(
                "prover boundary evidence content identity mismatch"
            )
        return result


_DEFAULT_CLASSES: Final[Mapping[PropertyKind, tuple[EvidenceClass, ...]]] = {
    PropertyKind.FINITE_CONSTRAINT: (EvidenceClass.SOLVER_CANDIDATE,),
    PropertyKind.STATE_MACHINE: (EvidenceClass.BOUNDED_MODEL_CHECKED,),
    PropertyKind.AUTHORIZATION: (EvidenceClass.AUTHORIZATION_CHECKED,),
    PropertyKind.PROTOCOL: (EvidenceClass.PROTOCOL_CHECKED,),
    PropertyKind.HYPERPROPERTY: (EvidenceClass.HYPERPROPERTY_CHECKED,),
    PropertyKind.RUNTIME_TRACE: (EvidenceClass.RUNTIME_CHECKED,),
    PropertyKind.KERNEL_CHECK: (
        EvidenceClass.KERNEL_VERIFIED,
        EvidenceClass.ATTESTED,
    ),
    PropertyKind.TYPED_PLANNING: (
        EvidenceClass.SOLVER_CANDIDATE,
        EvidenceClass.BOUNDED_MODEL_CHECKED,
        EvidenceClass.KERNEL_VERIFIED,
    ),
    PropertyKind.TEMPORAL_DEONTIC: (
        EvidenceClass.SOLVER_CANDIDATE,
        EvidenceClass.BOUNDED_MODEL_CHECKED,
        EvidenceClass.KERNEL_VERIFIED,
    ),
    PropertyKind.FIRST_ORDER_THEOREM: (
        EvidenceClass.SOLVER_CANDIDATE,
        EvidenceClass.KERNEL_VERIFIED,
    ),
}


@dataclass(frozen=True)
class AdversarialPolicy:
    """Admission policy for one exact property request."""

    property_class: PropertyKind | str
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED
    accepted_evidence_classes: tuple[EvidenceClass | str, ...] = ()
    require_conformance: bool = True
    max_cache_age_ms: int = DEFAULT_CACHE_MAX_AGE_MS
    now_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    forbidden_public_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        property_class = _enum(self.property_class, PropertyKind, "property_class")
        object.__setattr__(self, "property_class", property_class)
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        classes = self.accepted_evidence_classes or _DEFAULT_CLASSES[property_class]
        object.__setattr__(
            self,
            "accepted_evidence_classes",
            tuple(
                sorted(
                    {
                        _enum(item, EvidenceClass, "accepted_evidence_classes")
                        for item in classes
                    },
                    key=lambda item: item.value,
                )
            ),
        )
        if not isinstance(self.require_conformance, bool):
            raise AdversarialValidationError("require_conformance must be boolean")
        object.__setattr__(
            self,
            "max_cache_age_ms",
            _nonnegative(self.max_cache_age_ms, "max_cache_age_ms"),
        )
        object.__setattr__(self, "now_ms", _nonnegative(self.now_ms, "now_ms"))
        object.__setattr__(
            self,
            "forbidden_public_values",
            _strings(self.forbidden_public_values, "forbidden_public_values"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ADVERSARIAL_POLICY_SCHEMA,
            "version": FORMAL_PLANNING_ADVERSARIAL_VERSION,
            "property_class": self.property_class.value,
            "required_assurance": self.required_assurance.value,
            "accepted_evidence_classes": [
                item.value for item in self.accepted_evidence_classes
            ],
            "require_conformance": self.require_conformance,
            "max_cache_age_ms": self.max_cache_age_ms,
            "now_ms": self.now_ms,
            # Values are identities for local leakage detection and must not be
            # copied into durable public outcomes.
            "forbidden_public_value_ids": [
                _digest(item) for item in self.forbidden_public_values
            ],
        }


@dataclass(frozen=True)
class AdversarialFinding:
    code: FindingCode
    boundary: BoundaryKind
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _enum(self.code, FindingCode, "code"))
        object.__setattr__(
            self, "boundary", _enum(self.boundary, BoundaryKind, "boundary")
        )
        object.__setattr__(
            self, "message", _public_message(_text(self.message, "message"))
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "boundary": self.boundary.value,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdversarialFinding":
        return cls(
            code=value.get("code", ""),
            boundary=value.get("boundary", ""),
            message=value.get("message", ""),
        )


@dataclass(frozen=True)
class AdversarialAdmission:
    """Fail-closed result of one boundary evaluation."""

    binding_id: str
    evidence_id: str
    policy_id: str
    disposition: AdmissionDisposition
    evidence_class: EvidenceClass
    authoritative_assurance: AssuranceLevel
    conclusion: EvidenceConclusion
    findings: tuple[AdversarialFinding, ...] = ()

    def __post_init__(self) -> None:
        for name in ("binding_id", "evidence_id", "policy_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, AdmissionDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "evidence_class",
            _enum(self.evidence_class, EvidenceClass, "evidence_class"),
        )
        object.__setattr__(
            self,
            "authoritative_assurance",
            _enum(
                self.authoritative_assurance,
                AssuranceLevel,
                "authoritative_assurance",
            ),
        )
        object.__setattr__(
            self,
            "conclusion",
            _enum(self.conclusion, EvidenceConclusion, "conclusion"),
        )
        if any(not isinstance(item, AdversarialFinding) for item in self.findings):
            raise AdversarialValidationError(
                "findings must contain typed adversarial findings"
            )
        object.__setattr__(self, "findings", tuple(self.findings))
        if self.disposition is AdmissionDisposition.ADMITTED and self.findings:
            raise AdversarialValidationError(
                "an admitted result cannot contain rejection findings"
            )

    @property
    def admitted(self) -> bool:
        return self.disposition is AdmissionDisposition.ADMITTED

    @property
    def fail_closed(self) -> bool:
        return not self.admitted

    @property
    def promotable(self) -> bool:
        return self.admitted and self.conclusion is EvidenceConclusion.HOLDS

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(item.code.value for item in self.findings)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ADVERSARIAL_ADMISSION_SCHEMA,
            "version": FORMAL_PLANNING_ADVERSARIAL_VERSION,
            "binding_id": self.binding_id,
            "evidence_id": self.evidence_id,
            "policy_id": self.policy_id,
            "disposition": self.disposition.value,
            "evidence_class": self.evidence_class.value,
            "authoritative_assurance": self.authoritative_assurance.value,
            "conclusion": self.conclusion.value,
            "findings": [item.to_dict() for item in self.findings],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @property
    def content_id(self) -> str:
        return "formal-planning-admission:" + _digest(self._payload())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdversarialAdmission":
        if not isinstance(value, Mapping):
            raise AdversarialValidationError("admission must be an object")
        if value.get("schema") not in (None, "", ADVERSARIAL_ADMISSION_SCHEMA):
            raise AdversarialValidationError("unsupported admission schema")
        result = cls(
            binding_id=_text(value.get("binding_id"), "binding_id"),
            evidence_id=_text(value.get("evidence_id"), "evidence_id"),
            policy_id=_text(value.get("policy_id"), "policy_id"),
            disposition=_enum(
                value.get("disposition"), AdmissionDisposition, "disposition"
            ),
            evidence_class=_enum(
                value.get("evidence_class"), EvidenceClass, "evidence_class"
            ),
            authoritative_assurance=_enum(
                value.get("authoritative_assurance"),
                AssuranceLevel,
                "authoritative_assurance",
            ),
            conclusion=_enum(value.get("conclusion"), EvidenceConclusion, "conclusion"),
            findings=tuple(
                AdversarialFinding.from_dict(item)
                for item in value.get("findings") or ()
            ),
        )
        claimed = value.get("content_id")
        if claimed and claimed != result.content_id:
            raise AdversarialValidationError("admission content identity mismatch")
        return result


def _policy_id(policy: AdversarialPolicy) -> str:
    return "formal-planning-adversarial-policy:" + _digest(policy.to_dict())


def _finding(
    code: FindingCode,
    boundary: BoundaryKind,
    message: str,
) -> AdversarialFinding:
    return AdversarialFinding(code=code, boundary=boundary, message=message)


class FormalPlanningAdversarialGate:
    """Re-derive trust from typed evidence and exact request bindings."""

    def cancelled(
        self,
        binding: PlanTrustBinding,
        evidence: ProverBoundaryEvidence,
        policy: AdversarialPolicy,
    ) -> AdversarialAdmission:
        return AdversarialAdmission(
            binding_id=binding.content_id,
            evidence_id=evidence.content_id,
            policy_id=_policy_id(policy),
            disposition=AdmissionDisposition.REJECTED,
            evidence_class=EvidenceClass.UNVERIFIED,
            authoritative_assurance=AssuranceLevel.UNVERIFIED,
            conclusion=EvidenceConclusion.UNKNOWN,
            findings=(
                _finding(
                    FindingCode.VALIDATION_CANCELLED,
                    BoundaryKind.SINGLE_FLIGHT,
                    "validation was cancelled; no assurance was derived",
                ),
            ),
        )

    def execution_failed(
        self,
        binding: PlanTrustBinding,
        evidence: ProverBoundaryEvidence,
        policy: AdversarialPolicy,
    ) -> AdversarialAdmission:
        return AdversarialAdmission(
            binding_id=binding.content_id,
            evidence_id=evidence.content_id,
            policy_id=_policy_id(policy),
            disposition=AdmissionDisposition.REJECTED,
            evidence_class=EvidenceClass.UNVERIFIED,
            authoritative_assurance=AssuranceLevel.UNVERIFIED,
            conclusion=EvidenceConclusion.UNKNOWN,
            findings=(
                _finding(
                    FindingCode.SINGLE_FLIGHT_FAILED,
                    BoundaryKind.SINGLE_FLIGHT,
                    "validation execution failed; retry from a fresh fenced claim",
                ),
            ),
        )

    def evaluate(
        self,
        binding: PlanTrustBinding,
        evidence: ProverBoundaryEvidence,
        policy: AdversarialPolicy,
    ) -> AdversarialAdmission:
        if not isinstance(binding, PlanTrustBinding):
            raise AdversarialValidationError("binding must be a PlanTrustBinding")
        if not isinstance(evidence, ProverBoundaryEvidence):
            raise AdversarialValidationError("evidence must be ProverBoundaryEvidence")
        if not isinstance(policy, AdversarialPolicy):
            raise AdversarialValidationError("policy must be AdversarialPolicy")

        findings: list[AdversarialFinding] = []
        for name, boundary in _BOUNDARY_FOR_FIELD.items():
            if _canonical_value(getattr(binding, name)) != _canonical_value(
                getattr(evidence, name)
            ):
                findings.append(
                    _finding(
                        FindingCode.BINDING_MISMATCH,
                        boundary,
                        f"{name} does not match the authorized request",
                    )
                )

        if evidence.property_class is not policy.property_class:
            findings.append(
                _finding(
                    FindingCode.PROPERTY_CLASS_MISMATCH,
                    BoundaryKind.PLAN,
                    "evidence property class does not match policy",
                )
            )
        if not evidence.actor_authorized:
            findings.append(
                _finding(
                    FindingCode.ACTOR_NOT_AUTHORIZED,
                    BoundaryKind.ACTOR_AUTHORITY,
                    "the actor was not authorized for this transition",
                )
            )
        if not evidence.tool_available:
            findings.append(
                _finding(
                    FindingCode.TOOL_UNAVAILABLE,
                    BoundaryKind.CAPABILITY,
                    "the required executable capability was unavailable",
                )
            )
        if not evidence.executable_versions_verified:
            findings.append(
                _finding(
                    FindingCode.TOOL_VERSION_UNVERIFIED,
                    BoundaryKind.TOOLCHAIN,
                    "the executable version and file identity were not verified",
                )
            )
        if evidence.status is not EvidenceExecutionStatus.SUCCEEDED:
            findings.append(
                _finding(
                    FindingCode.EXECUTION_NOT_SUCCESSFUL,
                    BoundaryKind.CAPABILITY,
                    "the prover execution did not complete successfully",
                )
            )
        if policy.require_conformance and not evidence.conformance_passed:
            findings.append(
                _finding(
                    FindingCode.CONFORMANCE_NOT_PASSED,
                    BoundaryKind.CONFORMANCE,
                    "the exact executable path did not pass its fixture set",
                )
            )
        if (
            not evidence.receipt_digest_verified
            or not evidence.receipt_bindings_verified
        ):
            findings.append(
                _finding(
                    FindingCode.RECEIPT_NOT_VERIFIED,
                    BoundaryKind.RECEIPT,
                    "the receipt digest or semantic bindings were not verified",
                )
            )
        if not evidence.authoritative:
            findings.append(
                _finding(
                    FindingCode.SOURCE_NOT_AUTHORITATIVE,
                    BoundaryKind.ASSURANCE,
                    "the producing lane is not an authoritative source for this result",
                )
            )

        evidence_class, assurance = self._derive_source_trust(
            binding, evidence, policy, findings
        )

        if evidence.claimed_assurance.rank > assurance.rank:
            findings.append(
                _finding(
                    FindingCode.FORGED_ASSURANCE,
                    BoundaryKind.ASSURANCE,
                    "the provider assurance label exceeds independently derived assurance",
                )
            )
        if evidence_class not in policy.accepted_evidence_classes:
            findings.append(
                _finding(
                    FindingCode.EVIDENCE_CLASS_NOT_ALLOWED,
                    BoundaryKind.ASSURANCE,
                    "this property policy does not accept the derived evidence class",
                )
            )
        if not assurance.satisfies(policy.required_assurance):
            findings.append(
                _finding(
                    FindingCode.ASSURANCE_INSUFFICIENT,
                    BoundaryKind.ASSURANCE,
                    "derived assurance does not satisfy the configured requirement",
                )
            )

        # A trustworthy counterexample is admitted evidence but can never be
        # promoted as evidence that the property holds.
        if findings:
            disposition = (
                AdmissionDisposition.INCONCLUSIVE
                if all(
                    item.code
                    in {
                        FindingCode.ASSURANCE_INSUFFICIENT,
                        FindingCode.EVIDENCE_CLASS_NOT_ALLOWED,
                    }
                    for item in findings
                )
                else AdmissionDisposition.REJECTED
            )
            assurance = AssuranceLevel.UNVERIFIED
            evidence_class = EvidenceClass.UNVERIFIED
        else:
            disposition = AdmissionDisposition.ADMITTED
        return AdversarialAdmission(
            binding_id=binding.content_id,
            evidence_id=evidence.content_id,
            policy_id=_policy_id(policy),
            disposition=disposition,
            evidence_class=evidence_class,
            authoritative_assurance=assurance,
            conclusion=evidence.conclusion,
            findings=tuple(findings),
        )

    validate = evaluate
    admit = evaluate

    def _derive_source_trust(
        self,
        binding: PlanTrustBinding,
        evidence: ProverBoundaryEvidence,
        policy: AdversarialPolicy,
        findings: list[AdversarialFinding],
    ) -> tuple[EvidenceClass, AssuranceLevel]:
        source = evidence.source

        if source is EvidenceSource.MODEL_TEXT:
            findings.append(
                _finding(
                    FindingCode.MODEL_TEXT_UNTRUSTED,
                    BoundaryKind.ASSURANCE,
                    "model text is a proposal and cannot establish formal assurance",
                )
            )
            return EvidenceClass.UNVERIFIED, AssuranceLevel.CANDIDATE
        if source is EvidenceSource.NATIVE_HEURISTIC:
            findings.append(
                _finding(
                    FindingCode.HEURISTIC_PROOF_UNTRUSTED,
                    BoundaryKind.ASSURANCE,
                    "native heuristic success requires independent reconstruction",
                )
            )
            return EvidenceClass.UNVERIFIED, AssuranceLevel.CANDIDATE

        if source in _BOUNDED_SOURCES:
            if not evidence.bounded or not _has_explicit_bounds(
                evidence.temporal_bounds
            ):
                findings.append(
                    _finding(
                        FindingCode.BOUNDS_MISSING,
                        BoundaryKind.TEMPORAL_BOUNDS,
                        "bounded evidence lacks explicit finite bounds",
                    )
                )
            if not evidence.exploration_complete:
                findings.append(
                    _finding(
                        FindingCode.EXPLORATION_INCOMPLETE,
                        BoundaryKind.EXPLORATION,
                        "the configured bounded exploration did not complete",
                    )
                )

        if source is EvidenceSource.SOLVER:
            verdicts = {
                str(value).strip().lower()
                for value in evidence.solver_verdicts.values()
                if str(value).strip().lower()
                in {"sat", "unsat", "holds", "violated", "proved", "disproved"}
            }
            if len(verdicts) > 1:
                findings.append(
                    _finding(
                        FindingCode.SOLVER_DISAGREEMENT,
                        BoundaryKind.CONFORMANCE,
                        "independent solvers disagree; the portfolio fails closed",
                    )
                )
            return EvidenceClass.SOLVER_CANDIDATE, AssuranceLevel.SOLVER_CHECKED
        if source is EvidenceSource.MODEL_CHECKER:
            return (
                EvidenceClass.BOUNDED_MODEL_CHECKED,
                AssuranceLevel.SOLVER_CHECKED,
            )
        if source is EvidenceSource.AUTHORIZATION_ENGINE:
            return (
                EvidenceClass.AUTHORIZATION_CHECKED,
                AssuranceLevel.SOLVER_CHECKED,
            )
        if source is EvidenceSource.PROTOCOL_ENGINE:
            if not evidence.negative_fixtures_passed:
                findings.append(
                    _finding(
                        FindingCode.PROTOCOL_FALSE_POSITIVE,
                        BoundaryKind.PROTOCOL,
                        "the protocol lane accepted a reviewed attack fixture",
                    )
                )
            return EvidenceClass.PROTOCOL_CHECKED, AssuranceLevel.SOLVER_CHECKED
        if source is EvidenceSource.HYPERPROPERTY_ENGINE:
            leaked = _contains_private_material(
                evidence.hypertrace,
                forbidden_values=frozenset(policy.forbidden_public_values),
            )
            if (
                leaked
                or not evidence.hypertrace_redacted
                or not evidence.hypertrace_isolated
            ):
                findings.append(
                    _finding(
                        FindingCode.HYPERTRACE_LEAKAGE,
                        BoundaryKind.HYPERTRACE,
                        "the public hypertrace violates redaction or lane isolation",
                    )
                )
            return (
                EvidenceClass.HYPERPROPERTY_CHECKED,
                AssuranceLevel.SOLVER_CHECKED,
            )
        if source is EvidenceSource.RUNTIME_MONITOR:
            if not evidence.monitor_coverage_complete or evidence.monitor_gaps:
                findings.append(
                    _finding(
                        FindingCode.MONITOR_GAP,
                        BoundaryKind.MONITOR,
                        "the runtime trace has missing or unobserved monitor intervals",
                    )
                )
            # Observing a finite trace is useful property-specific evidence but
            # never a proof over unobserved executions.
            return EvidenceClass.RUNTIME_CHECKED, AssuranceLevel.CANDIDATE
        if source is EvidenceSource.KERNEL:
            return EvidenceClass.KERNEL_VERIFIED, AssuranceLevel.KERNEL_VERIFIED
        if source is EvidenceSource.ZKP:
            if evidence.simulated:
                findings.append(
                    _finding(
                        FindingCode.SIMULATED_ZKP,
                        BoundaryKind.ASSURANCE,
                        "a simulated ZKP backend cannot provide authoritative attestation",
                    )
                )
            if evidence.origin is not None:
                # ZKP origins are forbidden by the contract because public
                # attestation evidence must bind a receipt, not embed witnesses.
                findings.append(
                    _finding(
                        FindingCode.ATTESTATION_SUBJECT_UNTRUSTED,
                        BoundaryKind.RECEIPT,
                        "attestation evidence may bind but not embed an origin",
                    )
                )
            if (
                evidence.claimed_assurance is not AssuranceLevel.ATTESTED
                or not evidence.receipt_bindings_verified
                or not evidence.attested_receipt_assurance.satisfies(
                    AssuranceLevel.KERNEL_VERIFIED
                )
            ):
                findings.append(
                    _finding(
                        FindingCode.ATTESTATION_SUBJECT_UNTRUSTED,
                        BoundaryKind.RECEIPT,
                        "attestation is not bound to a trusted receipt",
                    )
                )
            return EvidenceClass.ATTESTED, AssuranceLevel.ATTESTED
        if source is EvidenceSource.CACHE:
            if evidence.origin is None:
                findings.append(
                    _finding(
                        FindingCode.CACHE_ORIGIN_MISSING,
                        BoundaryKind.CACHE,
                        "cache evidence has no independently revalidatable origin",
                    )
                )
                return EvidenceClass.UNVERIFIED, AssuranceLevel.UNVERIFIED
            origin = self.evaluate(binding, evidence.origin, policy)
            if not origin.admitted:
                findings.append(
                    _finding(
                        FindingCode.CACHE_ORIGIN_MISSING,
                        BoundaryKind.CACHE,
                        "the cached origin no longer satisfies current policy",
                    )
                )
            if not evidence.cache_digest_verified:
                findings.append(
                    _finding(
                        FindingCode.CACHE_POISONED,
                        BoundaryKind.CACHE,
                        "the cache envelope digest does not match its contents",
                    )
                )
            if evidence.cache_invalidated:
                findings.append(
                    _finding(
                        FindingCode.CACHE_INVALIDATED,
                        BoundaryKind.CACHE,
                        "the cache entry was explicitly invalidated",
                    )
                )
            age = policy.now_ms - evidence.cache_created_at_ms
            if (
                evidence.cache_created_at_ms <= 0
                or evidence.cache_expires_at_ms <= policy.now_ms
                or age < 0
                or age > policy.max_cache_age_ms
            ):
                findings.append(
                    _finding(
                        FindingCode.CACHE_STALE,
                        BoundaryKind.CACHE,
                        "the cache entry is expired or older than current policy permits",
                    )
                )
            return (
                (origin.evidence_class, origin.authoritative_assurance)
                if origin.admitted
                else (EvidenceClass.UNVERIFIED, AssuranceLevel.UNVERIFIED)
            )

        return EvidenceClass.UNVERIFIED, AssuranceLevel.UNVERIFIED


@dataclass(frozen=True)
class CoordinatedAdmission:
    admission: AdversarialAdmission
    owner: bool
    fencing_token: int

    @property
    def shared(self) -> bool:
        return not self.owner


class AdversarialValidationCoordinator:
    """Cross-thread/process fenced single-flight for boundary validation.

    Only canonical admission results enter the durable outcome channel.
    Crashes, cancellation, malformed outcomes, and fenced owners are projected
    into an explicit rejected result, so callers cannot accidentally interpret
    absence of an exception as assurance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        gate: FormalPlanningAdversarialGate | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.gate = gate or FormalPlanningAdversarialGate()
        self.store = ProverEvidenceStore(path, clock=clock)

    @staticmethod
    def _flight_key(
        binding: PlanTrustBinding,
        evidence: ProverBoundaryEvidence,
        policy: AdversarialPolicy,
    ) -> Any:
        versions = {str(key): value for key, value in binding.tool_versions.items()}
        return build_prover_evidence_key(
            property_class=policy.property_class,
            normalized_model={
                "binding": binding.to_dict(),
                "evidence_id": evidence.content_id,
            },
            translator_profile={
                "boundary": PROVER_BOUNDARY_EVIDENCE_SCHEMA,
                "source": evidence.source.value,
            },
            assumptions=tuple(binding.premise_ids),
            finite_bounds=dict(binding.temporal_bounds),
            prover_versions=versions,
            kernel_versions={"admission-boundary": FORMAL_PLANNING_ADVERSARIAL_VERSION},
            policy=_policy_id(policy),
            repository_tree_id=binding.repository_tree_id,
            conformance_fixture_set_id=binding.conformance_fixture_set_id,
        )

    def evaluate(
        self,
        binding: PlanTrustBinding,
        evidence: ProverBoundaryEvidence,
        policy: AdversarialPolicy,
        *,
        cancel_event: threading.Event | None = None,
        evaluator: Callable[
            [PlanTrustBinding, ProverBoundaryEvidence, AdversarialPolicy],
            AdversarialAdmission,
        ]
        | None = None,
        owner_id: str | None = None,
        lease_seconds: int = 30,
        wait_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.01,
        outcome_ttl_seconds: int = 600,
    ) -> CoordinatedAdmission:
        key = self._flight_key(binding, evidence, policy)
        validate = evaluator or self.gate.evaluate

        def produce() -> dict[str, Any]:
            if cancel_event is not None and cancel_event.is_set():
                return self.gate.cancelled(binding, evidence, policy).to_dict()
            try:
                result = validate(binding, evidence, policy)
                if not isinstance(result, AdversarialAdmission):
                    raise AdversarialValidationError(
                        "adversarial evaluator returned an untyped result"
                    )
            except Exception:
                # Never hand provider exception text to the durable
                # single-flight outcome channel.  It may contain witnesses,
                # source excerpts, credentials, or raw prover transcripts.
                return self.gate.execution_failed(binding, evidence, policy).to_dict()
            if cancel_event is not None and cancel_event.is_set():
                return self.gate.cancelled(binding, evidence, policy).to_dict()
            return result.to_dict()

        try:
            result = self.store.single_flight(
                key,
                produce,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
                wait_timeout_seconds=wait_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                outcome_ttl_seconds=outcome_ttl_seconds,
            )
            admission = AdversarialAdmission.from_dict(result.value)
            if (
                admission.binding_id != binding.content_id
                or admission.evidence_id != evidence.content_id
                or admission.policy_id != _policy_id(policy)
            ):
                raise AdversarialValidationError(
                    "single-flight admission bindings do not match the request"
                )
            return CoordinatedAdmission(
                admission=admission,
                owner=result.owner,
                fencing_token=result.fencing_token,
            )
        except Exception:
            # The intentionally broad projection is safe here: provider
            # exception text is never reflected into the public finding.
            return CoordinatedAdmission(
                admission=self.gate.execution_failed(binding, evidence, policy),
                owner=False,
                fencing_token=0,
            )

    validate = evaluate
    coordinate = evaluate


def evaluate_formal_planning_evidence(
    binding: PlanTrustBinding,
    evidence: ProverBoundaryEvidence,
    policy: AdversarialPolicy,
) -> AdversarialAdmission:
    """Convenience entry point for a deterministic boundary evaluation."""

    return FormalPlanningAdversarialGate().evaluate(binding, evidence, policy)


# Compatibility spellings for callers using trust-boundary terminology.
FormalPlanningTrustBoundary = FormalPlanningAdversarialGate
FormalPlanningAdversarialValidator = FormalPlanningAdversarialGate
AdversarialEvidence = ProverBoundaryEvidence
TrustBoundaryPolicy = AdversarialPolicy


__all__ = [
    "ADVERSARIAL_ADMISSION_SCHEMA",
    "ADVERSARIAL_POLICY_SCHEMA",
    "AdversarialAdmission",
    "AdversarialEvidence",
    "AdversarialFinding",
    "AdversarialPolicy",
    "AdversarialValidationCoordinator",
    "AdversarialValidationError",
    "AdmissionDisposition",
    "BoundaryKind",
    "CoordinatedAdmission",
    "EvidenceClass",
    "EvidenceConclusion",
    "EvidenceExecutionStatus",
    "EvidenceSource",
    "FORMAL_PLANNING_ADVERSARIAL_VERSION",
    "FindingCode",
    "FormalPlanningAdversarialGate",
    "FormalPlanningAdversarialValidator",
    "FormalPlanningTrustBoundary",
    "PLAN_TRUST_BINDING_SCHEMA",
    "PROVER_BOUNDARY_EVIDENCE_SCHEMA",
    "PlanTrustBinding",
    "ProverBoundaryEvidence",
    "TrustBoundaryPolicy",
    "evaluate_formal_planning_evidence",
]
