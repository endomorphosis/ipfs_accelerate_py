"""Compile runtime state-machine claims into canonical, solver-neutral obligations.

Interface: ``RuntimeContractObligation@1``

This module is an adapter, not a theorem generator.  It accepts only the closed
:class:`RuntimeClaimFamily` vocabulary and exact, identifier-only premises and
bounds.  The theorem statement is a canonical structured expression selected
from the reviewed family table below; callers cannot provide prose or source
text.

The resulting record keeps three existing representations bound together:

* a domain-separated shared logic-IR view from ``ipfs_datasets_py``;
* the accelerator's existing :class:`CodeProofObligation`; and
* the accelerator's existing ``CodeClaimRecord@1`` lifecycle record.

No representation establishes the theorem.  Analysis results are observations
(``proved`` / ``refuted`` / ``unknown`` / ``unsupported`` / ``timed_out``) and
supported obligations remain open until independently discharged at the
required assurance level.  Unsupported program semantics stay ``unknown`` and
must never be silently treated as refutation.  Compact counterexamples identify
the failed edge, transition, or invariant.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..analysis.content_identity_bridge import (
    LOGIC_IR_PROFILE,
    identify_logic_ir,
)
from ..analysis.runtime_component_catalog import RuntimeComponentCatalog
from .code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
    CodeClaimRecord,
    claim_from_obligation,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    canonical_json,
    content_identity,
)


RUNTIME_CONTRACT_OBLIGATIONS_INTERFACE: Final = "RuntimeContractObligation@1"
RUNTIME_CONTRACT_OBLIGATION_INTERFACE: Final = (
    RUNTIME_CONTRACT_OBLIGATIONS_INTERFACE
)
RUNTIME_CONTRACT_OBLIGATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-obligation@1"
)
RUNTIME_LOGIC_VIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-logic-view@1"
)
RUNTIME_LOGIC_EXPRESSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-logic-expression@1"
)
RUNTIME_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-claim@1"
)
RUNTIME_COUNTEREXAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-counterexample@1"
)
RUNTIME_CONTRACT_OBLIGATION_VERSION: Final = "1"
RUNTIME_LOGIC_IR_DOMAIN: Final = "ipfs-accelerate/runtime-contract-obligation"
RUNTIME_LOGIC_IR_SCHEMA_VERSION: Final = "runtime-contract-logic/v1"
RUNTIME_OBLIGATION_COMPILER_ID: Final = (
    "runtime-contract-obligation-compiler@1"
)
RUNTIME_CATALOG_VERSION: Final = "1"

_MAX_IDENTIFIER_BYTES: Final = 2_048
_MAX_COUNTEREXAMPLES: Final = 32
_CLAIM_FIELDS: Final = frozenset(
    {
        "schema",
        "claim_id",
        "family",
        "state",
        "subject_id",
        "property_id",
        "component_ids",
        "premise_ids",
        "bound_ids",
        "reason_codes",
        "counterexamples",
    }
)


class RuntimeContractObligationError(ValueError):
    """A claim cannot safely be compiled into a reviewed runtime obligation."""


class RuntimeClaimFamily(str, Enum):
    """Closed runtime claim families compiled by this adapter."""

    LIFECYCLE = "lifecycle"
    SCHEMA = "schema"
    REACHABILITY = "reachability"
    DOMINANCE = "dominance"
    TEMPORAL = "temporal"
    CONSERVATION = "conservation"
    IDEMPOTENCE = "idempotence"
    BOUNDED_CONCURRENCY = "bounded_concurrency"


class RuntimeClaimState(str, Enum):
    """Closed, mutually distinct observation states for runtime claims.

    These are analysis / observation dispositions.  ``proved`` does **not**
    mint kernel assurance; it is recorded as an open claim awaiting independent
    discharge.  ``unknown`` covers unmodeled or unsupported program semantics
    that must never be silently treated as refutation.
    """

    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    TIMED_OUT = "timed_out"


class RuntimeCounterexampleKind(str, Enum):
    """What structural element a compact counterexample identifies."""

    EDGE = "edge"
    TRANSITION = "transition"
    INVARIANT = "invariant"


class LogicFragment(str, Enum):
    """Closed solver-neutral fragments used by runtime obligations."""

    GRAPH = "graph"
    RELATION = "relation"
    SCHEMA = "schema"
    DEONTIC = "deontic"
    TEMPORAL = "temporal"
    BOUNDED_CONCURRENCY = "bounded_concurrency"
    UNSUPPORTED = "unsupported"

    # Descriptive aliases retained for callers that name the proof technique.
    GRAPH_REACHABILITY = "graph"
    RELATIONAL = "relation"
    JSON_SCHEMA = "schema"
    DEONTIC_ORDER = "deontic"
    BOUNDED_TEMPORAL = "temporal"
    STATE_MACHINE = "graph"


class LogicOperator(str, Enum):
    """Reviewed operators; there is deliberately no free-form predicate."""

    LIFECYCLE_EDGE_LEGAL = "lifecycle_edge_legal"
    SCHEMA_MATCHES = "schema_matches"
    INVOCATION_REACHABLE = "invocation_reachable"
    POLICY_DOMINATES_EFFECT = "policy_dominates_effect"
    TEMPORAL_INVARIANT_HOLDS = "temporal_invariant_holds"
    QUEUE_CONSERVED = "queue_conserved"
    OPERATION_IDEMPOTENT = "operation_idempotent"
    BOUNDED_INTERLEAVING_SAFE = "bounded_interleaving_safe"
    UNSUPPORTED = "unsupported"


_FAMILY_LOGIC: Final[
    Mapping[RuntimeClaimFamily, tuple[LogicFragment, LogicOperator]]
] = MappingProxyType(
    {
        RuntimeClaimFamily.LIFECYCLE: (
            LogicFragment.GRAPH,
            LogicOperator.LIFECYCLE_EDGE_LEGAL,
        ),
        RuntimeClaimFamily.SCHEMA: (
            LogicFragment.SCHEMA,
            LogicOperator.SCHEMA_MATCHES,
        ),
        RuntimeClaimFamily.REACHABILITY: (
            LogicFragment.GRAPH,
            LogicOperator.INVOCATION_REACHABLE,
        ),
        RuntimeClaimFamily.DOMINANCE: (
            LogicFragment.DEONTIC,
            LogicOperator.POLICY_DOMINATES_EFFECT,
        ),
        RuntimeClaimFamily.TEMPORAL: (
            LogicFragment.TEMPORAL,
            LogicOperator.TEMPORAL_INVARIANT_HOLDS,
        ),
        RuntimeClaimFamily.CONSERVATION: (
            LogicFragment.RELATION,
            LogicOperator.QUEUE_CONSERVED,
        ),
        RuntimeClaimFamily.IDEMPOTENCE: (
            LogicFragment.RELATION,
            LogicOperator.OPERATION_IDEMPOTENT,
        ),
        RuntimeClaimFamily.BOUNDED_CONCURRENCY: (
            LogicFragment.BOUNDED_CONCURRENCY,
            LogicOperator.BOUNDED_INTERLEAVING_SAFE,
        ),
    }
)

_CODE_FAMILY: Final[Mapping[RuntimeClaimFamily, ClaimFamily]] = MappingProxyType(
    {
        RuntimeClaimFamily.LIFECYCLE: ClaimFamily.SUPERVISOR_LIFECYCLE,
        RuntimeClaimFamily.SCHEMA: ClaimFamily.API_CONTRACT,
        RuntimeClaimFamily.REACHABILITY: ClaimFamily.DEPENDENCY_REACHABILITY,
        RuntimeClaimFamily.DOMINANCE: ClaimFamily.SECURITY_PROPERTY,
        RuntimeClaimFamily.TEMPORAL: ClaimFamily.BEHAVIORAL_INVARIANT,
        RuntimeClaimFamily.CONSERVATION: ClaimFamily.BEHAVIORAL_INVARIANT,
        RuntimeClaimFamily.IDEMPOTENCE: ClaimFamily.BEHAVIORAL_INVARIANT,
        RuntimeClaimFamily.BOUNDED_CONCURRENCY: ClaimFamily.BEHAVIORAL_INVARIANT,
    }
)

_UNKNOWN_SEMANTICS_REASONS: Final[frozenset[str]] = frozenset(
    {
        "dynamic_unresolved",
        "incomplete_state_machine",
        "observation_only",
        "program_semantics_unknown",
        "unmodeled_transition",
        "unsupported_program_semantics",
    }
)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise RuntimeContractObligationError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise RuntimeContractObligationError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise RuntimeContractObligationError(f"{name} is required")
    if len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
        raise RuntimeContractObligationError(f"{name} is oversized")
    return value


def _identifier(value: Any, name: str) -> str:
    """Validate an identifier and reject inline source/graph/corpus material."""

    result = _text(value, name)
    lowered = result.lower()
    if (
        any(character.isspace() for character in result)
        or result.startswith(("{", "["))
        or lowered.startswith(
            (
                "def ",
                "class ",
                "function ",
                "import ",
                "from ",
                "<graph",
                "digraph ",
                "source:",
            )
        )
    ):
        raise RuntimeContractObligationError(
            f"{name} must be a compact identifier, not source or graph data"
        )
    return result


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise RuntimeContractObligationError(
            f"{name} must be a sequence of compact identifiers"
        )
    result = tuple(sorted({_identifier(item, name) for item in values}))
    if required and not result:
        raise RuntimeContractObligationError(f"{name} must not be empty")
    return result


def _assurance(value: AssuranceLevel | str) -> AssuranceLevel:
    if isinstance(value, AssuranceLevel):
        return value
    try:
        return AssuranceLevel(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeContractObligationError(
            f"unknown required_assurance: {value!r}"
        ) from exc


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeContractObligationError(
            f"unknown {name}: {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class RuntimeCounterexample:
    """Compact witness identifying a failed edge, transition, or invariant."""

    kind: RuntimeCounterexampleKind
    subject_id: str
    reason_code: str
    failed_edge: str = ""
    failed_transition: str = ""
    failed_invariant: str = ""
    expected: str = ""
    actual: str = ""
    premise_ids: tuple[str, ...] = ()
    counterexample_id: str = ""

    def __post_init__(self) -> None:
        kind = _enum(self.kind, RuntimeCounterexampleKind, "kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "subject_id", _identifier(self.subject_id, "subject_id")
        )
        object.__setattr__(
            self, "reason_code", _identifier(self.reason_code, "reason_code")
        )
        for name in (
            "failed_edge",
            "failed_transition",
            "failed_invariant",
            "expected",
            "actual",
        ):
            raw = getattr(self, name)
            object.__setattr__(
                self,
                name,
                _identifier(raw, name) if raw else "",
            )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids")
        )
        if kind is RuntimeCounterexampleKind.EDGE and not self.failed_edge:
            raise RuntimeContractObligationError(
                "edge counterexample requires failed_edge"
            )
        if (
            kind is RuntimeCounterexampleKind.TRANSITION
            and not self.failed_transition
        ):
            raise RuntimeContractObligationError(
                "transition counterexample requires failed_transition"
            )
        if (
            kind is RuntimeCounterexampleKind.INVARIANT
            and not self.failed_invariant
        ):
            raise RuntimeContractObligationError(
                "invariant counterexample requires failed_invariant"
            )
        expected_id = content_identity(self._identity_payload())
        if self.counterexample_id and self.counterexample_id != expected_id:
            raise RuntimeContractObligationError(
                "counterexample identity does not match canonical content"
            )
        object.__setattr__(self, "counterexample_id", expected_id)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_COUNTEREXAMPLE_SCHEMA,
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "reason_code": self.reason_code,
            "failed_edge": self.failed_edge,
            "failed_transition": self.failed_transition,
            "failed_invariant": self.failed_invariant,
            "expected": self.expected,
            "actual": self.actual,
            "premise_ids": list(self.premise_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "counterexample_id": self.counterexample_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeCounterexample":
        if not isinstance(value, Mapping):
            raise RuntimeContractObligationError(
                "counterexample must be an object"
            )
        allowed = {
            "schema",
            "kind",
            "subject_id",
            "reason_code",
            "failed_edge",
            "failed_transition",
            "failed_invariant",
            "expected",
            "actual",
            "premise_ids",
            "counterexample_id",
        }
        if set(value).difference(allowed):
            raise RuntimeContractObligationError(
                "counterexample contains unsupported fields"
            )
        if value.get("schema") not in (None, RUNTIME_COUNTEREXAMPLE_SCHEMA):
            raise RuntimeContractObligationError(
                "unsupported counterexample schema"
            )
        return cls(
            kind=value.get("kind", ""),
            subject_id=str(value.get("subject_id") or ""),
            reason_code=str(value.get("reason_code") or ""),
            failed_edge=str(value.get("failed_edge") or ""),
            failed_transition=str(value.get("failed_transition") or ""),
            failed_invariant=str(value.get("failed_invariant") or ""),
            expected=str(value.get("expected") or ""),
            actual=str(value.get("actual") or ""),
            premise_ids=tuple(value.get("premise_ids") or ()),
            counterexample_id=str(value.get("counterexample_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class RuntimeContractClaim:
    """One runtime state-machine claim with exact premises and witnesses."""

    family: RuntimeClaimFamily
    state: RuntimeClaimState
    subject_id: str
    property_id: str
    premise_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    component_ids: tuple[str, ...] = ()
    bound_ids: tuple[str, ...] = ()
    counterexamples: tuple[RuntimeCounterexample, ...] = ()
    claim_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _enum(self.family, RuntimeClaimFamily, "family"),
        )
        object.__setattr__(
            self,
            "state",
            _enum(self.state, RuntimeClaimState, "state"),
        )
        object.__setattr__(
            self, "subject_id", _identifier(self.subject_id, "subject_id")
        )
        object.__setattr__(
            self, "property_id", _identifier(self.property_id, "property_id")
        )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        if not self.reason_codes:
            raise RuntimeContractObligationError(
                "claim requires at least one reason_code"
            )
        object.__setattr__(
            self, "component_ids", _ids(self.component_ids, "component_ids")
        )
        object.__setattr__(
            self, "bound_ids", _ids(self.bound_ids, "bound_ids")
        )
        items = tuple(
            item
            if isinstance(item, RuntimeCounterexample)
            else RuntimeCounterexample.from_dict(item)
            for item in self.counterexamples
        )
        if len(items) > _MAX_COUNTEREXAMPLES:
            raise RuntimeContractObligationError(
                "claim exceeds counterexample bound"
            )
        by_id = {item.counterexample_id: item for item in items}
        object.__setattr__(
            self,
            "counterexamples",
            tuple(by_id[key] for key in sorted(by_id)),
        )
        if self.state is RuntimeClaimState.PROVED and self.counterexamples:
            raise RuntimeContractObligationError(
                "proved claim cannot contain counterexamples"
            )
        if self.state is RuntimeClaimState.REFUTED and not self.counterexamples:
            raise RuntimeContractObligationError(
                "refuted claim requires a compact counterexample"
            )
        if self.state is RuntimeClaimState.REFUTED and any(
            code in _UNKNOWN_SEMANTICS_REASONS for code in self.reason_codes
        ):
            raise RuntimeContractObligationError(
                "unsupported program semantics must remain unknown, not refuted"
            )
        expected = content_identity(self._identity_payload())
        if self.claim_id and self.claim_id != expected:
            raise RuntimeContractObligationError(
                "claim identity does not match canonical content"
            )
        object.__setattr__(self, "claim_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_CLAIM_SCHEMA,
            "family": self.family.value,
            "state": self.state.value,
            "subject_id": self.subject_id,
            "property_id": self.property_id,
            "component_ids": list(self.component_ids),
            "premise_ids": list(self.premise_ids),
            "bound_ids": list(self.bound_ids),
            "reason_codes": list(self.reason_codes),
            "counterexamples": [item.to_dict() for item in self.counterexamples],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "claim_id": self.claim_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeContractClaim":
        if not isinstance(value, Mapping):
            raise RuntimeContractObligationError("claim must be an object")
        if set(value).difference(_CLAIM_FIELDS | {"claim_id"}):
            raise RuntimeContractObligationError(
                "claim contains unsupported fields; source/theorem payloads are not accepted"
            )
        if value.get("schema") not in (None, RUNTIME_CLAIM_SCHEMA):
            raise RuntimeContractObligationError("unsupported claim schema")
        return cls(
            family=value.get("family", ""),
            state=value.get("state", ""),
            subject_id=str(value.get("subject_id") or ""),
            property_id=str(value.get("property_id") or ""),
            premise_ids=tuple(value.get("premise_ids") or ()),
            reason_codes=tuple(value.get("reason_codes") or ()),
            component_ids=tuple(value.get("component_ids") or ()),
            bound_ids=tuple(value.get("bound_ids") or ()),
            counterexamples=tuple(value.get("counterexamples") or ()),
            claim_id=str(value.get("claim_id") or ""),
        )


def _claim(value: RuntimeContractClaim | Mapping[str, Any]) -> RuntimeContractClaim:
    if isinstance(value, RuntimeContractClaim):
        return value
    if not isinstance(value, Mapping):
        raise RuntimeContractObligationError(
            "claim must be a RuntimeContractClaim or canonical mapping"
        )
    if set(value).difference(_CLAIM_FIELDS | {"claim_id"}):
        raise RuntimeContractObligationError(
            "claim contains unsupported fields; source/theorem payloads are not accepted"
        )
    return RuntimeContractClaim.from_dict(value)


def _logic_for_claim(
    claim: RuntimeContractClaim,
) -> tuple[LogicFragment, LogicOperator, str]:
    if claim.state is RuntimeClaimState.UNSUPPORTED:
        reason = (
            claim.reason_codes[0] if claim.reason_codes else "unsupported_claim"
        )
        return LogicFragment.UNSUPPORTED, LogicOperator.UNSUPPORTED, reason
    try:
        fragment, operator = _FAMILY_LOGIC[claim.family]
    except KeyError:
        return (
            LogicFragment.UNSUPPORTED,
            LogicOperator.UNSUPPORTED,
            "unsupported_claim_family",
        )
    return fragment, operator, ""


def _claim_status(
    state: RuntimeClaimState,
    supported: bool,
) -> ClaimStatus:
    """Map observation state onto CodeClaimRecord lifecycle without collapsing.

    Distinct observation states remain recoverable via claim metadata
    (``observation_state``).  ``proved`` never mints kernel assurance.
    """

    if not supported or state is RuntimeClaimState.UNSUPPORTED:
        return ClaimStatus.UNSUPPORTED
    if state is RuntimeClaimState.REFUTED:
        return ClaimStatus.REFUTED
    if state is RuntimeClaimState.UNKNOWN:
        return ClaimStatus.UNKNOWN
    if state is RuntimeClaimState.TIMED_OUT:
        return ClaimStatus.NOT_MEASURED
    # proved (and any residual open observation) remains open for discharge.
    return ClaimStatus.OPEN


def _resolve_catalog(
    catalog: RuntimeComponentCatalog | Mapping[str, Any] | None,
    catalog_id: str = "",
    catalog_version: str = "",
) -> tuple[str, str]:
    if catalog is None:
        if not catalog_id:
            raise RuntimeContractObligationError(
                "runtime catalog or catalog_id is required"
            )
        return (
            _identifier(catalog_id, "catalog_id"),
            _identifier(
                catalog_version or RUNTIME_CATALOG_VERSION,
                "catalog_version",
            ),
        )
    if isinstance(catalog, RuntimeComponentCatalog):
        cid = _identifier(catalog.catalog_cid, "catalog_id")
        if catalog_id and _identifier(catalog_id, "catalog_id") != cid:
            raise RuntimeContractObligationError(
                "catalog_id does not match the supplied RuntimeComponentCatalog"
            )
        version = _identifier(
            catalog_version or RUNTIME_CATALOG_VERSION,
            "catalog_version",
        )
        return cid, version
    if isinstance(catalog, Mapping):
        cid = (
            catalog.get("catalog_cid")
            or catalog.get("catalogCid")
            or catalog_id
        )
        version = (
            catalog.get("catalog_version")
            or catalog.get("catalogVersion")
            or catalog_version
            or RUNTIME_CATALOG_VERSION
        )
        if not cid:
            raise RuntimeContractObligationError(
                "catalog mapping requires catalog_cid"
            )
        return (
            _identifier(cid, "catalog_id"),
            _identifier(version, "catalog_version"),
        )
    raise RuntimeContractObligationError(
        "catalog must be a RuntimeComponentCatalog, mapping, or None"
    )


@dataclass(frozen=True, slots=True)
class RuntimeLogicView:
    """Canonical structured theorem view under the shared logic-IR profile."""

    family: RuntimeClaimFamily
    fragment: LogicFragment
    operator: LogicOperator
    subject_id: str
    property_id: str
    claim_id: str
    premise_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    bound_ids: tuple[str, ...] = ()
    supported: bool = True
    unsupported_reason: str = ""

    def __post_init__(self) -> None:
        try:
            family = (
                self.family
                if isinstance(self.family, RuntimeClaimFamily)
                else RuntimeClaimFamily(str(self.family))
            )
            fragment = (
                self.fragment
                if isinstance(self.fragment, LogicFragment)
                else LogicFragment(str(self.fragment))
            )
            operator = (
                self.operator
                if isinstance(self.operator, LogicOperator)
                else LogicOperator(str(self.operator))
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeContractObligationError(
                "logic view uses an unknown family, fragment, or operator"
            ) from exc
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "fragment", fragment)
        object.__setattr__(self, "operator", operator)
        for name in ("subject_id", "property_id", "claim_id"):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self, "premise_ids", _ids(self.premise_ids, "premise_ids")
        )
        object.__setattr__(
            self, "assumption_ids", _ids(self.assumption_ids, "assumption_ids")
        )
        object.__setattr__(
            self, "bound_ids", _ids(self.bound_ids, "bound_ids")
        )
        reason = (
            _identifier(self.unsupported_reason, "unsupported_reason")
            if self.unsupported_reason
            else ""
        )
        object.__setattr__(self, "unsupported_reason", reason)
        expected_supported = fragment is not LogicFragment.UNSUPPORTED
        if bool(self.supported) != expected_supported:
            raise RuntimeContractObligationError(
                "supported flag must agree with the logic fragment"
            )
        if expected_supported:
            expected = _FAMILY_LOGIC.get(family)
            if expected != (fragment, operator):
                raise RuntimeContractObligationError(
                    "family is not bound to its reviewed logic operator"
                )
            if reason:
                raise RuntimeContractObligationError(
                    "supported logic view cannot carry unsupported_reason"
                )
        elif operator is not LogicOperator.UNSUPPORTED or not reason:
            raise RuntimeContractObligationError(
                "unsupported logic view requires an explicit reason"
            )

    def expression_dict(self) -> dict[str, Any]:
        """Return the closed theorem expression (never caller-authored prose)."""

        return {
            "schema": RUNTIME_LOGIC_EXPRESSION_SCHEMA,
            "operator": self.operator.value,
            "terms": {
                "claim_id": self.claim_id,
                "subject_id": self.subject_id,
                "property_id": self.property_id,
                "bound_ids": list(self.bound_ids),
            },
        }

    @property
    def statement(self) -> str:
        return canonical_json(self.expression_dict())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_LOGIC_VIEW_SCHEMA,
            "version": RUNTIME_CONTRACT_OBLIGATION_VERSION,
            "family": self.family.value,
            "fragment": self.fragment.value,
            "expression": self.expression_dict(),
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "bound_ids": list(self.bound_ids),
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
        }

    @property
    def identity(self) -> Any:
        return identify_logic_ir(
            self._identity_payload(),
            domain=RUNTIME_LOGIC_IR_DOMAIN,
            schema_version=RUNTIME_LOGIC_IR_SCHEMA_VERSION,
        )

    @property
    def logic_id(self) -> str:
        return self.identity.cid

    @property
    def content_id(self) -> str:
        return self.logic_id

    @property
    def identity_profile(self) -> str:
        return LOGIC_IR_PROFILE

    def to_dict(self) -> dict[str, Any]:
        identity = self.identity
        return {
            **self._identity_payload(),
            "logic_id": identity.cid,
            "identity": identity.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeLogicView":
        if not isinstance(value, Mapping):
            raise RuntimeContractObligationError("logic view must be an object")
        allowed = {
            "schema",
            "version",
            "family",
            "fragment",
            "expression",
            "premise_ids",
            "assumption_ids",
            "bound_ids",
            "supported",
            "unsupported_reason",
            "logic_id",
            "identity",
        }
        if set(value).difference(allowed):
            raise RuntimeContractObligationError(
                "logic view contains unsupported fields"
            )
        if value.get("schema") not in (None, RUNTIME_LOGIC_VIEW_SCHEMA):
            raise RuntimeContractObligationError(
                "unsupported logic-view schema"
            )
        if value.get("version") not in (
            None,
            RUNTIME_CONTRACT_OBLIGATION_VERSION,
        ):
            raise RuntimeContractObligationError(
                "unsupported logic-view version"
            )
        expression = value.get("expression")
        if not isinstance(expression, Mapping):
            raise RuntimeContractObligationError(
                "logic view requires a structured expression"
            )
        if set(expression) != {"schema", "operator", "terms"}:
            raise RuntimeContractObligationError(
                "logic expression must use the reviewed closed shape"
            )
        if expression.get("schema") != RUNTIME_LOGIC_EXPRESSION_SCHEMA:
            raise RuntimeContractObligationError(
                "unsupported logic-expression schema"
            )
        terms = expression.get("terms")
        if not isinstance(terms, Mapping):
            raise RuntimeContractObligationError(
                "logic expression requires compact terms"
            )
        if set(terms) != {
            "claim_id",
            "subject_id",
            "property_id",
            "bound_ids",
        }:
            raise RuntimeContractObligationError(
                "logic expression contains unsupported terms"
            )
        result = cls(
            family=value.get("family", ""),
            fragment=value.get("fragment", ""),
            operator=expression.get("operator", ""),
            subject_id=terms.get("subject_id", ""),
            property_id=terms.get("property_id", ""),
            claim_id=terms.get("claim_id", ""),
            premise_ids=tuple(value.get("premise_ids") or ()),
            assumption_ids=tuple(value.get("assumption_ids") or ()),
            bound_ids=tuple(value.get("bound_ids") or terms.get("bound_ids") or ()),
            supported=bool(value.get("supported", False)),
            unsupported_reason=str(value.get("unsupported_reason") or ""),
        )
        claimed_id = value.get("logic_id")
        if claimed_id is not None and claimed_id != result.logic_id:
            raise RuntimeContractObligationError(
                "logic-view identity does not match canonical content"
            )
        identity = value.get("identity")
        if identity is not None:
            if not isinstance(identity, Mapping):
                raise RuntimeContractObligationError(
                    "identity must be an object"
                )
            expected = result.identity
            for key, expected_value in (
                ("profile", expected.profile),
                ("cid", expected.cid),
                ("digest", expected.digest),
                ("domain", expected.domain),
            ):
                if identity.get(key) != expected_value:
                    raise RuntimeContractObligationError(
                        "logic-view identity metadata mismatch"
                    )
        return result

    @classmethod
    def from_json(cls, value: str) -> "RuntimeLogicView":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeContractObligationError(
                "logic-view JSON is malformed"
            ) from exc
        return cls.from_dict(payload)


def _load_shared_ir_claim(value: Mapping[str, Any]) -> Any:
    try:
        from ipfs_datasets_py.logic.ir_core.claims import IRClaim
    except ImportError as exc:
        raise RuntimeContractObligationError(
            "ipfs_datasets_py shared logic IR is unavailable"
        ) from exc
    try:
        return IRClaim.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeContractObligationError(
            f"invalid shared logic IR claim: {exc}"
        ) from exc


def _build_shared_ir(
    logic_view: RuntimeLogicView,
    *,
    catalog_id: str,
    catalog_version: str,
    repository_id: str,
    snapshot_id: str,
    scope_ids: tuple[str, ...],
    toolchain_id: str,
    policy_id: str,
    required_assurance: AssuranceLevel,
) -> Any:
    try:
        from ipfs_datasets_py.logic.ir_core.claims import (
            Assumption,
            FrozenMap,
            IRClaim,
            ProofObligation,
        )
    except ImportError as exc:
        raise RuntimeContractObligationError(
            "ipfs_datasets_py shared logic IR is unavailable"
        ) from exc

    assumptions = tuple(
        Assumption(
            assumption_id=assumption_id,
            statement=canonical_json(
                {
                    "operator": "assumption_ref",
                    "terms": [assumption_id],
                }
            ),
            source_refs=(assumption_id,),
        )
        for assumption_id in logic_view.assumption_ids
    )
    obligation = ProofObligation(
        obligation_id=logic_view.logic_id,
        statement=logic_view.statement,
        assumption_ids=logic_view.assumption_ids,
        logic_family=logic_view.fragment.value,
        source_refs=logic_view.premise_ids,
        metadata=FrozenMap(
            {
                "bound_ids": list(logic_view.bound_ids),
                "catalog_id": catalog_id,
                "catalog_version": catalog_version,
                "identity_profile": LOGIC_IR_PROFILE,
                "policy_id": policy_id,
                "property_id": logic_view.property_id,
                "repository_id": repository_id,
                "required_assurance": required_assurance.value,
                "scope_ids": list(scope_ids),
                "snapshot_id": snapshot_id,
                "supported": logic_view.supported,
                "toolchain_id": toolchain_id,
                "unsupported_reason": logic_view.unsupported_reason,
            }
        ),
    )
    return IRClaim(
        claim_id=logic_view.claim_id,
        declaration_id=logic_view.property_id,
        statement=logic_view.statement,
        assumptions=assumptions,
        obligations=(obligation,),
        domain=RUNTIME_LOGIC_IR_DOMAIN,
        source_refs=tuple(
            sorted(set(logic_view.premise_ids) | {logic_view.property_id})
        ),
        metadata=FrozenMap(
            {
                "logic_id": logic_view.logic_id,
                "subject_id": logic_view.subject_id,
                "schema": RUNTIME_LOGIC_VIEW_SCHEMA,
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimeContractObligation:
    """Bound canonical logic, code-proof, and claim-lifecycle projections."""

    logic_view: RuntimeLogicView
    code_obligation: CodeProofObligation
    code_claim: CodeClaimRecord
    shared_ir_claim: Any
    catalog_id: str
    catalog_version: str
    property_id: str
    toolchain_id: str
    policy_id: str
    bound_ids: tuple[str, ...]
    observation_state: RuntimeClaimState
    counterexamples: tuple[RuntimeCounterexample, ...]
    invalidators: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.logic_view, RuntimeLogicView):
            raise RuntimeContractObligationError(
                "logic_view must be a RuntimeLogicView"
            )
        if not isinstance(self.code_obligation, CodeProofObligation):
            raise RuntimeContractObligationError(
                "code_obligation must be a CodeProofObligation"
            )
        if not isinstance(self.code_claim, CodeClaimRecord):
            raise RuntimeContractObligationError(
                "code_claim must be a CodeClaimRecord"
            )
        for name in (
            "catalog_id",
            "catalog_version",
            "property_id",
            "toolchain_id",
            "policy_id",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self, "bound_ids", _ids(self.bound_ids, "bound_ids")
        )
        object.__setattr__(
            self,
            "observation_state",
            _enum(self.observation_state, RuntimeClaimState, "observation_state"),
        )
        counterexamples = tuple(
            item
            if isinstance(item, RuntimeCounterexample)
            else RuntimeCounterexample.from_dict(item)
            for item in self.counterexamples
        )
        by_id = {item.counterexample_id: item for item in counterexamples}
        object.__setattr__(
            self,
            "counterexamples",
            tuple(by_id[key] for key in sorted(by_id)),
        )
        shared = self.shared_ir_claim
        if isinstance(shared, Mapping):
            shared = _load_shared_ir_claim(shared)
            object.__setattr__(self, "shared_ir_claim", shared)
        if not callable(getattr(shared, "to_dict", None)):
            raise RuntimeContractObligationError(
                "shared_ir_claim must be an ipfs_datasets IRClaim"
            )
        invalidators = tuple(
            MappingProxyType(
                {
                    "kind": _identifier(
                        item.get("kind", ""), "invalidator.kind"
                    ),
                    "reason_code": _text(
                        item.get("reason_code", ""),
                        "invalidator.reason_code",
                        required=False,
                    ),
                    "source": _identifier(
                        item.get("source", ""), "invalidator.source"
                    ),
                    "value": _identifier(
                        item.get("value", ""), "invalidator.value"
                    ),
                }
            )
            for item in self.invalidators
        )
        if not invalidators:
            raise RuntimeContractObligationError(
                "compiled obligation requires invalidators"
            )
        compiler_kinds = {
            item["kind"]
            for item in invalidators
            if item["source"] == "compiler"
        }
        required_compiler_kinds = {
            "assumption_set",
            "bound_set",
            "catalog",
            "policy",
            "premise_set",
            "required_assurance",
            "scope_set",
            "snapshot",
            "toolchain",
        }
        if not required_compiler_kinds.issubset(compiler_kinds):
            raise RuntimeContractObligationError(
                "compiled obligation has incomplete binding invalidators"
            )
        object.__setattr__(
            self,
            "invalidators",
            tuple(
                sorted(
                    invalidators,
                    key=lambda item: (
                        item["source"],
                        item["kind"],
                        item["value"],
                        item["reason_code"],
                    ),
                )
            ),
        )
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        logic = self.logic_view
        code = self.code_obligation
        claim = self.code_claim
        shared = self.shared_ir_claim
        metadata = code.metadata
        if (
            code.statement != logic.statement
            or code.premise_ids != logic.premise_ids
            or claim.premise_ids != logic.premise_ids
            or claim.assumption_ids != logic.assumption_ids
            or claim.property_id != logic.property_id
            or claim.obligation_id != code.obligation_id
            or claim.repository_id != code.repository_id
            or claim.repository_tree_id != code.repository_tree_id
            or claim.scope_ids != code.ast_scope_ids
            or claim.required_assurance is not code.required_assurance
            or claim.toolchain_id != self.toolchain_id
            or claim.policy_id != self.policy_id
            or claim.catalog_version != self.catalog_version
            or self.property_id != logic.property_id
            or self.bound_ids != logic.bound_ids
        ):
            raise RuntimeContractObligationError(
                "logic, code obligation, and claim bindings disagree"
            )
        required_metadata = {
            "assumption_ids": list(logic.assumption_ids),
            "bound_ids": list(self.bound_ids),
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "logic_id": logic.logic_id,
            "logic_fragment": logic.fragment.value,
            "observation_state": self.observation_state.value,
            "policy_id": self.policy_id,
            "property_id": self.property_id,
            "snapshot_id": code.repository_tree_id,
            "supported": logic.supported,
            "toolchain_id": self.toolchain_id,
        }
        if any(metadata.get(key) != value for key, value in required_metadata.items()):
            raise RuntimeContractObligationError(
                "code obligation omits or changes a mandatory binding"
            )
        shared_obligation = shared.obligations[0]
        shared_metadata = shared_obligation.metadata.to_dict()
        required_shared_metadata = {
            "bound_ids": list(self.bound_ids),
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "identity_profile": LOGIC_IR_PROFILE,
            "policy_id": self.policy_id,
            "property_id": self.property_id,
            "repository_id": code.repository_id,
            "required_assurance": code.required_assurance.value,
            "scope_ids": list(code.ast_scope_ids),
            "snapshot_id": code.repository_tree_id,
            "supported": logic.supported,
            "toolchain_id": self.toolchain_id,
            "unsupported_reason": logic.unsupported_reason,
        }
        if (
            shared.claim_id != logic.claim_id
            or shared_obligation.obligation_id != logic.logic_id
            or tuple(shared_obligation.assumption_ids) != logic.assumption_ids
            or tuple(shared_obligation.source_refs) != logic.premise_ids
            or shared_obligation.statement != logic.statement
            or any(
                shared_metadata.get(key) != value
                for key, value in required_shared_metadata.items()
            )
        ):
            raise RuntimeContractObligationError(
                "shared logic IR is detached from the canonical logic view"
            )
        if (
            self.observation_state is RuntimeClaimState.REFUTED
            and not self.counterexamples
        ):
            raise RuntimeContractObligationError(
                "refuted obligation requires compact counterexamples"
            )
        for item in self.counterexamples:
            if not (
                item.failed_edge
                or item.failed_transition
                or item.failed_invariant
            ):
                raise RuntimeContractObligationError(
                    "counterexample must identify failed edge, transition, or invariant"
                )

    @property
    def obligation_id(self) -> str:
        return self.code_obligation.obligation_id

    @property
    def premise_ids(self) -> tuple[str, ...]:
        return self.logic_view.premise_ids

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        return self.logic_view.assumption_ids

    @property
    def snapshot_id(self) -> str:
        return self.code_obligation.repository_tree_id

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return self.code_obligation.ast_scope_ids

    @property
    def required_assurance(self) -> AssuranceLevel:
        return self.code_obligation.required_assurance

    @property
    def logic_fragment(self) -> LogicFragment:
        return self.logic_view.fragment

    @property
    def supported(self) -> bool:
        return self.logic_view.supported

    @property
    def code_proof_obligation(self) -> CodeProofObligation:
        """Descriptive alias for callers using the full interface name."""

        return self.code_obligation

    @property
    def claim_record(self) -> CodeClaimRecord:
        return self.code_claim

    @property
    def logic_ir(self) -> Any:
        return self.shared_ir_claim

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_CONTRACT_OBLIGATION_SCHEMA,
            "interface": RUNTIME_CONTRACT_OBLIGATIONS_INTERFACE,
            "version": RUNTIME_CONTRACT_OBLIGATION_VERSION,
            "logic_view": self.logic_view.to_dict(),
            "code_obligation": self.code_obligation.to_dict(),
            "code_claim": self.code_claim.to_dict(),
            "shared_ir_claim": self.shared_ir_claim.to_dict(),
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "property_id": self.property_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "bound_ids": list(self.bound_ids),
            "observation_state": self.observation_state.value,
            "counterexamples": [item.to_dict() for item in self.counterexamples],
            "invalidators": [dict(item) for item in self.invalidators],
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def compiled_obligation_id(self) -> str:
        return self.content_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "compiled_obligation_id": self.compiled_obligation_id,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeContractObligation":
        if not isinstance(value, Mapping):
            raise RuntimeContractObligationError(
                "compiled runtime obligation must be an object"
            )
        allowed = {
            "schema",
            "interface",
            "version",
            "logic_view",
            "code_obligation",
            "code_claim",
            "shared_ir_claim",
            "catalog_id",
            "catalog_version",
            "property_id",
            "toolchain_id",
            "policy_id",
            "bound_ids",
            "observation_state",
            "counterexamples",
            "invalidators",
            "compiled_obligation_id",
        }
        if set(value).difference(allowed):
            raise RuntimeContractObligationError(
                "compiled obligation contains unsupported fields"
            )
        if value.get("schema") not in (
            None,
            RUNTIME_CONTRACT_OBLIGATION_SCHEMA,
        ):
            raise RuntimeContractObligationError(
                "unsupported compiled-obligation schema"
            )
        if value.get("interface") not in (
            None,
            RUNTIME_CONTRACT_OBLIGATIONS_INTERFACE,
        ):
            raise RuntimeContractObligationError(
                "unsupported compiled-obligation interface"
            )
        if value.get("version") not in (
            None,
            RUNTIME_CONTRACT_OBLIGATION_VERSION,
        ):
            raise RuntimeContractObligationError(
                "unsupported compiled-obligation version"
            )
        try:
            result = cls(
                logic_view=RuntimeLogicView.from_dict(
                    value.get("logic_view") or {}
                ),
                code_obligation=CodeProofObligation.from_dict(
                    value.get("code_obligation") or {}
                ),
                code_claim=CodeClaimRecord.from_dict(
                    value.get("code_claim") or {}
                ),
                shared_ir_claim=_load_shared_ir_claim(
                    value.get("shared_ir_claim") or {}
                ),
                catalog_id=str(value.get("catalog_id") or ""),
                catalog_version=str(value.get("catalog_version") or ""),
                property_id=str(value.get("property_id") or ""),
                toolchain_id=str(value.get("toolchain_id") or ""),
                policy_id=str(value.get("policy_id") or ""),
                bound_ids=tuple(value.get("bound_ids") or ()),
                observation_state=value.get("observation_state") or "",
                counterexamples=tuple(value.get("counterexamples") or ()),
                invalidators=tuple(value.get("invalidators") or ()),
            )
        except RuntimeContractObligationError:
            raise
        except (TypeError, ValueError) as exc:
            raise RuntimeContractObligationError(
                f"invalid compiled obligation: {exc}"
            ) from exc
        claimed_id = value.get("compiled_obligation_id")
        if (
            claimed_id is not None
            and claimed_id != result.compiled_obligation_id
        ):
            raise RuntimeContractObligationError(
                "compiled-obligation identity does not match canonical content"
            )
        return result

    @classmethod
    def from_json(cls, value: str) -> "RuntimeContractObligation":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeContractObligationError(
                "compiled-obligation JSON is malformed"
            ) from exc
        return cls.from_dict(payload)


def compile_runtime_claim(
    claim: RuntimeContractClaim | Mapping[str, Any],
    *,
    catalog: RuntimeComponentCatalog | Mapping[str, Any] | None = None,
    catalog_id: str = "",
    catalog_version: str = "",
    repository_id: str,
    snapshot_id: str = "",
    repository_tree_id: str = "",
    tree_id: str = "",
    scope_ids: Sequence[str] = (),
    ast_scope_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    bound_ids: Sequence[str] = (),
    toolchain_id: str,
    policy_id: str,
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
) -> RuntimeContractObligation:
    """Compile one runtime claim using only closed, structured templates.

    ``snapshot_id`` and ``repository_tree_id`` are aliases.  If both are
    supplied they must be identical.  Premise and bound order have set
    semantics; the resulting identity is invariant to input order.
    """

    normalized_claim = _claim(claim)
    resolved_catalog_id, resolved_catalog_version = _resolve_catalog(
        catalog,
        catalog_id=catalog_id,
        catalog_version=catalog_version,
    )
    repository = _identifier(repository_id, "repository_id")
    snapshot_values = tuple(
        value for value in (snapshot_id, repository_tree_id, tree_id) if value
    )
    if len(set(snapshot_values)) > 1:
        raise RuntimeContractObligationError(
            "snapshot_id, repository_tree_id, and tree_id disagree"
        )
    snapshot = snapshot_values[0] if snapshot_values else ""
    snapshot = _identifier(snapshot, "snapshot_id")
    if scope_ids and ast_scope_ids and tuple(scope_ids) != tuple(ast_scope_ids):
        raise RuntimeContractObligationError(
            "scope_ids and ast_scope_ids disagree"
        )
    scopes = _ids(scope_ids or ast_scope_ids, "scope_ids", required=True)
    assumptions = _ids(assumption_ids, "assumption_ids")
    bounds = _ids(
        bound_ids if bound_ids else normalized_claim.bound_ids,
        "bound_ids",
    )
    if (
        normalized_claim.family is RuntimeClaimFamily.BOUNDED_CONCURRENCY
        and not bounds
    ):
        raise RuntimeContractObligationError(
            "bounded_concurrency claims require non-empty bound_ids"
        )
    toolchain = _identifier(toolchain_id, "toolchain_id")
    policy = _identifier(policy_id, "policy_id")
    required = _assurance(required_assurance)
    premises = _ids(
        normalized_claim.premise_ids, "premise_ids", required=True
    )

    fragment, operator, unsupported_reason = _logic_for_claim(normalized_claim)
    logic_view = RuntimeLogicView(
        family=normalized_claim.family,
        fragment=fragment,
        operator=operator,
        subject_id=normalized_claim.subject_id,
        property_id=normalized_claim.property_id,
        claim_id=normalized_claim.claim_id,
        premise_ids=premises,
        assumption_ids=assumptions,
        bound_ids=bounds,
        supported=fragment is not LogicFragment.UNSUPPORTED,
        unsupported_reason=unsupported_reason,
    )

    compiler_invalidators = (
        {
            "source": "compiler",
            "kind": "snapshot",
            "value": snapshot,
            "reason_code": "snapshot_changed",
        },
        {
            "source": "compiler",
            "kind": "scope_set",
            "value": content_identity({"scope_ids": list(scopes)}),
            "reason_code": "scope_changed",
        },
        {
            "source": "compiler",
            "kind": "premise_set",
            "value": content_identity({"premise_ids": list(premises)}),
            "reason_code": "premises_changed",
        },
        {
            "source": "compiler",
            "kind": "assumption_set",
            "value": content_identity({"assumption_ids": list(assumptions)}),
            "reason_code": "assumptions_changed",
        },
        {
            "source": "compiler",
            "kind": "bound_set",
            "value": content_identity({"bound_ids": list(bounds)}),
            "reason_code": "bounds_changed",
        },
        {
            "source": "compiler",
            "kind": "catalog",
            "value": resolved_catalog_id,
            "reason_code": "catalog_changed",
        },
        {
            "source": "compiler",
            "kind": "toolchain",
            "value": toolchain,
            "reason_code": "toolchain_changed",
        },
        {
            "source": "compiler",
            "kind": "policy",
            "value": policy,
            "reason_code": "policy_changed",
        },
        {
            "source": "compiler",
            "kind": "required_assurance",
            "value": required.value,
            "reason_code": "required_assurance_changed",
        },
    )
    invalidators = compiler_invalidators
    metadata = {
        "interface": RUNTIME_CONTRACT_OBLIGATIONS_INTERFACE,
        "assumption_ids": list(assumptions),
        "bound_ids": list(bounds),
        "catalog_id": resolved_catalog_id,
        "catalog_version": resolved_catalog_version,
        "claim_id": normalized_claim.claim_id,
        "component_ids": list(normalized_claim.component_ids),
        "identity_profile": LOGIC_IR_PROFILE,
        "invalidators": [dict(item) for item in invalidators],
        "logic_fragment": fragment.value,
        "logic_id": logic_view.logic_id,
        "observation_state": normalized_claim.state.value,
        "policy_id": policy,
        "property_id": normalized_claim.property_id,
        "snapshot_id": snapshot,
        "supported": logic_view.supported,
        "toolchain_id": toolchain,
        "unsupported_reason": unsupported_reason,
        "counterexample_ids": [
            item.counterexample_id for item in normalized_claim.counterexamples
        ],
    }
    code_obligation = CodeProofObligation(
        repository_id=repository,
        repository_tree_id=snapshot,
        ast_scope_ids=scopes,
        statement=logic_view.statement,
        premise_ids=premises,
        template_id=f"runtime-contract/{fragment.value}",
        template_version=RUNTIME_CONTRACT_OBLIGATION_VERSION,
        template_semantic_hash=logic_view.logic_id,
        invariant_class=f"runtime_contract:{normalized_claim.family.value}",
        task_id=normalized_claim.subject_id,
        required_assurance=required,
        fallback_checks=(
            ("runtime-contract:unsupported-fragment",)
            if not logic_view.supported
            else ()
        ),
        metadata=metadata,
    )
    code_claim = claim_from_obligation(
        code_obligation,
        property_id=normalized_claim.property_id,
        claim_family=(
            ClaimFamily.UNSUPPORTED
            if not logic_view.supported
            else _CODE_FAMILY[normalized_claim.family]
        ),
        assumption_ids=assumptions,
        producer_id=RUNTIME_OBLIGATION_COMPILER_ID,
        toolchain_id=toolchain,
        policy_id=policy,
        catalog_version=resolved_catalog_version,
        status=_claim_status(normalized_claim.state, logic_view.supported),
        metadata={
            "bound_ids": list(bounds),
            "catalog_id": resolved_catalog_id,
            "counterexample_ids": [
                item.counterexample_id
                for item in normalized_claim.counterexamples
            ],
            "logic_fragment": fragment.value,
            "logic_id": logic_view.logic_id,
            "observation_state": normalized_claim.state.value,
            "runtime_claim_id": normalized_claim.claim_id,
            "supported": logic_view.supported,
            "unsupported_reason": unsupported_reason,
        },
    )
    # ``claim_from_obligation`` forces unsupported templates to unsupported;
    # for supported templates preserve only the analysis-derived lifecycle.
    if logic_view.supported:
        code_claim = code_claim.with_updates(
            status=_claim_status(normalized_claim.state, True)
        )
    shared_ir = _build_shared_ir(
        logic_view,
        catalog_id=resolved_catalog_id,
        catalog_version=resolved_catalog_version,
        repository_id=repository,
        snapshot_id=snapshot,
        scope_ids=scopes,
        toolchain_id=toolchain,
        policy_id=policy,
        required_assurance=required,
    )
    return RuntimeContractObligation(
        logic_view=logic_view,
        code_obligation=code_obligation,
        code_claim=code_claim,
        shared_ir_claim=shared_ir,
        catalog_id=resolved_catalog_id,
        catalog_version=resolved_catalog_version,
        property_id=normalized_claim.property_id,
        toolchain_id=toolchain,
        policy_id=policy,
        bound_ids=bounds,
        observation_state=normalized_claim.state,
        counterexamples=normalized_claim.counterexamples,
        invalidators=invalidators,
    )


def compile_runtime_claims(
    claims: Sequence[RuntimeContractClaim | Mapping[str, Any]],
    **bindings: Any,
) -> tuple[RuntimeContractObligation, ...]:
    """Compile a deterministic set of claims with shared exact bindings."""

    if isinstance(claims, (str, bytes, bytearray)) or not isinstance(
        claims, Sequence
    ):
        raise RuntimeContractObligationError("claims must be a sequence")
    results = tuple(compile_runtime_claim(item, **bindings) for item in claims)
    by_id = {item.compiled_obligation_id: item for item in results}
    if len(by_id) != len(results):
        raise RuntimeContractObligationError(
            "claims compile to duplicate obligations"
        )
    return tuple(by_id[key] for key in sorted(by_id))
