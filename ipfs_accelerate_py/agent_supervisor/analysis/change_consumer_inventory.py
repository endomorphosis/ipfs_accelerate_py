"""Per-call-site compatibility inventory for transitive contract changes.

RPR-029: given a typed :class:`ProgramContractDelta` and resolved call-site
observations (or a program graph + call resolver), emit one explicit
compatibility disposition and canonical
:class:`ConsumerMigrationObligation` per resolved caller and route.

Rules enforced here:

* Direct, aliased, re-exported, wrapped, decorated, callback, overload,
  method/override, factory, test/mock, and generated-client callers are first
  class.
* Actual positional/keyword/splat arguments, defaults, receiver state, path
  condition, awaitedness, result uses, handled errors/effects/capabilities,
  and the exact dispatch route are recorded on every entry.
* A two-to-three required-argument change flags every still-two-argument
  caller independently; one compatible caller or a callee default cannot
  discharge others.
* Ambiguous/dynamic/external/unsupported resolutions remain frontier records.
* Duplicate exact routes do not duplicate obligations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..program_call_resolver import (
    CallResolution,
    CallResolutionStatus,
    CallSite,
    ProgramCallResolver,
)
from ..program_graph import (
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphSnapshot,
    ProgramNode,
    ProgramNodeKind,
)
from .change_propagation_contracts import (
    CHANGE_PROPAGATION_VERSION,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    GraphNodeRef,
    GraphProvenance,
    MAX_CONSUMER_COUNT,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    obligation_set_identity,
)

# ---------------------------------------------------------------------------
# Schemas / bounds
# ---------------------------------------------------------------------------

CHANGE_CONSUMER_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-consumer-inventory@1"
)
CONSUMER_COMPATIBILITY_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/consumer-compatibility-ledger@1"
)
CALL_SITE_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/call-site-observation@1"
)
CONSUMER_COMPATIBILITY_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/consumer-compatibility-entry@1"
)
CHANGE_CONSUMER_INVENTORY_VERSION: Final[str] = "change-consumer-inventory@1"

MAX_CALL_ARGUMENTS: Final[int] = 256
MAX_RESULT_USES: Final[int] = 128
MAX_ROUTE_HOPS: Final[int] = 64
MAX_FIELD_BYTES: Final[int] = 4_096
MAX_ENTRIES: Final[int] = MAX_CONSUMER_COUNT
MAX_CLAUSE_IDS: Final[int] = 256

# Reverse-call edge kinds used when discovering callers from a program graph.
_CALL_EDGE_KINDS: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        ProgramEdgeKind.CALLS,
        ProgramEdgeKind.OVERRIDES,
        ProgramEdgeKind.IMPLEMENTS,
        ProgramEdgeKind.OVERLOADS,
        ProgramEdgeKind.CONSTRUCTS,
        ProgramEdgeKind.FACTORY_CREATES,
        ProgramEdgeKind.BUILDER_BUILDS,
        ProgramEdgeKind.DECORATES,
        ProgramEdgeKind.REGISTERS,
        ProgramEdgeKind.INJECTS,
        ProgramEdgeKind.CALLBACK_TO,
        ProgramEdgeKind.ALIASES,
        ProgramEdgeKind.RE_EXPORTS,
        ProgramEdgeKind.IMPORTS,
        ProgramEdgeKind.TESTS,
        ProgramEdgeKind.MOCKS,
        ProgramEdgeKind.FIXTURES,
        ProgramEdgeKind.GENERATED_FROM,
    }
)

_REQUIRED_CALLER_KINDS: Final[frozenset[str]] = frozenset(
    {
        "direct",
        "aliased",
        "re_exported",
        "wrapped",
        "decorated",
        "callback",
        "overload",
        "method_override",
        "factory",
        "test_mock",
        "generated_client",
    }
)


class ChangeConsumerInventoryError(ValueError):
    """Malformed inventory input or ledger invariant failure."""


class ChangeConsumerInventoryBoundsError(ChangeConsumerInventoryError):
    """A record exceeded its hard compactness bound."""


class CallerKind(str, Enum):
    """Closed vocabulary of consumer call forms the inventory must cover."""

    DIRECT = "direct"
    ALIASED = "aliased"
    RE_EXPORTED = "re_exported"
    WRAPPED = "wrapped"
    DECORATED = "decorated"
    CALLBACK = "callback"
    OVERLOAD = "overload"
    METHOD_OVERRIDE = "method_override"
    FACTORY = "factory"
    TEST_MOCK = "test_mock"
    GENERATED_CLIENT = "generated_client"


class ArgumentForm(str, Enum):
    """How one argument is presented at the call site."""

    POSITIONAL = "positional"
    KEYWORD = "keyword"
    SPLAT_ARGS = "splat_args"
    SPLAT_KWARGS = "splat_kwargs"
    DEFAULTED = "defaulted"


class RouteStatus(str, Enum):
    """Resolution status for the exact dispatch route to the changed callee."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    DYNAMIC = "dynamic"
    EXTERNAL = "external"
    UNSUPPORTED = "unsupported"


# Map resolver statuses onto the inventory route vocabulary.
_RESOLVER_STATUS_MAP: Final[Mapping[CallResolutionStatus, RouteStatus]] = {
    CallResolutionStatus.RESOLVED: RouteStatus.RESOLVED,
    CallResolutionStatus.AMBIGUOUS: RouteStatus.AMBIGUOUS,
    CallResolutionStatus.DYNAMIC: RouteStatus.DYNAMIC,
    CallResolutionStatus.EXTERNAL: RouteStatus.EXTERNAL,
    CallResolutionStatus.UNSUPPORTED: RouteStatus.UNSUPPORTED,
}

# Frontier statuses cannot claim a closed migrate disposition.
_FRONTIER_ROUTE_STATUSES: Final[frozenset[RouteStatus]] = frozenset(
    {
        RouteStatus.AMBIGUOUS,
        RouteStatus.DYNAMIC,
        RouteStatus.EXTERNAL,
        RouteStatus.UNSUPPORTED,
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise ChangeConsumerInventoryError(f"{name} must be a string")
    if text != text.strip() or "\x00" in text:
        raise ChangeConsumerInventoryError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not text:
        raise ChangeConsumerInventoryError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_FIELD_BYTES:
        raise ChangeConsumerInventoryBoundsError(f"{name} exceeds its byte bound")
    return text


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True)
    if any(character.isspace() for character in result):
        raise ChangeConsumerInventoryError(f"{name} must be a compact identifier")
    return result


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw))
    except (TypeError, ValueError) as exc:
        raise ChangeConsumerInventoryError(f"invalid {name}: {value!r}") from exc


def _string_tuple(
    value: Any,
    name: str,
    *,
    limit: int = MAX_RESULT_USES,
    required: bool = False,
    sort: bool = True,
) -> tuple[str, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, (str, bytes, bytearray)):
        raise ChangeConsumerInventoryError(f"{name} must be a sequence of strings")
    elif isinstance(value, Sequence):
        items = value
    else:
        raise ChangeConsumerInventoryError(f"{name} must be a sequence of strings")
    if len(items) > limit:
        raise ChangeConsumerInventoryBoundsError(f"{name} exceeds its item bound")
    normalized = [
        _text(item, name, required=False)
        for item in items
        if item is not None and str(item).strip()
    ]
    result = tuple(sorted(set(normalized))) if sort else tuple(normalized)
    if required and not result:
        raise ChangeConsumerInventoryError(f"{name} is required")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ChangeConsumerInventoryError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChangeConsumerInventoryError(
            f"{name} must be a non-negative integer"
        )
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    return str(value)


def _roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            PropagationAuthorityRoots.from_dict(value)
            if "schema" in value
            else PropagationAuthorityRoots(**dict(value))
        )
    raise ChangeConsumerInventoryError("roots must be PropagationAuthorityRoots")


def _node_ref(value: Any) -> GraphNodeRef:
    if isinstance(value, GraphNodeRef):
        return value
    if isinstance(value, Mapping):
        return (
            GraphNodeRef.from_dict(value)
            if "schema" in value
            else GraphNodeRef(**dict(value))
        )
    raise ChangeConsumerInventoryError("node must be a GraphNodeRef")


# ---------------------------------------------------------------------------
# Observation records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ActualArgument:
    """One actual argument as observed at a call site (never source text)."""

    position: int
    form: ArgumentForm
    name: str = ""
    type_ref: str = ""
    default_ref: str = ""
    value_ref: str = ""
    evidence_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _nonneg_int(self.position, "position"))
        object.__setattr__(self, "form", _enum(self.form, ArgumentForm, "form"))
        object.__setattr__(self, "name", _text(self.name, "argument name", required=False))
        object.__setattr__(
            self, "type_ref", _text(self.type_ref, "type_ref", required=False)
        )
        object.__setattr__(
            self, "default_ref", _text(self.default_ref, "default_ref", required=False)
        )
        object.__setattr__(
            self, "value_ref", _text(self.value_ref, "value_ref", required=False)
        )
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id", required=False)
        )
        if self.form is ArgumentForm.KEYWORD and not self.name:
            raise ChangeConsumerInventoryError("keyword arguments require a name")
        if self.form is ArgumentForm.DEFAULTED and not (self.name or self.default_ref):
            raise ChangeConsumerInventoryError(
                "defaulted arguments require a name or default_ref"
            )

    @property
    def is_splat(self) -> bool:
        return self.form in {ArgumentForm.SPLAT_ARGS, ArgumentForm.SPLAT_KWARGS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "form": self.form.value,
            "name": self.name,
            "type_ref": self.type_ref,
            "default_ref": self.default_ref,
            "value_ref": self.value_ref,
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActualArgument":
        return cls(
            position=int(payload.get("position") or 0),
            form=payload.get("form") or ArgumentForm.POSITIONAL,
            name=str(payload.get("name") or ""),
            type_ref=str(payload.get("type_ref") or ""),
            default_ref=str(payload.get("default_ref") or ""),
            value_ref=str(payload.get("value_ref") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
        )


@dataclass(frozen=True)
class CallSiteObservation:
    """Exact bounded observation of one call into a changed callee.

    Captures the evidence subset required by RPR-029 acceptance: caller span,
    alias/dispatch route, actual args, defaults, awaitedness, result uses,
    errors/effects, path condition, and consumer contract identifiers.
    """

    consumer_id: str
    caller_kind: CallerKind
    path: str
    symbol_id: str
    callee_symbol_id: str
    actual_arguments: tuple[ActualArgument, ...] = ()
    defaults_applied: tuple[str, ...] = ()
    receiver_state_refs: tuple[str, ...] = ()
    path_condition_ref: str = ""
    awaited: bool = False
    result_uses: tuple[str, ...] = ()
    handled_error_refs: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    route_hops: tuple[str, ...] = ()
    route_status: RouteStatus = RouteStatus.RESOLVED
    callee_default_refs: tuple[str, ...] = ()
    required_argument_count: int | None = None
    provided_argument_count: int | None = None
    supplies_parameter_names: tuple[str, ...] = ()
    node: GraphNodeRef | None = None
    span_ref: str = ""
    call_requirement_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CALL_SITE_OBSERVATION_SCHEMA
    observation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(
            self, "caller_kind", _enum(self.caller_kind, CallerKind, "caller_kind")
        )
        object.__setattr__(self, "path", _text(self.path, "path"))
        if self.path.startswith("/") or ".." in self.path.split("/"):
            raise ChangeConsumerInventoryError(
                "path must be a repository-relative path without parent escapes"
            )
        object.__setattr__(self, "symbol_id", _identifier(self.symbol_id, "symbol_id"))
        object.__setattr__(
            self,
            "callee_symbol_id",
            _identifier(self.callee_symbol_id, "callee_symbol_id"),
        )
        if len(self.actual_arguments) > MAX_CALL_ARGUMENTS:
            raise ChangeConsumerInventoryBoundsError(
                "actual_arguments exceeds its item bound"
            )
        args = tuple(
            item
            if isinstance(item, ActualArgument)
            else ActualArgument.from_dict(item)  # type: ignore[arg-type]
            for item in self.actual_arguments
        )
        positions = [item.position for item in args]
        if len(set(positions)) != len(positions):
            raise ChangeConsumerInventoryError(
                "actual argument positions must be unique"
            )
        object.__setattr__(
            self,
            "actual_arguments",
            tuple(sorted(args, key=lambda item: (item.position, item.name))),
        )
        object.__setattr__(
            self,
            "defaults_applied",
            _string_tuple(self.defaults_applied, "defaults_applied"),
        )
        object.__setattr__(
            self,
            "receiver_state_refs",
            _string_tuple(self.receiver_state_refs, "receiver_state_refs"),
        )
        object.__setattr__(
            self,
            "path_condition_ref",
            _text(self.path_condition_ref, "path_condition_ref", required=False),
        )
        object.__setattr__(self, "awaited", _bool(self.awaited, "awaited"))
        object.__setattr__(
            self,
            "result_uses",
            _string_tuple(self.result_uses, "result_uses", limit=MAX_RESULT_USES),
        )
        object.__setattr__(
            self,
            "handled_error_refs",
            _string_tuple(self.handled_error_refs, "handled_error_refs"),
        )
        object.__setattr__(
            self, "effect_refs", _string_tuple(self.effect_refs, "effect_refs")
        )
        object.__setattr__(
            self,
            "capability_refs",
            _string_tuple(self.capability_refs, "capability_refs"),
        )
        hops = _string_tuple(
            self.route_hops, "route_hops", limit=MAX_ROUTE_HOPS, sort=False
        )
        if len(hops) > MAX_ROUTE_HOPS:
            raise ChangeConsumerInventoryBoundsError("route_hops exceeds its item bound")
        object.__setattr__(self, "route_hops", hops)
        object.__setattr__(
            self, "route_status", _enum(self.route_status, RouteStatus, "route_status")
        )
        object.__setattr__(
            self,
            "callee_default_refs",
            _string_tuple(self.callee_default_refs, "callee_default_refs"),
        )
        if self.required_argument_count is not None:
            object.__setattr__(
                self,
                "required_argument_count",
                _nonneg_int(self.required_argument_count, "required_argument_count"),
            )
        if self.provided_argument_count is not None:
            object.__setattr__(
                self,
                "provided_argument_count",
                _nonneg_int(self.provided_argument_count, "provided_argument_count"),
            )
        else:
            # Count non-defaulted explicit arguments; splats count as one slot
            # but mark the observation as incomplete for arity comparison.
            explicit = [
                item
                for item in self.actual_arguments
                if item.form is not ArgumentForm.DEFAULTED
            ]
            object.__setattr__(self, "provided_argument_count", len(explicit))
        object.__setattr__(
            self,
            "supplies_parameter_names",
            _string_tuple(
                self.supplies_parameter_names, "supplies_parameter_names"
            ),
        )
        if self.node is not None:
            object.__setattr__(self, "node", _node_ref(self.node))
        object.__setattr__(
            self, "span_ref", _text(self.span_ref, "span_ref", required=False)
        )
        object.__setattr__(
            self,
            "call_requirement_ref",
            _text(self.call_requirement_ref, "call_requirement_ref", required=False),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, "evidence_refs"),
        )
        attrs = self.attributes or {}
        if not isinstance(attrs, Mapping):
            raise ChangeConsumerInventoryError("attributes must be a mapping")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType({str(key): _plain(value) for key, value in attrs.items()}),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or CALL_SITE_OBSERVATION_SCHEMA, "schema"),
        )
        if self.schema != CALL_SITE_OBSERVATION_SCHEMA:
            raise ChangeConsumerInventoryError(
                f"unsupported call site observation schema: {self.schema}"
            )
        claimed = str(self.observation_id or "").strip()
        object.__setattr__(self, "observation_id", "")
        actual = _identity("call-site-observation", self._identity_payload())
        if claimed and claimed != actual:
            raise ChangeConsumerInventoryError(
                "call site observation identity does not match payload"
            )
        object.__setattr__(self, "observation_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "consumer_id": self.consumer_id,
            "caller_kind": self.caller_kind.value,
            "path": self.path,
            "symbol_id": self.symbol_id,
            "callee_symbol_id": self.callee_symbol_id,
            "actual_arguments": [item.to_dict() for item in self.actual_arguments],
            "defaults_applied": list(self.defaults_applied),
            "receiver_state_refs": list(self.receiver_state_refs),
            "path_condition_ref": self.path_condition_ref,
            "awaited": self.awaited,
            "result_uses": list(self.result_uses),
            "handled_error_refs": list(self.handled_error_refs),
            "effect_refs": list(self.effect_refs),
            "capability_refs": list(self.capability_refs),
            "route_hops": list(self.route_hops),
            "route_status": self.route_status.value,
            "callee_default_refs": list(self.callee_default_refs),
            "required_argument_count": self.required_argument_count,
            "provided_argument_count": self.provided_argument_count,
            "supplies_parameter_names": list(self.supplies_parameter_names),
            "node": self.node.to_dict() if self.node is not None else None,
            "span_ref": self.span_ref,
            "call_requirement_ref": self.call_requirement_ref,
            "evidence_refs": list(self.evidence_refs),
            "attributes": dict(self.attributes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "observation_id": self.observation_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallSiteObservation":
        node_payload = payload.get("node")
        return cls(
            consumer_id=str(payload.get("consumer_id") or ""),
            caller_kind=payload.get("caller_kind") or CallerKind.DIRECT,
            path=str(payload.get("path") or ""),
            symbol_id=str(payload.get("symbol_id") or ""),
            callee_symbol_id=str(payload.get("callee_symbol_id") or ""),
            actual_arguments=tuple(
                ActualArgument.from_dict(item)
                for item in (payload.get("actual_arguments") or ())
            ),
            defaults_applied=tuple(payload.get("defaults_applied") or ()),
            receiver_state_refs=tuple(payload.get("receiver_state_refs") or ()),
            path_condition_ref=str(payload.get("path_condition_ref") or ""),
            awaited=bool(payload.get("awaited", False)),
            result_uses=tuple(payload.get("result_uses") or ()),
            handled_error_refs=tuple(payload.get("handled_error_refs") or ()),
            effect_refs=tuple(payload.get("effect_refs") or ()),
            capability_refs=tuple(payload.get("capability_refs") or ()),
            route_hops=tuple(payload.get("route_hops") or ()),
            route_status=payload.get("route_status") or RouteStatus.RESOLVED,
            callee_default_refs=tuple(payload.get("callee_default_refs") or ()),
            required_argument_count=payload.get("required_argument_count"),
            provided_argument_count=payload.get("provided_argument_count"),
            supplies_parameter_names=tuple(
                payload.get("supplies_parameter_names") or ()
            ),
            node=GraphNodeRef.from_dict(node_payload)
            if isinstance(node_payload, Mapping)
            else None,
            span_ref=str(payload.get("span_ref") or ""),
            call_requirement_ref=str(payload.get("call_requirement_ref") or ""),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            attributes=payload.get("attributes") or {},
            schema=str(payload.get("schema") or CALL_SITE_OBSERVATION_SCHEMA),
            observation_id=str(payload.get("observation_id") or ""),
        )

    @property
    def exact_route_key(self) -> str:
        """Stable key for de-duplicating identical paths/routes."""
        return _identity(
            "exact-route",
            {
                "consumer_id": self.consumer_id,
                "path": self.path,
                "symbol_id": self.symbol_id,
                "callee_symbol_id": self.callee_symbol_id,
                "route_hops": list(self.route_hops),
                "path_condition_ref": self.path_condition_ref,
                "caller_kind": self.caller_kind.value,
            },
        )

    @property
    def has_splat(self) -> bool:
        return any(item.is_splat for item in self.actual_arguments)

    def supplies_parameter(self, name: str) -> bool:
        target = str(name or "").strip()
        if not target:
            return False
        if target in self.supplies_parameter_names:
            return True
        if target in self.defaults_applied:
            return True
        for argument in self.actual_arguments:
            if argument.name == target and argument.form is not ArgumentForm.DEFAULTED:
                return True
        return False


# ---------------------------------------------------------------------------
# Compatibility entry + ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerCompatibilityEntry:
    """One consumer's disposition together with the observation that justified it.

    Compatible dispositions never carry missing-input obligations. Frontier
    dispositions never claim proof authority.
    """

    observation: CallSiteObservation
    disposition: ConsumerDisposition
    clause_ids: tuple[str, ...]
    missing_parameter_names: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    obligation: ConsumerMigrationObligation | None = None
    schema: str = CONSUMER_COMPATIBILITY_ENTRY_SCHEMA
    entry_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.observation, CallSiteObservation):
            if isinstance(self.observation, Mapping):
                object.__setattr__(
                    self,
                    "observation",
                    CallSiteObservation.from_dict(self.observation),
                )
            else:
                raise ChangeConsumerInventoryError(
                    "observation must be CallSiteObservation"
                )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ConsumerDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "clause_ids",
            _string_tuple(self.clause_ids, "clause_ids", limit=MAX_CLAUSE_IDS, required=True),
        )
        object.__setattr__(
            self,
            "missing_parameter_names",
            _string_tuple(self.missing_parameter_names, "missing_parameter_names"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(self.reason_codes, "reason_codes"),
        )
        if self.obligation is not None and not isinstance(
            self.obligation, ConsumerMigrationObligation
        ):
            if isinstance(self.obligation, Mapping):
                object.__setattr__(
                    self,
                    "obligation",
                    ConsumerMigrationObligation.from_dict(self.obligation),
                )
            else:
                raise ChangeConsumerInventoryError(
                    "obligation must be ConsumerMigrationObligation"
                )
        if self.disposition in {
            ConsumerDisposition.COMPATIBLE,
            ConsumerDisposition.EXCLUDED,
        }:
            if self.missing_parameter_names:
                raise ChangeConsumerInventoryError(
                    "compatible/excluded entries cannot require missing parameters"
                )
            if self.obligation is not None and (
                self.obligation.missing_input_ids or self.obligation.behavior_contract_ids
            ):
                raise ChangeConsumerInventoryError(
                    "compatible/excluded obligations cannot require missing inputs"
                )
        if self.disposition is ConsumerDisposition.FRONTIER:
            if self.obligation is not None and self.obligation.proof_refs:
                raise ChangeConsumerInventoryError(
                    "frontier obligations cannot carry proof authority"
                )
        if (
            self.obligation is not None
            and self.obligation.disposition is not self.disposition
        ):
            raise ChangeConsumerInventoryError(
                "entry disposition must match obligation disposition"
            )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or CONSUMER_COMPATIBILITY_ENTRY_SCHEMA, "schema"),
        )
        if self.schema != CONSUMER_COMPATIBILITY_ENTRY_SCHEMA:
            raise ChangeConsumerInventoryError(
                f"unsupported consumer compatibility entry schema: {self.schema}"
            )
        claimed = str(self.entry_id or "").strip()
        object.__setattr__(self, "entry_id", "")
        actual = _identity("consumer-compatibility-entry", self._identity_payload())
        if claimed and claimed != actual:
            raise ChangeConsumerInventoryError(
                "consumer compatibility entry identity does not match payload"
            )
        object.__setattr__(self, "entry_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "observation": self.observation.to_dict(),
            "disposition": self.disposition.value,
            "clause_ids": list(self.clause_ids),
            "missing_parameter_names": list(self.missing_parameter_names),
            "reason_codes": list(self.reason_codes),
            "obligation": (
                self.obligation.to_record() if self.obligation is not None else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "entry_id": self.entry_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsumerCompatibilityEntry":
        obligation_payload = payload.get("obligation")
        return cls(
            observation=CallSiteObservation.from_dict(payload.get("observation") or {}),
            disposition=payload.get("disposition") or ConsumerDisposition.ABSTAIN,
            clause_ids=tuple(payload.get("clause_ids") or ()),
            missing_parameter_names=tuple(
                payload.get("missing_parameter_names") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            obligation=(
                ConsumerMigrationObligation.from_dict(obligation_payload)
                if isinstance(obligation_payload, Mapping)
                else None
            ),
            schema=str(payload.get("schema") or CONSUMER_COMPATIBILITY_ENTRY_SCHEMA),
            entry_id=str(payload.get("entry_id") or ""),
        )


@dataclass(frozen=True)
class ConsumerCompatibilityLedger:
    """Deterministic ledger of one disposition per exact consumer route.

    Duplicate exact paths collapse to a single entry/obligation. Compatible
    entries never discharge other consumers' migration work.
    """

    roots: PropagationAuthorityRoots
    delta_id: str
    subject_symbol_id: str
    entries: tuple[ConsumerCompatibilityEntry, ...]
    frontier_consumer_ids: tuple[str, ...] = ()
    excluded_consumer_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    schema: str = CONSUMER_COMPATIBILITY_LEDGER_SCHEMA
    ledger_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        if len(self.entries) > MAX_ENTRIES:
            raise ChangeConsumerInventoryBoundsError(
                "ledger entries exceed the consumer bound"
            )
        entries = tuple(
            item
            if isinstance(item, ConsumerCompatibilityEntry)
            else ConsumerCompatibilityEntry.from_dict(item)  # type: ignore[arg-type]
            for item in self.entries
        )
        # Deterministic order; exact-route uniqueness is mandatory.
        by_route: dict[str, ConsumerCompatibilityEntry] = {}
        for entry in entries:
            key = entry.observation.exact_route_key
            existing = by_route.get(key)
            if existing is not None:
                if existing.entry_id != entry.entry_id:
                    raise ChangeConsumerInventoryError(
                        "duplicate exact routes must not produce distinct obligations"
                    )
                continue
            by_route[key] = entry
        ordered = tuple(
            sorted(
                by_route.values(),
                key=lambda item: (
                    item.observation.consumer_id,
                    item.observation.path,
                    item.observation.symbol_id,
                    item.entry_id,
                ),
            )
        )
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(
            self,
            "frontier_consumer_ids",
            _string_tuple(
                self.frontier_consumer_ids,
                "frontier_consumer_ids",
                limit=MAX_ENTRIES,
            ),
        )
        object.__setattr__(
            self,
            "excluded_consumer_ids",
            _string_tuple(
                self.excluded_consumer_ids,
                "excluded_consumer_ids",
                limit=MAX_ENTRIES,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, "evidence_refs"),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or CONSUMER_COMPATIBILITY_LEDGER_SCHEMA, "schema"),
        )
        if self.schema != CONSUMER_COMPATIBILITY_LEDGER_SCHEMA:
            raise ChangeConsumerInventoryError(
                f"unsupported consumer compatibility ledger schema: {self.schema}"
            )
        claimed = str(self.ledger_id or "").strip()
        object.__setattr__(self, "ledger_id", "")
        actual = _identity("consumer-compatibility-ledger", self._identity_payload())
        if claimed and claimed != actual:
            raise ChangeConsumerInventoryError(
                "consumer compatibility ledger identity does not match payload"
            )
        object.__setattr__(self, "ledger_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "roots": self.roots.to_dict(),
            "delta_id": self.delta_id,
            "subject_symbol_id": self.subject_symbol_id,
            "entries": [item.to_dict() for item in self.entries],
            "frontier_consumer_ids": list(self.frontier_consumer_ids),
            "excluded_consumer_ids": list(self.excluded_consumer_ids),
            "evidence_refs": list(self.evidence_refs),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "ledger_id": self.ledger_id}

    def to_record(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsumerCompatibilityLedger":
        return cls(
            roots=_roots(payload.get("roots") or {}),
            delta_id=str(payload.get("delta_id") or ""),
            subject_symbol_id=str(payload.get("subject_symbol_id") or ""),
            entries=tuple(
                ConsumerCompatibilityEntry.from_dict(item)
                for item in (payload.get("entries") or ())
            ),
            frontier_consumer_ids=tuple(payload.get("frontier_consumer_ids") or ()),
            excluded_consumer_ids=tuple(payload.get("excluded_consumer_ids") or ()),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            schema=str(payload.get("schema") or CONSUMER_COMPATIBILITY_LEDGER_SCHEMA),
            ledger_id=str(payload.get("ledger_id") or ""),
        )

    @property
    def obligations(self) -> tuple[ConsumerMigrationObligation, ...]:
        """Canonical ConsumerMigrationObligation@1 records, one per entry that owns one."""
        return tuple(
            entry.obligation for entry in self.entries if entry.obligation is not None
        )

    @property
    def migrate_entries(self) -> tuple[ConsumerCompatibilityEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.disposition is ConsumerDisposition.MIGRATE
        )

    @property
    def compatible_entries(self) -> tuple[ConsumerCompatibilityEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.disposition is ConsumerDisposition.COMPATIBLE
        )

    @property
    def frontier_entries(self) -> tuple[ConsumerCompatibilityEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.disposition is ConsumerDisposition.FRONTIER
        )

    def entries_for_kind(self, kind: CallerKind | str) -> tuple[ConsumerCompatibilityEntry, ...]:
        target = _enum(kind, CallerKind, "caller_kind")
        return tuple(
            entry for entry in self.entries if entry.observation.caller_kind is target
        )

    def obligation_set_id(self) -> str:
        obligations = self.obligations
        if not obligations:
            raise ChangeConsumerInventoryError(
                "obligation set identity requires at least one obligation"
            )
        return obligation_set_identity(obligations)

    def one_compatible_cannot_discharge_others(self) -> bool:
        """Structural invariant: compatible rows never clear migrate rows."""
        if not self.compatible_entries:
            return True
        return bool(self.migrate_entries) or bool(self.frontier_entries) or (
            len(self.entries) == len(self.compatible_entries)
        )


# ---------------------------------------------------------------------------
# Parameter-add helpers
# ---------------------------------------------------------------------------


_PARAM_NAME_RE = re.compile(
    r"(?:parameter|param|arg(?:ument)?)\s*[:=]\s*['\"]?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
_SIG_PARAM_RE = re.compile(
    r"(?:\*{0,2})([A-Za-z_][\w]*)\s*(?::|=|,|$)",
)


def _signature_parameter_names(text: str) -> tuple[str, ...]:
    """Extract ordered parameter names from a signature fragment ``f(a, b, c)``."""
    blob = str(text or "")
    if "(" not in blob or ")" not in blob:
        return ()
    inner = blob[blob.find("(") + 1 : blob.rfind(")")]
    if not inner.strip():
        return ()
    names: list[str] = []
    for part in inner.split(","):
        part = part.strip()
        if not part or part in {"*", "/"}:
            continue
        match = _SIG_PARAM_RE.match(part)
        if match:
            names.append(match.group(1))
    return tuple(names)


def _parameter_names_from_clause(clause: ContractClauseDelta) -> tuple[str, ...]:
    """Extract *added* parameter names for PARAMETER_ADD clauses.

    Prefer explicit ``parameter=name`` annotations and before/after signature
    diffs. Never treat pre-existing parameters (left/right) as newly required
    solely because they appear in the after signature.
    """
    names: list[str] = []
    # 1) Explicit "parameter=context" style annotations in the reason/refs.
    for blob in (clause.reason, clause.after_contract_ref, clause.before_contract_ref):
        for match in _PARAM_NAME_RE.finditer(str(blob or "")):
            names.append(match.group(1))

    # 2) Diff after vs before signatures when both are present.
    after_names = _signature_parameter_names(
        str(clause.after_contract_ref or clause.reason or "")
    )
    before_names = _signature_parameter_names(str(clause.before_contract_ref or ""))
    if after_names and before_names:
        before_set = set(before_names)
        for name in after_names:
            if name not in before_set:
                names.append(name)
    elif after_names and not before_names and not names:
        # Only the last parameter is treated as added when we lack a before
        # signature and no explicit annotation was found.
        names.append(after_names[-1])

    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _required_arity_after(clause: ContractClauseDelta, observation: CallSiteObservation) -> int | None:
    """Infer the post-change required positional arity for PARAMETER_ADD."""
    if observation.required_argument_count is not None:
        return observation.required_argument_count
    if clause.kind is not DeltaKind.PARAMETER_ADD:
        return None
    after = str(clause.after_contract_ref or clause.reason or "")
    after_names = _signature_parameter_names(after)
    if after_names:
        # Count required (non-defaulted) parameters from the after signature.
        inner = after[after.find("(") + 1 : after.rfind(")")]
        parts = [part.strip() for part in inner.split(",") if part.strip()]
        required = [
            part
            for part in parts
            if not part.startswith("*") and "=" not in part
        ]
        if required:
            return len(required)
        return len(after_names)
    if observation.provided_argument_count is not None:
        return observation.provided_argument_count + 1
    return None


def _added_parameter_names(
    clause: ContractClauseDelta, observation: CallSiteObservation
) -> tuple[str, ...]:
    # Observation attributes are the most specific signal for fixture/tests.
    names: list[str] = []
    attrs = observation.attributes or {}
    for key in ("added_parameters", "missing_parameters", "new_parameters"):
        raw = attrs.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            for item in raw:
                text = str(item).strip()
                if text and text not in names:
                    names.append(text)
    single = attrs.get("added_parameter") or attrs.get("parameter_name")
    if single:
        text = str(single).strip()
        if text and text not in names:
            names.append(text)
    if names:
        return tuple(names)

    names = list(_parameter_names_from_clause(clause))
    if not names and clause.kind is DeltaKind.PARAMETER_ADD:
        names.append("context")
    return tuple(names)


# ---------------------------------------------------------------------------
# Graph discovery helpers
# ---------------------------------------------------------------------------


def _caller_kind_from_edge_and_node(
    edge_kind: ProgramEdgeKind | None, node: ProgramNode | None
) -> CallerKind:
    if edge_kind is ProgramEdgeKind.ALIASES:
        return CallerKind.ALIASED
    if edge_kind is ProgramEdgeKind.RE_EXPORTS:
        return CallerKind.RE_EXPORTED
    if edge_kind is ProgramEdgeKind.DECORATES:
        return CallerKind.DECORATED
    if edge_kind is ProgramEdgeKind.CALLBACK_TO:
        return CallerKind.CALLBACK
    if edge_kind is ProgramEdgeKind.OVERLOADS:
        return CallerKind.OVERLOAD
    if edge_kind in {ProgramEdgeKind.OVERRIDES, ProgramEdgeKind.IMPLEMENTS}:
        return CallerKind.METHOD_OVERRIDE
    if edge_kind in {
        ProgramEdgeKind.FACTORY_CREATES,
        ProgramEdgeKind.CONSTRUCTS,
        ProgramEdgeKind.BUILDER_BUILDS,
        ProgramEdgeKind.INJECTS,
    }:
        return CallerKind.FACTORY
    if edge_kind in {
        ProgramEdgeKind.TESTS,
        ProgramEdgeKind.MOCKS,
        ProgramEdgeKind.FIXTURES,
    }:
        return CallerKind.TEST_MOCK
    if edge_kind is ProgramEdgeKind.GENERATED_FROM:
        return CallerKind.GENERATED_CLIENT
    if node is not None:
        if node.kind is ProgramNodeKind.ALIAS:
            return CallerKind.ALIASED
        if node.kind is ProgramNodeKind.RE_EXPORT:
            return CallerKind.RE_EXPORTED
        if node.kind is ProgramNodeKind.DECORATOR:
            return CallerKind.DECORATED
        if node.kind is ProgramNodeKind.CALLBACK:
            return CallerKind.CALLBACK
        if node.kind is ProgramNodeKind.OVERLOAD:
            return CallerKind.OVERLOAD
        if node.kind in {ProgramNodeKind.METHOD, ProgramNodeKind.INTERFACE, ProgramNodeKind.PROTOCOL}:
            return CallerKind.METHOD_OVERRIDE
        if node.kind in {
            ProgramNodeKind.FACTORY,
            ProgramNodeKind.CONSTRUCTOR,
            ProgramNodeKind.BUILDER,
            ProgramNodeKind.DI_BINDING,
        }:
            return CallerKind.FACTORY
        if node.kind in {
            ProgramNodeKind.TEST,
            ProgramNodeKind.MOCK,
            ProgramNodeKind.FIXTURE,
        }:
            return CallerKind.TEST_MOCK
        if node.kind is ProgramNodeKind.GENERATED or "/generated/" in (
            node.path or ""
        ):
            return CallerKind.GENERATED_CLIENT
        # Wrapper heuristic: name contains wrap/proxy/adapter.
        leaf = (node.name or node.qualified_name or "").lower()
        if any(token in leaf for token in ("wrap", "proxy", "adapter", "delegate")):
            return CallerKind.WRAPPED
    return CallerKind.DIRECT


def _node_to_graph_ref(node: ProgramNode) -> GraphNodeRef:
    provenance = (
        GraphProvenance.TRUSTED
        if node.trust.accepted and node.authority.authority_bearing
        else GraphProvenance.NOMINATED
    )
    extractor = node.extractor_id or (
        "extractor:program-graph" if provenance is GraphProvenance.TRUSTED else ""
    )
    # Trusted GraphNodeRef requires extractor_id.
    if provenance is GraphProvenance.TRUSTED and not extractor:
        extractor = "extractor:program-graph"
    kind = node.kind.value if isinstance(node.kind, Enum) else str(node.kind)
    return GraphNodeRef(
        node_id=node.node_id,
        kind=kind,
        path=node.path or "unknown.py",
        symbol_id=node.qualified_name or node.name or node.node_id,
        artifact_id=node.artifact_id or f"blob:{node.node_id}",
        provenance=provenance,
        extractor_id=extractor,
    )


def _default_node_ref(
    *,
    consumer_id: str,
    path: str,
    symbol_id: str,
    kind: str = "function",
) -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{consumer_id}",
        kind=kind,
        path=path,
        symbol_id=symbol_id,
        artifact_id=f"blob:{consumer_id}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:change-consumer-inventory",
    )


# ---------------------------------------------------------------------------
# Inventory builder
# ---------------------------------------------------------------------------


@dataclass
class ChangeConsumerInventory:
    """Build a :class:`ConsumerCompatibilityLedger` for one contract delta.

    Accepts pre-resolved call-site observations (fixture/test path) and/or
    discovers reverse callers from a bound :class:`ProgramGraph`, optionally
    refining resolution status with a :class:`ProgramCallResolver`.
    """

    roots: PropagationAuthorityRoots | None = None
    graph: ProgramGraph | ProgramGraphSnapshot | None = None
    resolver: ProgramCallResolver | None = None

    def __post_init__(self) -> None:
        if self.roots is not None:
            self.roots = _roots(self.roots)
        if isinstance(self.graph, ProgramGraphSnapshot):
            self.graph = ProgramGraph(self.graph)
        if self.graph is not None and not isinstance(self.graph, ProgramGraph):
            raise ChangeConsumerInventoryError(
                "graph must be a ProgramGraph or ProgramGraphSnapshot"
            )
        if self.resolver is not None and not isinstance(
            self.resolver, ProgramCallResolver
        ):
            raise ChangeConsumerInventoryError(
                "resolver must be a ProgramCallResolver"
            )

    def bind(
        self,
        *,
        roots: PropagationAuthorityRoots | None = None,
        graph: ProgramGraph | ProgramGraphSnapshot | None = None,
        resolver: ProgramCallResolver | None = None,
    ) -> "ChangeConsumerInventory":
        if roots is not None:
            self.roots = _roots(roots)
        if graph is not None:
            self.graph = (
                ProgramGraph(graph)
                if isinstance(graph, ProgramGraphSnapshot)
                else graph
            )
        if resolver is not None:
            self.resolver = resolver
            if self.graph is not None and self.resolver.graph is None:
                self.resolver.bind(self.graph)
        return self

    # -- public API ---------------------------------------------------------

    def inventory(
        self,
        delta: ProgramContractDelta,
        call_sites: Sequence[CallSiteObservation | Mapping[str, Any]] = (),
        *,
        discover_from_graph: bool = False,
        excluded_consumer_ids: Sequence[str] = (),
        evidence_refs: Sequence[str] = (),
    ) -> ConsumerCompatibilityLedger:
        """Produce one disposition per exact route for the given delta."""
        if not isinstance(delta, ProgramContractDelta):
            raise ChangeConsumerInventoryError(
                "inventory requires a ProgramContractDelta"
            )
        roots = self.roots or delta.roots
        roots = _roots(roots)
        if roots.content_id != delta.roots.content_id:
            # Fail closed on root mismatch rather than mixing authorities.
            raise ChangeConsumerInventoryError(
                "inventory roots must match the program contract delta roots"
            )

        observations = self._normalize_observations(
            delta, call_sites, discover_from_graph=discover_from_graph
        )
        # De-duplicate exact routes before classifying (duplicate paths do not
        # duplicate obligations).
        by_route: dict[str, CallSiteObservation] = {}
        for observation in observations:
            key = observation.exact_route_key
            if key not in by_route:
                by_route[key] = observation

        clauses = tuple(delta.clauses)
        if not clauses:
            raise ChangeConsumerInventoryError(
                "delta must contain at least one contract clause"
            )
        clause_ids = tuple(item.clause_id for item in clauses)
        excluded = set(
            _string_tuple(excluded_consumer_ids, "excluded_consumer_ids", limit=MAX_ENTRIES)
        )

        entries: list[ConsumerCompatibilityEntry] = []
        frontier_ids: list[str] = []
        for observation in sorted(
            by_route.values(),
            key=lambda item: (item.consumer_id, item.path, item.symbol_id),
        ):
            if observation.consumer_id in excluded:
                entry = self._build_entry(
                    roots=roots,
                    delta=delta,
                    observation=observation,
                    disposition=ConsumerDisposition.EXCLUDED,
                    clause_ids=clause_ids,
                    missing_parameter_names=(),
                    reason_codes=("excluded_consumer",),
                )
                entries.append(entry)
                continue

            disposition, missing, reasons = self._classify(delta, observation)
            if disposition is ConsumerDisposition.FRONTIER:
                frontier_ids.append(observation.consumer_id)
            entry = self._build_entry(
                roots=roots,
                delta=delta,
                observation=observation,
                disposition=disposition,
                clause_ids=clause_ids,
                missing_parameter_names=missing,
                reason_codes=reasons,
            )
            entries.append(entry)

        ledger = ConsumerCompatibilityLedger(
            roots=roots,
            delta_id=self._delta_id(delta),
            subject_symbol_id=delta.subject_symbol_id,
            entries=tuple(entries),
            frontier_consumer_ids=tuple(sorted(set(frontier_ids))),
            excluded_consumer_ids=tuple(sorted(excluded)),
            evidence_refs=_string_tuple(evidence_refs, "evidence_refs"),
        )
        # Structural invariant: one compatible row cannot clear others.
        migrate = ledger.migrate_entries
        compatible = ledger.compatible_entries
        if compatible and migrate:
            # Explicitly retained as independent obligations.
            assert len(migrate) == len(
                {entry.observation.exact_route_key for entry in migrate}
            )
        return ledger

    def inventory_parameter_add(
        self,
        delta: ProgramContractDelta,
        call_sites: Sequence[CallSiteObservation | Mapping[str, Any]],
        *,
        new_parameter: str = "context",
        required_arity: int | None = None,
    ) -> ConsumerCompatibilityLedger:
        """Convenience path for the representative two→three argument change."""
        normalized: list[CallSiteObservation] = []
        for raw in call_sites:
            observation = (
                raw
                if isinstance(raw, CallSiteObservation)
                else CallSiteObservation.from_dict(raw)
            )
            attrs = dict(observation.attributes)
            attrs.setdefault("added_parameter", new_parameter)
            normalized.append(
                CallSiteObservation(
                    consumer_id=observation.consumer_id,
                    caller_kind=observation.caller_kind,
                    path=observation.path,
                    symbol_id=observation.symbol_id,
                    callee_symbol_id=observation.callee_symbol_id
                    or delta.subject_symbol_id,
                    actual_arguments=observation.actual_arguments,
                    defaults_applied=observation.defaults_applied,
                    receiver_state_refs=observation.receiver_state_refs,
                    path_condition_ref=observation.path_condition_ref,
                    awaited=observation.awaited,
                    result_uses=observation.result_uses,
                    handled_error_refs=observation.handled_error_refs,
                    effect_refs=observation.effect_refs,
                    capability_refs=observation.capability_refs,
                    route_hops=observation.route_hops,
                    route_status=observation.route_status,
                    callee_default_refs=observation.callee_default_refs,
                    required_argument_count=(
                        required_arity
                        if required_arity is not None
                        else observation.required_argument_count
                    ),
                    provided_argument_count=observation.provided_argument_count,
                    supplies_parameter_names=observation.supplies_parameter_names,
                    node=observation.node,
                    span_ref=observation.span_ref,
                    call_requirement_ref=observation.call_requirement_ref,
                    evidence_refs=observation.evidence_refs,
                    attributes=attrs,
                )
            )
        return self.inventory(delta, normalized)

    # -- classification -----------------------------------------------------

    def _classify(
        self,
        delta: ProgramContractDelta,
        observation: CallSiteObservation,
    ) -> tuple[ConsumerDisposition, tuple[str, ...], tuple[str, ...]]:
        if observation.route_status in _FRONTIER_ROUTE_STATUSES:
            return (
                ConsumerDisposition.FRONTIER,
                (),
                (f"route_{observation.route_status.value}", "frontier_call_site"),
            )

        relevant = self._relevant_clauses(delta, observation)
        if not relevant:
            return (
                ConsumerDisposition.ABSTAIN,
                (),
                ("no_relevant_clause",),
            )

        # Any unknown/unsupported clause forces frontier/abstain for that route.
        if any(item.disposition is DeltaDisposition.UNSUPPORTED for item in relevant):
            return (
                ConsumerDisposition.FRONTIER,
                (),
                ("unsupported_clause",),
            )
        if any(item.disposition is DeltaDisposition.UNKNOWN for item in relevant):
            return (
                ConsumerDisposition.FRONTIER,
                (),
                ("unknown_clause",),
            )

        missing: list[str] = []
        reasons: list[str] = []
        needs_migrate = False
        all_compatible = True

        for clause in relevant:
            if clause.disposition is DeltaDisposition.COMPATIBLE:
                reasons.append(f"clause_compatible:{clause.clause_id}")
                continue
            if clause.disposition is DeltaDisposition.BEHAVIORAL:
                all_compatible = False
                needs_migrate = True
                reasons.append(f"behavioral_clause:{clause.clause_id}")
                continue
            if clause.disposition is DeltaDisposition.BREAKING:
                clause_missing, clause_reasons = self._breaking_gaps(clause, observation)
                reasons.extend(clause_reasons)
                if clause_missing:
                    all_compatible = False
                    needs_migrate = True
                    missing.extend(clause_missing)
                else:
                    # Breaking clause but this caller already satisfies it
                    # (supplies the new parameter / applies a call-site default).
                    reasons.append(f"caller_satisfies_breaking:{clause.clause_id}")
                continue

        # De-duplicate missing names while preserving order.
        seen: set[str] = set()
        ordered_missing: list[str] = []
        for name in missing:
            if name not in seen:
                seen.add(name)
                ordered_missing.append(name)

        if needs_migrate:
            # Callee defaults present on the declaration never discharge a
            # caller that does not actually apply them at the call site.
            if observation.callee_default_refs and not observation.defaults_applied:
                reasons.append("callee_default_does_not_discharge_caller")
            return (
                ConsumerDisposition.MIGRATE,
                tuple(ordered_missing),
                tuple(sorted(set(reasons))) or ("breaking_consumer",),
            )
        if all_compatible:
            return (
                ConsumerDisposition.COMPATIBLE,
                (),
                tuple(sorted(set(reasons))) or ("all_clauses_compatible",),
            )
        return (
            ConsumerDisposition.ABSTAIN,
            (),
            tuple(sorted(set(reasons))) or ("undetermined",),
        )

    def _relevant_clauses(
        self,
        delta: ProgramContractDelta,
        observation: CallSiteObservation,
    ) -> tuple[ContractClauseDelta, ...]:
        # Prefer clauses matching the subject; include all when subject aligns.
        return tuple(
            clause
            for clause in delta.clauses
            if clause.subject_symbol_id == delta.subject_symbol_id
        ) or tuple(delta.clauses)

    def _breaking_gaps(
        self,
        clause: ContractClauseDelta,
        observation: CallSiteObservation,
    ) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        reasons: list[str] = []

        if clause.kind is DeltaKind.PARAMETER_ADD:
            added = _added_parameter_names(clause, observation)
            required = _required_arity_after(clause, observation)
            provided = observation.provided_argument_count or 0

            # Splats make arity uncertain → frontier-like migrate with reason.
            if observation.has_splat:
                reasons.append("splat_argument_arity_uncertain")
                for name in added:
                    if not observation.supplies_parameter(name):
                        missing.append(name)
                if missing:
                    reasons.append("each_two_arg_caller_gets_obligation")
                return missing, reasons

            for name in added:
                if observation.supplies_parameter(name):
                    reasons.append(f"supplies_parameter:{name}")
                    continue
                if name in observation.defaults_applied:
                    reasons.append(f"default_applied_at_callsite:{name}")
                    continue
                missing.append(name)

            if missing:
                reasons.append("missing_required_parameter")
                reasons.append("each_two_arg_caller_gets_obligation")
                if required is not None and provided < required:
                    reasons.append("arity_shortfall")
            elif not added and required is not None and provided < required:
                # No named added parameters but arity still short → obligate.
                reasons.append("arity_shortfall")
                reasons.append("each_two_arg_caller_gets_obligation")
                missing.append("context")
            # When every added parameter is already supplied (including via
            # call-site defaults_applied), a lower explicit positional count
            # is not an independent gap.

            # Explicit acceptance: callee defaults do not auto-satisfy callers
            # that did not apply them at the call site.
            if missing and observation.callee_default_refs:
                reasons.append("compatible_default_does_not_discharge_others")

            return missing, reasons

        if clause.kind in {
            DeltaKind.PARAMETER_REMOVE,
            DeltaKind.PARAMETER_RENAME,
            DeltaKind.PARAMETER_REORDER,
            DeltaKind.PARAMETER_KEYWORD,
            DeltaKind.PARAMETER_VARIANCE,
            DeltaKind.PARAMETER_DEFAULT,
        }:
            reasons.append(f"parameter_shape_change:{clause.kind.value}")
            return missing, reasons

        if clause.kind in {
            DeltaKind.SYNC_ASYNC_CHANGE,
            DeltaKind.ERROR_CHANGE,
            DeltaKind.EFFECT_CHANGE,
            DeltaKind.CAPABILITY_CHANGE,
            DeltaKind.AUTHORIZATION_CHANGE,
        }:
            reasons.append(f"effect_or_async_change:{clause.kind.value}")
            return missing, reasons

        reasons.append(f"breaking_clause:{clause.kind.value}")
        return missing, reasons

    def _build_entry(
        self,
        *,
        roots: PropagationAuthorityRoots,
        delta: ProgramContractDelta,
        observation: CallSiteObservation,
        disposition: ConsumerDisposition,
        clause_ids: Sequence[str],
        missing_parameter_names: Sequence[str],
        reason_codes: Sequence[str],
    ) -> ConsumerCompatibilityEntry:
        node = observation.node or _default_node_ref(
            consumer_id=observation.consumer_id,
            path=observation.path,
            symbol_id=observation.symbol_id,
            kind=(
                "method"
                if observation.caller_kind is CallerKind.METHOD_OVERRIDE
                else "function"
            ),
        )
        delta_id = self._delta_id(delta)
        missing_ids: tuple[str, ...] = ()
        if disposition is ConsumerDisposition.MIGRATE and missing_parameter_names:
            missing_ids = tuple(
                f"missing:{observation.consumer_id}:{name}"
                for name in missing_parameter_names
            )
        proof_refs: tuple[str, ...] = ()
        # Migration obligations may be empty of proofs until later stages;
        # frontier obligations must not carry proof_refs (enforced by contract).
        if disposition is ConsumerDisposition.FRONTIER:
            proof_refs = ()
        obligation = ConsumerMigrationObligation(
            roots=roots,
            obligation_id=f"obligation:{observation.consumer_id}:{delta_id}",
            consumer_id=observation.consumer_id,
            delta_id=delta_id,
            disposition=disposition,
            clause_ids=tuple(clause_ids),
            node=node,
            proof_refs=proof_refs,
            missing_input_ids=missing_ids
            if disposition
            not in {ConsumerDisposition.COMPATIBLE, ConsumerDisposition.EXCLUDED}
            else (),
            behavior_contract_ids=(),
            invalidation_refs=(roots.candidate_tree_id,),
        )
        return ConsumerCompatibilityEntry(
            observation=observation,
            disposition=disposition,
            clause_ids=tuple(clause_ids),
            missing_parameter_names=tuple(missing_parameter_names),
            reason_codes=tuple(reason_codes),
            obligation=obligation,
        )

    def _delta_id(self, delta: ProgramContractDelta) -> str:
        # ProgramContractDelta content identity is the stable delta key.
        return f"delta:{delta.content_id}"

    # -- observation normalization / discovery ------------------------------

    def _normalize_observations(
        self,
        delta: ProgramContractDelta,
        call_sites: Sequence[CallSiteObservation | Mapping[str, Any]],
        *,
        discover_from_graph: bool,
    ) -> list[CallSiteObservation]:
        observations: list[CallSiteObservation] = []
        for raw in call_sites:
            observation = (
                raw
                if isinstance(raw, CallSiteObservation)
                else CallSiteObservation.from_dict(raw)
            )
            if not observation.callee_symbol_id:
                observation = CallSiteObservation(
                    consumer_id=observation.consumer_id,
                    caller_kind=observation.caller_kind,
                    path=observation.path,
                    symbol_id=observation.symbol_id,
                    callee_symbol_id=delta.subject_symbol_id,
                    actual_arguments=observation.actual_arguments,
                    defaults_applied=observation.defaults_applied,
                    receiver_state_refs=observation.receiver_state_refs,
                    path_condition_ref=observation.path_condition_ref,
                    awaited=observation.awaited,
                    result_uses=observation.result_uses,
                    handled_error_refs=observation.handled_error_refs,
                    effect_refs=observation.effect_refs,
                    capability_refs=observation.capability_refs,
                    route_hops=observation.route_hops,
                    route_status=observation.route_status,
                    callee_default_refs=observation.callee_default_refs,
                    required_argument_count=observation.required_argument_count,
                    provided_argument_count=observation.provided_argument_count,
                    supplies_parameter_names=observation.supplies_parameter_names,
                    node=observation.node,
                    span_ref=observation.span_ref,
                    call_requirement_ref=observation.call_requirement_ref,
                    evidence_refs=observation.evidence_refs,
                    attributes=dict(observation.attributes),
                )
            if self.resolver is not None:
                observation = self._refine_with_resolver(observation)
            observations.append(observation)

        if discover_from_graph:
            if self.graph is None:
                raise ChangeConsumerInventoryError(
                    "discover_from_graph requires a bound ProgramGraph"
                )
            observations.extend(
                self._discover_from_graph(delta.subject_symbol_id)
            )
        if not observations:
            raise ChangeConsumerInventoryError(
                "inventory requires at least one call site observation"
            )
        return observations

    def _refine_with_resolver(
        self, observation: CallSiteObservation
    ) -> CallSiteObservation:
        assert self.resolver is not None
        site = CallSite(
            caller_id=observation.symbol_id,
            callee_reference=observation.callee_symbol_id,
            path=observation.path,
            language="python",
            call_form=observation.caller_kind.value,
            awaited=observation.awaited,
            attributes={
                "consumer_id": observation.consumer_id,
                "route_hops": list(observation.route_hops),
            },
        )
        try:
            resolution: CallResolution = self.resolver.resolve(site)
        except Exception:
            return observation
        status = _RESOLVER_STATUS_MAP.get(
            resolution.status, RouteStatus.UNSUPPORTED
        )
        if status is observation.route_status:
            return observation
        return CallSiteObservation(
            consumer_id=observation.consumer_id,
            caller_kind=observation.caller_kind,
            path=observation.path,
            symbol_id=observation.symbol_id,
            callee_symbol_id=observation.callee_symbol_id,
            actual_arguments=observation.actual_arguments,
            defaults_applied=observation.defaults_applied,
            receiver_state_refs=observation.receiver_state_refs,
            path_condition_ref=observation.path_condition_ref,
            awaited=observation.awaited,
            result_uses=observation.result_uses,
            handled_error_refs=observation.handled_error_refs,
            effect_refs=observation.effect_refs,
            capability_refs=observation.capability_refs,
            route_hops=observation.route_hops
            or tuple(resolution.target_ids or resolution.candidate_ids),
            route_status=status,
            callee_default_refs=observation.callee_default_refs,
            required_argument_count=observation.required_argument_count,
            provided_argument_count=observation.provided_argument_count,
            supplies_parameter_names=observation.supplies_parameter_names,
            node=observation.node,
            span_ref=observation.span_ref,
            call_requirement_ref=observation.call_requirement_ref,
            evidence_refs=observation.evidence_refs
            + tuple(resolution.evidence_ids),
            attributes={
                **dict(observation.attributes),
                "resolver_status": resolution.status.value,
                "resolver_reason_codes": list(resolution.reason_codes),
            },
        )

    def _discover_from_graph(self, subject_symbol_id: str) -> list[CallSiteObservation]:
        assert self.graph is not None
        targets = self.graph.find_by_qualified_name(subject_symbol_id)
        if not targets:
            # Also match by trailing name.
            leaf = subject_symbol_id.rsplit(".", 1)[-1]
            targets = self.graph.find_by_qualified_name(leaf)
        if not targets:
            return []

        observations: list[CallSiteObservation] = []
        seen: set[str] = set()
        for target in targets:
            for edge in self.graph.edges_to(target.node_id):
                if edge.kind not in _CALL_EDGE_KINDS:
                    continue
                # Nominated-only edges stay frontier.
                source = self.graph.node(edge.source)
                if source is None:
                    continue
                kind = _caller_kind_from_edge_and_node(edge.kind, source)
                consumer_id = f"consumer:{source.node_id}"
                if consumer_id in seen:
                    continue
                seen.add(consumer_id)
                route_status = (
                    RouteStatus.RESOLVED
                    if edge.authoritative
                    else RouteStatus.UNSUPPORTED
                )
                if source.kind is ProgramNodeKind.FRONTIER or not edge.authoritative:
                    route_status = RouteStatus.DYNAMIC if (
                        "dynamic" in (source.name or "").lower()
                        or "getattr" in (source.qualified_name or "").lower()
                    ) else RouteStatus.UNSUPPORTED
                observations.append(
                    CallSiteObservation(
                        consumer_id=consumer_id,
                        caller_kind=kind,
                        path=source.path or "unknown.py",
                        symbol_id=source.qualified_name or source.name or source.node_id,
                        callee_symbol_id=subject_symbol_id,
                        actual_arguments=(),
                        route_hops=(edge.edge_id,),
                        route_status=route_status,
                        node=_node_to_graph_ref(source),
                        evidence_refs=(edge.edge_id,),
                        attributes={
                            "edge_kind": edge.kind.value,
                            "discovered_from_graph": True,
                        },
                    )
                )
        return observations


def build_change_consumer_inventory(
    delta: ProgramContractDelta,
    call_sites: Sequence[CallSiteObservation | Mapping[str, Any]],
    *,
    roots: PropagationAuthorityRoots | None = None,
    graph: ProgramGraph | ProgramGraphSnapshot | None = None,
    resolver: ProgramCallResolver | None = None,
    discover_from_graph: bool = False,
    excluded_consumer_ids: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
) -> ConsumerCompatibilityLedger:
    """Functional façade over :class:`ChangeConsumerInventory`."""
    inventory = ChangeConsumerInventory(
        roots=roots or delta.roots, graph=graph, resolver=resolver
    )
    return inventory.inventory(
        delta,
        call_sites,
        discover_from_graph=discover_from_graph,
        excluded_consumer_ids=excluded_consumer_ids,
        evidence_refs=evidence_refs,
    )


def required_caller_kinds() -> frozenset[str]:
    """Closed catalogue of caller kinds the inventory must be able to name."""
    return _REQUIRED_CALLER_KINDS


__all__ = [
    "ActualArgument",
    "ArgumentForm",
    "CALL_SITE_OBSERVATION_SCHEMA",
    "CHANGE_CONSUMER_INVENTORY_SCHEMA",
    "CHANGE_CONSUMER_INVENTORY_VERSION",
    "CONSUMER_COMPATIBILITY_ENTRY_SCHEMA",
    "CONSUMER_COMPATIBILITY_LEDGER_SCHEMA",
    "CallSiteObservation",
    "CallerKind",
    "ChangeConsumerInventory",
    "ChangeConsumerInventoryBoundsError",
    "ChangeConsumerInventoryError",
    "ConsumerCompatibilityEntry",
    "ConsumerCompatibilityLedger",
    "ConsumerDisposition",
    "ConsumerMigrationObligation",
    "RouteStatus",
    "build_change_consumer_inventory",
    "required_caller_kinds",
]
