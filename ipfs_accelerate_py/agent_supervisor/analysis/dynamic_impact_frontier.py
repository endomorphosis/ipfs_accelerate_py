"""Preserve dynamic, reflection, registry, generated, and FFI impact frontiers.

Coverage reports must not silently drop consumers hidden behind reflection,
string dispatch, monkey patches, plugins, runtime DI/registries, callbacks,
generated sources, native/FFI boundaries, remote services, excluded roots, or
unbounded resource limits.  Each such route becomes an explicit bounded
:class:`ImpactFrontierEntry`.

Closure is fail-closed and route-scoped:

* reviewed manifests and root-bound runtime witnesses may close *only* the
  observed route under policy;
* admitted extractors / conservative resolvers may close only when they bind
  the same roots and the exact observed route;
* vector, knowledge-graph, and LLM claims may nominate but never close;
* absent evidence and timeouts remain :attr:`FrontierDisposition.UNKNOWN`;
* complete impact is impossible while any *required* entry is still open.

This adapter does not invent graph edges.  It turns already-collected
observations (graph frontier refs, resolver dynamic dispositions, capability
timeouts, manifests, runtime witnesses) into a deterministic frontier receipt
that impact-closure consumers can attach without redefining
:class:`ImpactClosureReceipt`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from .change_propagation_contracts import (
    CHANGE_PROPAGATION_VERSION,
    ImpactClosureReceipt,
    ImpactCompleteness,
    PropagationAuthorityRoots,
)


DYNAMIC_IMPACT_FRONTIER_VERSION: Final[int] = 1
DYNAMIC_IMPACT_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dynamic-impact-frontier@1"
)
IMPACT_FRONTIER_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/impact-frontier-entry@1"
)
MAX_ENTRIES: Final[int] = 1_024
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_MECHANISM_COUNT: Final[int] = 32
MAX_REASON_CODES: Final[int] = 128

# Producer identity retained on every emission for audit, never authority.
PRODUCER_ID: Final[str] = "dynamic-impact-frontier@1"


class DynamicImpactFrontierError(ValueError):
    """Malformed frontier observation, entry, or policy input."""


class DynamicImpactFrontierBoundsError(DynamicImpactFrontierError):
    """A frontier record exceeded its deterministic compactness bound."""


class DynamicImpactFrontierAuthorityError(DynamicImpactFrontierError):
    """Roots, routes, or closure mechanisms violated the trust boundary."""


class FrontierKind(str, Enum):
    """Closed vocabulary of dynamic / unmodeled impact-frontier categories."""

    REFLECTION = "reflection"
    INTROSPECTION = "introspection"
    STRING_DISPATCH = "string_dispatch"  # getattr / eval / import strings
    MONKEY_PATCH = "monkey_patch"
    PLUGIN_ENTRY_POINT = "plugin_entry_point"
    RUNTIME_DI_REGISTRY = "runtime_di_registry"
    CALLBACK = "callback"
    GENERATED_CODE = "generated_code"
    NATIVE_FFI = "native_ffi"
    REMOTE_SERVICE = "remote_service"
    EXCLUDED_ROOT = "excluded_root"  # vendored / read-only / excluded
    UNBOUNDED_RESOURCE = "unbounded_resource"

    @classmethod
    def coerce(cls, value: Any) -> "FrontierKind":
        """Map fixture aliases and synonyms onto the closed vocabulary."""
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, FrontierKind] = {
            "reflection": cls.REFLECTION,
            "introspection": cls.INTROSPECTION,
            "getattr": cls.STRING_DISPATCH,
            "eval": cls.STRING_DISPATCH,
            "import_string": cls.STRING_DISPATCH,
            "import_strings": cls.STRING_DISPATCH,
            "string_dispatch": cls.STRING_DISPATCH,
            "string_import": cls.STRING_DISPATCH,
            "dynamic_import": cls.STRING_DISPATCH,
            "monkey_patch": cls.MONKEY_PATCH,
            "monkeypatch": cls.MONKEY_PATCH,
            "plugin": cls.PLUGIN_ENTRY_POINT,
            "plugins": cls.PLUGIN_ENTRY_POINT,
            "plugin_registry": cls.PLUGIN_ENTRY_POINT,
            "plugin_entry_point": cls.PLUGIN_ENTRY_POINT,
            "entry_point": cls.PLUGIN_ENTRY_POINT,
            "entry_points": cls.PLUGIN_ENTRY_POINT,
            "registry": cls.RUNTIME_DI_REGISTRY,
            "runtime_registry": cls.RUNTIME_DI_REGISTRY,
            "runtime_di_registry": cls.RUNTIME_DI_REGISTRY,
            "di": cls.RUNTIME_DI_REGISTRY,
            "di_registry": cls.RUNTIME_DI_REGISTRY,
            "callback": cls.CALLBACK,
            "callbacks": cls.CALLBACK,
            "generated": cls.GENERATED_CODE,
            "generated_code": cls.GENERATED_CODE,
            "generated_binding": cls.GENERATED_CODE,
            "ffi": cls.NATIVE_FFI,
            "native": cls.NATIVE_FFI,
            "native_extension": cls.NATIVE_FFI,
            "native_ffi": cls.NATIVE_FFI,
            "remote": cls.REMOTE_SERVICE,
            "remote_service": cls.REMOTE_SERVICE,
            "excluded": cls.EXCLUDED_ROOT,
            "excluded_root": cls.EXCLUDED_ROOT,
            "vendored": cls.EXCLUDED_ROOT,
            "read_only": cls.EXCLUDED_ROOT,
            "readonly": cls.EXCLUDED_ROOT,
            "unbounded_resource": cls.UNBOUNDED_RESOURCE,
            "resource_bound": cls.UNBOUNDED_RESOURCE,
            "unbounded": cls.UNBOUNDED_RESOURCE,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DynamicImpactFrontierError(
                f"unsupported frontier kind: {value!r}"
            ) from exc


class FrontierDisposition(str, Enum):
    """Closed outcome for one frontier entry after policy evaluation."""

    OPEN = "open"
    """Required open frontier; blocks complete impact."""

    CLOSED_OBSERVED_ROUTE = "closed_observed_route"
    """Closed only for the observed route under an admitted mechanism."""

    UNKNOWN = "unknown"
    """Absent evidence or timeout; remains non-authoritative."""

    UNSUPPORTED = "unsupported"
    """No supported closure path for this kind under current policy."""

    NOMINATED_ONLY = "nominated_only"
    """Vector/KG/LLM (or other non-admitted) nomination without closure."""


class ClosureMechanism(str, Enum):
    """Mechanisms that may *close* a frontier entry under policy.

    Vector, knowledge-graph, and LLM claims are intentionally absent: they
    may only nominate and never appear here as closing mechanisms.
    """

    REVIEWED_MANIFEST = "reviewed_manifest"
    ROOT_BOUND_RUNTIME_WITNESS = "root_bound_runtime_witness"
    ADMITTED_EXTRACTOR = "admitted_extractor"
    CONSERVATIVE_RESOLVER = "conservative_resolver"


# Mechanisms that are never allowed to close a route.
_NON_CLOSING_CLAIM_KINDS: Final[frozenset[str]] = frozenset(
    {
        "vector",
        "vector_hit",
        "vector_nomination",
        "kg",
        "knowledge_graph",
        "graphrag",
        "llm",
        "llm_claim",
        "model",
        "history",
        "same_name",
    }
)

# Default supported closure mechanisms per kind (route-scoped only).
_DEFAULT_SUPPORTED_MECHANISMS: Final[Mapping[FrontierKind, tuple[ClosureMechanism, ...]]] = {
    FrontierKind.REFLECTION: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
    ),
    FrontierKind.INTROSPECTION: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
    ),
    FrontierKind.STRING_DISPATCH: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
        ClosureMechanism.CONSERVATIVE_RESOLVER,
    ),
    FrontierKind.MONKEY_PATCH: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
    ),
    FrontierKind.PLUGIN_ENTRY_POINT: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
        ClosureMechanism.ADMITTED_EXTRACTOR,
    ),
    FrontierKind.RUNTIME_DI_REGISTRY: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
        ClosureMechanism.ADMITTED_EXTRACTOR,
    ),
    FrontierKind.CALLBACK: (
        ClosureMechanism.CONSERVATIVE_RESOLVER,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
        ClosureMechanism.REVIEWED_MANIFEST,
    ),
    FrontierKind.GENERATED_CODE: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ADMITTED_EXTRACTOR,
    ),
    FrontierKind.NATIVE_FFI: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
    ),
    FrontierKind.REMOTE_SERVICE: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
    ),
    FrontierKind.EXCLUDED_ROOT: (
        ClosureMechanism.REVIEWED_MANIFEST,
    ),
    FrontierKind.UNBOUNDED_RESOURCE: (
        ClosureMechanism.REVIEWED_MANIFEST,
        ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
    ),
}


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise DynamicImpactFrontierError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise DynamicImpactFrontierError(f"{name} must not be empty")
    if len(text.encode("utf-8")) > limit:
        raise DynamicImpactFrontierBoundsError(f"{name} exceeds its byte bound")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True)
    if any(character.isspace() for character in text):
        raise DynamicImpactFrontierError(f"{name} must be a compact identifier")
    return text


def _route(value: Any, name: str = "route") -> str:
    text = _text(value, name, required=True, limit=MAX_PATH_BYTES)
    return text


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum)
        raise DynamicImpactFrontierError(f"{name} must be one of: {choices}") from exc


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DynamicImpactFrontierError(f"{name} must be a sequence of identifiers")
    else:
        raw = values
    if len(raw) > limit:
        raise DynamicImpactFrontierBoundsError(f"{name} exceeds its item bound")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        ident = _identifier(item, name)
        if ident not in seen:
            seen.add(ident)
            result.append(ident)
    ordered = tuple(sorted(result))
    if required and not ordered:
        raise DynamicImpactFrontierError(f"{name} must not be empty")
    return ordered


def _mechanisms(
    values: Any,
    name: str = "supported_closure_mechanisms",
) -> tuple[ClosureMechanism, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DynamicImpactFrontierError(f"{name} must be a sequence")
    if len(values) > MAX_MECHANISM_COUNT:
        raise DynamicImpactFrontierBoundsError(f"{name} exceeds its item bound")
    result: list[ClosureMechanism] = []
    seen: set[ClosureMechanism] = set()
    for item in values:
        mechanism = _enum(item, ClosureMechanism, name)
        if mechanism not in seen:
            seen.add(mechanism)
            result.append(mechanism)  # type: ignore[arg-type]
    return tuple(sorted(result, key=lambda item: item.value))


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DynamicImpactFrontierError(f"{name} must be a boolean")
    return value


def _roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            PropagationAuthorityRoots.from_dict(value)
            if "schema" in value
            else PropagationAuthorityRoots(**value)
        )
    raise DynamicImpactFrontierError("roots must be PropagationAuthorityRoots")


@dataclass(frozen=True)
class ImpactFrontierEntry:
    """One bounded, explicit dynamic/unmodeled impact frontier.

    Records the observed route, the affected contract reference, evidence,
    supported closure mechanisms, disposition, and reason.  Closure never
    extends past the observed route.
    """

    entry_id: str
    kind: FrontierKind
    disposition: FrontierDisposition
    route: str
    affected_contract_ref: str
    reason: str
    evidence_refs: tuple[str, ...] = ()
    supported_closure_mechanisms: tuple[ClosureMechanism, ...] = ()
    required: bool = True
    closed_by: ClosureMechanism | None = None
    closed_route_only: bool = False
    claim_kind: str = ""
    graph_node_id: str = ""
    graph_edge_id: str = ""
    schema: str = IMPACT_FRONTIER_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _identifier(self.entry_id, "entry_id"))
        object.__setattr__(self, "kind", FrontierKind.coerce(self.kind))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, FrontierDisposition, "disposition"),
        )
        object.__setattr__(self, "route", _route(self.route))
        object.__setattr__(
            self,
            "affected_contract_ref",
            _identifier(self.affected_contract_ref, "affected_contract_ref"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason", required=True))
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(
            self,
            "supported_closure_mechanisms",
            _mechanisms(self.supported_closure_mechanisms),
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        if self.closed_by is not None:
            object.__setattr__(
                self, "closed_by", _enum(self.closed_by, ClosureMechanism, "closed_by")
            )
        object.__setattr__(
            self, "closed_route_only", _bool(self.closed_route_only, "closed_route_only")
        )
        object.__setattr__(
            self, "claim_kind", _text(self.claim_kind, "claim_kind", required=False)
        )
        object.__setattr__(
            self,
            "graph_node_id",
            _text(self.graph_node_id, "graph_node_id", required=False),
        )
        object.__setattr__(
            self,
            "graph_edge_id",
            _text(self.graph_edge_id, "graph_edge_id", required=False),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or IMPACT_FRONTIER_ENTRY_SCHEMA, "schema"),
        )
        if self.schema != IMPACT_FRONTIER_ENTRY_SCHEMA:
            raise DynamicImpactFrontierError(
                f"unsupported frontier entry schema: {self.schema}"
            )
        self._validate_disposition_invariants()

    def _validate_disposition_invariants(self) -> None:
        disposition = self.disposition
        if disposition is FrontierDisposition.CLOSED_OBSERVED_ROUTE:
            if self.closed_by is None:
                raise DynamicImpactFrontierAuthorityError(
                    "closed_observed_route requires closed_by"
                )
            if self.closed_by not in self.supported_closure_mechanisms:
                raise DynamicImpactFrontierAuthorityError(
                    "closed_by must be one of supported_closure_mechanisms"
                )
            if not self.closed_route_only:
                raise DynamicImpactFrontierAuthorityError(
                    "closure is route-scoped; closed_route_only must be true"
                )
            if not self.evidence_refs:
                raise DynamicImpactFrontierAuthorityError(
                    "closed_observed_route requires bounded evidence"
                )
            claim = self.claim_kind.casefold()
            if claim in _NON_CLOSING_CLAIM_KINDS:
                raise DynamicImpactFrontierAuthorityError(
                    "vector/kg/llm claims cannot close a frontier entry"
                )
        else:
            if self.closed_by is not None:
                raise DynamicImpactFrontierAuthorityError(
                    f"{disposition.value} disposition cannot carry closed_by"
                )
            if self.closed_route_only:
                raise DynamicImpactFrontierAuthorityError(
                    f"{disposition.value} disposition cannot claim closed_route_only"
                )

    @property
    def is_open_required(self) -> bool:
        """True when this required entry still blocks complete impact."""
        if not self.required:
            return False
        return self.disposition in {
            FrontierDisposition.OPEN,
            FrontierDisposition.UNKNOWN,
            FrontierDisposition.UNSUPPORTED,
            FrontierDisposition.NOMINATED_ONLY,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_version": DYNAMIC_IMPACT_FRONTIER_VERSION,
            "entry_id": self.entry_id,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "route": self.route,
            "affected_contract_ref": self.affected_contract_ref,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "supported_closure_mechanisms": [
                item.value for item in self.supported_closure_mechanisms
            ],
            "required": self.required,
            "closed_by": self.closed_by.value if self.closed_by is not None else None,
            "closed_route_only": self.closed_route_only,
            "claim_kind": self.claim_kind,
            "graph_node_id": self.graph_node_id,
            "graph_edge_id": self.graph_edge_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImpactFrontierEntry":
        if not isinstance(payload, Mapping):
            raise DynamicImpactFrontierError("frontier entry payload must be a mapping")
        schema = payload.get("schema", IMPACT_FRONTIER_ENTRY_SCHEMA)
        if schema != IMPACT_FRONTIER_ENTRY_SCHEMA:
            raise DynamicImpactFrontierError(
                f"unsupported frontier entry schema: {schema}"
            )
        closed_by = payload.get("closed_by")
        return cls(
            entry_id=str(payload.get("entry_id") or ""),
            kind=payload.get("kind", FrontierKind.REFLECTION),
            disposition=payload.get("disposition", FrontierDisposition.OPEN),
            route=str(payload.get("route") or ""),
            affected_contract_ref=str(payload.get("affected_contract_ref") or ""),
            reason=str(payload.get("reason") or ""),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            supported_closure_mechanisms=tuple(
                payload.get("supported_closure_mechanisms") or ()
            ),
            required=bool(payload.get("required", True)),
            closed_by=closed_by,
            closed_route_only=bool(payload.get("closed_route_only", False)),
            claim_kind=str(payload.get("claim_kind") or ""),
            graph_node_id=str(payload.get("graph_node_id") or ""),
            graph_edge_id=str(payload.get("graph_edge_id") or ""),
        )


@dataclass(frozen=True)
class DynamicImpactFrontier:
    """Bounded collection of dynamic frontier entries for one delta.

    ``impact_completeness_possible`` is false whenever any required entry is
    still open/unknown/unsupported/nominated-only.  Complete impact closure
    receipts cannot be emitted while this report reports open required entries.
    """

    roots: PropagationAuthorityRoots
    delta_id: str
    entries: tuple[ImpactFrontierEntry, ...]
    completeness: ImpactCompleteness
    open_required_entry_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    graph_id: str = ""
    timeout: bool = False
    producer_id: str = PRODUCER_ID
    schema: str = DYNAMIC_IMPACT_FRONTIER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        if isinstance(self.entries, (str, bytes, bytearray)) or not isinstance(
            self.entries, Sequence
        ):
            raise DynamicImpactFrontierError("entries must be a sequence")
        if len(self.entries) > MAX_ENTRIES:
            raise DynamicImpactFrontierBoundsError("entries exceeds its item bound")
        entries = tuple(self.entries)
        if not all(isinstance(item, ImpactFrontierEntry) for item in entries):
            raise DynamicImpactFrontierError(
                "entries must contain ImpactFrontierEntry values"
            )
        entry_ids = [item.entry_id for item in entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise DynamicImpactFrontierError("frontier entry_ids must be unique")
        # Deterministic order by (kind, route, entry_id).
        ordered = tuple(
            sorted(
                entries,
                key=lambda item: (item.kind.value, item.route, item.entry_id),
            )
        )
        object.__setattr__(self, "entries", ordered)
        open_required = tuple(
            item.entry_id for item in ordered if item.is_open_required
        )
        object.__setattr__(self, "open_required_entry_ids", open_required)
        object.__setattr__(
            self,
            "completeness",
            _enum(self.completeness, ImpactCompleteness, "completeness"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes", limit=MAX_REASON_CODES)
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(
            self, "graph_id", _text(self.graph_id, "graph_id", required=False)
        )
        object.__setattr__(self, "timeout", _bool(self.timeout, "timeout"))
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or DYNAMIC_IMPACT_FRONTIER_SCHEMA, "schema"),
        )
        if self.schema != DYNAMIC_IMPACT_FRONTIER_SCHEMA:
            raise DynamicImpactFrontierError(
                f"unsupported frontier schema: {self.schema}"
            )
        if self.graph_id and self.graph_id != self.roots.graph_id:
            raise DynamicImpactFrontierAuthorityError(
                "graph_id must match propagation authority roots"
            )
        # Fail closed: complete is impossible with open required entries.
        if open_required:
            if self.completeness is ImpactCompleteness.COMPLETE:
                raise DynamicImpactFrontierAuthorityError(
                    "complete impact is impossible while a required entry is open"
                )
        elif self.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER:
            # No open required entries — partial_with_frontier only when some
            # non-required entries remain open, which is still a frontier.
            non_required_open = any(
                not item.required
                and item.disposition
                in {
                    FrontierDisposition.OPEN,
                    FrontierDisposition.UNKNOWN,
                    FrontierDisposition.UNSUPPORTED,
                    FrontierDisposition.NOMINATED_ONLY,
                }
                for item in ordered
            )
            if not non_required_open and ordered:
                # All entries closed and none open → should be COMPLETE.
                raise DynamicImpactFrontierAuthorityError(
                    "partial_with_frontier requires an open frontier entry"
                )

    @property
    def impact_completeness_possible(self) -> bool:
        return not self.open_required_entry_ids

    @property
    def open_entries(self) -> tuple[ImpactFrontierEntry, ...]:
        return tuple(item for item in self.entries if item.is_open_required)

    def entry_ids_by_kind(self) -> Mapping[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for item in self.entries:
            grouped.setdefault(item.kind.value, []).append(item.entry_id)
        return MappingProxyType({key: tuple(value) for key, value in sorted(grouped.items())})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_version": DYNAMIC_IMPACT_FRONTIER_VERSION,
            "change_propagation_version": CHANGE_PROPAGATION_VERSION,
            "roots": self.roots.to_dict(),
            "delta_id": self.delta_id,
            "entries": [item.to_dict() for item in self.entries],
            "completeness": self.completeness.value,
            "open_required_entry_ids": list(self.open_required_entry_ids),
            "reason_codes": list(self.reason_codes),
            "evidence_refs": list(self.evidence_refs),
            "graph_id": self.graph_id,
            "timeout": self.timeout,
            "producer_id": self.producer_id,
            "impact_completeness_possible": self.impact_completeness_possible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DynamicImpactFrontier":
        if not isinstance(payload, Mapping):
            raise DynamicImpactFrontierError("frontier payload must be a mapping")
        schema = payload.get("schema", DYNAMIC_IMPACT_FRONTIER_SCHEMA)
        if schema != DYNAMIC_IMPACT_FRONTIER_SCHEMA:
            raise DynamicImpactFrontierError(f"unsupported frontier schema: {schema}")
        entries = tuple(
            ImpactFrontierEntry.from_dict(item)
            if isinstance(item, Mapping)
            else item
            for item in (payload.get("entries") or ())
        )
        return cls(
            roots=_roots(payload.get("roots")),
            delta_id=str(payload.get("delta_id") or ""),
            entries=entries,
            completeness=payload.get(
                "completeness", ImpactCompleteness.PARTIAL_WITH_FRONTIER
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            graph_id=str(payload.get("graph_id") or ""),
            timeout=bool(payload.get("timeout", False)),
            producer_id=str(payload.get("producer_id") or PRODUCER_ID),
        )

    def apply_to_closure_receipt(
        self, receipt: ImpactClosureReceipt
    ) -> ImpactClosureReceipt:
        """Project open frontier node/edge ids onto an existing closure receipt.

        Never upgrades completeness to COMPLETE while required entries remain
        open.  Returns a new :class:`ImpactClosureReceipt` with the union of
        frontier node/edge ids and a fail-closed completeness disposition.
        """
        if not isinstance(receipt, ImpactClosureReceipt):
            raise DynamicImpactFrontierError(
                "receipt must be an ImpactClosureReceipt"
            )
        if receipt.roots != self.roots:
            raise DynamicImpactFrontierAuthorityError(
                "closure receipt roots must match the frontier roots"
            )
        if receipt.delta_id != self.delta_id:
            raise DynamicImpactFrontierAuthorityError(
                "closure receipt delta_id must match the frontier delta_id"
            )
        node_ids = set(receipt.frontier_node_ids)
        edge_ids = set(receipt.frontier_edge_ids)
        for entry in self.entries:
            if entry.is_open_required:
                if entry.graph_node_id:
                    node_ids.add(entry.graph_node_id)
                else:
                    node_ids.add(f"frontier:{entry.entry_id}")
                if entry.graph_edge_id:
                    edge_ids.add(entry.graph_edge_id)
        evidence = tuple(
            sorted(set(receipt.evidence_refs) | set(self.evidence_refs))
        )
        if self.open_required_entry_ids:
            completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            if not node_ids and not edge_ids:
                node_ids = {f"frontier:{entry_id}" for entry_id in self.open_required_entry_ids}
        else:
            completeness = receipt.completeness
            if completeness is ImpactCompleteness.COMPLETE and (node_ids or edge_ids):
                completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
        return ImpactClosureReceipt(
            roots=receipt.roots,
            delta_id=receipt.delta_id,
            completeness=completeness,
            consumers=receipt.consumers,
            sccs=receipt.sccs,
            frontier_node_ids=tuple(sorted(node_ids)),
            frontier_edge_ids=tuple(sorted(edge_ids)),
            excluded_refs=receipt.excluded_refs,
            validation_refs=receipt.validation_refs,
            resource_bound_refs=receipt.resource_bound_refs,
            evidence_refs=evidence,
        )


@dataclass(frozen=True)
class FrontierObservation:
    """One raw observation before policy disposition is applied.

    Callers supply already-collected graph/resolver/manifest/runtime facts.
    The analyzer never invents sites from package presence or vector hits.
    """

    kind: FrontierKind | str
    route: str
    affected_contract_ref: str
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    required: bool = True
    claim_kind: str = ""
    graph_node_id: str = ""
    graph_edge_id: str = ""
    entry_id: str = ""
    supported_closure_mechanisms: tuple[ClosureMechanism | str, ...] | None = None
    timed_out: bool = False
    absent_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", FrontierKind.coerce(self.kind))
        object.__setattr__(self, "route", _route(self.route))
        object.__setattr__(
            self,
            "affected_contract_ref",
            _identifier(self.affected_contract_ref, "affected_contract_ref"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self, "evidence_refs", _ids(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(
            self, "claim_kind", _text(self.claim_kind, "claim_kind", required=False)
        )
        object.__setattr__(
            self,
            "graph_node_id",
            _text(self.graph_node_id, "graph_node_id", required=False),
        )
        object.__setattr__(
            self,
            "graph_edge_id",
            _text(self.graph_edge_id, "graph_edge_id", required=False),
        )
        if self.entry_id:
            object.__setattr__(
                self, "entry_id", _identifier(self.entry_id, "entry_id")
            )
        if self.supported_closure_mechanisms is not None:
            object.__setattr__(
                self,
                "supported_closure_mechanisms",
                _mechanisms(self.supported_closure_mechanisms),
            )
        object.__setattr__(self, "timed_out", _bool(self.timed_out, "timed_out"))
        object.__setattr__(
            self, "absent_evidence", _bool(self.absent_evidence, "absent_evidence")
        )


@dataclass(frozen=True)
class ClosureAttempt:
    """A proposed route-scoped closure under an admitted mechanism.

    Vector/KG/LLM claims are rejected even if supplied here: they cannot
    appear as :class:`ClosureMechanism` values.
    """

    entry_route: str
    mechanism: ClosureMechanism | str
    evidence_refs: tuple[str, ...]
    roots_graph_id: str = ""
    claim_kind: str = ""
    observed_route_only: bool = True
    kind: FrontierKind | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_route", _route(self.entry_route, "entry_route"))
        object.__setattr__(
            self, "mechanism", _enum(self.mechanism, ClosureMechanism, "mechanism")
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _ids(self.evidence_refs, "evidence_refs", required=True),
        )
        object.__setattr__(
            self,
            "roots_graph_id",
            _text(self.roots_graph_id, "roots_graph_id", required=False),
        )
        object.__setattr__(
            self, "claim_kind", _text(self.claim_kind, "claim_kind", required=False)
        )
        object.__setattr__(
            self,
            "observed_route_only",
            _bool(self.observed_route_only, "observed_route_only"),
        )
        if self.kind is not None:
            object.__setattr__(self, "kind", FrontierKind.coerce(self.kind))


@runtime_checkable
class ProgramGraphFrontierSource(Protocol):
    """Minimal graph surface used to harvest explicit frontier refs."""

    @property
    def frontier_refs(self) -> Sequence[str]: ...

    @property
    def exclusion_refs(self) -> Sequence[str]: ...

    @property
    def complete(self) -> bool: ...


@runtime_checkable
class ProgramCallResolverFrontierSource(Protocol):
    """Minimal resolver surface that can report dynamic dispositions."""

    def frontier_observations(
        self, roots: PropagationAuthorityRoots, delta_id: str
    ) -> Sequence[FrontierObservation | Mapping[str, Any]]: ...


def _default_entry_id(kind: FrontierKind, route: str, index: int = 0) -> str:
    """Deterministic id from kind + route (index only disambiguates collisions)."""
    safe_route = route.replace("/", ".").replace(":", ".")
    if index:
        return f"frontier:{kind.value}:{safe_route}:{index}"
    return f"frontier:{kind.value}:{safe_route}"


def _default_reason(kind: FrontierKind, disposition: FrontierDisposition) -> str:
    if disposition is FrontierDisposition.CLOSED_OBSERVED_ROUTE:
        return (
            f"{kind.value} frontier closed for the observed route only under policy"
        )
    if disposition is FrontierDisposition.UNKNOWN:
        return f"{kind.value} frontier remains unknown (absent evidence or timeout)"
    if disposition is FrontierDisposition.NOMINATED_ONLY:
        return (
            f"{kind.value} nominated by non-authoritative claim; cannot close coverage"
        )
    if disposition is FrontierDisposition.UNSUPPORTED:
        return f"{kind.value} frontier has no supported closure path under policy"
    return (
        f"{kind.value} remains an explicit impact frontier until admitted evidence "
        "closes the observed route"
    )


def _normalize_observation(
    value: FrontierObservation | Mapping[str, Any] | ImpactFrontierEntry,
) -> FrontierObservation:
    if isinstance(value, FrontierObservation):
        return value
    if isinstance(value, ImpactFrontierEntry):
        return FrontierObservation(
            kind=value.kind,
            route=value.route,
            affected_contract_ref=value.affected_contract_ref,
            reason=value.reason,
            evidence_refs=value.evidence_refs,
            required=value.required,
            claim_kind=value.claim_kind,
            graph_node_id=value.graph_node_id,
            graph_edge_id=value.graph_edge_id,
            entry_id=value.entry_id,
            supported_closure_mechanisms=value.supported_closure_mechanisms,
        )
    if isinstance(value, Mapping):
        return FrontierObservation(
            kind=value.get("kind", FrontierKind.REFLECTION),
            route=str(value.get("route") or value.get("site") or ""),
            affected_contract_ref=str(
                value.get("affected_contract_ref")
                or value.get("contract_ref")
                or "contract:unspecified"
            ),
            reason=str(value.get("reason") or ""),
            evidence_refs=tuple(value.get("evidence_refs") or ()),
            required=bool(value.get("required", True)),
            claim_kind=str(value.get("claim_kind") or ""),
            graph_node_id=str(value.get("graph_node_id") or ""),
            graph_edge_id=str(value.get("graph_edge_id") or ""),
            entry_id=str(value.get("entry_id") or ""),
            supported_closure_mechanisms=(
                tuple(value["supported_closure_mechanisms"])
                if "supported_closure_mechanisms" in value
                else None
            ),
            timed_out=bool(value.get("timed_out", False)),
            absent_evidence=bool(value.get("absent_evidence", False)),
        )
    raise DynamicImpactFrontierError(
        "observation must be FrontierObservation, ImpactFrontierEntry, or mapping"
    )


def _observations_from_graph(
    graph: ProgramGraphFrontierSource | None,
    *,
    affected_contract_ref: str,
) -> list[FrontierObservation]:
    if graph is None:
        return []
    observations: list[FrontierObservation] = []
    try:
        frontier_refs = tuple(graph.frontier_refs or ())
        exclusion_refs = tuple(graph.exclusion_refs or ())
    except (AttributeError, TypeError):
        return []
    for ref in frontier_refs:
        text = str(ref or "").strip()
        if not text:
            continue
        kind = _infer_kind_from_ref(text)
        observations.append(
            FrontierObservation(
                kind=kind,
                route=text if ":" in text or "/" in text else f"graph:{text}",
                affected_contract_ref=affected_contract_ref,
                reason=f"graph frontier ref {text}",
                evidence_refs=(f"graph-frontier:{text}",),
                graph_node_id=text if text.startswith("node:") else "",
            )
        )
    for ref in exclusion_refs:
        text = str(ref or "").strip()
        if not text:
            continue
        observations.append(
            FrontierObservation(
                kind=FrontierKind.EXCLUDED_ROOT,
                route=text if "/" in text else f"excluded:{text}",
                affected_contract_ref=affected_contract_ref,
                reason=f"excluded or vendored root {text}",
                evidence_refs=(f"graph-exclusion:{text}",),
            )
        )
    return observations


def _infer_kind_from_ref(ref: str) -> FrontierKind:
    lower = ref.casefold()
    for token, kind in (
        ("reflection", FrontierKind.REFLECTION),
        ("introspect", FrontierKind.INTROSPECTION),
        ("getattr", FrontierKind.STRING_DISPATCH),
        ("eval", FrontierKind.STRING_DISPATCH),
        ("import", FrontierKind.STRING_DISPATCH),
        ("string_dispatch", FrontierKind.STRING_DISPATCH),
        ("monkey", FrontierKind.MONKEY_PATCH),
        ("plugin", FrontierKind.PLUGIN_ENTRY_POINT),
        ("entry_point", FrontierKind.PLUGIN_ENTRY_POINT),
        ("registry", FrontierKind.RUNTIME_DI_REGISTRY),
        ("di_", FrontierKind.RUNTIME_DI_REGISTRY),
        ("callback", FrontierKind.CALLBACK),
        ("generated", FrontierKind.GENERATED_CODE),
        ("ffi", FrontierKind.NATIVE_FFI),
        ("native", FrontierKind.NATIVE_FFI),
        ("remote", FrontierKind.REMOTE_SERVICE),
        ("vendor", FrontierKind.EXCLUDED_ROOT),
        ("excluded", FrontierKind.EXCLUDED_ROOT),
        ("read_only", FrontierKind.EXCLUDED_ROOT),
        ("resource", FrontierKind.UNBOUNDED_RESOURCE),
        ("unbounded", FrontierKind.UNBOUNDED_RESOURCE),
    ):
        if token in lower:
            return kind
    return FrontierKind.REFLECTION


def _capability_timed_out(report: Any) -> bool:
    if report is None:
        return False
    # Accept ChangePropagationCapabilityReport or duck-typed equivalents.
    capabilities = getattr(report, "capabilities", None)
    if capabilities is None and isinstance(report, Mapping):
        capabilities = report.get("capabilities")
    if not capabilities:
        status = getattr(report, "status", None)
        if status is None and isinstance(report, Mapping):
            status = report.get("status")
        if status is not None and str(getattr(status, "value", status)).casefold() in {
            "timed_out",
            "timeout",
        }:
            return True
        return bool(getattr(report, "timeout", False) or getattr(report, "timed_out", False))
    for item in capabilities:
        status = getattr(item, "status", None)
        if status is None and isinstance(item, Mapping):
            status = item.get("status")
        raw = str(getattr(status, "value", status) or "").casefold()
        if raw in {"timed_out", "timeout"}:
            return True
    return False


class DynamicImpactFrontierAnalyzer:
    """Emit and close (route-scoped) dynamic impact frontier entries.

    Inputs are observations plus optional graph / resolver / capability
    surfaces.  Policy evaluation is deterministic and fail-closed.
    """

    def __init__(
        self,
        *,
        default_required: bool = True,
        allow_non_required_kinds: Iterable[FrontierKind] | None = None,
    ) -> None:
        self._default_required = bool(default_required)
        self._allow_non_required = frozenset(allow_non_required_kinds or ())

    def analyze(
        self,
        roots: PropagationAuthorityRoots | Mapping[str, Any],
        delta_id: str,
        observations: Sequence[
            FrontierObservation | Mapping[str, Any] | ImpactFrontierEntry
        ] = (),
        *,
        graph: ProgramGraphFrontierSource | None = None,
        resolver: ProgramCallResolverFrontierSource | None = None,
        capability_report: Any = None,
        closure_attempts: Sequence[ClosureAttempt | Mapping[str, Any]] = (),
        affected_contract_ref: str = "contract:delta",
        timeout: bool = False,
    ) -> DynamicImpactFrontier:
        authority = _roots(roots)
        delta = _identifier(delta_id, "delta_id")
        contract_ref = _identifier(affected_contract_ref, "affected_contract_ref")

        collected: list[FrontierObservation] = [
            _normalize_observation(item) for item in observations
        ]
        collected.extend(
            _observations_from_graph(graph, affected_contract_ref=contract_ref)
        )
        if resolver is not None:
            try:
                extra = resolver.frontier_observations(authority, delta)
            except (AttributeError, NotImplementedError, TypeError, ValueError):
                extra = ()
            for item in extra or ():
                collected.append(_normalize_observation(item))

        global_timeout = bool(timeout) or _capability_timed_out(capability_report)
        report_timeout = global_timeout or any(item.timed_out for item in collected)
        attempts = self._normalize_attempts(closure_attempts)

        # Sort before disposal so entry_ids and output order are input-order
        # independent (kind, route, contract, then stable original index).
        indexed = list(enumerate(collected))
        indexed.sort(
            key=lambda pair: (
                pair[1].kind.value,  # type: ignore[union-attr]
                pair[1].route,
                pair[1].affected_contract_ref,
                pair[0],
            )
        )

        entries: list[ImpactFrontierEntry] = []
        reason_codes: list[str] = []
        evidence: list[str] = []
        used_ids: dict[str, int] = {}

        for _original_index, observation in indexed:
            kind = observation.kind  # type: ignore[assignment]
            assert isinstance(kind, FrontierKind)
            base_id = observation.entry_id or _default_entry_id(kind, observation.route)
            collision = used_ids.get(base_id, 0)
            used_ids[base_id] = collision + 1
            entry_id = base_id if collision == 0 else f"{base_id}:{collision}"
            entry = self._dispose_entry(
                observation,
                entry_id=entry_id,
                authority=authority,
                attempts=attempts,
                # Global/capability timeout applies to every entry; per-observation
                # timeout only affects that route.
                timed_out=global_timeout or observation.timed_out,
                default_contract_ref=contract_ref,
            )
            entries.append(entry)
            evidence.extend(entry.evidence_refs)
            reason_codes.append(f"{entry.kind.value}:{entry.disposition.value}")

        if report_timeout and not any(
            item.disposition is FrontierDisposition.UNKNOWN for item in entries
        ):
            # Global timeout without per-entry observations still records unknown.
            reason_codes.append("timeout")

        open_required = [item for item in entries if item.is_open_required]
        if open_required:
            completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            reason_codes.append("required_frontier_open")
        elif entries:
            # All required closed; still may have optional open (handled above).
            if any(
                item.disposition
                in {
                    FrontierDisposition.OPEN,
                    FrontierDisposition.UNKNOWN,
                    FrontierDisposition.UNSUPPORTED,
                    FrontierDisposition.NOMINATED_ONLY,
                }
                for item in entries
            ):
                completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            else:
                completeness = ImpactCompleteness.COMPLETE
        else:
            # No dynamic frontier observations — not a claim of complete impact;
            # that belongs to ImpactClosureReceipt.  Empty frontier is complete
            # *for this adapter's scope*.
            completeness = ImpactCompleteness.COMPLETE

        return DynamicImpactFrontier(
            roots=authority,
            delta_id=delta,
            entries=tuple(entries),
            completeness=completeness,
            reason_codes=tuple(sorted(set(reason_codes))),
            evidence_refs=tuple(sorted(set(evidence))),
            graph_id=authority.graph_id,
            timeout=report_timeout,
        )

    def _normalize_attempts(
        self, attempts: Sequence[ClosureAttempt | Mapping[str, Any]]
    ) -> tuple[ClosureAttempt, ...]:
        result: list[ClosureAttempt] = []
        for item in attempts or ():
            if isinstance(item, ClosureAttempt):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(
                    ClosureAttempt(
                        entry_route=str(item.get("entry_route") or item.get("route") or ""),
                        mechanism=item.get(
                            "mechanism", ClosureMechanism.REVIEWED_MANIFEST
                        ),
                        evidence_refs=tuple(item.get("evidence_refs") or ()),
                        roots_graph_id=str(item.get("roots_graph_id") or ""),
                        claim_kind=str(item.get("claim_kind") or ""),
                        observed_route_only=bool(
                            item.get("observed_route_only", True)
                        ),
                        kind=item.get("kind"),
                    )
                )
            else:
                raise DynamicImpactFrontierError(
                    "closure_attempts must be ClosureAttempt or mapping values"
                )
        return tuple(result)

    def _dispose_entry(
        self,
        observation: FrontierObservation,
        *,
        entry_id: str,
        authority: PropagationAuthorityRoots,
        attempts: Sequence[ClosureAttempt],
        timed_out: bool,
        default_contract_ref: str,
    ) -> ImpactFrontierEntry:
        kind = observation.kind  # type: ignore[assignment]
        assert isinstance(kind, FrontierKind)
        supported = (
            observation.supported_closure_mechanisms
            if observation.supported_closure_mechanisms is not None
            else _DEFAULT_SUPPORTED_MECHANISMS.get(kind, ())
        )
        assert isinstance(supported, tuple)
        required = (
            observation.required
            if observation.required is not None
            else self._default_required
        )
        if kind in self._allow_non_required:
            required = False

        claim = observation.claim_kind.casefold()
        evidence = list(observation.evidence_refs)

        # Timeout / absent evidence → UNKNOWN (never closed).
        if timed_out or observation.timed_out:
            disposition = FrontierDisposition.UNKNOWN
            reason = observation.reason or _default_reason(kind, disposition)
            return ImpactFrontierEntry(
                entry_id=entry_id,
                kind=kind,
                disposition=disposition,
                route=observation.route,
                affected_contract_ref=observation.affected_contract_ref
                or default_contract_ref,
                reason=reason,
                evidence_refs=tuple(evidence),
                supported_closure_mechanisms=supported,  # type: ignore[arg-type]
                required=required,
                claim_kind=observation.claim_kind,
                graph_node_id=observation.graph_node_id,
                graph_edge_id=observation.graph_edge_id,
            )

        if observation.absent_evidence and not evidence:
            disposition = FrontierDisposition.UNKNOWN
            reason = observation.reason or _default_reason(kind, disposition)
            return ImpactFrontierEntry(
                entry_id=entry_id,
                kind=kind,
                disposition=disposition,
                route=observation.route,
                affected_contract_ref=observation.affected_contract_ref
                or default_contract_ref,
                reason=reason,
                evidence_refs=(),
                supported_closure_mechanisms=supported,  # type: ignore[arg-type]
                required=required,
                claim_kind=observation.claim_kind,
                graph_node_id=observation.graph_node_id,
                graph_edge_id=observation.graph_edge_id,
            )

        # Non-authoritative nominations never close.
        if claim in _NON_CLOSING_CLAIM_KINDS:
            disposition = FrontierDisposition.NOMINATED_ONLY
            reason = observation.reason or _default_reason(kind, disposition)
            return ImpactFrontierEntry(
                entry_id=entry_id,
                kind=kind,
                disposition=disposition,
                route=observation.route,
                affected_contract_ref=observation.affected_contract_ref
                or default_contract_ref,
                reason=reason,
                evidence_refs=tuple(evidence),
                supported_closure_mechanisms=supported,  # type: ignore[arg-type]
                required=required,
                claim_kind=observation.claim_kind,
                graph_node_id=observation.graph_node_id,
                graph_edge_id=observation.graph_edge_id,
            )

        matching = [
            attempt
            for attempt in attempts
            if attempt.entry_route == observation.route
            and (attempt.kind is None or attempt.kind == kind)
        ]
        for attempt in matching:
            claim_attempt = attempt.claim_kind.casefold()
            if claim_attempt in _NON_CLOSING_CLAIM_KINDS:
                # Explicit rejection of vector/kg/llm closure attempts.
                continue
            if not attempt.observed_route_only:
                # Policy: may close only the observed route.
                continue
            if attempt.roots_graph_id and attempt.roots_graph_id != authority.graph_id:
                continue
            mechanism = attempt.mechanism
            assert isinstance(mechanism, ClosureMechanism)
            if mechanism not in supported:
                continue
            if not attempt.evidence_refs:
                continue
            merged_evidence = tuple(sorted(set(evidence) | set(attempt.evidence_refs)))
            disposition = FrontierDisposition.CLOSED_OBSERVED_ROUTE
            reason = observation.reason or _default_reason(kind, disposition)
            return ImpactFrontierEntry(
                entry_id=entry_id,
                kind=kind,
                disposition=disposition,
                route=observation.route,
                affected_contract_ref=observation.affected_contract_ref
                or default_contract_ref,
                reason=reason,
                evidence_refs=merged_evidence,
                supported_closure_mechanisms=supported,  # type: ignore[arg-type]
                required=required,
                closed_by=mechanism,
                closed_route_only=True,
                claim_kind=observation.claim_kind,
                graph_node_id=observation.graph_node_id,
                graph_edge_id=observation.graph_edge_id,
            )

        # No admitted closure → remain OPEN (or UNKNOWN if no evidence at all).
        if not evidence:
            disposition = FrontierDisposition.UNKNOWN
        else:
            disposition = FrontierDisposition.OPEN
        reason = observation.reason or _default_reason(kind, disposition)
        return ImpactFrontierEntry(
            entry_id=entry_id,
            kind=kind,
            disposition=disposition,
            route=observation.route,
            affected_contract_ref=observation.affected_contract_ref
            or default_contract_ref,
            reason=reason,
            evidence_refs=tuple(evidence),
            supported_closure_mechanisms=supported,  # type: ignore[arg-type]
            required=required,
            claim_kind=observation.claim_kind,
            graph_node_id=observation.graph_node_id,
            graph_edge_id=observation.graph_edge_id,
        )


def all_frontier_kinds() -> tuple[FrontierKind, ...]:
    """Stable ordered list of every required frontier kind."""
    return tuple(FrontierKind)


def required_kind_coverage(entries: Sequence[ImpactFrontierEntry]) -> frozenset[FrontierKind]:
    """Return the set of frontier kinds represented in *entries*."""
    return frozenset(item.kind for item in entries)


__all__ = [
    "CLOSURE_MECHANISM_ALIASES",
    "ClosureAttempt",
    "ClosureMechanism",
    "DYNAMIC_IMPACT_FRONTIER_SCHEMA",
    "DYNAMIC_IMPACT_FRONTIER_VERSION",
    "DynamicImpactFrontier",
    "DynamicImpactFrontierAnalyzer",
    "DynamicImpactFrontierAuthorityError",
    "DynamicImpactFrontierBoundsError",
    "DynamicImpactFrontierError",
    "FrontierDisposition",
    "FrontierKind",
    "FrontierObservation",
    "IMPACT_FRONTIER_ENTRY_SCHEMA",
    "ImpactFrontierEntry",
    "MAX_ENTRIES",
    "PRODUCER_ID",
    "ProgramCallResolverFrontierSource",
    "ProgramGraphFrontierSource",
    "all_frontier_kinds",
    "required_kind_coverage",
]


# Back-compat alias map exposed for tests / callers that introspect policy.
CLOSURE_MECHANISM_ALIASES: Final[Mapping[str, ClosureMechanism]] = MappingProxyType(
    {
        "reviewed_manifest": ClosureMechanism.REVIEWED_MANIFEST,
        "manifest": ClosureMechanism.REVIEWED_MANIFEST,
        "root_bound_runtime_witness": ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
        "runtime_witness": ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
        "admitted_extractor": ClosureMechanism.ADMITTED_EXTRACTOR,
        "extractor": ClosureMechanism.ADMITTED_EXTRACTOR,
        "conservative_resolver": ClosureMechanism.CONSERVATIVE_RESOLVER,
        "resolver": ClosureMechanism.CONSERVATIVE_RESOLVER,
    }
)
