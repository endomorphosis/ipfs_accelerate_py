"""Conservative semantic conflict sets (EAAEF-081).

Overlapping symbols, files, interfaces, schemas, authorities, resources, or
effects serialize unless an explicit named merge contract proves compatibility
for that exact overlap. Unknown scope always conflicts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


CONFLICT_GRAPH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-conflict-graph@1"
)
CONFLICT_GRAPH_INTERFACE: Final[str] = "ExternalConflictGraph@1"

SCOPE_DOMAINS: Final[tuple[str, ...]] = (
    "files",
    "effects",
    "symbols",
    "interfaces",
    "schemas",
    "authorities",
    "resources",
)
_DOMAIN_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "file": "files",
        "files": "files",
        "write": "files",
        "writes": "files",
        "write_scope": "files",
        "effect": "effects",
        "effects": "effects",
        "effect_scope": "effects",
        "symbol": "symbols",
        "symbols": "symbols",
        "interface": "interfaces",
        "interfaces": "interfaces",
        "schema": "schemas",
        "schemas": "schemas",
        "authority": "authorities",
        "authorities": "authorities",
        "resource": "resources",
        "resources": "resources",
    }
)
_UNKNOWN_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "*",
        "?",
        "<unknown>",
        "unknown",
        "unscoped",
        "unspecified",
    }
)


class ConflictGraphError(ValueError):
    """Malformed conflict-graph input."""


def _text(value: object, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ConflictGraphError(f"{name} is required")
    return text


def _tuple_text(values: object, name: str) -> tuple[str, ...]:
    if values is None:
        items: Sequence[object] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence):
        items = values
    else:
        raise ConflictGraphError(f"{name} must be a list of strings")
    result = tuple(_text(item, name, required=True) for item in items)
    if len(set(result)) != len(result):
        raise ConflictGraphError(f"{name} contains duplicates")
    return result


def _is_unknown_token(value: str) -> bool:
    return value.strip().lower() in _UNKNOWN_TOKENS


def _domain_items(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {domain: [] for domain in SCOPE_DOMAINS}
    for key, raw in payload.items():
        domain = _DOMAIN_ALIASES.get(str(key).strip())
        if domain is None or raw is None:
            continue
        for item in _tuple_text(raw, key):
            if item not in collected[domain]:
                collected[domain].append(item)
    return {domain: tuple(items) for domain, items in collected.items()}


def _unknown_from_items(items_by_domain: Mapping[str, Sequence[str]]) -> bool:
    return any(_is_unknown_token(item) for items in items_by_domain.values() for item in items)


def _mapping_declares_scope(payload: Mapping[str, Any]) -> bool:
    return any(str(key).strip() in _DOMAIN_ALIASES for key in payload)


@dataclass(frozen=True)
class TaskScope:
    """Write/effect and related semantic scopes for one task."""

    task_id: str
    files: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    authorities: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    unknown: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        for domain in SCOPE_DOMAINS:
            object.__setattr__(self, domain, _tuple_text(getattr(self, domain), domain))
        unknown = bool(self.unknown) or _unknown_from_items(self.items_by_domain())
        object.__setattr__(self, "unknown", unknown)

    def items_by_domain(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType({domain: tuple(getattr(self, domain)) for domain in SCOPE_DOMAINS})

    def to_dict(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "write_scope": list(self.files),
            "effect_scope": list(self.effects),
            "unknown": self.unknown,
        }
        for domain in SCOPE_DOMAINS:
            if domain in {"files", "effects"}:
                continue
            payload[domain] = list(getattr(self, domain))
        return MappingProxyType(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | TaskScope) -> "TaskScope":
        if isinstance(payload, TaskScope):
            return payload
        if not isinstance(payload, Mapping):
            raise ConflictGraphError("task scope must be an object")
        items = _domain_items(payload)
        unknown = bool(payload.get("unknown")) or not _mapping_declares_scope(payload)
        return cls(
            task_id=str(payload.get("task_id") or payload.get("id") or ""),
            unknown=unknown,
            **items,
        )


@dataclass(frozen=True)
class MergeContract:
    """Named contract that may admit an exact overlap, never an unknown scope."""

    name: str
    files: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    authorities: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "merge contract name"))
        for domain in SCOPE_DOMAINS:
            object.__setattr__(self, domain, _tuple_text(getattr(self, domain), domain))
            if _unknown_from_items({domain: getattr(self, domain)}):
                raise ConflictGraphError("merge contract cannot admit unknown scope")

    def items_by_domain(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType({domain: tuple(getattr(self, domain)) for domain in SCOPE_DOMAINS})

    def to_dict(self) -> Mapping[str, Any]:
        payload = {"name": self.name}
        for domain in SCOPE_DOMAINS:
            items = list(getattr(self, domain))
            if items:
                payload[domain] = items
        return MappingProxyType(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | MergeContract) -> "MergeContract":
        if isinstance(payload, MergeContract):
            return payload
        if not isinstance(payload, Mapping):
            raise ConflictGraphError("merge contract must be an object")
        nested = payload.get("admits")
        combined: dict[str, Any] = dict(payload)
        if isinstance(nested, Mapping):
            combined.update(dict(nested))
        domain = str(payload.get("domain") or "").strip()
        item = payload.get("item")
        if domain and item is not None:
            combined.setdefault(domain, item)
        items = _domain_items(combined)
        return cls(name=str(payload.get("name") or ""), **items)


def _normalize_contracts(
    contracts: Sequence[MergeContract | Mapping[str, Any]] | None,
) -> tuple[MergeContract, ...]:
    if contracts is None:
        return ()
    compiled = tuple(MergeContract.from_mapping(contract) for contract in contracts)
    names = tuple(contract.name for contract in compiled)
    if len(set(names)) != len(names):
        raise ConflictGraphError("merge contract names must be unique")
    return compiled


def _evaluate(
    left: TaskScope,
    right: TaskScope,
    contracts: Sequence[MergeContract],
) -> tuple[
    Mapping[str, tuple[str, ...]],
    Mapping[str, tuple[str, ...]],
    tuple[str, ...],
    tuple[str, ...],
    bool,
]:
    overlaps: dict[str, tuple[str, ...]] = {}
    for domain in SCOPE_DOMAINS:
        shared = tuple(
            sorted(set(getattr(left, domain)).intersection(getattr(right, domain)))
        )
        if shared:
            overlaps[domain] = shared

    admitted: dict[str, list[str]] = {domain: [] for domain in SCOPE_DOMAINS}
    admitted_by: list[str] = []
    remaining: dict[str, tuple[str, ...]] = {}
    for domain, items in overlaps.items():
        leftover = set(items)
        for contract in contracts:
            covered = leftover.intersection(getattr(contract, domain))
            if not covered:
                continue
            leftover.difference_update(covered)
            for item in sorted(covered):
                if item not in admitted[domain]:
                    admitted[domain].append(item)
            if contract.name not in admitted_by:
                admitted_by.append(contract.name)
        if leftover:
            remaining[domain] = tuple(sorted(leftover))

    reasons: list[str] = []
    unknown = left.unknown or right.unknown
    if unknown:
        reasons.append("unknown scope")
    for domain, items in remaining.items():
        reasons.append(f"overlapping {domain}: {', '.join(items)}")

    admitted_map = {
        domain: tuple(items) for domain, items in admitted.items() if items
    }
    conflicts = unknown or bool(remaining)
    return (
        MappingProxyType(overlaps),
        MappingProxyType(admitted_map),
        tuple(reasons),
        tuple(admitted_by),
        conflicts,
    )


@dataclass(frozen=True)
class ConflictGraph:
    """Pairwise conservative conflict set over two task write/effect scopes."""

    left: TaskScope | Mapping[str, Any]
    right: TaskScope | Mapping[str, Any]
    merge_contracts: tuple[MergeContract | Mapping[str, Any], ...] = ()
    overlaps: Mapping[str, tuple[str, ...]] = field(init=False)
    admitted: Mapping[str, tuple[str, ...]] = field(init=False)
    reasons: tuple[str, ...] = field(init=False)
    admitted_by: tuple[str, ...] = field(init=False)
    conflicts: bool = field(init=False)

    def __post_init__(self) -> None:
        left = TaskScope.from_mapping(self.left)
        right = TaskScope.from_mapping(self.right)
        if left.task_id == right.task_id:
            raise ConflictGraphError("conflict comparison requires two distinct tasks")
        contracts = _normalize_contracts(self.merge_contracts)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "merge_contracts", contracts)
        overlaps, admitted, reasons, admitted_by, conflicts = _evaluate(
            left, right, contracts
        )
        object.__setattr__(self, "overlaps", overlaps)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "admitted_by", admitted_by)
        object.__setattr__(self, "conflicts", conflicts)

    @property
    def must_serialize(self) -> bool:
        return self.conflicts

    def to_dict(self) -> Mapping[str, Any]:
        left = self.left
        right = self.right
        assert isinstance(left, TaskScope)
        assert isinstance(right, TaskScope)
        return MappingProxyType(
            {
                "schema": CONFLICT_GRAPH_SCHEMA,
                "interface": CONFLICT_GRAPH_INTERFACE,
                "left": dict(left.to_dict()),
                "right": dict(right.to_dict()),
                "merge_contracts": [
                    dict(contract.to_dict()) for contract in self.merge_contracts
                ],
                "overlaps": {domain: list(items) for domain, items in self.overlaps.items()},
                "admitted": {domain: list(items) for domain, items in self.admitted.items()},
                "reasons": list(self.reasons),
                "admitted_by": list(self.admitted_by),
                "conflicts": self.conflicts,
                "must_serialize": self.must_serialize,
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))

    @classmethod
    def derive(
        cls,
        left: TaskScope | Mapping[str, Any],
        right: TaskScope | Mapping[str, Any],
        merge_contracts: Sequence[MergeContract | Mapping[str, Any]] = (),
    ) -> "ConflictGraph":
        return cls(left=left, right=right, merge_contracts=tuple(merge_contracts))
