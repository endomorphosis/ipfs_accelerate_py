"""Semantic affected-check and test selection for incremental verification.

``select_affected_verification`` / :class:`AffectedCheckSelector` walk supplied
semantic, test, fixture, config, and proof dependency edges from changed
symbols and paths.  Selection is pure, deterministic, and fail-closed:

* Direct and transitive exact edges expand the dependency cone and select
  affected tests, static/type checks, and proof obligations.
* Unrelated edits never invent edges; exact selection stays empty when no
  supplied edge or catalog mapping intersects the change set.
* Unknown, dynamic, opaque, uncovered, truncated, or conflicting *critical*
  edges force broader and/or full-suite fallback with stable reason codes.
* Changed obligation dependencies select the corresponding proofs.
* Output order and reason chains are lexicographically stable.

This module never executes tests, mutates caches, or invents missing edges as
exact coverage.  Importing it performs no I/O.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .datasets_adapter import (
    DependencyEdgeView,
    EdgeDisposition,
    InvalidationPlanView,
    SemanticCapsuleView,
    ValidationSelectionView,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

AFFECTED_VERIFICATION_SELECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/affected-verification-selection@1"
)
AFFECTED_VERIFICATION_SELECTION_INTERFACE: Final[str] = (
    "AffectedVerificationSelection@1"
)
SELECTION_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-selection-policy@1"
)
VERIFICATION_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-catalog@1"
)
SELECTION_EVIDENCE: Final[str] = "ivp/test-selection@1"
AFFECTED_CHECK_SELECTOR_INTERFACE: Final[str] = "AffectedCheckSelector@1"

MAX_EDGES: Final[int] = 50_000
MAX_CATALOG_ITEMS: Final[int] = 50_000
MAX_REASON_CHAIN: Final[int] = 64
MAX_TEXT_BYTES: Final[int] = 4_096
DEFAULT_MAX_REASON_CHAIN: Final[int] = 16

# Edge kinds that reverse-walk from a changed provider to dependents.
_REVERSE_IMPACT_KINDS: Final[frozenset[str]] = frozenset(
    {
        "depends_on",
        "imports",
        "calls",
    }
)
# Edge kinds that select a verification target from a symbol/path in the cone.
_TEST_EDGE_KINDS: Final[frozenset[str]] = frozenset({"tested_by"})
_PROOF_EDGE_KINDS: Final[frozenset[str]] = frozenset({"proved_by"})
_FIXTURE_EDGE_KINDS: Final[frozenset[str]] = frozenset({"fixtures"})
_CONFIG_EDGE_KINDS: Final[frozenset[str]] = frozenset({"configures"})
_UNCERTAIN_KINDS: Final[frozenset[str]] = frozenset(
    {"opaque", "dynamic", "unknown"}
)
_UNCERTAIN_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        EdgeDisposition.OPAQUE.value,
        EdgeDisposition.UNCOVERED.value,
        EdgeDisposition.TRUNCATED.value,
        EdgeDisposition.MISSING.value,
        EdgeDisposition.CONSERVATIVE.value,
    }
)

# Reason codes (stable tokens).
REASON_DIRECT_SYMBOL: Final[str] = "direct_symbol_change"
REASON_DIRECT_PATH: Final[str] = "direct_path_change"
REASON_TRANSITIVE_DEPENDENCY: Final[str] = "transitive_dependency"
REASON_TESTED_BY: Final[str] = "tested_by_edge"
REASON_PROVED_BY: Final[str] = "proved_by_edge"
REASON_FIXTURE_EDGE: Final[str] = "fixture_edge"
REASON_CONFIG_EDGE: Final[str] = "config_edge"
REASON_VALIDATION_MAPPING: Final[str] = "validation_id_mapping"
REASON_STATIC_TARGET: Final[str] = "static_check_target"
REASON_TYPE_TARGET: Final[str] = "type_check_target"
REASON_PROOF_DEPENDENCY: Final[str] = "proof_obligation_dependency"
REASON_CATALOG_FALLBACK: Final[str] = "catalog_fallback"
REASON_OPAQUE_CRITICAL: Final[str] = "opaque_critical_edge"
REASON_DYNAMIC_CRITICAL: Final[str] = "dynamic_critical_edge"
REASON_UNKNOWN_CRITICAL: Final[str] = "unknown_critical_edge"
REASON_UNCOVERED_IMPACT: Final[str] = "uncovered_impact"
REASON_TRUNCATED_FRONTIER: Final[str] = "truncated_frontier"
REASON_CONFLICTING_CRITICAL: Final[str] = "conflicting_critical_edges"
REASON_MISSING_CRITICAL: Final[str] = "missing_critical_edge"
REASON_CONSERVATIVE_CLOSURE: Final[str] = "conservative_closure"
REASON_BROADER_REQUIRED: Final[str] = "broader_selection_required"
REASON_FULL_SUITE_POLICY: Final[str] = "policy_requires_full_suite"
REASON_FULL_SUITE_UNCERTAINTY: Final[str] = "uncertainty_requires_full_suite"
REASON_VALIDATION_MAPPING_INCOMPLETE: Final[str] = (
    "validation_id_mapping_incomplete"
)
REASON_INPUT_REQUIRES_BROADER: Final[str] = "input_requires_broader_selection"
REASON_UNRELATED_NO_EXPANSION: Final[str] = "unrelated_edit_no_expansion"


class SelectionError(ValueError):
    """Malformed selection input or policy contract violation."""


class SelectionBoundsError(SelectionError):
    """A selection input exceeded deterministic compactness bounds."""


class FallbackMode(str, Enum):
    """How far selection expands under uncertainty."""

    EXACT = "exact"
    BROADER = "broader"
    FULL_SUITE = "full_suite"


class SelectionDisposition(str, Enum):
    """Per-item selection disposition (exact vs fallback-driven)."""

    EXACT = "exact"
    BROADER = "broader"
    FULL_SUITE = "full_suite"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        if required:
            raise SelectionError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise SelectionError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise SelectionError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise SelectionError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise SelectionBoundsError(
            f"{field_name} exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        )
    return text


def _optional_text(value: Any, *, field_name: str) -> str:
    if value is None or value == "":
        return ""
    return _text(value, field_name=field_name, required=True)


def _boolean(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raise SelectionError(f"{field_name} must be a boolean")


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int = 10_000_000,
    default: int | None = None,
) -> int:
    if value is None:
        if default is not None:
            return default
        raise SelectionError(f"{field_name} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise SelectionBoundsError(
            f"{field_name} must be in [{minimum}, {maximum}]"
        )
    return value


def _string_tuple(
    value: Any,
    *,
    field_name: str,
    sort: bool = True,
    maximum: int = MAX_CATALOG_ITEMS,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        items = value
    else:
        raise SelectionError(f"{field_name} must be a sequence of strings")
    if len(items) > maximum:
        raise SelectionBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _text(item, field_name=f"{field_name}[{index}]")
        if text not in seen:
            seen.add(text)
            result.append(text)
    if sort:
        result.sort()
    return tuple(result)


def _string_to_string_tuple_map(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, tuple[str, ...]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise SelectionError(f"{field_name} must be a mapping")
    if len(value) > MAX_CATALOG_ITEMS:
        raise SelectionBoundsError(
            f"{field_name} exceeds {MAX_CATALOG_ITEMS} items"
        )
    result: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_targets in value.items():
        key = _text(raw_key, field_name=f"{field_name}.key")
        result[key] = _string_tuple(
            raw_targets, field_name=f"{field_name}[{key}]"
        )
    return MappingProxyType(dict(sorted(result.items())))


def _unique_sorted(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(item) for item in items if str(item)}))


def _stable_unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)


def _truncate_chain(
    chain: Sequence[str], *, maximum: int
) -> tuple[str, ...]:
    if maximum < 1:
        return ()
    if len(chain) <= maximum:
        return tuple(chain)
    # Keep root and the final hops when truncating for stability.
    if maximum == 1:
        return (chain[-1],)
    head = chain[0]
    tail = tuple(chain[-(maximum - 1) :])
    return (head, *tail)


def _prefer_chain(
    existing: Sequence[str] | None, candidate: Sequence[str]
) -> bool:
    """Return True when *candidate* should replace *existing*."""

    if existing is None:
        return True
    existing_t = tuple(existing)
    candidate_t = tuple(candidate)
    return (len(candidate_t), candidate_t) < (len(existing_t), existing_t)


# ---------------------------------------------------------------------------
# Edge normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionEdge:
    """Normalized selection edge (non-authoritative)."""

    edge_id: str
    source: str
    target: str
    kind: str
    disposition: str
    truncated: bool = False
    opaque: bool = False
    uncovered: bool = False
    critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "disposition": self.disposition,
            "truncated": self.truncated,
            "opaque": self.opaque,
            "uncovered": self.uncovered,
            "critical": self.critical,
        }


def _disposition_token(value: Any) -> str:
    if isinstance(value, EdgeDisposition):
        return value.value
    if isinstance(value, Enum):
        return str(value.value)
    text = str(value or EdgeDisposition.EXACT.value).strip().lower()
    return text or EdgeDisposition.EXACT.value


def _normalize_edge(value: Any, *, index: int) -> SelectionEdge:
    if isinstance(value, SelectionEdge):
        return value
    if isinstance(value, DependencyEdgeView):
        critical = True
        if isinstance(value.record, Mapping):
            critical = _boolean(
                value.record.get("critical", True),
                field_name=f"edges[{index}].critical",
            )
        return SelectionEdge(
            edge_id=value.edge_id,
            source=value.source,
            target=value.target,
            kind=str(value.kind).strip().lower(),
            disposition=_disposition_token(value.disposition),
            truncated=bool(value.truncated),
            opaque=bool(value.opaque),
            uncovered=bool(value.uncovered),
            critical=critical,
        )
    if not isinstance(value, Mapping):
        raise SelectionError(f"edges[{index}] must be a mapping or edge view")
    source = _text(
        value.get("source") or value.get("source_id"),
        field_name=f"edges[{index}].source",
    )
    target = _text(
        value.get("target") or value.get("target_id"),
        field_name=f"edges[{index}].target",
    )
    kind = _optional_text(
        value.get("kind") or value.get("edge_kind"),
        field_name=f"edges[{index}].kind",
    )
    if not kind:
        kind = "unknown"
    kind = kind.strip().lower()
    truncated = _boolean(
        value.get("truncated"), field_name=f"edges[{index}].truncated"
    )
    uncovered = _boolean(
        value.get("uncovered"), field_name=f"edges[{index}].uncovered"
    )
    opaque_flag = _boolean(
        value.get("opaque"), field_name=f"edges[{index}].opaque"
    )
    opaque = opaque_flag or kind in _UNCERTAIN_KINDS
    if truncated:
        disposition = EdgeDisposition.TRUNCATED.value
    elif uncovered:
        disposition = EdgeDisposition.UNCOVERED.value
    elif opaque:
        disposition = EdgeDisposition.OPAQUE.value
    else:
        disposition = _disposition_token(
            value.get("disposition") or EdgeDisposition.EXACT.value
        )
    edge_id = _optional_text(
        value.get("edge_id"), field_name=f"edges[{index}].edge_id"
    )
    if not edge_id:
        edge_id = f"edge:{kind}:{source}->{target}"
    critical = _boolean(
        value.get("critical", True), field_name=f"edges[{index}].critical"
    )
    return SelectionEdge(
        edge_id=edge_id,
        source=source,
        target=target,
        kind=kind,
        disposition=disposition,
        truncated=truncated,
        opaque=opaque,
        uncovered=uncovered,
        critical=critical,
    )


def _normalize_edges(value: Any) -> tuple[SelectionEdge, ...]:
    if value is None:
        return ()
    if isinstance(value, (SelectionEdge, DependencyEdgeView, Mapping)):
        items: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        items = value
    else:
        raise SelectionError("edges must be a sequence of edge mappings")
    if len(items) > MAX_EDGES:
        raise SelectionBoundsError(f"edges exceeds {MAX_EDGES} items")
    edges = [_normalize_edge(item, index=index) for index, item in enumerate(items)]
    edges.sort(key=lambda e: (e.kind, e.source, e.target, e.edge_id))
    return tuple(edges)


# ---------------------------------------------------------------------------
# Policy and catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Fail-closed policy knobs for broader/full-suite escalation."""

    schema: str = SELECTION_POLICY_SCHEMA
    force_full_suite: bool = False
    # When True, any broader-selection uncertainty escalates to full suite.
    broader_escalates_to_full_suite: bool = False
    # Critical uncertain edge kinds always escalate to full suite when True.
    critical_uncertainty_requires_full_suite: bool = True
    # Non-critical uncertain edges only require broader (not full) when False.
    non_critical_uncertainty_requires_broader: bool = True
    max_reason_chain: int = DEFAULT_MAX_REASON_CHAIN
    # Optional path-prefix broader expansion when broader but not full suite.
    broader_includes_sibling_tests: bool = True
    policy_id: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "force_full_suite",
            _boolean(self.force_full_suite, field_name="force_full_suite"),
        )
        object.__setattr__(
            self,
            "broader_escalates_to_full_suite",
            _boolean(
                self.broader_escalates_to_full_suite,
                field_name="broader_escalates_to_full_suite",
            ),
        )
        object.__setattr__(
            self,
            "critical_uncertainty_requires_full_suite",
            _boolean(
                self.critical_uncertainty_requires_full_suite,
                field_name="critical_uncertainty_requires_full_suite",
            ),
        )
        object.__setattr__(
            self,
            "non_critical_uncertainty_requires_broader",
            _boolean(
                self.non_critical_uncertainty_requires_broader,
                field_name="non_critical_uncertainty_requires_broader",
            ),
        )
        object.__setattr__(
            self,
            "broader_includes_sibling_tests",
            _boolean(
                self.broader_includes_sibling_tests,
                field_name="broader_includes_sibling_tests",
            ),
        )
        object.__setattr__(
            self,
            "max_reason_chain",
            _integer(
                self.max_reason_chain,
                field_name="max_reason_chain",
                minimum=1,
                maximum=MAX_REASON_CHAIN,
                default=DEFAULT_MAX_REASON_CHAIN,
            ),
        )
        object.__setattr__(
            self,
            "policy_id",
            _text(self.policy_id, field_name="policy_id"),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="schema"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "force_full_suite": self.force_full_suite,
            "broader_escalates_to_full_suite": (
                self.broader_escalates_to_full_suite
            ),
            "critical_uncertainty_requires_full_suite": (
                self.critical_uncertainty_requires_full_suite
            ),
            "non_critical_uncertainty_requires_broader": (
                self.non_critical_uncertainty_requires_broader
            ),
            "max_reason_chain": self.max_reason_chain,
            "broader_includes_sibling_tests": (
                self.broader_includes_sibling_tests
            ),
            "policy_id": self.policy_id,
        }

    @classmethod
    def from_value(cls, value: Any | None) -> SelectionPolicy:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise SelectionError("policy must be a SelectionPolicy or mapping")
        return cls(
            schema=str(value.get("schema") or SELECTION_POLICY_SCHEMA),
            force_full_suite=bool(value.get("force_full_suite", False)),
            broader_escalates_to_full_suite=bool(
                value.get("broader_escalates_to_full_suite", False)
            ),
            critical_uncertainty_requires_full_suite=bool(
                value.get("critical_uncertainty_requires_full_suite", True)
            ),
            non_critical_uncertainty_requires_broader=bool(
                value.get("non_critical_uncertainty_requires_broader", True)
            ),
            max_reason_chain=int(
                value.get("max_reason_chain", DEFAULT_MAX_REASON_CHAIN)
            ),
            broader_includes_sibling_tests=bool(
                value.get("broader_includes_sibling_tests", True)
            ),
            policy_id=str(value.get("policy_id") or "default"),
        )


@dataclass(frozen=True, slots=True)
class VerificationCatalog:
    """Known verification inventory used for exact and fallback selection."""

    schema: str = VERIFICATION_CATALOG_SCHEMA
    tests: tuple[str, ...] = ()
    static_checks: tuple[str, ...] = ()
    type_checks: tuple[str, ...] = ()
    proof_obligations: tuple[str, ...] = ()
    # check_id / obligation_id -> symbols and/or paths it covers.
    static_check_targets: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    type_check_targets: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    proof_obligation_dependencies: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tests", _string_tuple(self.tests, field_name="tests")
        )
        object.__setattr__(
            self,
            "static_checks",
            _string_tuple(self.static_checks, field_name="static_checks"),
        )
        object.__setattr__(
            self,
            "type_checks",
            _string_tuple(self.type_checks, field_name="type_checks"),
        )
        object.__setattr__(
            self,
            "proof_obligations",
            _string_tuple(
                self.proof_obligations, field_name="proof_obligations"
            ),
        )
        object.__setattr__(
            self,
            "static_check_targets",
            _string_to_string_tuple_map(
                self.static_check_targets, field_name="static_check_targets"
            ),
        )
        object.__setattr__(
            self,
            "type_check_targets",
            _string_to_string_tuple_map(
                self.type_check_targets, field_name="type_check_targets"
            ),
        )
        object.__setattr__(
            self,
            "proof_obligation_dependencies",
            _string_to_string_tuple_map(
                self.proof_obligation_dependencies,
                field_name="proof_obligation_dependencies",
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema, field_name="schema"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tests": list(self.tests),
            "static_checks": list(self.static_checks),
            "type_checks": list(self.type_checks),
            "proof_obligations": list(self.proof_obligations),
            "static_check_targets": {
                key: list(value)
                for key, value in self.static_check_targets.items()
            },
            "type_check_targets": {
                key: list(value)
                for key, value in self.type_check_targets.items()
            },
            "proof_obligation_dependencies": {
                key: list(value)
                for key, value in self.proof_obligation_dependencies.items()
            },
        }

    @classmethod
    def from_value(cls, value: Any | None) -> VerificationCatalog:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise SelectionError(
                "catalog must be a VerificationCatalog or mapping"
            )
        return cls(
            schema=str(value.get("schema") or VERIFICATION_CATALOG_SCHEMA),
            tests=tuple(value.get("tests") or ()),
            static_checks=tuple(value.get("static_checks") or ()),
            type_checks=tuple(value.get("type_checks") or ()),
            proof_obligations=tuple(value.get("proof_obligations") or ()),
            static_check_targets=value.get("static_check_targets") or {},
            type_check_targets=value.get("type_check_targets") or {},
            proof_obligation_dependencies=(
                value.get("proof_obligation_dependencies") or {}
            ),
        )


# ---------------------------------------------------------------------------
# Selection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AffectedVerificationSelection:
    """Deterministic selection of affected checks and fallback posture."""

    SCHEMA: Final[str] = AFFECTED_VERIFICATION_SELECTION_SCHEMA
    INTERFACE: Final[str] = AFFECTED_VERIFICATION_SELECTION_INTERFACE

    schema: str = AFFECTED_VERIFICATION_SELECTION_SCHEMA
    interface: str = AFFECTED_VERIFICATION_SELECTION_INTERFACE
    evidence: str = SELECTION_EVIDENCE
    affected_tests: tuple[str, ...] = ()
    fallback_tests: tuple[str, ...] = ()
    required_static_checks: tuple[str, ...] = ()
    required_type_checks: tuple[str, ...] = ()
    affected_proof_obligation_cids: tuple[str, ...] = ()
    dependency_cone_symbols: tuple[str, ...] = ()
    dependency_cone_paths: tuple[str, ...] = ()
    reason_chains: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    fallback_mode: FallbackMode = FallbackMode.EXACT
    broader_selection_required: bool = False
    full_suite_required: bool = False
    fallback_reason_codes: tuple[str, ...] = ()
    full_suite_reason_codes: tuple[str, ...] = ()
    selection_reason_codes: tuple[str, ...] = ()
    critical_uncertain_edges: tuple[str, ...] = ()
    conflicting_edge_ids: tuple[str, ...] = ()
    policy_id: str = "default"
    authoritative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(
            self, "affected_tests", _unique_sorted(self.affected_tests)
        )
        object.__setattr__(
            self, "fallback_tests", _unique_sorted(self.fallback_tests)
        )
        object.__setattr__(
            self,
            "required_static_checks",
            _unique_sorted(self.required_static_checks),
        )
        object.__setattr__(
            self,
            "required_type_checks",
            _unique_sorted(self.required_type_checks),
        )
        object.__setattr__(
            self,
            "affected_proof_obligation_cids",
            _unique_sorted(self.affected_proof_obligation_cids),
        )
        object.__setattr__(
            self,
            "dependency_cone_symbols",
            _unique_sorted(self.dependency_cone_symbols),
        )
        object.__setattr__(
            self,
            "dependency_cone_paths",
            _unique_sorted(self.dependency_cone_paths),
        )
        chains: dict[str, tuple[str, ...]] = {}
        raw_chains = self.reason_chains or {}
        if not isinstance(raw_chains, Mapping):
            raise SelectionError("reason_chains must be a mapping")
        for key, chain in raw_chains.items():
            key_text = _text(key, field_name="reason_chains.key")
            if isinstance(chain, str):
                chain_items = (chain,)
            elif isinstance(chain, Sequence) and not isinstance(
                chain, (bytes, bytearray)
            ):
                chain_items = tuple(
                    _text(item, field_name=f"reason_chains[{key_text}]")
                    for item in chain
                )
            else:
                raise SelectionError(
                    f"reason_chains[{key_text}] must be a sequence of strings"
                )
            chains[key_text] = chain_items
        object.__setattr__(
            self, "reason_chains", MappingProxyType(dict(sorted(chains.items())))
        )
        mode = self.fallback_mode
        if isinstance(mode, str):
            mode = FallbackMode(mode)
        if not isinstance(mode, FallbackMode):
            raise SelectionError("fallback_mode must be a FallbackMode")
        object.__setattr__(self, "fallback_mode", mode)
        object.__setattr__(
            self,
            "broader_selection_required",
            _boolean(
                self.broader_selection_required,
                field_name="broader_selection_required",
            ),
        )
        object.__setattr__(
            self,
            "full_suite_required",
            _boolean(self.full_suite_required, field_name="full_suite_required"),
        )
        object.__setattr__(
            self,
            "fallback_reason_codes",
            _stable_unique(self.fallback_reason_codes),
        )
        object.__setattr__(
            self,
            "full_suite_reason_codes",
            _stable_unique(self.full_suite_reason_codes),
        )
        object.__setattr__(
            self,
            "selection_reason_codes",
            _stable_unique(self.selection_reason_codes),
        )
        object.__setattr__(
            self,
            "critical_uncertain_edges",
            _unique_sorted(self.critical_uncertain_edges),
        )
        object.__setattr__(
            self,
            "conflicting_edge_ids",
            _unique_sorted(self.conflicting_edge_ids),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id")
        )
        # Consistency: full suite implies broader.
        if self.full_suite_required and not self.broader_selection_required:
            object.__setattr__(self, "broader_selection_required", True)
        if self.full_suite_required and self.fallback_mode is FallbackMode.EXACT:
            object.__setattr__(self, "fallback_mode", FallbackMode.FULL_SUITE)
        if (
            self.broader_selection_required
            and not self.full_suite_required
            and self.fallback_mode is FallbackMode.EXACT
        ):
            object.__setattr__(self, "fallback_mode", FallbackMode.BROADER)

    @property
    def selected_tests(self) -> tuple[str, ...]:
        """Union of exact affected and fallback tests (deterministic)."""

        return _unique_sorted((*self.affected_tests, *self.fallback_tests))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence": self.evidence,
            "affected_tests": list(self.affected_tests),
            "fallback_tests": list(self.fallback_tests),
            "selected_tests": list(self.selected_tests),
            "required_static_checks": list(self.required_static_checks),
            "required_type_checks": list(self.required_type_checks),
            "affected_proof_obligation_cids": list(
                self.affected_proof_obligation_cids
            ),
            "dependency_cone_symbols": list(self.dependency_cone_symbols),
            "dependency_cone_paths": list(self.dependency_cone_paths),
            "reason_chains": {
                key: list(chain) for key, chain in self.reason_chains.items()
            },
            "fallback_mode": self.fallback_mode.value,
            "broader_selection_required": self.broader_selection_required,
            "full_suite_required": self.full_suite_required,
            "fallback_reason_codes": list(self.fallback_reason_codes),
            "full_suite_reason_codes": list(self.full_suite_reason_codes),
            "selection_reason_codes": list(self.selection_reason_codes),
            "critical_uncertain_edges": list(self.critical_uncertain_edges),
            "conflicting_edge_ids": list(self.conflicting_edge_ids),
            "policy_id": self.policy_id,
            "authoritative": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AffectedVerificationSelection:
        if not isinstance(payload, Mapping):
            raise SelectionError("selection payload must be a mapping")
        mode = payload.get("fallback_mode", FallbackMode.EXACT.value)
        return cls(
            schema=str(
                payload.get("schema") or AFFECTED_VERIFICATION_SELECTION_SCHEMA
            ),
            interface=str(
                payload.get("interface")
                or AFFECTED_VERIFICATION_SELECTION_INTERFACE
            ),
            evidence=str(payload.get("evidence") or SELECTION_EVIDENCE),
            affected_tests=tuple(payload.get("affected_tests") or ()),
            fallback_tests=tuple(payload.get("fallback_tests") or ()),
            required_static_checks=tuple(
                payload.get("required_static_checks") or ()
            ),
            required_type_checks=tuple(
                payload.get("required_type_checks") or ()
            ),
            affected_proof_obligation_cids=tuple(
                payload.get("affected_proof_obligation_cids") or ()
            ),
            dependency_cone_symbols=tuple(
                payload.get("dependency_cone_symbols") or ()
            ),
            dependency_cone_paths=tuple(
                payload.get("dependency_cone_paths") or ()
            ),
            reason_chains=payload.get("reason_chains") or {},
            fallback_mode=FallbackMode(mode),
            broader_selection_required=bool(
                payload.get("broader_selection_required", False)
            ),
            full_suite_required=bool(payload.get("full_suite_required", False)),
            fallback_reason_codes=tuple(
                payload.get("fallback_reason_codes") or ()
            ),
            full_suite_reason_codes=tuple(
                payload.get("full_suite_reason_codes") or ()
            ),
            selection_reason_codes=tuple(
                payload.get("selection_reason_codes") or ()
            ),
            critical_uncertain_edges=tuple(
                payload.get("critical_uncertain_edges") or ()
            ),
            conflicting_edge_ids=tuple(
                payload.get("conflicting_edge_ids") or ()
            ),
            policy_id=str(payload.get("policy_id") or "default"),
        )


# ---------------------------------------------------------------------------
# Input view extraction
# ---------------------------------------------------------------------------


def _edges_from_view(view: Any) -> tuple[SelectionEdge, ...]:
    if view is None:
        return ()
    edges = getattr(view, "edges", None)
    if edges is None and isinstance(view, Mapping):
        edges = view.get("edges")
    return _normalize_edges(edges)


def _changed_from_invalidation(
    plan: InvalidationPlanView | Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], bool, bool, tuple[str, ...], tuple[str, ...]]:
    if plan is None:
        return (), (), False, False, (), ()
    if isinstance(plan, InvalidationPlanView):
        return (
            tuple(plan.changed_symbols),
            tuple(plan.changed_paths),
            bool(plan.truncated),
            bool(plan.requires_broader_selection),
            tuple(plan.uncovered_symbols),
            tuple(plan.uncovered_paths),
        )
    if not isinstance(plan, Mapping):
        raise SelectionError(
            "invalidation_plan must be an InvalidationPlanView or mapping"
        )
    return (
        _string_tuple(plan.get("changed_symbols"), field_name="changed_symbols"),
        _string_tuple(plan.get("changed_paths"), field_name="changed_paths"),
        _boolean(plan.get("truncated"), field_name="truncated"),
        _boolean(
            plan.get("requires_broader_selection"),
            field_name="requires_broader_selection",
        ),
        _string_tuple(
            plan.get("uncovered_symbols"), field_name="uncovered_symbols"
        ),
        _string_tuple(plan.get("uncovered_paths"), field_name="uncovered_paths"),
    )


def _validation_nodes(
    value: ValidationSelectionView | Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], bool, tuple[str, ...]]:
    if value is None:
        return (), False, ()
    if isinstance(value, ValidationSelectionView):
        return (
            tuple(value.mapped_pytest_node_ids),
            bool(value.requires_broader_selection),
            tuple(value.unmapped_validation_ids),
        )
    if not isinstance(value, Mapping):
        raise SelectionError(
            "validation_selection must be a ValidationSelectionView or mapping"
        )
    return (
        _string_tuple(
            value.get("mapped_pytest_node_ids"),
            field_name="mapped_pytest_node_ids",
        ),
        _boolean(
            value.get("requires_broader_selection"),
            field_name="requires_broader_selection",
        ),
        _string_tuple(
            value.get("unmapped_validation_ids"),
            field_name="unmapped_validation_ids",
        ),
    )


# ---------------------------------------------------------------------------
# Core selection engine
# ---------------------------------------------------------------------------


def _looks_like_path(token: str) -> bool:
    return "/" in token or token.endswith((".py", ".toml"))


def _looks_like_test(token: str) -> bool:
    return "::" in token or token.startswith("test/") or "/test_" in token


def _test_file_prefix(node_id: str) -> str:
    if "::" in node_id:
        return node_id.split("::", 1)[0]
    return node_id


def _sibling_tests(
    exact_tests: Iterable[str], catalog_tests: Sequence[str]
) -> tuple[str, ...]:
    prefixes = {_test_file_prefix(test) for test in exact_tests}
    if not prefixes:
        return ()
    selected: list[str] = []
    for test in catalog_tests:
        if _test_file_prefix(test) in prefixes:
            selected.append(test)
    return _unique_sorted(selected)


def _edge_is_uncertain(edge: SelectionEdge) -> bool:
    if edge.truncated or edge.uncovered or edge.opaque:
        return True
    if edge.kind in _UNCERTAIN_KINDS:
        return True
    return edge.disposition in _UNCERTAIN_DISPOSITIONS


def _uncertainty_reason(edge: SelectionEdge) -> str:
    if edge.truncated or edge.disposition == EdgeDisposition.TRUNCATED.value:
        return REASON_TRUNCATED_FRONTIER
    if edge.uncovered or edge.disposition == EdgeDisposition.UNCOVERED.value:
        return REASON_UNCOVERED_IMPACT
    if edge.kind == "dynamic":
        return REASON_DYNAMIC_CRITICAL
    if edge.kind == "unknown" or edge.disposition == EdgeDisposition.MISSING.value:
        if edge.disposition == EdgeDisposition.MISSING.value:
            return REASON_MISSING_CRITICAL
        return REASON_UNKNOWN_CRITICAL
    if edge.kind == "opaque" or edge.opaque:
        return REASON_OPAQUE_CRITICAL
    if edge.disposition == EdgeDisposition.CONSERVATIVE.value:
        return REASON_CONSERVATIVE_CLOSURE
    if edge.disposition == EdgeDisposition.OPAQUE.value:
        return REASON_OPAQUE_CRITICAL
    return REASON_OPAQUE_CRITICAL


def _detect_conflicts(edges: Sequence[SelectionEdge]) -> tuple[str, ...]:
    """Conflicting critical edges share source+kind with mixed dispositions."""

    groups: dict[tuple[str, str], list[SelectionEdge]] = {}
    for edge in edges:
        if not edge.critical:
            continue
        groups.setdefault((edge.source, edge.kind), []).append(edge)
    conflicting: list[str] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        dispositions = {edge.disposition for edge in group}
        uncertain = any(_edge_is_uncertain(edge) for edge in group)
        exact = any(
            not _edge_is_uncertain(edge)
            and edge.disposition == EdgeDisposition.EXACT.value
            for edge in group
        )
        # Conflict when exact and uncertain evidence compete, or dispositions
        # disagree across more than one critical edge.
        if (exact and uncertain) or (len(dispositions) > 1 and uncertain):
            for edge in group:
                conflicting.append(edge.edge_id)
    return _unique_sorted(conflicting)


def _build_reverse_adjacency(
    edges: Sequence[SelectionEdge],
) -> dict[str, list[tuple[str, SelectionEdge]]]:
    """provider -> list of (dependent, edge) for reverse-impact kinds."""

    reverse: dict[str, list[tuple[str, SelectionEdge]]] = {}
    for edge in edges:
        if edge.kind not in _REVERSE_IMPACT_KINDS:
            continue
        # source depends on target => changing target impacts source.
        reverse.setdefault(edge.target, []).append((edge.source, edge))
    for value in reverse.values():
        value.sort(key=lambda item: (item[0], item[1].edge_id))
    return reverse


def _collect_selector_edges(
    edges: Sequence[SelectionEdge],
    kinds: frozenset[str],
) -> dict[str, list[tuple[str, SelectionEdge]]]:
    """source -> list of (target, edge) for selection edge kinds."""

    mapping: dict[str, list[tuple[str, SelectionEdge]]] = {}
    for edge in edges:
        if edge.kind not in kinds:
            continue
        mapping.setdefault(edge.source, []).append((edge.target, edge))
    for value in mapping.values():
        value.sort(key=lambda item: (item[0], item[1].edge_id))
    return mapping


def compute_dependency_cone(
    *,
    changed_symbols: Sequence[str],
    changed_paths: Sequence[str],
    edges: Sequence[SelectionEdge],
    max_reason_chain: int = DEFAULT_MAX_REASON_CHAIN,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    Mapping[str, tuple[str, ...]],
]:
    """Reverse transitive closure over depends_on/imports/calls edges."""

    roots = _unique_sorted((*changed_symbols, *changed_paths))
    reverse = _build_reverse_adjacency(edges)
    chains: dict[str, tuple[str, ...]] = {
        root: (root,) for root in roots
    }
    queue: deque[str] = deque(roots)
    while queue:
        current = queue.popleft()
        for dependent, edge in reverse.get(current, ()):
            # Uncertain reverse edges still expand the cone conservatively;
            # callers classify fallback from the same edges.
            candidate = (*chains[current], dependent)
            if _prefer_chain(chains.get(dependent), candidate):
                chains[dependent] = candidate
                queue.append(dependent)
    symbols: list[str] = []
    paths: list[str] = []
    truncated_chains: dict[str, tuple[str, ...]] = {}
    for node, chain in sorted(chains.items()):
        truncated_chains[node] = _truncate_chain(
            chain, maximum=max_reason_chain
        )
        if _looks_like_path(node) and not _looks_like_test(node):
            paths.append(node)
        else:
            # Symbols and ambiguous tokens land in the symbol cone.
            if not _looks_like_test(node):
                symbols.append(node)
            if _looks_like_path(node):
                paths.append(node)
    return _unique_sorted(symbols), _unique_sorted(paths), MappingProxyType(
        dict(sorted(truncated_chains.items()))
    )


def _record_selection(
    *,
    selected: dict[str, tuple[str, ...]],
    item: str,
    chain: Sequence[str],
    max_reason_chain: int,
) -> None:
    candidate = _truncate_chain(chain, maximum=max_reason_chain)
    existing = selected.get(item)
    if _prefer_chain(existing, candidate):
        selected[item] = candidate


def select_affected_verification(
    *,
    changed_symbols: Sequence[str] | None = None,
    changed_paths: Sequence[str] | None = None,
    edges: Sequence[Any] | None = None,
    uncovered_symbols: Sequence[str] | None = None,
    uncovered_paths: Sequence[str] | None = None,
    truncated: bool = False,
    requires_broader_selection: bool = False,
    invalidation_plan: InvalidationPlanView | Mapping[str, Any] | None = None,
    semantic_capsule: SemanticCapsuleView | Mapping[str, Any] | None = None,
    validation_selection: (
        ValidationSelectionView | Mapping[str, Any] | None
    ) = None,
    catalog: VerificationCatalog | Mapping[str, Any] | None = None,
    policy: SelectionPolicy | Mapping[str, Any] | None = None,
    known_tests: Sequence[str] | None = None,
    known_static_checks: Sequence[str] | None = None,
    known_type_checks: Sequence[str] | None = None,
    known_proof_obligations: Sequence[str] | None = None,
    static_check_targets: Mapping[str, Sequence[str]] | None = None,
    type_check_targets: Mapping[str, Sequence[str]] | None = None,
    proof_obligation_dependencies: Mapping[str, Sequence[str]] | None = None,
) -> AffectedVerificationSelection:
    """Select affected tests/static/type checks/proofs from semantic edges.

    Accepts either flat arguments or normalized datasets-adapter views.  Missing
    edges are never inferred as exact; uncertainty always broadens or forces a
    full-suite fallback according to policy.
    """

    policy_obj = SelectionPolicy.from_value(policy)
    catalog_obj = VerificationCatalog.from_value(catalog)

    # Merge catalog overrides from keyword arguments.
    if known_tests is not None:
        catalog_obj = VerificationCatalog(
            **{
                **catalog_obj.to_dict(),
                "tests": tuple(known_tests),
            }
        )
    if known_static_checks is not None:
        catalog_obj = VerificationCatalog(
            **{
                **catalog_obj.to_dict(),
                "static_checks": tuple(known_static_checks),
            }
        )
    if known_type_checks is not None:
        catalog_obj = VerificationCatalog(
            **{
                **catalog_obj.to_dict(),
                "type_checks": tuple(known_type_checks),
            }
        )
    if known_proof_obligations is not None:
        catalog_obj = VerificationCatalog(
            **{
                **catalog_obj.to_dict(),
                "proof_obligations": tuple(known_proof_obligations),
            }
        )
    if static_check_targets is not None:
        catalog_obj = VerificationCatalog(
            **{
                **catalog_obj.to_dict(),
                "static_check_targets": dict(static_check_targets),
            }
        )
    if type_check_targets is not None:
        catalog_obj = VerificationCatalog(
            **{
                **catalog_obj.to_dict(),
                "type_check_targets": dict(type_check_targets),
            }
        )
    if proof_obligation_dependencies is not None:
        catalog_obj = VerificationCatalog(
            **{
                **catalog_obj.to_dict(),
                "proof_obligation_dependencies": dict(
                    proof_obligation_dependencies
                ),
            }
        )

    plan_symbols, plan_paths, plan_truncated, plan_broader, plan_uncov_sym, plan_uncov_path = (
        _changed_from_invalidation(invalidation_plan)
    )

    symbols = _string_tuple(
        changed_symbols if changed_symbols is not None else plan_symbols,
        field_name="changed_symbols",
    )
    paths = _string_tuple(
        changed_paths if changed_paths is not None else plan_paths,
        field_name="changed_paths",
    )
    uncovered_sym = _string_tuple(
        uncovered_symbols
        if uncovered_symbols is not None
        else plan_uncov_sym,
        field_name="uncovered_symbols",
    )
    uncovered_pth = _string_tuple(
        uncovered_paths if uncovered_paths is not None else plan_uncov_path,
        field_name="uncovered_paths",
    )
    truncated_flag = bool(truncated) or bool(plan_truncated)
    broader_flag = bool(requires_broader_selection) or bool(plan_broader)

    edge_list: list[SelectionEdge] = []
    if edges is not None:
        edge_list.extend(_normalize_edges(edges))
    if invalidation_plan is not None:
        edge_list.extend(_edges_from_view(invalidation_plan))
    if semantic_capsule is not None:
        edge_list.extend(_edges_from_view(semantic_capsule))
        if isinstance(semantic_capsule, SemanticCapsuleView):
            if semantic_capsule.truncated:
                truncated_flag = True
            if semantic_capsule.requires_broader_selection:
                broader_flag = True
        elif isinstance(semantic_capsule, Mapping):
            if _boolean(
                semantic_capsule.get("truncated"), field_name="truncated"
            ):
                truncated_flag = True
            if _boolean(
                semantic_capsule.get("requires_broader_selection"),
                field_name="requires_broader_selection",
            ):
                broader_flag = True

    # Deduplicate edges by edge_id (stable first-wins after sort).
    edge_list.sort(key=lambda e: (e.kind, e.source, e.target, e.edge_id))
    deduped: list[SelectionEdge] = []
    seen_ids: set[str] = set()
    for edge in edge_list:
        if edge.edge_id in seen_ids:
            continue
        seen_ids.add(edge.edge_id)
        deduped.append(edge)
    all_edges = tuple(deduped)

    max_chain = policy_obj.max_reason_chain
    cone_symbols, cone_paths, cone_chains = compute_dependency_cone(
        changed_symbols=symbols,
        changed_paths=paths,
        edges=all_edges,
        max_reason_chain=max_chain,
    )
    cone_nodes = set(cone_symbols) | set(cone_paths) | set(symbols) | set(paths)

    # --- Uncertainty classification ---
    fallback_reasons: list[str] = []
    full_suite_reasons: list[str] = []
    selection_reasons: list[str] = []
    critical_uncertain: list[str] = []
    conflicting = list(_detect_conflicts(all_edges))

    if symbols:
        selection_reasons.append(REASON_DIRECT_SYMBOL)
    if paths:
        selection_reasons.append(REASON_DIRECT_PATH)

    if uncovered_sym or uncovered_pth:
        broader_flag = True
        fallback_reasons.append(REASON_UNCOVERED_IMPACT)
    if truncated_flag:
        broader_flag = True
        fallback_reasons.append(REASON_TRUNCATED_FRONTIER)
    if (
        broader_flag
        and REASON_INPUT_REQUIRES_BROADER not in fallback_reasons
        and (requires_broader_selection or plan_broader)
    ):
        fallback_reasons.append(REASON_INPUT_REQUIRES_BROADER)

    for edge in all_edges:
        if not _edge_is_uncertain(edge):
            continue
        reason = _uncertainty_reason(edge)
        # Critical uncertain edges that touch the change frontier or cone.
        touches = (
            edge.source in cone_nodes
            or edge.target in cone_nodes
            or edge.source in symbols
            or edge.source in paths
            or edge.target in symbols
            or edge.target in paths
            or not cone_nodes  # no cone yet but uncertain edges exist globally
        )
        # Always treat critical uncertain edges as selection-relevant when they
        # are reverse-impact, test, proof, fixture, or config edges.
        relevant_kind = edge.kind in (
            _REVERSE_IMPACT_KINDS
            | _TEST_EDGE_KINDS
            | _PROOF_EDGE_KINDS
            | _FIXTURE_EDGE_KINDS
            | _CONFIG_EDGE_KINDS
            | _UNCERTAIN_KINDS
        )
        if edge.critical and (touches or relevant_kind):
            critical_uncertain.append(edge.edge_id)
            broader_flag = True
            fallback_reasons.append(reason)
            if policy_obj.critical_uncertainty_requires_full_suite:
                full_suite_reasons.append(reason)
                full_suite_reasons.append(REASON_FULL_SUITE_UNCERTAINTY)
        elif (
            not edge.critical
            and policy_obj.non_critical_uncertainty_requires_broader
            and (touches or relevant_kind)
        ):
            broader_flag = True
            fallback_reasons.append(reason)

    if conflicting:
        broader_flag = True
        fallback_reasons.append(REASON_CONFLICTING_CRITICAL)
        if policy_obj.critical_uncertainty_requires_full_suite:
            full_suite_reasons.append(REASON_CONFLICTING_CRITICAL)
            full_suite_reasons.append(REASON_FULL_SUITE_UNCERTAINTY)

    mapped_nodes, validation_broader, unmapped = _validation_nodes(
        validation_selection
    )
    if validation_broader or unmapped:
        broader_flag = True
        fallback_reasons.append(REASON_VALIDATION_MAPPING_INCOMPLETE)

    # --- Exact selections ---
    selected_tests: dict[str, tuple[str, ...]] = {}
    selected_static: dict[str, tuple[str, ...]] = {}
    selected_type: dict[str, tuple[str, ...]] = {}
    selected_proofs: dict[str, tuple[str, ...]] = {}

    test_edges = _collect_selector_edges(all_edges, _TEST_EDGE_KINDS)
    proof_edges = _collect_selector_edges(all_edges, _PROOF_EDGE_KINDS)
    fixture_edges = _collect_selector_edges(all_edges, _FIXTURE_EDGE_KINDS)
    config_edges = _collect_selector_edges(all_edges, _CONFIG_EDGE_KINDS)

    # Transitive cone nodes select tests/proofs via exact edges.
    for node in sorted(cone_nodes):
        base_chain = cone_chains.get(node, (node,))
        if node not in symbols and node not in paths and len(base_chain) > 1:
            selection_reasons.append(REASON_TRANSITIVE_DEPENDENCY)

        for target, edge in test_edges.get(node, ()):
            if _edge_is_uncertain(edge):
                continue  # uncertain test edges broaden; do not exact-select
            _record_selection(
                selected=selected_tests,
                item=target,
                chain=(*base_chain, REASON_TESTED_BY, target),
                max_reason_chain=max_chain,
            )
            selection_reasons.append(REASON_TESTED_BY)

        for target, edge in proof_edges.get(node, ()):
            if _edge_is_uncertain(edge):
                continue
            _record_selection(
                selected=selected_proofs,
                item=target,
                chain=(*base_chain, REASON_PROVED_BY, target),
                max_reason_chain=max_chain,
            )
            selection_reasons.append(REASON_PROVED_BY)

        for target, edge in fixture_edges.get(node, ()):
            if _edge_is_uncertain(edge):
                continue
            # Fixture edge: source is fixture/symbol, target is test or reverse.
            if _looks_like_test(target):
                test_id = target
            elif _looks_like_test(edge.source):
                test_id = edge.source
            else:
                test_id = target
            _record_selection(
                selected=selected_tests,
                item=test_id,
                chain=(*base_chain, REASON_FIXTURE_EDGE, test_id),
                max_reason_chain=max_chain,
            )
            selection_reasons.append(REASON_FIXTURE_EDGE)

        for target, edge in config_edges.get(node, ()):
            if _edge_is_uncertain(edge):
                continue
            # Config change selecting a configured module/test.
            if _looks_like_test(target):
                _record_selection(
                    selected=selected_tests,
                    item=target,
                    chain=(*base_chain, REASON_CONFIG_EDGE, target),
                    max_reason_chain=max_chain,
                )
                selection_reasons.append(REASON_CONFIG_EDGE)
            else:
                # Configured symbols/paths join the cone for further selection.
                if target not in cone_nodes:
                    cone_nodes.add(target)
                    # Re-process via a local one-step: tests/proofs on target.
                    for t2, e2 in test_edges.get(target, ()):
                        if _edge_is_uncertain(e2):
                            continue
                        _record_selection(
                            selected=selected_tests,
                            item=t2,
                            chain=(
                                *base_chain,
                                REASON_CONFIG_EDGE,
                                target,
                                REASON_TESTED_BY,
                                t2,
                            ),
                            max_reason_chain=max_chain,
                        )
                        selection_reasons.append(REASON_CONFIG_EDGE)
                        selection_reasons.append(REASON_TESTED_BY)
                    for t2, e2 in proof_edges.get(target, ()):
                        if _edge_is_uncertain(e2):
                            continue
                        _record_selection(
                            selected=selected_proofs,
                            item=t2,
                            chain=(
                                *base_chain,
                                REASON_CONFIG_EDGE,
                                target,
                                REASON_PROVED_BY,
                                t2,
                            ),
                            max_reason_chain=max_chain,
                        )
                        selection_reasons.append(REASON_CONFIG_EDGE)
                        selection_reasons.append(REASON_PROVED_BY)

    # Also walk config edges where the changed node is the config source.
    for edge in all_edges:
        if edge.kind not in _CONFIG_EDGE_KINDS:
            continue
        if _edge_is_uncertain(edge):
            continue
        if edge.source not in (set(symbols) | set(paths) | cone_nodes):
            continue
        base_chain = cone_chains.get(edge.source, (edge.source,))
        if _looks_like_test(edge.target):
            _record_selection(
                selected=selected_tests,
                item=edge.target,
                chain=(*base_chain, REASON_CONFIG_EDGE, edge.target),
                max_reason_chain=max_chain,
            )
            selection_reasons.append(REASON_CONFIG_EDGE)

    # Fixture edges where fixture path itself changed.
    for edge in all_edges:
        if edge.kind not in _FIXTURE_EDGE_KINDS:
            continue
        if _edge_is_uncertain(edge):
            continue
        if edge.source in (set(symbols) | set(paths) | cone_nodes) or edge.target in (
            set(symbols) | set(paths) | cone_nodes
        ):
            base = edge.source if edge.source in cone_nodes or edge.source in symbols or edge.source in paths else edge.target
            base_chain = cone_chains.get(base, (base,))
            test_id = edge.target if _looks_like_test(edge.target) else (
                edge.source if _looks_like_test(edge.source) else edge.target
            )
            if _looks_like_test(test_id):
                _record_selection(
                    selected=selected_tests,
                    item=test_id,
                    chain=(*base_chain, REASON_FIXTURE_EDGE, test_id),
                    max_reason_chain=max_chain,
                )
                selection_reasons.append(REASON_FIXTURE_EDGE)

    # Validation ID mapping contributes exact pytest nodes.
    for node in mapped_nodes:
        _record_selection(
            selected=selected_tests,
            item=node,
            chain=(REASON_VALIDATION_MAPPING, node),
            max_reason_chain=max_chain,
        )
        selection_reasons.append(REASON_VALIDATION_MAPPING)

    # Catalog-driven static / type / proof selection from cone intersection.
    for check_id, targets in catalog_obj.static_check_targets.items():
        hit = sorted(set(targets) & cone_nodes)
        if not hit:
            continue
        root = hit[0]
        base_chain = cone_chains.get(root, (root,))
        _record_selection(
            selected=selected_static,
            item=check_id,
            chain=(*base_chain, REASON_STATIC_TARGET, check_id),
            max_reason_chain=max_chain,
        )
        selection_reasons.append(REASON_STATIC_TARGET)

    for check_id, targets in catalog_obj.type_check_targets.items():
        hit = sorted(set(targets) & cone_nodes)
        if not hit:
            continue
        root = hit[0]
        base_chain = cone_chains.get(root, (root,))
        _record_selection(
            selected=selected_type,
            item=check_id,
            chain=(*base_chain, REASON_TYPE_TARGET, check_id),
            max_reason_chain=max_chain,
        )
        selection_reasons.append(REASON_TYPE_TARGET)

    for obligation_id, deps in catalog_obj.proof_obligation_dependencies.items():
        hit = sorted(set(deps) & cone_nodes)
        if not hit:
            continue
        root = hit[0]
        base_chain = cone_chains.get(root, (root,))
        _record_selection(
            selected=selected_proofs,
            item=obligation_id,
            chain=(*base_chain, REASON_PROOF_DEPENDENCY, obligation_id),
            max_reason_chain=max_chain,
        )
        selection_reasons.append(REASON_PROOF_DEPENDENCY)

    # Proof obligations listed in catalog without deps are selected only via
    # proved_by edges (already handled). Obligation IDs in catalog that match
    # selected proofs stay; orphans are not auto-selected.

    # Restrict exact tests to catalog when a catalog is provided.
    catalog_tests = set(catalog_obj.tests)
    if catalog_tests:
        selected_tests = {
            key: chain
            for key, chain in selected_tests.items()
            if key in catalog_tests
        }

    if catalog_obj.static_checks:
        allowed = set(catalog_obj.static_checks)
        selected_static = {
            key: chain
            for key, chain in selected_static.items()
            if key in allowed
        }
    if catalog_obj.type_checks:
        allowed = set(catalog_obj.type_checks)
        selected_type = {
            key: chain
            for key, chain in selected_type.items()
            if key in allowed
        }
    if catalog_obj.proof_obligations:
        allowed = set(catalog_obj.proof_obligations)
        selected_proofs = {
            key: chain
            for key, chain in selected_proofs.items()
            if key in allowed
        }

    exact_tests = _unique_sorted(selected_tests)
    exact_static = _unique_sorted(selected_static)
    exact_type = _unique_sorted(selected_type)
    exact_proofs = _unique_sorted(selected_proofs)

    # Unrelated edit signal: changes present but zero exact expansion.
    if (
        (symbols or paths)
        and not (exact_tests or exact_static or exact_type or exact_proofs)
        and not broader_flag
        and not critical_uncertain
        and not conflicting
    ):
        # Only annotate when there is no uncertainty forcing broader.
        selection_reasons.append(REASON_UNRELATED_NO_EXPANSION)

    # --- Fallback escalation ---
    full_suite = bool(policy_obj.force_full_suite)
    if full_suite:
        full_suite_reasons.append(REASON_FULL_SUITE_POLICY)
        broader_flag = True

    if broader_flag and policy_obj.broader_escalates_to_full_suite:
        full_suite = True
        full_suite_reasons.append(REASON_FULL_SUITE_UNCERTAINTY)
        full_suite_reasons.append(REASON_BROADER_REQUIRED)

    if full_suite_reasons:
        full_suite = True
        broader_flag = True

    fallback_tests: list[str] = []
    if full_suite:
        fallback_mode = FallbackMode.FULL_SUITE
        if catalog_obj.tests:
            fallback_tests = list(catalog_obj.tests)
        else:
            # Without a catalog, fallback is empty but full_suite_required is set
            # so the planner can require an external full suite.
            fallback_tests = []
        fallback_reasons.append(REASON_BROADER_REQUIRED)
        if REASON_CATALOG_FALLBACK not in fallback_reasons and catalog_obj.tests:
            fallback_reasons.append(REASON_CATALOG_FALLBACK)
        # Full suite also pulls in all catalog static/type/proofs as required.
        if catalog_obj.static_checks:
            for check_id in catalog_obj.static_checks:
                selected_static.setdefault(
                    check_id, (REASON_CATALOG_FALLBACK, check_id)
                )
            exact_static = _unique_sorted(selected_static)
        if catalog_obj.type_checks:
            for check_id in catalog_obj.type_checks:
                selected_type.setdefault(
                    check_id, (REASON_CATALOG_FALLBACK, check_id)
                )
            exact_type = _unique_sorted(selected_type)
        if catalog_obj.proof_obligations:
            for obligation_id in catalog_obj.proof_obligations:
                selected_proofs.setdefault(
                    obligation_id, (REASON_CATALOG_FALLBACK, obligation_id)
                )
            exact_proofs = _unique_sorted(selected_proofs)
    elif broader_flag:
        fallback_mode = FallbackMode.BROADER
        fallback_reasons.append(REASON_BROADER_REQUIRED)
        if policy_obj.broader_includes_sibling_tests and catalog_obj.tests:
            siblings = _sibling_tests(exact_tests, catalog_obj.tests)
            # Broader fallback is the sibling expansion beyond exact.
            fallback_tests = [t for t in siblings if t not in set(exact_tests)]
            if not fallback_tests and catalog_obj.tests:
                # No exact tests: broader means all catalog tests under changed
                # path prefixes when possible, else entire catalog.
                path_prefixes = {
                    p for p in paths if p.endswith(".py") or "/" in p
                }
                for test in catalog_obj.tests:
                    file_part = _test_file_prefix(test)
                    if any(
                        file_part == prefix
                        or file_part.startswith(prefix.rstrip("/") + "/")
                        or prefix in file_part
                        for prefix in path_prefixes
                    ):
                        fallback_tests.append(test)
                if not fallback_tests:
                    fallback_tests = list(catalog_obj.tests)
            if fallback_tests:
                fallback_reasons.append(REASON_CATALOG_FALLBACK)
        elif catalog_obj.tests and not exact_tests:
            fallback_tests = list(catalog_obj.tests)
            fallback_reasons.append(REASON_CATALOG_FALLBACK)
    else:
        fallback_mode = FallbackMode.EXACT

    # Reason chains for all selected items (tests, checks, proofs).
    reason_chains: dict[str, tuple[str, ...]] = {}
    for item, chain in selected_tests.items():
        reason_chains[item] = chain
    for item, chain in selected_static.items():
        reason_chains[item] = chain
    for item, chain in selected_type.items():
        reason_chains[item] = chain
    for item, chain in selected_proofs.items():
        reason_chains[item] = chain
    # Include cone chains for transparency (do not overwrite selected items).
    for node, chain in cone_chains.items():
        reason_chains.setdefault(f"cone:{node}", chain)

    # Final cone symbols/paths include config-expanded nodes.
    final_symbols = set(cone_symbols) | set(symbols)
    final_paths = set(cone_paths) | set(paths)
    for node in cone_nodes:
        if _looks_like_path(node) and not _looks_like_test(node):
            final_paths.add(node)
        elif not _looks_like_test(node):
            final_symbols.add(node)

    return AffectedVerificationSelection(
        affected_tests=exact_tests,
        fallback_tests=_unique_sorted(fallback_tests),
        required_static_checks=exact_static,
        required_type_checks=exact_type,
        affected_proof_obligation_cids=exact_proofs,
        dependency_cone_symbols=_unique_sorted(final_symbols),
        dependency_cone_paths=_unique_sorted(final_paths),
        reason_chains=reason_chains,
        fallback_mode=fallback_mode,
        broader_selection_required=broader_flag or full_suite,
        full_suite_required=full_suite,
        fallback_reason_codes=_stable_unique(fallback_reasons),
        full_suite_reason_codes=_stable_unique(full_suite_reasons),
        selection_reason_codes=_stable_unique(selection_reasons),
        critical_uncertain_edges=_unique_sorted(critical_uncertain),
        conflicting_edge_ids=_unique_sorted(conflicting),
        policy_id=policy_obj.policy_id,
    )


# ---------------------------------------------------------------------------
# Selector object (planner-friendly collaborator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AffectedCheckSelector:
    """Reusable selector with a fixed catalog and policy."""

    INTERFACE: Final[str] = AFFECTED_CHECK_SELECTOR_INTERFACE

    catalog: VerificationCatalog = field(default_factory=VerificationCatalog)
    policy: SelectionPolicy = field(default_factory=SelectionPolicy)

    def select(
        self,
        *,
        changed_symbols: Sequence[str] | None = None,
        changed_paths: Sequence[str] | None = None,
        edges: Sequence[Any] | None = None,
        uncovered_symbols: Sequence[str] | None = None,
        uncovered_paths: Sequence[str] | None = None,
        truncated: bool = False,
        requires_broader_selection: bool = False,
        invalidation_plan: InvalidationPlanView | Mapping[str, Any] | None = None,
        semantic_capsule: SemanticCapsuleView | Mapping[str, Any] | None = None,
        validation_selection: (
            ValidationSelectionView | Mapping[str, Any] | None
        ) = None,
    ) -> AffectedVerificationSelection:
        return select_affected_verification(
            changed_symbols=changed_symbols,
            changed_paths=changed_paths,
            edges=edges,
            uncovered_symbols=uncovered_symbols,
            uncovered_paths=uncovered_paths,
            truncated=truncated,
            requires_broader_selection=requires_broader_selection,
            invalidation_plan=invalidation_plan,
            semantic_capsule=semantic_capsule,
            validation_selection=validation_selection,
            catalog=self.catalog,
            policy=self.policy,
        )


def create_affected_check_selector(
    *,
    catalog: VerificationCatalog | Mapping[str, Any] | None = None,
    policy: SelectionPolicy | Mapping[str, Any] | None = None,
) -> AffectedCheckSelector:
    return AffectedCheckSelector(
        catalog=VerificationCatalog.from_value(catalog),
        policy=SelectionPolicy.from_value(policy),
    )


__all__ = [
    "AFFECTED_CHECK_SELECTOR_INTERFACE",
    "AFFECTED_VERIFICATION_SELECTION_INTERFACE",
    "AFFECTED_VERIFICATION_SELECTION_SCHEMA",
    "DEFAULT_MAX_REASON_CHAIN",
    "REASON_BROADER_REQUIRED",
    "REASON_CONFIG_EDGE",
    "REASON_CONFLICTING_CRITICAL",
    "REASON_CONSERVATIVE_CLOSURE",
    "REASON_DIRECT_PATH",
    "REASON_DIRECT_SYMBOL",
    "REASON_DYNAMIC_CRITICAL",
    "REASON_FIXTURE_EDGE",
    "REASON_FULL_SUITE_POLICY",
    "REASON_FULL_SUITE_UNCERTAINTY",
    "REASON_INPUT_REQUIRES_BROADER",
    "REASON_MISSING_CRITICAL",
    "REASON_OPAQUE_CRITICAL",
    "REASON_PROOF_DEPENDENCY",
    "REASON_PROVED_BY",
    "REASON_STATIC_TARGET",
    "REASON_TESTED_BY",
    "REASON_TRANSITIVE_DEPENDENCY",
    "REASON_TRUNCATED_FRONTIER",
    "REASON_TYPE_TARGET",
    "REASON_UNCOVERED_IMPACT",
    "REASON_UNKNOWN_CRITICAL",
    "REASON_UNRELATED_NO_EXPANSION",
    "REASON_VALIDATION_MAPPING",
    "REASON_VALIDATION_MAPPING_INCOMPLETE",
    "SELECTION_EVIDENCE",
    "SELECTION_POLICY_SCHEMA",
    "VERIFICATION_CATALOG_SCHEMA",
    "AffectedCheckSelector",
    "AffectedVerificationSelection",
    "FallbackMode",
    "SelectionBoundsError",
    "SelectionDisposition",
    "SelectionEdge",
    "SelectionError",
    "SelectionPolicy",
    "VerificationCatalog",
    "compute_dependency_cone",
    "create_affected_check_selector",
    "select_affected_verification",
]
