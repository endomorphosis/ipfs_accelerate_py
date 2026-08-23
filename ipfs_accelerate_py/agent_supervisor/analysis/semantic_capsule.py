"""FACP-047 — content-addressed semantic capsules over the dependency graph.

Extends :mod:`semantic_dependency_graph` with a capsule projection, hermetic
Datalog invalidation/reuse derivations, incremental compile that equals a clean
rebuild, minimal path explanations, and automatic demotion of stale historical
receipts.

This module does **not**:

* recompile datasets ``SemanticCapsuleCompiler@1`` producer capsules
* restate accelerate admission / raw-source substitution
* rebuild a parallel graph authority (the dependency graph remains sole edge
  authority)
* embed raw repository dumps inside capsules (only opaque ``source_cids``)
* allow reuse when a required dependency is unknown
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph import (
    DEFAULT_MAX_CLOSURE_DEPTH,
    DEFAULT_MAX_CLOSURE_NODES,
    SemanticDependencyGraph,
    SemanticEdge,
    SemanticEdgeKind,
    SemanticNode,
    SemanticNodeKind,
    canonical_semantic_json,
)

# ---------------------------------------------------------------------------
# FACP evidence envelope
# ---------------------------------------------------------------------------

SCHEMA: Final[str] = "facp/semantic-capsule@1"
INDEX_SCHEMA: Final[str] = "facp/semantic-capsule-index@1"
INVALIDATION_SCHEMA: Final[str] = "facp/invalidation-soundness@1"
EVIDENCE_SCHEMA: Final[str] = SCHEMA
TASK_ID: Final[str] = "FACP-047"
GOAL_ID: Final[str] = "FACP-G610"
BUNDLE: Final[str] = "facp/incremental/capsules"
ANALYZER_VERSION: Final[str] = "semantic-capsule/v1"
HERMETIC_EVALUATOR_ID: Final[str] = (
    "semantic_capsule.hermetic_reference_evaluator/v1"
)

DEFAULT_MAX_CAPSULES: Final[int] = 16_384
DEFAULT_MAX_DATALOG_FACTS: Final[int] = 100_000
DEFAULT_MAX_DATALOG_ITERATIONS: Final[int] = 256

# Edge kinds that establish a capsule dependency (subject -> dependency).
_DEPENDENCY_EDGE_KINDS: Final[frozenset[SemanticEdgeKind]] = frozenset(
    {
        SemanticEdgeKind.REQUIRES,
        SemanticEdgeKind.DEPENDS_ON,
        SemanticEdgeKind.PROVEN_BY,
        SemanticEdgeKind.CONSTRAINED_BY,
        SemanticEdgeKind.IMPLEMENTS,
        SemanticEdgeKind.MONITORED_BY,
        SemanticEdgeKind.SOURCED_FROM,
        SemanticEdgeKind.AFFECTS,
    }
)

# Capsules of these kinds are "required validations" under G610 acceptance.
_REQUIRED_VALIDATION_KINDS: Final[frozenset[str]] = frozenset(
    {"proof", "test", "release"}
)

EVIDENCE_FIELDS: Final[tuple[str, ...]] = (
    "exports",
    "requires",
    "effects",
    "authority",
    "abstract_state",
    "assumptions",
    "guarantees",
    "proofs",
    "tests",
    "public_data",
    "environment",
    "source_cids",
)


class SemanticCapsuleError(ValueError):
    """Malformed capsule, index, or invalidation input."""


class CapsuleBoundsError(SemanticCapsuleError):
    """A capsule compile or derivation exceeded a deterministic bound."""


class UnknownDependencyError(SemanticCapsuleError):
    """Reuse requested for a capsule with an unknown required dependency."""


class StaleReceiptError(SemanticCapsuleError):
    """A caller treated a demoted historical receipt as live authority."""


class CapsuleKind(str, Enum):
    SYMBOL = "symbol"
    CONTRACT = "contract"
    EFFECT = "effect"
    POLICY = "policy"
    PROOF = "proof"
    TEST = "test"
    ENVIRONMENT = "environment"
    RELEASE = "release"


class CapsuleAction(str, Enum):
    REUSE = "reuse"
    INVALIDATE = "invalidate"
    DEMOTE = "demote"


class ReceiptStatus(str, Enum):
    LIVE = "live"
    DEMOTED = "demoted"


class ReceiptKind(str, Enum):
    PROOF = "proof"
    TEST = "test"
    RELEASE = "release"


class CapsuleRuleId(str, Enum):
    SEEDED_CHANGE = "capsule.rule.seeded_change"
    TRANSITIVE_INVALIDATION = "capsule.rule.transitive_invalidation"
    UNKNOWN_DEPENDENCY = "capsule.rule.unknown_dependency"
    INPUTS_UNCHANGED = "capsule.rule.inputs_unchanged"
    STALE_HISTORICAL_RECEIPT = "capsule.rule.stale_historical_receipt"
    REQUIRED_VALIDATION = "capsule.rule.required_validation"


# Kind projection: priority order when a node matches multiple buckets.
_KIND_PRIORITY: Final[tuple[CapsuleKind, ...]] = (
    CapsuleKind.RELEASE,
    CapsuleKind.PROOF,
    CapsuleKind.TEST,
    CapsuleKind.POLICY,
    CapsuleKind.CONTRACT,
    CapsuleKind.EFFECT,
    CapsuleKind.ENVIRONMENT,
    CapsuleKind.SYMBOL,
)

_NODE_KIND_TO_CAPSULE: Final[Mapping[SemanticNodeKind, CapsuleKind]] = (
    MappingProxyType(
        {
            SemanticNodeKind.SYMBOL: CapsuleKind.SYMBOL,
            SemanticNodeKind.AST: CapsuleKind.SYMBOL,
            SemanticNodeKind.FILE: CapsuleKind.SYMBOL,
            SemanticNodeKind.INTERFACE: CapsuleKind.SYMBOL,
            SemanticNodeKind.CALL: CapsuleKind.SYMBOL,
            SemanticNodeKind.PROGRAM: CapsuleKind.SYMBOL,
            SemanticNodeKind.DATA_FLOW: CapsuleKind.SYMBOL,
            SemanticNodeKind.OBLIGATION: CapsuleKind.CONTRACT,
            SemanticNodeKind.LEGAL_OBLIGATION: CapsuleKind.CONTRACT,
            SemanticNodeKind.LEGAL_PROHIBITION: CapsuleKind.CONTRACT,
            SemanticNodeKind.INTENT_PRECONDITION: CapsuleKind.CONTRACT,
            SemanticNodeKind.INTENT_POSTCONDITION: CapsuleKind.CONTRACT,
            SemanticNodeKind.INTENT_INVARIANT: CapsuleKind.CONTRACT,
            SemanticNodeKind.EFFECT: CapsuleKind.EFFECT,
            SemanticNodeKind.INTENT_EFFECT: CapsuleKind.EFFECT,
            SemanticNodeKind.ACTION: CapsuleKind.EFFECT,
            SemanticNodeKind.SECURITY_POLICY: CapsuleKind.POLICY,
            SemanticNodeKind.AUTHORIZATION: CapsuleKind.POLICY,
            SemanticNodeKind.LEGAL_PERMISSION: CapsuleKind.POLICY,
            SemanticNodeKind.LEGAL_POWER: CapsuleKind.POLICY,
            SemanticNodeKind.LEGAL_EXCEPTION: CapsuleKind.POLICY,
            SemanticNodeKind.PROOF: CapsuleKind.PROOF,
            SemanticNodeKind.VALIDATION: CapsuleKind.TEST,
            SemanticNodeKind.MONITOR: CapsuleKind.TEST,
            SemanticNodeKind.ENVIRONMENT: CapsuleKind.ENVIRONMENT,
            SemanticNodeKind.TOOLCHAIN: CapsuleKind.ENVIRONMENT,
            SemanticNodeKind.WORKTREE: CapsuleKind.ENVIRONMENT,
            SemanticNodeKind.REPOSITORY_TREE: CapsuleKind.ENVIRONMENT,
            SemanticNodeKind.MERGE_EVIDENCE: CapsuleKind.RELEASE,
            SemanticNodeKind.DECISION: CapsuleKind.RELEASE,
        }
    )
)


# ---------------------------------------------------------------------------
# Canonical identity helpers
# ---------------------------------------------------------------------------


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise CapsuleBoundsError("capsule record exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise SemanticCapsuleError("floating values are not canonical capsule data")
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise CapsuleBoundsError("capsule mapping is invalid")
        return {key: _plain(value[key], depth=depth + 1) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 16_384:
            raise CapsuleBoundsError("capsule sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise SemanticCapsuleError(
        f"unsupported capsule value: {type(value).__name__}"
    )


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(
        canonical_semantic_json(value).encode("utf-8")
    ).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise SemanticCapsuleError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise SemanticCapsuleError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise SemanticCapsuleError(f"{name} is required")
    if len(value.encode("utf-8")) > 8_192:
        raise CapsuleBoundsError(f"{name} is oversized")
    return value


def _unique_sorted(values: Iterable[Any], name: str) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = _text(str(getattr(raw, "value", raw)), name, required=True)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(sorted(out))


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw))
    except (TypeError, ValueError) as exc:
        raise SemanticCapsuleError(f"invalid {name}: {value!r}") from exc


def project_capsule_kind(node: SemanticNode) -> CapsuleKind | None:
    """Map an authoritative graph node to a closed capsule kind, or omit it."""

    if not isinstance(node, SemanticNode):
        raise SemanticCapsuleError("node must be a SemanticNode")
    if not node.authoritative:
        return None
    mapped = _NODE_KIND_TO_CAPSULE.get(node.kind)
    return mapped


# ---------------------------------------------------------------------------
# Hermetic Datalog (module-local; IPA-shaped, no IPA rule IDs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatalogAtom:
    predicate: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        if not str(self.predicate).strip():
            raise SemanticCapsuleError("datalog atom predicate is required")
        if len(self.args) > 16:
            raise CapsuleBoundsError("datalog atom arity exceeds bound")
        object.__setattr__(
            self, "args", tuple(str(arg) for arg in self.args)
        )

    @property
    def is_ground(self) -> bool:
        return all(not _is_variable(arg) for arg in self.args)

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "args": list(self.args)}


@dataclass(frozen=True)
class DatalogRule:
    rule_id: str
    head: DatalogAtom
    body: tuple[DatalogAtom, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.rule_id).strip():
            raise SemanticCapsuleError("datalog rule_id is required")
        object.__setattr__(self, "body", tuple(self.body))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "head": self.head.to_dict(),
            "body": [atom.to_dict() for atom in self.body],
        }


@dataclass(frozen=True)
class DatalogEvaluationResult:
    relations: Mapping[str, frozenset[tuple[str, ...]]]
    derived_rule_ids: tuple[str, ...]
    evaluator_id: str = HERMETIC_EVALUATOR_ID
    iterations: int = 0

    def facts(self, predicate: str) -> frozenset[tuple[str, ...]]:
        return self.relations.get(predicate, frozenset())

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "iterations": self.iterations,
            "derived_rule_ids": list(self.derived_rule_ids),
            "relations": {
                name: [list(row) for row in sorted(rows)]
                for name, rows in sorted(self.relations.items())
            },
        }


def _is_variable(token: str) -> bool:
    return bool(token) and token[0].isupper()


class HermeticReferenceEvaluator:
    """Bounded bottom-up Horn evaluator used for capsule impact derivations."""

    def __init__(
        self,
        *,
        max_iterations: int = DEFAULT_MAX_DATALOG_ITERATIONS,
        max_facts: int = DEFAULT_MAX_DATALOG_FACTS,
    ) -> None:
        if max_iterations < 1 or max_iterations > 10_000:
            raise SemanticCapsuleError("max_iterations out of bounds")
        if max_facts < 1 or max_facts > 1_000_000:
            raise SemanticCapsuleError("max_facts out of bounds")
        self.max_iterations = max_iterations
        self.max_facts = max_facts
        self.evaluator_id = HERMETIC_EVALUATOR_ID

    def evaluate(
        self,
        facts: Sequence[DatalogAtom],
        rules: Sequence[DatalogRule],
    ) -> DatalogEvaluationResult:
        relations: dict[str, set[tuple[str, ...]]] = {}
        for fact in facts:
            if not fact.is_ground:
                raise SemanticCapsuleError(f"fact must be ground: {fact!r}")
            relations.setdefault(fact.predicate, set()).add(fact.args)
        derived_rules: set[str] = set()
        iterations = 0
        changed = True
        while changed and iterations < self.max_iterations:
            changed = False
            iterations += 1
            for rule in rules:
                for binding in self._match_body(rule.body, relations):
                    head_args = tuple(
                        binding.get(arg, arg) if _is_variable(arg) else arg
                        for arg in rule.head.args
                    )
                    if any(_is_variable(arg) for arg in head_args):
                        continue
                    bucket = relations.setdefault(rule.head.predicate, set())
                    before = len(bucket)
                    bucket.add(head_args)
                    if len(bucket) > before:
                        changed = True
                        derived_rules.add(rule.rule_id)
                    total = sum(len(rows) for rows in relations.values())
                    if total > self.max_facts:
                        raise CapsuleBoundsError(
                            "hermetic datalog evaluation exceeded fact bound"
                        )
        frozen = {
            name: frozenset(rows) for name, rows in sorted(relations.items())
        }
        return DatalogEvaluationResult(
            relations=frozen,
            derived_rule_ids=tuple(sorted(derived_rules)),
            evaluator_id=self.evaluator_id,
            iterations=iterations,
        )

    def _match_body(
        self,
        body: Sequence[DatalogAtom],
        relations: Mapping[str, set[tuple[str, ...]]],
    ) -> Iterator[dict[str, str]]:
        if not body:
            yield {}
            return

        def rec(index: int, binding: dict[str, str]) -> Iterator[dict[str, str]]:
            if index >= len(body):
                yield dict(binding)
                return
            atom = body[index]
            for row in sorted(relations.get(atom.predicate, set())):
                if len(row) != len(atom.args):
                    continue
                next_binding = dict(binding)
                ok = True
                for schema, value in zip(atom.args, row):
                    if _is_variable(schema):
                        existing = next_binding.get(schema)
                        if existing is None:
                            next_binding[schema] = value
                        elif existing != value:
                            ok = False
                            break
                    elif schema != value:
                        ok = False
                        break
                if ok:
                    yield from rec(index + 1, next_binding)

        yield from rec(0, {})


def default_capsule_datalog_rules() -> tuple[DatalogRule, ...]:
    """Positive Horn rules for seeded and transitive capsule invalidation."""

    return (
        DatalogRule(
            rule_id=CapsuleRuleId.SEEDED_CHANGE.value,
            head=DatalogAtom("Invalidates", ("X", "X")),
            body=(DatalogAtom("Changed", ("X",)),),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.TRANSITIVE_INVALIDATION.value,
            head=DatalogAtom("Invalidates", ("X", "Y")),
            body=(
                DatalogAtom("Changed", ("X",)),
                DatalogAtom("DependsOn", ("Y", "X")),
            ),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.TRANSITIVE_INVALIDATION.value,
            head=DatalogAtom("Invalidates", ("X", "Z")),
            body=(
                DatalogAtom("Invalidates", ("X", "Y")),
                DatalogAtom("DependsOn", ("Z", "Y")),
            ),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.REQUIRED_VALIDATION.value,
            head=DatalogAtom("RequiresRevalidation", ("Y", "Kind")),
            body=(
                DatalogAtom("Invalidates", ("X", "Y")),
                DatalogAtom("Capsule", ("Y", "Kind")),
                DatalogAtom("RequiredKind", ("Kind",)),
            ),
        ),
        # When a subject is invalidated, its required proof/test/release
        # dependencies must be revalidated even if their own content is stable.
        DatalogRule(
            rule_id=CapsuleRuleId.REQUIRED_VALIDATION.value,
            head=DatalogAtom("RequiresRevalidation", ("P", "Kind")),
            body=(
                DatalogAtom("Invalidates", ("X", "Y")),
                DatalogAtom("DependsOn", ("Y", "P")),
                DatalogAtom("Capsule", ("P", "Kind")),
                DatalogAtom("RequiredKind", ("Kind",)),
            ),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.REQUIRED_VALIDATION.value,
            head=DatalogAtom("Invalidates", ("X", "P")),
            body=(
                DatalogAtom("RequiresRevalidation", ("P", "Kind")),
                DatalogAtom("Invalidates", ("X", "Y")),
                DatalogAtom("DependsOn", ("Y", "P")),
            ),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.STALE_HISTORICAL_RECEIPT.value,
            head=DatalogAtom("Demote", ("R", CapsuleRuleId.STALE_HISTORICAL_RECEIPT.value)),
            body=(
                DatalogAtom("BindsReceipt", ("R", "C")),
                DatalogAtom("Invalidates", ("X", "C")),
            ),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.STALE_HISTORICAL_RECEIPT.value,
            head=DatalogAtom("Demote", ("R", CapsuleRuleId.STALE_HISTORICAL_RECEIPT.value)),
            body=(
                DatalogAtom("BindsReceipt", ("R", "C")),
                DatalogAtom("AbsentCapsule", ("C",)),
            ),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.UNKNOWN_DEPENDENCY.value,
            head=DatalogAtom("NoReuse", ("Y",)),
            body=(DatalogAtom("UnknownDep", ("Y", "Z")),),
        ),
        DatalogRule(
            rule_id=CapsuleRuleId.UNKNOWN_DEPENDENCY.value,
            head=DatalogAtom("NoReuse", ("Y",)),
            body=(DatalogAtom("Invalidates", ("X", "Y")),),
        ),
    )


# ---------------------------------------------------------------------------
# Capsule records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapsuleEvidence:
    """Closed evidence subset bound into every capsule identity."""

    exports: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    authority: str = "authoritative"
    abstract_state: Mapping[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    guarantees: tuple[str, ...] = ()
    proofs: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    public_data: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    source_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "exports", _unique_sorted(self.exports, "exports")
        )
        object.__setattr__(
            self, "requires", _unique_sorted(self.requires, "requires")
        )
        object.__setattr__(
            self, "effects", _unique_sorted(self.effects, "effects")
        )
        object.__setattr__(
            self, "authority", _text(self.authority, "authority")
        )
        object.__setattr__(
            self,
            "abstract_state",
            MappingProxyType(_plain(self.abstract_state) if self.abstract_state else {}),
        )
        object.__setattr__(
            self, "assumptions", _unique_sorted(self.assumptions, "assumptions")
        )
        object.__setattr__(
            self, "guarantees", _unique_sorted(self.guarantees, "guarantees")
        )
        object.__setattr__(
            self, "proofs", _unique_sorted(self.proofs, "proofs")
        )
        object.__setattr__(
            self, "tests", _unique_sorted(self.tests, "tests")
        )
        object.__setattr__(
            self, "public_data", _unique_sorted(self.public_data, "public_data")
        )
        object.__setattr__(
            self, "environment", _unique_sorted(self.environment, "environment")
        )
        object.__setattr__(
            self, "source_cids", _unique_sorted(self.source_cids, "source_cids")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exports": list(self.exports),
            "requires": list(self.requires),
            "effects": list(self.effects),
            "authority": self.authority,
            "abstract_state": dict(self.abstract_state),
            "assumptions": list(self.assumptions),
            "guarantees": list(self.guarantees),
            "proofs": list(self.proofs),
            "tests": list(self.tests),
            "public_data": list(self.public_data),
            "environment": list(self.environment),
            "source_cids": list(self.source_cids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapsuleEvidence":
        return cls(
            exports=tuple(payload.get("exports") or ()),
            requires=tuple(payload.get("requires") or ()),
            effects=tuple(payload.get("effects") or ()),
            authority=str(payload.get("authority") or "authoritative"),
            abstract_state=payload.get("abstract_state") or {},
            assumptions=tuple(payload.get("assumptions") or ()),
            guarantees=tuple(payload.get("guarantees") or ()),
            proofs=tuple(payload.get("proofs") or ()),
            tests=tuple(payload.get("tests") or ()),
            public_data=tuple(payload.get("public_data") or ()),
            environment=tuple(payload.get("environment") or ()),
            source_cids=tuple(payload.get("source_cids") or ()),
        )


@dataclass(frozen=True)
class SemanticCapsuleRecord:
    """Content-addressed capsule projected from one authoritative graph node."""

    kind: CapsuleKind
    node_id: str
    node_content_id: str
    graph_id: str
    root_id: str
    source_root_id: str
    authority: str
    trust: str
    provenance: str
    version: str
    evidence: CapsuleEvidence
    dependency_node_ids: tuple[str, ...] = ()
    dependency_cids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    unknown_dependency_refs: tuple[str, ...] = ()
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, CapsuleKind, "kind"))
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(
            self, "node_content_id", _text(self.node_content_id, "node_content_id")
        )
        object.__setattr__(self, "graph_id", _text(self.graph_id, "graph_id"))
        object.__setattr__(self, "root_id", _text(self.root_id, "root_id"))
        object.__setattr__(
            self, "source_root_id", _text(self.source_root_id, "source_root_id")
        )
        object.__setattr__(self, "authority", _text(self.authority, "authority"))
        object.__setattr__(self, "trust", _text(self.trust, "trust"))
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        if not isinstance(self.evidence, CapsuleEvidence):
            object.__setattr__(
                self, "evidence", CapsuleEvidence.from_dict(self.evidence)
            )
        object.__setattr__(
            self,
            "dependency_node_ids",
            _unique_sorted(self.dependency_node_ids, "dependency_node_ids"),
        )
        object.__setattr__(
            self,
            "dependency_cids",
            _unique_sorted(self.dependency_cids, "dependency_cids"),
        )
        object.__setattr__(
            self, "edge_ids", _unique_sorted(self.edge_ids, "edge_ids")
        )
        object.__setattr__(
            self,
            "unknown_dependency_refs",
            _unique_sorted(self.unknown_dependency_refs, "unknown_dependency_refs"),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != SCHEMA:
            raise SemanticCapsuleError(f"unsupported capsule schema: {self.schema}")

    def identity_payload(self) -> dict[str, Any]:
        # Intentionally excludes graph_id so an unrelated edit elsewhere in the
        # same root does not churn this capsule's content address. The index
        # binds the exact graph_id for authority.
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "node_id": self.node_id,
            "node_content_id": self.node_content_id,
            "root_id": self.root_id,
            "source_root_id": self.source_root_id,
            "authority": self.authority,
            "trust": self.trust,
            "provenance": self.provenance,
            "version": self.version,
            "evidence": self.evidence.to_dict(),
            "dependency_node_ids": list(self.dependency_node_ids),
            "dependency_cids": list(self.dependency_cids),
            "edge_ids": list(self.edge_ids),
            "unknown_dependency_refs": list(self.unknown_dependency_refs),
        }

    @property
    def capsule_cid(self) -> str:
        return _identity("semantic-capsule", self.identity_payload())

    @property
    def has_unknown_dependency(self) -> bool:
        return bool(self.unknown_dependency_refs)

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["capsule_cid"] = self.capsule_cid
        payload["graph_id"] = self.graph_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticCapsuleRecord":
        schema = str(payload.get("schema") or SCHEMA)
        if schema != SCHEMA:
            raise SemanticCapsuleError(f"unsupported capsule schema: {schema}")
        record = cls(
            kind=payload.get("kind", ""),
            node_id=str(payload.get("node_id") or ""),
            node_content_id=str(payload.get("node_content_id") or ""),
            graph_id=str(payload.get("graph_id") or ""),
            root_id=str(payload.get("root_id") or ""),
            source_root_id=str(payload.get("source_root_id") or ""),
            authority=str(payload.get("authority") or ""),
            trust=str(payload.get("trust") or ""),
            provenance=str(payload.get("provenance") or ""),
            version=str(payload.get("version") or ""),
            evidence=CapsuleEvidence.from_dict(payload.get("evidence") or {}),
            dependency_node_ids=tuple(payload.get("dependency_node_ids") or ()),
            dependency_cids=tuple(payload.get("dependency_cids") or ()),
            edge_ids=tuple(payload.get("edge_ids") or ()),
            unknown_dependency_refs=tuple(
                payload.get("unknown_dependency_refs") or ()
            ),
            schema=schema,
        )
        claimed = str(payload.get("capsule_cid") or "")
        if claimed and claimed != record.capsule_cid:
            raise SemanticCapsuleError("capsule content identity mismatch")
        return record


@dataclass(frozen=True)
class CapsulePath:
    """Minimal simple path of capsule CIDs with graph edge kinds along it."""

    capsule_cids: tuple[str, ...]
    edge_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capsule_cids",
            tuple(_text(item, "capsule_cid") for item in self.capsule_cids),
        )
        object.__setattr__(
            self,
            "edge_kinds",
            tuple(_text(item, "edge_kind") for item in self.edge_kinds),
        )
        if not self.capsule_cids:
            raise SemanticCapsuleError("path must contain at least one capsule")
        if len(self.edge_kinds) not in {0, len(self.capsule_cids) - 1}:
            raise SemanticCapsuleError(
                "edge_kinds length must be zero or path_length-1"
            )
        if len(set(self.capsule_cids)) != len(self.capsule_cids):
            raise SemanticCapsuleError("path must be simple (no cycles)")

    @property
    def path_cid(self) -> str:
        return _identity(
            "capsule-path",
            {
                "capsule_cids": list(self.capsule_cids),
                "edge_kinds": list(self.edge_kinds),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_cids": list(self.capsule_cids),
            "edge_kinds": list(self.edge_kinds),
            "path_cid": self.path_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapsulePath":
        return cls(
            capsule_cids=tuple(payload.get("capsule_cids") or ()),
            edge_kinds=tuple(payload.get("edge_kinds") or ()),
        )


@dataclass(frozen=True)
class InvalidationExplanation:
    """Minimal path explanation for one reuse, invalidation, or demotion."""

    action: CapsuleAction
    subject_cid: str
    seed_cid: str
    path: CapsulePath
    rule_id: str
    reason: str
    schema: str = INVALIDATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action", _enum(self.action, CapsuleAction, "action")
        )
        object.__setattr__(
            self, "subject_cid", _text(self.subject_cid, "subject_cid")
        )
        object.__setattr__(self, "seed_cid", _text(self.seed_cid, "seed_cid"))
        if not isinstance(self.path, CapsulePath):
            object.__setattr__(self, "path", CapsulePath.from_dict(self.path))
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.path.capsule_cids[-1] != self.subject_cid:
            raise SemanticCapsuleError("explanation path must end at subject")
        if self.path.capsule_cids[0] != self.seed_cid:
            raise SemanticCapsuleError("explanation path must start at seed")

    @property
    def explanation_cid(self) -> str:
        return _identity("capsule-explanation", self.to_dict_without_cid())

    def to_dict_without_cid(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action": self.action.value,
            "subject_cid": self.subject_cid,
            "seed_cid": self.seed_cid,
            "path": self.path.to_dict(),
            "rule_id": self.rule_id,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_dict_without_cid()
        payload["explanation_cid"] = self.explanation_cid
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvalidationExplanation":
        return cls(
            action=payload.get("action", ""),
            subject_cid=str(payload.get("subject_cid") or ""),
            seed_cid=str(payload.get("seed_cid") or ""),
            path=CapsulePath.from_dict(payload.get("path") or {}),
            rule_id=str(payload.get("rule_id") or ""),
            reason=str(payload.get("reason") or ""),
            schema=str(payload.get("schema") or INVALIDATION_SCHEMA),
        )


@dataclass(frozen=True)
class HistoricalReceipt:
    """Historical proof/test/release receipt bound to a capsule CID."""

    receipt_id: str
    kind: ReceiptKind
    bound_capsule_cid: str
    produced_graph_id: str = ""
    status: ReceiptStatus = ReceiptStatus.LIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, "receipt_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, ReceiptKind, "kind"))
        object.__setattr__(
            self,
            "bound_capsule_cid",
            _text(self.bound_capsule_cid, "bound_capsule_cid"),
        )
        object.__setattr__(
            self,
            "produced_graph_id",
            _text(self.produced_graph_id, "produced_graph_id", required=False),
        )
        object.__setattr__(
            self, "status", _enum(self.status, ReceiptStatus, "status")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "kind": self.kind.value,
            "bound_capsule_cid": self.bound_capsule_cid,
            "produced_graph_id": self.produced_graph_id,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HistoricalReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id") or ""),
            kind=payload.get("kind", ""),
            bound_capsule_cid=str(payload.get("bound_capsule_cid") or ""),
            produced_graph_id=str(payload.get("produced_graph_id") or ""),
            status=payload.get("status", ReceiptStatus.LIVE.value),
        )


@dataclass(frozen=True)
class CapsuleIndex:
    graph_id: str
    capsules: tuple[SemanticCapsuleRecord, ...]
    schema: str = INDEX_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_id", _text(self.graph_id, "graph_id"))
        ordered = tuple(
            sorted(self.capsules, key=lambda item: (item.kind.value, item.node_id))
        )
        if len(ordered) > DEFAULT_MAX_CAPSULES:
            raise CapsuleBoundsError("capsule index exceeds max capsules")
        seen: set[str] = set()
        for capsule in ordered:
            if capsule.capsule_cid in seen:
                raise SemanticCapsuleError(
                    f"duplicate capsule cid: {capsule.capsule_cid}"
                )
            seen.add(capsule.capsule_cid)
        object.__setattr__(self, "capsules", ordered)
        object.__setattr__(self, "schema", _text(self.schema, "schema"))

    @property
    def index_cid(self) -> str:
        return _identity(
            "semantic-capsule-index",
            {
                "schema": self.schema,
                "graph_id": self.graph_id,
                "capsule_cids": [item.capsule_cid for item in self.capsules],
                "capsules": [item.identity_payload() for item in self.capsules],
            },
        )

    def by_node_id(self) -> Mapping[str, SemanticCapsuleRecord]:
        return MappingProxyType({item.node_id: item for item in self.capsules})

    def by_cid(self) -> Mapping[str, SemanticCapsuleRecord]:
        return MappingProxyType(
            {item.capsule_cid: item for item in self.capsules}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "graph_id": self.graph_id,
            "index_cid": self.index_cid,
            "capsule_count": len(self.capsules),
            "capsules": [item.to_dict() for item in self.capsules],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapsuleIndex":
        schema = str(payload.get("schema") or INDEX_SCHEMA)
        if schema != INDEX_SCHEMA:
            raise SemanticCapsuleError(f"unsupported index schema: {schema}")
        capsules = tuple(
            SemanticCapsuleRecord.from_dict(item)
            for item in (payload.get("capsules") or ())
        )
        index = cls(
            graph_id=str(payload.get("graph_id") or ""),
            capsules=capsules,
            schema=schema,
        )
        claimed = str(payload.get("index_cid") or "")
        if claimed and claimed != index.index_cid:
            raise SemanticCapsuleError("capsule index identity mismatch")
        return index


@dataclass(frozen=True)
class CapsuleCompileResult:
    """Envelope for a cold or incremental capsule compile."""

    index: CapsuleIndex
    explanations: tuple[InvalidationExplanation, ...] = ()
    reused_cids: tuple[str, ...] = ()
    invalidated_cids: tuple[str, ...] = ()
    demoted_receipts: tuple[HistoricalReceipt, ...] = ()
    datalog: DatalogEvaluationResult | None = None
    schema: str = SCHEMA
    evidence_schema: str = EVIDENCE_SCHEMA
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    bundle: str = BUNDLE
    analyzer_version: str = ANALYZER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.index, CapsuleIndex):
            object.__setattr__(self, "index", CapsuleIndex.from_dict(self.index))
        object.__setattr__(
            self,
            "explanations",
            tuple(
                item
                if isinstance(item, InvalidationExplanation)
                else InvalidationExplanation.from_dict(item)
                for item in self.explanations
            ),
        )
        object.__setattr__(
            self, "reused_cids", _unique_sorted(self.reused_cids, "reused_cids")
        )
        object.__setattr__(
            self,
            "invalidated_cids",
            _unique_sorted(self.invalidated_cids, "invalidated_cids"),
        )
        object.__setattr__(
            self,
            "demoted_receipts",
            tuple(
                item
                if isinstance(item, HistoricalReceipt)
                else HistoricalReceipt.from_dict(item)
                for item in self.demoted_receipts
            ),
        )

    @property
    def capsules(self) -> tuple[SemanticCapsuleRecord, ...]:
        return self.index.capsules

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_schema": self.evidence_schema,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "bundle": self.bundle,
            "analyzer_version": self.analyzer_version,
            "index": self.index.to_dict(),
            "capsule_count": len(self.index.capsules),
            "reused_cids": list(self.reused_cids),
            "invalidated_cids": list(self.invalidated_cids),
            "explanations": [item.to_dict() for item in self.explanations],
            "demoted_receipts": [
                item.to_dict() for item in self.demoted_receipts
            ],
            "datalog": None if self.datalog is None else self.datalog.to_dict(),
        }


# ---------------------------------------------------------------------------
# Graph projection
# ---------------------------------------------------------------------------


def _dependency_edges(
    graph: SemanticDependencyGraph,
) -> tuple[SemanticEdge, ...]:
    return tuple(
        edge
        for edge in graph.edges
        if edge.authoritative
        and edge.mandatory
        and edge.kind in _DEPENDENCY_EDGE_KINDS
    )


def _build_evidence(
    graph: SemanticDependencyGraph,
    node: SemanticNode,
    dep_edges: Sequence[SemanticEdge],
) -> CapsuleEvidence:
    node_by_id = {item.node_id: item for item in graph.nodes}
    requires: list[str] = []
    effects: list[str] = []
    assumptions: list[str] = []
    guarantees: list[str] = []
    proofs: list[str] = []
    tests: list[str] = []
    environment: list[str] = []
    public_data: list[str] = []
    for edge in dep_edges:
        if edge.source != node.node_id:
            continue
        target = node_by_id.get(edge.target)
        requires.append(edge.target)
        if target is None:
            continue
        target_kind = project_capsule_kind(target)
        if edge.kind is SemanticEdgeKind.PROVEN_BY or target.kind is SemanticNodeKind.PROOF:
            proofs.append(edge.target)
        if edge.kind is SemanticEdgeKind.MONITORED_BY or target.kind in {
            SemanticNodeKind.VALIDATION,
            SemanticNodeKind.MONITOR,
        }:
            tests.append(edge.target)
        if target.kind in {
            SemanticNodeKind.ENVIRONMENT,
            SemanticNodeKind.TOOLCHAIN,
            SemanticNodeKind.WORKTREE,
            SemanticNodeKind.REPOSITORY_TREE,
        }:
            environment.append(edge.target)
        if target.kind in {
            SemanticNodeKind.ASSUMPTION,
            SemanticNodeKind.INTENT_ASSUMPTION,
            SemanticNodeKind.LEGAL_ASSUMPTION,
            SemanticNodeKind.SECURITY_THREAT_ASSUMPTION,
        }:
            assumptions.append(edge.target)
        if target.kind in {
            SemanticNodeKind.OBLIGATION,
            SemanticNodeKind.LEGAL_OBLIGATION,
            SemanticNodeKind.INTENT_POSTCONDITION,
            SemanticNodeKind.INTENT_CLAIM,
            SemanticNodeKind.LEGAL_CLAIM,
            SemanticNodeKind.SECURITY_CLAIM,
        }:
            guarantees.append(edge.target)
        if target.kind is SemanticNodeKind.RESOURCE:
            public_data.append(edge.target)
        if target.kind in {
            SemanticNodeKind.EFFECT,
            SemanticNodeKind.INTENT_EFFECT,
        } or edge.kind is SemanticEdgeKind.AFFECTS:
            effects.append(edge.target)
        if target_kind is CapsuleKind.EFFECT:
            effects.append(edge.target)

    record = dict(node.record)
    abstract_state = {}
    for key in ("abstract_state", "ipa_labels", "product_domain", "effects"):
        if key in record:
            abstract_state[key] = record[key]
    source_cids = [node.source_root_id, node.content_id]
    if isinstance(record.get("source_cid"), str) and record["source_cid"]:
        source_cids.append(record["source_cid"])
    exports = [node.node_id]
    if isinstance(record.get("export"), str) and record["export"]:
        exports.append(record["export"])
    if isinstance(record.get("exports"), Sequence) and not isinstance(
        record.get("exports"), (str, bytes, bytearray)
    ):
        exports.extend(str(item) for item in record["exports"])

    return CapsuleEvidence(
        exports=tuple(exports),
        requires=tuple(requires),
        effects=tuple(effects),
        authority=(
            "authoritative" if node.authoritative else node.authority.value
        ),
        abstract_state=abstract_state,
        assumptions=tuple(assumptions),
        guarantees=tuple(guarantees),
        proofs=tuple(proofs),
        tests=tuple(tests),
        public_data=tuple(public_data),
        environment=tuple(environment),
        source_cids=tuple(source_cids),
    )


def _capsule_candidates(
    graph: SemanticDependencyGraph,
) -> tuple[SemanticNode, ...]:
    return tuple(
        node
        for node in graph.nodes
        if project_capsule_kind(node) is not None
    )


def _topological_capsule_nodes(
    graph: SemanticDependencyGraph,
    candidates: Sequence[SemanticNode],
) -> tuple[SemanticNode, ...]:
    """Order capsules so dependency targets appear before dependents."""

    candidate_ids = {node.node_id for node in candidates}
    indegree = {node.node_id: 0 for node in candidates}
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in candidates}
    for edge in _dependency_edges(graph):
        if edge.source not in candidate_ids or edge.target not in candidate_ids:
            continue
        # edge: subject(source) -> dependency(target)
        # process dependency before subject: edge target -> source in topo
        if edge.source not in adjacency[edge.target]:
            adjacency[edge.target].add(edge.source)
            indegree[edge.source] += 1
    ready = deque(
        sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    )
    ordered_ids: list[str] = []
    while ready:
        current = ready.popleft()
        ordered_ids.append(current)
        for nxt in sorted(adjacency.get(current, ())):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if len(ordered_ids) != len(candidate_ids):
        # Residual cycles among capsule nodes are unexpected (graph rejects
        # unsafe mandatory cycles). Fall back to sorted node_id order.
        ordered_ids = sorted(candidate_ids)
    by_id = {node.node_id: node for node in candidates}
    return tuple(by_id[node_id] for node_id in ordered_ids)


def _project_capsules(
    graph: SemanticDependencyGraph,
) -> tuple[SemanticCapsuleRecord, ...]:
    if not isinstance(graph, SemanticDependencyGraph):
        raise SemanticCapsuleError("graph must be a SemanticDependencyGraph")
    candidates = _capsule_candidates(graph)
    if len(candidates) > DEFAULT_MAX_CAPSULES:
        raise CapsuleBoundsError("too many capsule candidates")
    dep_edges = _dependency_edges(graph)
    candidate_ids = {node.node_id for node in candidates}
    node_by_id = {node.node_id: node for node in graph.nodes}
    built: dict[str, SemanticCapsuleRecord] = {}

    for node in _topological_capsule_nodes(graph, candidates):
        kind = project_capsule_kind(node)
        assert kind is not None
        node_edges = [edge for edge in dep_edges if edge.source == node.node_id]
        dependency_node_ids: list[str] = []
        dependency_cids: list[str] = []
        edge_ids: list[str] = []
        unknown: list[str] = []
        for edge in node_edges:
            edge_ids.append(edge.edge_id)
            dependency_node_ids.append(edge.target)
            if edge.target not in node_by_id:
                unknown.append(edge.target)
                continue
            if edge.target not in candidate_ids:
                # Required mandatory edge to a non-capsule node is unknown for
                # reuse purposes (fail closed: never reuse on unknown dep).
                unknown.append(edge.target)
                continue
            dep_capsule = built.get(edge.target)
            if dep_capsule is None:
                unknown.append(edge.target)
            else:
                dependency_cids.append(dep_capsule.capsule_cid)
        evidence = _build_evidence(graph, node, node_edges)
        built[node.node_id] = SemanticCapsuleRecord(
            kind=kind,
            node_id=node.node_id,
            node_content_id=node.content_id,
            graph_id=graph.graph_id,
            root_id=node.root_id,
            source_root_id=node.source_root_id,
            authority=node.authority.value,
            trust=node.trust.value,
            provenance=node.provenance.value,
            version=node.version,
            evidence=evidence,
            dependency_node_ids=tuple(dependency_node_ids),
            dependency_cids=tuple(dependency_cids),
            edge_ids=tuple(edge_ids),
            unknown_dependency_refs=tuple(unknown),
        )
    return tuple(
        built[node_id] for node_id in sorted(built)
    )


def compile_semantic_capsule(
    graph: SemanticDependencyGraph,
    node_id: str,
) -> SemanticCapsuleRecord:
    """Compile the capsule for one node; unknown/unmapped nodes fail closed."""

    try:
        node = graph.node(node_id)
    except KeyError as exc:
        raise UnknownDependencyError(f"unknown graph node: {node_id}") from exc
    kind = project_capsule_kind(node)
    if kind is None:
        raise SemanticCapsuleError(
            f"node {node_id!r} does not project to a capsule"
        )
    for capsule in _project_capsules(graph):
        if capsule.node_id == node_id:
            return capsule
    raise SemanticCapsuleError(f"failed to project capsule for {node_id}")


# ---------------------------------------------------------------------------
# Impact / path explanations
# ---------------------------------------------------------------------------


def _depends_adjacency(
    capsules: Sequence[SemanticCapsuleRecord],
) -> tuple[
    dict[str, list[tuple[str, str]]],
    dict[str, list[tuple[str, str]]],
]:
    """Return (forward subject->deps, reverse dep->subjects) with edge kinds.

    Forward list entries are ``(dependency_cid, edge_kind_or_depends_on)``.
    Reverse list entries are ``(subject_cid, edge_kind)``.
    """

    by_node = {item.node_id: item for item in capsules}
    forward: dict[str, list[tuple[str, str]]] = {
        item.capsule_cid: [] for item in capsules
    }
    reverse: dict[str, list[tuple[str, str]]] = {
        item.capsule_cid: [] for item in capsules
    }
    for capsule in capsules:
        for dep_node_id in capsule.dependency_node_ids:
            dep = by_node.get(dep_node_id)
            if dep is None:
                continue
            # Without per-edge kind on the capsule, use depends_on.
            kind = SemanticEdgeKind.DEPENDS_ON.value
            forward[capsule.capsule_cid].append((dep.capsule_cid, kind))
            reverse.setdefault(dep.capsule_cid, []).append(
                (capsule.capsule_cid, kind)
            )
    for mapping in (forward, reverse):
        for key, values in mapping.items():
            mapping[key] = sorted(set(values))
    return forward, reverse


def _shortest_invalidation_path(
    seed_cid: str,
    subject_cid: str,
    reverse_adj: Mapping[str, Sequence[tuple[str, str]]],
    *,
    max_depth: int = DEFAULT_MAX_CLOSURE_DEPTH,
) -> CapsulePath:
    if seed_cid == subject_cid:
        return CapsulePath(capsule_cids=(seed_cid,), edge_kinds=())
    parent: dict[str, tuple[str, str]] = {}
    queue: deque[str] = deque((seed_cid,))
    seen = {seed_cid}
    depth = {seed_cid: 0}
    while queue:
        current = queue.popleft()
        if depth[current] >= max_depth:
            continue
        for nxt, edge_kind in reverse_adj.get(current, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (current, edge_kind)
            depth[nxt] = depth[current] + 1
            if nxt == subject_cid:
                queue.clear()
                break
            queue.append(nxt)
    if subject_cid not in parent and subject_cid != seed_cid:
        raise SemanticCapsuleError(
            f"no invalidation path from {seed_cid} to {subject_cid}"
        )
    nodes = [subject_cid]
    kinds: list[str] = []
    cursor = subject_cid
    while cursor != seed_cid:
        prev, edge_kind = parent[cursor]
        nodes.append(prev)
        kinds.append(edge_kind)
        cursor = prev
    nodes.reverse()
    kinds.reverse()
    return CapsulePath(capsule_cids=tuple(nodes), edge_kinds=tuple(kinds))


def _seed_cids_from_previous(
    previous: CapsuleIndex | None,
    current: CapsuleIndex,
) -> set[str]:
    if previous is None:
        return set()
    prev_by_node = previous.by_node_id()
    seeds: set[str] = set()
    for capsule in current.capsules:
        prior = prev_by_node.get(capsule.node_id)
        if prior is None:
            seeds.add(capsule.capsule_cid)
            continue
        if prior.node_content_id != capsule.node_content_id:
            seeds.add(capsule.capsule_cid)
        elif prior.capsule_cid != capsule.capsule_cid:
            seeds.add(capsule.capsule_cid)
    # Removed nodes: their previous CIDs are seeds for receipt demotion only.
    return seeds


def _content_changed_node_ids(
    previous: CapsuleIndex | None,
    current: CapsuleIndex,
) -> set[str]:
    if previous is None:
        return set()
    prev_by_node = previous.by_node_id()
    changed: set[str] = set()
    for capsule in current.capsules:
        prior = prev_by_node.get(capsule.node_id)
        if prior is None or prior.node_content_id != capsule.node_content_id:
            changed.add(capsule.node_id)
    return changed


def demote_stale_receipts(
    index: CapsuleIndex,
    receipts: Sequence[HistoricalReceipt | Mapping[str, Any]],
    *,
    invalidated_cids: Iterable[str] = (),
) -> tuple[HistoricalReceipt, ...]:
    """Demote receipts whose bound capsule is absent or invalidated."""

    live_cids = {item.capsule_cid for item in index.capsules}
    invalidated = set(invalidated_cids)
    assessed: list[HistoricalReceipt] = []
    for raw in receipts:
        receipt = (
            raw
            if isinstance(raw, HistoricalReceipt)
            else HistoricalReceipt.from_dict(raw)
        )
        stale = (
            receipt.bound_capsule_cid not in live_cids
            or receipt.bound_capsule_cid in invalidated
            or (
                receipt.produced_graph_id
                and receipt.produced_graph_id != index.graph_id
            )
        )
        status = ReceiptStatus.DEMOTED if stale else ReceiptStatus.LIVE
        assessed.append(
            HistoricalReceipt(
                receipt_id=receipt.receipt_id,
                kind=receipt.kind,
                bound_capsule_cid=receipt.bound_capsule_cid,
                produced_graph_id=receipt.produced_graph_id,
                status=status,
            )
        )
    return tuple(
        sorted(assessed, key=lambda item: (item.kind.value, item.receipt_id))
    )


def explain_path(
    result: CapsuleCompileResult,
    subject_cid: str,
) -> InvalidationExplanation:
    """Return the recorded minimal-path explanation for ``subject_cid``."""

    for explanation in result.explanations:
        if explanation.subject_cid == subject_cid:
            return explanation
    raise SemanticCapsuleError(
        f"no explanation recorded for subject {subject_cid}"
    )


def _derive_impact(
    previous: CapsuleIndex | None,
    current: CapsuleIndex,
    *,
    receipts: Sequence[HistoricalReceipt] = (),
    explicit_seed_node_ids: Iterable[str] | None = None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[InvalidationExplanation, ...],
    tuple[HistoricalReceipt, ...],
    DatalogEvaluationResult,
]:
    forward, reverse = _depends_adjacency(current.capsules)
    by_cid = current.by_cid()
    by_node = current.by_node_id()

    if explicit_seed_node_ids is not None:
        seed_cids = {
            by_node[node_id].capsule_cid
            for node_id in explicit_seed_node_ids
            if node_id in by_node
        }
    else:
        seed_cids = _seed_cids_from_previous(previous, current)

    # Datalog ground facts
    facts: list[DatalogAtom] = []
    for kind in _REQUIRED_VALIDATION_KINDS:
        facts.append(DatalogAtom("RequiredKind", (kind,)))
    for capsule in current.capsules:
        facts.append(
            DatalogAtom("Capsule", (capsule.capsule_cid, capsule.kind.value))
        )
        for dep_cid, _edge_kind in forward.get(capsule.capsule_cid, ()):
            facts.append(
                DatalogAtom("DependsOn", (capsule.capsule_cid, dep_cid))
            )
        for ref in capsule.unknown_dependency_refs:
            facts.append(
                DatalogAtom("UnknownDep", (capsule.capsule_cid, ref))
            )
    for seed in sorted(seed_cids):
        facts.append(DatalogAtom("Changed", (seed,)))

    prev_cids = (
        {item.capsule_cid for item in previous.capsules}
        if previous is not None
        else set()
    )
    current_cids = {item.capsule_cid for item in current.capsules}
    for receipt in receipts:
        facts.append(
            DatalogAtom(
                "BindsReceipt", (receipt.receipt_id, receipt.bound_capsule_cid)
            )
        )
        if receipt.bound_capsule_cid not in current_cids:
            facts.append(
                DatalogAtom("AbsentCapsule", (receipt.bound_capsule_cid,))
            )

    evaluator = HermeticReferenceEvaluator()
    datalog = evaluator.evaluate(facts, default_capsule_datalog_rules())

    invalidated: set[str] = set()
    for row in datalog.facts("Invalidates"):
        if len(row) >= 2:
            invalidated.add(row[1])
        elif len(row) == 1:
            invalidated.add(row[0])

    # Close via BFS over reverse DependsOn (dependents of seeds).
    queue = deque(sorted(seed_cids))
    reached = set(seed_cids)
    while queue:
        current_cid = queue.popleft()
        if len(reached) > DEFAULT_MAX_CLOSURE_NODES:
            raise CapsuleBoundsError("invalidation closure exceeds max nodes")
        for nxt, _kind in reverse.get(current_cid, ()):
            if nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    invalidated |= reached

    # Required validations that impacted subjects depend on must revalidate.
    required_extra: set[str] = set()
    for subject_cid in sorted(invalidated):
        for dep_cid, _kind in forward.get(subject_cid, ()):
            dep = by_cid.get(dep_cid)
            if dep is not None and dep.kind.value in _REQUIRED_VALIDATION_KINDS:
                required_extra.add(dep_cid)
    invalidated |= required_extra
    for row in datalog.facts("RequiresRevalidation"):
        if row:
            invalidated.add(row[0])

    no_reuse = {row[0] for row in datalog.facts("NoReuse") if row}
    no_reuse |= {
        capsule.capsule_cid
        for capsule in current.capsules
        if capsule.has_unknown_dependency
    }
    no_reuse |= invalidated

    reused: set[str] = set()
    explanations: list[InvalidationExplanation] = []

    # Seed self-explanations
    for seed in sorted(seed_cids):
        if seed not in by_cid:
            continue
        explanations.append(
            InvalidationExplanation(
                action=CapsuleAction.INVALIDATE,
                subject_cid=seed,
                seed_cid=seed,
                path=CapsulePath(capsule_cids=(seed,)),
                rule_id=CapsuleRuleId.SEEDED_CHANGE.value,
                reason="seeded content_id change",
            )
        )

    for subject in sorted(invalidated - seed_cids):
        # Prefer a seed that reaches the subject via reverse DependsOn.
        chosen_seed = None
        chosen_path = None
        for seed in sorted(seed_cids):
            try:
                path = _shortest_invalidation_path(seed, subject, reverse)
            except SemanticCapsuleError:
                continue
            if chosen_path is None or (
                len(path.capsule_cids),
                path.capsule_cids,
            ) < (len(chosen_path.capsule_cids), chosen_path.capsule_cids):
                chosen_seed = seed
                chosen_path = path

        # Required proof/test/release deps of an impacted subject: path is
        # seed -> ... -> impacted_subject -> required_validation.
        if chosen_path is None and subject in required_extra:
            for impacted in sorted(invalidated - required_extra):
                deps = {dep for dep, _kind in forward.get(impacted, ())}
                if subject not in deps:
                    continue
                for seed in sorted(seed_cids):
                    try:
                        prefix = _shortest_invalidation_path(
                            seed, impacted, reverse
                        )
                    except SemanticCapsuleError:
                        if seed == impacted:
                            prefix = CapsulePath(capsule_cids=(seed,))
                        else:
                            continue
                    candidate = CapsulePath(
                        capsule_cids=prefix.capsule_cids + (subject,),
                        edge_kinds=prefix.edge_kinds
                        + (SemanticEdgeKind.DEPENDS_ON.value,),
                    )
                    if chosen_path is None or (
                        len(candidate.capsule_cids),
                        candidate.capsule_cids,
                    ) < (
                        len(chosen_path.capsule_cids),
                        chosen_path.capsule_cids,
                    ):
                        chosen_seed = seed
                        chosen_path = candidate

        if chosen_seed is None or chosen_path is None:
            # Fall back to self path when subject is newly introduced.
            chosen_seed = subject
            chosen_path = CapsulePath(capsule_cids=(subject,))
        rule = CapsuleRuleId.TRANSITIVE_INVALIDATION.value
        if by_cid[subject].kind.value in _REQUIRED_VALIDATION_KINDS:
            rule = CapsuleRuleId.REQUIRED_VALIDATION.value
        explanations.append(
            InvalidationExplanation(
                action=CapsuleAction.INVALIDATE,
                subject_cid=subject,
                seed_cid=chosen_seed,
                path=chosen_path,
                rule_id=rule,
                reason="seeded semantic change requires revalidation",
            )
        )

    for capsule in current.capsules:
        cid = capsule.capsule_cid
        if cid in no_reuse:
            if capsule.has_unknown_dependency and cid not in invalidated:
                explanations.append(
                    InvalidationExplanation(
                        action=CapsuleAction.INVALIDATE,
                        subject_cid=cid,
                        seed_cid=cid,
                        path=CapsulePath(capsule_cids=(cid,)),
                        rule_id=CapsuleRuleId.UNKNOWN_DEPENDENCY.value,
                        reason="unknown required dependency refuses reuse",
                    )
                )
            continue
        # Reusable: present in previous with identical CID, or first compile.
        if previous is not None and cid in prev_cids:
            reused.add(cid)
            explanations.append(
                InvalidationExplanation(
                    action=CapsuleAction.REUSE,
                    subject_cid=cid,
                    seed_cid=cid,
                    path=CapsulePath(capsule_cids=(cid,)),
                    rule_id=CapsuleRuleId.INPUTS_UNCHANGED.value,
                    reason="inputs_unchanged",
                )
            )
        elif previous is None:
            # Cold compile: treat unchanged-self as reuse-equivalent identity.
            reused.add(cid)
            explanations.append(
                InvalidationExplanation(
                    action=CapsuleAction.REUSE,
                    subject_cid=cid,
                    seed_cid=cid,
                    path=CapsulePath(capsule_cids=(cid,)),
                    rule_id=CapsuleRuleId.INPUTS_UNCHANGED.value,
                    reason="clean_rebuild_identity",
                )
            )

    demoted = demote_stale_receipts(
        current, receipts, invalidated_cids=invalidated
    )
    for receipt in demoted:
        if receipt.status is not ReceiptStatus.DEMOTED:
            continue
        seed = (
            receipt.bound_capsule_cid
            if receipt.bound_capsule_cid in by_cid
            or receipt.bound_capsule_cid in invalidated
            else receipt.bound_capsule_cid
        )
        # Demotion explanations bind the receipt id as subject surrogate via
        # reason text; path uses bound capsule when present in index, else a
        # single-node synthetic path on the bound cid string.
        path_cid = (
            receipt.bound_capsule_cid
            if receipt.bound_capsule_cid in by_cid
            else receipt.bound_capsule_cid
        )
        explanations.append(
            InvalidationExplanation(
                action=CapsuleAction.DEMOTE,
                subject_cid=path_cid,
                seed_cid=path_cid,
                path=CapsulePath(capsule_cids=(path_cid,)),
                rule_id=CapsuleRuleId.STALE_HISTORICAL_RECEIPT.value,
                reason=f"stale_historical_receipt:{receipt.receipt_id}",
            )
        )

    # Deduplicate explanations by (action, subject, seed, rule).
    dedup: dict[tuple[str, str, str, str], InvalidationExplanation] = {}
    for item in explanations:
        key = (
            item.action.value,
            item.subject_cid,
            item.seed_cid,
            item.rule_id,
        )
        prior = dedup.get(key)
        if prior is None or (
            len(item.path.capsule_cids),
            item.path.capsule_cids,
        ) < (len(prior.path.capsule_cids), prior.path.capsule_cids):
            dedup[key] = item
    ordered_explanations = tuple(
        sorted(
            dedup.values(),
            key=lambda item: (
                item.action.value,
                item.subject_cid,
                item.seed_cid,
                item.rule_id,
            ),
        )
    )

    # invalidated_cids reported to callers: only those whose CID changed vs
    # previous, or required validations newly requiring revalidation.
    reported_invalidated = set(invalidated)
    if previous is not None:
        prev_by_node = previous.by_node_id()
        # Capsules whose node content changed necessarily have new CIDs; the
        # previous CIDs are the ones "invalidated" historically.
        for node_id in _content_changed_node_ids(previous, current):
            prior = prev_by_node.get(node_id)
            if prior is not None:
                reported_invalidated.add(prior.capsule_cid)
            current_capsule = by_node.get(node_id)
            if current_capsule is not None:
                reported_invalidated.add(current_capsule.capsule_cid)

    return (
        tuple(sorted(reused)),
        tuple(sorted(reported_invalidated)),
        ordered_explanations,
        tuple(
            item for item in demoted if item.status is ReceiptStatus.DEMOTED
        ),
        datalog,
    )


# ---------------------------------------------------------------------------
# Public compile / update entry points
# ---------------------------------------------------------------------------


def compile_semantic_capsules(
    graph: SemanticDependencyGraph,
    *,
    previous: CapsuleCompileResult | CapsuleIndex | None = None,
    receipts: Sequence[HistoricalReceipt | Mapping[str, Any]] = (),
    seed_node_ids: Iterable[str] | None = None,
) -> CapsuleCompileResult:
    """Project capsules from ``graph``.

    When ``previous`` is supplied, reuse/invalidation explanations are derived
    against that snapshot. A clean rebuild (``previous is None``) and an
    incremental update against an identical graph share the same ``index_cid``.
    """

    capsules = _project_capsules(graph)
    index = CapsuleIndex(graph_id=graph.graph_id, capsules=capsules)
    prev_index: CapsuleIndex | None
    if previous is None:
        prev_index = None
    elif isinstance(previous, CapsuleCompileResult):
        prev_index = previous.index
    elif isinstance(previous, CapsuleIndex):
        prev_index = previous
    else:
        raise SemanticCapsuleError("previous must be a CapsuleCompileResult or CapsuleIndex")

    normalized_receipts = tuple(
        item
        if isinstance(item, HistoricalReceipt)
        else HistoricalReceipt.from_dict(item)
        for item in receipts
    )
    reused, invalidated, explanations, demoted, datalog = _derive_impact(
        prev_index,
        index,
        receipts=normalized_receipts,
        explicit_seed_node_ids=seed_node_ids,
    )

    # Incremental reuse diagnostic: when previous exists and CID unchanged.
    if prev_index is not None and seed_node_ids is None:
        prev_cids = {item.capsule_cid for item in prev_index.capsules}
        reused = tuple(
            sorted(
                cid
                for cid in (item.capsule_cid for item in index.capsules)
                if cid in prev_cids and cid not in invalidated
            )
        )

    return CapsuleCompileResult(
        index=index,
        explanations=explanations,
        reused_cids=reused,
        invalidated_cids=invalidated,
        demoted_receipts=demoted,
        datalog=datalog,
    )


def update_semantic_capsules(
    previous: CapsuleCompileResult | CapsuleIndex,
    graph: SemanticDependencyGraph,
    *,
    receipts: Sequence[HistoricalReceipt | Mapping[str, Any]] = (),
    seed_node_ids: Iterable[str] | None = None,
) -> CapsuleCompileResult:
    """Incremental update. Result index must equal a clean rebuild of ``graph``."""

    return compile_semantic_capsules(
        graph,
        previous=previous,
        receipts=receipts,
        seed_node_ids=seed_node_ids,
    )


def invalidate_capsules(
    previous: CapsuleCompileResult | CapsuleIndex,
    graph: SemanticDependencyGraph,
    *,
    seeds: Iterable[str],
    receipts: Sequence[HistoricalReceipt | Mapping[str, Any]] = (),
) -> CapsuleCompileResult:
    """Recompile ``graph`` and mark invalidation from explicit seed node ids."""

    return compile_semantic_capsules(
        graph,
        previous=previous,
        receipts=receipts,
        seed_node_ids=tuple(seeds),
    )


def verify_capsule_compile_result(
    result: CapsuleCompileResult | Mapping[str, Any],
) -> CapsuleCompileResult:
    """Recompute every claimed CID; reject forged or non-canonical payloads."""

    if isinstance(result, Mapping):
        index = CapsuleIndex.from_dict(result.get("index") or {})
        explanations = tuple(
            InvalidationExplanation.from_dict(item)
            for item in (result.get("explanations") or ())
        )
        restored = CapsuleCompileResult(
            index=index,
            explanations=explanations,
            reused_cids=tuple(result.get("reused_cids") or ()),
            invalidated_cids=tuple(result.get("invalidated_cids") or ()),
            demoted_receipts=tuple(
                HistoricalReceipt.from_dict(item)
                for item in (result.get("demoted_receipts") or ())
            ),
        )
    else:
        restored = CapsuleCompileResult(
            index=CapsuleIndex.from_dict(result.index.to_dict()),
            explanations=tuple(
                InvalidationExplanation.from_dict(item.to_dict())
                for item in result.explanations
            ),
            reused_cids=result.reused_cids,
            invalidated_cids=result.invalidated_cids,
            demoted_receipts=tuple(
                HistoricalReceipt.from_dict(item.to_dict())
                for item in result.demoted_receipts
            ),
            datalog=result.datalog,
        )
    # Round-trip every capsule.
    for capsule in restored.capsules:
        again = SemanticCapsuleRecord.from_dict(capsule.to_dict())
        if again.capsule_cid != capsule.capsule_cid:
            raise SemanticCapsuleError(
                f"capsule {capsule.node_id} failed reverify"
            )
    return restored


def capsules_from_graph(
    graph: SemanticDependencyGraph,
    **kwargs: Any,
) -> CapsuleCompileResult:
    """Alias for :func:`compile_semantic_capsules`."""

    return compile_semantic_capsules(graph, **kwargs)


def require_live_receipt(receipt: HistoricalReceipt) -> HistoricalReceipt:
    """Fail closed when a demoted receipt is used as live authority."""

    if receipt.status is ReceiptStatus.DEMOTED:
        raise StaleReceiptError(
            f"receipt {receipt.receipt_id} is demoted and cannot authorize reuse"
        )
    return receipt


def assert_reuse_allowed(capsule: SemanticCapsuleRecord) -> None:
    """Fail closed when reuse is requested for an unknown-dependency capsule."""

    if capsule.has_unknown_dependency:
        raise UnknownDependencyError(
            f"capsule {capsule.node_id} has unknown dependencies "
            f"{list(capsule.unknown_dependency_refs)}; reuse refused"
        )


__all__ = [
    "ANALYZER_VERSION",
    "BUNDLE",
    "DEFAULT_MAX_CAPSULES",
    "EVIDENCE_FIELDS",
    "EVIDENCE_SCHEMA",
    "GOAL_ID",
    "HERMETIC_EVALUATOR_ID",
    "INDEX_SCHEMA",
    "INVALIDATION_SCHEMA",
    "SCHEMA",
    "TASK_ID",
    "CapsuleAction",
    "CapsuleBoundsError",
    "CapsuleCompileResult",
    "CapsuleEvidence",
    "CapsuleIndex",
    "CapsuleKind",
    "CapsulePath",
    "CapsuleRuleId",
    "DatalogAtom",
    "DatalogEvaluationResult",
    "DatalogRule",
    "HermeticReferenceEvaluator",
    "HistoricalReceipt",
    "InvalidationExplanation",
    "ReceiptKind",
    "ReceiptStatus",
    "SemanticCapsuleError",
    "SemanticCapsuleRecord",
    "StaleReceiptError",
    "UnknownDependencyError",
    "assert_reuse_allowed",
    "capsules_from_graph",
    "compile_semantic_capsule",
    "compile_semantic_capsules",
    "default_capsule_datalog_rules",
    "demote_stale_receipts",
    "explain_path",
    "invalidate_capsules",
    "project_capsule_kind",
    "require_live_receipt",
    "update_semantic_capsules",
    "verify_capsule_compile_result",
]
