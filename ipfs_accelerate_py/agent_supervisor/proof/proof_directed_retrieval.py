"""Proof-directed retrieval receipts with deterministic authority closure.

Approximate retrieval in this module is deliberately advisory.  BM25, vector,
AST, and GraphRAG providers may nominate compact node identifiers, but every
nomination is checked against one exact graph/index/model/configuration
snapshot and partition.  The mandatory authority and proof population is
always computed independently by following the semantic graph's authoritative
typed edges to a fixed point.

Consequently an unavailable index, poisoned embedding, stale neighbor,
cross-partition hit, provider disagreement, or top-k truncation can remove only
optional explanatory evidence.  A mandatory closure that exceeds its graph or
receipt budget raises a typed fail-closed error instead of returning a partial
receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..analysis.analysis_retrieval import (
    BoundRetrievalCandidate,
    RetrievalBindingError,
    RetrievalSnapshotBinding,
    validate_bound_retrieval_candidate,
)
from ..context.decision_contracts import DecisionRequest
from ..analysis.semantic_dependency_graph import (
    ClosureBounds,
    MandatoryClosure,
    SemanticDependencyGraph,
    SemanticGraphBoundsError,
    SemanticNode,
    SemanticNodeKind,
    canonical_semantic_json,
)


PROOF_DIRECTED_RETRIEVAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-directed-retrieval-receipt@1"
)
RETRIEVAL_SEED_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/retrieval-seed@1"
)
RETRIEVAL_CANDIDATE_AUDIT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/retrieval-candidate-audit@1"
)
RETRIEVAL_CLOSURE_REQUIREMENT_ID = (
    "agent-supervisor.requirement.retrieval-authoritative-closure@1"
)

APPROXIMATE_SOURCES = ("ast", "bm25", "graphrag", "vector")
_MAX_TEXT_BYTES = 8_192


class ProofDirectedRetrievalError(ValueError):
    """A retrieval input or authoritative result violates the contract."""


class ProofDirectedRetrievalBudgetError(ProofDirectedRetrievalError):
    """A complete mandatory closure or its receipt cannot fit a hard budget."""


class MissingRequiredIndexError(ProofDirectedRetrievalError):
    """A required advisory index has neither an exact fallback nor a result."""


class StaleRetrievalRootError(ProofDirectedRetrievalError):
    """The decision, graph, or declared snapshot roots do not agree."""


class CandidateDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TRUNCATED = "truncated"


class RetrievalBackendState(str, Enum):
    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"
    UNHEALTHY = "unhealthy"
    EXACT_FALLBACK = "exact_fallback"


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise ProofDirectedRetrievalError("retrieval receipt exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProofDirectedRetrievalError(
                "retrieval receipt cannot contain non-finite values"
            )
        return format(value, ".17g")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProofDirectedRetrievalError(
                "retrieval receipt mapping keys must be strings"
            )
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_plain(item, depth=depth + 1) for item in value]
        return sorted(normalized, key=_canonical_json)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise ProofDirectedRetrievalError(
        f"unsupported retrieval receipt value: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ProofDirectedRetrievalError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise ProofDirectedRetrievalError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise ProofDirectedRetrievalError(f"{name} is required")
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ProofDirectedRetrievalError(f"{name} is oversized")
    return value


def _strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({_text(str(item), "identity") for item in values if str(item)}))


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise ProofDirectedRetrievalError(
        f"retrieval record must be a mapping, got {type(value).__name__}"
    )


def _deep_string_values(value: Any, *, depth: int = 0) -> frozenset[str]:
    if depth > 12:
        return frozenset()
    if isinstance(value, str):
        return frozenset((value,))
    if isinstance(value, Mapping):
        result: set[str] = set()
        for item in value.values():
            result.update(_deep_string_values(item, depth=depth + 1))
        return frozenset(result)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        result = set()
        for item in value:
            result.update(_deep_string_values(item, depth=depth + 1))
        return frozenset(result)
    return frozenset()


def embedding_fingerprint(
    embedding: Sequence[Any],
    *,
    model_id: str,
    configuration_id: str,
) -> str:
    """Hash a finite vector and its exact producer bindings without exposing it."""

    values: list[str] = []
    for item in embedding:
        if isinstance(item, bool):
            raise ProofDirectedRetrievalError("embedding values must be finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ProofDirectedRetrievalError(
                "embedding values must be finite numbers"
            ) from exc
        if not math.isfinite(number):
            raise ProofDirectedRetrievalError(
                "embedding values must be finite numbers"
            )
        values.append(format(number, ".17g"))
    return _identity(
        "embedding",
        {
            "model_id": _text(model_id, "embedding model_id"),
            "configuration_id": _text(
                configuration_id, "embedding configuration_id"
            ),
            "dimensions": len(values),
            "values": values,
        },
    )


@dataclass(frozen=True)
class ProofRetrievalBudget:
    max_candidates: int
    max_optional_nodes: int
    max_graph_nodes: int
    max_graph_edges: int
    max_graph_depth: int
    max_receipt_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "max_candidates",
            "max_optional_nodes",
            "max_graph_nodes",
            "max_graph_edges",
            "max_graph_depth",
            "max_receipt_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProofDirectedRetrievalBudgetError(
                    f"{name} must be a positive integer"
                )

    @classmethod
    def from_decision(cls, request: DecisionRequest) -> "ProofRetrievalBudget":
        return cls(
            max_candidates=request.budget.max_retrieval_results,
            max_optional_nodes=request.budget.max_retrieval_results,
            max_graph_nodes=request.budget.max_items,
            max_graph_edges=min(
                request.budget.max_items * 16,
                250_000,
            ),
            max_graph_depth=request.budget.max_graph_hops,
            max_receipt_bytes=request.budget.max_serialized_bytes,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_optional_nodes": self.max_optional_nodes,
            "max_graph_nodes": self.max_graph_nodes,
            "max_graph_edges": self.max_graph_edges,
            "max_graph_depth": self.max_graph_depth,
            "max_receipt_bytes": self.max_receipt_bytes,
        }


@dataclass(frozen=True)
class RetrievalSeed:
    selector_kind: str
    value: str
    matched_node_ids: tuple[str, ...] = ()
    mandatory: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selector_kind", _text(self.selector_kind, "seed selector_kind")
        )
        object.__setattr__(self, "value", _text(self.value, "seed value"))
        object.__setattr__(
            self, "matched_node_ids", _strings(self.matched_node_ids)
        )
        if not isinstance(self.mandatory, bool):
            raise ProofDirectedRetrievalError("seed mandatory must be a boolean")

    @property
    def seed_id(self) -> str:
        return _identity(
            "retrieval-seed",
            self.to_dict(include_seed_id=False),
        )

    def to_dict(self, *, include_seed_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": RETRIEVAL_SEED_SCHEMA,
            "selector_kind": self.selector_kind,
            "value": self.value,
            "matched_node_ids": list(self.matched_node_ids),
            "mandatory": self.mandatory,
            "source": "decision_request",
        }
        if include_seed_id:
            payload["seed_id"] = self.seed_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetrievalSeed":
        unknown = set(payload).difference(
            {
                "schema",
                "seed_id",
                "selector_kind",
                "value",
                "matched_node_ids",
                "mandatory",
                "source",
            }
        )
        if unknown:
            raise ProofDirectedRetrievalError(
                "retrieval seed contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        if payload.get("schema") not in (None, RETRIEVAL_SEED_SCHEMA):
            raise ProofDirectedRetrievalError("unsupported retrieval seed schema")
        if payload.get("source") not in (None, "decision_request"):
            raise ProofDirectedRetrievalError(
                "retrieval seeds must be derived from the DecisionRequest"
            )
        result = cls(
            selector_kind=payload.get("selector_kind", ""),
            value=payload.get("value", ""),
            matched_node_ids=tuple(payload.get("matched_node_ids") or ()),
            mandatory=payload.get("mandatory", False),
        )
        claimed = payload.get("seed_id")
        if claimed not in (None, result.seed_id):
            raise ProofDirectedRetrievalError("retrieval seed identity mismatch")
        return result


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    node_id: str
    source: str
    disposition: CandidateDisposition
    score_millionths: int | None
    rank: int
    binding_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        for name, required in (
            ("candidate_id", True),
            ("node_id", False),
            ("source", True),
            ("binding_id", False),
            ("reason", False),
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), f"candidate audit {name}", required=required),
            )
        object.__setattr__(
            self, "disposition", CandidateDisposition(self.disposition)
        )
        if self.score_millionths is not None and (
            isinstance(self.score_millionths, bool)
            or not isinstance(self.score_millionths, int)
            or not 0 <= self.score_millionths <= 1_000_000
        ):
            raise ProofDirectedRetrievalError("candidate audit score is invalid")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ProofDirectedRetrievalError("candidate audit rank is invalid")
        if self.disposition is CandidateDisposition.ACCEPTED and self.reason:
            raise ProofDirectedRetrievalError(
                "accepted candidates cannot carry a rejection reason"
            )
        if self.disposition is not CandidateDisposition.ACCEPTED and not self.reason:
            raise ProofDirectedRetrievalError(
                "rejected or truncated candidates require a reason"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RETRIEVAL_CANDIDATE_AUDIT_SCHEMA,
            "candidate_id": self.candidate_id,
            "node_id": self.node_id,
            "source": self.source,
            "disposition": self.disposition.value,
            "score_millionths": self.score_millionths,
            "rank": self.rank,
            "binding_id": self.binding_id,
            "reason": self.reason,
            "authority": "context_only",
            "proof_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateAudit":
        unknown = set(payload).difference(
            {
                "schema",
                "candidate_id",
                "node_id",
                "source",
                "disposition",
                "score_millionths",
                "rank",
                "binding_id",
                "reason",
                "authority",
                "proof_authority",
            }
        )
        if unknown:
            raise ProofDirectedRetrievalError(
                "candidate audit contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        if payload.get("schema") not in (
            None,
            RETRIEVAL_CANDIDATE_AUDIT_SCHEMA,
        ):
            raise ProofDirectedRetrievalError(
                "unsupported retrieval candidate audit schema"
            )
        if payload.get("proof_authority") not in (None, False):
            raise ProofDirectedRetrievalError(
                "candidate audit cannot claim proof authority"
            )
        return cls(
            candidate_id=payload.get("candidate_id", ""),
            node_id=payload.get("node_id", ""),
            source=payload.get("source", ""),
            disposition=payload.get("disposition", ""),
            score_millionths=payload.get("score_millionths"),
            rank=payload.get("rank", 0),
            binding_id=payload.get("binding_id", ""),
            reason=payload.get("reason", ""),
        )


@dataclass(frozen=True)
class ProofDirectedRetrievalReceipt:
    decision_request_id: str
    query: Mapping[str, Any]
    roots: Mapping[str, Any]
    snapshot: RetrievalSnapshotBinding
    budgets: ProofRetrievalBudget
    seeds: tuple[RetrievalSeed, ...]
    candidates: tuple[CandidateAudit, ...]
    closure_id: str
    closure_node_ids: tuple[str, ...]
    closure_edge_ids: tuple[str, ...]
    paths: Mapping[str, tuple[str, ...]]
    included_node_ids: tuple[str, ...]
    optional_node_ids: tuple[str, ...]
    omitted_node_ids: tuple[str, ...]
    truncation: Mapping[str, Any]
    disagreement: tuple[str, ...]
    fallback: tuple[str, ...]
    backend_states: Mapping[str, str]
    fixed_point_iterations: int
    closure_fixed_point: bool = True
    closure_complete: bool = True
    schema: str = PROOF_DIRECTED_RETRIEVAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_request_id",
            _text(self.decision_request_id, "decision_request_id"),
        )
        object.__setattr__(self, "closure_id", _text(self.closure_id, "closure_id"))
        if not isinstance(self.snapshot, RetrievalSnapshotBinding):
            object.__setattr__(
                self,
                "snapshot",
                RetrievalSnapshotBinding.from_dict(self.snapshot),
            )
        if not isinstance(self.budgets, ProofRetrievalBudget):
            if not isinstance(self.budgets, Mapping):
                raise ProofDirectedRetrievalError("receipt budgets are invalid")
            object.__setattr__(self, "budgets", ProofRetrievalBudget(**self.budgets))
        object.__setattr__(self, "query", MappingProxyType(_plain(self.query)))
        object.__setattr__(self, "roots", MappingProxyType(_plain(self.roots)))
        normalized_seeds = tuple(
            item
            if isinstance(item, RetrievalSeed)
            else RetrievalSeed.from_dict(item)
            for item in self.seeds
        )
        object.__setattr__(
            self,
            "seeds",
            tuple(
                sorted(
                    normalized_seeds,
                    key=lambda item: (
                        item.selector_kind,
                        item.value,
                        item.seed_id,
                    ),
                )
            ),
        )
        normalized_candidates = tuple(
            item
            if isinstance(item, CandidateAudit)
            else CandidateAudit.from_dict(item)
            for item in self.candidates
        )
        object.__setattr__(
            self,
            "candidates",
            tuple(
                sorted(
                    normalized_candidates,
                    key=lambda item: (
                        item.source,
                        item.rank,
                        item.candidate_id,
                    ),
                )
            ),
        )
        for name in (
            "closure_node_ids",
            "closure_edge_ids",
            "included_node_ids",
            "optional_node_ids",
            "omitted_node_ids",
            "disagreement",
            "fallback",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        canonical_paths = {
            str(key): tuple(str(item) for item in value)
            for key, value in sorted(self.paths.items())
        }
        object.__setattr__(self, "paths", MappingProxyType(canonical_paths))
        object.__setattr__(
            self, "truncation", MappingProxyType(_plain(self.truncation))
        )
        states = {
            _text(str(key), "backend name"): RetrievalBackendState(value).value
            for key, value in sorted(self.backend_states.items())
        }
        object.__setattr__(self, "backend_states", MappingProxyType(states))

        closure_nodes = set(self.closure_node_ids)
        included = set(self.included_node_ids)
        optional = set(self.optional_node_ids)
        omitted = set(self.omitted_node_ids)
        if not self.closure_complete or not self.closure_fixed_point:
            raise ProofDirectedRetrievalError(
                "proof-directed receipts require a complete closure fixed point"
            )
        if not closure_nodes or not closure_nodes.issubset(included):
            raise ProofDirectedRetrievalError(
                "included nodes must contain every mandatory closure node"
            )
        if closure_nodes.intersection(omitted):
            raise ProofDirectedRetrievalError(
                "a mandatory dependency cannot be listed as omitted"
            )
        if optional.intersection(closure_nodes) or not optional.issubset(included):
            raise ProofDirectedRetrievalError(
                "optional nodes must be included and outside mandatory closure"
            )
        if optional.intersection(omitted):
            raise ProofDirectedRetrievalError(
                "an optional node cannot be both included and omitted"
            )
        if set(canonical_paths) != closure_nodes:
            raise ProofDirectedRetrievalError(
                "receipt paths must cover exactly the mandatory closure"
            )
        path_roots = {
            path[0]
            for path in canonical_paths.values()
            if path
        }
        for node_id, path in canonical_paths.items():
            if (
                not path
                or path[-1] != node_id
                or len(path) != len(set(path))
            ):
                raise ProofDirectedRetrievalError(
                    f"invalid mandatory closure path for {node_id}"
                )
        if len(path_roots) != 1:
            raise ProofDirectedRetrievalError(
                "mandatory closure paths must share one decision root"
            )
        if (
            isinstance(self.fixed_point_iterations, bool)
            or not isinstance(self.fixed_point_iterations, int)
            or self.fixed_point_iterations < 1
        ):
            raise ProofDirectedRetrievalError(
                "fixed_point_iterations must be a positive integer"
            )
        expected_iterations = max(
            (len(path) for path in canonical_paths.values()),
            default=1,
        )
        if self.fixed_point_iterations != expected_iterations:
            raise ProofDirectedRetrievalError(
                "fixed_point_iterations does not match closure paths"
            )
        graph_root = str(self.roots.get("semantic_graph_root_id") or "")
        if not graph_root:
            raise ProofDirectedRetrievalError(
                "receipt roots omit semantic_graph_root_id"
            )
        reconstructed = MandatoryClosure(
            root_id=graph_root,
            decision_id=next(iter(path_roots)),
            node_ids=self.closure_node_ids,
            edge_ids=self.closure_edge_ids,
            paths=canonical_paths,
            bounds=ClosureBounds(
                max_nodes=self.budgets.max_graph_nodes,
                max_edges=self.budgets.max_graph_edges,
                max_depth=self.budgets.max_graph_depth,
                max_annotations=1,
            ),
        )
        if reconstructed.closure_id != self.closure_id:
            raise ProofDirectedRetrievalError(
                "receipt closure identity does not match its fixed point"
            )

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def proof_authority(self) -> bool:
        return False

    @property
    def receipt_id(self) -> str:
        return _identity(
            "proof-directed-retrieval",
            self.to_dict(include_receipt_id=False),
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "requirement_id": RETRIEVAL_CLOSURE_REQUIREMENT_ID,
            "decision_request_id": self.decision_request_id,
            "query": _plain(self.query),
            "roots": _plain(self.roots),
            "snapshot": self.snapshot.to_dict(),
            "budgets": self.budgets.to_dict(),
            "seeds": [item.to_dict() for item in self.seeds],
            "candidates": [item.to_dict() for item in self.candidates],
            "closure": {
                "closure_id": self.closure_id,
                "node_ids": list(self.closure_node_ids),
                "edge_ids": list(self.closure_edge_ids),
                "paths": {
                    key: list(value) for key, value in self.paths.items()
                },
                "fixed_point_iterations": self.fixed_point_iterations,
                "fixed_point": self.closure_fixed_point,
                "complete": self.closure_complete,
                "truncated": False,
            },
            "included_node_ids": list(self.included_node_ids),
            "optional_node_ids": list(self.optional_node_ids),
            "omitted_node_ids": list(self.omitted_node_ids),
            "truncation": _plain(self.truncation),
            "disagreement": list(self.disagreement),
            "fallback": list(self.fallback),
            "backend_states": dict(self.backend_states),
            "authority": "context_only",
            "proof_authority": False,
            "completion_authority": False,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return _canonical_json(self.to_dict())
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ProofDirectedRetrievalReceipt":
        allowed = {
            "schema",
            "requirement_id",
            "receipt_id",
            "decision_request_id",
            "query",
            "roots",
            "snapshot",
            "budgets",
            "seeds",
            "candidates",
            "closure",
            "included_node_ids",
            "optional_node_ids",
            "omitted_node_ids",
            "truncation",
            "disagreement",
            "fallback",
            "backend_states",
            "authority",
            "proof_authority",
            "completion_authority",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise ProofDirectedRetrievalError(
                "retrieval receipt contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        if payload.get("schema") != PROOF_DIRECTED_RETRIEVAL_SCHEMA:
            raise ProofDirectedRetrievalError(
                "unsupported proof-directed retrieval receipt schema"
            )
        if payload.get("proof_authority") not in (None, False):
            raise ProofDirectedRetrievalError(
                "retrieval receipt cannot claim proof authority"
            )
        if payload.get("completion_authority") not in (None, False):
            raise ProofDirectedRetrievalError(
                "retrieval receipt cannot claim completion authority"
            )
        if payload.get("authority") not in (None, "context_only"):
            raise ProofDirectedRetrievalError(
                "retrieval receipt authority must be context_only"
            )
        if payload.get("requirement_id") not in (
            None,
            RETRIEVAL_CLOSURE_REQUIREMENT_ID,
        ):
            raise ProofDirectedRetrievalError(
                "retrieval receipt requirement identity mismatch"
            )
        closure = payload.get("closure")
        if not isinstance(closure, Mapping):
            raise ProofDirectedRetrievalError("receipt closure must be an object")
        result = cls(
            decision_request_id=payload.get("decision_request_id", ""),
            query=payload.get("query") or {},
            roots=payload.get("roots") or {},
            snapshot=RetrievalSnapshotBinding.from_dict(
                payload.get("snapshot") or {}
            ),
            budgets=ProofRetrievalBudget(**dict(payload.get("budgets") or {})),
            seeds=tuple(
                RetrievalSeed.from_dict(item)
                for item in payload.get("seeds") or ()
            ),
            candidates=tuple(
                CandidateAudit.from_dict(item)
                for item in payload.get("candidates") or ()
            ),
            closure_id=closure.get("closure_id", ""),
            closure_node_ids=tuple(closure.get("node_ids") or ()),
            closure_edge_ids=tuple(closure.get("edge_ids") or ()),
            paths={
                str(key): tuple(value)
                for key, value in (closure.get("paths") or {}).items()
            },
            included_node_ids=tuple(payload.get("included_node_ids") or ()),
            optional_node_ids=tuple(payload.get("optional_node_ids") or ()),
            omitted_node_ids=tuple(payload.get("omitted_node_ids") or ()),
            truncation=payload.get("truncation") or {},
            disagreement=tuple(payload.get("disagreement") or ()),
            fallback=tuple(payload.get("fallback") or ()),
            backend_states=payload.get("backend_states") or {},
            fixed_point_iterations=closure.get("fixed_point_iterations", 0),
            closure_fixed_point=closure.get("fixed_point", False),
            closure_complete=closure.get("complete", False)
            and not closure.get("truncated", False),
        )
        claimed = payload.get("receipt_id")
        if claimed not in (None, result.receipt_id):
            raise ProofDirectedRetrievalError("retrieval receipt identity mismatch")
        return result

    @classmethod
    def from_json(cls, payload: str) -> "ProofDirectedRetrievalReceipt":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProofDirectedRetrievalError(
                "proof-directed retrieval JSON is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise ProofDirectedRetrievalError(
                "proof-directed retrieval JSON must contain an object"
            )
        return cls.from_dict(value)


def _request_selectors(request: DecisionRequest) -> tuple[tuple[str, str], ...]:
    values: set[tuple[str, str]] = {
        ("decision_request", request.content_id),
        ("objective", request.objective_id),
        ("objective_revision", request.objective_revision),
        ("acceptance", request.acceptance_id),
        ("repository", request.repository_id),
        ("environment", request.environment_id),
        ("model", request.model_id),
        ("toolchain", request.toolchain_id),
        ("principal", request.authority.principal_id),
        ("action", request.action.action_id),
        ("action_name", request.action.action),
        ("tool", request.action.tool_id),
    }
    if request.jurisdiction:
        values.add(("jurisdiction", request.jurisdiction))
    for target in request.action.targets:
        values.add(("target", target.target_id))
        values.add(("resource_type", target.resource_type))
        values.update(("path", path) for path in target.repository_paths)
    for effect in request.expected_effects:
        values.add(("effect", effect.effect_id))
        values.update(("target", target_id) for target_id in effect.target_ids)
        values.update(("path", path) for path in effect.repository_paths)
    for root in request.semantic_roots:
        values.add((f"root:{root.kind.value}", root.artifact.artifact_id))
        values.add((f"root_cid:{root.kind.value}", root.artifact.cid_v1))
        values.add(
            (f"root_digest:{root.kind.value}", root.artifact.supervisor_digest)
        )
    for fact in request.applicability_facts:
        values.add(("applicability_fact", fact.fact_id))
        values.add(("applicability_predicate", fact.predicate))
        values.add(("applicability_source", fact.source.artifact_id))
    for capability in request.capabilities:
        values.add(("capability", capability.capability_id))
    values.update(
        ("capability", item) for item in request.authority.capability_ids
    )
    if request.authority.authorization is not None:
        values.add(
            ("authorization", request.authority.authorization.artifact_id)
        )
    return tuple(sorted(values))


def _matching_nodes(
    graph: SemanticDependencyGraph,
    selector_kind: str,
    value: str,
) -> tuple[str, ...]:
    matches: list[str] = []
    for node in graph.nodes:
        direct = {
            node.node_id,
            node.source_root_id,
            node.provenance_id,
        }
        record_values = _deep_string_values(node.record)
        if value in direct or value in record_values:
            matches.append(node.node_id)
    return tuple(sorted(matches))


def derive_exact_retrieval_seeds(
    request: DecisionRequest,
    graph: SemanticDependencyGraph,
) -> tuple[RetrievalSeed, ...]:
    """Derive stable exact selectors exclusively from the decision envelope."""

    if not isinstance(request, DecisionRequest):
        raise ProofDirectedRetrievalError("request must be a DecisionRequest")
    if not isinstance(graph, SemanticDependencyGraph):
        raise ProofDirectedRetrievalError(
            "graph must be a SemanticDependencyGraph"
        )
    return tuple(
        RetrievalSeed(
            selector_kind=kind,
            value=value,
            matched_node_ids=_matching_nodes(graph, kind, value),
            mandatory=kind == "decision_request",
        )
        for kind, value in _request_selectors(request)
    )


def _decision_node(
    request: DecisionRequest,
    graph: SemanticDependencyGraph,
    explicit: str,
) -> SemanticNode:
    decisions = graph.nodes_by_kind(SemanticNodeKind.DECISION)
    if explicit:
        try:
            node = graph.node(explicit)
        except KeyError as exc:
            raise StaleRetrievalRootError(
                "declared decision node is absent from the graph"
            ) from exc
        candidates = (node,)
    else:
        candidates = decisions
    exact: list[SemanticNode] = []
    for node in candidates:
        if node.kind is not SemanticNodeKind.DECISION:
            continue
        record = node.record
        request_ids = {
            str(record.get(name) or "")
            for name in (
                "decision_request_id",
                "request_id",
                "decision_id",
                "content_id",
            )
        }
        exact_binding = request.content_id in {
            node.node_id,
            node.provenance_id,
            *request_ids,
        }
        semantic_binding = (
            str(record.get("objective_id") or "") == request.objective_id
            and str(record.get("action_id") or "") == request.action.action_id
        )
        if exact_binding or semantic_binding:
            exact.append(node)
    if len(exact) != 1:
        raise StaleRetrievalRootError(
            "semantic graph must contain exactly one decision node bound to "
            "the DecisionRequest"
        )
    if not exact[0].authoritative:
        raise StaleRetrievalRootError(
            "DecisionRequest graph node is not authority-bearing"
        )
    return exact[0]


def _configuration_id(request: DecisionRequest, index_roots: Mapping[str, str]) -> str:
    return _identity(
        "retrieval-configuration",
        {
            "model_id": request.model_id,
            "toolchain_id": request.toolchain_id,
            "index_roots": dict(sorted(index_roots.items())),
        },
    )


def _query_payload(
    request: DecisionRequest,
    *,
    fingerprint: str,
) -> Mapping[str, Any]:
    selectors = _request_selectors(request)
    terms = {
        request.objective_id,
        request.action.action,
        request.action.tool_id,
        *(path for target in request.action.targets for path in target.repository_paths),
        *(path for effect in request.expected_effects for path in effect.repository_paths),
    }
    text = " ".join(sorted(item for item in terms if item))
    payload = {
        "query_id": _identity(
            "proof-retrieval-query",
            {
                "decision_request_id": request.content_id,
                "text": text,
                "selectors": selectors,
                "embedding_fingerprint": fingerprint,
            },
        ),
        "text": text,
        "exact_selectors": [
            {"kind": kind, "value": value} for kind, value in selectors
        ],
        "embedding_fingerprint": fingerprint,
    }
    return MappingProxyType(payload)


def _root_payload(
    request: DecisionRequest,
    graph: SemanticDependencyGraph,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "decision_request_id": request.content_id,
            "repository_id": request.repository_id,
            "semantic_roots": {
                root.kind.value: {
                    "artifact_id": root.artifact.artifact_id,
                    "cid_v1": root.artifact.cid_v1,
                    "supervisor_digest": root.artifact.supervisor_digest,
                    "reference_id": root.artifact.content_id,
                }
                for root in request.semantic_roots
            },
            "semantic_graph_root_id": graph.root_id,
            "semantic_graph_id": graph.graph_id,
        }
    )


def _invoke_provider(provider: Any, query: Mapping[str, Any], limit: int) -> Any:
    method = None
    for name in ("search", "query", "retrieve"):
        candidate = getattr(provider, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None and callable(provider):
        method = provider
    if method is None:
        return provider
    attempts = (
        lambda: method(query, limit=limit),
        lambda: method(query=query, limit=limit),
        lambda: method(query.get("text", ""), limit=limit),
        lambda: method(query),
    )
    last: TypeError | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last = exc
    assert last is not None
    raise last


def _provider_rows(value: Any) -> tuple[Mapping[str, Any], tuple[Any, ...]]:
    metadata: Mapping[str, Any] = {}
    if isinstance(value, Mapping):
        metadata = value
        rows = (
            value.get("candidates")
            or value.get("results")
            or value.get("matches")
            or value.get("rows")
            or ()
        )
    else:
        rows = value
    if isinstance(rows, (str, bytes, bytearray, Mapping)) or not isinstance(
        rows, Sequence
    ):
        raise ProofDirectedRetrievalError(
            "retrieval provider rows must be a sequence"
        )
    return metadata, tuple(rows)


def _score_millionths(value: Any, *, already_millionths: bool = False) -> int:
    if isinstance(value, bool):
        raise RetrievalBindingError("candidate score must be finite")
    if already_millionths:
        if not isinstance(value, int):
            raise RetrievalBindingError(
                "candidate score_millionths must be an integer"
            )
        if 0 <= value <= 1_000_000:
            return value
        raise RetrievalBindingError("candidate score is out of range")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalBindingError("candidate score must be finite") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RetrievalBindingError("candidate score is out of range")
    return int(round(score * 1_000_000))


def _raw_candidate_id(source: str, rank: int, value: Any) -> str:
    try:
        normalized = _plain(value)
    except ProofDirectedRetrievalError:
        normalized = {"type": type(value).__name__}
    return _identity(
        "rejected-retrieval-candidate",
        {"source": source, "rank": rank, "record": normalized},
    )


def _normalize_candidate(
    raw: Any,
    *,
    source: str,
    rank: int,
    metadata: Mapping[str, Any],
    snapshot: RetrievalSnapshotBinding,
    node_ids: frozenset[str],
) -> BoundRetrievalCandidate:
    if isinstance(raw, BoundRetrievalCandidate):
        if raw.source != source:
            raise RetrievalBindingError("candidate source differs from its provider")
        return validate_bound_retrieval_candidate(
            raw, snapshot, node_ids=node_ids
        )
    row = _mapping(raw)
    claimed_source = str(row.get("source") or source)
    if claimed_source != source:
        raise RetrievalBindingError("candidate source differs from its provider")
    if row.get("proof_authority") not in (None, False):
        raise RetrievalBindingError("retrieval candidate claims proof authority")
    if row.get("authority") not in (None, "context_only", "proposal_only"):
        raise RetrievalBindingError("retrieval candidate claims elevated authority")
    binding_value = row.get("binding", metadata.get("binding"))
    if binding_value is None:
        combined = {**metadata, **row}
        binding_value = {
            "graph_root_id": combined.get("graph_root_id", ""),
            "graph_id": combined.get("graph_id", ""),
            "partition_id": combined.get("partition_id", ""),
            "configuration_id": combined.get("configuration_id", ""),
            "model_id": combined.get("model_id", ""),
            "embedding_fingerprint": combined.get("embedding_fingerprint", ""),
            "index_roots": combined.get("index_roots") or {},
        }
    binding = RetrievalSnapshotBinding.from_dict(binding_value)
    if "score_millionths" in row:
        score = _score_millionths(
            row.get("score_millionths"), already_millionths=True
        )
    else:
        score = _score_millionths(row.get("score", row.get("similarity")))
    combined = {**metadata, **row}
    candidate = BoundRetrievalCandidate(
        node_id=str(
            row.get("node_id")
            or row.get("candidate_node_id")
            or row.get("id")
            or ""
        ),
        source=source,
        score_millionths=score,
        binding=binding,
        index_root_id=str(
            combined.get("index_root_id")
            or (binding.index_roots.get(source, ""))
        ),
        rank=rank,
        candidate_id=str(row.get("candidate_id") or ""),
    )
    return validate_bound_retrieval_candidate(
        candidate, snapshot, node_ids=node_ids
    )


def _fixed_point_iterations(closure: MandatoryClosure) -> int:
    return max((len(path) for path in closure.paths.values()), default=1)


def retrieve_proof_directed(
    request: DecisionRequest,
    graph: SemanticDependencyGraph,
    *,
    decision_node_id: str = "",
    candidate_providers: Mapping[str, Any] | None = None,
    candidates: Iterable[Any] = (),
    index_roots: Mapping[str, str] | None = None,
    partition_id: str = "",
    configuration_id: str = "",
    query_embedding: Sequence[Any] = (),
    query_embedding_fingerprint: str = "",
    required_indexes: Iterable[str] = (),
    allow_exact_fallback: bool = True,
    budget: ProofRetrievalBudget | None = None,
) -> ProofDirectedRetrievalReceipt:
    """Retrieve optional candidates and independently close mandatory edges."""

    if not isinstance(request, DecisionRequest):
        raise ProofDirectedRetrievalError("request must be a DecisionRequest")
    if not isinstance(graph, SemanticDependencyGraph):
        raise ProofDirectedRetrievalError(
            "graph must be a SemanticDependencyGraph"
        )
    limits = budget or ProofRetrievalBudget.from_decision(request)
    if (
        limits.max_graph_depth > request.budget.max_graph_hops
        or limits.max_graph_nodes > request.budget.max_items
        or limits.max_candidates > request.budget.max_retrieval_results
        or limits.max_receipt_bytes > request.budget.max_serialized_bytes
    ):
        raise ProofDirectedRetrievalBudgetError(
            "retrieval budget cannot exceed its DecisionRequest budget"
        )
    roots = {
        _text(str(name), "index name"): _text(str(value), "index root")
        for name, value in sorted((index_roots or {}).items())
    }
    unknown_roots = set(roots).difference(APPROXIMATE_SOURCES)
    if unknown_roots:
        raise ProofDirectedRetrievalError(
            "unsupported retrieval index roots: " + ", ".join(sorted(unknown_roots))
        )
    required = _strings(required_indexes)
    unknown_required = set(required).difference(APPROXIMATE_SOURCES)
    if unknown_required:
        raise ProofDirectedRetrievalError(
            "unsupported required indexes: " + ", ".join(sorted(unknown_required))
        )
    config_id = configuration_id or _configuration_id(request, roots)
    embedding_error = ""
    try:
        derived_fingerprint = embedding_fingerprint(
            query_embedding,
            model_id=request.model_id,
            configuration_id=config_id,
        )
    except ProofDirectedRetrievalError:
        embedding_error = "poisoned_or_non_finite_query_embedding"
        # Bind the invalid input's structural shape without serializing NaN,
        # infinity, model output, or the vector itself into the receipt.
        derived_fingerprint = _identity(
            "invalid-embedding",
            {
                "model_id": request.model_id,
                "configuration_id": config_id,
                "dimensions": len(query_embedding),
                "value_types": [
                    type(item).__name__ for item in query_embedding
                ],
                "reason": embedding_error,
            },
        )
    if query_embedding_fingerprint:
        claimed_fingerprint = _text(
            query_embedding_fingerprint, "query_embedding_fingerprint"
        )
        if not embedding_error and claimed_fingerprint != derived_fingerprint:
            raise ProofDirectedRetrievalError(
                "supplied embedding fingerprint does not match the vector"
            )
        fingerprint = (
            derived_fingerprint if embedding_error else claimed_fingerprint
        )
    else:
        fingerprint = derived_fingerprint
    snapshot = RetrievalSnapshotBinding(
        graph_root_id=graph.root_id,
        graph_id=graph.graph_id,
        partition_id=partition_id or request.repository_id,
        configuration_id=config_id,
        model_id=request.model_id,
        embedding_fingerprint=fingerprint,
        index_roots=roots,
    )
    decision = _decision_node(request, graph, decision_node_id)
    seeds = derive_exact_retrieval_seeds(request, graph)
    try:
        closure = graph.mandatory_closure(
            decision.node_id,
            bounds=ClosureBounds(
                max_nodes=limits.max_graph_nodes,
                max_edges=limits.max_graph_edges,
                max_depth=limits.max_graph_depth,
                # Annotation enumeration is an implementation detail of the
                # shared graph closure.  It is ignored here and must never
                # make optional corpus growth fail a mandatory traversal.
                max_annotations=max(1, len(graph.nodes)),
            ),
        )
    except SemanticGraphBoundsError as exc:
        raise ProofDirectedRetrievalBudgetError(
            "mandatory authority/proof closure exhausted its graph budget"
        ) from exc

    query = _query_payload(request, fingerprint=fingerprint)
    providers = dict(candidate_providers or {})
    direct_by_source: dict[str, list[Any]] = {}
    for raw in candidates:
        if isinstance(raw, BoundRetrievalCandidate):
            source = raw.source
        elif isinstance(raw, Mapping):
            source = str(raw.get("source") or "")
        else:
            raise ProofDirectedRetrievalError(
                "direct candidates must be bound candidates or mappings"
            )
        if source not in APPROXIMATE_SOURCES:
            raise ProofDirectedRetrievalError(
                "direct candidates require an explicit bm25, vector, ast, "
                "or graphrag source"
            )
        direct_by_source.setdefault(source, []).append(raw)
    for source, rows in direct_by_source.items():
        if source in providers:
            raise ProofDirectedRetrievalError(
                f"{source} candidates were supplied both directly and by provider"
            )
        providers[source] = tuple(rows)
    backend_states: dict[str, str] = {
        name: RetrievalBackendState.UNAVAILABLE.value
        for name in APPROXIMATE_SOURCES
    }
    audits: list[CandidateAudit] = []
    accepted: list[BoundRetrievalCandidate] = []
    node_ids = frozenset(item.node_id for item in graph.nodes)
    considered = 0
    provider_returned = 0
    provider_dropped = 0
    provider_failures: dict[str, str] = {}

    for source in sorted(providers):
        provider = providers[source]
        if source not in APPROXIMATE_SOURCES:
            raise ProofDirectedRetrievalError(
                f"unsupported retrieval candidate source: {source}"
            )
        effective_source = source
        if source == "vector" and embedding_error:
            provider_failures[source] = "PoisonedQueryEmbedding"
            backend_states[source] = RetrievalBackendState.UNHEALTHY.value
            continue
        if source not in roots:
            provider_failures[source] = "MissingPinnedIndexRoot"
            backend_states[source] = RetrievalBackendState.UNHEALTHY.value
            continue
        try:
            response = _invoke_provider(provider, query, limits.max_candidates)
            metadata, rows = _provider_rows(response)
            if source in APPROXIMATE_SOURCES:
                backend_states[source] = RetrievalBackendState.HEALTHY.value
        except Exception as exc:
            provider_failures[source] = type(exc).__name__
            if source in APPROXIMATE_SOURCES:
                backend_states[source] = RetrievalBackendState.UNHEALTHY.value
            continue
        provider_returned += len(rows)
        remaining = max(0, limits.max_candidates - considered)
        bounded_rows = rows[:remaining]
        provider_dropped += len(rows) - len(bounded_rows)
        for rank, raw in enumerate(bounded_rows):
            considered += 1
            raw_id = _raw_candidate_id(effective_source, rank, raw)
            try:
                candidate = _normalize_candidate(
                    raw,
                    source=effective_source,
                    rank=rank,
                    metadata=metadata,
                    snapshot=snapshot,
                    node_ids=node_ids,
                )
            except Exception as exc:
                row = raw if isinstance(raw, Mapping) else {}
                audits.append(
                    CandidateAudit(
                        candidate_id=raw_id,
                        node_id=str(row.get("node_id") or row.get("id") or ""),
                        source=effective_source,
                        disposition=CandidateDisposition.REJECTED,
                        score_millionths=None,
                        rank=rank,
                        binding_id="",
                        reason=f"{type(exc).__name__}:{str(exc)[:256]}",
                    )
                )
                continue
            accepted.append(candidate)
            audits.append(
                CandidateAudit(
                    candidate_id=candidate.candidate_id,
                    node_id=candidate.node_id,
                    source=candidate.source,
                    disposition=CandidateDisposition.ACCEPTED,
                    score_millionths=candidate.score_millionths,
                    rank=candidate.rank,
                    binding_id=candidate.binding.binding_id,
                )
            )

    fallback: list[str] = []
    for name in required:
        state = backend_states.get(name, RetrievalBackendState.UNAVAILABLE.value)
        if name not in roots or state in {
            RetrievalBackendState.UNAVAILABLE.value,
            RetrievalBackendState.UNHEALTHY.value,
        }:
            if not allow_exact_fallback:
                raise MissingRequiredIndexError(
                    f"required {name} index is unavailable and exact fallback is disabled"
                )
            backend_states[name] = RetrievalBackendState.EXACT_FALLBACK.value
            fallback.append(f"{name}:deterministic_exact_graph_scan")

    accepted.sort(
        key=lambda item: (
            -item.score_millionths,
            item.node_id,
            item.source,
            item.candidate_id,
        )
    )
    best_by_node: dict[str, BoundRetrievalCandidate] = {}
    scores_by_node: dict[str, set[int]] = {}
    for item in accepted:
        scores_by_node.setdefault(item.node_id, set()).add(item.score_millionths)
        best_by_node.setdefault(item.node_id, item)
    disagreement = [
        f"candidate_score_disagreement:{node_id}"
        for node_id, scores in sorted(scores_by_node.items())
        if len(scores) > 1
    ]
    disagreement.extend(
        f"provider_failure:{source}:{error}"
        for source, error in sorted(provider_failures.items())
    )
    if embedding_error:
        disagreement.append(embedding_error)

    closure_nodes = set(closure.node_ids)
    optional: list[str] = []
    omitted: set[str] = set()
    for item in best_by_node.values():
        if item.node_id in closure_nodes:
            continue
        if len(optional) < limits.max_optional_nodes:
            optional.append(item.node_id)
        else:
            omitted.add(item.node_id)
    for audit in audits:
        if (
            audit.node_id
            and audit.disposition is not CandidateDisposition.ACCEPTED
            and audit.node_id not in closure_nodes
            and audit.node_id not in best_by_node
        ):
            omitted.add(audit.node_id)
    optional = sorted(set(optional))
    included = sorted(closure_nodes.union(optional))
    truncated_candidates = provider_dropped + sum(
        item.disposition is CandidateDisposition.TRUNCATED for item in audits
    )
    rejected_candidates = sum(
        item.disposition is CandidateDisposition.REJECTED for item in audits
    )
    truncation = {
        "candidate_returned_count": provider_returned,
        "candidate_considered_count": considered,
        "candidate_audit_count": len(audits),
        "candidate_budget": limits.max_candidates,
        "optional_node_budget": limits.max_optional_nodes,
        "candidate_truncation_count": truncated_candidates,
        "optional_node_omission_count": len(omitted),
        "rejected_candidate_count": rejected_candidates,
        "retrieval_truncated": bool(truncated_candidates or omitted),
        "mandatory_truncated": False,
    }

    receipt = ProofDirectedRetrievalReceipt(
        decision_request_id=request.content_id,
        query=query,
        roots=_root_payload(request, graph),
        snapshot=snapshot,
        budgets=limits,
        seeds=seeds,
        candidates=tuple(audits),
        closure_id=closure.closure_id,
        closure_node_ids=closure.node_ids,
        closure_edge_ids=closure.edge_ids,
        paths=closure.paths,
        included_node_ids=tuple(included),
        optional_node_ids=tuple(optional),
        omitted_node_ids=tuple(omitted),
        truncation=truncation,
        disagreement=tuple(disagreement),
        fallback=tuple(fallback),
        backend_states=backend_states,
        fixed_point_iterations=_fixed_point_iterations(closure),
    )

    # Optional audit rows are the only part that may be reduced to make room.
    # Mandatory roots, seeds, paths, included closure nodes, and fixed-point
    # evidence are never trimmed.
    while (
        len(receipt.to_json().encode("utf-8")) > limits.max_receipt_bytes
        and receipt.candidates
    ):
        removed = receipt.candidates[-1]
        new_omitted = set(receipt.omitted_node_ids)
        remaining_candidates = receipt.candidates[:-1]
        has_remaining_acceptance = bool(
            removed.node_id
            and any(
                item.disposition is CandidateDisposition.ACCEPTED
                and item.node_id == removed.node_id
                for item in remaining_candidates
            )
        )
        if (
            removed.node_id
            and removed.node_id not in closure_nodes
            and not has_remaining_acceptance
        ):
            new_omitted.add(removed.node_id)
        new_optional = set(receipt.optional_node_ids)
        if (
            removed.disposition is CandidateDisposition.ACCEPTED
            and removed.node_id
            and not has_remaining_acceptance
        ):
            new_optional.discard(removed.node_id)
        new_included = closure_nodes.union(new_optional)
        new_truncation = dict(receipt.truncation)
        new_truncation["receipt_candidate_omission_count"] = (
            int(new_truncation.get("receipt_candidate_omission_count", 0)) + 1
        )
        new_truncation["retrieval_truncated"] = True
        receipt = replace(
            receipt,
            candidates=remaining_candidates,
            included_node_ids=tuple(new_included),
            optional_node_ids=tuple(new_optional),
            omitted_node_ids=tuple(new_omitted),
            truncation=new_truncation,
        )
    if len(receipt.to_json().encode("utf-8")) > limits.max_receipt_bytes:
        raise ProofDirectedRetrievalBudgetError(
            "complete mandatory retrieval receipt exceeds max_receipt_bytes"
        )
    return receipt


# Compatibility/descriptive spellings for callers and later context compilation.
ProofDirectedRetrievalResult = ProofDirectedRetrievalReceipt
RetrievalClosureReceipt = ProofDirectedRetrievalReceipt
RetrievalCandidate = BoundRetrievalCandidate
RetrievalCandidateBinding = RetrievalSnapshotBinding
derive_retrieval_seeds = derive_exact_retrieval_seeds
retrieve_authoritative_closure = retrieve_proof_directed
retrieve_proof_directed_evidence = retrieve_proof_directed
proof_directed_retrieve = retrieve_proof_directed
build_proof_directed_retrieval_receipt = retrieve_proof_directed
build_retrieval_closure_receipt = retrieve_proof_directed


RETRIEVAL_CONTEXT_SLICE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/retrieval-context-slice@1"
)


@dataclass(frozen=True)
class RetrievalContextSlice:
    """Body-free proof-directed retrieval projection for Planner/Doctor context.

    Satisfied mandatory closure and optional nominations are represented only
    as compact node identifiers and expansion CIDs/handles.  No source bodies,
    proof transcripts, embeddings, or secrets are embedded.
    """

    receipt_id: str
    closure_id: str
    decision_request_id: str
    mandatory_node_ids: tuple[str, ...]
    optional_node_ids: tuple[str, ...]
    omitted_node_ids: tuple[str, ...]
    seed_ids: tuple[str, ...]
    expansion_cids: tuple[str, ...]
    paths: Mapping[str, tuple[str, ...]]
    closure_complete: bool
    closure_fixed_point: bool
    schema: str = RETRIEVAL_CONTEXT_SLICE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "closure_id", _text(self.closure_id, "closure_id")
        )
        object.__setattr__(
            self,
            "decision_request_id",
            _text(self.decision_request_id, "decision_request_id"),
        )
        for name in (
            "mandatory_node_ids",
            "optional_node_ids",
            "omitted_node_ids",
            "seed_ids",
            "expansion_cids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        object.__setattr__(
            self,
            "paths",
            MappingProxyType(
                {
                    str(key): tuple(str(item) for item in value)
                    for key, value in sorted(self.paths.items())
                }
            ),
        )
        if not isinstance(self.closure_complete, bool):
            raise ProofDirectedRetrievalError("closure_complete must be boolean")
        if not isinstance(self.closure_fixed_point, bool):
            raise ProofDirectedRetrievalError(
                "closure_fixed_point must be boolean"
            )
        if self.schema != RETRIEVAL_CONTEXT_SLICE_SCHEMA:
            raise ProofDirectedRetrievalError(
                "unsupported retrieval context slice schema"
            )

    @property
    def slice_id(self) -> str:
        return _identity(
            "retrieval-context-slice",
            self.to_dict(include_identity=False),
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "closure_id": self.closure_id,
            "decision_request_id": self.decision_request_id,
            "mandatory_node_ids": list(self.mandatory_node_ids),
            "optional_node_ids": list(self.optional_node_ids),
            "omitted_node_ids": list(self.omitted_node_ids),
            "seed_ids": list(self.seed_ids),
            "expansion_cids": list(self.expansion_cids),
            "paths": {
                key: list(value) for key, value in self.paths.items()
            },
            "closure_complete": self.closure_complete,
            "closure_fixed_point": self.closure_fixed_point,
            "body_embedded": False,
            "proof_body_embedded": False,
            "source_body_embedded": False,
            "completion_authority": False,
            "proof_authority": False,
        }
        if include_identity:
            payload["slice_id"] = self.slice_id
        return payload

    def to_causal_ast_slice(self) -> dict[str, Any]:
        """Compact causal/AST slice handles for PlannerDoctor context cores."""

        return {
            "closure_id": self.closure_id,
            "receipt_id": self.receipt_id,
            "closure_node_ids": list(self.mandatory_node_ids),
            "optional_node_ids": list(self.optional_node_ids),
            "omitted_node_ids": list(self.omitted_node_ids),
            "seed_ids": list(self.seed_ids),
            "expansion_cids": list(self.expansion_cids),
            "paths": {key: list(value) for key, value in self.paths.items()},
            "closure_complete": self.closure_complete,
            "closure_fixed_point": self.closure_fixed_point,
            "digest_handles_only": True,
        }


def project_retrieval_context_slice(
    receipt: ProofDirectedRetrievalReceipt | Mapping[str, Any],
) -> RetrievalContextSlice:
    """Project a retrieval receipt into a body-free context slice.

    Approximate optional nominations become expansion CIDs/handles.  The
    mandatory authority/proof closure remains independently named and never
    depends on optional ranking for completeness.
    """

    if isinstance(receipt, ProofDirectedRetrievalReceipt):
        payload = receipt.to_dict()
        receipt_id = receipt.receipt_id
        seeds = receipt.seeds
        paths = dict(receipt.paths)
        mandatory = receipt.closure_node_ids
        optional = receipt.optional_node_ids
        omitted = receipt.omitted_node_ids
        closure_id = receipt.closure_id
        decision_request_id = receipt.decision_request_id
        closure_complete = receipt.closure_complete
        closure_fixed_point = receipt.closure_fixed_point
    elif isinstance(receipt, Mapping):
        payload = dict(receipt)
        receipt_id = str(
            payload.get("receipt_id") or payload.get("content_id") or ""
        )
        if not receipt_id:
            receipt_id = _identity("retrieval-receipt", payload)
        seeds = tuple(payload.get("seeds") or ())
        paths = dict(payload.get("paths") or {})
        mandatory = tuple(payload.get("closure_node_ids") or ())
        optional = tuple(payload.get("optional_node_ids") or ())
        omitted = tuple(payload.get("omitted_node_ids") or ())
        closure_id = str(payload.get("closure_id") or "")
        decision_request_id = str(payload.get("decision_request_id") or "")
        closure_complete = bool(payload.get("closure_complete", True))
        closure_fixed_point = bool(payload.get("closure_fixed_point", True))
    else:
        raise ProofDirectedRetrievalError(
            "receipt must be a ProofDirectedRetrievalReceipt or mapping"
        )

    seed_ids: list[str] = []
    for item in seeds:
        if isinstance(item, RetrievalSeed):
            seed_ids.append(item.seed_id)
        elif isinstance(item, Mapping):
            seed_id = str(item.get("seed_id") or "")
            if seed_id:
                seed_ids.append(seed_id)
            else:
                seed_ids.append(_identity("retrieval-seed", item))
        else:
            seed_ids.append(_text(str(item), "seed_id"))

    # Expansion CIDs are content digests over optional/omitted node handles —
    # never source or proof bodies.
    expansion_cids = tuple(
        sorted(
            {
                _identity("expansion-node", {"node_id": node_id, "kind": "optional"})
                for node_id in optional
            }
            | {
                _identity("expansion-node", {"node_id": node_id, "kind": "omitted"})
                for node_id in omitted
            }
            | ({_identity("expansion-receipt", receipt_id)} if receipt_id else set())
        )
    )

    return RetrievalContextSlice(
        receipt_id=receipt_id,
        closure_id=closure_id or _identity("closure", mandatory),
        decision_request_id=decision_request_id or "decision:unknown",
        mandatory_node_ids=tuple(mandatory),
        optional_node_ids=tuple(optional),
        omitted_node_ids=tuple(omitted),
        seed_ids=tuple(seed_ids),
        expansion_cids=expansion_cids,
        paths=paths,
        closure_complete=closure_complete,
        closure_fixed_point=closure_fixed_point,
    )


def retrieval_slice_for_planner_doctor_context(
    receipt: ProofDirectedRetrievalReceipt | Mapping[str, Any],
) -> dict[str, Any]:
    """Convenience projection consumed by PlannerDoctorContextRequest builders."""

    slice_ = project_retrieval_context_slice(receipt)
    return {
        "retrieval_receipt_id": slice_.receipt_id,
        "retrieval_closure_id": slice_.closure_id,
        "retrieval_slice_node_ids": list(slice_.mandatory_node_ids),
        "expansion_cids": list(slice_.expansion_cids),
        "causal_ast_slice": slice_.to_causal_ast_slice(),
        "slice": slice_.to_dict(),
    }


__all__ = [
    "APPROXIMATE_SOURCES",
    "CandidateAudit",
    "CandidateDisposition",
    "MissingRequiredIndexError",
    "PROOF_DIRECTED_RETRIEVAL_SCHEMA",
    "ProofDirectedRetrievalBudgetError",
    "ProofDirectedRetrievalError",
    "ProofDirectedRetrievalReceipt",
    "ProofDirectedRetrievalResult",
    "ProofRetrievalBudget",
    "RETRIEVAL_CLOSURE_REQUIREMENT_ID",
    "RETRIEVAL_CONTEXT_SLICE_SCHEMA",
    "RetrievalBackendState",
    "RetrievalCandidate",
    "RetrievalCandidateBinding",
    "RetrievalClosureReceipt",
    "RetrievalContextSlice",
    "RetrievalSeed",
    "StaleRetrievalRootError",
    "build_proof_directed_retrieval_receipt",
    "build_retrieval_closure_receipt",
    "derive_exact_retrieval_seeds",
    "derive_retrieval_seeds",
    "embedding_fingerprint",
    "project_retrieval_context_slice",
    "proof_directed_retrieve",
    "retrieval_slice_for_planner_doctor_context",
    "retrieve_authoritative_closure",
    "retrieve_proof_directed",
    "retrieve_proof_directed_evidence",
]
