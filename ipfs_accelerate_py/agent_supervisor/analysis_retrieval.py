"""Deterministic, bounded multi-signal retrieval for analysis evidence.

This module is intentionally a projection layer rather than a source or
artifact reader.  It accepts the supervisor's existing graph, todo-vector,
dependency, goal-coverage, and proof-scope records and returns only compact
evidence references.  Source bodies, model output, AST bodies, and nested
artifact graphs are never copied into a result.

Fusion uses fixed weights.  A missing or failed optional signal contributes
zero and remains visible in ``backend_health``; weights are never redistributed
among the remaining signals.  That property makes a replay comparable across
machines with different optional analysis backends.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ANALYSIS_RETRIEVAL_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.analysis-retrieval@1"
)
ANALYSIS_EVIDENCE_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.analysis-evidence-reference@1"
)
DEFAULT_SIGNAL_WEIGHTS: Mapping[str, float] = {
    "lexical": 0.28,
    "vector": 0.18,
    "ast_symbol": 0.18,
    "dependency_neighborhood": 0.14,
    "goal_coverage": 0.11,
    "proof_gap": 0.11,
}
SIGNAL_ORDER = tuple(DEFAULT_SIGNAL_WEIGHTS)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*")
_SPACE_RE = re.compile(r"\s+")
_MAX_TEXT = 320
_MAX_DETAIL = 180
_MAX_REFERENCES_PER_RESULT = 8
_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "source",
        "source_body",
        "source_text",
        "body",
        "content",
        "text",
        "raw",
        "raw_text",
        "decoded",
        "decoded_text",
        "decoded_model_text",
        "model_output",
        "model_response",
        "prompt",
        "completion",
        "transcript",
        "ast",
        "ast_body",
        "graph",
        "objective_graph",
        "artifact_graph",
        "nested_graph",
        "payload",
        "metadata",
        "embedding",
    }
)


class RetrievalValidationError(ValueError):
    """Raised when a retrieval query, limit, or backend response is unsafe."""


class RetrievalBudgetError(RetrievalValidationError):
    """Raised when mandatory health and truncation metadata cannot fit."""


class BackendState(str, Enum):
    """Observable state of one retrieval signal."""

    HEALTHY = "healthy"
    UNAVAILABLE = "unavailable"
    UNHEALTHY = "unhealthy"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        converted = [_canonical_value(item) for item in value]
        return sorted(converted, key=_canonical_json)
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RetrievalValidationError("retrieval values must be finite")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _bounded_text(value: Any, maximum: int = _MAX_TEXT) -> str:
    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) <= maximum:
        return text
    if maximum <= 1:
        return text[:maximum]
    return text[: maximum - 1].rstrip() + "…"


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
        return dict(result) if isinstance(result, Mapping) else {}
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _record_values(value: Any) -> tuple[Any, ...]:
    """Accept a record, a record sequence, or an ID-to-record mapping."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        record_markers = {
            "task_id",
            "obligation_id",
            "receipt_id",
            "criterion_id",
            "node_id",
            "record_id",
            "title",
            "kind",
        }
        if record_markers.intersection(value):
            return (value,)
        if all(isinstance(item, Mapping) for item in value.values()):
            return tuple(value[key] for key in sorted(value, key=str))
        return ()
    projected = _mapping(value)
    if projected and {
        "task_id",
        "obligation_id",
        "receipt_id",
        "criterion_id",
        "node_id",
        "record_id",
        "title",
        "kind",
    }.intersection(projected):
        return (value,)
    return _sequence(value)


def _strings(value: Any, *, maximum: int = 128) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        values = value.keys()
    else:
        try:
            values = iter(value)
        except TypeError:
            values = (value,)
    result = {
        _bounded_text(item)
        for item in values
        if _bounded_text(item)
    }
    return tuple(sorted(result, key=lambda item: (item.casefold(), item))[:maximum])


def _first(record: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value not in (None, "") and not isinstance(value, (Mapping, list, tuple)):
            text = _bounded_text(getattr(value, "value", value))
            if text:
                return text
    return ""


def _tokens(value: Any) -> frozenset[str]:
    text = str(value or "").casefold()
    tokens: set[str] = set()
    for match in _TOKEN_RE.findall(text):
        normalized = match.strip("._:/-")
        if not normalized:
            continue
        tokens.add(normalized)
        for part in re.split(r"[._:/-]+", normalized):
            if part:
                tokens.add(part)
    return frozenset(tokens)


def _score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(max(0.0, min(1.0, number)), 6)


def _cosine(left: Sequence[Any], right: Sequence[Any]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    try:
        left_values = [float(item) for item in left]
        right_values = [float(item) for item in right]
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(item) for item in (*left_values, *right_values)):
        return 0.0
    left_norm = math.sqrt(sum(item * item for item in left_values))
    right_norm = math.sqrt(sum(item * item for item in right_values))
    if not left_norm or not right_norm:
        return 0.0
    return _score(
        sum(a * b for a, b in zip(left_values, right_values))
        / (left_norm * right_norm)
    )


@dataclass(frozen=True)
class RetrievalLimits:
    """Hard limits applied to candidate expansion and serialized output."""

    max_results: int = 20
    max_bytes: int = 64 * 1024
    max_candidates: int = 1000
    max_hops: int = 2
    max_backend_results: int = 200

    def __post_init__(self) -> None:
        values = {
            "max_results": (self.max_results, 1, 1000),
            "max_bytes": (self.max_bytes, 1, 16 * 1024 * 1024),
            "max_candidates": (self.max_candidates, 1, 100_000),
            "max_hops": (self.max_hops, 0, 8),
            "max_backend_results": (self.max_backend_results, 1, 10_000),
        }
        for name, (raw, minimum, maximum) in values.items():
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise RetrievalValidationError(f"{name} must be an integer") from exc
            if value < minimum or value > maximum:
                raise RetrievalValidationError(
                    f"{name} must be between {minimum} and {maximum}"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_value(
        cls, value: "RetrievalLimits | Mapping[str, Any] | None"
    ) -> "RetrievalLimits":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise RetrievalValidationError(
                "limits must be RetrievalLimits or a mapping"
            )
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise RetrievalValidationError(
                f"unknown retrieval limit fields: {', '.join(map(str, unknown))}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class RetrievalQuery:
    """One replayable retrieval request.

    ``embedding`` is deliberately omitted from :meth:`to_dict`; it may be
    consumed for vector scoring but does not enter result records or query
    identity.  ``embedding_id`` binds a supplied vector to a stable model or
    cache identity when callers need vector-sensitive replay identities.
    """

    text: str
    task_ids: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    embedding: tuple[float, ...] = ()
    embedding_id: str = ""

    def __post_init__(self) -> None:
        text = _bounded_text(self.text, 4096)
        dimensions = {
            "task_ids": _strings(self.task_ids, maximum=64),
            "goal_ids": _strings(self.goal_ids, maximum=64),
            "symbols": _strings(self.symbols, maximum=128),
            "obligation_ids": _strings(self.obligation_ids, maximum=64),
            "paths": _strings(self.paths, maximum=128),
        }
        if not text and not any(dimensions.values()):
            raise RetrievalValidationError(
                "retrieval query requires text or an exact selector"
            )
        object.__setattr__(self, "text", text)
        for name, value in dimensions.items():
            object.__setattr__(self, name, value)
        vector: list[float] = []
        for item in self.embedding or ():
            try:
                number = float(item)
            except (TypeError, ValueError) as exc:
                raise RetrievalValidationError(
                    "query embedding must contain numbers"
                ) from exc
            if not math.isfinite(number):
                raise RetrievalValidationError("query embedding must be finite")
            vector.append(number)
        object.__setattr__(self, "embedding", tuple(vector))
        embedding_id = _bounded_text(self.embedding_id, 256)
        if vector and not embedding_id:
            embedding_id = _digest("query-embedding", vector)
        object.__setattr__(self, "embedding_id", embedding_id)

    @property
    def query_id(self) -> str:
        return _digest("analysis-query", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "task_ids": list(self.task_ids),
            "goal_ids": list(self.goal_ids),
            "symbols": list(self.symbols),
            "obligation_ids": list(self.obligation_ids),
            "paths": list(self.paths),
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_value(cls, value: "RetrievalQuery | str | Mapping[str, Any]") -> "RetrievalQuery":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if not isinstance(value, Mapping):
            raise RetrievalValidationError(
                "query must be text, a mapping, or RetrievalQuery"
            )
        return cls(
            text=str(value.get("text") or value.get("query") or ""),
            task_ids=_strings(value.get("task_ids")),
            goal_ids=_strings(value.get("goal_ids")),
            symbols=_strings(value.get("symbols") or value.get("ast_symbols")),
            obligation_ids=_strings(value.get("obligation_ids")),
            paths=_strings(value.get("paths") or value.get("files")),
            embedding=tuple(value.get("embedding") or value.get("query_embedding") or ()),
            embedding_id=str(value.get("embedding_id") or ""),
        )


@dataclass(frozen=True, order=True)
class EvidenceReference:
    """Stable provenance link to a durable input record, never its body."""

    source_kind: str
    source_id: str
    record_id: str = ""
    provenance: str = ""
    artifact_id: str = ""

    def __post_init__(self) -> None:
        for name, maximum in (
            ("source_kind", 80),
            ("source_id", 320),
            ("record_id", 320),
            ("provenance", 80),
            ("artifact_id", 320),
        ):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), maximum)
            )
        if not self.source_kind or not self.source_id:
            raise RetrievalValidationError(
                "evidence references require source_kind and source_id"
            )

    @property
    def reference_id(self) -> str:
        return _digest(
            "evidence-ref",
            {
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "record_id": self.record_id,
                "provenance": self.provenance,
                "artifact_id": self.artifact_id,
            },
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": ANALYSIS_EVIDENCE_REFERENCE_SCHEMA,
            "reference_id": self.reference_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "record_id": self.record_id,
            "provenance": self.provenance,
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True)
class SignalScore:
    """One fixed-weight contribution to a fused rank."""

    score: float
    weight: float
    contribution: float
    available: bool
    explanation: str = ""

    def __post_init__(self) -> None:
        for name in ("score", "weight", "contribution"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise RetrievalValidationError(
                    f"signal {name} must be numeric"
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise RetrievalValidationError(
                    f"signal {name} must be finite and non-negative"
                )
            if name in {"score", "weight"} and value > 1.0:
                raise RetrievalValidationError(
                    f"signal {name} must be between zero and one"
                )
            object.__setattr__(self, name, round(value, 6))
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(
            self, "explanation", _bounded_text(self.explanation, _MAX_DETAIL)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "weight": round(self.weight, 6),
            "contribution": round(self.contribution, 6),
            "available": self.available,
            "explanation": _bounded_text(self.explanation, _MAX_DETAIL),
        }


@dataclass(frozen=True)
class BackendHealth:
    """Health and semantic participation of one signal."""

    signal: str
    state: BackendState
    detail: str
    candidate_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal", _bounded_text(self.signal, 80))
        object.__setattr__(self, "state", BackendState(self.state))
        object.__setattr__(self, "detail", _bounded_text(self.detail, _MAX_DETAIL))
        try:
            count = int(self.candidate_count)
        except (TypeError, ValueError) as exc:
            raise RetrievalValidationError(
                "backend candidate_count must be an integer"
            ) from exc
        if count < 0:
            raise RetrievalValidationError(
                "backend candidate_count must be non-negative"
            )
        object.__setattr__(self, "candidate_count", count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "state": self.state.value,
            "available": self.state is BackendState.HEALTHY,
            "detail": _bounded_text(self.detail, _MAX_DETAIL),
            "candidate_count": max(0, int(self.candidate_count)),
        }


@dataclass(frozen=True)
class RetrievalResult:
    """Compact ranked evidence record."""

    evidence_id: str
    entity_kind: str
    title: str
    task_id: str
    goal_id: str
    obligation_id: str
    symbol: str
    path: str
    status: str
    score: float
    signal_scores: Mapping[str, SignalScore]
    ranking_explanation: str
    evidence_references: tuple[EvidenceReference, ...]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "evidence_id": self.evidence_id,
            "entity_kind": self.entity_kind,
            "title": self.title,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "obligation_id": self.obligation_id,
            "symbol": self.symbol,
            "path": self.path,
            "status": self.status,
            "score": round(self.score, 6),
            "signal_scores": {
                name: self.signal_scores[name].to_dict() for name in SIGNAL_ORDER
            },
            "ranking_explanation": self.ranking_explanation,
            "evidence_references": [
                item.to_dict() for item in self.evidence_references
            ],
        }
        if _FORBIDDEN_RESULT_KEYS.intersection(value):
            raise AssertionError("unsafe fields entered a retrieval result")
        return value


@dataclass(frozen=True)
class TruncationMetadata:
    """Auditable account of every applied retrieval bound."""

    considered_count: int
    eligible_count: int
    returned_count: int
    dropped_by_candidate_limit: int
    dropped_by_count_limit: int
    dropped_by_byte_limit: int
    reference_truncation_count: int
    max_candidates: int
    max_results: int
    max_bytes: int
    max_hops: int
    max_backend_results: int
    output_bytes: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResponse:
    """Deterministic response envelope."""

    query_id: str
    results: tuple[RetrievalResult, ...]
    backend_health: Mapping[str, BackendHealth]
    truncation: TruncationMetadata
    weights: Mapping[str, float]
    schema: str = ANALYSIS_RETRIEVAL_SCHEMA

    @property
    def response_id(self) -> str:
        payload = self.to_dict(include_response_id=False)
        # output_bytes is an encoding fact, not evidence identity.
        payload["truncation"]["output_bytes"] = 0
        return _digest("analysis-retrieval", payload)

    def to_dict(self, *, include_response_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "query_id": self.query_id,
            "ranking": {
                "method": "fixed_weight_linear_fusion",
                "unavailable_signal_semantics": "zero_contribution_without_weight_redistribution",
                "weights": {
                    name: round(float(self.weights[name]), 6)
                    for name in SIGNAL_ORDER
                },
            },
            "backend_health": {
                name: self.backend_health[name].to_dict() for name in SIGNAL_ORDER
            },
            "results": [item.to_dict() for item in self.results],
            "truncation": self.truncation.to_dict(),
        }
        if include_response_id:
            payload["response_id"] = self.response_id
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return _canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )


@dataclass
class _Candidate:
    key: str
    kind: str
    title: str = ""
    task_id: str = ""
    goal_id: str = ""
    obligation_id: str = ""
    receipt_id: str = ""
    criterion_id: str = ""
    symbol: str = ""
    path: str = ""
    status: str = ""
    terms: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    vectors: list[tuple[float, ...]] = field(default_factory=list)
    references: dict[str, EvidenceReference] = field(default_factory=dict)
    goal_statuses: set[str] = field(default_factory=set)
    proof_gap: float = 0.0

    @property
    def evidence_id(self) -> str:
        return _digest("analysis-evidence", self.key)

    def add_reference(self, reference: EvidenceReference) -> None:
        self.references[reference.reference_id] = reference

    def absorb_text(self, *values: Any) -> None:
        for value in values:
            self.terms.update(_tokens(value))


class BoundedGraphRAGRetriever:
    """Build and query a compact in-memory projection of supervisor evidence.

    Inputs may be typed repository objects or their persisted dictionary
    representations.  Optional vector and AST backends are invoked only during
    retrieval and are isolated: exceptions become explicit ``unhealthy``
    states and contribute zero under the unchanged fusion weights.
    """

    def __init__(
        self,
        *,
        evidence_graph: Any = None,
        records: Iterable[Any] = (),
        todo_records: Iterable[Any] = (),
        dependency_graph: Any = None,
        goal_coverage: Any = None,
        proof_scope_index: Any = None,
        vector_backend: Any = None,
        vector_embedder: Callable[[str], Sequence[float]] | None = None,
        ast_backend: Any = None,
        artifact_id: str = "",
        signal_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.vector_backend = vector_backend
        self.vector_embedder = vector_embedder
        self.ast_backend = ast_backend
        self.artifact_id = _bounded_text(artifact_id, 320)
        self.weights = self._weights(signal_weights)
        self._candidates: dict[str, _Candidate] = {}
        self._aliases: dict[str, str] = {}
        self._node_keys: dict[str, str] = {}
        self._dependency_edges: set[tuple[str, str]] = set()
        self._task_obligations: dict[str, set[str]] = {}
        self._obligation_symbols: dict[str, set[str]] = {}
        self._coverage_present = goal_coverage is not None
        self._proof_present = proof_scope_index is not None
        self._graph_present = evidence_graph is not None

        self._ingest_graph(evidence_graph)
        self._ingest_records(_record_values(records), "analysis_record")
        self._ingest_records(_record_values(todo_records), "todo_index_record")
        self._ingest_dependency_graph(dependency_graph)
        self._ingest_goal_coverage(goal_coverage)
        self._ingest_proof_scope(proof_scope_index)
        self._propagate_graph_context()

    @staticmethod
    def _weights(value: Mapping[str, float] | None) -> dict[str, float]:
        supplied = dict(DEFAULT_SIGNAL_WEIGHTS if value is None else value)
        if set(supplied) != set(SIGNAL_ORDER):
            missing = sorted(set(SIGNAL_ORDER) - set(supplied))
            extra = sorted(set(supplied) - set(SIGNAL_ORDER))
            raise RetrievalValidationError(
                f"signal weights require exactly {list(SIGNAL_ORDER)}; "
                f"missing={missing}, extra={extra}"
            )
        result: dict[str, float] = {}
        for name in SIGNAL_ORDER:
            try:
                weight = float(supplied[name])
            except (TypeError, ValueError) as exc:
                raise RetrievalValidationError(
                    f"signal weight {name!r} must be numeric"
                ) from exc
            if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
                raise RetrievalValidationError(
                    f"signal weight {name!r} must be between zero and one"
                )
            result[name] = weight
        total = sum(result.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RetrievalValidationError(
                "signal weights must sum to 1.0; weights are never normalized implicitly"
            )
        return result

    def _candidate(
        self,
        *,
        kind: str,
        task_id: str = "",
        obligation_id: str = "",
        receipt_id: str = "",
        criterion_id: str = "",
        symbol: str = "",
        path: str = "",
        stable_id: str = "",
    ) -> _Candidate:
        normalized_kind = str(kind or "evidence").strip().casefold()
        obligation_kinds = {"obligation", "proof_obligation"}
        receipt_kinds = {
            "proof",
            "proof_receipt",
            "validation",
            "validation_receipt",
            "merge",
            "merge_receipt",
            "receipt",
        }
        if obligation_id and (
            normalized_kind in obligation_kinds
            or (not task_id and not receipt_id and normalized_kind not in receipt_kinds)
        ):
            key = f"obligation:{obligation_id}"
        elif receipt_id and normalized_kind in receipt_kinds:
            key = f"receipt:{receipt_id}"
        elif task_id:
            key = f"task:{task_id}"
        elif obligation_id:
            key = f"obligation:{obligation_id}"
        elif receipt_id:
            key = f"receipt:{receipt_id}"
        elif criterion_id:
            key = f"criterion:{criterion_id}"
        elif symbol:
            key = f"symbol:{path}:{symbol}"
        elif stable_id:
            key = f"{kind}:{stable_id}"
        elif path:
            key = f"{kind}:path:{path}"
        else:
            key = _digest("anonymous-evidence", {"kind": kind})
        candidate = self._candidates.get(key)
        if candidate is None:
            candidate = _Candidate(
                key=key,
                kind=_bounded_text(kind or "evidence", 80),
                task_id=_bounded_text(task_id),
                obligation_id=_bounded_text(obligation_id),
                receipt_id=_bounded_text(receipt_id),
                criterion_id=_bounded_text(criterion_id),
                symbol=_bounded_text(symbol),
                path=_bounded_text(path),
            )
            self._candidates[key] = candidate
        return candidate

    def _reference(
        self,
        source_kind: str,
        source_id: str,
        *,
        record_id: str = "",
        provenance: str = "",
        artifact_id: str = "",
    ) -> EvidenceReference:
        return EvidenceReference(
            source_kind=source_kind,
            source_id=source_id,
            record_id=record_id,
            provenance=provenance,
            artifact_id=artifact_id or self.artifact_id,
        )

    def _ingest_graph(self, graph: Any) -> None:
        if graph is None:
            return
        if isinstance(graph, (str, Path)):
            from .artifact_store import read_code_evidence_graph

            graph = read_code_evidence_graph(graph)
        if hasattr(graph, "nodes") and not isinstance(graph, Mapping):
            nodes = _sequence(getattr(graph, "nodes", ()))
            edges = _sequence(getattr(graph, "edges", ()))
            graph_id = _bounded_text(getattr(graph, "graph_id", ""), 320)
        else:
            payload = _mapping(graph)
            nodes = _sequence(payload.get("nodes"))
            edges = _sequence(payload.get("edges"))
            graph_id = _first(payload, "graph_id")
        for value in nodes:
            row = _mapping(value)
            record = _mapping(row.get("record"))
            kind = _first(row, "kind", "node_kind") or "evidence"
            node_id = _first(row, "node_id") or _digest(
                "graph-node",
                {
                    "kind": kind,
                    "record_key": _first(row, "record_key"),
                },
            )
            task_id = _first(row, "task_id")
            if not task_id and kind == "task":
                task_id = _first(record, "task_id", "id")
            obligation_id = _first(row, "obligation_id")
            if not obligation_id and kind == "obligation":
                obligation_id = _first(record, "obligation_id", "id")
            record_key = _first(row, "record_key") or _first(
                record, "receipt_id", "record_id", "id"
            )
            receipt_id = (
                record_key
                if kind in {"proof", "validation", "merge", "receipt"}
                else _first(record, "receipt_id")
            )
            symbol = _first(row, "symbol") or _first(
                record, "qualified_name", "symbol", "name"
            )
            path = _first(record, "path", "file", "source_path")
            candidate = self._candidate(
                kind=kind,
                task_id=task_id,
                obligation_id=obligation_id,
                receipt_id=receipt_id,
                symbol=symbol if kind == "symbol" else "",
                path=path,
                stable_id=record_key or node_id,
            )
            candidate.title = candidate.title or _first(
                record, "title", "name", "label", "summary"
            )
            candidate.goal_id = candidate.goal_id or _first(record, "goal_id")
            candidate.status = candidate.status or _first(
                record, "status", "outcome", "freshness"
            )
            if symbol:
                candidate.symbols.add(symbol)
            if path:
                candidate.paths.add(path)
            candidate.absorb_text(
                candidate.title,
                task_id,
                obligation_id,
                record_key,
                symbol,
                path,
                candidate.goal_id,
                candidate.status,
                *self._safe_search_values(record),
            )
            reference = self._reference(
                "code_evidence_node",
                node_id,
                record_id=record_key,
                provenance=_first(row, "provenance"),
                artifact_id=graph_id,
            )
            candidate.add_reference(reference)
            self._node_keys[node_id] = candidate.key
            for alias in (node_id, record_key):
                if alias:
                    self._aliases.setdefault(alias, candidate.key)
            if task_id and candidate.key.startswith("task:"):
                self._aliases[task_id] = candidate.key
            if obligation_id and candidate.key.startswith("obligation:"):
                self._aliases[obligation_id] = candidate.key
            if receipt_id and candidate.key.startswith("receipt:"):
                self._aliases[receipt_id] = candidate.key

        proved_obligations: set[str] = set()
        for value in edges:
            row = _mapping(value)
            source_id = _first(row, "source", "source_node_id")
            target_id = _first(row, "target", "target_node_id")
            kind = _first(row, "kind", "edge_kind").casefold()
            source_key = self._node_keys.get(source_id)
            target_key = self._node_keys.get(target_id)
            if not source_key or not target_key:
                continue
            source = self._candidates[source_key]
            target = self._candidates[target_key]
            if kind == "depends_on":
                if source.task_id and target.task_id:
                    self._dependency_edges.add((source.task_id, target.task_id))
                elif source.obligation_id and target.obligation_id:
                    # Proof dependency neighborhoods remain represented through
                    # proof-gap scoring; task neighborhoods stay task-specific.
                    pass
            elif kind == "has_obligation":
                task = source.task_id or target.task_id
                obligation = source.obligation_id or target.obligation_id
                if task and obligation:
                    self._task_obligations.setdefault(task, set()).add(obligation)
            elif kind == "covers":
                obligation = source.obligation_id or target.obligation_id
                symbol = source.symbol or target.symbol
                if obligation and symbol:
                    self._obligation_symbols.setdefault(obligation, set()).add(symbol)
            elif kind == "proves":
                obligation = source.obligation_id or target.obligation_id
                if obligation:
                    proved_obligations.add(obligation)
                    item = self._candidates.get(f"obligation:{obligation}")
                    if item is not None:
                        item.proof_gap = max(item.proof_gap, 0.1)

        for candidate in self._candidates.values():
            if candidate.obligation_id and any(
                reference.source_kind == "code_evidence_node"
                for reference in candidate.references.values()
            ):
                candidate.proof_gap = (
                    0.1
                    if candidate.obligation_id in proved_obligations
                    else max(candidate.proof_gap, 1.0)
                )

    @staticmethod
    def _safe_search_values(record: Mapping[str, Any]) -> tuple[str, ...]:
        """Select bounded scalar/list fields for matching without retaining them."""

        allowed = (
            "title",
            "name",
            "label",
            "summary",
            "description",
            "task_id",
            "goal_id",
            "obligation_id",
            "receipt_id",
            "criterion",
            "acceptance",
            "acceptance_criteria",
            "missing_evidence",
            "outputs",
            "predicted_files",
            "changed_paths",
            "paths",
            "path",
            "interfaces",
            "ast_symbols",
            "symbols",
            "qualified_name",
            "validation",
            "track",
            "status",
            "kind",
            "scope_keys",
        )
        result: list[str] = []
        for name in allowed:
            value = record.get(name)
            if isinstance(value, (str, int, float, bool)):
                result.append(_bounded_text(value, 1024))
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for item in value[:128]:
                    if isinstance(item, (str, int, float, bool)):
                        result.append(_bounded_text(item))
                    elif isinstance(item, Mapping):
                        # Proof scope keys have a tiny, known shape.
                        kind = _first(item, "kind")
                        key_value = _first(item, "value")
                        if kind or key_value:
                            result.append(f"{kind}:{key_value}")
        return tuple(result)

    def _ingest_records(self, records: Iterable[Any], source_kind: str) -> None:
        for value in records:
            row = _mapping(value)
            if not row:
                continue
            task_id = _first(row, "task_id", "canonical_task_id")
            obligation_id = _first(row, "obligation_id")
            receipt_id = _first(row, "receipt_id", "validation_receipt_id")
            criterion_id = _first(row, "criterion_id")
            symbol = _first(row, "qualified_name", "symbol")
            path = _first(row, "path", "file", "source_path")
            kind = _first(row, "kind", "entity_kind", "type") or (
                "task" if task_id else "evidence"
            )
            explicit_stable_id = _first(
                row,
                "node_id",
                "record_id",
                "task_cid",
                "scope_id",
                "vector_key",
                "id",
            )
            stable_id = explicit_stable_id or _digest(
                f"{source_kind}-identity",
                {
                    "kind": kind,
                    "task_id": task_id,
                    "obligation_id": obligation_id,
                    "receipt_id": receipt_id,
                    "criterion_id": criterion_id,
                    "symbol": symbol,
                    "path": path,
                    "search_values": self._safe_search_values(row),
                },
            )
            candidate = self._candidate(
                kind=kind,
                task_id=task_id,
                obligation_id=obligation_id,
                receipt_id=receipt_id,
                criterion_id=criterion_id,
                symbol=symbol if not task_id and not obligation_id else "",
                path=path,
                stable_id=stable_id,
            )
            candidate.title = candidate.title or _first(
                row, "title", "name", "label", "summary", "acceptance"
            )
            candidate.goal_id = candidate.goal_id or _first(row, "goal_id")
            candidate.status = candidate.status or _first(row, "status", "outcome")
            candidate.symbols.update(_strings(row.get("ast_symbols") or row.get("symbols")))
            if symbol:
                candidate.symbols.add(symbol)
            candidate.paths.update(
                _strings(
                    row.get("predicted_files")
                    or row.get("paths")
                    or row.get("outputs")
                )
            )
            candidate.paths.update(_strings(row.get("changed_paths")))
            if path:
                candidate.paths.add(path)
            vector = row.get("embedding") or row.get("vector")
            if isinstance(vector, Sequence) and not isinstance(
                vector, (str, bytes, bytearray)
            ):
                try:
                    normalized = tuple(float(item) for item in vector)
                except (TypeError, ValueError):
                    normalized = ()
                if normalized and all(math.isfinite(item) for item in normalized):
                    candidate.vectors.append(normalized)
            candidate.absorb_text(
                candidate.title,
                task_id,
                obligation_id,
                receipt_id,
                criterion_id,
                candidate.goal_id,
                candidate.status,
                *candidate.symbols,
                *candidate.paths,
                *self._safe_search_values(row),
            )
            reference_id = explicit_stable_id or stable_id
            reference = self._reference(
                source_kind,
                reference_id,
                record_id=task_id or obligation_id or receipt_id or criterion_id,
                provenance=_first(row, "provenance", "provenance_cid"),
            )
            candidate.add_reference(reference)
            self._aliases.setdefault(stable_id, candidate.key)
            if task_id and candidate.key.startswith("task:"):
                self._aliases[task_id] = candidate.key
            if obligation_id and candidate.key.startswith("obligation:"):
                self._aliases[obligation_id] = candidate.key
            if receipt_id and candidate.key.startswith("receipt:"):
                self._aliases[receipt_id] = candidate.key
            if criterion_id and candidate.key.startswith("criterion:"):
                self._aliases[criterion_id] = candidate.key

    def _ingest_dependency_graph(self, graph: Any) -> None:
        if graph is None:
            return
        payload = _mapping(graph)
        nodes_value = (
            getattr(graph, "nodes", None)
            if not isinstance(graph, Mapping)
            else payload.get("nodes")
        )
        edges_value = (
            getattr(graph, "edges", None)
            if not isinstance(graph, Mapping)
            else payload.get("edges")
        )
        if isinstance(nodes_value, Mapping):
            nodes = tuple(nodes_value.values())
        else:
            nodes = _sequence(nodes_value)
        for value in nodes:
            row = _mapping(value)
            task_id = _first(row, "task_id", "task_cid")
            if not task_id:
                continue
            candidate = self._candidate(kind="task", task_id=task_id)
            candidate.title = candidate.title or _first(row, "title", "name")
            candidate.goal_id = candidate.goal_id or _first(row, "goal_id")
            candidate.status = candidate.status or _first(row, "status")
            candidate.absorb_text(
                task_id, candidate.title, candidate.goal_id, candidate.status
            )
            source_id = _first(row, "task_cid") or task_id
            candidate.add_reference(
                self._reference(
                    "task_dependency_node",
                    source_id,
                    record_id=task_id,
                    provenance="task",
                )
            )
            self._aliases.setdefault(task_id, candidate.key)
            self._aliases.setdefault(source_id, candidate.key)
        for value in _sequence(edges_value):
            row = _mapping(value)
            source = _first(
                row, "source_task_cid", "source_task_id", "source", "from"
            )
            target = _first(
                row, "target_task_cid", "target_task_id", "target", "to"
            )
            source_key = self._aliases.get(source, f"task:{source}")
            target_key = self._aliases.get(target, f"task:{target}")
            source_candidate = self._candidates.get(source_key)
            target_candidate = self._candidates.get(target_key)
            if source_candidate and target_candidate:
                source_task = source_candidate.task_id
                target_task = target_candidate.task_id
                if source_task and target_task and source_task != target_task:
                    self._dependency_edges.add((source_task, target_task))

    def _ingest_goal_coverage(self, coverage: Any) -> None:
        if coverage is None:
            return
        if hasattr(coverage, "criteria") and not isinstance(coverage, Mapping):
            criteria = _sequence(getattr(coverage, "criteria", ()))
            edges = _sequence(getattr(coverage, "edges", ()))
            graph_id = _bounded_text(getattr(coverage, "graph_id", ""), 320)
        else:
            payload = _mapping(coverage)
            criteria = _sequence(payload.get("criteria"))
            edges = _sequence(payload.get("edges"))
            graph_id = _first(payload, "graph_id")
        criterion_goals: dict[str, str] = {}
        criterion_statuses: dict[str, str] = {}
        for value in criteria:
            row = _mapping(value)
            criterion_id = _first(row, "criterion_id", "id")
            goal_id = _first(row, "goal_id")
            if not criterion_id:
                criterion_id = _digest(
                    "criterion",
                    {"goal_id": goal_id, "criterion": _first(row, "criterion")},
                )
            candidate = self._candidate(
                kind="acceptance_criterion", criterion_id=criterion_id
            )
            candidate.title = candidate.title or _first(
                row, "criterion", "title", "name"
            )
            candidate.goal_id = candidate.goal_id or goal_id
            candidate.status = candidate.status or _first(row, "status")
            if candidate.status:
                candidate.goal_statuses.add(candidate.status.casefold())
            candidate.absorb_text(
                candidate.title,
                goal_id,
                criterion_id,
                candidate.status,
                *self._safe_search_values(row),
            )
            candidate.add_reference(
                self._reference(
                    "goal_coverage_criterion",
                    criterion_id,
                    record_id=goal_id,
                    provenance="goal_coverage",
                    artifact_id=graph_id,
                )
            )
            criterion_goals[criterion_id] = goal_id
            criterion_statuses[criterion_id] = candidate.status.casefold()
            for task_id in _strings(row.get("task_ids")):
                task = self._candidate(kind="task", task_id=task_id)
                task.goal_id = task.goal_id or goal_id
                if candidate.status:
                    task.goal_statuses.add(candidate.status.casefold())
                task.absorb_text(candidate.title, goal_id)

        for value in edges:
            row = _mapping(value)
            criterion_id = _first(row, "criterion_id", "source")
            goal_id = _first(row, "goal_id") or criterion_goals.get(criterion_id, "")
            task_id = _first(row, "task_id")
            status = _first(row, "status").casefold() or criterion_statuses.get(
                criterion_id, ""
            )
            if task_id:
                candidate = self._candidate(kind="task", task_id=task_id)
            elif criterion_id and f"criterion:{criterion_id}" in self._candidates:
                candidate = self._candidates[f"criterion:{criterion_id}"]
            else:
                edge_id = _first(row, "edge_id") or _digest(
                    "coverage-edge",
                    {
                        "criterion_id": criterion_id,
                        "goal_id": goal_id,
                        "value": _first(row, "value"),
                    },
                )
                candidate = self._candidate(
                    kind="coverage_surface", stable_id=edge_id
                )
            candidate.goal_id = candidate.goal_id or goal_id
            if status:
                candidate.goal_statuses.add(status)
                candidate.status = candidate.status or status
            candidate.absorb_text(
                criterion_id,
                goal_id,
                _first(row, "value"),
                _first(row, "relation", "kind"),
                status,
            )
            edge_id = _first(row, "edge_id") or _digest(
                "coverage-edge-reference",
                {
                    "criterion_id": criterion_id,
                    "goal_id": goal_id,
                    "task_id": task_id,
                    "value": _first(row, "value"),
                    "status": status,
                },
            )
            candidate.add_reference(
                self._reference(
                    "goal_coverage_edge",
                    edge_id,
                    record_id=criterion_id or goal_id,
                    provenance="goal_coverage",
                    artifact_id=graph_id,
                )
            )

    def _ingest_proof_scope(self, index: Any) -> None:
        if index is None:
            return
        if hasattr(index, "obligations") and not isinstance(index, Mapping):
            obligations = _sequence(getattr(index, "obligations", ()))
            receipts = _sequence(getattr(index, "receipts", ()))
            invalidations = _sequence(getattr(index, "invalidations", ()))
            scopes = _sequence(getattr(index, "scope_records", ()))
            index_id = _bounded_text(getattr(index, "index_id", ""), 320)
            active_obligations = set(
                _strings(getattr(index, "active_obligation_ids", ()))
            )
            active_receipts = set(_strings(getattr(index, "active_receipt_ids", ())))
        else:
            payload = _mapping(index)
            obligations = _sequence(payload.get("obligations"))
            receipts = _sequence(payload.get("receipts"))
            invalidations = _sequence(payload.get("invalidations"))
            scopes = tuple(
                scope
                for blob in _sequence(payload.get("blobs"))
                for scope in _sequence(_mapping(blob).get("scopes"))
            )
            index_id = _first(payload, "index_id")
            stale_obligations = {
                _first(_mapping(item), "subject_id")
                for item in invalidations
                if _first(_mapping(item), "subject_kind") == "obligation"
            }
            stale_receipts = {
                _first(_mapping(item), "subject_id")
                for item in invalidations
                if _first(_mapping(item), "subject_kind") == "receipt"
            }
            active_obligations = {
                _first(_mapping(item), "obligation_id") for item in obligations
            } - stale_obligations
            active_receipts = {
                _first(_mapping(item), "receipt_id") for item in receipts
            } - stale_receipts

        receipts_by_obligation: dict[str, set[str]] = {}
        for value in receipts:
            row = _mapping(value)
            receipt_id = _first(row, "receipt_id")
            obligation_id = _first(row, "obligation_id")
            if not receipt_id or not obligation_id:
                continue
            receipts_by_obligation.setdefault(obligation_id, set()).add(receipt_id)
            candidate = self._candidate(
                kind="proof_receipt", receipt_id=receipt_id
            )
            candidate.obligation_id = candidate.obligation_id or obligation_id
            candidate.status = candidate.status or (
                "active" if receipt_id in active_receipts else "invalidated"
            )
            candidate.proof_gap = 0.0 if receipt_id in active_receipts else 0.9
            candidate.absorb_text(
                receipt_id,
                obligation_id,
                candidate.status,
                *self._safe_search_values(row),
            )
            candidate.add_reference(
                self._reference(
                    "proof_scope_receipt",
                    receipt_id,
                    record_id=obligation_id,
                    provenance="proof",
                    artifact_id=index_id,
                )
            )
            self._aliases[receipt_id] = candidate.key

        for value in obligations:
            row = _mapping(value)
            obligation_id = _first(row, "obligation_id")
            if not obligation_id:
                continue
            candidate = self._candidate(
                kind="proof_obligation", obligation_id=obligation_id
            )
            receipt_ids = receipts_by_obligation.get(obligation_id, set())
            active = sorted(receipt_ids & active_receipts)
            if obligation_id not in active_obligations:
                candidate.proof_gap = 1.0
                candidate.status = candidate.status or "invalidated"
            elif not active:
                candidate.proof_gap = max(candidate.proof_gap, 1.0)
                candidate.status = candidate.status or "unproved"
            else:
                candidate.proof_gap = max(candidate.proof_gap, 0.1)
                candidate.status = candidate.status or "proved"
            candidate.absorb_text(
                obligation_id,
                candidate.status,
                *self._safe_search_values(row),
            )
            candidate.add_reference(
                self._reference(
                    "proof_scope_obligation",
                    obligation_id,
                    record_id=obligation_id,
                    provenance="proof",
                    artifact_id=index_id,
                )
            )
            self._aliases[obligation_id] = candidate.key

        for value in scopes:
            row = _mapping(value)
            scope_id = _first(row, "scope_id")
            path = _first(row, "path")
            if not scope_id:
                continue
            candidate = self._candidate(
                kind="proof_scope", path=path, stable_id=scope_id
            )
            candidate.title = candidate.title or scope_id
            if path:
                candidate.paths.add(path)
            candidate.absorb_text(scope_id, path, *self._safe_search_values(row))
            candidate.add_reference(
                self._reference(
                    "proof_scope",
                    scope_id,
                    record_id=_first(row, "blob_id"),
                    provenance="proof",
                    artifact_id=index_id,
                )
            )

        for value in invalidations:
            row = _mapping(value)
            subject_id = _first(row, "subject_id")
            key = self._aliases.get(subject_id)
            candidate = self._candidates.get(key or "")
            if candidate is not None:
                candidate.proof_gap = max(candidate.proof_gap, 1.0)
                candidate.status = candidate.status or "invalidated"

    def _propagate_graph_context(self) -> None:
        for task_id, obligations in self._task_obligations.items():
            task = self._candidates.get(f"task:{task_id}")
            if task is None:
                continue
            for obligation_id in sorted(obligations):
                obligation = self._candidates.get(f"obligation:{obligation_id}")
                if obligation is not None:
                    task.proof_gap = max(task.proof_gap, obligation.proof_gap * 0.85)
                    task.absorb_text(obligation_id)
                    for reference in obligation.references.values():
                        task.add_reference(reference)
                for symbol in self._obligation_symbols.get(obligation_id, ()):
                    task.symbols.add(symbol)
                    task.absorb_text(symbol)

    def _base_health(self) -> dict[str, BackendHealth]:
        ast_candidates = sum(
            bool(item.symbols or item.symbol)
            for item in self._candidates.values()
        )
        dependency_candidates = len(
            {
                task_id
                for edge in self._dependency_edges
                for task_id in edge
            }
        )
        coverage_candidates = sum(
            bool(item.goal_id or item.goal_statuses)
            for item in self._candidates.values()
        )
        proof_candidates = sum(
            bool(item.obligation_id or item.proof_gap)
            for item in self._candidates.values()
        )
        return {
            "lexical": BackendHealth(
                "lexical",
                BackendState.HEALTHY,
                "built-in deterministic token matcher",
                len(self._candidates),
            ),
            "vector": BackendHealth(
                "vector",
                BackendState.UNAVAILABLE,
                "no query vector or vector backend",
                0,
            ),
            "ast_symbol": BackendHealth(
                "ast_symbol",
                BackendState.HEALTHY if ast_candidates else BackendState.UNAVAILABLE,
                (
                    "compact symbol index available"
                    if ast_candidates
                    else "no AST-symbol evidence or backend"
                ),
                ast_candidates,
            ),
            "dependency_neighborhood": BackendHealth(
                "dependency_neighborhood",
                BackendState.HEALTHY
                if self._dependency_edges
                else BackendState.UNAVAILABLE,
                (
                    "bounded dependency adjacency available"
                    if self._dependency_edges
                    else "no dependency graph evidence"
                ),
                dependency_candidates,
            ),
            "goal_coverage": BackendHealth(
                "goal_coverage",
                BackendState.HEALTHY
                if self._coverage_present
                else BackendState.UNAVAILABLE,
                (
                    "goal coverage map available"
                    if self._coverage_present
                    else "no goal coverage map"
                ),
                coverage_candidates,
            ),
            "proof_gap": BackendHealth(
                "proof_gap",
                BackendState.HEALTHY
                if self._proof_present or proof_candidates
                else BackendState.UNAVAILABLE,
                (
                    "proof obligations and invalidations available"
                    if self._proof_present or proof_candidates
                    else "no proof-scope evidence"
                ),
                proof_candidates,
            ),
        }

    @staticmethod
    def _backend_health(backend: Any) -> tuple[bool, str]:
        health = getattr(backend, "health", None)
        if not callable(health):
            health = getattr(backend, "health_check", None)
        if not callable(health):
            return True, "backend callable"
        value = health()
        if isinstance(value, Mapping):
            status = str(value.get("status") or value.get("state") or "").casefold()
            healthy = bool(value.get("healthy", status in {"healthy", "ok", "ready"}))
            detail = _bounded_text(
                value.get("detail") or value.get("reason") or status or "health response",
                _MAX_DETAIL,
            )
            return healthy, detail
        return (
            bool(value),
            "backend health check passed"
            if value
            else "backend health check failed",
        )

    @staticmethod
    def _invoke_backend(
        backend: Any,
        query: RetrievalQuery,
        limit: int,
        *,
        method_names: tuple[str, ...],
    ) -> Any:
        method = None
        for name in method_names:
            candidate = getattr(backend, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None and callable(backend):
            method = backend
        if method is None:
            raise TypeError("backend has no supported query method")
        attempts = (
            lambda: method(query.text, limit=limit),
            lambda: method(query=query.text, limit=limit),
            lambda: method(query.to_dict(), limit=limit),
            lambda: method(query.text),
        )
        last_error: TypeError | None = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _backend_scores(
        self,
        backend: Any,
        query: RetrievalQuery,
        limit: int,
        *,
        method_names: tuple[str, ...],
    ) -> dict[str, float]:
        response = self._invoke_backend(
            backend, query, limit, method_names=method_names
        )
        if isinstance(response, Mapping):
            nested = (
                response.get("results")
                or response.get("matches")
                or response.get("rows")
            )
            if nested is None and all(
                isinstance(value, (int, float)) for value in response.values()
            ):
                response = [
                    {"id": key, "score": value}
                    for key, value in sorted(response.items(), key=lambda pair: str(pair[0]))
                ]
            elif isinstance(nested, Mapping):
                response = [
                    {"id": key, "score": value}
                    for key, value in sorted(nested.items(), key=lambda pair: str(pair[0]))
                    if isinstance(value, (int, float))
                ]
            else:
                response = nested or ()
        values = _sequence(response)
        scores: dict[str, float] = {}
        for value in values[:limit]:
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray, Mapping)
            ):
                row = {"id": value[0], "score": value[1]} if len(value) >= 2 else {}
            else:
                row = _mapping(value)
            identity = _first(
                row,
                "evidence_id",
                "candidate_id",
                "node_id",
                "task_id",
                "obligation_id",
                "receipt_id",
                "record_id",
                "id",
            )
            key = self._aliases.get(identity)
            if identity in self._candidates:
                key = identity
            if identity.startswith("analysis-evidence:"):
                key = next(
                    (
                        candidate.key
                        for candidate in self._candidates.values()
                        if candidate.evidence_id == identity
                    ),
                    key,
                )
            if key and key in self._candidates:
                scores[key] = max(
                    scores.get(key, 0.0),
                    _score(row.get("score", row.get("similarity", 0.0))),
                )
        return scores

    def _vector_scores(
        self,
        query: RetrievalQuery,
        limits: RetrievalLimits,
        health: dict[str, BackendHealth],
    ) -> dict[str, float]:
        if self.vector_backend is not None:
            try:
                healthy, detail = self._backend_health(self.vector_backend)
                if not healthy:
                    health["vector"] = BackendHealth(
                        "vector", BackendState.UNHEALTHY, detail, 0
                    )
                    return {}
                scores = self._backend_scores(
                    self.vector_backend,
                    query,
                    limits.max_backend_results,
                    method_names=("search", "query", "retrieve"),
                )
                health["vector"] = BackendHealth(
                    "vector",
                    BackendState.HEALTHY,
                    detail,
                    len(scores),
                )
                return scores
            except Exception as exc:
                health["vector"] = BackendHealth(
                    "vector",
                    BackendState.UNHEALTHY,
                    f"{type(exc).__name__}: vector backend query failed",
                    0,
                )
                return {}

        vector = query.embedding
        if not vector and self.vector_embedder is not None:
            try:
                vector = tuple(float(item) for item in self.vector_embedder(query.text))
                if not vector or not all(math.isfinite(item) for item in vector):
                    raise ValueError("embedder returned an empty or non-finite vector")
            except Exception as exc:
                health["vector"] = BackendHealth(
                    "vector",
                    BackendState.UNHEALTHY,
                    f"{type(exc).__name__}: vector embedding failed",
                    0,
                )
                return {}
        if not vector:
            return {}
        scores: dict[str, float] = {}
        for key, candidate in self._candidates.items():
            values = [
                _cosine(vector, item)
                for item in candidate.vectors
                if len(item) == len(vector)
            ]
            if values:
                scores[key] = max(values)
        if scores:
            health["vector"] = BackendHealth(
                "vector",
                BackendState.HEALTHY,
                "local compact vectors available",
                len(scores),
            )
        else:
            health["vector"] = BackendHealth(
                "vector",
                BackendState.UNAVAILABLE,
                "query vector supplied but no compatible evidence vectors",
                0,
            )
        return scores

    def _ast_backend_scores(
        self,
        query: RetrievalQuery,
        limits: RetrievalLimits,
        health: dict[str, BackendHealth],
    ) -> dict[str, float]:
        if self.ast_backend is None:
            return {}
        try:
            healthy, detail = self._backend_health(self.ast_backend)
            if not healthy:
                health["ast_symbol"] = BackendHealth(
                    "ast_symbol", BackendState.UNHEALTHY, detail, 0
                )
                return {}
            scores = self._backend_scores(
                self.ast_backend,
                query,
                limits.max_backend_results,
                method_names=("search_symbols", "search", "query"),
            )
            health["ast_symbol"] = BackendHealth(
                "ast_symbol", BackendState.HEALTHY, detail, len(scores)
            )
            return scores
        except Exception as exc:
            health["ast_symbol"] = BackendHealth(
                "ast_symbol",
                BackendState.UNHEALTHY,
                f"{type(exc).__name__}: AST backend query failed",
                0,
            )
            return {}

    @staticmethod
    def _lexical(query_tokens: frozenset[str], candidate: _Candidate) -> float:
        if not query_tokens or not candidate.terms:
            return 0.0
        overlap = query_tokens & candidate.terms
        if not overlap:
            return 0.0
        query_coverage = len(overlap) / len(query_tokens)
        jaccard = len(overlap) / len(query_tokens | candidate.terms)
        exact_bonus = 0.1 if query_tokens <= candidate.terms else 0.0
        return _score(0.72 * query_coverage + 0.28 * jaccard + exact_bonus)

    @staticmethod
    def _ast_score(query: RetrievalQuery, candidate: _Candidate) -> float:
        symbols = set(candidate.symbols)
        if candidate.symbol:
            symbols.add(candidate.symbol)
        if not symbols:
            return 0.0
        expected = {item.casefold() for item in query.symbols}
        symbol_values = {item.casefold() for item in symbols}
        if expected:
            exact = expected & symbol_values
            leaf_values = {
                re.split(r"[.:/]", item)[-1] for item in symbol_values
            }
            leaf_matches = {
                item
                for item in expected
                if re.split(r"[.:/]", item)[-1] in leaf_values
            }
            return _score(
                len(exact) / len(expected)
                + 0.75 * len(leaf_matches - exact) / len(expected)
            )
        query_tokens = _tokens(query.text)
        symbol_tokens = frozenset(
            token
            for symbol in symbols
            for token in _tokens(symbol)
        )
        overlap = query_tokens & symbol_tokens
        return _score(0.8 * len(overlap) / max(1, len(query_tokens)))

    def _dependency_scores(
        self, query: RetrievalQuery, limits: RetrievalLimits
    ) -> dict[str, float]:
        if not self._dependency_edges:
            return {}
        seeds = set(query.task_ids)
        query_tokens = _tokens(query.text)
        for candidate in self._candidates.values():
            if candidate.task_id and candidate.task_id.casefold() in query_tokens:
                seeds.add(candidate.task_id)
        if not seeds:
            return {}
        adjacency: dict[str, set[str]] = {}
        for source, target in self._dependency_edges:
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        distances = {seed: 0 for seed in seeds}
        frontier = sorted(seeds)
        for distance in range(1, limits.max_hops + 1):
            next_frontier: list[str] = []
            for task_id in frontier:
                for neighbor in sorted(adjacency.get(task_id, ())):
                    if neighbor not in distances:
                        distances[neighbor] = distance
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return {
            f"task:{task_id}": _score(1.0 / (distance + 1))
            for task_id, distance in distances.items()
            if f"task:{task_id}" in self._candidates
        }

    @staticmethod
    def _coverage_score(query: RetrievalQuery, candidate: _Candidate) -> float:
        if not candidate.goal_id and not candidate.goal_statuses:
            return 0.0
        if query.goal_ids and candidate.goal_id not in query.goal_ids:
            return 0.0
        if not query.goal_ids:
            query_tokens = _tokens(query.text)
            if candidate.goal_id and candidate.goal_id.casefold() not in query_tokens:
                if not (query_tokens & candidate.terms):
                    return 0.0
        priority = {
            "contradicted": 1.0,
            "uncovered": 1.0,
            "stale": 0.85,
            "weakly_inferred": 0.65,
            "verified": 0.35,
        }
        statuses = set(candidate.goal_statuses)
        if candidate.status:
            statuses.add(candidate.status.casefold())
        status_score = max((priority.get(item, 0.5) for item in statuses), default=0.5)
        return _score(status_score)

    @staticmethod
    def _proof_score(query: RetrievalQuery, candidate: _Candidate) -> float:
        if candidate.proof_gap <= 0.0 and not candidate.obligation_id:
            return 0.0
        if query.obligation_ids:
            if candidate.obligation_id not in query.obligation_ids:
                return 0.0
        elif candidate.obligation_id:
            query_tokens = _tokens(query.text)
            if (
                candidate.obligation_id.casefold() not in query_tokens
                and not (query_tokens & candidate.terms)
            ):
                # General gap-oriented queries should still retrieve gaps.
                if not query_tokens.intersection(
                    {"proof", "gap", "unproved", "invalidated", "stale", "obligation"}
                ):
                    return 0.0
        return _score(candidate.proof_gap)

    @staticmethod
    def _explanation(
        signal_scores: Mapping[str, SignalScore],
        health: Mapping[str, BackendHealth],
    ) -> str:
        contributing = [
            f"{name}={signal_scores[name].score:.3f}"
            for name in SIGNAL_ORDER
            if signal_scores[name].contribution > 0.0
        ]
        degraded = [
            f"{name}:{health[name].state.value}"
            for name in SIGNAL_ORDER
            if health[name].state is not BackendState.HEALTHY
        ]
        text = (
            "Fixed-weight fusion; "
            + (", ".join(contributing) if contributing else "no positive signal")
        )
        if degraded:
            text += "; zero contribution from " + ", ".join(degraded)
        return _bounded_text(text, _MAX_TEXT)

    def _rank(
        self,
        query: RetrievalQuery,
        limits: RetrievalLimits,
        health: dict[str, BackendHealth],
    ) -> tuple[list[RetrievalResult], int]:
        query_tokens = _tokens(
            " ".join(
                (
                    query.text,
                    *query.task_ids,
                    *query.goal_ids,
                    *query.symbols,
                    *query.obligation_ids,
                    *query.paths,
                )
            )
        )
        vector_scores = self._vector_scores(query, limits, health)
        ast_backend_scores = self._ast_backend_scores(query, limits, health)
        dependency_scores = self._dependency_scores(query, limits)
        ranked: list[RetrievalResult] = []
        reference_truncations = 0
        available = {
            name: health[name].state is BackendState.HEALTHY for name in SIGNAL_ORDER
        }
        for key in sorted(self._candidates):
            candidate = self._candidates[key]
            ast_score = max(
                self._ast_score(query, candidate),
                ast_backend_scores.get(key, 0.0),
            )
            raw_scores = {
                "lexical": self._lexical(query_tokens, candidate),
                "vector": vector_scores.get(key, 0.0),
                "ast_symbol": ast_score,
                "dependency_neighborhood": dependency_scores.get(key, 0.0),
                "goal_coverage": self._coverage_score(query, candidate)
                if available["goal_coverage"]
                else 0.0,
                "proof_gap": self._proof_score(query, candidate)
                if available["proof_gap"]
                else 0.0,
            }
            signal_scores: dict[str, SignalScore] = {}
            for name in SIGNAL_ORDER:
                score = _score(raw_scores[name]) if available[name] else 0.0
                contribution = round(score * self.weights[name], 6)
                signal_scores[name] = SignalScore(
                    score=score,
                    weight=self.weights[name],
                    contribution=contribution,
                    available=available[name],
                    explanation=(
                        "matched"
                        if score > 0.0
                        else "no match"
                        if available[name]
                        else f"backend {health[name].state.value}"
                    ),
                )
            fused = round(
                sum(item.contribution for item in signal_scores.values()), 6
            )
            if fused <= 0.0:
                continue
            references = tuple(
                candidate.references[item]
                for item in sorted(candidate.references)
            )
            if not references:
                references = (
                    self._reference(
                        "retrieval_projection",
                        candidate.key,
                        record_id=candidate.evidence_id,
                        provenance="deterministic_projection",
                    ),
                )
            if len(references) > _MAX_REFERENCES_PER_RESULT:
                reference_truncations += len(references) - _MAX_REFERENCES_PER_RESULT
                references = references[:_MAX_REFERENCES_PER_RESULT]
            ranked.append(
                RetrievalResult(
                    evidence_id=candidate.evidence_id,
                    entity_kind=candidate.kind,
                    title=_bounded_text(candidate.title),
                    task_id=candidate.task_id,
                    goal_id=candidate.goal_id,
                    obligation_id=candidate.obligation_id,
                    symbol=candidate.symbol
                    or (sorted(candidate.symbols)[0] if candidate.symbols else ""),
                    path=candidate.path
                    or (sorted(candidate.paths)[0] if candidate.paths else ""),
                    status=candidate.status,
                    score=fused,
                    signal_scores=signal_scores,
                    ranking_explanation=self._explanation(signal_scores, health),
                    evidence_references=references,
                )
            )
        ranked.sort(
            key=lambda item: (
                -item.score,
                -sum(
                    score.score > 0.0 for score in item.signal_scores.values()
                ),
                item.evidence_id,
            )
        )
        return ranked, reference_truncations

    def retrieve(
        self,
        query: RetrievalQuery | str | Mapping[str, Any],
        *,
        limits: RetrievalLimits | Mapping[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Return a deterministic response within count and UTF-8 byte limits."""

        request = RetrievalQuery.from_value(query)
        bounds = RetrievalLimits.from_value(limits)
        health = self._base_health()
        ranked, reference_truncations = self._rank(request, bounds, health)
        considered = len(self._candidates)
        candidate_limited = ranked[: bounds.max_candidates]
        dropped_candidate = max(0, len(ranked) - len(candidate_limited))
        count_limited = candidate_limited[: bounds.max_results]
        dropped_count = max(0, len(candidate_limited) - len(count_limited))

        selected: list[RetrievalResult] = []
        dropped_bytes = 0

        def response_for(
            items: Sequence[RetrievalResult],
            *,
            byte_drops: int,
            output_bytes: int = 0,
        ) -> RetrievalResponse:
            truncated = bool(
                dropped_candidate
                or dropped_count
                or byte_drops
                or reference_truncations
            )
            return RetrievalResponse(
                query_id=request.query_id,
                results=tuple(items),
                backend_health=health,
                weights=self.weights,
                truncation=TruncationMetadata(
                    considered_count=considered,
                    eligible_count=len(ranked),
                    returned_count=len(items),
                    dropped_by_candidate_limit=dropped_candidate,
                    dropped_by_count_limit=dropped_count,
                    dropped_by_byte_limit=byte_drops,
                    reference_truncation_count=reference_truncations,
                    max_candidates=bounds.max_candidates,
                    max_results=bounds.max_results,
                    max_bytes=bounds.max_bytes,
                    max_hops=bounds.max_hops,
                    max_backend_results=bounds.max_backend_results,
                    output_bytes=output_bytes,
                    truncated=truncated,
                ),
            )

        for result in count_limited:
            proposed = [*selected, result]
            remaining = len(count_limited) - len(proposed)
            trial = response_for(proposed, byte_drops=remaining)
            if len(trial.to_json().encode("utf-8")) <= bounds.max_bytes:
                selected = proposed
            else:
                dropped_bytes += 1
        dropped_bytes += len(count_limited) - len(selected) - dropped_bytes
        response = response_for(selected, byte_drops=dropped_bytes)

        # output_bytes participates in its own encoding length.  Converge in a
        # handful of steps, removing the lowest-ranked record if digit growth
        # crosses the bound.
        output_bytes = 0
        for _ in range(8):
            response = response_for(
                selected, byte_drops=dropped_bytes, output_bytes=output_bytes
            )
            encoded = len(response.to_json().encode("utf-8"))
            if encoded > bounds.max_bytes and selected:
                selected.pop()
                dropped_bytes += 1
                output_bytes = 0
                continue
            if encoded == output_bytes:
                break
            output_bytes = encoded
        response = response_for(
            selected, byte_drops=dropped_bytes, output_bytes=output_bytes
        )
        final_size = len(response.to_json().encode("utf-8"))
        if final_size != response.truncation.output_bytes:
            response = response_for(
                selected, byte_drops=dropped_bytes, output_bytes=final_size
            )
            final_size = len(response.to_json().encode("utf-8"))
        if final_size > bounds.max_bytes:
            raise RetrievalBudgetError(
                "max_bytes is too small for mandatory retrieval metadata"
            )
        return response

    query = retrieve
    search = retrieve


def retrieve_analysis_evidence(
    query: RetrievalQuery | str | Mapping[str, Any],
    *,
    evidence_graph: Any = None,
    records: Iterable[Any] = (),
    todo_records: Iterable[Any] = (),
    dependency_graph: Any = None,
    goal_coverage: Any = None,
    proof_scope_index: Any = None,
    vector_backend: Any = None,
    vector_embedder: Callable[[str], Sequence[float]] | None = None,
    ast_backend: Any = None,
    artifact_id: str = "",
    signal_weights: Mapping[str, float] | None = None,
    limits: RetrievalLimits | Mapping[str, Any] | None = None,
) -> RetrievalResponse:
    """Convenience entry point for one bounded retrieval."""

    retriever = BoundedGraphRAGRetriever(
        evidence_graph=evidence_graph,
        records=records,
        todo_records=todo_records,
        dependency_graph=dependency_graph,
        goal_coverage=goal_coverage,
        proof_scope_index=proof_scope_index,
        vector_backend=vector_backend,
        vector_embedder=vector_embedder,
        ast_backend=ast_backend,
        artifact_id=artifact_id,
        signal_weights=signal_weights,
    )
    return retriever.retrieve(query, limits=limits)


# Descriptive aliases for callers that use the task title's terminology.
AnalysisRetriever = BoundedGraphRAGRetriever
MultiSignalGraphRAGRetriever = BoundedGraphRAGRetriever
GraphRAGRetriever = BoundedGraphRAGRetriever
AnalysisRetrievalQuery = RetrievalQuery
AnalysisRetrievalLimits = RetrievalLimits
AnalysisRetrievalResponse = RetrievalResponse
AnalysisRetrievalError = RetrievalValidationError
retrieve_graph_evidence = retrieve_analysis_evidence
retrieve = retrieve_analysis_evidence


__all__ = [
    "ANALYSIS_EVIDENCE_REFERENCE_SCHEMA",
    "ANALYSIS_RETRIEVAL_SCHEMA",
    "DEFAULT_SIGNAL_WEIGHTS",
    "SIGNAL_ORDER",
    "AnalysisRetrievalLimits",
    "AnalysisRetrievalError",
    "AnalysisRetrievalQuery",
    "AnalysisRetrievalResponse",
    "AnalysisRetriever",
    "BackendHealth",
    "BackendState",
    "BoundedGraphRAGRetriever",
    "EvidenceReference",
    "GraphRAGRetriever",
    "MultiSignalGraphRAGRetriever",
    "RetrievalLimits",
    "RetrievalBudgetError",
    "RetrievalQuery",
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalValidationError",
    "SignalScore",
    "TruncationMetadata",
    "retrieve",
    "retrieve_analysis_evidence",
    "retrieve_graph_evidence",
]
