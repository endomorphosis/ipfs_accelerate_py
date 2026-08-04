"""Compile live, body-free hybrid evidence for Planner and Doctor.

``PlanningEvidenceBundle@1`` is the composition boundary over existing AST,
lexical, graph, vector, lineage, contract, value-provenance, test, and proof
providers.  Providers are queried lazily and their output is treated as
untrusted data.  Only compact references and labels enter the bundle.

Retrieval and history lanes are nomination-only.  They can improve recall and
ranking, but they cannot satisfy proof, authorization, mutation, promotion, or
completion claims.  Required evidence coverage fails closed: each missing slot
is either represented by a bounded scheduled query or makes the coverage
receipt reject planning.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from .analysis_retrieval import (
    BackendState,
    RetrievalResponse,
    assess_retrieval_index_health,
)


PLANNING_EVIDENCE_BUNDLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/planning-evidence-bundle@1"
)
EVIDENCE_COVERAGE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/evidence-coverage-receipt@1"
)
PLANNING_EVIDENCE_ITEM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/planning-evidence-item@1"
)
EVIDENCE_QUERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/planning-evidence-query@1"
)
PLANNING_EVIDENCE_BUNDLE_INTERFACE = "PlanningEvidenceBundle@1"
EVIDENCE_COVERAGE_RECEIPT_INTERFACE = "EvidenceCoverageReceipt@1"

_MAX_TEXT = 512
_MAX_RESULTS_PER_SLOT = 128
_MAX_RESULTS = 512
_MAX_BYTES = 2 * 1024 * 1024
_SPACE_RE = re.compile(r"\s+")
_FORBIDDEN_BODY_KEYS = frozenset(
    {
        "ast",
        "ast_body",
        "body",
        "completion",
        "content",
        "decoded",
        "decoded_model_text",
        "embedding",
        "graph",
        "instructions",
        "messages",
        "metadata",
        "model_output",
        "model_response",
        "nested_graph",
        "payload",
        "prompt",
        "raw",
        "raw_text",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "text",
        "transcript",
        "vector",
    }
)


class PlanningEvidenceError(ValueError):
    """Base error for invalid evidence inputs and receipts."""


class MissingRequiredEvidenceError(PlanningEvidenceError):
    """Required coverage was neither satisfied nor scheduled."""

    def __init__(self, slots: Iterable["EvidenceSlot"]) -> None:
        self.slots = tuple(sorted(set(slots), key=lambda item: item.value))
        super().__init__(
            "missing required evidence slots: "
            + ", ".join(item.value for item in self.slots)
        )


class EvidenceSlot(str, Enum):
    """Closed evidence coverage vocabulary."""

    AST = "ast"
    BM25 = "bm25"
    KG_GRAPHRAG = "kg_graphrag"
    VECTOR_EMBEDDING = "vector_embedding"
    LINEAGE_HISTORY = "lineage_history"
    CONTRACTS = "contracts"
    VALUE_PROVENANCE = "value_provenance"
    TESTS = "tests"
    PROOFS = "proofs"


EVIDENCE_SLOT_ORDER = tuple(EvidenceSlot)
DEFAULT_REQUIRED_EVIDENCE_SLOTS = EVIDENCE_SLOT_ORDER
NOMINATION_ONLY_SLOTS = frozenset(
    {
        EvidenceSlot.BM25,
        EvidenceSlot.KG_GRAPHRAG,
        EvidenceSlot.VECTOR_EMBEDDING,
        EvidenceSlot.LINEAGE_HISTORY,
    }
)


class CoverageDisposition(str, Enum):
    SATISFIED = "satisfied"
    SCHEDULED = "scheduled"
    REJECTED = "rejected"
    OPTIONAL_MISSING = "optional_missing"


class CoverageDecision(str, Enum):
    READY = "ready"
    QUERIES_SCHEDULED = "queries_scheduled"
    REJECTED = "rejected"


_SLOT_ALIASES: Mapping[str, EvidenceSlot] = {
    "ast": EvidenceSlot.AST,
    "symbol": EvidenceSlot.AST,
    "symbols": EvidenceSlot.AST,
    "lexical": EvidenceSlot.BM25,
    "bm25": EvidenceSlot.BM25,
    "kg": EvidenceSlot.KG_GRAPHRAG,
    "graph": EvidenceSlot.KG_GRAPHRAG,
    "graphrag": EvidenceSlot.KG_GRAPHRAG,
    "kg_graphrag": EvidenceSlot.KG_GRAPHRAG,
    "vector": EvidenceSlot.VECTOR_EMBEDDING,
    "vectors": EvidenceSlot.VECTOR_EMBEDDING,
    "embedding": EvidenceSlot.VECTOR_EMBEDDING,
    "embeddings": EvidenceSlot.VECTOR_EMBEDDING,
    "vector_embedding": EvidenceSlot.VECTOR_EMBEDDING,
    "history": EvidenceSlot.LINEAGE_HISTORY,
    "lineage": EvidenceSlot.LINEAGE_HISTORY,
    "lineage_history": EvidenceSlot.LINEAGE_HISTORY,
    "contract": EvidenceSlot.CONTRACTS,
    "contracts": EvidenceSlot.CONTRACTS,
    "value": EvidenceSlot.VALUE_PROVENANCE,
    "provenance": EvidenceSlot.VALUE_PROVENANCE,
    "value_provenance": EvidenceSlot.VALUE_PROVENANCE,
    "test": EvidenceSlot.TESTS,
    "tests": EvidenceSlot.TESTS,
    "proof": EvidenceSlot.PROOFS,
    "proofs": EvidenceSlot.PROOFS,
}

_AUTHORITY_BY_SLOT: Mapping[EvidenceSlot, str] = {
    EvidenceSlot.AST: "current_root_fact",
    EvidenceSlot.BM25: "nomination_only",
    EvidenceSlot.KG_GRAPHRAG: "nomination_only",
    EvidenceSlot.VECTOR_EMBEDDING: "nomination_only",
    EvidenceSlot.LINEAGE_HISTORY: "nomination_only",
    EvidenceSlot.CONTRACTS: "reviewed_contract",
    EvidenceSlot.VALUE_PROVENANCE: "current_root_fact",
    EvidenceSlot.TESTS: "bounded_observation",
    EvidenceSlot.PROOFS: "proof_receipt",
}

_METHODS_BY_SLOT: Mapping[EvidenceSlot, tuple[str, ...]] = {
    EvidenceSlot.AST: ("query_ast", "search_symbols", "search", "query"),
    EvidenceSlot.BM25: ("query_bm25", "bm25", "search", "query"),
    EvidenceSlot.KG_GRAPHRAG: (
        "query_graphrag",
        "query_graph",
        "retrieve",
        "search",
        "query",
    ),
    EvidenceSlot.VECTOR_EMBEDDING: (
        "query_vectors",
        "vector_search",
        "search",
        "query",
    ),
    EvidenceSlot.LINEAGE_HISTORY: (
        "query_history",
        "history",
        "search",
        "query",
    ),
    EvidenceSlot.CONTRACTS: (
        "query_contracts",
        "contracts",
        "search",
        "query",
    ),
    EvidenceSlot.VALUE_PROVENANCE: (
        "query_value_provenance",
        "trace",
        "search",
        "query",
    ),
    EvidenceSlot.TESTS: ("query_tests", "tests", "search", "query"),
    EvidenceSlot.PROOFS: ("query_proofs", "proofs", "search", "query"),
}


def _slot(value: Any) -> EvidenceSlot:
    if isinstance(value, EvidenceSlot):
        return value
    key = str(getattr(value, "value", value) or "").strip().casefold()
    try:
        return _SLOT_ALIASES[key]
    except KeyError as exc:
        raise PlanningEvidenceError(f"unsupported evidence slot: {value!r}") from exc


def _text(value: Any, *, maximum: int = _MAX_TEXT, required: bool = False) -> str:
    result = _SPACE_RE.sub(" ", str(value or "")).strip()
    if "\x00" in result:
        raise PlanningEvidenceError("evidence text must not contain NUL")
    if len(result) > maximum:
        result = result[: maximum - 1].rstrip() + "…"
    if required and not result:
        raise PlanningEvidenceError("required evidence text is empty")
    return result


def _identity_text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise PlanningEvidenceError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise PlanningEvidenceError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise PlanningEvidenceError(f"{name} is required")
    if len(value.encode("utf-8")) > 2048:
        raise PlanningEvidenceError(f"{name} exceeds 2048 UTF-8 bytes")
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=_canonical_json)
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanningEvidenceError("evidence values must be finite")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Mapping):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return ()


def _strings(value: Any, *, maximum: int = 128) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = (value,) if isinstance(value, str) else _sequence(value)
    result = {_text(item, maximum=320) for item in values}
    result.discard("")
    return tuple(sorted(result)[:maximum])


def _finite_score_millionths(value: Any) -> int:
    if isinstance(value, bool):
        raise PlanningEvidenceError("evidence score must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanningEvidenceError("evidence score must be numeric") from exc
    if not math.isfinite(number):
        raise PlanningEvidenceError("evidence score must be finite")
    return round(max(0.0, min(1.0, number)) * 1_000_000)


def _first(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, "") and not isinstance(
            value, (Mapping, list, tuple)
        ):
            result = _text(getattr(value, "value", value), maximum=512)
            if result:
                return result
    return ""


def _assert_body_free(value: Any, name: str = "evidence") -> None:
    if isinstance(value, Mapping):
        forbidden = {
            str(key).strip().casefold()
            for key in value
        }.intersection(_FORBIDDEN_BODY_KEYS)
        if forbidden:
            raise PlanningEvidenceError(
                f"{name} contains forbidden body fields: "
                + ", ".join(sorted(forbidden))
            )
        for key, item in value.items():
            _assert_body_free(item, f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_body_free(item, f"{name}[{index}]")


@dataclass(frozen=True)
class EvidenceQuery:
    """One bounded, replayable hybrid query."""

    text: str
    task_ids: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    max_results_per_slot: int = 32
    max_total_results: int = 128
    max_bytes: int = 256 * 1024
    timeout_ms: int = 5_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, maximum=4096))
        for name in (
            "task_ids",
            "goal_ids",
            "symbols",
            "paths",
            "obligation_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name)))
        if not self.text and not any(
            getattr(self, name)
            for name in (
                "task_ids",
                "goal_ids",
                "symbols",
                "paths",
                "obligation_ids",
            )
        ):
            raise PlanningEvidenceError(
                "evidence query requires text or an exact selector"
            )
        for name, minimum, maximum in (
            ("max_results_per_slot", 1, _MAX_RESULTS_PER_SLOT),
            ("max_total_results", 1, _MAX_RESULTS),
            ("max_bytes", 1024, _MAX_BYTES),
            ("timeout_ms", 1, 120_000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise PlanningEvidenceError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise PlanningEvidenceError(
                    f"{name} must be between {minimum} and {maximum}"
                )

    @property
    def query_id(self) -> str:
        return _digest("planning-evidence-query", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_QUERY_SCHEMA,
            "text_digest": _digest("query-text", self.text),
            "task_ids": list(self.task_ids),
            "goal_ids": list(self.goal_ids),
            "symbols": list(self.symbols),
            "paths": list(self.paths),
            "obligation_ids": list(self.obligation_ids),
            "max_results_per_slot": self.max_results_per_slot,
            "max_total_results": self.max_total_results,
            "max_bytes": self.max_bytes,
            "timeout_ms": self.timeout_ms,
            "prompt_material_is_inert": True,
        }

    def provider_payload(self) -> dict[str, Any]:
        """Ephemeral provider input; never serialized into the bundle."""

        return {
            "text": self.text,
            "task_ids": self.task_ids,
            "goal_ids": self.goal_ids,
            "symbols": self.symbols,
            "paths": self.paths,
            "obligation_ids": self.obligation_ids,
        }

    @classmethod
    def from_value(cls, value: "EvidenceQuery | str | Mapping[str, Any]") -> "EvidenceQuery":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if not isinstance(value, Mapping):
            raise PlanningEvidenceError(
                "query must be EvidenceQuery, text, or a mapping"
            )
        return cls(
            text=str(value.get("text") or value.get("query") or ""),
            task_ids=_strings(value.get("task_ids")),
            goal_ids=_strings(value.get("goal_ids")),
            symbols=_strings(value.get("symbols") or value.get("ast_symbols")),
            paths=_strings(value.get("paths") or value.get("files")),
            obligation_ids=_strings(value.get("obligation_ids")),
            max_results_per_slot=int(value.get("max_results_per_slot", 32)),
            max_total_results=int(value.get("max_total_results", 128)),
            max_bytes=int(value.get("max_bytes", 256 * 1024)),
            timeout_ms=int(value.get("timeout_ms", 5_000)),
        )


@dataclass(frozen=True)
class EvidenceLabels:
    """Mandatory authority, provenance, root, capability, and cache labels."""

    authority_label: str
    provenance_label: str
    current_root_label: str
    capability_label: str
    cache_label: str
    nomination_only: bool
    inert_data: bool = True

    def __post_init__(self) -> None:
        for name in (
            "authority_label",
            "provenance_label",
            "current_root_label",
            "capability_label",
            "cache_label",
        ):
            object.__setattr__(
                self,
                name,
                _identity_text(getattr(self, name), name),
            )
        if self.inert_data is not True:
            raise PlanningEvidenceError("all evidence material must remain inert data")

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority_label,
            "provenance": self.provenance_label,
            "current_root": self.current_root_label,
            "capability": self.capability_label,
            "cache": self.cache_label,
            "nomination_only": self.nomination_only,
            "inert_data": True,
            "completion_authority": False,
            "write_authority": False,
        }


@dataclass(frozen=True)
class PlanningEvidenceItem:
    """One compact evidence handle; source and proof bodies are excluded."""

    slot: EvidenceSlot
    reference_id: str
    entity_kind: str
    title: str
    path: str
    symbol: str
    status: str
    score_millionths: int
    ranking_explanation: str
    labels: EvidenceLabels
    source_record_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _slot(self.slot))
        for name in ("reference_id",):
            object.__setattr__(
                self, name, _identity_text(getattr(self, name), name)
            )
        for name, maximum in (
            ("entity_kind", 80),
            ("title", 320),
            ("path", 512),
            ("symbol", 320),
            ("status", 80),
            ("ranking_explanation", 512),
            ("source_record_id", 512),
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), maximum=maximum)
            )
        if (
            isinstance(self.score_millionths, bool)
            or not isinstance(self.score_millionths, int)
            or not 0 <= self.score_millionths <= 1_000_000
        ):
            raise PlanningEvidenceError(
                "score_millionths must be an integer from 0 through 1000000"
            )
        if not self.ranking_explanation:
            raise PlanningEvidenceError("every evidence item needs a ranking explanation")
        if not isinstance(self.labels, EvidenceLabels):
            raise PlanningEvidenceError("every evidence item needs EvidenceLabels")
        if self.labels.current_root_label == "":
            raise PlanningEvidenceError("evidence item is missing its current-root label")
        if (
            self.slot in NOMINATION_ONLY_SLOTS
            and (
                not self.labels.nomination_only
                or self.labels.authority_label != "nomination_only"
            )
        ):
            raise PlanningEvidenceError(
                f"{self.slot.value} evidence must remain nomination-only"
            )

    @property
    def evidence_id(self) -> str:
        return _digest(
            "planning-evidence",
            self.to_dict(include_evidence_id=False),
        )

    def to_dict(self, *, include_evidence_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLANNING_EVIDENCE_ITEM_SCHEMA,
            "slot": self.slot.value,
            "reference_id": self.reference_id,
            "source_record_id": self.source_record_id,
            "entity_kind": self.entity_kind,
            "title": self.title,
            "path": self.path,
            "symbol": self.symbol,
            "status": self.status,
            "score_millionths": self.score_millionths,
            "ranking_explanation": self.ranking_explanation,
            "authority_label": self.labels.authority_label,
            "provenance_label": self.labels.provenance_label,
            "current_root_label": self.labels.current_root_label,
            "capability_label": self.labels.capability_label,
            "cache_label": self.labels.cache_label,
            "labels": self.labels.to_dict(),
        }
        _assert_body_free(payload)
        if include_evidence_id:
            payload["evidence_id"] = self.evidence_id
        return payload


@dataclass(frozen=True)
class EvidenceAdapterHealth:
    """Health receipt for one queried evidence lane."""

    slot: EvidenceSlot
    state: BackendState
    detail: str
    reason_codes: tuple[str, ...] = ()
    provider_id: str = ""
    capability_id: str = ""
    index_root_id: str = ""
    queried: bool = False
    result_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _slot(self.slot))
        object.__setattr__(self, "state", BackendState(self.state))
        object.__setattr__(self, "detail", _text(self.detail, maximum=320))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({_text(item, maximum=80) for item in self.reason_codes if item})),
        )
        for name in ("provider_id", "capability_id", "index_root_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), maximum=512)
            )
        if (
            isinstance(self.result_count, bool)
            or not isinstance(self.result_count, int)
            or self.result_count < 0
        ):
            raise PlanningEvidenceError("adapter result_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot.value,
            "state": self.state.value,
            "available": self.state is BackendState.HEALTHY,
            "detail": self.detail,
            "reason_codes": list(self.reason_codes),
            "provider_id": self.provider_id,
            "capability_id": self.capability_id,
            "index_root_id": self.index_root_id,
            "queried": self.queried,
            "result_count": self.result_count,
        }


@dataclass(frozen=True)
class ScheduledEvidenceQuery:
    """A missing-slot query handed to the next deterministic query stage."""

    slot: EvidenceSlot
    query_id: str
    current_root_id: str
    reason_code: str
    capability_id: str
    max_results: int
    max_bytes: int
    timeout_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _slot(self.slot))
        for name in (
            "query_id",
            "current_root_id",
            "reason_code",
            "capability_id",
        ):
            object.__setattr__(
                self, name, _identity_text(getattr(self, name), name)
            )

    @property
    def scheduled_query_id(self) -> str:
        return _digest("scheduled-evidence-query", self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "slot": self.slot.value,
            "query_id": self.query_id,
            "current_root_id": self.current_root_id,
            "reason_code": self.reason_code,
            "capability_id": self.capability_id,
            "max_results": self.max_results,
            "max_bytes": self.max_bytes,
            "timeout_ms": self.timeout_ms,
            "source_and_prompt_material_is_inert": True,
        }
        if include_id:
            payload["scheduled_query_id"] = self.scheduled_query_id
        return payload


@dataclass(frozen=True)
class EvidenceSlotCoverage:
    slot: EvidenceSlot
    required: bool
    disposition: CoverageDisposition
    result_count: int
    health_state: BackendState
    reason_codes: tuple[str, ...] = ()
    scheduled_query_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot", _slot(self.slot))
        object.__setattr__(
            self, "disposition", CoverageDisposition(self.disposition)
        )
        object.__setattr__(self, "health_state", BackendState(self.health_state))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({_text(item, maximum=80) for item in self.reason_codes if item})),
        )
        object.__setattr__(
            self,
            "scheduled_query_id",
            _text(self.scheduled_query_id, maximum=512),
        )
        if self.disposition is CoverageDisposition.SCHEDULED and not self.scheduled_query_id:
            raise PlanningEvidenceError("scheduled coverage requires a query id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot.value,
            "required": self.required,
            "disposition": self.disposition.value,
            "result_count": self.result_count,
            "health_state": self.health_state.value,
            "reason_codes": list(self.reason_codes),
            "scheduled_query_id": self.scheduled_query_id,
        }


@dataclass(frozen=True)
class EvidenceCoverageReceipt:
    """Proof that required slots are present, scheduled, or rejected."""

    current_root_id: str
    query_id: str
    slots: tuple[EvidenceSlotCoverage, ...]
    scheduled_queries: tuple[ScheduledEvidenceQuery, ...] = ()
    decision: CoverageDecision = CoverageDecision.READY
    interface: str = EVIDENCE_COVERAGE_RECEIPT_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "current_root_id", _identity_text(self.current_root_id, "current_root_id")
        )
        object.__setattr__(self, "query_id", _identity_text(self.query_id, "query_id"))
        object.__setattr__(self, "decision", CoverageDecision(self.decision))
        by_slot = {item.slot: item for item in self.slots}
        if len(by_slot) != len(self.slots):
            raise PlanningEvidenceError("coverage receipt contains duplicate slots")
        object.__setattr__(
            self,
            "slots",
            tuple(by_slot[slot] for slot in EVIDENCE_SLOT_ORDER if slot in by_slot),
        )
        queries = tuple(
            sorted(
                self.scheduled_queries,
                key=lambda item: (item.slot.value, item.scheduled_query_id),
            )
        )
        object.__setattr__(self, "scheduled_queries", queries)
        scheduled_ids = {item.scheduled_query_id for item in queries}
        if any(
            item.disposition is CoverageDisposition.SCHEDULED
            and item.scheduled_query_id not in scheduled_ids
            for item in self.slots
        ):
            raise PlanningEvidenceError(
                "coverage references an absent scheduled query"
            )
        rejected = any(
            item.required and item.disposition is CoverageDisposition.REJECTED
            for item in self.slots
        )
        scheduled = any(
            item.required and item.disposition is CoverageDisposition.SCHEDULED
            for item in self.slots
        )
        expected = (
            CoverageDecision.REJECTED
            if rejected
            else CoverageDecision.QUERIES_SCHEDULED
            if scheduled
            else CoverageDecision.READY
        )
        if self.decision is not expected:
            raise PlanningEvidenceError(
                f"coverage decision {self.decision.value} does not match slot state"
            )

    @property
    def receipt_id(self) -> str:
        return _digest(
            "evidence-coverage-receipt",
            self.to_dict(include_receipt_id=False),
        )

    @property
    def ready(self) -> bool:
        return self.decision is CoverageDecision.READY

    @property
    def planning_blocked(self) -> bool:
        return not self.ready

    @property
    def missing_required_slots(self) -> tuple[EvidenceSlot, ...]:
        return tuple(
            item.slot
            for item in self.slots
            if item.required
            and item.disposition is not CoverageDisposition.SATISFIED
        )

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": EVIDENCE_COVERAGE_RECEIPT_SCHEMA,
            "interface": self.interface,
            "current_root_id": self.current_root_id,
            "query_id": self.query_id,
            "decision": self.decision.value,
            "planning_blocked": self.planning_blocked,
            "required_slots_complete": self.ready,
            "slots": [item.to_dict() for item in self.slots],
            "scheduled_queries": [
                item.to_dict() for item in self.scheduled_queries
            ],
            "authority": {
                "completion_authority": False,
                "proof_authority": False,
                "retrieval_is_nomination_only": True,
            },
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class PlanningEvidenceBundle:
    """Body-free evidence and its coverage/health receipts."""

    current_root_id: str
    query: EvidenceQuery
    results: tuple[PlanningEvidenceItem, ...]
    coverage: EvidenceCoverageReceipt
    backend_health: Mapping[EvidenceSlot, EvidenceAdapterHealth]
    considered_count: int
    dropped_count: int = 0
    output_bytes: int = 0
    interface: str = PLANNING_EVIDENCE_BUNDLE_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "current_root_id", _identity_text(self.current_root_id, "current_root_id")
        )
        if self.coverage.current_root_id != self.current_root_id:
            raise PlanningEvidenceError("coverage receipt is bound to another root")
        if self.coverage.query_id != self.query.query_id:
            raise PlanningEvidenceError("coverage receipt is bound to another query")
        results = tuple(
            sorted(
                self.results,
                key=lambda item: (
                    -item.score_millionths,
                    EVIDENCE_SLOT_ORDER.index(item.slot),
                    item.evidence_id,
                ),
            )
        )
        if any(
            item.labels.current_root_label != self.current_root_id
            for item in results
        ):
            raise PlanningEvidenceError("evidence item is stale or cross-root")
        object.__setattr__(self, "results", results)
        health = {
            _slot(slot): value for slot, value in dict(self.backend_health).items()
        }
        if set(health) != set(EVIDENCE_SLOT_ORDER):
            raise PlanningEvidenceError("bundle health must label every evidence slot")
        object.__setattr__(
            self,
            "backend_health",
            MappingProxyType(
                {slot: health[slot] for slot in EVIDENCE_SLOT_ORDER}
            ),
        )

    @property
    def bundle_id(self) -> str:
        payload = self.to_dict(include_bundle_id=False)
        payload["truncation"]["output_bytes"] = 0
        return _digest("planning-evidence-bundle", payload)

    @property
    def ready(self) -> bool:
        return self.coverage.ready

    def to_dict(self, *, include_bundle_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PLANNING_EVIDENCE_BUNDLE_SCHEMA,
            "interface": self.interface,
            "current_root_id": self.current_root_id,
            "query": self.query.to_dict(),
            "query_id": self.query.query_id,
            "results": [item.to_dict() for item in self.results],
            "coverage": self.coverage.to_dict(),
            "backend_health": {
                slot.value: self.backend_health[slot].to_dict()
                for slot in EVIDENCE_SLOT_ORDER
            },
            "truncation": {
                "considered_count": self.considered_count,
                "returned_count": len(self.results),
                "dropped_count": self.dropped_count,
                "max_results": self.query.max_total_results,
                "max_bytes": self.query.max_bytes,
                "output_bytes": self.output_bytes,
                "truncated": self.dropped_count > 0,
            },
            "body_free": True,
            "source_and_prompt_instructions_are_inert_data": True,
            "authority": {
                "completion_authority": False,
                "write_authority": False,
                "retrieval_is_nomination_only": True,
            },
        }
        _assert_body_free(
            {
                "results": payload["results"],
            }
        )
        if include_bundle_id:
            payload["bundle_id"] = self.bundle_id
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


def _adapter_value(adapter: Any, *names: str) -> str:
    for name in names:
        if isinstance(adapter, Mapping):
            value = adapter.get(name)
        else:
            value = getattr(adapter, name, None)
        if value not in (None, "") and not callable(value):
            result = _text(value, maximum=512)
            if result:
                return result
    return ""


def _adapter_health_payload(adapter: Any) -> Any:
    for name in ("health", "health_check", "status"):
        value = (
            adapter.get(name)
            if isinstance(adapter, Mapping)
            else getattr(adapter, name, None)
        )
        if callable(value):
            return value()
        if isinstance(value, Mapping):
            return value
    return {"healthy": True, "status": "ready"}


def _invoke_adapter(
    adapter: Any,
    slot: EvidenceSlot,
    query: EvidenceQuery,
) -> Any:
    if isinstance(adapter, Mapping):
        for name in ("results", "matches", "rows", "items"):
            if name in adapter:
                return adapter[name]
    method: Callable[..., Any] | None = None
    for name in _METHODS_BY_SLOT[slot]:
        candidate = getattr(adapter, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None and callable(adapter):
        method = adapter
    if method is None:
        raise TypeError(f"{slot.value} adapter has no supported query method")
    payload = query.provider_payload()
    attempts = (
        lambda: method(
            payload,
            limit=query.max_results_per_slot,
            timeout_ms=query.timeout_ms,
        ),
        lambda: method(payload, limit=query.max_results_per_slot),
        lambda: method(query.text, limit=query.max_results_per_slot),
        lambda: method(payload),
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


def _response_rows(value: Any) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    if isinstance(value, RetrievalResponse):
        return tuple(item.to_dict() for item in value.results), value.to_dict()
    envelope = _mapping(value)
    if envelope:
        nested: Any = None
        for name in ("results", "matches", "rows", "items", "evidence"):
            if name in envelope:
                nested = envelope[name]
                break
        if nested is not None:
            if isinstance(nested, Mapping):
                rows = tuple(
                    {"reference_id": str(key), "score": item}
                    if isinstance(item, (int, float))
                    else {"reference_id": str(key), **_mapping(item)}
                    for key, item in sorted(nested.items(), key=lambda pair: str(pair[0]))
                )
            else:
                rows = tuple(_mapping(item) for item in _sequence(nested))
            return tuple(item for item in rows if item), envelope
        if any(
            name in envelope
            for name in (
                "evidence_id",
                "reference_id",
                "record_id",
                "node_id",
                "receipt_id",
                "path",
                "symbol",
                "title",
            )
        ):
            return (envelope,), {}
    return tuple(
        row for row in (_mapping(item) for item in _sequence(value)) if row
    ), {}


def _row_root(row: Mapping[str, Any]) -> str:
    return _first(
        row,
        "current_root_id",
        "tree_id",
        "repository_root_id",
        "root_id",
    )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    slot: EvidenceSlot,
    current_root_id: str,
    expected_dimension: int,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    dimensions: set[int] = set()
    fingerprints: set[str] = set()
    vector_count = 0
    observed_scores: list[float] = []
    for row in rows:
        root = _row_root(row)
        if root and root != current_root_id:
            reasons.add("stale_or_cross_root_index")
        if any(
            row.get(name) is True
            for name in (
                "poisoned",
                "corrupt",
                "corrupted",
                "integrity_failed",
            )
        ) or row.get("integrity_valid") is False:
            reasons.add("poisoned_index")
        if any(
            row.get(name) is True
            for name in ("stale", "expired", "invalidated")
        ):
            reasons.add("stale_index")
        if any(
            row.get(name) is True
            for name in ("constant", "is_constant", "collapsed")
        ):
            reasons.add("constant_index")
        if row.get("dimension_drift") is True:
            reasons.add("dimension_drift")
        score = row.get(
            "score",
            row.get(
                "similarity",
                int(row.get("score_millionths", 0)) / 1_000_000
                if isinstance(row.get("score_millionths"), int)
                else 0.0,
            ),
        )
        try:
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                reasons.add("non_finite_index")
            else:
                observed_scores.append(numeric_score)
        except (TypeError, ValueError):
            reasons.add("invalid_index_score")
        vector = row.get("embedding", row.get("vector"))
        if vector is not None:
            vector_count += 1
            if not isinstance(vector, Sequence) or isinstance(
                vector, (str, bytes, bytearray)
            ):
                reasons.add("invalid_vector")
                continue
            try:
                numbers = tuple(float(item) for item in vector)
            except (TypeError, ValueError):
                reasons.add("invalid_vector")
                continue
            if not numbers or not all(math.isfinite(item) for item in numbers):
                reasons.add("non_finite_index")
                continue
            dimensions.add(len(numbers))
            fingerprints.add(_digest("vector", numbers))
    if len(dimensions) > 1:
        reasons.add("dimension_drift")
    if expected_dimension and dimensions and dimensions != {expected_dimension}:
        reasons.add("dimension_drift")
    if (
        slot is EvidenceSlot.VECTOR_EMBEDDING
        and vector_count > 1
        and len(fingerprints) <= 1
    ):
        reasons.add("constant_index")
    if (
        slot is EvidenceSlot.VECTOR_EMBEDDING
        and len(observed_scores) > 1
        and max(observed_scores) == min(observed_scores)
    ):
        reasons.add("constant_index")
    return tuple(sorted(reasons))


class PlanningEvidenceBundleCompiler:
    """Query live evidence adapters and compile one bounded coverage bundle."""

    INTERFACE = PLANNING_EVIDENCE_BUNDLE_INTERFACE

    def __init__(
        self,
        *,
        current_root_id: str,
        adapters: Mapping[EvidenceSlot | str, Any] | None = None,
        ast_adapter: Any = None,
        bm25_adapter: Any = None,
        kg_graphrag_adapter: Any = None,
        vector_adapter: Any = None,
        lineage_adapter: Any = None,
        contract_adapter: Any = None,
        value_provenance_adapter: Any = None,
        test_adapter: Any = None,
        proof_adapter: Any = None,
        expected_vector_dimension: int = 0,
        schedule_callback: Callable[[ScheduledEvidenceQuery], Any] | None = None,
        **backend_aliases: Any,
    ) -> None:
        self.current_root_id = _identity_text(current_root_id, "current_root_id")
        alias_slots = {
            "ast_backend": EvidenceSlot.AST,
            "bm25_backend": EvidenceSlot.BM25,
            "lexical_backend": EvidenceSlot.BM25,
            "kg_backend": EvidenceSlot.KG_GRAPHRAG,
            "graphrag_backend": EvidenceSlot.KG_GRAPHRAG,
            "graph_backend": EvidenceSlot.KG_GRAPHRAG,
            "vector_backend": EvidenceSlot.VECTOR_EMBEDDING,
            "embedding_backend": EvidenceSlot.VECTOR_EMBEDDING,
            "history_backend": EvidenceSlot.LINEAGE_HISTORY,
            "lineage_backend": EvidenceSlot.LINEAGE_HISTORY,
            "contracts_backend": EvidenceSlot.CONTRACTS,
            "contract_backend": EvidenceSlot.CONTRACTS,
            "value_provenance_backend": EvidenceSlot.VALUE_PROVENANCE,
            "tests_backend": EvidenceSlot.TESTS,
            "test_backend": EvidenceSlot.TESTS,
            "proofs_backend": EvidenceSlot.PROOFS,
            "proof_backend": EvidenceSlot.PROOFS,
        }
        unknown_aliases = set(backend_aliases).difference(alias_slots)
        if unknown_aliases:
            raise PlanningEvidenceError(
                "unsupported live adapter arguments: "
                + ", ".join(sorted(unknown_aliases))
            )
        supplied = {
            _slot(key): value for key, value in dict(adapters or {}).items()
        }
        explicit = {
            EvidenceSlot.AST: ast_adapter,
            EvidenceSlot.BM25: bm25_adapter,
            EvidenceSlot.KG_GRAPHRAG: kg_graphrag_adapter,
            EvidenceSlot.VECTOR_EMBEDDING: vector_adapter,
            EvidenceSlot.LINEAGE_HISTORY: lineage_adapter,
            EvidenceSlot.CONTRACTS: contract_adapter,
            EvidenceSlot.VALUE_PROVENANCE: value_provenance_adapter,
            EvidenceSlot.TESTS: test_adapter,
            EvidenceSlot.PROOFS: proof_adapter,
        }
        for name, value in backend_aliases.items():
            slot = alias_slots[name]
            if value is not None and explicit.get(slot) is None:
                explicit[slot] = value
        supplied.update({slot: value for slot, value in explicit.items() if value is not None})
        self.adapters = MappingProxyType(supplied)
        if (
            isinstance(expected_vector_dimension, bool)
            or not isinstance(expected_vector_dimension, int)
            or expected_vector_dimension < 0
        ):
            raise PlanningEvidenceError(
                "expected_vector_dimension must be a non-negative integer"
            )
        self.expected_vector_dimension = expected_vector_dimension
        self.schedule_callback = schedule_callback

    def _query_slot(
        self,
        slot: EvidenceSlot,
        query: EvidenceQuery,
    ) -> tuple[tuple[PlanningEvidenceItem, ...], EvidenceAdapterHealth]:
        adapter = self.adapters.get(slot)
        if adapter is None:
            return (), EvidenceAdapterHealth(
                slot=slot,
                state=BackendState.UNAVAILABLE,
                detail="no live adapter registered",
                reason_codes=("adapter_unavailable",),
            )
        provider_id = _adapter_value(
            adapter, "provider_id", "adapter_id", "name"
        ) or f"{slot.value}-adapter"
        capability_id = _adapter_value(
            adapter, "capability_id", "capability", "interface"
        ) or f"{slot.value}@1"
        index_root_id = _adapter_value(
            adapter, "index_root_id", "index_id", "graph_id"
        )
        expected_dimension = (
            self.expected_vector_dimension
            if slot is EvidenceSlot.VECTOR_EMBEDDING
            else 0
        )
        try:
            raw_health = _adapter_health_payload(adapter)
            assessment = assess_retrieval_index_health(
                raw_health,
                current_root_id=self.current_root_id,
                expected_dimension=expected_dimension,
            )
            if not assessment.healthy:
                return (), EvidenceAdapterHealth(
                    slot=slot,
                    state=BackendState.UNHEALTHY,
                    detail=assessment.detail,
                    reason_codes=assessment.reason_codes,
                    provider_id=provider_id,
                    capability_id=capability_id,
                    index_root_id=index_root_id,
                )
            raw_response = _invoke_adapter(adapter, slot, query)
            rows, envelope = _response_rows(raw_response)
            row_reasons = _validate_rows(
                rows,
                slot=slot,
                current_root_id=self.current_root_id,
                expected_dimension=expected_dimension or assessment.dimension,
            )
            if row_reasons:
                return (), EvidenceAdapterHealth(
                    slot=slot,
                    state=BackendState.UNHEALTHY,
                    detail=", ".join(row_reasons),
                    reason_codes=row_reasons,
                    provider_id=provider_id,
                    capability_id=capability_id,
                    index_root_id=index_root_id,
                    queried=True,
                )
            envelope_root = _row_root(envelope)
            if envelope_root and envelope_root != self.current_root_id:
                return (), EvidenceAdapterHealth(
                    slot=slot,
                    state=BackendState.UNHEALTHY,
                    detail="stale_or_cross_root_index",
                    reason_codes=("stale_or_cross_root_index",),
                    provider_id=provider_id,
                    capability_id=capability_id,
                    index_root_id=index_root_id,
                    queried=True,
                )
            cache_label = _first(
                envelope, "cache_label", "cache_status", "cache_outcome"
            ) or _adapter_value(adapter, "cache_label", "cache_status") or "cache_miss"
            items: list[PlanningEvidenceItem] = []
            for rank, row in enumerate(rows[: query.max_results_per_slot], start=1):
                reference_id = _first(
                    row,
                    "reference_id",
                    "evidence_id",
                    "candidate_id",
                    "node_id",
                    "receipt_id",
                    "record_id",
                    "id",
                )
                if not reference_id:
                    reference_id = _digest(
                        f"{slot.value}-reference",
                        {
                            "kind": _first(row, "entity_kind", "kind", "type"),
                            "title": _first(row, "title", "name", "label", "summary"),
                            "path": _first(row, "path", "file", "source_path"),
                            "symbol": _first(row, "symbol", "qualified_name"),
                        },
                    )
                raw_score: Any
                if isinstance(row.get("score_millionths"), int):
                    raw_score = int(row["score_millionths"]) / 1_000_000
                else:
                    raw_score = row.get("score", row.get("similarity", 1.0 / rank))
                score = _finite_score_millionths(raw_score)
                provenance = _first(
                    row,
                    "provenance_label",
                    "provenance",
                    "source_kind",
                    "producer_id",
                ) or f"live:{provider_id}"
                row_cache = _first(
                    row, "cache_label", "cache_status", "cache_outcome"
                ) or cache_label
                explanation = _first(row, "ranking_explanation", "explanation")
                if not explanation:
                    explanation = (
                        f"{slot.value} adapter rank {rank}; "
                        f"score={score}/1000000; fixed slot authority="
                        f"{_AUTHORITY_BY_SLOT[slot]}"
                    )
                labels = EvidenceLabels(
                    authority_label=_AUTHORITY_BY_SLOT[slot],
                    provenance_label=provenance,
                    current_root_label=self.current_root_id,
                    capability_label=capability_id,
                    cache_label=row_cache,
                    nomination_only=slot in NOMINATION_ONLY_SLOTS,
                )
                item = PlanningEvidenceItem(
                    slot=slot,
                    reference_id=reference_id,
                    source_record_id=_first(
                        row, "source_record_id", "record_id", "receipt_id"
                    ),
                    entity_kind=_first(row, "entity_kind", "kind", "type")
                    or slot.value,
                    title=_first(row, "title", "name", "label", "summary"),
                    path=_first(row, "path", "file", "source_path"),
                    symbol=_first(row, "symbol", "qualified_name"),
                    status=_first(row, "status", "outcome", "state"),
                    score_millionths=score,
                    ranking_explanation=explanation,
                    labels=labels,
                )
                items.append(item)
            deduplicated = {
                item.evidence_id: item for item in items
            }
            results = tuple(
                sorted(
                    deduplicated.values(),
                    key=lambda item: (-item.score_millionths, item.evidence_id),
                )
            )
            return results, EvidenceAdapterHealth(
                slot=slot,
                state=BackendState.HEALTHY,
                detail=assessment.detail or "live query completed",
                provider_id=provider_id,
                capability_id=capability_id,
                index_root_id=index_root_id,
                queried=True,
                result_count=len(results),
            )
        except Exception as exc:
            reason = getattr(exc, "reason_code", "adapter_query_failed")
            return (), EvidenceAdapterHealth(
                slot=slot,
                state=BackendState.UNHEALTHY,
                detail=f"{type(exc).__name__}: {reason}",
                reason_codes=(_text(reason, maximum=80),),
                provider_id=provider_id,
                capability_id=capability_id,
                index_root_id=index_root_id,
                queried=True,
            )

    def compile(
        self,
        query: EvidenceQuery | str | Mapping[str, Any],
        *,
        required_slots: Iterable[EvidenceSlot | str] = DEFAULT_REQUIRED_EVIDENCE_SLOTS,
        schedule_missing: bool = True,
        raise_on_rejection: bool = False,
    ) -> PlanningEvidenceBundle:
        request = EvidenceQuery.from_value(query)
        required = frozenset(_slot(item) for item in required_slots)
        all_results: list[PlanningEvidenceItem] = []
        health: dict[EvidenceSlot, EvidenceAdapterHealth] = {}
        coverage: list[EvidenceSlotCoverage] = []
        scheduled: list[ScheduledEvidenceQuery] = []

        for slot in EVIDENCE_SLOT_ORDER:
            results, slot_health = self._query_slot(slot, request)
            health[slot] = slot_health
            all_results.extend(results)
            if results:
                coverage.append(
                    EvidenceSlotCoverage(
                        slot=slot,
                        required=slot in required,
                        disposition=CoverageDisposition.SATISFIED,
                        result_count=len(results),
                        health_state=slot_health.state,
                    )
                )
                continue
            reasons = slot_health.reason_codes or (
                "query_returned_no_evidence"
                if slot_health.state is BackendState.HEALTHY
                else "adapter_unavailable",
            )
            if slot not in required:
                coverage.append(
                    EvidenceSlotCoverage(
                        slot=slot,
                        required=False,
                        disposition=CoverageDisposition.OPTIONAL_MISSING,
                        result_count=0,
                        health_state=slot_health.state,
                        reason_codes=reasons,
                    )
                )
                continue
            if schedule_missing:
                query_record = ScheduledEvidenceQuery(
                    slot=slot,
                    query_id=request.query_id,
                    current_root_id=self.current_root_id,
                    reason_code=reasons[0],
                    capability_id=slot_health.capability_id
                    or f"{slot.value}@1",
                    max_results=request.max_results_per_slot,
                    max_bytes=request.max_bytes,
                    timeout_ms=request.timeout_ms,
                )
                if self.schedule_callback is not None:
                    try:
                        accepted = self.schedule_callback(query_record)
                    except Exception:
                        accepted = False
                    if accepted is False:
                        coverage.append(
                            EvidenceSlotCoverage(
                                slot=slot,
                                required=True,
                                disposition=CoverageDisposition.REJECTED,
                                result_count=0,
                                health_state=slot_health.state,
                                reason_codes=(*reasons, "query_schedule_rejected"),
                            )
                        )
                        continue
                scheduled.append(query_record)
                coverage.append(
                    EvidenceSlotCoverage(
                        slot=slot,
                        required=True,
                        disposition=CoverageDisposition.SCHEDULED,
                        result_count=0,
                        health_state=slot_health.state,
                        reason_codes=reasons,
                        scheduled_query_id=query_record.scheduled_query_id,
                    )
                )
            else:
                coverage.append(
                    EvidenceSlotCoverage(
                        slot=slot,
                        required=True,
                        disposition=CoverageDisposition.REJECTED,
                        result_count=0,
                        health_state=slot_health.state,
                        reason_codes=reasons,
                    )
                )

        rejected = any(
            item.disposition is CoverageDisposition.REJECTED
            for item in coverage
            if item.required
        )
        decision = (
            CoverageDecision.REJECTED
            if rejected
            else CoverageDecision.QUERIES_SCHEDULED
            if scheduled
            else CoverageDecision.READY
        )
        receipt = EvidenceCoverageReceipt(
            current_root_id=self.current_root_id,
            query_id=request.query_id,
            slots=tuple(coverage),
            scheduled_queries=tuple(scheduled),
            decision=decision,
        )
        if rejected and raise_on_rejection:
            raise MissingRequiredEvidenceError(receipt.missing_required_slots)

        ranked = sorted(
            all_results,
            key=lambda item: (
                -item.score_millionths,
                EVIDENCE_SLOT_ORDER.index(item.slot),
                item.evidence_id,
            ),
        )
        considered = len(ranked)
        selected = ranked[: request.max_total_results]
        dropped = considered - len(selected)

        def make_bundle(items: Sequence[PlanningEvidenceItem], output_bytes: int = 0) -> PlanningEvidenceBundle:
            return PlanningEvidenceBundle(
                current_root_id=self.current_root_id,
                query=request,
                results=tuple(items),
                coverage=receipt,
                backend_health=health,
                considered_count=considered,
                dropped_count=considered - len(items),
                output_bytes=output_bytes,
            )

        # Keep mandatory health and coverage metadata, then fit ranked handles.
        while selected:
            trial = make_bundle(selected)
            if len(trial.to_json().encode("utf-8")) <= request.max_bytes:
                break
            selected.pop()
            dropped += 1
        bundle = make_bundle(selected)
        output_bytes = 0
        for _ in range(8):
            bundle = make_bundle(selected, output_bytes)
            encoded = len(bundle.to_json().encode("utf-8"))
            if encoded > request.max_bytes and selected:
                selected.pop()
                dropped += 1
                output_bytes = 0
                continue
            if encoded == output_bytes:
                break
            output_bytes = encoded
        bundle = make_bundle(selected, output_bytes)
        final_size = len(bundle.to_json().encode("utf-8"))
        if final_size != bundle.output_bytes:
            bundle = make_bundle(selected, final_size)
            final_size = len(bundle.to_json().encode("utf-8"))
        if final_size > request.max_bytes:
            raise PlanningEvidenceError(
                "max_bytes is too small for mandatory evidence coverage metadata"
            )
        return bundle

    build = compile
    query = compile


def compile_planning_evidence_bundle(
    query: EvidenceQuery | str | Mapping[str, Any],
    *,
    current_root_id: str,
    adapters: Mapping[EvidenceSlot | str, Any] | None = None,
    required_slots: Iterable[EvidenceSlot | str] = DEFAULT_REQUIRED_EVIDENCE_SLOTS,
    schedule_missing: bool = True,
    raise_on_rejection: bool = False,
    expected_vector_dimension: int = 0,
    schedule_callback: Callable[[ScheduledEvidenceQuery], Any] | None = None,
    **adapter_kwargs: Any,
) -> PlanningEvidenceBundle:
    """One-shot production entry point."""

    compiler = PlanningEvidenceBundleCompiler(
        current_root_id=current_root_id,
        adapters=adapters,
        expected_vector_dimension=expected_vector_dimension,
        schedule_callback=schedule_callback,
        **adapter_kwargs,
    )
    return compiler.compile(
        query,
        required_slots=required_slots,
        schedule_missing=schedule_missing,
        raise_on_rejection=raise_on_rejection,
    )


# Descriptive aliases used by query-planning and Doctor callers.
PlanningEvidenceCompiler = PlanningEvidenceBundleCompiler
LiveHybridEvidenceCompiler = PlanningEvidenceBundleCompiler
EvidenceResult = PlanningEvidenceItem
EvidenceCoverage = EvidenceCoverageReceipt
build_planning_evidence_bundle = compile_planning_evidence_bundle


__all__ = [
    "DEFAULT_REQUIRED_EVIDENCE_SLOTS",
    "EVIDENCE_COVERAGE_RECEIPT_INTERFACE",
    "EVIDENCE_COVERAGE_RECEIPT_SCHEMA",
    "EVIDENCE_QUERY_SCHEMA",
    "EVIDENCE_SLOT_ORDER",
    "NOMINATION_ONLY_SLOTS",
    "PLANNING_EVIDENCE_BUNDLE_INTERFACE",
    "PLANNING_EVIDENCE_BUNDLE_SCHEMA",
    "PLANNING_EVIDENCE_ITEM_SCHEMA",
    "CoverageDecision",
    "CoverageDisposition",
    "EvidenceAdapterHealth",
    "EvidenceCoverage",
    "EvidenceCoverageReceipt",
    "EvidenceLabels",
    "EvidenceQuery",
    "EvidenceResult",
    "EvidenceSlot",
    "EvidenceSlotCoverage",
    "LiveHybridEvidenceCompiler",
    "MissingRequiredEvidenceError",
    "PlanningEvidenceBundle",
    "PlanningEvidenceBundleCompiler",
    "PlanningEvidenceCompiler",
    "PlanningEvidenceError",
    "PlanningEvidenceItem",
    "ScheduledEvidenceQuery",
    "build_planning_evidence_bundle",
    "compile_planning_evidence_bundle",
]
