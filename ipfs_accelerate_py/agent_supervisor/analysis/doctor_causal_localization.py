"""Deterministic, fail-closed causal localization for Doctor findings.

Interface: ``DoctorCausalLocalization@1``

The localizer joins a current :class:`DoctorEvidenceSnapshot` with compact,
root-bound facts produced by exact contract, graph, data-flow, runtime,
delta-debugging, and solver adapters.  Approximate retrieval can nominate a
candidate for follow-up analysis, but it can never select a cause.  Stale,
poisoned, unverified, or root-mismatched evidence is retained in the audit and
excluded from the causal slice.

The output deliberately separates three identities:

* ``diagnostic_finding_cid`` identifies the evidence-rich source finding;
* ``issue_cid`` identifies the semantic mismatch and is stable as evidence is
  reordered, enriched, or poisoned nominations are added; and
* ``localization_cid`` identifies this exact localization receipt.

This module is report-only.  It grants no repair or mutation authority.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, TYPE_CHECKING

from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .doctor_repository_diagnostics import (
    DoctorDiagnosticFinding,
    DoctorEvidenceSnapshot,
    ExpectationSourceKind,
    FindingDisposition,
    FindingKind,
)

if TYPE_CHECKING:  # Avoid importing the large impact compiler at module import time.
    from .deterministic_doctor_impact import DoctorImpactClosureReceipt


DOCTOR_CAUSAL_LOCALIZATION_INTERFACE: Final[str] = "DoctorCausalLocalization@1"
DOCTOR_CAUSAL_LOCALIZATION_VERSION: Final[int] = 1
DOCTOR_CAUSAL_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-causal-evidence@1"
)
DOCTOR_MISMATCH_SLICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-mismatch-slice@1"
)
DOCTOR_CAUSAL_LOCALIZATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-causal-localization@1"
)

MAX_EVIDENCE = 4_096
MAX_REFERENCES = 16_384
MAX_TEXT = 4_096

_BODY_FIELDS = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "source_bytes",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "trace_body",
        "model",
    }
)


class DoctorCausalLocalizationError(ValueError):
    """Malformed or authority-invalid localization input."""


class DoctorCausalLocalizationBoundsError(DoctorCausalLocalizationError):
    """A compact localization record exceeded a deterministic bound."""


class CausalEvidenceKind(str, Enum):
    """Closed evidence vocabulary with authority determined by kind."""

    CONTRACT_DELTA = "contract_delta"
    STATIC_SLICE = "static_slice"
    DYNAMIC_SLICE = "dynamic_slice"
    CALL_GRAPH = "call_graph"
    DEPENDENCY_GRAPH = "dependency_graph"
    VALUE_GRAPH = "value_graph"
    DATAFLOW = "dataflow"
    RUNTIME_FACT = "runtime_fact"
    FAILING_TRACE = "failing_trace"
    DELTA_DEBUG = "delta_debug"
    UNSAT_CORE = "unsat_core"
    COUNTEREXAMPLE = "counterexample"
    AST_FACT = "ast_fact"
    TYPE_FACT = "type_fact"
    RETRIEVAL = "retrieval"
    VECTOR_NEAREST = "vector_nearest"
    GRAPHRAG = "graphrag"
    CACHE = "cache"
    MODEL_NOMINATION = "model_nomination"


class CausalEvidenceDisposition(str, Enum):
    """How one evidence item was handled by localization."""

    EXACT = "exact"
    NOMINATION_ONLY = "nomination_only"
    STALE = "stale"
    POISONED = "poisoned"
    UNVERIFIED = "unverified"
    ROOT_MISMATCH = "root_mismatch"
    INSUFFICIENT = "insufficient"


class CausalLocalizationDisposition(str, Enum):
    """Deterministic localization outcome."""

    LOCALIZED = "localized"
    ABSTAINED = "abstained"


_NOMINATION_KINDS: Final[frozenset[CausalEvidenceKind]] = frozenset(
    {
        CausalEvidenceKind.RETRIEVAL,
        CausalEvidenceKind.VECTOR_NEAREST,
        CausalEvidenceKind.GRAPHRAG,
        CausalEvidenceKind.CACHE,
        CausalEvidenceKind.MODEL_NOMINATION,
    }
)

_DECISIVE_KINDS: Final[frozenset[CausalEvidenceKind]] = frozenset(
    {
        CausalEvidenceKind.DELTA_DEBUG,
        CausalEvidenceKind.UNSAT_CORE,
        CausalEvidenceKind.COUNTEREXAMPLE,
    }
)

_CURRENT_CHECKOUT_FACT_KINDS: Final[frozenset[CausalEvidenceKind]] = frozenset(
    {
        CausalEvidenceKind.CONTRACT_DELTA,
        CausalEvidenceKind.STATIC_SLICE,
        CausalEvidenceKind.CALL_GRAPH,
        CausalEvidenceKind.DEPENDENCY_GRAPH,
        CausalEvidenceKind.VALUE_GRAPH,
        CausalEvidenceKind.DATAFLOW,
        CausalEvidenceKind.AST_FACT,
        CausalEvidenceKind.TYPE_FACT,
    }
)

_KIND_DOMAIN: Final[Mapping[CausalEvidenceKind, str]] = MappingProxyType(
    {
        CausalEvidenceKind.CONTRACT_DELTA: "contract",
        CausalEvidenceKind.STATIC_SLICE: "graph",
        CausalEvidenceKind.CALL_GRAPH: "graph",
        CausalEvidenceKind.DEPENDENCY_GRAPH: "graph",
        CausalEvidenceKind.AST_FACT: "graph",
        CausalEvidenceKind.VALUE_GRAPH: "dataflow",
        CausalEvidenceKind.DATAFLOW: "dataflow",
        CausalEvidenceKind.TYPE_FACT: "dataflow",
        CausalEvidenceKind.DYNAMIC_SLICE: "runtime",
        CausalEvidenceKind.RUNTIME_FACT: "runtime",
        CausalEvidenceKind.FAILING_TRACE: "runtime",
        CausalEvidenceKind.DELTA_DEBUG: "intervention",
        CausalEvidenceKind.UNSAT_CORE: "solver",
        CausalEvidenceKind.COUNTEREXAMPLE: "solver",
        CausalEvidenceKind.RETRIEVAL: "nomination",
        CausalEvidenceKind.VECTOR_NEAREST: "nomination",
        CausalEvidenceKind.GRAPHRAG: "nomination",
        CausalEvidenceKind.CACHE: "nomination",
        CausalEvidenceKind.MODEL_NOMINATION: "nomination",
    }
)


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str):
        raise DoctorCausalLocalizationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise DoctorCausalLocalizationError(f"{name} is required")
    if len(result.encode("utf-8")) > limit:
        raise DoctorCausalLocalizationBoundsError(f"{name} exceeds its byte bound")
    return result


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCES,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DoctorCausalLocalizationError(f"{name} must be a sequence")
    else:
        raw = values
    if len(raw) > limit:
        raise DoctorCausalLocalizationBoundsError(f"{name} exceeds its item bound")
    result = tuple(sorted({_text(item, name, limit=1024) for item in raw}))
    if required and not result:
        raise DoctorCausalLocalizationError(f"{name} is required")
    return result


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise DoctorCausalLocalizationError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        raise DoctorCausalLocalizationBoundsError("metadata exceeds nesting bound")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        raise DoctorCausalLocalizationError("floating point evidence is not canonical")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DoctorCausalLocalizationError("metadata keys must be strings")
            normalized = key.casefold().replace("-", "_").strip()
            if normalized in _BODY_FIELDS:
                raise DoctorCausalLocalizationError("causal records must be body-free")
            result[key] = _plain(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, depth=depth + 1) for item in value]
    raise DoctorCausalLocalizationError(
        f"unsupported metadata type: {type(value).__name__}"
    )


def _semantic_issue_cid(
    snapshot: DoctorEvidenceSnapshot, finding: DoctorDiagnosticFinding
) -> str:
    """Identity of the mismatch, excluding evidence/ranking/cache volatility."""

    semantic_detail_keys = (
        "diagnostic_code",
        "failure_kind",
        "trace_disposition",
        "receiver_reference",
        "resolution",
        "target_path",
        "expected_argument_count",
        "observed_argument_count",
        "missing_required_parameters",
        "unexpected_keywords",
        "duplicate_arguments",
        "contract_clause_id",
    )
    details = {
        key: finding.details[key]
        for key in semantic_detail_keys
        if key in finding.details
    }
    return content_identity(
        {
            "schema": "doctor-semantic-issue@1",
            "repository_id": snapshot.authority_roots.repository_id,
            "kind": finding.kind.value,
            "path": finding.path,
            "symbol": finding.symbol,
            "expectation_source": finding.expectation_source.value,
            "expectation_ref": finding.expectation_ref,
            "semantic_details": details,
        }
    )


@dataclass(frozen=True)
class CausalEvidence(CanonicalContract):
    """One compact causal fact or non-authoritative nomination."""

    SCHEMA: ClassVar[str] = DOCTOR_CAUSAL_EVIDENCE_SCHEMA

    evidence_id: str
    kind: CausalEvidenceKind | str
    cause_ids: tuple[str, ...] = ()
    fact_refs: tuple[str, ...] = ()
    consumer_ids: tuple[str, ...] = ()
    frontier_refs: tuple[str, ...] = ()
    snapshot_cid: str = ""
    tree_id: str = ""
    graph_id: str = ""
    index_id: str = ""
    verified: bool = True
    minimized: bool = False
    stale: bool = False
    poisoned: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "kind", _enum(self.kind, CausalEvidenceKind, "kind"))
        object.__setattr__(self, "cause_ids", _ids(self.cause_ids, "cause_ids"))
        object.__setattr__(self, "fact_refs", _ids(self.fact_refs, "fact_refs"))
        object.__setattr__(self, "consumer_ids", _ids(self.consumer_ids, "consumer_ids"))
        object.__setattr__(self, "frontier_refs", _ids(self.frontier_refs, "frontier_refs"))
        for name in ("snapshot_cid", "tree_id", "graph_id", "index_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False, limit=1024)
            )
        for name in ("verified", "minimized", "stale", "poisoned"):
            if not isinstance(getattr(self, name), bool):
                raise DoctorCausalLocalizationError(f"{name} must be a boolean")
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(_plain(self.metadata or {})))
        )

    @property
    def nomination_only(self) -> bool:
        return self.kind in _NOMINATION_KINDS

    @property
    def domain(self) -> str:
        return _KIND_DOMAIN[self.kind]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DOCTOR_CAUSAL_LOCALIZATION_VERSION,
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "cause_ids": list(self.cause_ids),
            "fact_refs": list(self.fact_refs),
            "consumer_ids": list(self.consumer_ids),
            "frontier_refs": list(self.frontier_refs),
            "snapshot_cid": self.snapshot_cid,
            "tree_id": self.tree_id,
            "graph_id": self.graph_id,
            "index_id": self.index_id,
            "verified": self.verified,
            "minimized": self.minimized,
            "stale": self.stale,
            "poisoned": self.poisoned,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalEvidence":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DoctorCausalLocalizationError("unsupported causal evidence schema")
        value = cls(
            evidence_id=str(payload.get("evidence_id") or ""),
            kind=payload.get("kind", ""),
            cause_ids=tuple(payload.get("cause_ids") or ()),
            fact_refs=tuple(payload.get("fact_refs") or ()),
            consumer_ids=tuple(payload.get("consumer_ids") or ()),
            frontier_refs=tuple(payload.get("frontier_refs") or ()),
            snapshot_cid=str(payload.get("snapshot_cid") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            graph_id=str(payload.get("graph_id") or ""),
            index_id=str(payload.get("index_id") or ""),
            verified=payload.get("verified", True),
            minimized=payload.get("minimized", False),
            stale=payload.get("stale", False),
            poisoned=payload.get("poisoned", False),
            metadata=payload.get("metadata") or {},
        )
        supplied = payload.get("content_id", payload.get("cid", ""))
        if supplied not in (None, "", value.content_id):
            raise DoctorCausalLocalizationError("causal evidence content identity mismatch")
        return value


@dataclass(frozen=True)
class MinimalMismatchSlice(CanonicalContract):
    """Minimal exact fact cover explaining one selected cause."""

    SCHEMA: ClassVar[str] = DOCTOR_MISMATCH_SLICE_SCHEMA

    issue_cid: str
    cause_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    contract_refs: tuple[str, ...] = ()
    graph_refs: tuple[str, ...] = ()
    dataflow_refs: tuple[str, ...] = ()
    runtime_refs: tuple[str, ...] = ()
    delta_debug_refs: tuple[str, ...] = ()
    unsat_core_refs: tuple[str, ...] = ()
    counterexample_refs: tuple[str, ...] = ()
    mandatory_consumer_ids: tuple[str, ...] = ()
    open_frontier_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_cid", _text(self.issue_cid, "issue_cid"))
        object.__setattr__(self, "cause_id", _text(self.cause_id, "cause_id", required=False))
        for name in (
            "evidence_ids",
            "contract_refs",
            "graph_refs",
            "dataflow_refs",
            "runtime_refs",
            "delta_debug_refs",
            "unsat_core_refs",
            "counterexample_refs",
            "mandatory_consumer_ids",
            "open_frontier_refs",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DOCTOR_CAUSAL_LOCALIZATION_VERSION,
            "issue_cid": self.issue_cid,
            "cause_id": self.cause_id,
            "evidence_ids": list(self.evidence_ids),
            "contract_refs": list(self.contract_refs),
            "graph_refs": list(self.graph_refs),
            "dataflow_refs": list(self.dataflow_refs),
            "runtime_refs": list(self.runtime_refs),
            "delta_debug_refs": list(self.delta_debug_refs),
            "unsat_core_refs": list(self.unsat_core_refs),
            "counterexample_refs": list(self.counterexample_refs),
            "mandatory_consumer_ids": list(self.mandatory_consumer_ids),
            "open_frontier_refs": list(self.open_frontier_refs),
        }

    @property
    def slice_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MinimalMismatchSlice":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DoctorCausalLocalizationError("unsupported mismatch slice schema")
        value = cls(
            **{
                name: payload.get(name, "" if name in {"issue_cid", "cause_id"} else ())
                for name in (
                    "issue_cid",
                    "cause_id",
                    "evidence_ids",
                    "contract_refs",
                    "graph_refs",
                    "dataflow_refs",
                    "runtime_refs",
                    "delta_debug_refs",
                    "unsat_core_refs",
                    "counterexample_refs",
                    "mandatory_consumer_ids",
                    "open_frontier_refs",
                )
            }
        )
        supplied = payload.get("content_id", payload.get("cid", ""))
        if supplied not in (None, "", value.content_id):
            raise DoctorCausalLocalizationError("mismatch slice identity mismatch")
        return value


@dataclass(frozen=True)
class DoctorCausalLocalizationRequest:
    """Current snapshot and exact/advisory facts for one mismatch."""

    snapshot: DoctorEvidenceSnapshot
    finding: DoctorDiagnosticFinding | Mapping[str, Any] | None = None
    issue_cid: str = ""
    evidence: tuple[CausalEvidence, ...] = ()
    impact_closure: Any = None
    required_frontiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, DoctorEvidenceSnapshot):
            raise DoctorCausalLocalizationError("snapshot must be DoctorEvidenceSnapshot")
        finding = self.finding
        if isinstance(finding, Mapping):
            finding = DoctorDiagnosticFinding(
                kind=finding.get("kind", FindingKind.CONTRACT),
                disposition=finding.get(
                    "disposition", FindingDisposition.ABSTAIN
                ),
                path=str(finding.get("path") or ""),
                symbol=str(finding.get("symbol") or ""),
                message=str(finding.get("message") or ""),
                observation_refs=tuple(finding.get("observation_refs") or ()),
                expectation_source=finding.get(
                    "expectation_source", ExpectationSourceKind.NONE
                ),
                expectation_ref=str(finding.get("expectation_ref") or ""),
                expectation_precedence=int(
                    finding.get("expectation_precedence") or 0
                ),
                open_frontier_refs=tuple(
                    finding.get("open_frontier_refs") or ()
                ),
                evidence_refs=tuple(finding.get("evidence_refs") or ()),
                details=finding.get("details") or {},
            )
        if finding is None and self.issue_cid:
            # Accept a source finding CID as a lookup convenience.  Semantic
            # issue CIDs cannot be reversed and therefore need ``finding``.
            finding = self.snapshot.finding_for_cid(self.issue_cid)
        if finding is None:
            candidates = tuple(
                item
                for item in self.snapshot.findings
                if item.kind is not FindingKind.COMPLETENESS
            )
            if len(candidates) != 1:
                raise DoctorCausalLocalizationError(
                    "finding is required when snapshot has zero or multiple findings"
                )
            finding = candidates[0]
        if not isinstance(finding, DoctorDiagnosticFinding):
            raise DoctorCausalLocalizationError("finding must be DoctorDiagnosticFinding")
        if self.snapshot.finding_for_cid(finding.finding_cid) is None:
            raise DoctorCausalLocalizationError("finding is not part of the current snapshot")
        semantic_cid = _semantic_issue_cid(self.snapshot, finding)
        if self.issue_cid and self.issue_cid not in {finding.finding_cid, semantic_cid}:
            raise DoctorCausalLocalizationError("issue_cid does not identify the finding")
        object.__setattr__(self, "finding", finding)
        object.__setattr__(self, "issue_cid", semantic_cid)
        if len(self.evidence) > MAX_EVIDENCE:
            raise DoctorCausalLocalizationBoundsError("evidence exceeds item bound")
        normalized: list[CausalEvidence] = []
        for item in self.evidence:
            if isinstance(item, CausalEvidence):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(
                    CausalEvidence.from_dict(item)
                    if item.get("schema")
                    else CausalEvidence(**item)
                )
            else:
                raise DoctorCausalLocalizationError("evidence contains an invalid item")
        evidence_ids = [item.evidence_id for item in normalized]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise DoctorCausalLocalizationError("evidence IDs must be unique")
        object.__setattr__(
            self, "evidence", tuple(sorted(normalized, key=lambda item: item.evidence_id))
        )
        object.__setattr__(
            self, "required_frontiers", _ids(self.required_frontiers, "required_frontiers")
        )


@dataclass(frozen=True)
class DoctorCausalLocalizationReceipt(CanonicalContract):
    """Auditable localization result with a stable semantic issue CID."""

    SCHEMA: ClassVar[str] = DOCTOR_CAUSAL_LOCALIZATION_SCHEMA

    repository_id: str
    snapshot_cid: str
    diagnostic_finding_cid: str
    issue_cid: str
    disposition: CausalLocalizationDisposition | str
    selected_cause_id: str = ""
    candidate_cause_ids: tuple[str, ...] = ()
    mismatch_slice: MinimalMismatchSlice | None = None
    exact_evidence_ids: tuple[str, ...] = ()
    nomination_evidence_ids: tuple[str, ...] = ()
    rejected_evidence_ids: tuple[str, ...] = ()
    evidence_dispositions: Mapping[str, str] = field(default_factory=dict)
    mandatory_consumer_ids: tuple[str, ...] = ()
    open_frontier_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    complete_frontier_accounting: bool = False

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "snapshot_cid",
            "diagnostic_finding_cid",
            "issue_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CausalLocalizationDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "selected_cause_id",
            _text(self.selected_cause_id, "selected_cause_id", required=False),
        )
        for name in (
            "candidate_cause_ids",
            "exact_evidence_ids",
            "nomination_evidence_ids",
            "rejected_evidence_ids",
            "mandatory_consumer_ids",
            "open_frontier_refs",
            "reason_codes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        dispositions = {
            _text(key, "evidence disposition key"): _text(
                value, "evidence disposition value"
            )
            for key, value in (self.evidence_dispositions or {}).items()
        }
        object.__setattr__(
            self, "evidence_dispositions", MappingProxyType(dict(sorted(dispositions.items())))
        )
        if self.mismatch_slice is not None and not isinstance(
            self.mismatch_slice, MinimalMismatchSlice
        ):
            if isinstance(self.mismatch_slice, Mapping):
                object.__setattr__(
                    self,
                    "mismatch_slice",
                    MinimalMismatchSlice.from_dict(self.mismatch_slice),
                )
            else:
                raise DoctorCausalLocalizationError("mismatch_slice has invalid type")
        if self.mismatch_slice is None:
            raise DoctorCausalLocalizationError("mismatch_slice is required")
        if self.mismatch_slice.issue_cid != self.issue_cid:
            raise DoctorCausalLocalizationError("mismatch slice issue CID mismatch")
        if self.disposition is CausalLocalizationDisposition.LOCALIZED:
            if not self.selected_cause_id:
                raise DoctorCausalLocalizationError("localized result requires a cause")
            if self.selected_cause_id not in self.candidate_cause_ids:
                raise DoctorCausalLocalizationError("selected cause must be a candidate")
            if self.mismatch_slice.cause_id != self.selected_cause_id:
                raise DoctorCausalLocalizationError("mismatch slice cause mismatch")
        elif self.selected_cause_id:
            raise DoctorCausalLocalizationError("abstention cannot select a cause")
        if not isinstance(self.complete_frontier_accounting, bool):
            raise DoctorCausalLocalizationError(
                "complete_frontier_accounting must be a boolean"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DOCTOR_CAUSAL_LOCALIZATION_VERSION,
            "repository_id": self.repository_id,
            "snapshot_cid": self.snapshot_cid,
            "diagnostic_finding_cid": self.diagnostic_finding_cid,
            "issue_cid": self.issue_cid,
            "disposition": self.disposition.value,
            "selected_cause_id": self.selected_cause_id,
            "candidate_cause_ids": list(self.candidate_cause_ids),
            "mismatch_slice": self.mismatch_slice.to_dict(),
            "exact_evidence_ids": list(self.exact_evidence_ids),
            "nomination_evidence_ids": list(self.nomination_evidence_ids),
            "rejected_evidence_ids": list(self.rejected_evidence_ids),
            "evidence_dispositions": dict(self.evidence_dispositions),
            "mandatory_consumer_ids": list(self.mandatory_consumer_ids),
            "open_frontier_refs": list(self.open_frontier_refs),
            "reason_codes": list(self.reason_codes),
            "complete_frontier_accounting": self.complete_frontier_accounting,
        }

    @property
    def localization_cid(self) -> str:
        return self.content_id

    @property
    def localized(self) -> bool:
        return self.disposition is CausalLocalizationDisposition.LOCALIZED

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorCausalLocalizationReceipt":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DoctorCausalLocalizationError("unsupported localization receipt schema")
        value = cls(
            repository_id=str(payload.get("repository_id") or ""),
            snapshot_cid=str(payload.get("snapshot_cid") or ""),
            diagnostic_finding_cid=str(payload.get("diagnostic_finding_cid") or ""),
            issue_cid=str(payload.get("issue_cid") or ""),
            disposition=payload.get("disposition", ""),
            selected_cause_id=str(payload.get("selected_cause_id") or ""),
            candidate_cause_ids=tuple(payload.get("candidate_cause_ids") or ()),
            mismatch_slice=payload.get("mismatch_slice"),
            exact_evidence_ids=tuple(payload.get("exact_evidence_ids") or ()),
            nomination_evidence_ids=tuple(payload.get("nomination_evidence_ids") or ()),
            rejected_evidence_ids=tuple(payload.get("rejected_evidence_ids") or ()),
            evidence_dispositions=payload.get("evidence_dispositions") or {},
            mandatory_consumer_ids=tuple(payload.get("mandatory_consumer_ids") or ()),
            open_frontier_refs=tuple(payload.get("open_frontier_refs") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            complete_frontier_accounting=payload.get(
                "complete_frontier_accounting", False
            ),
        )
        supplied = payload.get("content_id", payload.get("cid", ""))
        if supplied not in (None, "", value.content_id):
            raise DoctorCausalLocalizationError("localization receipt identity mismatch")
        return value


def _current_fact_refs(snapshot: DoctorEvidenceSnapshot) -> frozenset[str]:
    refs: set[str] = {snapshot.snapshot_cid, snapshot.snapshot_id}
    refs.update(snapshot.finding_cids)
    for finding in snapshot.findings:
        refs.update(finding.observation_refs)
        refs.update(finding.evidence_refs)
    for hits in snapshot.query_index.values():
        for hit in hits:
            if hit.fact_id:
                refs.add(hit.fact_id)
    for item in snapshot.ast_index.path_records:
        refs.add(item.ast_record.record_id)
    return frozenset(refs)


def _evidence_disposition(
    evidence: CausalEvidence,
    snapshot: DoctorEvidenceSnapshot,
    current_refs: frozenset[str],
) -> CausalEvidenceDisposition:
    metadata_status = str(evidence.metadata.get("status") or "").casefold()
    if (
        evidence.poisoned
        or evidence.metadata.get("poisoned") is True
        or "poison" in metadata_status
    ):
        return CausalEvidenceDisposition.POISONED
    if (
        evidence.stale
        or evidence.metadata.get("stale") is True
        or "stale" in metadata_status
    ):
        return CausalEvidenceDisposition.STALE
    if evidence.nomination_only:
        return CausalEvidenceDisposition.NOMINATION_ONLY
    roots = snapshot.authority_roots
    mismatches = (
        evidence.snapshot_cid and evidence.snapshot_cid != snapshot.snapshot_cid,
        evidence.tree_id and roots.tree_id and evidence.tree_id != roots.tree_id,
        evidence.graph_id
        and roots.dependency_graph_id
        and evidence.graph_id != roots.dependency_graph_id,
        evidence.index_id
        and roots.ast_index_id
        and evidence.index_id != roots.ast_index_id,
    )
    if any(mismatches):
        return CausalEvidenceDisposition.ROOT_MISMATCH
    if not evidence.verified:
        return CausalEvidenceDisposition.UNVERIFIED
    if evidence.kind in _DECISIVE_KINDS and not evidence.minimized:
        return CausalEvidenceDisposition.INSUFFICIENT
    root_bound = bool(
        evidence.snapshot_cid
        or evidence.tree_id
        or evidence.graph_id
        or evidence.index_id
    )
    fact_bound = bool(set(evidence.fact_refs) & current_refs)
    if evidence.kind in _CURRENT_CHECKOUT_FACT_KINDS and not fact_bound:
        return CausalEvidenceDisposition.UNVERIFIED
    if not root_bound and not fact_bound:
        return CausalEvidenceDisposition.UNVERIFIED
    if not evidence.cause_ids:
        return CausalEvidenceDisposition.INSUFFICIENT
    return CausalEvidenceDisposition.EXACT


def _impact_facts(
    impact: Any, snapshot: DoctorEvidenceSnapshot
) -> tuple[tuple[str, ...], tuple[str, ...], bool, bool]:
    if impact is None:
        return (), (), False, False
    impact_roots = getattr(impact, "roots", None)
    snapshot_roots = snapshot.authority_roots
    stale = bool(
        impact_roots is None
        or (
            snapshot_roots.repository_id
            and getattr(impact_roots, "repository_id", "")
            != snapshot_roots.repository_id
        )
        or (
            snapshot_roots.tree_id
            and getattr(impact_roots, "tree_id", "") != snapshot_roots.tree_id
        )
        or (
            snapshot_roots.dependency_graph_id
            and getattr(impact, "current_graph_cid", "")
            != snapshot_roots.dependency_graph_id
        )
        or (
            snapshot_roots.ast_index_id
            and getattr(impact, "current_index_cid", "")
            != snapshot_roots.ast_index_id
        )
    )
    if stale:
        return (), ("frontier:stale_impact_closure",), False, True
    consumers = getattr(impact, "consumers", ())
    mandatory = tuple(
        sorted(
            {
                str(item.consumer_id)
                for item in consumers
                if bool(getattr(item, "mandatory", False))
            }
        )
    )
    frontiers = tuple(
        sorted(str(item) for item in getattr(impact, "open_required_frontiers", ()))
    )
    completeness = getattr(getattr(impact, "completeness", None), "value", "")
    complete = (
        completeness in {"complete", "partial_with_frontier"}
        and not getattr(impact, "missed_consumer_ids", ())
        and not getattr(impact, "stale_consumer_ids", ())
        and not getattr(impact, "duplicate_consumer_ids", ())
    )
    return mandatory, frontiers, complete, False


def _select_cause(
    exact: Sequence[CausalEvidence],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return selected cause, all candidates, and deterministic reason codes."""

    support_domains: dict[str, set[str]] = defaultdict(set)
    support_count: dict[str, int] = defaultdict(int)
    decisive_sets: list[set[str]] = []
    for item in exact:
        causes = set(item.cause_ids)
        if item.kind in _DECISIVE_KINDS:
            decisive_sets.append(causes)
        for cause in causes:
            support_domains[cause].add(item.domain)
            support_count[cause] += 1
    candidates = tuple(sorted(support_domains))
    if not candidates:
        return "", (), ("no_exact_causal_candidate", "correct_abstention")
    decisive_intersection = set.intersection(*decisive_sets) if decisive_sets else set()
    if decisive_sets and not decisive_intersection:
        return "", candidates, ("contradictory_decisive_evidence", "correct_abstention")

    eligible: list[str] = []
    for candidate in candidates:
        domains = support_domains[candidate]
        has_decisive = candidate in decisive_intersection if decisive_sets else False
        fused_foundational = len(domains & {"contract", "graph", "dataflow", "runtime"}) >= 3
        decisive_fusion = has_decisive and bool(
            domains & {"contract", "graph", "dataflow", "runtime"}
        )
        if fused_foundational or decisive_fusion:
            eligible.append(candidate)
    if not eligible:
        return "", candidates, ("insufficient_independent_exact_evidence", "correct_abstention")

    ranks = {
        candidate: (len(support_domains[candidate]), support_count[candidate])
        for candidate in eligible
    }
    best_rank = max(ranks.values())
    winners = sorted(candidate for candidate, rank in ranks.items() if rank == best_rank)
    if len(winners) != 1:
        return "", candidates, ("ambiguous_exact_causes", "correct_abstention")
    selected = winners[0]
    if decisive_sets and selected not in decisive_intersection:
        return "", candidates, ("decisive_evidence_disagrees", "correct_abstention")
    return selected, candidates, ("unique_fused_exact_cause",)


def _minimal_cover(
    selected: str, exact: Sequence[CausalEvidence]
) -> tuple[CausalEvidence, ...]:
    """One deterministic item per independent domain, preferring small facts."""

    by_domain: dict[str, list[CausalEvidence]] = defaultdict(list)
    for item in exact:
        if selected in item.cause_ids:
            by_domain[item.domain].append(item)
    chosen: list[CausalEvidence] = []
    for domain in sorted(by_domain):
        chosen.append(
            min(
                by_domain[domain],
                key=lambda item: (
                    0 if item.kind in _DECISIVE_KINDS else 1,
                    len(item.fact_refs),
                    item.evidence_id,
                ),
            )
        )
    return tuple(chosen)


def _slice_for(
    *,
    issue_cid: str,
    selected: str,
    selected_evidence: Sequence[CausalEvidence],
    mandatory_consumers: Sequence[str],
    open_frontiers: Sequence[str],
) -> MinimalMismatchSlice:
    buckets: dict[str, set[str]] = defaultdict(set)
    evidence_ids: list[str] = []
    for item in selected_evidence:
        evidence_ids.append(item.evidence_id)
        bucket = item.domain
        if item.kind is CausalEvidenceKind.DELTA_DEBUG:
            bucket = "delta_debug"
        elif item.kind is CausalEvidenceKind.UNSAT_CORE:
            bucket = "unsat_core"
        elif item.kind is CausalEvidenceKind.COUNTEREXAMPLE:
            bucket = "counterexample"
        buckets[bucket].update(item.fact_refs or (item.evidence_id,))
    return MinimalMismatchSlice(
        issue_cid=issue_cid,
        cause_id=selected,
        evidence_ids=tuple(evidence_ids),
        contract_refs=tuple(buckets["contract"]),
        graph_refs=tuple(buckets["graph"]),
        dataflow_refs=tuple(buckets["dataflow"]),
        runtime_refs=tuple(buckets["runtime"]),
        delta_debug_refs=tuple(buckets["delta_debug"]),
        unsat_core_refs=tuple(buckets["unsat_core"]),
        counterexample_refs=tuple(buckets["counterexample"]),
        mandatory_consumer_ids=tuple(mandatory_consumers),
        open_frontier_refs=tuple(open_frontiers),
    )


class DoctorCausalLocalizer:
    """Fuse exact causal facts and correctly abstain on residual ambiguity."""

    INTERFACE: ClassVar[str] = DOCTOR_CAUSAL_LOCALIZATION_INTERFACE

    def localize(
        self,
        request: DoctorCausalLocalizationRequest | Mapping[str, Any],
    ) -> DoctorCausalLocalizationReceipt:
        if isinstance(request, Mapping):
            request = DoctorCausalLocalizationRequest(**request)
        if not isinstance(request, DoctorCausalLocalizationRequest):
            raise DoctorCausalLocalizationError("request has invalid type")
        snapshot = request.snapshot
        finding = request.finding
        assert finding is not None  # normalized by request
        current_refs = _current_fact_refs(snapshot)

        exact: list[CausalEvidence] = []
        nominations: list[str] = []
        rejected: list[str] = []
        dispositions: dict[str, str] = {}
        evidence_frontiers: set[str] = set()
        for item in request.evidence:
            disposition = _evidence_disposition(item, snapshot, current_refs)
            dispositions[item.evidence_id] = disposition.value
            evidence_frontiers.update(item.frontier_refs)
            if disposition is CausalEvidenceDisposition.EXACT:
                exact.append(item)
            elif disposition is CausalEvidenceDisposition.NOMINATION_ONLY:
                nominations.append(item.evidence_id)
            else:
                rejected.append(item.evidence_id)

        selected, candidates, selection_reasons = _select_cause(exact)
        mandatory, impact_frontiers, impact_complete, impact_stale = _impact_facts(
            request.impact_closure, snapshot
        )
        open_frontiers = tuple(
            sorted(
                set(snapshot.open_frontiers)
                | set(finding.open_frontier_refs)
                | set(request.required_frontiers)
                | evidence_frontiers
                | set(impact_frontiers)
            )
        )
        chosen = _minimal_cover(selected, exact) if selected else ()
        mismatch_slice = _slice_for(
            issue_cid=request.issue_cid,
            selected=selected,
            selected_evidence=chosen,
            mandatory_consumers=mandatory,
            open_frontiers=open_frontiers,
        )
        reasons = list(selection_reasons)
        if nominations:
            reasons.append("retrieval_nomination_non_authoritative")
        if rejected:
            reasons.append("rejected_untrusted_evidence")
        if request.impact_closure is None:
            reasons.append("impact_closure_unavailable")
        elif impact_stale:
            reasons.append("stale_impact_closure_rejected")
        elif not impact_complete:
            reasons.append("impact_frontier_open")
        elif impact_frontiers:
            reasons.append("impact_frontier_open")
        if open_frontiers:
            reasons.append("open_frontiers_preserved")

        return DoctorCausalLocalizationReceipt(
            repository_id=snapshot.authority_roots.repository_id,
            snapshot_cid=snapshot.snapshot_cid,
            diagnostic_finding_cid=finding.finding_cid,
            issue_cid=request.issue_cid,
            disposition=(
                CausalLocalizationDisposition.LOCALIZED
                if selected
                else CausalLocalizationDisposition.ABSTAINED
            ),
            selected_cause_id=selected,
            candidate_cause_ids=candidates,
            mismatch_slice=mismatch_slice,
            exact_evidence_ids=tuple(item.evidence_id for item in exact),
            nomination_evidence_ids=tuple(nominations),
            rejected_evidence_ids=tuple(rejected),
            evidence_dispositions=dispositions,
            mandatory_consumer_ids=mandatory,
            open_frontier_refs=open_frontiers,
            reason_codes=tuple(reasons),
            complete_frontier_accounting=bool(
                request.impact_closure is not None and impact_complete
            ),
        )

    diagnose = localize


def localize_doctor_cause(
    request: DoctorCausalLocalizationRequest | Mapping[str, Any],
) -> DoctorCausalLocalizationReceipt:
    """Functional entry point for deterministic causal localization."""

    return DoctorCausalLocalizer().localize(request)


# Friendly aliases for plan terminology and adjacent adapters.
DoctorCausalLocalization = DoctorCausalLocalizer
DoctorCausalLocalizationResult = DoctorCausalLocalizationReceipt
CausalEvidenceObservation = CausalEvidence
CausalMismatchSlice = MinimalMismatchSlice
CausalDisposition = CausalLocalizationDisposition
localize_cause = localize_doctor_cause


__all__ = [
    "DOCTOR_CAUSAL_EVIDENCE_SCHEMA",
    "DOCTOR_CAUSAL_LOCALIZATION_INTERFACE",
    "DOCTOR_CAUSAL_LOCALIZATION_SCHEMA",
    "DOCTOR_CAUSAL_LOCALIZATION_VERSION",
    "DOCTOR_MISMATCH_SLICE_SCHEMA",
    "CausalDisposition",
    "CausalEvidence",
    "CausalEvidenceDisposition",
    "CausalEvidenceKind",
    "CausalEvidenceObservation",
    "CausalLocalizationDisposition",
    "CausalMismatchSlice",
    "DoctorCausalLocalization",
    "DoctorCausalLocalizationBoundsError",
    "DoctorCausalLocalizationError",
    "DoctorCausalLocalizationReceipt",
    "DoctorCausalLocalizationRequest",
    "DoctorCausalLocalizationResult",
    "DoctorCausalLocalizer",
    "MinimalMismatchSlice",
    "localize_cause",
    "localize_doctor_cause",
]
