"""SCA / MCP contract diagnosis bridge for the deterministic Doctor (DCR-051).

Interfaces
----------
* ``DoctorFinding@1`` — one earliest-broken-edge diagnosis with minimal evidence.
* ``ScaDoctorBridge@1`` — pure diagnose surface over DCR-024 mismatch findings.

Predicted symbols: :func:`diagnose_contract_failure`, :class:`DoctorFinding`.

Normative rules (fail-closed)
-----------------------------
* Exact finding enums and graph order replace substring matching / lexical
  guesses.
* Earliest failing edge is selected on the mandatory consumer path:
  declaration → registration → dispatcher → handler → effect → response.
* Same graph + transcript epoch always yields the same diagnosis CID.
* Ambiguity, stale / mixed-epoch bytes, and unsupported logic return typed
  abstain / defer dispositions and never invent a transform.
* The bridge never loads an LLM surface or executes a repair operator.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .analysis.mcp_contract_graph import (
    MANDATORY_EDGE_KINDS,
    McpContractGraph,
    McpContractGraphError,
    load_mcp_contract_graph,
    materialize_mcp_contract_graph,
)
from .analysis.mcp_contract_mismatch import (
    CONTRACT_MISMATCH_INTERFACE,
    MISMATCH_EVIDENCE_TERM,
    ContractMismatch,
    McpContractMismatchError,
    MismatchClass,
    build_mismatch_findings,
    classify_and_deduplicate,
    earliest_broken_edge,
    materialize_mcp_contract_mismatch_findings,
)
from .analysis.mcp_live_observer import load_mcp_live_transcript
from .analysis.deterministic_doctor_contracts import DoctorRepairDisposition
from .proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Closed interface / evidence constants
# ---------------------------------------------------------------------------

SCA_DOCTOR_BRIDGE_INTERFACE: Final[str] = "ScaDoctorBridge@1"
DOCTOR_FINDING_INTERFACE: Final[str] = "DoctorFinding@1"
SCA_DOCTOR_BRIDGE_VERSION: Final[int] = 1
SCA_DOCTOR_DIAGNOSIS_EVIDENCE: Final[str] = "dcr/doctor-diagnosis@1"

DOCTOR_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-finding@1"
)
DOCTOR_DIAGNOSIS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-diagnosis@1"
)
DOCTOR_FINDINGS_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-findings-catalog@1"
)

DEFAULT_DOCTOR_FINDINGS_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/doctor-findings.json"
)

# Mandatory consumer path (authoritative order).
MANDATORY_PATH_ORDER: Final[tuple[str, ...]] = tuple(MANDATORY_EDGE_KINDS)


class DoctorDiagnosisDisposition(str, Enum):  # noqa: UP042
    """Closed diagnosis outcomes for DCR-051 (no transform authority)."""

    DIAGNOSED = "diagnosed"
    ABSTAIN_REVIEW = "abstain_review"
    DEFER_CAPABILITY = "defer_capability"


class ScaDoctorBridgeError(ContractValidationError):
    """Malformed diagnosis input or closed-boundary violation."""


# ---------------------------------------------------------------------------
# DoctorFinding@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorFinding(CanonicalContract):
    """Earliest-broken-edge diagnosis with minimal supporting evidence.

    Interface: ``DoctorFinding@1``

    Wraps a DCR-024 :class:`ContractMismatch` without granting write/transform
    authority.  The finding is content-addressed and reconstructible.
    """

    SCHEMA: ClassVar[str] = DOCTOR_FINDING_SCHEMA
    INTERFACE: ClassVar[str] = DOCTOR_FINDING_INTERFACE

    mismatch: ContractMismatch
    edge_key: str
    epoch_cid: str
    source_hashes: Mapping[str, str] = MappingProxyType({})
    source_spans: tuple[Mapping[str, Any], ...] = ()
    transcript_cid: str = ""
    logic_family: str = ""
    reason_codes: tuple[str, ...] = ()
    grants_transform_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mismatch, ContractMismatch):
            if isinstance(self.mismatch, Mapping):
                object.__setattr__(
                    self, "mismatch", ContractMismatch.from_dict(self.mismatch)
                )
            else:
                raise ScaDoctorBridgeError("mismatch must be a ContractMismatch")
        edge_key = str(self.edge_key or self.mismatch.edge_kind).strip()
        if not edge_key:
            raise ScaDoctorBridgeError("edge_key is required")
        object.__setattr__(self, "edge_key", edge_key)
        epoch = str(self.epoch_cid or "").strip()
        if not epoch:
            raise ScaDoctorBridgeError("epoch_cid is required")
        object.__setattr__(self, "epoch_cid", epoch)
        hashes = {
            str(k): str(v)
            for k, v in dict(self.source_hashes or {}).items()
            if str(k).strip() and str(v).strip()
        }
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(sorted(hashes.items()))))
        spans: list[Mapping[str, Any]] = []
        for item in self.source_spans or ():
            if isinstance(item, Mapping):
                spans.append(MappingProxyType(dict(item)))
        object.__setattr__(self, "source_spans", tuple(spans))
        object.__setattr__(self, "transcript_cid", str(self.transcript_cid or "").strip())
        object.__setattr__(self, "logic_family", str(self.logic_family or "").strip())
        codes: list[str] = []
        for raw in self.reason_codes or ():
            text = str(raw).strip()
            if text and text not in codes:
                codes.append(text)
        if not codes:
            codes.append(self.mismatch.reason_code or "diagnosed")
        object.__setattr__(self, "reason_codes", tuple(codes))
        # Diagnosis never grants transform authority (DCR-051 / Doctor contract).
        object.__setattr__(self, "grants_transform_authority", False)

    @property
    def finding_enum(self) -> str:
        return self.mismatch.mismatch_class.value

    @property
    def finding_id(self) -> str:
        return self.mismatch.finding_id or self.content_id

    @property
    def counterexample(self) -> Mapping[str, Any]:
        return dict(self.mismatch.counterexample_seed)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "mismatch": self.mismatch.to_dict(),
            "edge_key": self.edge_key,
            "epoch_cid": self.epoch_cid,
            "source_hashes": dict(self.source_hashes),
            "source_spans": [dict(item) for item in self.source_spans],
            "transcript_cid": self.transcript_cid,
            "logic_family": self.logic_family,
            "reason_codes": list(self.reason_codes),
            "grants_transform_authority": False,
            "finding_enum": self.finding_enum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorFinding":
        if not isinstance(payload, Mapping):
            raise ScaDoctorBridgeError("doctor finding must be an object")
        return cls(
            mismatch=payload.get("mismatch") or {},
            edge_key=str(payload.get("edge_key") or ""),
            epoch_cid=str(payload.get("epoch_cid") or ""),
            source_hashes=payload.get("source_hashes") or {},
            source_spans=tuple(payload.get("source_spans") or ()),
            transcript_cid=str(payload.get("transcript_cid") or ""),
            logic_family=str(payload.get("logic_family") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )

    @classmethod
    def from_mismatch(
        cls,
        mismatch: ContractMismatch,
        *,
        epoch_cid: str,
        source_hashes: Mapping[str, str] | None = None,
        source_spans: Sequence[Mapping[str, Any]] | None = None,
        transcript_cid: str = "",
        logic_family: str = "",
        reason_codes: Sequence[str] | None = None,
    ) -> "DoctorFinding":
        return cls(
            mismatch=mismatch,
            edge_key=mismatch.edge_kind,
            epoch_cid=epoch_cid,
            source_hashes=source_hashes or {},
            source_spans=tuple(source_spans or ()),
            transcript_cid=transcript_cid,
            logic_family=logic_family,
            reason_codes=tuple(reason_codes or (mismatch.reason_code,)),
        )


# ---------------------------------------------------------------------------
# Diagnosis result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorDiagnosis(CanonicalContract):
    """Closed result of :func:`diagnose_contract_failure`."""

    SCHEMA: ClassVar[str] = DOCTOR_DIAGNOSIS_SCHEMA

    disposition: DoctorDiagnosisDisposition
    epoch_cid: str
    findings: tuple[DoctorFinding, ...] = ()
    earliest: DoctorFinding | None = None
    reason_codes: tuple[str, ...] = ()
    grants_transform_authority: bool = False
    evidence_id: str = SCA_DOCTOR_DIAGNOSIS_EVIDENCE

    def __post_init__(self) -> None:
        try:
            disposition = DoctorDiagnosisDisposition(
                str(getattr(self.disposition, "value", self.disposition))
            )
        except ValueError as exc:
            raise ScaDoctorBridgeError(
                f"unsupported diagnosis disposition: {self.disposition!r}"
            ) from exc
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "epoch_cid", str(self.epoch_cid or "").strip())
        findings = tuple(
            item if isinstance(item, DoctorFinding) else DoctorFinding.from_dict(item)
            for item in (self.findings or ())
        )
        object.__setattr__(self, "findings", findings)
        earliest = self.earliest
        if earliest is not None and not isinstance(earliest, DoctorFinding):
            earliest = DoctorFinding.from_dict(earliest)
        if earliest is None and findings and disposition is DoctorDiagnosisDisposition.DIAGNOSED:
            earliest = findings[0]
        object.__setattr__(self, "earliest", earliest)
        codes: list[str] = []
        for raw in self.reason_codes or ():
            text = str(raw).strip()
            if text and text not in codes:
                codes.append(text)
        if not codes:
            codes.append(disposition.value)
        object.__setattr__(self, "reason_codes", tuple(codes))
        # Diagnosis never authorizes a transform (DCR-051 acceptance).
        object.__setattr__(self, "grants_transform_authority", False)
        object.__setattr__(self, "evidence_id", SCA_DOCTOR_DIAGNOSIS_EVIDENCE)

    @property
    def is_actionable(self) -> bool:
        return (
            self.disposition is DoctorDiagnosisDisposition.DIAGNOSED
            and self.earliest is not None
            and not self.grants_transform_authority
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": SCA_DOCTOR_BRIDGE_INTERFACE,
            "evidence_id": self.evidence_id,
            "disposition": self.disposition.value,
            "epoch_cid": self.epoch_cid,
            "findings": [item.to_dict() for item in self.findings],
            "earliest": None if self.earliest is None else self.earliest.to_dict(),
            "reason_codes": list(self.reason_codes),
            "grants_transform_authority": False,
            "version": SCA_DOCTOR_BRIDGE_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorDiagnosis":
        if not isinstance(payload, Mapping):
            raise ScaDoctorBridgeError("doctor diagnosis must be an object")
        return cls(
            disposition=payload.get("disposition") or DoctorDiagnosisDisposition.ABSTAIN_REVIEW,
            epoch_cid=str(payload.get("epoch_cid") or ""),
            findings=tuple(payload.get("findings") or ()),
            earliest=payload.get("earliest"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


# ---------------------------------------------------------------------------
# Diagnosis pipeline
# ---------------------------------------------------------------------------


def _epoch_cid(
    graph: McpContractGraph,
    transcript: Mapping[str, Any] | None,
) -> str:
    graph_cid = getattr(graph, "content_id", None) or content_identity(
        graph.to_dict() if hasattr(graph, "to_dict") else {"graph": True}
    )
    if transcript is None:
        return content_identity({"graph_cid": graph_cid, "transcript": None})
    for key in ("transcript_cid", "receipt_cid", "local_cid", "epoch_cid"):
        value = transcript.get(key)
        if isinstance(value, str) and value.strip():
            return content_identity(
                {"graph_cid": graph_cid, "transcript_cid": value.strip()}
            )
    return content_identity(
        {
            "graph_cid": graph_cid,
            "transcript": {
                k: transcript.get(k)
                for k in sorted(transcript)
                if k
                in (
                    "schema",
                    "evidence_term",
                    "observations",
                    "epoch",
                    "snapshot_id",
                )
            },
        }
    )


def _source_evidence(mismatch: ContractMismatch) -> tuple[dict[str, str], tuple[dict[str, Any], ...]]:
    hashes: dict[str, str] = {}
    spans: list[dict[str, Any]] = []
    for edge in (mismatch.expected_edge, mismatch.observed_edge):
        if not isinstance(edge, Mapping):
            continue
        path = str(edge.get("path") or edge.get("source_path") or "").strip()
        digest = str(edge.get("content_hash") or edge.get("sha256") or "").strip()
        if path and digest:
            hashes[path] = digest
        span = edge.get("span")
        if isinstance(span, Mapping):
            spans.append(dict(span))
        elif path:
            spans.append({"path": path})
    # Counterexample may carry span/hash clues without source bodies.
    seed = mismatch.counterexample_seed
    if isinstance(seed, Mapping):
        path = str(seed.get("path") or "").strip()
        digest = str(seed.get("content_hash") or seed.get("sha256") or "").strip()
        if path and digest:
            hashes[path] = digest
        if path and not any(item.get("path") == path for item in spans):
            spans.append(
                {
                    "path": path,
                    "start": seed.get("span_start"),
                    "end": seed.get("span_end"),
                }
            )
    return hashes, tuple(spans)


def _is_ambiguous(mismatch: ContractMismatch) -> bool:
    return mismatch.mismatch_class in {
        MismatchClass.AMBIGUOUS,
        MismatchClass.UNOBSERVED,
    } or "ambiguous" in (mismatch.reason_code or "").lower()


def _is_stale_or_unsupported(reason_codes: Sequence[str]) -> bool:
    tokens = " ".join(reason_codes).lower()
    return any(
        marker in tokens
        for marker in (
            "stale",
            "mixed_epoch",
            "epoch_mismatch",
            "unsupported",
            "transcript_load_failed",
            "logic_unavailable",
            "capability",
        )
    )


def diagnose_contract_failure(
    graph: McpContractGraph | None = None,
    transcript: Mapping[str, Any] | None = None,
    *,
    consumer_id: str | None = None,
    require_shared_epoch: bool = True,
    repo_root: str | Path | None = None,
    logic_family: str = "",
) -> DoctorDiagnosis:
    """Diagnose the earliest broken contract edge for one graph/transcript epoch.

    Returns a content-addressed :class:`DoctorDiagnosis`.  On success the
    ``earliest`` finding is set and ``grants_transform_authority`` is always
    false — transform selection is DCR-052.
    """

    root = Path(repo_root).resolve() if repo_root is not None else None
    mismatches: tuple[ContractMismatch, ...] = ()
    epoch = ""
    working_graph: McpContractGraph | None = graph

    try:
        if working_graph is None and root is not None:
            catalog = materialize_mcp_contract_mismatch_findings(
                repo_root=root,
                require_shared_epoch=require_shared_epoch,
            )
            mismatches = tuple(catalog.findings)
            epoch = content_identity(
                {
                    "graph_cid": catalog.graph_cid,
                    "transcript_epoch": catalog.transcript_epoch,
                    "snapshot_id": catalog.snapshot_id,
                }
            )
        else:
            if working_graph is None:
                working_graph = materialize_mcp_contract_graph()
            if not isinstance(working_graph, McpContractGraph):
                raise ScaDoctorBridgeError("graph must be an McpContractGraph")
            catalog = build_mismatch_findings(
                working_graph,
                transcript,
                require_shared_epoch=require_shared_epoch,
            )
            mismatches = tuple(catalog.findings)
            epoch = _epoch_cid(working_graph, transcript)

            if consumer_id:
                blocker = earliest_broken_edge(working_graph, consumer_id)
                if blocker is not None:
                    focused = [
                        item
                        for item in mismatches
                        if item.consumer_id == consumer_id
                        and item.edge_kind == blocker.edge_kind
                    ]
                    if focused:
                        mismatches = tuple(focused) + tuple(
                            item for item in mismatches if item not in focused
                        )
    except (McpContractGraphError, McpContractMismatchError, OSError, ValueError, TypeError) as exc:
        reason = getattr(exc, "reason_code", None) or type(exc).__name__
        disposition = (
            DoctorDiagnosisDisposition.DEFER_CAPABILITY
            if _is_stale_or_unsupported((str(reason), str(exc)))
            else DoctorDiagnosisDisposition.ABSTAIN_REVIEW
        )
        return DoctorDiagnosis(
            disposition=disposition,
            epoch_cid=content_identity({"error": str(exc)[:200]}),
            findings=(),
            earliest=None,
            reason_codes=(str(reason), disposition.value, "no_transform"),
        )

    if not mismatches:
        return DoctorDiagnosis(
            disposition=DoctorDiagnosisDisposition.ABSTAIN_REVIEW,
            epoch_cid=epoch or content_identity({"empty": True}),
            findings=(),
            earliest=None,
            reason_codes=("no_mismatches", "abstain_review", "no_transform"),
        )

    doctor_findings: list[DoctorFinding] = []
    for mismatch in mismatches:
        hashes, spans = _source_evidence(mismatch)
        doctor_findings.append(
            DoctorFinding.from_mismatch(
                mismatch,
                epoch_cid=epoch,
                source_hashes=hashes,
                source_spans=spans,
                transcript_cid=(
                    str((transcript or {}).get("transcript_cid") or "")
                    if transcript
                    else ""
                ),
                logic_family=logic_family,
                reason_codes=(mismatch.reason_code, mismatch.mismatch_class.value),
            )
        )

    path_rank = {kind: index for index, kind in enumerate(MANDATORY_PATH_ORDER)}

    def _rank(item: DoctorFinding) -> tuple[int, str, str]:
        return (
            path_rank.get(item.edge_key, len(path_rank)),
            item.mismatch.consumer_id,
            item.finding_id,
        )

    ordered = tuple(sorted(doctor_findings, key=_rank))
    earliest = ordered[0]

    if _is_ambiguous(earliest.mismatch):
        return DoctorDiagnosis(
            disposition=DoctorDiagnosisDisposition.ABSTAIN_REVIEW,
            epoch_cid=epoch,
            findings=ordered,
            earliest=earliest,
            reason_codes=(
                "ambiguous_or_unobserved_earliest_edge",
                earliest.finding_enum,
                "no_transform",
            ),
        )

    return DoctorDiagnosis(
        disposition=DoctorDiagnosisDisposition.DIAGNOSED,
        epoch_cid=epoch,
        findings=ordered,
        earliest=earliest,
        reason_codes=(
            "earliest_broken_edge",
            earliest.edge_key,
            earliest.finding_enum,
            "no_transform",
        ),
    )



def materialize_doctor_findings(
    *,
    repo_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize ``doctor-findings.json`` for DCR-051 evidence."""

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    diagnosis = diagnose_contract_failure(repo_root=root, require_shared_epoch=True)
    payload = {
        "schema": DOCTOR_FINDINGS_CATALOG_SCHEMA,
        "interface": SCA_DOCTOR_BRIDGE_INTERFACE,
        "evidence_id": SCA_DOCTOR_DIAGNOSIS_EVIDENCE,
        "version": SCA_DOCTOR_BRIDGE_VERSION,
        "diagnosis": diagnosis.to_dict(),
        "mandatory_path_order": list(MANDATORY_PATH_ORDER),
        "runtime_model_calls": 0,
    }
    path = (
        Path(destination)
        if destination is not None
        else root.joinpath(*PurePosixPath(DEFAULT_DOCTOR_FINDINGS_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_DOCTOR_FINDINGS_PATH",
    "DOCTOR_DIAGNOSIS_SCHEMA",
    "DOCTOR_FINDING_INTERFACE",
    "DOCTOR_FINDING_SCHEMA",
    "DOCTOR_FINDINGS_CATALOG_SCHEMA",
    "MANDATORY_PATH_ORDER",
    "SCA_DOCTOR_BRIDGE_INTERFACE",
    "SCA_DOCTOR_BRIDGE_VERSION",
    "SCA_DOCTOR_DIAGNOSIS_EVIDENCE",
    "DoctorDiagnosis",
    "DoctorDiagnosisDisposition",
    "DoctorFinding",
    "ScaDoctorBridgeError",
    "diagnose_contract_failure",
    "materialize_doctor_findings",
]
