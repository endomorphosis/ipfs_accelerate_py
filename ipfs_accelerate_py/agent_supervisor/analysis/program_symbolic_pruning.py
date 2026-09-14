"""Authoritative symbolic pruning of conservative program successors.

Interface: ``SymbolicSuccessorPruner@1``
Evidence: ``sawm/static-successors@1``

Pruning may remove a successor only when an authoritative type, effect, path
condition, contract, capability, abstract-interpretation, or solver reason
proves it impossible.  Unknown, timeout, incomplete, and unavailable evidence
widen the unresolved frontier and never shrink the candidate set.

This module does not construct a second scanner, solver, or router.  It
evaluates caller-supplied symbolic evidence against a generated
:class:`StaticSuccessorPlan` and records every stage as selected, skipped,
unavailable, rejected, or escalated.  Model output cannot self-approve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph import (
    DynamicFrontierReason,
)

from .program_successors import (
    FORBIDDEN_FIELD_MARKERS,
    SAWM_STATIC_SUCCESSORS_EVIDENCE,
    StaticSuccessorError,
    StaticSuccessorPlan,
    SuccessorCandidate,
    SuccessorDecisionReceipt,
    SuccessorDisposition,
    SuccessorReasonCode,
    SuccessorStage,
    UnresolvedDynamicFrontier,
    _identity_cid,
    _optional_text,
    _plain,
    _sorted_unique,
    _text,
    record_stage,
)


SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE: Final[str] = "SymbolicSuccessorPruner@1"
SYMBOLIC_PRUNING_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-pruning-result@1"
)
SYMBOLIC_PRUNING_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-pruning-evidence@1"
)

PRUNER_VERSION: Final[str] = "1"
MAX_EVIDENCE_ITEMS: Final[int] = 4_096

PRUNING_STAGE_ORDER: Final[tuple[SuccessorStage, ...]] = (
    SuccessorStage.FRESHNESS,
    SuccessorStage.TYPE,
    SuccessorStage.EFFECT,
    SuccessorStage.PATH_CONDITION,
    SuccessorStage.CONTRACT,
    SuccessorStage.CAPABILITY,
    SuccessorStage.ABSTRACT_INTERPRETATION,
    SuccessorStage.SOLVER,
    SuccessorStage.RESIDUAL_ESCALATION,
    SuccessorStage.COMPLETENESS,
)

AUTHORITATIVE_PRUNE_STAGES: Final[frozenset[str]] = frozenset(
    stage.value for stage in PRUNING_STAGE_ORDER
)

_STAGE_IMPOSSIBLE_REASON: Final[dict[str, SuccessorReasonCode]] = {
    SuccessorStage.TYPE.value: SuccessorReasonCode.INCOMPATIBLE_TYPE,
    SuccessorStage.EFFECT.value: SuccessorReasonCode.INCOMPATIBLE_EFFECT,
    SuccessorStage.PATH_CONDITION.value: SuccessorReasonCode.UNSAT_PATH,
    SuccessorStage.CONTRACT.value: SuccessorReasonCode.CONTRACT_UNSAT,
    SuccessorStage.ABSTRACT_INTERPRETATION.value: (
        SuccessorReasonCode.EMPTY_ABSTRACT_STATE
    ),
    SuccessorStage.SOLVER.value: SuccessorReasonCode.SOLVER_UNSAT,
}

_STAGE_UNKNOWN_REASON: Final[dict[str, SuccessorReasonCode]] = {
    SuccessorStage.TYPE.value: SuccessorReasonCode.UNKNOWN_WIDENED,
    SuccessorStage.EFFECT.value: SuccessorReasonCode.UNKNOWN_WIDENED,
    SuccessorStage.PATH_CONDITION.value: SuccessorReasonCode.UNKNOWN_WIDENED,
    SuccessorStage.CONTRACT.value: SuccessorReasonCode.CONTRACT_UNKNOWN,
    SuccessorStage.CAPABILITY.value: SuccessorReasonCode.CAPABILITY_UNAVAILABLE,
    SuccessorStage.ABSTRACT_INTERPRETATION.value: (
        SuccessorReasonCode.ABSTRACT_INCOMPLETE
    ),
    SuccessorStage.SOLVER.value: SuccessorReasonCode.SOLVER_UNKNOWN,
}

_STAGE_FRONTIER_REASON: Final[dict[str, str]] = {
    SuccessorStage.TYPE.value: DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,
    SuccessorStage.EFFECT.value: DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,
    SuccessorStage.PATH_CONDITION.value: DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,
    SuccessorStage.CONTRACT.value: DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,
    SuccessorStage.CAPABILITY.value: DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,
    SuccessorStage.ABSTRACT_INTERPRETATION.value: (
        DynamicFrontierReason.INCOMPLETE_ANALYSIS.value
    ),
    SuccessorStage.SOLVER.value: DynamicFrontierReason.SOLVER_UNKNOWN.value,
}

ALLOWED_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "type",
        "effect",
        "path_condition",
        "contract",
        "capability",
        "abstract_interpretation",
        "solver",
        "ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph",
        "ipfs_datasets_py.logic.software_verification.program_graph_builder",
    }
)


class SymbolicPruningError(StaticSuccessorError):
    """Symbolic pruning evidence is malformed or non-authoritative."""


class PruningVerdict(str, Enum):
    """Closed evidence verdicts.  Only ``impossible`` may remove a successor."""

    IMPOSSIBLE = "impossible"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"
    UNAVAILABLE = "unavailable"


class SolverStatus(str, Enum):
    """Closed solver outcomes.  Timeout and unknown widen."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


_NON_PRUNING_VERDICTS: Final[frozenset[str]] = frozenset(
    {
        PruningVerdict.UNKNOWN.value,
        PruningVerdict.TIMEOUT.value,
        PruningVerdict.INCOMPLETE.value,
        PruningVerdict.UNAVAILABLE.value,
    }
)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SymbolicPruningError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise SymbolicPruningError(f"{name} has unsupported value {value!r}") from exc


@dataclass(frozen=True, slots=True)
class SymbolicPruningEvidence:
    """One typed symbolic fact about a successor.  Similarity cannot appear."""

    stage: SuccessorStage | str
    successor_node_cid: str
    verdict: PruningVerdict | str
    reason_code: SuccessorReasonCode | str
    authoritative: bool = False
    evidence_cid: str | None = None
    authority: str = ""
    candidate_cid: str | None = None
    diagnostic: str = ""

    SCHEMA: ClassVar[str] = SYMBOLIC_PRUNING_EVIDENCE_SCHEMA
    CID_FIELD: ClassVar[str] = "pruning_evidence_cid"

    def __post_init__(self) -> None:
        stage = _enum(self.stage, SuccessorStage, "stage")
        if stage not in AUTHORITATIVE_PRUNE_STAGES:
            raise SymbolicPruningError(f"unsupported pruning stage {stage!r}")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self,
            "successor_node_cid",
            _text(self.successor_node_cid, "successor_node_cid"),
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, PruningVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, SuccessorReasonCode, "reason_code"),
        )
        object.__setattr__(
            self, "authoritative", _bool(self.authoritative, "authoritative")
        )
        object.__setattr__(
            self, "evidence_cid", _optional_text(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self, "authority", _text(self.authority, "authority", empty=True)
        )
        object.__setattr__(
            self, "candidate_cid", _optional_text(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self, "diagnostic", _text(self.diagnostic, "diagnostic", empty=True)
        )
        _plain(self.identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stage": self.stage,
            "successor_node_cid": self.successor_node_cid,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "authoritative": self.authoritative,
            "evidence_cid": self.evidence_cid,
            "authority": self.authority,
            "candidate_cid": self.candidate_cid,
            "diagnostic": self.diagnostic,
        }

    @property
    def pruning_evidence_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["pruning_evidence_cid"] = self.pruning_evidence_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SymbolicPruningEvidence":
        payload = dict(data)
        claimed = payload.pop("pruning_evidence_cid", None)
        payload.pop("schema", None)
        result = cls(**payload)
        if claimed is not None and claimed != result.pruning_evidence_cid:
            raise SymbolicPruningError("pruning evidence identity does not match")
        return result

    @property
    def can_remove(self) -> bool:
        return (
            self.authoritative
            and self.verdict == PruningVerdict.IMPOSSIBLE.value
        )

    @property
    def widens(self) -> bool:
        return self.verdict in _NON_PRUNING_VERDICTS


def _coerce_plan(value: Any) -> StaticSuccessorPlan:
    if isinstance(value, StaticSuccessorPlan):
        return value
    if isinstance(value, Mapping):
        return StaticSuccessorPlan.from_dict(value)
    raise SymbolicPruningError("plan must be a StaticSuccessorPlan or mapping")


def _coerce_evidence_item(value: Any, default_stage: str) -> SymbolicPruningEvidence:
    if isinstance(value, SymbolicPruningEvidence):
        return value
    if not isinstance(value, Mapping):
        raise SymbolicPruningError("pruning evidence must be a mapping")
    payload = dict(value)
    forbidden = set(payload) & FORBIDDEN_FIELD_MARKERS
    if forbidden:
        raise SymbolicPruningError(
            "pruning evidence rejects non-semantic fields "
            + ", ".join(sorted(forbidden))
        )
    payload.setdefault("stage", default_stage)
    if "successor_node_cid" not in payload:
        for alias in ("successor_cid", "target_node_cid", "node_cid"):
            if alias in payload:
                payload["successor_node_cid"] = payload[alias]
                break
    if "verdict" not in payload and "status" in payload:
        payload["verdict"] = payload["status"]
    if "reason_code" not in payload:
        verdict = str(payload.get("verdict", PruningVerdict.UNKNOWN.value))
        if verdict == PruningVerdict.IMPOSSIBLE.value:
            payload["reason_code"] = _STAGE_IMPOSSIBLE_REASON.get(
                default_stage, SuccessorReasonCode.IMPOSSIBLE
            )
        elif verdict == PruningVerdict.TIMEOUT.value:
            payload["reason_code"] = SuccessorReasonCode.SOLVER_TIMEOUT
        elif verdict == PruningVerdict.UNAVAILABLE.value:
            payload["reason_code"] = SuccessorReasonCode.CAPABILITY_UNAVAILABLE
        else:
            payload["reason_code"] = _STAGE_UNKNOWN_REASON.get(
                default_stage, SuccessorReasonCode.UNKNOWN_WIDENED
            )
    payload.setdefault("authoritative", False)
    return SymbolicPruningEvidence.from_dict(payload) if "schema" in payload else SymbolicPruningEvidence(**{
        key: payload[key]
        for key in (
            "stage",
            "successor_node_cid",
            "verdict",
            "reason_code",
            "authoritative",
            "evidence_cid",
            "authority",
            "candidate_cid",
            "diagnostic",
        )
        if key in payload
    })


def _coerce_evidence(
    values: Any,
    default_stage: str,
) -> tuple[SymbolicPruningEvidence, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise SymbolicPruningError("evidence must be a sequence")
    items = tuple(
        _coerce_evidence_item(item, default_stage) for item in values
    )
    if len(items) > MAX_EVIDENCE_ITEMS:
        raise SymbolicPruningError("pruning evidence bound exceeded")
    return items


def _matches_candidate(
    evidence: SymbolicPruningEvidence,
    candidate: SuccessorCandidate,
) -> bool:
    if evidence.candidate_cid and evidence.candidate_cid == candidate.candidate_cid:
        return True
    return evidence.successor_node_cid == candidate.target_node_cid


def _fill_pruning_receipts(
    receipts: Sequence[SuccessorDecisionReceipt],
    *,
    blocked: bool,
    candidate_cids: Sequence[str] = (),
) -> tuple[SuccessorDecisionReceipt, ...]:
    recorded = {item.stage: item for item in receipts}
    filled: list[SuccessorDecisionReceipt] = []
    for stage in PRUNING_STAGE_ORDER:
        existing = recorded.get(stage.value)
        if existing is not None:
            filled.append(existing)
            continue
        filled.append(
            record_stage(
                stage,
                SuccessorDisposition.SKIPPED,
                SuccessorReasonCode.FRESHNESS_BLOCKED
                if blocked
                else SuccessorReasonCode.NO_EVIDENCE,
                candidate_cids_in=candidate_cids,
                candidate_cids_out=candidate_cids,
            )
        )
    return tuple(filled)


@dataclass(frozen=True, slots=True)
class SymbolicPruningResult:
    """Pruned successor set with impossible/incomplete split and stage receipts."""

    plan_cid: str
    snapshot_cid: str
    graph_cid: str
    language: str
    retained: Sequence[SuccessorCandidate]
    removed: Sequence[SuccessorCandidate] = ()
    frontier: UnresolvedDynamicFrontier | None = None
    receipts: Sequence[SuccessorDecisionReceipt] = ()
    generation_receipts: Sequence[SuccessorDecisionReceipt] = ()
    complete: bool = False
    impossible_cids: Sequence[str] = ()
    incomplete_cids: Sequence[str] = ()
    unavailable_dimensions: Sequence[str] = ()
    original_plan_cid: str | None = None

    SCHEMA: ClassVar[str] = SYMBOLIC_PRUNING_RESULT_SCHEMA
    INTERFACE: ClassVar[str] = SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE
    CID_FIELD: ClassVar[str] = "pruning_result_cid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_cid", _text(self.plan_cid, "plan_cid"))
        object.__setattr__(self, "snapshot_cid", _text(self.snapshot_cid, "snapshot_cid"))
        object.__setattr__(self, "graph_cid", _text(self.graph_cid, "graph_cid"))
        object.__setattr__(self, "language", _text(self.language, "language"))
        retained = tuple(self.retained)
        removed = tuple(self.removed)
        for item in (*retained, *removed):
            if not isinstance(item, SuccessorCandidate):
                raise SymbolicPruningError(
                    "pruning results must contain SuccessorCandidate values"
                )
        object.__setattr__(
            self,
            "retained",
            tuple(sorted(retained, key=lambda item: item.candidate_cid)),
        )
        object.__setattr__(
            self,
            "removed",
            tuple(sorted(removed, key=lambda item: item.candidate_cid)),
        )
        frontier = self.frontier
        if frontier is None:
            frontier = UnresolvedDynamicFrontier(language=self.language)
        elif not isinstance(frontier, UnresolvedDynamicFrontier):
            raise SymbolicPruningError("frontier must be UnresolvedDynamicFrontier")
        object.__setattr__(self, "frontier", frontier)
        object.__setattr__(self, "receipts", tuple(self.receipts))
        object.__setattr__(
            self, "generation_receipts", tuple(self.generation_receipts)
        )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(
            self,
            "impossible_cids",
            _sorted_unique(self.impossible_cids, "impossible_cids"),
        )
        object.__setattr__(
            self,
            "incomplete_cids",
            _sorted_unique(self.incomplete_cids, "incomplete_cids"),
        )
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _sorted_unique(self.unavailable_dimensions, "unavailable_dimensions"),
        )
        object.__setattr__(
            self,
            "original_plan_cid",
            _optional_text(self.original_plan_cid, "original_plan_cid"),
        )
        retained_ids = {item.candidate_cid for item in self.retained}
        if retained_ids & set(self.impossible_cids):
            raise SymbolicPruningError(
                "impossible successors cannot remain in the retained set"
            )
        if self.complete and (
            not frontier.empty
            or self.incomplete_cids
            or self.unavailable_dimensions
            or any(item.unknown or not item.complete for item in self.retained)
        ):
            raise SymbolicPruningError(
                "complete pruning results cannot retain unknown or incomplete residue"
            )

    @property
    def candidate_cids(self) -> tuple[str, ...]:
        return tuple(item.candidate_cid for item in self.retained)

    @property
    def target_node_cids(self) -> tuple[str, ...]:
        return _sorted_unique(
            (item.target_node_cid for item in self.retained), "target_node_cids"
        )

    def identity_payload(self) -> dict[str, Any]:
        assert self.frontier is not None
        return {
            "schema": self.SCHEMA,
            "plan_cid": self.plan_cid,
            "snapshot_cid": self.snapshot_cid,
            "graph_cid": self.graph_cid,
            "language": self.language,
            "retained": [item.to_dict() for item in self.retained],
            "removed": [item.to_dict() for item in self.removed],
            "frontier": self.frontier.to_dict(),
            "receipts": [item.to_dict() for item in self.receipts],
            "generation_receipts": [
                item.to_dict() for item in self.generation_receipts
            ],
            "complete": self.complete,
            "impossible_cids": list(self.impossible_cids),
            "incomplete_cids": list(self.incomplete_cids),
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "original_plan_cid": self.original_plan_cid,
        }

    @property
    def pruning_result_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["pruning_result_cid"] = self.pruning_result_cid
        return value

    def receipt_for(self, stage: SuccessorStage | str) -> SuccessorDecisionReceipt | None:
        stage_value = stage.value if isinstance(stage, SuccessorStage) else str(stage)
        for receipt in self.receipts:
            if receipt.stage == stage_value:
                return receipt
        return None

    def as_plan(self) -> StaticSuccessorPlan:
        """Project retained successors back into a successor plan."""

        return StaticSuccessorPlan(
            snapshot_cid=self.snapshot_cid,
            graph_cid=self.graph_cid,
            language=self.language,
            query_families=(
                "next_call",
                "next_event",
            ),
            environment_binding_cid="pruned",
            sealed_binding_cid="pruned",
            subject_node_cid=None,
            candidates=self.retained,
            frontier=self.frontier,
            receipts=tuple(self.generation_receipts) + tuple(self.receipts),
            complete=self.complete,
            unavailable_dimensions=self.unavailable_dimensions,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SymbolicPruningResult":
        payload = dict(data)
        claimed = payload.pop("pruning_result_cid", None)
        payload.pop("schema", None)
        payload["retained"] = tuple(
            SuccessorCandidate.from_dict(item) if isinstance(item, Mapping) else item
            for item in payload.get("retained", ())
        )
        payload["removed"] = tuple(
            SuccessorCandidate.from_dict(item) if isinstance(item, Mapping) else item
            for item in payload.get("removed", ())
        )
        frontier = payload.get("frontier")
        if isinstance(frontier, Mapping):
            payload["frontier"] = UnresolvedDynamicFrontier.from_dict(frontier)
        payload["receipts"] = tuple(
            SuccessorDecisionReceipt.from_dict(item)
            if isinstance(item, Mapping)
            else item
            for item in payload.get("receipts", ())
        )
        payload["generation_receipts"] = tuple(
            SuccessorDecisionReceipt.from_dict(item)
            if isinstance(item, Mapping)
            else item
            for item in payload.get("generation_receipts", ())
        )
        result = cls(**payload)
        if claimed is not None and claimed != result.pruning_result_cid:
            raise SymbolicPruningError("symbolic pruning result identity does not match")
        return result


def _apply_stage(
    *,
    stage: SuccessorStage,
    evidence: Sequence[SymbolicPruningEvidence],
    retained: list[SuccessorCandidate],
    removed: list[SuccessorCandidate],
    frontier: UnresolvedDynamicFrontier,
    solver_status: str | None = None,
) -> tuple[
    list[SuccessorCandidate],
    list[SuccessorCandidate],
    UnresolvedDynamicFrontier,
    SuccessorDecisionReceipt,
]:
    incoming_cids = tuple(item.candidate_cid for item in retained)
    if not evidence and solver_status is None:
        return retained, removed, frontier, record_stage(
            stage,
            SuccessorDisposition.SKIPPED,
            SuccessorReasonCode.NO_EVIDENCE,
            candidate_cids_in=incoming_cids,
            candidate_cids_out=incoming_cids,
        )

    if solver_status is not None and stage is SuccessorStage.SOLVER:
        status = _enum(solver_status, SolverStatus, "solver_status")
        if status in {
            SolverStatus.UNKNOWN.value,
            SolverStatus.TIMEOUT.value,
            SolverStatus.UNAVAILABLE.value,
        }:
            reason = (
                SuccessorReasonCode.SOLVER_TIMEOUT
                if status == SolverStatus.TIMEOUT.value
                else (
                    SuccessorReasonCode.CAPABILITY_UNAVAILABLE
                    if status == SolverStatus.UNAVAILABLE.value
                    else SuccessorReasonCode.SOLVER_UNKNOWN
                )
            )
            disposition = (
                SuccessorDisposition.UNAVAILABLE
                if status == SolverStatus.UNAVAILABLE.value
                else SuccessorDisposition.SELECTED
            )
            frontier = frontier.widen(
                reasons=(DynamicFrontierReason.SOLVER_UNKNOWN.value,),
                unavailable_dimensions=(
                    "solver_timeout"
                    if status == SolverStatus.TIMEOUT.value
                    else "solver_unknown"
                ),
            )
            return retained, removed, frontier, record_stage(
                stage,
                disposition,
                reason,
                candidate_cids_in=incoming_cids,
                candidate_cids_out=incoming_cids,
                unknown_cids=incoming_cids,
                unavailable_dimensions=frontier.unavailable_dimensions,
                widened=True,
                diagnostic=f"solver status {status} cannot prune successors",
            )

    removed_here: list[SuccessorCandidate] = []
    unknown_here: list[str] = []
    evidence_cids: list[str] = []
    non_authoritative = False
    widened = False
    unavailable = False
    remaining: list[SuccessorCandidate] = []
    still_retained = list(retained)

    for fact in evidence:
        if fact.stage != stage.value:
            continue
        evidence_cids.append(fact.pruning_evidence_cid)
        if fact.evidence_cid:
            evidence_cids.append(fact.evidence_cid)
        if fact.verdict == PruningVerdict.UNAVAILABLE.value:
            unavailable = True
            widened = True
            unknown_here.extend(incoming_cids)
            continue
        if fact.widens:
            widened = True
            unknown_here.extend(
                item.candidate_cid
                for item in still_retained
                if _matches_candidate(fact, item)
            )
            continue
        if fact.verdict == PruningVerdict.IMPOSSIBLE.value and not fact.authoritative:
            non_authoritative = True
            continue
        if not fact.can_remove:
            continue
        kept: list[SuccessorCandidate] = []
        for candidate in still_retained:
            if _matches_candidate(fact, candidate):
                removed_here.append(candidate)
            else:
                kept.append(candidate)
        still_retained = kept

    remaining = still_retained
    removed_cids = tuple(item.candidate_cid for item in removed_here)
    if removed_here:
        removed = [*removed, *removed_here]
        removed_targets = tuple(item.target_node_cid for item in removed_here)
        frontier = frontier.without_nodes(removed_targets)

    if widened:
        frontier = frontier.widen(
            reasons=(_STAGE_FRONTIER_REASON.get(stage.value, DynamicFrontierReason.INCOMPLETE_ANALYSIS.value),),
            unavailable_dimensions=(
                ("solver_unknown",)
                if stage is SuccessorStage.SOLVER
                else ("incomplete_analysis",)
            ),
            unresolved_node_cids=tuple(
                item.target_node_cid
                for item in remaining
                if item.unknown or item.candidate_cid in unknown_here
            ),
        )

    outgoing_cids = tuple(item.candidate_cid for item in remaining)
    if non_authoritative and not removed_here:
        receipt = record_stage(
            stage,
            SuccessorDisposition.REJECTED,
            SuccessorReasonCode.NON_AUTHORITATIVE,
            candidate_cids_in=incoming_cids,
            candidate_cids_out=outgoing_cids,
            evidence_cids=evidence_cids,
            diagnostic="non-authoritative evidence cannot prune successors",
        )
        return remaining, removed, frontier, receipt

    if unavailable and not removed_here:
        receipt = record_stage(
            stage,
            SuccessorDisposition.UNAVAILABLE,
            _STAGE_UNKNOWN_REASON.get(stage.value, SuccessorReasonCode.CAPABILITY_UNAVAILABLE),
            candidate_cids_in=incoming_cids,
            candidate_cids_out=outgoing_cids,
            unknown_cids=_sorted_unique(unknown_here, "unknown_here"),
            evidence_cids=evidence_cids,
            unavailable_dimensions=frontier.unavailable_dimensions,
            widened=True,
        )
        return remaining, removed, frontier, receipt

    if not evidence_cids and solver_status is None:
        receipt = record_stage(
            stage,
            SuccessorDisposition.SKIPPED,
            SuccessorReasonCode.NO_EVIDENCE,
            candidate_cids_in=incoming_cids,
            candidate_cids_out=outgoing_cids,
        )
        return remaining, removed, frontier, receipt

    reason = (
        _STAGE_IMPOSSIBLE_REASON.get(stage.value, SuccessorReasonCode.IMPOSSIBLE)
        if removed_here
        else (
            _STAGE_UNKNOWN_REASON.get(stage.value, SuccessorReasonCode.UNKNOWN_WIDENED)
            if widened
            else SuccessorReasonCode.NO_EVIDENCE
        )
    )
    if not removed_here and not widened:
        # Authoritative "possible" evidence confirms without shrinking.
        reason = SuccessorReasonCode.COMPLETE if remaining else SuccessorReasonCode.NO_EVIDENCE
    receipt = record_stage(
        stage,
        SuccessorDisposition.SELECTED,
        reason,
        candidate_cids_in=incoming_cids,
        candidate_cids_out=outgoing_cids,
        removed_cids=removed_cids,
        unknown_cids=_sorted_unique(unknown_here, "unknown_here"),
        evidence_cids=evidence_cids,
        unavailable_dimensions=frontier.unavailable_dimensions if widened else (),
        widened=widened,
    )
    return remaining, removed, frontier, receipt


def prune_program_successors(
    plan: StaticSuccessorPlan | Mapping[str, Any],
    *,
    type_evidence: Sequence[Any] | None = None,
    effect_evidence: Sequence[Any] | None = None,
    path_evidence: Sequence[Any] | None = None,
    contract_evidence: Sequence[Any] | None = None,
    capability_evidence: Sequence[Any] | None = None,
    abstract_interpretation: Sequence[Any] | None = None,
    solver_evidence: Sequence[Any] | None = None,
    solver_status: str | None = None,
    expected_snapshot_cid: str | None = None,
    expected_graph_cid: str | None = None,
    expected_plan_cid: str | None = None,
) -> SymbolicPruningResult:
    """Prune a successor plan using only authoritative symbolic reasons.

    Unknown widens.  Timeout, incomplete, and unavailable evidence cannot
    remove static possibilities or explicit unknown.  Every pruning stage is
    recorded.
    """

    bound = _coerce_plan(plan)
    receipts: list[SuccessorDecisionReceipt] = []
    incoming = list(bound.candidates)
    incoming_cids = tuple(item.candidate_cid for item in incoming)

    stale_reason: SuccessorReasonCode | None = None
    if expected_plan_cid is not None and expected_plan_cid != bound.plan_cid:
        stale_reason = SuccessorReasonCode.STALE_GRAPH
    elif expected_snapshot_cid is not None and expected_snapshot_cid != bound.snapshot_cid:
        stale_reason = SuccessorReasonCode.STALE_SNAPSHOT
    elif expected_graph_cid is not None and expected_graph_cid != bound.graph_cid:
        stale_reason = SuccessorReasonCode.STALE_GRAPH
    elif any(
        item.stage == SuccessorStage.FRESHNESS.value
        and item.disposition == SuccessorDisposition.REJECTED.value
        for item in bound.receipts
    ):
        stale_reason = SuccessorReasonCode.STALE_GRAPH

    if stale_reason is not None:
        receipts.append(
            record_stage(
                SuccessorStage.FRESHNESS,
                SuccessorDisposition.REJECTED,
                stale_reason,
                candidate_cids_in=incoming_cids,
                unavailable_dimensions=("stale_graph",),
                diagnostic="pruning refuses a stale successor plan",
            )
        )
        filled = _fill_pruning_receipts(receipts, blocked=True)
        frontier = bound.frontier or UnresolvedDynamicFrontier(language=bound.language)
        frontier = frontier.widen(
            reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
            unavailable_dimensions=("stale_graph",),
        )
        return SymbolicPruningResult(
            plan_cid=bound.plan_cid,
            snapshot_cid=bound.snapshot_cid,
            graph_cid=bound.graph_cid,
            language=bound.language,
            retained=(),
            removed=(),
            frontier=frontier,
            receipts=filled,
            generation_receipts=bound.receipts,
            complete=False,
            impossible_cids=(),
            incomplete_cids=incoming_cids,
            unavailable_dimensions=("stale_graph",),
            original_plan_cid=bound.plan_cid,
        )

    receipts.append(
        record_stage(
            SuccessorStage.FRESHNESS,
            SuccessorDisposition.SELECTED,
            SuccessorReasonCode.GRAPH_FRESH,
            candidate_cids_in=incoming_cids,
            candidate_cids_out=incoming_cids,
            evidence_cids=(bound.plan_cid, bound.snapshot_cid),
        )
    )

    type_facts = _coerce_evidence(type_evidence, SuccessorStage.TYPE.value)
    effect_facts = _coerce_evidence(effect_evidence, SuccessorStage.EFFECT.value)
    path_facts = _coerce_evidence(path_evidence, SuccessorStage.PATH_CONDITION.value)
    contract_facts = _coerce_evidence(contract_evidence, SuccessorStage.CONTRACT.value)
    capability_facts = _coerce_evidence(
        capability_evidence, SuccessorStage.CAPABILITY.value
    )
    abstract_facts = _coerce_evidence(
        abstract_interpretation, SuccessorStage.ABSTRACT_INTERPRETATION.value
    )
    solver_facts = _coerce_evidence(solver_evidence, SuccessorStage.SOLVER.value)

    retained = incoming
    removed: list[SuccessorCandidate] = []
    frontier = bound.frontier or UnresolvedDynamicFrontier(language=bound.language)

    stage_inputs: tuple[tuple[SuccessorStage, tuple[SymbolicPruningEvidence, ...]], ...] = (
        (SuccessorStage.TYPE, type_facts),
        (SuccessorStage.EFFECT, effect_facts),
        (SuccessorStage.PATH_CONDITION, path_facts),
        (SuccessorStage.CONTRACT, contract_facts),
        (SuccessorStage.CAPABILITY, capability_facts),
        (SuccessorStage.ABSTRACT_INTERPRETATION, abstract_facts),
        (SuccessorStage.SOLVER, solver_facts),
    )
    for stage, facts in stage_inputs:
        retained, removed, frontier, receipt = _apply_stage(
            stage=stage,
            evidence=facts,
            retained=retained,
            removed=removed,
            frontier=frontier,
            solver_status=solver_status if stage is SuccessorStage.SOLVER else None,
        )
        receipts.append(receipt)

    unknown_retained = [
        item for item in retained if item.unknown or not item.complete
    ]
    if unknown_retained or not frontier.empty:
        receipts.append(
            record_stage(
                SuccessorStage.RESIDUAL_ESCALATION,
                SuccessorDisposition.ESCALATED,
                SuccessorReasonCode.RESIDUAL_UNKNOWN,
                candidate_cids_in=tuple(item.candidate_cid for item in retained),
                candidate_cids_out=tuple(item.candidate_cid for item in retained),
                unknown_cids=tuple(item.candidate_cid for item in unknown_retained),
                unavailable_dimensions=frontier.unavailable_dimensions,
                diagnostic=(
                    "deterministic routes exhausted; residual is proposal-only "
                    "and cannot self-approve"
                ),
            )
        )
    else:
        receipts.append(
            record_stage(
                SuccessorStage.RESIDUAL_ESCALATION,
                SuccessorDisposition.SKIPPED,
                SuccessorReasonCode.NO_RESIDUAL,
                candidate_cids_in=tuple(item.candidate_cid for item in retained),
                candidate_cids_out=tuple(item.candidate_cid for item in retained),
            )
        )

    impossible_cids = tuple(item.candidate_cid for item in removed)
    incomplete_cids = tuple(item.candidate_cid for item in unknown_retained)
    unavailable = _sorted_unique(
        (
            *bound.unavailable_dimensions,
            *frontier.unavailable_dimensions,
            *(dim for item in retained for dim in item.unavailable_dimensions),
        ),
        "unavailable_dimensions",
    )
    complete = (
        frontier.empty
        and not incomplete_cids
        and not unavailable
        and all(item.complete and not item.unknown for item in retained)
    )
    if complete:
        receipts.append(
            record_stage(
                SuccessorStage.COMPLETENESS,
                SuccessorDisposition.SELECTED,
                SuccessorReasonCode.COMPLETE,
                candidate_cids_out=tuple(item.candidate_cid for item in retained),
                removed_cids=impossible_cids,
            )
        )
        unavailable = ()
    else:
        if not unavailable:
            unavailable = ("incomplete_analysis",)
        receipts.append(
            record_stage(
                SuccessorStage.COMPLETENESS,
                SuccessorDisposition.SELECTED,
                SuccessorReasonCode.INCOMPLETE,
                candidate_cids_out=tuple(item.candidate_cid for item in retained),
                unknown_cids=incomplete_cids,
                removed_cids=impossible_cids,
                unavailable_dimensions=unavailable,
                widened=frontier.widened,
            )
        )

    filled = _fill_pruning_receipts(receipts, blocked=False)
    return SymbolicPruningResult(
        plan_cid=bound.plan_cid,
        snapshot_cid=bound.snapshot_cid,
        graph_cid=bound.graph_cid,
        language=bound.language,
        retained=tuple(retained),
        removed=tuple(removed),
        frontier=frontier,
        receipts=filled,
        generation_receipts=bound.receipts,
        complete=complete,
        impossible_cids=impossible_cids,
        incomplete_cids=incomplete_cids,
        unavailable_dimensions=unavailable,
        original_plan_cid=bound.plan_cid,
    )


@dataclass(frozen=True, slots=True)
class SymbolicSuccessorPruner:
    """Operational pruner for authoritative symbolic successor reduction."""

    expected_snapshot_cid: str | None = None
    expected_graph_cid: str | None = None

    INTERFACE: ClassVar[str] = SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE
    EVIDENCE: ClassVar[str] = SAWM_STATIC_SUCCESSORS_EVIDENCE

    def prune(
        self,
        plan: StaticSuccessorPlan | Mapping[str, Any],
        **kwargs: Any,
    ) -> SymbolicPruningResult:
        kwargs.setdefault("expected_snapshot_cid", self.expected_snapshot_cid)
        kwargs.setdefault("expected_graph_cid", self.expected_graph_cid)
        return prune_program_successors(plan, **kwargs)


__all__ = [
    "ALLOWED_AUTHORITIES",
    "PRUNING_STAGE_ORDER",
    "SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE",
    "PruningVerdict",
    "SolverStatus",
    "SymbolicPruningError",
    "SymbolicPruningEvidence",
    "SymbolicPruningResult",
    "SymbolicSuccessorPruner",
    "prune_program_successors",
]
