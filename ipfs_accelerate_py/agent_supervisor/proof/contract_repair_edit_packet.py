"""Fail-closed hand-off packets for admitted contract-repair targets.

``RepairTargetDecision`` is the only object which grants mutation authority,
but it is deliberately not a prompt.  This module turns a *currently valid*
``AdmissionResult`` into a compact provider hand-off without copying source,
AST, solver, or proof bodies into that hand-off.  A packet records the exact
decision identity and writes exactly the paths in that decision; candidates
which were not selected never occur in the packet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    DecisionDisposition,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    RepairTargetDecision,
    SourceSpan,
)
from ..analysis.sender_receiver_contracts import (
    ClauseDisposition,
    ProgramContractComparison,
)
from ..planning.repair_target_admission import (
    AdmissionResult,
    RepairTargetAdmissionError,
    RepairTargetDecisionValidator,
    TargetRepositoryAuthority,
)
from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)


CONTRACT_REPAIR_EDIT_PACKET_INTERFACE: Final[str] = "ContractRepairEditPacket@2"
CONTRACT_REPAIR_EDIT_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-edit-packet@2"
)
CONTRACT_REPAIR_EDIT_PACKET_VERSION: Final[int] = 2
MAX_PACKET_BYTES: Final[int] = 262_144
MAX_CLAUSES: Final[int] = 64
MAX_REFERENCES: Final[int] = 256
MAX_HANDLES: Final[int] = 64
MAX_COMMANDS: Final[int] = 32
MAX_OBLIGATIONS: Final[int] = 128


class ContractRepairEditPacketError(ContractValidationError):
    """A packet would weaken the admitted repair authority boundary."""


class ContractRepairEditPacketReason(str, Enum):
    """Stable, machine-readable admission failures."""

    NOT_CURRENT = "not_current"
    NOT_ADMITTED = "not_admitted"
    ROOT_DRIFT = "root_drift"
    DECISION_DRIFT = "decision_drift"
    TARGET_DRIFT = "target_drift"
    SCOPE_MISMATCH = "scope_mismatch"
    FORBIDDEN_BODY = "forbidden_body"
    MALFORMED = "malformed"


def _text(value: Any, name: str, *, required: bool = True, limit: int = 4096) -> str:
    if not isinstance(value, str) or value != value.strip() or "\x00" in value:
        raise ContractRepairEditPacketError(f"{name} must be a trimmed string")
    if required and not value:
        raise ContractRepairEditPacketError(f"{name} is required")
    if len(value.encode("utf-8")) > limit:
        raise ContractRepairEditPacketError(f"{name} exceeds its byte bound")
    return value


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name)
    if any(char.isspace() for char in result):
        raise ContractRepairEditPacketError(f"{name} must be an opaque identifier")
    return result


def _path(value: Any, name: str) -> str:
    result = _text(value, name, limit=1024)
    if "\\" in result:
        raise ContractRepairEditPacketError(f"{name} must use POSIX separators")
    parsed = PurePosixPath(result)
    if (
        parsed.is_absolute()
        or result.startswith("./")
        or ".." in parsed.parts
        or parsed.as_posix() in {"", "."}
        or any(char in result for char in "*?[]{}")
    ):
        raise ContractRepairEditPacketError(f"{name} must be an exact repository-relative path")
    return parsed.as_posix()


def _paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ContractRepairEditPacketError(f"{name} must be a sequence of paths")
    result = tuple(sorted({_path(value, name) for value in values}))
    if required and not result:
        raise ContractRepairEditPacketError(f"{name} must not be empty")
    if len(result) > MAX_REFERENCES:
        raise ContractRepairEditPacketError(f"{name} exceeds its item bound")
    return result


def _ids(values: Any, name: str, *, required: bool = False, maximum: int = MAX_REFERENCES) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairEditPacketError(f"{name} must be a sequence of identifiers")
    result = tuple(sorted({_identifier(value, name) for value in values}))
    if required and not result:
        raise ContractRepairEditPacketError(f"{name} must not be empty")
    if len(result) > maximum:
        raise ContractRepairEditPacketError(f"{name} exceeds its item bound")
    return result


def _refs(values: Any, name: str, *, required: bool = False) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairEditPacketError(f"{name} must be EvidenceReference values")
    if not all(isinstance(value, EvidenceReference) for value in values):
        raise ContractRepairEditPacketError(f"{name} must contain EvidenceReference values")
    result = tuple(sorted(set(values), key=lambda value: value.content_id))
    if required and not result:
        raise ContractRepairEditPacketError(f"{name} must not be empty")
    if len(result) > MAX_REFERENCES:
        raise ContractRepairEditPacketError(f"{name} exceeds its item bound")
    return result


@dataclass(frozen=True)
class ContractClause:
    """A bounded, body-free projection of one sender/receiver clause."""

    aspect: str
    disposition: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "aspect", _identifier(self.aspect, "clause.aspect"))
        try:
            disposition = ClauseDisposition(str(getattr(self.disposition, "value", self.disposition)))
        except ValueError as exc:
            raise ContractRepairEditPacketError("clause.disposition is unknown") from exc
        object.__setattr__(self, "disposition", disposition.value)
        object.__setattr__(self, "reason", _text(self.reason, "clause.reason", limit=1024))

    def to_dict(self) -> dict[str, str]:
        return {"aspect": self.aspect, "disposition": self.disposition, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractClause":
        if not isinstance(payload, Mapping) or set(payload) != {"aspect", "disposition", "reason"}:
            raise ContractRepairEditPacketError("contract clause must contain exactly aspect, disposition, and reason")
        return cls(payload["aspect"], payload["disposition"], payload["reason"])


@dataclass(frozen=True)
class ExpansionHandle:
    """A bounded pointer to more evidence; it never contains the evidence."""

    handle_id: str
    kind: str
    reference_id: str
    permitted_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "handle_id", _identifier(self.handle_id, "handle_id"))
        object.__setattr__(self, "kind", _identifier(self.kind, "handle.kind"))
        if self.kind.casefold().replace("-", "_") in {"source", "source_body", "proof_body", "ast_body"}:
            raise ContractRepairEditPacketError("expansion handles may not name embedded bodies")
        object.__setattr__(self, "reference_id", _identifier(self.reference_id, "handle.reference_id"))
        object.__setattr__(self, "permitted_paths", _paths(self.permitted_paths, "handle.permitted_paths", required=False))

    def to_dict(self) -> dict[str, Any]:
        return {"handle_id": self.handle_id, "kind": self.kind, "reference_id": self.reference_id,
                "permitted_paths": list(self.permitted_paths), "body_embedded": False}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpansionHandle":
        allowed = {"handle_id", "kind", "reference_id", "permitted_paths", "body_embedded"}
        if not isinstance(payload, Mapping) or set(payload).difference(allowed):
            raise ContractRepairEditPacketError("expansion handle contains unsupported fields")
        if payload.get("body_embedded", False) is not False:
            raise ContractRepairEditPacketError("expansion handle cannot embed a body")
        return cls(payload.get("handle_id"), payload.get("kind"), payload.get("reference_id"),
                   tuple(payload.get("permitted_paths", ())))


def _contract_id(contract: Any, name: str) -> str:
    value = getattr(contract, "content_id", "")
    return _identifier(value, name)


@dataclass(frozen=True)
class ContractRepairEditPacket(CanonicalContract):
    """A content-addressed provider input bound to one validated decision."""

    SCHEMA: ClassVar[str] = CONTRACT_REPAIR_EDIT_PACKET_SCHEMA

    roots: AuthorityRoots
    decision_id: str
    candidate_set_id: str
    trace_id: str
    strategy: RepairStrategy
    target_span: SourceSpan
    read_paths: tuple[str, ...]
    write_paths: tuple[str, ...]
    sender_expected_contract_id: str
    receiver_expected_contract_id: str
    receiver_observed_contract_id: str
    clauses: tuple[ContractClause, ...]
    unsupported_clause_ids: tuple[str, ...]
    selection_rationale_refs: tuple[EvidenceReference, ...]
    proof_refs: tuple[EvidenceReference, ...]
    counterexample_refs: tuple[EvidenceReference, ...]
    index_refs: tuple[str, ...]
    post_edit_obligation_ids: tuple[str, ...]
    validation_commands: tuple[str, ...]
    reproof_commands: tuple[str, ...]
    invalidation_refs: tuple[str, ...]
    expansion_handles: tuple[ExpansionHandle, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(self.target_span, SourceSpan):
            raise ContractRepairEditPacketError("roots and target_span must be typed contracts")
        for name in ("decision_id", "candidate_set_id", "trace_id", "sender_expected_contract_id", "receiver_expected_contract_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "receiver_observed_contract_id", _text(self.receiver_observed_contract_id, "receiver_observed_contract_id", required=False))
        try:
            strategy = RepairStrategy(self.strategy)
        except ValueError as exc:
            raise ContractRepairEditPacketError("strategy is unknown") from exc
        if strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}:
            raise ContractRepairEditPacketError("a packet cannot materialize a reject or ambiguous strategy")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "read_paths", _paths(self.read_paths, "read_paths"))
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        if self.target_span.path not in self.read_paths or self.target_span.path not in self.write_paths:
            raise ContractRepairEditPacketError("target span must remain inside exact read and write authority")
        if not isinstance(self.clauses, Sequence) or not self.clauses or not all(isinstance(item, ContractClause) for item in self.clauses):
            raise ContractRepairEditPacketError("clauses must be a non-empty ContractClause sequence")
        clauses = tuple(sorted(self.clauses, key=lambda item: item.aspect))
        if len(clauses) > MAX_CLAUSES or len({item.aspect for item in clauses}) != len(clauses):
            raise ContractRepairEditPacketError("clauses must have unique bounded aspects")
        object.__setattr__(self, "clauses", clauses)
        unsupported = _ids(self.unsupported_clause_ids, "unsupported_clause_ids", maximum=MAX_CLAUSES)
        expected_unsupported = tuple(sorted(item.aspect for item in clauses if item.disposition == ClauseDisposition.UNSUPPORTED.value))
        if unsupported != expected_unsupported:
            raise ContractRepairEditPacketError("unsupported limits must exactly name unsupported clauses")
        object.__setattr__(self, "unsupported_clause_ids", unsupported)
        object.__setattr__(self, "selection_rationale_refs", _refs(self.selection_rationale_refs, "selection_rationale_refs", required=True))
        object.__setattr__(self, "proof_refs", _refs(self.proof_refs, "proof_refs", required=True))
        object.__setattr__(self, "counterexample_refs", _refs(self.counterexample_refs, "counterexample_refs"))
        index_refs = _ids(self.index_refs, "index_refs", required=True)
        if self.roots.index_id not in index_refs:
            raise ContractRepairEditPacketError("index_refs must bind the decision index root")
        object.__setattr__(self, "index_refs", index_refs)
        object.__setattr__(self, "post_edit_obligation_ids", _ids(self.post_edit_obligation_ids, "post_edit_obligation_ids", required=True, maximum=MAX_OBLIGATIONS))
        object.__setattr__(self, "validation_commands", _commands(self.validation_commands, "validation_commands"))
        object.__setattr__(self, "reproof_commands", _commands(self.reproof_commands, "reproof_commands"))
        object.__setattr__(self, "invalidation_refs", _ids(self.invalidation_refs, "invalidation_refs", required=True))
        if not isinstance(self.expansion_handles, Sequence) or not all(isinstance(item, ExpansionHandle) for item in self.expansion_handles):
            raise ContractRepairEditPacketError("expansion_handles must contain ExpansionHandle values")
        handles = tuple(sorted(self.expansion_handles, key=lambda item: item.handle_id))
        if len(handles) > MAX_HANDLES or len({item.handle_id for item in handles}) != len(handles):
            raise ContractRepairEditPacketError("expansion_handles must be unique and bounded")
        # Selection rationale may describe the complete ranking, including
        # rejected candidates.  It is audit-only and must never be expanded
        # into provider context.  Only selected-proof/counterexample/index
        # references may back an expansion handle.
        reference_ids = set(self.index_refs)
        for refs in (self.proof_refs, self.counterexample_refs):
            reference_ids.update(item.content_id for item in refs)
            reference_ids.update(item.artifact_id for item in refs)
        for handle in handles:
            if handle.reference_id not in reference_ids:
                raise ContractRepairEditPacketError("an expansion handle must point to packet-bound evidence")
            if not set(handle.permitted_paths).issubset(self.read_paths):
                raise ContractRepairEditPacketError("an expansion handle cannot expand read scope")
        object.__setattr__(self, "expansion_handles", handles)
        if len(canonical_json_bytes(self._payload())) > MAX_PACKET_BYTES:
            raise ContractRepairEditPacketError("packet exceeds its serialized byte bound")

    @property
    def interface(self) -> str:
        return CONTRACT_REPAIR_EDIT_PACKET_INTERFACE

    @property
    def packet_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_REPAIR_EDIT_PACKET_VERSION,
            "interface": self.interface,
            "roots": self.roots.to_dict(),
            "decision_id": self.decision_id,
            "candidate_set_id": self.candidate_set_id,
            "trace_id": self.trace_id,
            "strategy": self.strategy.value,
            "target_span": self.target_span.to_dict(),
            "read_paths": list(self.read_paths), "write_paths": list(self.write_paths),
            "sender_expected_contract_id": self.sender_expected_contract_id,
            "receiver_expected_contract_id": self.receiver_expected_contract_id,
            "receiver_observed_contract_id": self.receiver_observed_contract_id,
            "clauses": [item.to_dict() for item in self.clauses],
            "unsupported_clause_ids": list(self.unsupported_clause_ids),
            "selection_rationale_refs": [item.to_dict() for item in self.selection_rationale_refs],
            "proof_refs": [item.to_dict() for item in self.proof_refs],
            "counterexample_refs": [item.to_dict() for item in self.counterexample_refs],
            "index_refs": list(self.index_refs),
            "post_edit_obligation_ids": list(self.post_edit_obligation_ids),
            "validation_commands": list(self.validation_commands),
            "reproof_commands": list(self.reproof_commands),
            "invalidation_refs": list(self.invalidation_refs),
            "expansion_handles": [item.to_dict() for item in self.expansion_handles],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractRepairEditPacket":
        if not isinstance(payload, Mapping):
            raise ContractRepairEditPacketError("packet payload must be an object")
        fields = {
            "schema", "contract_version", "interface", "content_id", "roots", "decision_id", "candidate_set_id", "trace_id",
            "strategy", "target_span", "read_paths", "write_paths", "sender_expected_contract_id", "receiver_expected_contract_id",
            "receiver_observed_contract_id", "clauses", "unsupported_clause_ids", "selection_rationale_refs", "proof_refs",
            "counterexample_refs", "index_refs", "post_edit_obligation_ids", "validation_commands", "reproof_commands",
            "invalidation_refs", "expansion_handles",
        }
        if set(payload).difference(fields) or payload.get("schema") not in (None, "", cls.SCHEMA):
            raise ContractRepairEditPacketError("packet has unsupported fields or schema")
        if payload.get("contract_version") not in (None, CONTRACT_REPAIR_EDIT_PACKET_VERSION):
            raise ContractRepairEditPacketError("packet has an unsupported contract version")
        if payload.get("interface") not in (None, "", CONTRACT_REPAIR_EDIT_PACKET_INTERFACE):
            raise ContractRepairEditPacketError("packet has an unsupported interface")
        try:
            packet = cls(
                roots=AuthorityRoots.from_dict(payload["roots"]), decision_id=payload["decision_id"],
                candidate_set_id=payload["candidate_set_id"], trace_id=payload["trace_id"], strategy=payload["strategy"],
                target_span=SourceSpan.from_dict(payload["target_span"]), read_paths=tuple(payload["read_paths"]),
                write_paths=tuple(payload["write_paths"]), sender_expected_contract_id=payload["sender_expected_contract_id"],
                receiver_expected_contract_id=payload["receiver_expected_contract_id"],
                receiver_observed_contract_id=payload.get("receiver_observed_contract_id", ""),
                clauses=tuple(ContractClause.from_dict(item) for item in payload["clauses"]),
                unsupported_clause_ids=tuple(payload["unsupported_clause_ids"]),
                selection_rationale_refs=tuple(EvidenceReference.from_dict(item) for item in payload["selection_rationale_refs"]),
                proof_refs=tuple(EvidenceReference.from_dict(item) for item in payload["proof_refs"]),
                counterexample_refs=tuple(EvidenceReference.from_dict(item) for item in payload["counterexample_refs"]),
                index_refs=tuple(payload["index_refs"]), post_edit_obligation_ids=tuple(payload["post_edit_obligation_ids"]),
                validation_commands=tuple(payload["validation_commands"]), reproof_commands=tuple(payload["reproof_commands"]),
                invalidation_refs=tuple(payload["invalidation_refs"]),
                expansion_handles=tuple(ExpansionHandle.from_dict(item) for item in payload.get("expansion_handles", ())),
            )
        except ContractRepairEditPacketError:
            raise
        except (KeyError, TypeError, ContractValidationError) as exc:
            raise ContractRepairEditPacketError("packet payload is malformed") from exc
        claimed = payload.get("content_id")
        if claimed not in (None, "") and claimed != packet.content_id:
            raise ContractRepairEditPacketError("packet content identity is forged")
        return packet


def _commands(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairEditPacketError(f"{name} must be a sequence of commands")
    result = tuple(sorted({_text(value, name, limit=4096) for value in values}))
    if not result:
        raise ContractRepairEditPacketError(f"{name} must not be empty")
    if len(result) > MAX_COMMANDS or any("\n" in item or "\r" in item for item in result):
        raise ContractRepairEditPacketError(f"{name} must contain bounded one-line commands")
    return result


def materialize_contract_repair_edit_packet(
    admission: AdmissionResult,
    trace: BrokenContractTrace,
    comparison: ProgramContractComparison,
    *,
    roots: AuthorityRoots,
    candidates: Sequence[RepairCandidate],
    rerank_receipt: Any,
    authorities: Sequence[TargetRepositoryAuthority],
    now: int,
    post_edit_obligation_ids: Sequence[str],
    validation_commands: Sequence[str],
    reproof_commands: Sequence[str],
    counterexample_refs: Sequence[EvidenceReference] = (),
    index_refs: Sequence[str] = (),
    expansion_handles: Sequence[ExpansionHandle] = (),
    validator: RepairTargetDecisionValidator | None = None,
) -> ContractRepairEditPacket:
    """Materialize one packet after replaying the complete admission boundary.

    ``AdmissionResult`` is required rather than a bare decision because only it
    carries the expiry/audit data needed to distinguish a current decision from
    a stale replay.  ``RepairTargetDecisionValidator`` additionally catches
    proof, candidate-set, target, and authority drift immediately before the
    packet becomes provider-visible.
    """

    if not isinstance(admission, AdmissionResult):
        raise ContractRepairEditPacketError("a current AdmissionResult is required")
    if not isinstance(trace, BrokenContractTrace) or not isinstance(comparison, ProgramContractComparison):
        raise ContractRepairEditPacketError("trace and comparison must be typed contracts")
    if not isinstance(roots, AuthorityRoots):
        raise ContractRepairEditPacketError("roots must be AuthorityRoots")
    checker = validator or RepairTargetDecisionValidator()
    try:
        checker.require_valid(admission, roots=roots, candidates=candidates,
                              rerank_receipt=rerank_receipt, authorities=authorities, now=now)
    except RepairTargetAdmissionError as exc:
        raise ContractRepairEditPacketError("target decision is not current and admitted") from exc
    decision = admission.decision
    if decision.disposition is not DecisionDisposition.ADMITTED or decision.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}:
        raise ContractRepairEditPacketError("only a current admitted non-abstaining decision may materialize")
    selected = next((item for item in decision.candidates if item.content_id == decision.selected_candidate_id), None)
    if selected is None or selected.trace_id != trace.content_id or selected.target_span.path not in decision.write_paths:
        raise ContractRepairEditPacketError("decision target does not bind this broken trace and write authority")
    if decision.roots != roots or trace.roots != roots or comparison.sender.call_requirement.roots != roots:
        raise ContractRepairEditPacketError("trace, comparison, and decision must bind current roots")
    if decision.permitted_write_paths != tuple(sorted(decision.permitted_write_paths)) or not decision.permitted_write_paths:
        raise ContractRepairEditPacketError("decision has no exact write authority")
    clauses = tuple(
        ContractClause(item.aspect.value, item.disposition.value, item.reason)
        for item in comparison.clauses
    )
    allowed_refs = {
        *decision.evidence_refs, *decision.proof_refs, *selected.evidence_refs, *selected.proof_refs,
        *trace.evidence_refs, *trace.proof_refs, *comparison.call_requirement.evidence_refs,
        *comparison.call_requirement.proof_refs,
    }
    requested_counterexamples = _refs(counterexample_refs, "counterexample_refs")
    if not set(requested_counterexamples).issubset(allowed_refs):
        raise ContractRepairEditPacketError("counterexample refs must already bind the selected decision")
    requested_indexes = _ids(index_refs, "index_refs")
    packet_index_refs = tuple(sorted({roots.index_id, *requested_indexes}))
    return ContractRepairEditPacket(
        roots=roots, decision_id=decision.content_id, candidate_set_id=decision.candidate_set_id,
        trace_id=trace.content_id, strategy=decision.strategy, target_span=selected.target_span,
        read_paths=decision.permitted_read_paths, write_paths=decision.permitted_write_paths,
        sender_expected_contract_id=_contract_id(comparison.sender.contract, "sender_expected_contract_id"),
        receiver_expected_contract_id=_contract_id(comparison.receiver.contract, "receiver_expected_contract_id"),
        receiver_observed_contract_id=(
            _contract_id(comparison.receiver.observed, "receiver_observed_contract_id")
            if comparison.receiver.observed is not None else ""
        ),
        clauses=clauses,
        unsupported_clause_ids=tuple(item.aspect for item in clauses if item.disposition == ClauseDisposition.UNSUPPORTED.value),
        selection_rationale_refs=decision.evidence_refs, proof_refs=decision.proof_refs,
        counterexample_refs=requested_counterexamples, index_refs=packet_index_refs,
        post_edit_obligation_ids=tuple(post_edit_obligation_ids), validation_commands=tuple(validation_commands),
        reproof_commands=tuple(reproof_commands), invalidation_refs=decision.invalidation_refs,
        expansion_handles=tuple(expansion_handles),
    )


build_contract_repair_edit_packet = materialize_contract_repair_edit_packet


__all__ = [
    "CONTRACT_REPAIR_EDIT_PACKET_INTERFACE", "CONTRACT_REPAIR_EDIT_PACKET_SCHEMA",
    "CONTRACT_REPAIR_EDIT_PACKET_VERSION", "ContractClause", "ContractRepairEditPacket",
    "ContractRepairEditPacketError", "ContractRepairEditPacketReason", "ExpansionHandle",
    "build_contract_repair_edit_packet", "materialize_contract_repair_edit_packet",
]
