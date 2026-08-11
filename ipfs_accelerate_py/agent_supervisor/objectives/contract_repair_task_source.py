"""Project proof-gated contract-repair packets into exact supervisor tasks.

This is intentionally a projection boundary, not a task planner.  A provider
does not get to nominate files here: every emitted task derives its predicted
files and write scope directly from one canonical ``ContractRepairEditPacket``.
The source keeps a small in-memory decision index so a repeated finding or a
second packet for an already projected decision cannot mint another task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Final

from ..analysis.contract_repair_contracts import AuthorityRoots
from ..proof.contract_repair_edit_packet import (
    CONTRACT_REPAIR_EDIT_PACKET_INTERFACE,
    ContractRepairEditPacket,
    ContractRepairEditPacketError,
)
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .objective_graph import ObjectiveFinding, ObjectiveTaskRecord


CONTRACT_REPAIR_TASK_SOURCE_INTERFACE: Final[str] = "ContractRepairTaskSource@1"
CONTRACT_REPAIR_TASK_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-task-projection@1"
)
CONTRACT_REPAIR_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-supervisor-task@1"
)
TASK_ID_PREFIX: Final[str] = "RPR-PACKET-"
MAX_PROJECTED_PATHS: Final[int] = 256


class ContractRepairTaskSourceError(ValueError):
    """A packet cannot safely become an implementation task."""


class ContractRepairTaskProjectionReason(str, Enum):
    """Closed outcomes for the packet-to-task admission boundary."""

    EMITTED = "emitted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    MALFORMED = "malformed"
    SCOPE_MISMATCH = "scope_mismatch"


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ContractRepairTaskSourceError(f"{name} must be a non-empty trimmed identifier")
    if any(character.isspace() for character in value) or "\x00" in value:
        raise ContractRepairTaskSourceError(f"{name} must be an opaque identifier")
    return value


def _paths(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractRepairTaskSourceError(f"{name} must be a path sequence")
    if any(not isinstance(item, str) for item in value):
        raise ContractRepairTaskSourceError(f"{name} must contain only paths")
    result = tuple(sorted(set(value)))
    if not result or len(result) > MAX_PROJECTED_PATHS:
        raise ContractRepairTaskSourceError(f"{name} must be a bounded non-empty path sequence")
    return result


def deterministic_contract_repair_task_id(
    packet_id: str,
    decision_id: str,
    tree_id: str,
) -> str:
    """Return the stable task alias for one admitted decision in one tree.

    The packet identity is intentionally included even though task dedupe uses
    the decision/tree pair.  This makes the alias evidence-bearing while the
    source's decision index prevents a later packet from duplicating work.
    """

    digest = sha256(canonical_json_bytes({
        "schema": CONTRACT_REPAIR_TASK_SCHEMA,
        "packet_id": _identifier(packet_id, "packet_id"),
        "decision_id": _identifier(decision_id, "decision_id"),
        "tree_id": _identifier(tree_id, "tree_id"),
    })).hexdigest()
    return TASK_ID_PREFIX + digest[:24].upper()


def _packet_from(value: ContractRepairEditPacket | Mapping[str, Any]) -> ContractRepairEditPacket:
    if isinstance(value, ContractRepairEditPacket):
        # Reparse the canonical record.  This catches forged subclasses and
        # makes the boundary independent of incidental object mutation.
        return ContractRepairEditPacket.from_dict(value.to_record())
    if isinstance(value, Mapping):
        return ContractRepairEditPacket.from_dict(value)
    raise ContractRepairTaskSourceError("packet must be ContractRepairEditPacket@2 or its canonical record")


def _rejection_reason(value: Any) -> ContractRepairTaskProjectionReason:
    if isinstance(value, Mapping):
        strategy = str(value.get("strategy") or "").strip().lower()
        if strategy == "ambiguous":
            return ContractRepairTaskProjectionReason.AMBIGUOUS
        if strategy == "reject":
            return ContractRepairTaskProjectionReason.REJECTED
    return ContractRepairTaskProjectionReason.MALFORMED


def _format_refs(packet: ContractRepairEditPacket) -> str:
    return ", ".join(item.content_id for item in packet.proof_refs)


def _task_prompt(packet: ContractRepairEditPacket, task_id: str) -> str:
    """Render a bounded, source-free provider prompt from packet fields only."""

    observed = packet.receiver_observed_contract_id or "not-observed"
    clauses = "\n".join(
        f"- {clause.aspect}: {clause.disposition}; {clause.reason}"
        for clause in packet.clauses
    )
    unsupported = ", ".join(packet.unsupported_clause_ids) or "none"
    validation = "\n".join(f"- {command}" for command in packet.validation_commands)
    reproof = "\n".join(f"- {command}" for command in packet.reproof_commands)
    paths = ", ".join(packet.write_paths)
    return (
        f"Implement proof-gated contract repair task {task_id}.\n\n"
        "Authority binding:\n"
        f"- Packet: {packet.packet_id}\n- Decision: {packet.decision_id}\n"
        f"- Repository tree: {packet.roots.tree_id}\n"
        f"- Target: {packet.target_span.path}:{packet.target_span.start}-{packet.target_span.end}\n"
        f"- Selected strategy: {packet.strategy.value}\n"
        f"- Target reason evidence: {', '.join(item.content_id for item in packet.selection_rationale_refs)}\n\n"
        "Sender/receiver contract (precise identifiers):\n"
        f"- Sender expected: {packet.sender_expected_contract_id}\n"
        f"- Receiver expected: {packet.receiver_expected_contract_id}\n"
        f"- Receiver observed: {observed}\n"
        f"{clauses}\n\n"
        f"Unsupported limits: {unsupported}. Do not infer, implement, or claim support outside these limits.\n\n"
        "Scope is closed:\n"
        f"- Declared outputs: {paths}\n"
        f"- Allowed write paths: {paths}\n"
        f"- Allowed read paths: {', '.join(packet.read_paths)}\n"
        "- The provider must not add, modify, rename, or request any path outside the declared outputs.\n\n"
        f"Post-edit obligations: {', '.join(packet.post_edit_obligation_ids)}\n"
        f"Proof references: {_format_refs(packet)}\n"
        f"Index references: {', '.join(packet.index_refs)}\n\n"
        "Run focused validation:\n"
        f"{validation}\n\n"
        "Then re-prove the post-edit obligations:\n"
        f"{reproof}\n"
    )


@dataclass(frozen=True, slots=True)
class ContractRepairTaskProjection:
    """One emitted task or a typed no-task result.

    ``task_record`` is absent for rejected, ambiguous, stale, malformed, and
    duplicate input.  Consumers should use ``implementation_task`` rather
    than treating a failed packet as a partially specified task.
    """

    packet_id: str
    decision_id: str
    tree_id: str
    task_id: str
    reason: ContractRepairTaskProjectionReason
    predicted_files: tuple[str, ...]
    write_scope: tuple[str, ...]
    prompt: str = ""
    task_record: ObjectiveTaskRecord | None = None
    detail: str = ""

    @property
    def emitted(self) -> bool:
        return self.reason is ContractRepairTaskProjectionReason.EMITTED

    @property
    def implementation_task(self) -> ObjectiveTaskRecord | None:
        return self.task_record

    @property
    def declared_outputs(self) -> tuple[str, ...]:
        """The only outputs a provider may change for this task."""

        return self.predicted_files

    @property
    def allowed_write_paths(self) -> tuple[str, ...]:
        """Alias making the provider-facing authority fence explicit."""

        return self.write_scope

    @property
    def projection_id(self) -> str:
        return content_identity({
            "schema": CONTRACT_REPAIR_TASK_PROJECTION_SCHEMA,
            "packet_id": self.packet_id,
            "decision_id": self.decision_id,
            "tree_id": self.tree_id,
            "task_id": self.task_id,
            "reason": self.reason.value,
            "predicted_files": list(self.predicted_files),
            "write_scope": list(self.write_scope),
            "prompt": self.prompt,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_TASK_PROJECTION_SCHEMA,
            "interface": CONTRACT_REPAIR_TASK_SOURCE_INTERFACE,
            "projection_id": self.projection_id,
            "packet_id": self.packet_id,
            "decision_id": self.decision_id,
            "tree_id": self.tree_id,
            "task_id": self.task_id,
            "reason": self.reason.value,
            "predicted_files": list(self.predicted_files),
            "write_scope": list(self.write_scope),
            "prompt": self.prompt,
            "detail": self.detail,
            "emitted": self.emitted,
        }


class ContractRepairTaskSource:
    """Idempotently issue exact-scope supervisor tasks from admitted packets."""

    interface: Final[str] = CONTRACT_REPAIR_TASK_SOURCE_INTERFACE

    def __init__(
        self,
        *,
        current_roots: AuthorityRoots | None = None,
        roots: AuthorityRoots | None = None,
        current_tree_id: str | None = None,
    ) -> None:
        if current_roots is not None and roots is not None and current_roots != roots:
            raise ContractRepairTaskSourceError("current_roots and roots disagree")
        configured_roots = current_roots if current_roots is not None else roots
        if configured_roots is not None and not isinstance(configured_roots, AuthorityRoots):
            raise ContractRepairTaskSourceError("current_roots must be AuthorityRoots")
        self._lock = RLock()
        self._by_decision_tree: dict[tuple[str, str], ContractRepairTaskProjection] = {}
        self._current_roots = configured_roots
        self._current_tree_id = (
            _identifier(current_tree_id, "current_tree_id")
            if current_tree_id is not None else ""
        )

    def project(
        self,
        packet: ContractRepairEditPacket | Mapping[str, Any],
        *,
        current_roots: AuthorityRoots | None = None,
        roots: AuthorityRoots | None = None,
        current_tree_id: str | None = None,
        provider_outputs: Sequence[str] | None = None,
    ) -> ContractRepairTaskProjection:
        """Project one current packet, returning a no-task result on rejection.

        ``provider_outputs`` is accepted solely as a scope fence.  If supplied,
        it must be exactly the packet's writes; it never supplements them.
        ``roots`` is a compatibility spelling for ``current_roots``.
        """

        try:
            parsed = _packet_from(packet)
        except (ContractRepairEditPacketError, ContractRepairTaskSourceError, TypeError, ValueError) as exc:
            return ContractRepairTaskProjection(
                "", "", "", "", _rejection_reason(packet), (), (), detail=str(exc)
            )
        if current_roots is not None and roots is not None and current_roots != roots:
            raise ContractRepairTaskSourceError("current_roots and roots disagree")
        expected_roots = (
            current_roots if current_roots is not None
            else roots if roots is not None
            else self._current_roots
        )
        if expected_roots is not None and (not isinstance(expected_roots, AuthorityRoots) or parsed.roots != expected_roots):
            return ContractRepairTaskProjection(
                parsed.packet_id, parsed.decision_id, parsed.roots.tree_id, "",
                ContractRepairTaskProjectionReason.STALE, (), (),
                detail="packet authority roots are not current",
            )
        expected_tree_id = (
            _identifier(current_tree_id, "current_tree_id")
            if current_tree_id is not None else self._current_tree_id
        )
        if expected_tree_id and parsed.roots.tree_id != expected_tree_id:
            return ContractRepairTaskProjection(
                parsed.packet_id, parsed.decision_id, parsed.roots.tree_id, "",
                ContractRepairTaskProjectionReason.STALE, (), (),
                detail="packet tree is not current",
            )
        if provider_outputs is not None:
            try:
                supplied = _paths(provider_outputs, "provider_outputs")
            except ContractRepairTaskSourceError as exc:
                return ContractRepairTaskProjection(
                    parsed.packet_id, parsed.decision_id, parsed.roots.tree_id, "",
                    ContractRepairTaskProjectionReason.SCOPE_MISMATCH, (), (), detail=str(exc),
                )
            if supplied != parsed.write_paths:
                return ContractRepairTaskProjection(
                    parsed.packet_id, parsed.decision_id, parsed.roots.tree_id, "",
                    ContractRepairTaskProjectionReason.SCOPE_MISMATCH, (), (),
                    detail="provider outputs must exactly equal packet write_paths",
                )
        return self._emit(parsed)

    def project_packet(self, packet: ContractRepairEditPacket | Mapping[str, Any], **kwargs: Any) -> ContractRepairTaskProjection:
        """Explicit alias retained for integrations that name packet input."""

        return self.project(packet, **kwargs)

    def project_packets(
        self,
        packets: Sequence[ContractRepairEditPacket | Mapping[str, Any]],
        *,
        current_roots: AuthorityRoots | None = None,
    ) -> tuple[ContractRepairTaskProjection, ...]:
        """Compatibility spelling for a deterministic batch projection."""

        return self.project_many(packets, current_roots=current_roots)

    def project_many(
        self,
        packets: Sequence[ContractRepairEditPacket | Mapping[str, Any]],
        *,
        current_roots: AuthorityRoots | None = None,
    ) -> tuple[ContractRepairTaskProjection, ...]:
        """Project in canonical packet order without duplicate emitted tasks."""

        if isinstance(packets, (str, bytes, bytearray)) or not isinstance(packets, Sequence):
            raise ContractRepairTaskSourceError("packets must be a sequence")
        indexed = list(enumerate(packets))
        indexed.sort(key=lambda item: self._sort_key(item[1], item[0]))
        results: list[ContractRepairTaskProjection] = []
        emitted_decisions: set[tuple[str, str]] = set()
        for _, packet in indexed:
            projection = self.project(packet, current_roots=current_roots)
            if projection.emitted:
                key = (projection.decision_id, projection.tree_id)
                if key in emitted_decisions:
                    continue
                emitted_decisions.add(key)
            results.append(projection)
        return tuple(results)

    def _sort_key(self, packet: Any, ordinal: int) -> tuple[str, int]:
        try:
            return (_packet_from(packet).packet_id, ordinal)
        except (ContractRepairEditPacketError, ContractRepairTaskSourceError, TypeError, ValueError):
            return ("~invalid", ordinal)

    def _emit(self, packet: ContractRepairEditPacket) -> ContractRepairTaskProjection:
        key = (packet.decision_id, packet.roots.tree_id)
        with self._lock:
            existing = self._by_decision_tree.get(key)
            if existing is not None:
                if existing.packet_id == packet.packet_id:
                    return existing
                return ContractRepairTaskProjection(
                    packet.packet_id, packet.decision_id, packet.roots.tree_id, existing.task_id,
                    ContractRepairTaskProjectionReason.DUPLICATE, (), (),
                    detail="decision/tree already has an implementation task",
                )
            task_id = deterministic_contract_repair_task_id(packet.packet_id, packet.decision_id, packet.roots.tree_id)
            prompt = _task_prompt(packet, task_id)
            finding = ObjectiveFinding(
                fingerprint=packet.trace_id,
                goal_id="RPR-G080",
                title=f"Repair admitted contract target {packet.target_span.path}",
                summary="Repair the exact proof-gated sender/receiver contract target.",
                priority="P0",
                track="contract-repair",
                missing_evidence=list(packet.post_edit_obligation_ids),
                present_evidence={"proof_refs": [item.content_id for item in packet.proof_refs]},
                evidence_methods=["proof-gated-contract-repair"],
                objective_path="contract-repair-edit-packet",
                outputs=list(packet.write_paths),
                validation=" && ".join(packet.validation_commands),
                predicted_files=list(packet.write_paths),
                changed_paths=list(packet.write_paths),
                interfaces=[CONTRACT_REPAIR_EDIT_PACKET_INTERFACE, CONTRACT_REPAIR_TASK_SOURCE_INTERFACE],
                acceptance_subset=["write scope exactly equals packet write_paths", "validation and re-proof succeed"],
                evidence_subset=[item.content_id for item in packet.proof_refs],
                context_paths=list(packet.read_paths),
                semantic_identity=f"contract-repair:{packet.decision_id}:{packet.roots.tree_id}",
                dedupe_key=f"contract-repair:{packet.decision_id}:{packet.roots.tree_id}",
                completion_authority="",
            )
            task = ObjectiveTaskRecord(
                task_id=task_id,
                task_block=prompt,
                finding=finding,
                discovery_path=Path(packet.target_span.path),
            )
            projection = ContractRepairTaskProjection(
                packet.packet_id, packet.decision_id, packet.roots.tree_id, task_id,
                ContractRepairTaskProjectionReason.EMITTED, packet.write_paths,
                packet.write_paths, prompt, task,
            )
            self._by_decision_tree[key] = projection
            return projection


def project_contract_repair_task(
    packet: ContractRepairEditPacket | Mapping[str, Any],
    *,
    current_roots: AuthorityRoots | None = None,
    current_tree_id: str | None = None,
    provider_outputs: Sequence[str] | None = None,
) -> ContractRepairTaskProjection:
    """Stateless convenience entry point for one deterministic projection."""

    return ContractRepairTaskSource().project(
        packet, current_roots=current_roots, current_tree_id=current_tree_id,
        provider_outputs=provider_outputs
    )


__all__ = [
    "CONTRACT_REPAIR_TASK_PROJECTION_SCHEMA", "CONTRACT_REPAIR_TASK_SCHEMA",
    "CONTRACT_REPAIR_TASK_SOURCE_INTERFACE", "ContractRepairTaskProjection",
    "ContractRepairTaskProjectionReason", "ContractRepairTaskSource",
    "ContractRepairTaskSourceError", "deterministic_contract_repair_task_id",
    "project_contract_repair_task",
]
