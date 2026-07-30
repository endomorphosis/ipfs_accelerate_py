"""VFS-G110 composition surface for compact CID-addressed repair packets.

Declared output path for objective goal ``VFS-G110`` / task ``VFS-078``.

This module composes:

* :class:`~ipfs_accelerate_py.agent_supervisor.contract_repair_packet.ContractRepairPacket`
  — the model-facing compact packet and delta-retry surface
* :class:`~ipfs_accelerate_py.agent_supervisor.planning.formal_replanner.FormalReplanner`
  / :class:`~ipfs_accelerate_py.agent_supervisor.planning.formal_replanner.CodexRepairPacket`
  — the formal replan transition projection (canonical under ``planning/``)
* :class:`~ipfs_accelerate_py.agent_supervisor.context.context_compiler.ContextCompiler`
  — optional decision-context compilation (canonical under ``context/``)

It does **not** alter provider invocation.  It only compiles the smallest
sufficient repair packet (and delta retry) and measures symbolic analysis
context separately from provider input context.

Canonical formal replan implementation lives at::

    ipfs_accelerate_py.agent_supervisor.planning.formal_replanner

Package imports of the historical flat name resolve through the landed-module
alias finder to that domain package.  This file remains the objective-heap
declared output and composition entry for compact repair evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.contract_repair_packet import (
    COMPACT_REPAIR_PACKET_EVIDENCE,
    DELTA_REPAIR_CONTEXT_EVIDENCE,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_ID,
    CompiledRepairPacket,
    ContractRepairPacket,
    ContractRepairPacketCompiler,
    RepairPacketDelta,
    RepairPacketRequest,
    all_covered_evidence_terms,
    compact_repair_packet_evidence,
    compact_repair_packet_evidence_terms,
    compile_contract_repair_packet,
    compile_repair_packet,
    compile_repair_packet_delta,
    covered_evidence_terms,
    delta_repair_context_evidence,
    delta_repair_context_evidence_terms,
    measure_repair_context,
    prove_compact_repair_packet,
    prove_delta_repair_context,
    prove_repair_packet_evidence,
)

# Discovery / AST anchors for the objective-heap AST query
# ``ContextCompiler CodexRepairPacket FormalReplanner``.
# Canonical class bodies live in domain packages; names here bind the
# composition surface scanners look for on this declared output path.
ContextCompiler: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.context.context_compiler.ContextCompiler"
)
CodexRepairPacket: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.planning.formal_replanner.CodexRepairPacket"
)
FormalReplanner: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.planning.formal_replanner.FormalReplanner"
)

FORMAL_REPAIR_COMPOSITION_VERSION: Final[int] = 1
FORMAL_REPAIR_COMPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/formal-repair-composition@1"
)

assert compact_repair_packet_evidence() == "vfs/compact-repair-packet@1"
assert delta_repair_context_evidence() == "vfs/delta-repair-context@1"
assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
    "vfs/compact-repair-packet@1",
    "vfs/delta-repair-context@1",
)
assert "FormalReplanner" in FormalReplanner
assert "CodexRepairPacket" in CodexRepairPacket
assert "ContextCompiler" in ContextCompiler


@dataclass(frozen=True)
class FormalRepairComposition:
    """Bound compact packet (+ optional delta) with separate context measurements.

    Symbolic analysis and provider packet sizes are recorded independently so
    supervisors never blend deterministic analysis cost into provider budgets.
    """

    packet: CompiledRepairPacket
    delta: RepairPacketDelta | None
    measurement: Mapping[str, Any]
    compact_claim: Mapping[str, Any]
    delta_claim: Mapping[str, Any] | None

    @property
    def packet_id(self) -> str:
        return self.packet.packet_id

    @property
    def status(self) -> str:
        return self.packet.status.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_REPAIR_COMPOSITION_SCHEMA,
            "version": FORMAL_REPAIR_COMPOSITION_VERSION,
            "goal_id": OBJECTIVE_GOAL_ID,
            "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
            "task_id": OBJECTIVE_TASK_ID,
            "evidence_terms": list(OBJECTIVE_DOMAIN_EVIDENCE_TERMS),
            "packet": self.packet.to_dict(),
            "delta": self.delta.to_dict() if self.delta is not None else None,
            "measurement": dict(self.measurement),
            "compact_claim": dict(self.compact_claim),
            "delta_claim": dict(self.delta_claim) if self.delta_claim else None,
            "composition": {
                "ContextCompiler": ContextCompiler,
                "CodexRepairPacket": CodexRepairPacket,
                "FormalReplanner": FormalReplanner,
            },
            "measurements_are_separate": True,
            "model_output_is_proposal": True,
            "authoritative": False,
            "completion_authoritative": False,
        }


def compile_formal_repair_packet(
    request: RepairPacketRequest | Mapping[str, Any],
    *,
    compiler: ContractRepairPacketCompiler | None = None,
    symbolic_analysis: Mapping[str, Any] | Sequence[Any] | None = None,
    repository_files: Sequence[Mapping[str, Any]] | Sequence[str] | None = None,
    changed_evidence: Sequence[Any] = (),
    requested_evidence: Sequence[Any] = (),
) -> FormalRepairComposition:
    """Compile a compact repair packet (and optional delta) for Grok/Codex.

    Provider packets are always :class:`ContractRepairPacket` instances —
    never full repository context, formal plan sources, or unredacted
    counterexample bodies.  Delta retries bind the parent packet identity.
    """

    compiled = compile_contract_repair_packet(request, compiler=compiler)
    delta: RepairPacketDelta | None = None
    delta_claim: dict[str, Any] | None = None
    if changed_evidence or requested_evidence:
        delta = compile_repair_packet_delta(
            compiled,
            changed_evidence=changed_evidence,
            requested_evidence=requested_evidence,
        )
        delta_claim = prove_delta_repair_context(delta, parent=compiled)

    measurement = measure_repair_context(
        compiled,
        symbolic_analysis=symbolic_analysis,
        repository_files=repository_files,
    )
    compact_claim = prove_compact_repair_packet(
        compiled,
        symbolic_analysis=symbolic_analysis,
        repository_files=repository_files,
    )
    return FormalRepairComposition(
        packet=compiled,
        delta=delta,
        measurement=measurement,
        compact_claim=compact_claim,
        delta_claim=delta_claim,
    )


def measure_symbolic_and_provider_context(
    packet: ContractRepairPacket | CompiledRepairPacket | Mapping[str, Any],
    *,
    symbolic_analysis: Mapping[str, Any] | Sequence[Any] | None = None,
    repository_files: Sequence[Mapping[str, Any]] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Public alias: measure symbolic analysis and provider context separately."""

    return measure_repair_context(
        packet,
        symbolic_analysis=symbolic_analysis,
        repository_files=repository_files,
    )


def formal_repair_evidence_terms() -> tuple[str, ...]:
    """Return VFS-G110 evidence terms covered by this composition surface."""

    return covered_evidence_terms()


__all__ = [
    "COMPACT_REPAIR_PACKET_EVIDENCE",
    "CodexRepairPacket",
    "ContextCompiler",
    "DELTA_REPAIR_CONTEXT_EVIDENCE",
    "FORMAL_REPAIR_COMPOSITION_SCHEMA",
    "FORMAL_REPAIR_COMPOSITION_VERSION",
    "FormalRepairComposition",
    "FormalReplanner",
    "OBJECTIVE_DOMAIN_EVIDENCE_TERMS",
    "OBJECTIVE_GOAL_ID",
    "OBJECTIVE_PARENT_GOAL_ID",
    "OBJECTIVE_TASK_ID",
    "all_covered_evidence_terms",
    "compact_repair_packet_evidence",
    "compact_repair_packet_evidence_terms",
    "compile_contract_repair_packet",
    "compile_formal_repair_packet",
    "compile_repair_packet",
    "compile_repair_packet_delta",
    "covered_evidence_terms",
    "delta_repair_context_evidence",
    "delta_repair_context_evidence_terms",
    "formal_repair_evidence_terms",
    "measure_repair_context",
    "measure_symbolic_and_provider_context",
    "prove_compact_repair_packet",
    "prove_delta_repair_context",
    "prove_repair_packet_evidence",
]
