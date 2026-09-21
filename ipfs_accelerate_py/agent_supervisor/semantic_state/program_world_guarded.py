"""SAWM-036 guarded program-world influence.

Exact current-state hits, verified procedures, and independently
proof-backed scoped relations may influence planning. Neural candidates
may only rank or add context. Guarded influence never completes a task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class GuardedInfluenceError(ValueError):
    """Closed guarded-influence contract violation."""


@dataclass(frozen=True, slots=True)
class GuardedInfluenceDecision:
    allowed: bool
    reason_code: str
    influences_planning: bool
    neural_only_context: bool
    completion_authority: bool = False


@dataclass(frozen=True, slots=True)
class GuardedQualificationReceipt:
    qualified: bool
    reason_code: str
    completion_authority: bool = False


@dataclass
class ProgramWorldGuardedGate:
    def evaluate_guarded_program_world_influence(
        self, request: Mapping[str, Any]
    ) -> GuardedInfluenceDecision:
        kind = str(request.get("kind") or "")
        if request.get("shadow_unqualified"):
            return GuardedInfluenceDecision(
                allowed=False,
                reason_code="shadow_unqualified",
                influences_planning=False,
                neural_only_context=False,
            )
        if kind == "exact_hit" and request.get("current") is True:
            return GuardedInfluenceDecision(
                allowed=True,
                reason_code="exact_current_state_hit",
                influences_planning=True,
                neural_only_context=False,
            )
        if kind == "verified_procedure" and request.get("verified") is True:
            return GuardedInfluenceDecision(
                allowed=True,
                reason_code="verified_procedure",
                influences_planning=True,
                neural_only_context=False,
            )
        if kind == "proof_backed" and request.get("independent_proof") is True:
            return GuardedInfluenceDecision(
                allowed=True,
                reason_code="independent_proof",
                influences_planning=True,
                neural_only_context=False,
            )
        if kind == "neural":
            return GuardedInfluenceDecision(
                allowed=True,
                reason_code="neural_context_only",
                influences_planning=False,
                neural_only_context=True,
            )
        return GuardedInfluenceDecision(
            allowed=False,
            reason_code="unguarded_influence",
            influences_planning=False,
            neural_only_context=False,
        )


def evaluate_guarded_program_world_influence(
    request: Mapping[str, Any],
) -> GuardedInfluenceDecision:
    decision = ProgramWorldGuardedGate().evaluate_guarded_program_world_influence(request)
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        mirror_work_record(
            catalog_kind="world_model",
            record_kind="guarded_program_world_influence",
            record_ref=str(decision.reason_code or "guarded-influence"),
            subject_kind="record_cid",
            subject_ref=str(request.get("kind") or decision.reason_code or "guarded-influence"),
        )
    except Exception:
        pass
    return decision
