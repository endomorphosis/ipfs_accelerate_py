"""SAWM-035 shadow_read for hypothetical reuse/prediction/routing.

Shadow reads never change execution, planning, or completion.
Missing bindings are typed unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ShadowReadError(ValueError):
    """Closed shadow-read contract violation."""


@dataclass(frozen=True, slots=True)
class HypotheticalReuseDecision:
    query: str
    would_reuse: bool
    reason_code: str
    execution_changed: bool = False
    completion_authority: bool = False


@dataclass(frozen=True, slots=True)
class ShadowReadComparison:
    false_candidates: tuple[str, ...]
    missed_opportunities: tuple[str, ...]
    execution_changed: bool = False


@dataclass
class ProgramWorldShadowReader:
    def evaluate_program_world_shadow_reads(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        if request.get("binding_missing"):
            return {
                "status": "unavailable",
                "reason_code": "missing_binding",
                "execution_changed": False,
                "completion_authority": False,
            }
        gold = tuple(request.get("authoritative_hits") or ())
        shadow = tuple(request.get("shadow_hits") or ())
        false_candidates = tuple(item for item in shadow if item not in gold)
        missed = tuple(item for item in gold if item not in shadow)
        decision = HypotheticalReuseDecision(
            query=str(request.get("query") or ""),
            would_reuse=bool(shadow) and not false_candidates,
            reason_code="hypothetical_only",
        )
        comparison = ShadowReadComparison(
            false_candidates=false_candidates,
            missed_opportunities=missed,
        )
        return {
            "decision": decision,
            "comparison": comparison,
            "execution_changed": False,
            "completion_authority": False,
        }


def evaluate_program_world_shadow_reads(request: Mapping[str, Any]) -> dict[str, Any]:
    return ProgramWorldShadowReader().evaluate_program_world_shadow_reads(request)
