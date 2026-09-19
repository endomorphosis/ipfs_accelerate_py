"""SAWM-025 proposal-only call-target ranking specialist.

Ranks only existing static candidate symbol IDs plus explicit unknown.
Never erases a statically required possibility. Stale/OOD cases abstain.
Results cannot self-promote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence


UNKNOWN: Final[str] = "unknown"


class CallTargetRankingError(ValueError):
    """Closed ranking-specialist contract violation."""


@dataclass(frozen=True, slots=True)
class CallTargetRankingRequest:
    current_symbol: str
    static_candidates: tuple[str, ...]
    unknown_frontier: bool = False
    ood: bool = False
    stale: bool = False
    scores: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class CallTargetRankingResult:
    ranked: tuple[str, ...]
    abstained: bool
    reason_code: str
    proposal_only: bool = True
    admitted: bool = False
    coverage: float = 0.0


@dataclass(frozen=True, slots=True)
class CallTargetOODGate:
    def reject_or_abstain(self, request: CallTargetRankingRequest) -> str | None:
        if request.stale:
            return "stale_snapshot"
        if request.ood:
            return "ood_rejected"
        return None


@dataclass
class ProgramCallTargetRanker:
    ood_gate: CallTargetOODGate | None = None

    def rank_program_call_targets(
        self, request: CallTargetRankingRequest
    ) -> CallTargetRankingResult:
        gate = self.ood_gate or CallTargetOODGate()
        blocked = gate.reject_or_abstain(request)
        if blocked:
            return CallTargetRankingResult(
                ranked=(),
                abstained=True,
                reason_code=blocked,
                coverage=0.0,
            )
        static = tuple(dict.fromkeys(request.static_candidates))
        if not static and not request.unknown_frontier:
            raise CallTargetRankingError("no static candidates or unknown frontier")
        scores = dict(request.scores or {})
        illegal = set(scores) - set(static) - {UNKNOWN}
        if illegal:
            raise CallTargetRankingError(
                f"ranker cannot introduce non-candidate symbols {sorted(illegal)}"
            )
        ordered = tuple(sorted(static, key=lambda name: (-float(scores.get(name, 0.0)), name)))
        if request.unknown_frontier:
            ordered = (*ordered, UNKNOWN)
        if set(static) - set(ordered):
            raise CallTargetRankingError("statically required possibility was erased")
        return CallTargetRankingResult(
            ranked=ordered,
            abstained=False,
            reason_code="ranked_static_candidates",
            coverage=1.0 if static else 0.0,
        )


def rank_program_call_targets(
    request: CallTargetRankingRequest | Mapping[str, Any],
) -> CallTargetRankingResult:
    if not isinstance(request, CallTargetRankingRequest):
        request = CallTargetRankingRequest(
            current_symbol=str(request["current_symbol"]),
            static_candidates=tuple(request.get("static_candidates") or ()),
            unknown_frontier=bool(request.get("unknown_frontier")),
            ood=bool(request.get("ood")),
            stale=bool(request.get("stale")),
            scores=request.get("scores"),
        )
    return ProgramCallTargetRanker().rank_program_call_targets(request)
