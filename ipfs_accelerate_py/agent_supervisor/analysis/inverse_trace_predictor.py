"""SAWM-027 inverse trace reconstruction.

Ranks plausible predecessor state/event sets. Predictions never become
observations or completions. OOD/stale/environment mismatch abstain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class InverseTraceError(ValueError):
    """Closed inverse-trace contract violation."""


@dataclass(frozen=True, slots=True)
class PredecessorStateCandidate:
    state_id: str
    score: float


@dataclass(frozen=True, slots=True)
class PredecessorEventCandidate:
    event_id: str
    score: float


@dataclass(frozen=True, slots=True)
class FailureReproductionEvaluator:
    def useful(self, predicted: Sequence[str], reproducing: Sequence[str]) -> bool:
        return bool(predicted) and set(predicted) <= set(reproducing)


@dataclass
class InverseTracePredictor:
    def rank_inverse_trace_predecessors(
        self, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if request.get("stale") or request.get("ood") or request.get("environment_mismatch"):
            reason = (
                "stale_snapshot"
                if request.get("stale")
                else "ood_rejected"
                if request.get("ood")
                else "environment_mismatch"
            )
            return {
                "abstained": True,
                "reason_code": reason,
                "proposal_only": True,
                "observation": False,
                "completion_authority": False,
            }
        states = [
            PredecessorStateCandidate(str(item["state_id"]), float(item.get("score") or 0.0))
            for item in request.get("predecessor_states") or ()
        ]
        events = [
            PredecessorEventCandidate(str(item["event_id"]), float(item.get("score") or 0.0))
            for item in request.get("predecessor_events") or ()
        ]
        ranked_states = tuple(sorted(states, key=lambda item: (-item.score, item.state_id)))
        ranked_events = tuple(sorted(events, key=lambda item: (-item.score, item.event_id)))
        missing = tuple(str(item) for item in request.get("missing_inputs") or ())
        reproducing = tuple(str(item) for item in request.get("reproducing_inputs") or ())
        evaluator = FailureReproductionEvaluator()
        return {
            "states": [item.state_id for item in ranked_states],
            "events": [item.event_id for item in ranked_events],
            "missing_inputs": missing,
            "multiple_valid": len(ranked_states) > 1,
            "reproduction_useful": evaluator.useful(missing, reproducing),
            "proposal_only": True,
            "observation": False,
            "completion_authority": False,
        }


def rank_inverse_trace_predecessors(request: Mapping[str, Any]) -> Mapping[str, Any]:
    return InverseTracePredictor().rank_inverse_trace_predecessors(request)
