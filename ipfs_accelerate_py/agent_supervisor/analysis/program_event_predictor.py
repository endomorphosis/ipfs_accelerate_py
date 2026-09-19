"""SAWM-026 proposal-only next-event and next-state prediction.

Predictions never become observations, proofs, or completions. Observed
debugger/runtime events remain ground truth. Drift, OOD, and unavailable
checkpoints abstain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence


EVENT_GRAMMAR: Final[frozenset[str]] = frozenset(
    {
        "call",
        "return",
        "raise",
        "assign",
        "await",
        "yield",
        "unknown",
    }
)


class EventPredictionError(ValueError):
    """Closed event-prediction contract violation."""


@dataclass(frozen=True, slots=True)
class StructuredEventDecoder:
    allowed: frozenset[str] = EVENT_GRAMMAR

    def decode(self, event_type: str) -> str:
        text = str(event_type or "").strip()
        if text not in self.allowed:
            raise EventPredictionError(f"impossible event {text!r}")
        return text


@dataclass(frozen=True, slots=True)
class EventPredictionOODGate:
    def abstain_reason(self, request: Mapping[str, Any]) -> str | None:
        if request.get("stale"):
            return "stale_checkpoint"
        if request.get("ood"):
            return "ood_rejected"
        if request.get("checkpoint_unavailable"):
            return "checkpoint_unavailable"
        if request.get("drift"):
            return "calibration_drift"
        return None


@dataclass(frozen=True, slots=True)
class ProgramStateDeltaPredictor:
    def predict(self, current_state: str, event_type: str) -> str:
        return f"{current_state}->{event_type}"


@dataclass(frozen=True, slots=True)
class ProgramNextEventPredictor:
    decoder: StructuredEventDecoder | None = None
    ood_gate: EventPredictionOODGate | None = None
    delta: ProgramStateDeltaPredictor | None = None

    def predict_next_program_event(
        self, request: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        gate = self.ood_gate or EventPredictionOODGate()
        blocked = gate.abstain_reason(request)
        if blocked:
            return {
                "abstained": True,
                "reason_code": blocked,
                "proposal_only": True,
                "observation": False,
                "completion_authority": False,
            }
        decoder = self.decoder or StructuredEventDecoder()
        event_type = decoder.decode(str(request.get("event_type") or "unknown"))
        allowed_targets = tuple(request.get("allowed_targets") or ())
        target = request.get("target")
        if target is not None and allowed_targets and target not in allowed_targets:
            raise EventPredictionError("target is not an existing allowed symbol")
        observed = request.get("observed_event")
        delta = (self.delta or ProgramStateDeltaPredictor()).predict(
            str(request.get("current_state") or "state"), event_type
        )
        return {
            "event_type": event_type,
            "target": target,
            "state_delta": delta,
            "abstained": False,
            "proposal_only": True,
            "observation": False,
            "matches_observation": observed == event_type if observed is not None else None,
            "completion_authority": False,
        }


def predict_next_program_event(request: Mapping[str, Any]) -> Mapping[str, Any]:
    return ProgramNextEventPredictor().predict_next_program_event(request)
