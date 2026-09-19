"""SAWM-028 repair-operator and graph-delta prediction.

Analytical/deterministic routes run first. Predictions are proposal-only and
cannot weaken tests, touch protected paths, or complete tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PROTECTED_MARKERS: tuple[str, ...] = (
    "control.duckdb",
    "quack-owner",
    "completion_authority",
)


class RepairPredictionError(ValueError):
    """Closed repair-delta prediction failure."""


@dataclass(frozen=True, slots=True)
class RepairOperatorRanker:
    def rank(self, operators: Sequence[str], scores: Mapping[str, float] | None = None) -> tuple[str, ...]:
        weights = dict(scores or {})
        return tuple(sorted(operators, key=lambda name: (-float(weights.get(name, 0.0)), name)))


@dataclass(frozen=True, slots=True)
class PatchSketchDecoder:
    def decode(self, sketch: Mapping[str, Any]) -> Mapping[str, Any]:
        path = str(sketch.get("path") or "")
        if any(marker in path for marker in PROTECTED_MARKERS):
            raise RepairPredictionError("protected path")
        if sketch.get("weakens_tests"):
            raise RepairPredictionError("test weakening")
        return {"path": path, "operator": str(sketch.get("operator") or ""), "proposal_only": True}


@dataclass(frozen=True, slots=True)
class RepairPredictionAdmissionGate:
    def allow(self, request: Mapping[str, Any]) -> str | None:
        if request.get("ood"):
            return "ood_rejected"
        if request.get("analytical_route_open"):
            return "analytical_route_precedence"
        return None


@dataclass
class ProgramGraphDeltaPredictor:
    ranker: RepairOperatorRanker | None = None
    decoder: PatchSketchDecoder | None = None
    gate: RepairPredictionAdmissionGate | None = None

    def rank_repair_operators(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        blocked = (self.gate or RepairPredictionAdmissionGate()).allow(request)
        if blocked:
            return {
                "abstained": True,
                "reason_code": blocked,
                "proposal_only": True,
                "completion_authority": False,
            }
        ranked = (self.ranker or RepairOperatorRanker()).rank(
            tuple(request.get("operators") or ()),
            request.get("scores"),
        )
        return {
            "ranked": ranked,
            "proposal_only": True,
            "completion_authority": False,
        }

    def predict_program_graph_delta(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        blocked = (self.gate or RepairPredictionAdmissionGate()).allow(request)
        if blocked:
            return {
                "abstained": True,
                "reason_code": blocked,
                "proposal_only": True,
                "completion_authority": False,
            }
        sketch = (self.decoder or PatchSketchDecoder()).decode(request.get("sketch") or {})
        return {
            "sketch": sketch,
            "proposal_only": True,
            "completion_authority": False,
        }


def rank_repair_operators(request: Mapping[str, Any]) -> Mapping[str, Any]:
    return ProgramGraphDeltaPredictor().rank_repair_operators(request)


def predict_program_graph_delta(request: Mapping[str, Any]) -> Mapping[str, Any]:
    return ProgramGraphDeltaPredictor().predict_program_graph_delta(request)
