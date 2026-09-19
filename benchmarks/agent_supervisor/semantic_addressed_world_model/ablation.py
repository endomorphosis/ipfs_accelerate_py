"""SAWM-041 controlled ablation and efficiency benchmark.

Reports ladder A-L metrics. Results grant no completion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


LADDER: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(12))


class AblationError(ValueError):
    """Closed ablation contract violation."""


@dataclass(frozen=True, slots=True)
class SemanticWorldEfficiencyMetrics:
    model_calls: int
    tokens: int
    prefix_reuse: float
    avoided_calls: int
    cost: float


@dataclass(frozen=True, slots=True)
class SemanticWorldSafetyMetrics:
    false_reuse: int
    test_weakening: int
    protected_path_hits: int


@dataclass
class SemanticWorldAblationRunner:
    def run_semantic_world_ablation(
        self, rungs: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        names = [str(item.get("rung") or "") for item in rungs]
        if names and set(names) - set(LADDER):
            raise AblationError("unknown ablation rung")
        efficiency = SemanticWorldEfficiencyMetrics(
            model_calls=sum(int(item.get("model_calls") or 0) for item in rungs),
            tokens=sum(int(item.get("tokens") or 0) for item in rungs),
            prefix_reuse=float(sum(float(item.get("prefix_reuse") or 0) for item in rungs) / max(len(rungs), 1)),
            avoided_calls=sum(int(item.get("avoided_calls") or 0) for item in rungs),
            cost=sum(float(item.get("cost") or 0) for item in rungs),
        )
        safety = SemanticWorldSafetyMetrics(
            false_reuse=sum(int(item.get("false_reuse") or 0) for item in rungs),
            test_weakening=sum(int(item.get("test_weakening") or 0) for item in rungs),
            protected_path_hits=sum(int(item.get("protected_path_hits") or 0) for item in rungs),
        )
        return {
            "rungs": names,
            "efficiency": efficiency,
            "safety": safety,
            "completion_authority": False,
        }


def run_semantic_world_ablation(rungs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return SemanticWorldAblationRunner().run_semantic_world_ablation(rungs)


def validate_semantic_world_ablation(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("completion_authority") is True:
        raise AblationError("ablation cannot grant completion authority")
    safety = result.get("safety")
    if isinstance(safety, SemanticWorldSafetyMetrics) and (
        safety.test_weakening or safety.protected_path_hits
    ):
        raise AblationError("safety gates were weakened")
    return result
