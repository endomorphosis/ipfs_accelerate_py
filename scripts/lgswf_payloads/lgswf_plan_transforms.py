"""LGSWF plan-transform proposals over existing PlanDelta surfaces."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

ALLOWED = frozenset({"split", "coalesce", "rewire", "speculate"})
IMMUTABLE_LIFECYCLES = frozenset({"claimed", "running", "accepted"})


class PlanTransformError(ValueError):
    """A plan transform proposal was rejected."""


def propose_plan_transform(request: Mapping[str, Any]) -> Mapping[str, Any]:
    kind = request.get("kind")
    if kind not in ALLOWED:
        raise PlanTransformError(f"unsupported transform {kind!r}")
    lifecycle = request.get("lifecycle") or "future"
    if lifecycle in IMMUTABLE_LIFECYCLES:
        raise PlanTransformError("claimed-through-accepted tasks are immutable")
    if kind == "speculate" and request.get("mutate_canonical"):
        raise PlanTransformError("speculation cannot mutate canonical authority")
    if int(request.get("amplification", 1)) > int(request.get("amplification_bound", 4)):
        raise PlanTransformError("unbounded task growth")
    return MappingProxyType(
        {
            "kind": kind,
            "accepted": False,
            "proposal": True,
            "evidence": request.get("evidence") or "coverage",
            "coverage_equivalent": bool(request.get("coverage_equivalent", True)),
            "predicted_parallelism": int(request.get("predicted_parallelism", 1)),
            "critical_path_delta": int(request.get("critical_path_delta", 0)),
            "resource_delta": int(request.get("resource_delta", 0)),
            "amplification_bound": int(request.get("amplification_bound", 4)),
            "dedup_key": str(request.get("dedup_key") or f"{kind}:{request.get('target')}"),
            "fallback": str(request.get("fallback") or "keep-current"),
            "human_review": bool(request.get("human_review", False)),
            "cancellable": True,
        }
    )
