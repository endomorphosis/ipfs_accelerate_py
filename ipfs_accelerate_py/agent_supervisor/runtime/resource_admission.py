"""Join resource reservations with a selected frontier."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence


class ResourceAdmissionError(ValueError):
    """Frontier dispatch was refused by resource admission."""


def admit_frontier(
    selected: Sequence[Mapping[str, Any]],
    *,
    capacity: Mapping[str, int],
) -> Mapping[str, Any]:
    used: dict[str, int] = {key: 0 for key in capacity}
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    for task in selected:
        demand = dict(task.get("demand") or {})
        overflow = None
        for key, need in demand.items():
            if type(need) is not int:
                raise ResourceAdmissionError("demand must be integer")
            if used.get(key, 0) + need > int(capacity.get(key, 0)):
                overflow = key
                break
        if overflow:
            rejected.append({"task_id": str(task.get("task_id")), "reason": f"capacity:{overflow}"})
            continue
        for key, need in demand.items():
            used[key] = used.get(key, 0) + need
        accepted.append(str(task.get("task_id")))
    return MappingProxyType(
        {
            "accepted": tuple(accepted),
            "rejected": tuple(rejected),
            "used": MappingProxyType(used),
            "dispatched": False,
        }
    )
