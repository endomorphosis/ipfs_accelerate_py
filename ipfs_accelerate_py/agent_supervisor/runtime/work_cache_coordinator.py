"""Historical estimates, cache locality, and single-flight coordination."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


class WorkCacheError(ValueError):
    """Cache/single-flight coordination rejected the request."""


class WorkCacheCoordinator:
    def __init__(self) -> None:
        self._inflight: dict[str, str] = {}
        self._estimates: dict[str, int] = {}

    def remember_estimate(self, key: str, cost: int) -> None:
        if type(cost) is not int or cost < 0:
            raise WorkCacheError("estimate must be a non-negative int")
        self._estimates[key] = cost

    def estimate(self, key: str) -> int | None:
        return self._estimates.get(key)

    def begin_single_flight(self, key: str, owner: str) -> Mapping[str, Any]:
        current = self._inflight.get(key)
        if current and current != owner:
            return MappingProxyType({"joined": True, "owner": current, "duplicate": False})
        self._inflight[key] = owner
        return MappingProxyType({"joined": False, "owner": owner, "duplicate": False})

    def end_single_flight(self, key: str, owner: str) -> None:
        if self._inflight.get(key) == owner:
            self._inflight.pop(key, None)
