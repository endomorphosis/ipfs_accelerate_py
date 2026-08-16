"""SupervisorWorldView@1 — mutation-free queries over one verified snapshot."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

try:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_contracts import (
        REQUIRED_COMPONENTS,
        parse_world_snapshot,
    )
except ImportError:  # LGSWF-012 may merge before LGSWF-010
    REQUIRED_COMPONENTS = ()

    def parse_world_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(snapshot, Mapping) or "snapshot_cid" not in snapshot:
            raise ValueError("snapshot is not a verified SupervisorWorldSnapshot")
        return MappingProxyType(dict(snapshot))


class WorldViewError(ValueError):
    """A world-view query or mutation was rejected."""


_QUERY_NAMES = (
    "goal_state",
    "subgoal_state",
    "task_state",
    "semantic_binding",
    "dependencies",
    "conflicts",
    "resources",
    "claims",
    "capsules",
    "contracts",
    "obligations",
    "completion_evidence",
    "refill_eligibility",
)


class SupervisorWorldView:
    """Pure read model bound to one verified snapshot and injected views."""

    __slots__ = ("_snapshot", "_views")

    def __init__(
        self,
        snapshot: Mapping[str, Any],
        views: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        frozen = parse_world_snapshot(snapshot)
        injected = views or {}
        if not isinstance(injected, Mapping):
            raise WorldViewError("views must be an injected mapping")
        object.__setattr__(self, "_snapshot", frozen)
        object.__setattr__(
            self,
            "_views",
            MappingProxyType(
                {
                    str(key): MappingProxyType(dict(value))
                    for key, value in injected.items()
                }
            ),
        )

    @property
    def snapshot_cid(self) -> str:
        return str(self._snapshot["snapshot_cid"])

    def component(self, name: str) -> Mapping[str, Any]:
        components = self._snapshot["components"]
        if name not in components:
            raise WorldViewError(f"unknown reference: {name}")
        return components[name]

    def _lookup(self, collection: str, item_id: str) -> Mapping[str, Any]:
        if collection not in self._views:
            return MappingProxyType(
                {
                    "id": item_id,
                    "found": False,
                    "reason": "unknown-reference",
                    "snapshot_cid": self.snapshot_cid,
                }
            )
        record = self._views[collection].get(item_id)
        if record is None:
            return MappingProxyType(
                {
                    "id": item_id,
                    "found": False,
                    "reason": "unknown-reference",
                    "snapshot_cid": self.snapshot_cid,
                }
            )
        payload = dict(record)
        payload.setdefault("id", item_id)
        payload["found"] = True
        payload["snapshot_cid"] = self.snapshot_cid
        return MappingProxyType(payload)

    def goal_state(self, goal_id: str) -> Mapping[str, Any]:
        return self._lookup("goals", goal_id)

    def subgoal_state(self, subgoal_id: str) -> Mapping[str, Any]:
        return self._lookup("subgoals", subgoal_id)

    def task_state(self, task_id: str) -> Mapping[str, Any]:
        return self._lookup("tasks", task_id)

    def semantic_binding(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("bindings", entity_id)

    def dependencies(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("dependencies", entity_id)

    def conflicts(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("conflicts", entity_id)

    def resources(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("resources", entity_id)

    def claims(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("claims", entity_id)

    def capsules(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("capsules", entity_id)

    def contracts(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("contracts", entity_id)

    def obligations(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("obligations", entity_id)

    def completion_evidence(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("completion", entity_id)

    def refill_eligibility(self, entity_id: str) -> Mapping[str, Any]:
        return self._lookup("refill", entity_id)

    def query_matrix(self) -> tuple[str, ...]:
        return _QUERY_NAMES

    def __setattr__(self, name: str, value: Any) -> None:
        raise WorldViewError("SupervisorWorldView is immutable")

    def __delattr__(self, name: str) -> None:
        raise WorldViewError("SupervisorWorldView is immutable")


def required_query_names() -> tuple[str, ...]:
    return _QUERY_NAMES


def required_component_names() -> tuple[str, ...]:
    return REQUIRED_COMPONENTS
