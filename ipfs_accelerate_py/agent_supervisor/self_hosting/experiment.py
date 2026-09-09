"""Frozen input records for bounded self-hosting observations.

These records deliberately describe an experiment only.  They contain no
thresholds, scoring rules, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

PLAN_SCHEMA: Final[str] = "ipfs-accelerate.self-hosting.v0.1/experiment-plan"
TASK_SCHEMA: Final[str] = "ipfs-accelerate.self-hosting.v0.1/task"
EVIDENCE_KINDS: Final[tuple[str, ...]] = ("live", "replayed", "simulated")


def _json_value(value: Any) -> Any:
    """Materialize immutable runtime mappings before canonical serialization."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return the restricted canonical JSON used for stable experiment ids."""
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(kind: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json({"kind": kind, **dict(value)}).encode()).hexdigest()
    return f"sha256:{digest}"


def _required(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class SelfHostingTask:
    """A pinned task specification and its bounded external patch proposal."""

    task_id: str
    task_specification_cid: str
    proposal: Mapping[str, Any] = field(default_factory=dict)
    replay_record: Mapping[str, Any] | None = None
    schema: str = TASK_SCHEMA

    def __post_init__(self) -> None:
        _required(self.task_id, "task_id")
        _required(self.task_specification_cid, "task_specification_cid")
        object.__setattr__(self, "proposal", _mapping(self.proposal, "proposal"))
        if self.replay_record is not None:
            object.__setattr__(self, "replay_record", _mapping(self.replay_record, "replay_record"))
        if self.schema != TASK_SCHEMA:
            raise ValueError("task schema is not supported")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SelfHostingTask":
        return cls(
            schema=str(raw.get("schema", TASK_SCHEMA)),
            task_id=_required(raw.get("task_id"), "task_id"),
            task_specification_cid=_required(raw.get("task_specification_cid"), "task_specification_cid"),
            proposal=_mapping(raw.get("proposal", {}), "proposal"),
            replay_record=_mapping(raw["replay_record"], "replay_record") if raw.get("replay_record") is not None else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": self.schema, "task_id": self.task_id, "task_specification_cid": self.task_specification_cid, "proposal": dict(self.proposal)}
        if self.replay_record is not None:
            result["replay_record"] = dict(self.replay_record)
        return result


@dataclass(frozen=True)
class ExperimentPlan:
    """Identity-bound, frozen self-hosting observation plan."""

    engine_id: str
    package_id: str
    package_identity: str
    repository_id: str
    repository_state_cid: str
    configuration_id: str
    configuration_cid: str
    evidence_kind: str
    tasks: tuple[SelfHostingTask, ...]
    plan_id: str = ""
    schema: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        for name in ("engine_id", "package_id", "package_identity", "repository_id", "repository_state_cid", "configuration_id", "configuration_cid"):
            _required(getattr(self, name), name)
        if self.schema != PLAN_SCHEMA:
            raise ValueError("experiment plan schema is not supported")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError("evidence_kind must be live, replayed, or simulated")
        tasks = tuple(self.tasks)
        if not tasks:
            raise ValueError("tasks must not be empty")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("task_id values must be unique")
        if self.evidence_kind == "replayed" and any(task.replay_record is None for task in tasks):
            raise ValueError("replayed tasks require replay_record")
        object.__setattr__(self, "tasks", tasks)
        calculated = stable_id("self-hosting-experiment", self._identity_mapping())
        if self.plan_id and self.plan_id != calculated:
            raise ValueError("plan_id does not bind the plan contents")
        object.__setattr__(self, "plan_id", calculated)

    def _identity_mapping(self) -> dict[str, Any]:
        return {"schema": self.schema, "engine_id": self.engine_id, "package_id": self.package_id, "package_identity": self.package_identity, "repository_id": self.repository_id, "repository_state_cid": self.repository_state_cid, "configuration_id": self.configuration_id, "configuration_cid": self.configuration_cid, "evidence_kind": self.evidence_kind, "tasks": [task.to_mapping() for task in self.tasks]}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentPlan":
        tasks = raw.get("tasks")
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            raise ValueError("tasks must be an array")
        return cls(
            schema=str(raw.get("schema", PLAN_SCHEMA)), plan_id=str(raw.get("plan_id", "")),
            engine_id=_required(raw.get("engine_id"), "engine_id"), package_id=_required(raw.get("package_id"), "package_id"), package_identity=_required(raw.get("package_identity"), "package_identity"), repository_id=_required(raw.get("repository_id"), "repository_id"), repository_state_cid=_required(raw.get("repository_state_cid"), "repository_state_cid"), configuration_id=_required(raw.get("configuration_id"), "configuration_id"), configuration_cid=_required(raw.get("configuration_cid"), "configuration_cid"), evidence_kind=_required(raw.get("evidence_kind"), "evidence_kind"), tasks=tuple(SelfHostingTask.from_mapping(_mapping(item, "task")) for item in tasks),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {**self._identity_mapping(), "plan_id": self.plan_id}
