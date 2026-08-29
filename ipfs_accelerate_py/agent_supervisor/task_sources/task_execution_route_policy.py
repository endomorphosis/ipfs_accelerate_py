"""Immutable launch policy for per-task implementation routing.

The policy is deliberately small and data-only.  It is sealed from one
generation-stable typed task projection, transported to a managed executor on
its inherited owner bootstrap channel, and then matched against the exact task
revision before a shared claim is taken.  It is not a task-board projection,
provider configuration, or ambient process setting.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .control_plane_contracts import canonical_json_bytes, content_identity
from .database_task_source import (
    TaskRecord,
    TaskSourceBoundsError,
    TaskSourceIntegrityError,
    TaskSourceSnapshot,
)

TASK_EXECUTION_ROUTE_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-execution-route-policy@1"
)
TASK_EXECUTION_ROUTE_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-execution-route-binding@1"
)
TASK_EXECUTION_ROUTE_SUMMARY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-execution-route-summary@1"
)
TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_FIELD: Final = (
    "fresh_portal_revalidation_requirement"
)
TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "typed-database-blocked-retry-fresh-portal-revalidation@1"
)
TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_OPERATION: Final = (
    "database_operator_blocked_retry_fresh_portal_revalidation_required"
)
DETERMINISTIC_ONLY_EXECUTION_MODE: Final = "deterministic-only"
GROK_CODEX_EXECUTION_MODE: Final = "grok-codex"
TASK_EXECUTION_ROUTE_MODES: Final = frozenset(
    {DETERMINISTIC_ONLY_EXECUTION_MODE, GROK_CODEX_EXECUTION_MODE}
)
MAX_TASK_EXECUTION_ROUTE_ENTRIES: Final = 1_000
MAX_TASK_EXECUTION_ROUTE_POLICY_BYTES: Final = 49_152
_CONTENT_ID = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:)?[^\s]{1,4096}$")
_TASK_ID = re.compile(r"^[^\s]{1,1024}$")
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def _required_text(value: Any, *, noun: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TaskSourceIntegrityError(f"{noun} is missing or noncanonical")
    return value


def validated_typed_database_blocked_retry_revalidation_requirement(
    value: Any,
    *,
    task_cid: str,
) -> dict[str, Any]:
    """Validate the sole operational marker excluded from task semantics."""

    expected_fields = {
        "schema",
        "operation",
        "task_cid",
        "source_completion_receipt_id",
        "operator_handoff_receipt_id",
        "sidecar_evidence_id",
        "recovered_from_revision",
        "fresh_attempt_number",
        "requirement_id",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise TaskSourceIntegrityError(
            "fresh Portal revalidation requirement differs from its schema"
        )
    requirement = dict(value)
    requirement_id = requirement.pop("requirement_id", None)
    if (
        value.get("schema")
        != TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_SCHEMA
        or value.get("operation")
        != TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_OPERATION
        or value.get("task_cid") != task_cid
        or any(
            type(value.get(name)) is not str
            or _SHA256_ID.fullmatch(value[name]) is None
            for name in (
                "source_completion_receipt_id",
                "operator_handoff_receipt_id",
                "sidecar_evidence_id",
            )
        )
        or any(
            type(value.get(name)) is not int or int(value[name]) < 1
            for name in (
                "recovered_from_revision",
                "fresh_attempt_number",
            )
        )
        or requirement_id
        != content_identity(
            {"fresh_portal_revalidation_requirement": requirement}
        )
    ):
        raise TaskSourceIntegrityError(
            "fresh Portal revalidation requirement is malformed"
        )
    return dict(value)


def typed_database_blocked_retry_revalidation_requirement(
    *,
    task_cid: str,
    source_completion_receipt_id: str,
    operator_handoff_receipt_id: str,
    sidecar_evidence_id: str,
    recovered_from_revision: int,
    fresh_attempt_number: int,
) -> dict[str, Any]:
    """Seal the durable task-body marker requiring a fresh Portal proof."""

    requirement = {
        "schema": TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_SCHEMA,
        "operation": TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_OPERATION,
        "task_cid": task_cid,
        "source_completion_receipt_id": source_completion_receipt_id,
        "operator_handoff_receipt_id": operator_handoff_receipt_id,
        "sidecar_evidence_id": sidecar_evidence_id,
        "recovered_from_revision": recovered_from_revision,
        "fresh_attempt_number": fresh_attempt_number,
    }
    requirement["requirement_id"] = content_identity(
        {"fresh_portal_revalidation_requirement": requirement}
    )
    return validated_typed_database_blocked_retry_revalidation_requirement(
        requirement,
        task_cid=task_cid,
    )


def task_execution_contract_cid(task: TaskRecord) -> str:
    """Identify immutable task semantics while excluding operational status."""

    if not isinstance(task, TaskRecord):
        raise TaskSourceIntegrityError("execution route task is not canonical")
    body: dict[str, Any] = {}
    for key, value in task.body.items():
        selected = str(key)
        normalized = selected.strip().lower().replace("_", " ")
        if normalized in {"status", "completion receipt"}:
            continue
        if selected == TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_FIELD:
            validated_typed_database_blocked_retry_revalidation_requirement(
                value,
                task_cid=task.task_cid,
            )
            continue
        body[selected] = value
    return content_identity(
        {
            "task_cid": task.task_cid,
            "task_alias": task.task_alias,
            "goal_cid": task.goal_cid,
            "plan_cid": task.plan_cid,
            "objective_id": task.objective_id,
            "ordinal": int(task.ordinal),
            "priority": task.priority,
            "body": body,
            "dependencies": list(task.dependencies),
            "outputs": [dict(item) for item in task.outputs],
            "acceptance": [dict(item) for item in task.acceptance],
            "validations": [dict(item) for item in task.validations],
        }
    )


@dataclass(frozen=True)
class TaskExecutionRouteEntry:
    """One exact task revision and its closed execution mode."""

    task_cid: str
    task_alias: str
    task_revision: int
    task_contract_cid: str
    execution_mode: str

    def __post_init__(self) -> None:
        _required_text(self.task_cid, noun="route task CID", pattern=_TASK_ID)
        _required_text(self.task_alias, noun="route task alias", pattern=_TASK_ID)
        if (
            isinstance(self.task_revision, bool)
            or not isinstance(self.task_revision, int)
            or self.task_revision < 1
        ):
            raise TaskSourceIntegrityError("route task revision is invalid")
        if self.execution_mode not in TASK_EXECUTION_ROUTE_MODES:
            raise TaskSourceIntegrityError("task execution route mode is unknown")
        _required_text(
            self.task_contract_cid,
            noun="route task contract CID",
            pattern=_CONTENT_ID,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_cid": self.task_cid,
            "task_alias": self.task_alias,
            "task_revision": int(self.task_revision),
            "task_contract_cid": self.task_contract_cid,
            "execution_mode": self.execution_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskExecutionRouteEntry:
        fields = {
            "task_cid",
            "task_alias",
            "task_revision",
            "task_contract_cid",
            "execution_mode",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise TaskSourceIntegrityError(
                "task execution route entry differs from its closed schema"
            )
        return cls(
            task_cid=value["task_cid"],
            task_alias=value["task_alias"],
            task_revision=value["task_revision"],
            task_contract_cid=value["task_contract_cid"],
            execution_mode=value["execution_mode"],
        )


@dataclass(frozen=True)
class TaskExecutionRouteBinding:
    """Attempt-carried proof of one entry in a launch policy."""

    policy_id: str
    plan_root_cid: str
    repository_tree_id: str
    source_revision: int
    task_cid: str
    task_alias: str
    task_revision: int
    task_contract_cid: str
    execution_mode: str

    def __post_init__(self) -> None:
        _required_text(self.policy_id, noun="route policy ID", pattern=_CONTENT_ID)
        _required_text(
            self.plan_root_cid, noun="route plan root CID", pattern=_CONTENT_ID
        )
        _required_text(
            self.repository_tree_id,
            noun="route repository tree ID",
            pattern=_CONTENT_ID,
        )
        if (
            isinstance(self.source_revision, bool)
            or not isinstance(self.source_revision, int)
            or self.source_revision < 1
        ):
            raise TaskSourceIntegrityError("route source revision is invalid")
        TaskExecutionRouteEntry(
            task_cid=self.task_cid,
            task_alias=self.task_alias,
            task_revision=self.task_revision,
            task_contract_cid=self.task_contract_cid,
            execution_mode=self.execution_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_EXECUTION_ROUTE_BINDING_SCHEMA,
            "policy_id": self.policy_id,
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "source_revision": int(self.source_revision),
            "task_cid": self.task_cid,
            "task_alias": self.task_alias,
            "task_revision": int(self.task_revision),
            "task_contract_cid": self.task_contract_cid,
            "execution_mode": self.execution_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskExecutionRouteBinding:
        fields = {
            "schema",
            "policy_id",
            "plan_root_cid",
            "repository_tree_id",
            "source_revision",
            "task_cid",
            "task_alias",
            "task_revision",
            "task_contract_cid",
            "execution_mode",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != TASK_EXECUTION_ROUTE_BINDING_SCHEMA
        ):
            raise TaskSourceIntegrityError(
                "task execution route binding differs from its closed schema"
            )
        return cls(
            policy_id=value["policy_id"],
            plan_root_cid=value["plan_root_cid"],
            repository_tree_id=value["repository_tree_id"],
            source_revision=value["source_revision"],
            task_cid=value["task_cid"],
            task_alias=value["task_alias"],
            task_revision=value["task_revision"],
            task_contract_cid=value["task_contract_cid"],
            execution_mode=value["execution_mode"],
        )


@dataclass(frozen=True)
class TaskExecutionRoutePolicy:
    """One immutable plan-root policy over an exact task population."""

    plan_root_cid: str
    repository_tree_id: str
    source_revision: int
    source_projection_cid: str
    entries: tuple[TaskExecutionRouteEntry, ...]
    policy_id: str

    def __post_init__(self) -> None:
        _required_text(
            self.plan_root_cid, noun="route plan root CID", pattern=_CONTENT_ID
        )
        _required_text(
            self.repository_tree_id,
            noun="route repository tree ID",
            pattern=_CONTENT_ID,
        )
        _required_text(
            self.source_projection_cid,
            noun="route source projection CID",
            pattern=_CONTENT_ID,
        )
        _required_text(self.policy_id, noun="route policy ID", pattern=_CONTENT_ID)
        if (
            isinstance(self.source_revision, bool)
            or not isinstance(self.source_revision, int)
            or self.source_revision < 1
        ):
            raise TaskSourceIntegrityError("route source revision is invalid")
        if not self.entries or len(self.entries) > MAX_TASK_EXECUTION_ROUTE_ENTRIES:
            raise TaskSourceBoundsError("task execution route population is invalid")
        cids = [entry.task_cid for entry in self.entries]
        aliases = [entry.task_alias for entry in self.entries]
        if len(cids) != len(set(cids)) or len(aliases) != len(set(aliases)):
            raise TaskSourceIntegrityError(
                "task execution route population has duplicate identities"
            )
        if tuple(sorted(self.entries, key=lambda entry: entry.task_cid)) != self.entries:
            raise TaskSourceIntegrityError(
                "task execution route entries are not in canonical CID order"
            )
        body = self._body()
        if content_identity(body) != self.policy_id:
            raise TaskSourceIntegrityError("task execution route policy ID is invalid")
        if len(canonical_json_bytes(self.to_dict())) > MAX_TASK_EXECUTION_ROUTE_POLICY_BYTES:
            raise TaskSourceBoundsError("task execution route policy exceeds its bound")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": TASK_EXECUTION_ROUTE_POLICY_SCHEMA,
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "source_revision": int(self.source_revision),
            "source_projection_cid": self.source_projection_cid,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "policy_id": self.policy_id}

    @classmethod
    def seal(
        cls,
        *,
        snapshot: TaskSourceSnapshot,
        tasks: Sequence[TaskRecord],
        execution_modes: Mapping[str, str],
    ) -> TaskExecutionRoutePolicy:
        if not isinstance(snapshot, TaskSourceSnapshot):
            raise TaskSourceIntegrityError(
                "task execution route requires a typed task-source snapshot"
            )
        task_tuple = tuple(tasks)
        aliases = {task.task_alias for task in task_tuple}
        if (
            len(task_tuple) != snapshot.task_count
            or not aliases
            or set(execution_modes) != aliases
        ):
            raise TaskSourceIntegrityError(
                "task execution route does not cover the exact task population"
            )
        if any(task.plan_cid != snapshot.plan_root_cid for task in task_tuple):
            raise TaskSourceIntegrityError(
                "task execution route population differs from its plan root"
            )
        entries = tuple(
            sorted(
                (
                    TaskExecutionRouteEntry(
                        task_cid=task.task_cid,
                        task_alias=task.task_alias,
                        task_revision=int(task.revision),
                        task_contract_cid=task_execution_contract_cid(task),
                        execution_mode=str(execution_modes[task.task_alias]),
                    )
                    for task in task_tuple
                ),
                key=lambda entry: entry.task_cid,
            )
        )
        body = {
            "schema": TASK_EXECUTION_ROUTE_POLICY_SCHEMA,
            "plan_root_cid": snapshot.plan_root_cid,
            "repository_tree_id": snapshot.repository_tree_id,
            "source_revision": int(snapshot.revision),
            "source_projection_cid": snapshot.projection_cid,
            "entries": [entry.to_dict() for entry in entries],
        }
        return cls(
            plan_root_cid=snapshot.plan_root_cid,
            repository_tree_id=snapshot.repository_tree_id,
            source_revision=int(snapshot.revision),
            source_projection_cid=snapshot.projection_cid,
            entries=entries,
            policy_id=content_identity(body),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskExecutionRoutePolicy:
        fields = {
            "schema",
            "plan_root_cid",
            "repository_tree_id",
            "source_revision",
            "source_projection_cid",
            "entries",
            "policy_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != TASK_EXECUTION_ROUTE_POLICY_SCHEMA
            or not isinstance(value.get("entries"), list)
        ):
            raise TaskSourceIntegrityError(
                "task execution route policy differs from its closed schema"
            )
        return cls(
            plan_root_cid=value["plan_root_cid"],
            repository_tree_id=value["repository_tree_id"],
            source_revision=value["source_revision"],
            source_projection_cid=value["source_projection_cid"],
            entries=tuple(
                TaskExecutionRouteEntry.from_dict(entry)
                for entry in value["entries"]
            ),
            policy_id=value["policy_id"],
        )

    @property
    def entries_by_cid(self) -> Mapping[str, TaskExecutionRouteEntry]:
        return MappingProxyType({entry.task_cid: entry for entry in self.entries})

    def public_summary(self) -> dict[str, Any]:
        """Return the bounded, task-detail-free launch/status projection."""

        deterministic_count = sum(
            entry.execution_mode == DETERMINISTIC_ONLY_EXECUTION_MODE
            for entry in self.entries
        )
        model_count = sum(
            entry.execution_mode == GROK_CODEX_EXECUTION_MODE
            for entry in self.entries
        )
        return {
            "schema": TASK_EXECUTION_ROUTE_SUMMARY_SCHEMA,
            "policy_id": self.policy_id,
            "plan_root_cid": self.plan_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "source_revision": int(self.source_revision),
            "task_count": len(self.entries),
            "deterministic_task_count": deterministic_count,
            "model_task_count": model_count,
        }

    def binding_for_task(self, task: TaskRecord) -> TaskExecutionRouteBinding:
        entry = self.entries_by_cid.get(str(getattr(task, "task_cid", "") or ""))
        if entry is None:
            raise TaskSourceIntegrityError("task is absent from the launch route policy")
        if (
            entry.task_alias != str(getattr(task, "task_alias", "") or "")
            or entry.task_revision != int(getattr(task, "revision", 0) or 0)
            or entry.task_contract_cid != task_execution_contract_cid(task)
        ):
            raise TaskSourceIntegrityError(
                "task alias or revision differs from the launch route policy"
            )
        return TaskExecutionRouteBinding(
            policy_id=self.policy_id,
            plan_root_cid=self.plan_root_cid,
            repository_tree_id=self.repository_tree_id,
            source_revision=self.source_revision,
            task_cid=entry.task_cid,
            task_alias=entry.task_alias,
            task_revision=entry.task_revision,
            task_contract_cid=entry.task_contract_cid,
            execution_mode=entry.execution_mode,
        )

    def validate_binding(
        self,
        value: Mapping[str, Any],
    ) -> TaskExecutionRouteBinding:
        binding = TaskExecutionRouteBinding.from_dict(value)
        entry = self.entries_by_cid.get(binding.task_cid)
        expected = (
            TaskExecutionRouteBinding(
                policy_id=self.policy_id,
                plan_root_cid=self.plan_root_cid,
                repository_tree_id=self.repository_tree_id,
                source_revision=self.source_revision,
                task_cid=entry.task_cid,
                task_alias=entry.task_alias,
                task_revision=entry.task_revision,
                task_contract_cid=entry.task_contract_cid,
                execution_mode=entry.execution_mode,
            )
            if entry is not None
            else None
        )
        if expected is None or binding != expected:
            raise TaskSourceIntegrityError(
                "task execution route binding is not in the launch policy"
            )
        return binding


__all__ = [
    "DETERMINISTIC_ONLY_EXECUTION_MODE",
    "GROK_CODEX_EXECUTION_MODE",
    "MAX_TASK_EXECUTION_ROUTE_ENTRIES",
    "MAX_TASK_EXECUTION_ROUTE_POLICY_BYTES",
    "TASK_EXECUTION_ROUTE_BINDING_SCHEMA",
    "TASK_EXECUTION_ROUTE_MODES",
    "TASK_EXECUTION_ROUTE_POLICY_SCHEMA",
    "TASK_EXECUTION_ROUTE_SUMMARY_SCHEMA",
    "TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_FIELD",
    "TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_OPERATION",
    "TYPED_DATABASE_BLOCKED_RETRY_REVALIDATION_SCHEMA",
    "TaskExecutionRouteBinding",
    "TaskExecutionRouteEntry",
    "TaskExecutionRoutePolicy",
    "task_execution_contract_cid",
    "typed_database_blocked_retry_revalidation_requirement",
    "validated_typed_database_blocked_retry_revalidation_requirement",
]
