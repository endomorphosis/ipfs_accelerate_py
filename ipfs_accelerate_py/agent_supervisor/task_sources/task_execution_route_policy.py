"""Immutable launch policy for per-task implementation routing.

The policy is deliberately small and data-only.  It is sealed from one
generation-stable typed task projection, transported to a managed executor on
its inherited owner bootstrap channel, and then matched against the exact task
revision before a shared claim is taken.  It is not a task-board projection,
provider configuration, or ambient process setting.
"""

from __future__ import annotations

import hashlib
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
DETERMINISTIC_ONLY_EXECUTION_MODE: Final = "deterministic-only"
GROK_CODEX_EXECUTION_MODE: Final = "grok-codex"
TASK_EXECUTION_ROUTE_MODES: Final = frozenset(
    {DETERMINISTIC_ONLY_EXECUTION_MODE, GROK_CODEX_EXECUTION_MODE}
)
POST_MERGE_RETRY_RECOVERY_OPERATIONS: Final = frozenset(
    {
        "database_post_merge_declared_outputs_repair_recovery",
        "database_post_merge_declared_outputs_requalification_recovery",
        "database_post_merge_declared_outputs_callback_integration_recovery",
    }
)
EXECUTION_ROUTE_RECEIPT_FIELDS: Final = frozenset(
    {
        "execution_route_binding",
        "execution_route_policy_id",
        "execution_route_origin_revision",
    }
)
VIRGIN_TASK_TRANSFER_RECEIPT_FIELDS: Final = frozenset(
    {"virgin_task_transfer", "virgin_task_transfer_claim_cursor"}
)
MAX_TASK_EXECUTION_ROUTE_ENTRIES: Final = 1_000
MAX_TASK_EXECUTION_ROUTE_POLICY_BYTES: Final = 49_152
_CONTENT_ID = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:)?[^\s]{1,4096}$")
_TASK_ID = re.compile(r"^[^\s]{1,1024}$")


def _required_text(value: Any, *, noun: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise TaskSourceIntegrityError(f"{noun} is missing or noncanonical")
    return value


def task_execution_contract_cid(task: TaskRecord) -> str:
    """Identify immutable task semantics while excluding operational status."""

    if not isinstance(task, TaskRecord):
        raise TaskSourceIntegrityError("execution route task is not canonical")
    body = {
        str(key): value
        for key, value in task.body.items()
        if str(key).strip().lower().replace("_", " ")
        not in {"status", "completion receipt"}
    }
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


def resolve_post_merge_retry_predecessor_lineage(
    task: TaskRecord,
    revisions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Recover only the exact lineage dropped by the legacy retry writer.

    This is a compatibility verifier, not a generic history search.  It
    accepts one retrying post-merge recovery head whose immediate blocked
    predecessor carries a complete execution route (and, optionally, the
    complete virgin-transfer pair).  The current head, predecessor, recovery
    seed, immutable task body, and route contract all have to agree exactly.
    """

    if not isinstance(task, TaskRecord) or task.status != "retrying":
        raise TaskSourceIntegrityError(
            "post-merge route recovery requires one retrying task"
        )
    body = task.body if isinstance(task.body, Mapping) else {}
    receipt = body.get("completion_receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("operation") not in POST_MERGE_RETRY_RECOVERY_OPERATIONS
        or EXECUTION_ROUTE_RECEIPT_FIELDS.intersection(receipt)
    ):
        raise TaskSourceIntegrityError(
            "task is not one route-less post-merge recovery head"
        )
    seed = receipt.get("post_merge_completion_recovery_seed")
    if not isinstance(seed, Mapping):
        raise TaskSourceIntegrityError(
            "route-less post-merge recovery has no exact recovery seed"
        )
    seed_value = dict(seed)
    seed_body = dict(seed_value)
    seed_id = seed_body.pop("seed_id", None)
    seed_schema = seed_value.get("schema")
    seed_fields = {
        "schema",
        "task_cid",
        "task_alias",
        "attempt_id",
        "attempt_number",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "source_task_revision",
        "request_id",
        "candidate_commit",
        "qualified_target_commit",
        "qualification_kind",
        "qualification_receipt_id",
        "queue_source_attempt_id",
        "queue_source_claim_id",
        "queue_source_lease_id",
        "queue_source_fencing_token",
        "queue_source_fence_epoch",
        "queue_source_binding_id",
        "queue_source_projection_immutable_digest",
        "recovery_evidence_id",
        "terminal_reason",
        "seed_id",
    }
    if seed_schema == (
        "ipfs_accelerate_py/agent-supervisor/"
        "database-post-merge-completion-recovery-seed@2"
    ):
        seed_fields.add("recovery_control_revision")
        seed_revision = seed_value.get("recovery_control_revision")
    else:
        seed_revision = seed_value.get("source_task_revision")
    if (
        seed_schema
        not in {
            "ipfs_accelerate_py/agent-supervisor/"
            "database-post-merge-completion-recovery-seed@1",
            "ipfs_accelerate_py/agent-supervisor/"
            "database-post-merge-completion-recovery-seed@2",
        }
        or set(seed_value) != seed_fields
        or seed_id
        != "sha256:"
        + hashlib.sha256(canonical_json_bytes(seed_body)).hexdigest()
        or seed_value.get("task_cid") != task.task_cid
        or seed_value.get("task_alias") != task.task_alias
        or seed_revision != task.revision - 1
        or receipt.get("control_expected_status") != "blocked"
        or receipt.get("control_expected_revision") != task.revision - 1
    ):
        raise TaskSourceIntegrityError(
            "route-less post-merge recovery seed is invalid or stale"
        )
    revision_rows = tuple(revisions)
    if (
        len(revision_rows) != task.revision
        or any(
            not isinstance(row, Mapping)
            or row.get("revision") != index
            for index, row in enumerate(revision_rows, 1)
        )
    ):
        raise TaskSourceIntegrityError(
            "post-merge route recovery history is incomplete or noncanonical"
        )
    predecessor = revision_rows[-2] if len(revision_rows) >= 2 else None
    current = revision_rows[-1] if revision_rows else None
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("status") != "blocked"
        or not isinstance(current, Mapping)
        or current.get("status") != "retrying"
        or current.get("body") != dict(body)
    ):
        raise TaskSourceIntegrityError(
            "post-merge route recovery lacks its exact predecessor/head"
        )
    predecessor_body = predecessor.get("body")
    predecessor_receipt = (
        predecessor_body.get("completion_receipt")
        if isinstance(predecessor_body, Mapping)
        else None
    )
    if (
        not isinstance(predecessor_receipt, Mapping)
        or predecessor_receipt.get("operation")
        != "database_portal_terminal_failure"
        or predecessor_receipt.get("control_expected_status")
        != "in_progress"
        or predecessor_receipt.get("control_expected_revision")
        != task.revision - 2
        or {
            key: value
            for key, value in predecessor_body.items()
            if key != "completion_receipt"
        }
        != {key: value for key, value in body.items() if key != "completion_receipt"}
        or EXECUTION_ROUTE_RECEIPT_FIELDS.intersection(predecessor_receipt)
        != EXECUTION_ROUTE_RECEIPT_FIELDS
    ):
        raise TaskSourceIntegrityError(
            "post-merge route recovery predecessor authority is incomplete"
        )
    identity_fields = (
        "attempt_id",
        "attempt_number",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "execution_revision",
        "execution_finished_at_ms",
    )
    if any(
        type(receipt.get(name)) is not type(predecessor_receipt.get(name))
        or receipt.get(name) != predecessor_receipt.get(name)
        for name in identity_fields
    ):
        raise TaskSourceIntegrityError(
            "post-merge route recovery differs from its failed attempt"
        )
    binding = TaskExecutionRouteBinding.from_dict(
        predecessor_receipt["execution_route_binding"]
    )
    if (
        predecessor_receipt.get("execution_route_policy_id")
        != binding.policy_id
        or predecessor_receipt.get("execution_route_origin_revision")
        != binding.task_revision
        or binding.task_cid != task.task_cid
        or binding.task_alias != task.task_alias
        or binding.task_revision >= task.revision
        or binding.task_contract_cid != task_execution_contract_cid(task)
    ):
        raise TaskSourceIntegrityError(
            "post-merge route recovery predecessor binding is invalid"
        )
    transfer_fields = VIRGIN_TASK_TRANSFER_RECEIPT_FIELDS.intersection(
        predecessor_receipt
    )
    if transfer_fields not in (set(), VIRGIN_TASK_TRANSFER_RECEIPT_FIELDS):
        raise TaskSourceIntegrityError(
            "post-merge route recovery predecessor transfer is partial"
        )
    lineage = {
        "execution_route_binding": binding.to_dict(),
        "execution_route_policy_id": binding.policy_id,
        "execution_route_origin_revision": binding.task_revision,
    }
    if transfer_fields:
        transfer = predecessor_receipt.get("virgin_task_transfer")
        cursor = predecessor_receipt.get("virgin_task_transfer_claim_cursor")
        if (
            not isinstance(transfer, Mapping)
            or not isinstance(cursor, Mapping)
            or not str(transfer.get("binding_id") or "")
            or cursor.get("binding_id") != transfer.get("binding_id")
        ):
            raise TaskSourceIntegrityError(
                "post-merge route recovery predecessor transfer is invalid"
            )
        lineage.update(
            {
                "virgin_task_transfer": dict(transfer),
                "virgin_task_transfer_claim_cursor": dict(cursor),
            }
        )
    return MappingProxyType(lineage)


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
    "TaskExecutionRouteBinding",
    "TaskExecutionRouteEntry",
    "TaskExecutionRoutePolicy",
    "task_execution_contract_cid",
]
