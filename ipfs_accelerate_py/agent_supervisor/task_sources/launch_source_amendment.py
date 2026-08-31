"""Append-only current-source amendment for an immutable task-plan lineage.

Long-running configured boards preserve task CIDs, lifecycle history, and
completion receipts while accepted control-plane commits advance the source
tree.  The historical task execution-route policy therefore remains immutable;
it is not the source preimage authority for a later attempt.

``LaunchSourceAmendment@1`` records that second identity.  The operator appends
the closed, content-addressed record through ``PlanRevisionRepository@1`` and a
daemon independently reads it through the named Quack query surface before it
can create a provider-capable Portal.  Filesystem copies are diagnostics only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .control_plane_contracts import canonical_json_bytes, content_identity

LAUNCH_SOURCE_AMENDMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/launch-source-amendment@1"
)
LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "semantic-preserving-remodularization-launch-source-forest@1"
)
LAUNCH_SOURCE_TASK_HISTORY_POLICY: Final[str] = "preserve_immutable_task_cids_and_receipts"
LAUNCH_SOURCE_ATTEMPT_POLICY: Final[str] = "exact_launch_forest_and_worktree_preimage"
LAUNCH_SOURCE_COMPLETION_POLICY: Final[str] = "current_tree_requalification_required"
ATTEMPT_SOURCE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/attempt-source-policy@1"
)
TASK_CONTRACT_SET_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/task-contract-set@1"
MAX_LAUNCH_SOURCE_AMENDMENT_BYTES: Final[int] = 16_384

_GIT_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_CONTENT_ID = re.compile(r"(?:sha256:[0-9a-f]{64}|bagu[a-z2-7]{20,})")
_BOARD_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_PLAN_ALIAS = re.compile(r"[^\s\x00]{1,256}")


class LaunchSourceAmendmentError(ValueError):
    """A launch-source amendment is malformed or inconsistent."""


def _required_text(value: object, *, field: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise LaunchSourceAmendmentError(f"launch source {field} is invalid")
    return value


def _closed_json_loads(value: str, *, noun: str) -> Any:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=closed_object)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LaunchSourceAmendmentError(f"{noun} is malformed or ambiguous") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise LaunchSourceAmendmentError("launch source JSON keys must be strings")
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    raise LaunchSourceAmendmentError(
        f"launch source JSON contains unsupported {type(value).__name__}"
    )


def _sha256_identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def task_contract_set_cid(tasks: Iterable[Any]) -> str:
    """Identify the immutable execution contract of one exact task population."""

    from .database_task_source import TaskRecord
    from .task_execution_route_policy import task_execution_contract_cid

    records: list[dict[str, str]] = []
    for task in tasks:
        if not isinstance(task, TaskRecord):
            raise LaunchSourceAmendmentError("task contract set contains a noncanonical task")
        records.append(
            {
                "task_cid": task.task_cid,
                "task_alias": task.task_alias,
                "task_execution_contract_cid": task_execution_contract_cid(task),
            }
        )
    records.sort(key=lambda item: (item["task_cid"], item["task_alias"]))
    if (
        not records
        or len({item["task_cid"] for item in records}) != len(records)
        or len({item["task_alias"] for item in records}) != len(records)
    ):
        raise LaunchSourceAmendmentError(
            "task contract set is empty or contains duplicate identities"
        )
    return content_identity({"schema": TASK_CONTRACT_SET_SCHEMA, "tasks": records})


def task_contract_set_cid_from_route_entries(entries: Iterable[Any]) -> str:
    """Identify the sealed launch contract, ignoring later operational bodies."""

    records: list[dict[str, str]] = []
    for entry in entries:
        task_cid = str(getattr(entry, "task_cid", "") or "")
        task_alias = str(getattr(entry, "task_alias", "") or "")
        contract_cid = str(getattr(entry, "task_contract_cid", "") or "")
        if not task_cid or not task_alias or not contract_cid:
            raise LaunchSourceAmendmentError(
                "task contract set contains a noncanonical route entry"
            )
        records.append(
            {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "task_execution_contract_cid": contract_cid,
            }
        )
    records.sort(key=lambda item: (item["task_cid"], item["task_alias"]))
    if (
        not records
        or len({item["task_cid"] for item in records}) != len(records)
        or len({item["task_alias"] for item in records}) != len(records)
    ):
        raise LaunchSourceAmendmentError(
            "task contract set is empty or contains duplicate identities"
        )
    return content_identity({"schema": TASK_CONTRACT_SET_SCHEMA, "tasks": records})


@dataclass(frozen=True)
class LaunchSourceAmendment:
    """Closed content-addressed current-source amendment for one plan lineage."""

    board_namespace: str
    plan_alias: str
    bootstrap_receipt_id: str
    bootstrap_plan_root_cid: str
    bootstrap_source_head: str
    bootstrap_repository_tree_id: str
    launch_source_forest_receipt_id: str
    launch_source_forest_root: str
    launch_source_forest_receipt: Mapping[str, Any]
    launch_source_head: str
    launch_repository_tree_id: str
    immutable_objectives_cid: str
    immutable_plan_cid: str
    immutable_taskboard_cid: str
    immutable_validator_cid: str
    bootstrap_config_cid: str
    launch_config_cid: str
    dependency_seal_cid: str
    task_contract_set_cid: str
    parent_plan_revision: int
    amended_plan_revision: int
    predecessor_amendment_id: str | None = None
    task_history_policy: str = LAUNCH_SOURCE_TASK_HISTORY_POLICY
    attempt_source_policy: str = LAUNCH_SOURCE_ATTEMPT_POLICY
    completion_policy: str = LAUNCH_SOURCE_COMPLETION_POLICY
    amendment_id: str = ""

    def __post_init__(self) -> None:
        _required_text(
            self.board_namespace,
            field="board namespace",
            pattern=_BOARD_NAMESPACE,
        )
        _required_text(self.plan_alias, field="plan alias", pattern=_PLAN_ALIAS)
        for name in (
            "bootstrap_receipt_id",
            "bootstrap_plan_root_cid",
            "launch_source_forest_receipt_id",
            "launch_source_forest_root",
            "immutable_objectives_cid",
            "immutable_plan_cid",
            "immutable_taskboard_cid",
            "immutable_validator_cid",
            "bootstrap_config_cid",
            "launch_config_cid",
            "dependency_seal_cid",
            "task_contract_set_cid",
        ):
            _required_text(
                getattr(self, name),
                field=name.replace("_", " "),
                pattern=_CONTENT_ID,
            )
        if self.predecessor_amendment_id is not None:
            _required_text(
                self.predecessor_amendment_id,
                field="predecessor amendment id",
                pattern=_CONTENT_ID,
            )
        for name in (
            "bootstrap_source_head",
            "bootstrap_repository_tree_id",
            "launch_source_head",
            "launch_repository_tree_id",
        ):
            _required_text(
                getattr(self, name),
                field=name.replace("_", " "),
                pattern=_GIT_OID,
            )
        if (
            isinstance(self.parent_plan_revision, bool)
            or not isinstance(self.parent_plan_revision, int)
            or self.parent_plan_revision < 1
            or isinstance(self.amended_plan_revision, bool)
            or not isinstance(self.amended_plan_revision, int)
            or self.amended_plan_revision != self.parent_plan_revision + 1
        ):
            raise LaunchSourceAmendmentError("launch source plan revision transition is invalid")
        forest_receipt = _plain_json(self.launch_source_forest_receipt)
        forest_fields = {
            "schema",
            "source_head",
            "repository_tree",
            "source_forest_root",
            "source_forest",
            "receipt_id",
        }
        if not isinstance(forest_receipt, dict) or set(forest_receipt) != forest_fields:
            raise LaunchSourceAmendmentError(
                "launch source forest receipt differs from its closed schema"
            )
        receipt_body = dict(forest_receipt)
        receipt_id = str(receipt_body.pop("receipt_id", "") or "")
        source_forest = forest_receipt.get("source_forest")
        source_forest_fields = {
            "source_head",
            "nested_repositories",
            "cross_repository_writes",
            "source_forest_root",
        }
        nested_repository_fields = {
            "repository",
            "path",
            "head",
            "tree",
            "planning_revision",
            "planning_revision_is_ancestor",
            "access",
        }
        source_forest_body = dict(source_forest) if isinstance(source_forest, Mapping) else {}
        claimed_forest_root = source_forest_body.pop("source_forest_root", None)
        nested_repositories = source_forest_body.get("nested_repositories")
        if (
            receipt_id != self.launch_source_forest_receipt_id
            or forest_receipt.get("schema") != LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA
            or _sha256_identity(receipt_body) != receipt_id
            or forest_receipt.get("source_head") != self.launch_source_head
            or forest_receipt.get("repository_tree") != self.launch_repository_tree_id
            or forest_receipt.get("source_forest_root") != self.launch_source_forest_root
            or not isinstance(source_forest, Mapping)
            or set(source_forest) != source_forest_fields
            or source_forest.get("source_head") != self.launch_source_head
            or source_forest.get("source_forest_root") != self.launch_source_forest_root
            or claimed_forest_root != self.launch_source_forest_root
            or _sha256_identity(source_forest_body) != claimed_forest_root
            or source_forest.get("cross_repository_writes") is not False
            or not isinstance(nested_repositories, list)
            or any(
                not isinstance(item, Mapping) or set(item) != nested_repository_fields
                for item in nested_repositories
            )
        ):
            raise LaunchSourceAmendmentError(
                "launch source forest receipt identity or binding is invalid"
            )
        object.__setattr__(
            self,
            "launch_source_forest_receipt",
            _freeze_json(forest_receipt),
        )
        if self.task_history_policy != LAUNCH_SOURCE_TASK_HISTORY_POLICY:
            raise LaunchSourceAmendmentError("launch source task-history policy is unknown")
        if self.attempt_source_policy != LAUNCH_SOURCE_ATTEMPT_POLICY:
            raise LaunchSourceAmendmentError("launch source attempt policy is unknown")
        if self.completion_policy != LAUNCH_SOURCE_COMPLETION_POLICY:
            raise LaunchSourceAmendmentError("launch source completion policy is unknown")
        expected = content_identity(self._body())
        if not self.amendment_id:
            object.__setattr__(self, "amendment_id", expected)
        elif self.amendment_id != expected:
            raise LaunchSourceAmendmentError("launch source amendment identity does not rehash")
        if len(canonical_json_bytes(self.to_dict())) > (MAX_LAUNCH_SOURCE_AMENDMENT_BYTES):
            raise LaunchSourceAmendmentError("launch source amendment exceeds its byte bound")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LAUNCH_SOURCE_AMENDMENT_SCHEMA,
            "board_namespace": self.board_namespace,
            "plan_alias": self.plan_alias,
            "bootstrap_receipt_id": self.bootstrap_receipt_id,
            "bootstrap_plan_root_cid": self.bootstrap_plan_root_cid,
            "bootstrap_source_head": self.bootstrap_source_head,
            "bootstrap_repository_tree_id": self.bootstrap_repository_tree_id,
            "launch_source_forest_receipt_id": (self.launch_source_forest_receipt_id),
            "launch_source_forest_root": self.launch_source_forest_root,
            "launch_source_forest_receipt": _plain_json(self.launch_source_forest_receipt),
            "launch_source_head": self.launch_source_head,
            "launch_repository_tree_id": self.launch_repository_tree_id,
            "immutable_objectives_cid": self.immutable_objectives_cid,
            "immutable_plan_cid": self.immutable_plan_cid,
            "immutable_taskboard_cid": self.immutable_taskboard_cid,
            "immutable_validator_cid": self.immutable_validator_cid,
            "bootstrap_config_cid": self.bootstrap_config_cid,
            "launch_config_cid": self.launch_config_cid,
            "dependency_seal_cid": self.dependency_seal_cid,
            "task_contract_set_cid": self.task_contract_set_cid,
            "parent_plan_revision": self.parent_plan_revision,
            "amended_plan_revision": self.amended_plan_revision,
            "predecessor_amendment_id": self.predecessor_amendment_id,
            "task_history_policy": self.task_history_policy,
            "attempt_source_policy": self.attempt_source_policy,
            "completion_policy": self.completion_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "amendment_id": self.amendment_id}

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LaunchSourceAmendment:
        fields = {
            "schema",
            "board_namespace",
            "plan_alias",
            "bootstrap_receipt_id",
            "bootstrap_plan_root_cid",
            "bootstrap_source_head",
            "bootstrap_repository_tree_id",
            "launch_source_forest_receipt_id",
            "launch_source_forest_root",
            "launch_source_forest_receipt",
            "launch_source_head",
            "launch_repository_tree_id",
            "immutable_objectives_cid",
            "immutable_plan_cid",
            "immutable_taskboard_cid",
            "immutable_validator_cid",
            "bootstrap_config_cid",
            "launch_config_cid",
            "dependency_seal_cid",
            "task_contract_set_cid",
            "parent_plan_revision",
            "amended_plan_revision",
            "predecessor_amendment_id",
            "task_history_policy",
            "attempt_source_policy",
            "completion_policy",
            "amendment_id",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != LAUNCH_SOURCE_AMENDMENT_SCHEMA
        ):
            raise LaunchSourceAmendmentError(
                "launch source amendment differs from its closed schema"
            )
        return cls(**{name: value[name] for name in fields - {"schema"}})

    @classmethod
    def from_json(cls, value: str) -> LaunchSourceAmendment:
        if (
            type(value) is not str
            or not value
            or len(value.encode("utf-8")) > (MAX_LAUNCH_SOURCE_AMENDMENT_BYTES)
        ):
            raise LaunchSourceAmendmentError("launch source amendment JSON is absent or oversized")
        try:
            payload = _closed_json_loads(value, noun="launch source amendment JSON")
        except LaunchSourceAmendmentError:
            raise
        return cls.from_dict(payload)

    def validate_launch_git(
        self,
        *,
        source_head: str,
        repository_tree_id: str,
    ) -> None:
        if (
            source_head != self.launch_source_head
            or repository_tree_id != self.launch_repository_tree_id
        ):
            raise LaunchSourceAmendmentError(
                "launch Git generation differs from its source amendment"
            )

    def successor_for_current_generation(
        self,
        *,
        source_head: str,
        repository_tree_id: str,
        nested_repositories: Sequence[Mapping[str, Any]] | None = None,
    ) -> LaunchSourceAmendment:
        """Mint the next amendment for an accepted descendant Git generation.

        Used when merged work advances HEAD after launch.  Callers must prove
        ``source_head`` descends from ``launch_source_head`` before invoking.
        Immutable bootstrap/plan/taskboard identities are preserved.
        """

        _required_text(source_head, field="launch source head", pattern=_GIT_OID)
        _required_text(
            repository_tree_id,
            field="launch repository tree id",
            pattern=_GIT_OID,
        )
        if (
            source_head == self.launch_source_head
            and repository_tree_id == self.launch_repository_tree_id
        ):
            return self
        current_forest = _plain_json(self.launch_source_forest_receipt)
        current_source_forest = current_forest.get("source_forest")
        if not isinstance(current_source_forest, Mapping):
            raise LaunchSourceAmendmentError(
                "launch source forest is missing from its successor preimage"
            )
        if nested_repositories is None:
            nested = _plain_json(current_source_forest.get("nested_repositories") or [])
        else:
            nested = _plain_json(list(nested_repositories))
        source_forest_body = {
            "source_head": source_head,
            "nested_repositories": nested,
            "cross_repository_writes": False,
        }
        source_forest_root = _sha256_identity(source_forest_body)
        source_forest = {
            **source_forest_body,
            "source_forest_root": source_forest_root,
        }
        receipt_body = {
            "schema": LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA,
            "source_head": source_head,
            "repository_tree": repository_tree_id,
            "source_forest_root": source_forest_root,
            "source_forest": source_forest,
        }
        receipt_id = _sha256_identity(receipt_body)
        return LaunchSourceAmendment(
            board_namespace=self.board_namespace,
            plan_alias=self.plan_alias,
            bootstrap_receipt_id=self.bootstrap_receipt_id,
            bootstrap_plan_root_cid=self.bootstrap_plan_root_cid,
            bootstrap_source_head=self.bootstrap_source_head,
            bootstrap_repository_tree_id=self.bootstrap_repository_tree_id,
            launch_source_forest_receipt_id=receipt_id,
            launch_source_forest_root=source_forest_root,
            launch_source_forest_receipt={**receipt_body, "receipt_id": receipt_id},
            launch_source_head=source_head,
            launch_repository_tree_id=repository_tree_id,
            immutable_objectives_cid=self.immutable_objectives_cid,
            immutable_plan_cid=self.immutable_plan_cid,
            immutable_taskboard_cid=self.immutable_taskboard_cid,
            immutable_validator_cid=self.immutable_validator_cid,
            bootstrap_config_cid=self.bootstrap_config_cid,
            launch_config_cid=self.launch_config_cid,
            dependency_seal_cid=self.dependency_seal_cid,
            task_contract_set_cid=self.task_contract_set_cid,
            parent_plan_revision=self.amended_plan_revision,
            amended_plan_revision=self.amended_plan_revision + 1,
            predecessor_amendment_id=self.amendment_id,
        )

    def same_launch_generation(self, other: LaunchSourceAmendment) -> bool:
        """Compare launch material while deliberately ignoring chain position."""

        if type(other) is not LaunchSourceAmendment:
            return False
        names = set(self._body()) - {
            "predecessor_amendment_id",
            "parent_plan_revision",
            "amended_plan_revision",
        }
        left = self._body()
        right = other._body()
        return all(left[name] == right[name] for name in names)

    def attempt_policy_root(self, route_binding: Mapping[str, Any]) -> str:
        """Compose current source with one historical task route binding."""
        from .database_task_source import TaskSourceIntegrityError
        from .task_execution_route_policy import TaskExecutionRouteBinding

        try:
            binding = TaskExecutionRouteBinding.from_dict(route_binding)
        except TaskSourceIntegrityError as exc:
            raise LaunchSourceAmendmentError(
                "attempt execution-route binding differs from its closed schema"
            ) from exc
        if (
            binding.plan_root_cid != self.bootstrap_plan_root_cid
            or binding.repository_tree_id != self.bootstrap_repository_tree_id
        ):
            raise LaunchSourceAmendmentError(
                "attempt execution route differs from the amendment plan lineage"
            )
        canonical_binding = binding.to_dict()
        return content_identity(
            {
                "schema": ATTEMPT_SOURCE_POLICY_SCHEMA,
                "launch_source_amendment_id": self.amendment_id,
                "launch_source_forest_root": self.launch_source_forest_root,
                "launch_source_head": self.launch_source_head,
                "launch_repository_tree_id": self.launch_repository_tree_id,
                "execution_route_binding_id": content_identity(canonical_binding),
                "execution_route_policy_id": binding.policy_id,
                "task_cid": binding.task_cid,
                "task_alias": binding.task_alias,
                "task_contract_cid": binding.task_contract_cid,
            }
        )


__all__ = [
    "ATTEMPT_SOURCE_POLICY_SCHEMA",
    "LAUNCH_SOURCE_AMENDMENT_SCHEMA",
    "LAUNCH_SOURCE_ATTEMPT_POLICY",
    "LAUNCH_SOURCE_COMPLETION_POLICY",
    "LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA",
    "LAUNCH_SOURCE_TASK_HISTORY_POLICY",
    "MAX_LAUNCH_SOURCE_AMENDMENT_BYTES",
    "TASK_CONTRACT_SET_SCHEMA",
    "LaunchSourceAmendment",
    "LaunchSourceAmendmentError",
    "task_contract_set_cid",
    "task_contract_set_cid_from_route_entries",
]
