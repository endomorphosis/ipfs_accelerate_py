"""Signed, effect-bound EAAEF Plan-R2 transition gate.

This module is deliberately not a state repository or another command
protocol.  It authorizes one exact prepare/apply CAS against an independently
qualified Quack command fabric.  The mutable owner must consume and verify the
full authorization; a bare ``StateCommand`` or direct DuckDB connection can
never satisfy this interface.  Process birth is a separate launch-capsule
effect and is explicitly prohibited here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    ed25519_public_key_from_did,
    verify_did_key_signature,
)

PLAN_R2_TRANSITION_STATEMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-plan-r2-transition-authorization-statement@1"
)
PLAN_R2_TRANSITION_APPROVAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-transition-approval@1"
)
PLAN_R2_TRANSITION_AUTHORIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-transition-authorization@1"
)
PLAN_R2_TRANSITION_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-plan-r2-transition-authorization-verification@1"
)
PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-plan-r2-operational-capability@1"
)
PLAN_R2_TRANSITION_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-transition-decision@1"
)
PLAN_R2_PREPARED_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-prepared-projection@1"
)
PLAN_R2_TRANSITION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-transition-receipt@1"
)
PLAN_R2_STATE_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-state-observation@1"
)
AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE: Final = (
    "AuthorizedPlanR2TransitionRepository@1"
)
EAAEF_AUTHORITY_REGISTRY_PREFIX: Final = (
    "data/agent_supervisor/external_agent_autonomous_execution_fabric/authority"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_MAX_FRONTIER = 5
_MAX_TASKS = 512
_MAX_DEPENDENCIES = 4096
_MAX_LAUNCH_READBACK_AGE_MS = 30_000
_PROTECTED_STATUSES = frozenset(
    {"claimed", "running", "settling", "completed", "accepted"}
)
_TASK_STATUSES = frozenset(
    {
        "todo",
        "blocked",
        "claimed",
        "running",
        "settling",
        "completed",
        "accepted",
        "cancelled",
        "superseded",
    }
)
_AUTHORITY = {
    "effect": "atomic_plan_population_cas",
    "prepare_allowed": True,
    "apply_allowed": True,
    "maximum_apply_count": 1,
    "process_birth_allowed": False,
    "direct_duckdb_file_open": False,
    "command_transport": "signed_state_command_envelope",
    "maximum_frontier_tasks": _MAX_FRONTIER,
}
_PLAN_FIELDS = frozenset(
    {
        "plan_cid",
        "plan_alias",
        "plan_root_cid",
        "semantic_root_cid",
        "status",
        "revision",
        "body",
    }
)
_TASK_FIELDS = frozenset(
    {
        "task_cid",
        "task_alias",
        "goal_cid",
        "plan_cid",
        "objective_id",
        "ordinal",
        "status",
        "revision",
        "priority",
        "identity",
        "body",
    }
)
_DEPENDENCY_FIELDS = frozenset(
    {"task_cid", "dependency_task_cid", "kind"}
)
_PROTECTED_FIELDS = frozenset(
    {"task_cid", "status", "revision", "task_row", "task_row_cid"}
)
_STATEMENT_FIELDS = frozenset(
    {
        "schema",
        "board_namespace",
        "source_head",
        "source_tree",
        "source_generation_cid",
        "bootstrap_admission_cid",
        "r1_launch_capsule_cid",
        "quack_owner_qualification_cid",
        "quack_command_fabric_qualification_cid",
        "owner_principal_did",
        "shard_id",
        "store_id",
        "owner_generation",
        "expected_epoch",
        "fencing_token",
        "lease_id",
        "expected_version",
        "expected_active_plan_cid",
        "expected_active_plan_root_cid",
        "expected_active_plan_revision",
        "expected_event_cursor",
        "expected_semantic_root_cid",
        "new_plan",
        "tasks",
        "dependencies",
        "protected_tasks",
        "frontier_task_cids",
        "population_cid",
        "plan_root_cid",
        "task_population_cid",
        "dependency_population_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "delta_cid",
        "request_id",
        "idempotency_key",
        "deadline_ms",
        "issued_at_ms",
        "expires_at_ms",
        "one_use_nonce",
        "authority",
        "statement_cid",
    }
)
_APPROVAL_FIELDS = frozenset(
    {
        "schema",
        "role",
        "identity_did",
        "statement_cid",
        "one_use_nonce",
        "issued_at_ms",
        "expires_at_ms",
        "signature",
    }
)
_APPROVAL_SIGNING_FIELDS = _APPROVAL_FIELDS - {"signature"}
_AUTHORIZATION_FIELDS = frozenset(
    {*_STATEMENT_FIELDS, "operator_approval", "security_approval", "authorization_cid"}
)
_CAPABILITY_FIELDS = frozenset(
    {
        "schema",
        "allowed",
        "blockers",
        "source_head",
        "source_tree",
        "bootstrap_admission_cid",
        "quack_owner_qualification_cid",
        "quack_command_fabric_qualification_cid",
        "owner_principal_did",
        "shard_id",
        "owner_generation",
        "epoch",
        "fence",
        "duckdb_version",
        "quack_build",
        "authorized_state_command_schema",
        "ingress_authenticated",
        "ingress_append_only_single_relation",
        "ingress_accepts_signed_envelope_only",
        "bare_state_command_rejected",
        "owner_verifies_authorized_state_command",
        "authority_ref_binds_transition_authorization",
        "local_owner_verifies_transition_authorization",
        "operational_database_private",
        "one_mutable_owner",
        "atomic_plan_population_cas",
        "egress_read_only",
        "egress_append_denied",
        "durable_idempotent_receipts",
        "protected_full_rows_bound",
        "reviewer_identity_did",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_signature",
        "capability_cid",
    }
)
_CAPABILITY_SIGNING_FIELDS = _CAPABILITY_FIELDS - {
    "reviewer_signature",
    "capability_cid",
}
_CAPABILITY_REQUIRED_TRUE = (
    "ingress_authenticated",
    "ingress_append_only_single_relation",
    "ingress_accepts_signed_envelope_only",
    "bare_state_command_rejected",
    "owner_verifies_authorized_state_command",
    "authority_ref_binds_transition_authorization",
    "local_owner_verifies_transition_authorization",
    "operational_database_private",
    "one_mutable_owner",
    "atomic_plan_population_cas",
    "egress_read_only",
    "egress_append_denied",
    "durable_idempotent_receipts",
    "protected_full_rows_bound",
)
_PREPARED_FIELDS = frozenset(
    {
        "schema",
        "authorization_cid",
        "statement_cid",
        "capability_cid",
        "authorized_prepare_command_cid",
        "source_head",
        "source_tree",
        "shard_id",
        "owner_generation",
        "epoch",
        "fence",
        "before_plan_cid",
        "before_plan_root_cid",
        "before_plan_revision",
        "before_version",
        "before_event_cursor",
        "before_semantic_root_cid",
        "population_cid",
        "plan_root_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "prepared_at_ms",
        "expires_at_ms",
        "authority_mutated",
        "process_started",
        "projection_cid",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "authorization_cid",
        "statement_cid",
        "capability_cid",
        "authorized_prepare_command_cid",
        "authorized_apply_command_cid",
        "prepared_projection_cid",
        "source_head",
        "source_tree",
        "shard_id",
        "owner_generation",
        "epoch",
        "fence",
        "before_plan_cid",
        "after_plan_cid",
        "before_plan_root_cid",
        "after_plan_root_cid",
        "before_plan_revision",
        "after_plan_revision",
        "before_version",
        "after_version",
        "before_event_cursor",
        "after_event_cursor",
        "before_semantic_root_cid",
        "after_semantic_root_cid",
        "population_cid",
        "task_population_cid",
        "dependency_population_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "frontier_task_cids",
        "protected_tasks_unchanged",
        "transaction_cid",
        "replayed",
        "committed_at_ms",
        "receipt_cid",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "authorization_cid",
        "transition_receipt_cid",
        "transaction_cid",
        "authorized_prepare_command_cid",
        "authorized_apply_command_cid",
        "quack_command_fabric_qualification_cid",
        "source_head",
        "source_tree",
        "owner_principal_did",
        "shard_id",
        "owner_generation",
        "epoch",
        "fence",
        "store_version",
        "active_plan_cid",
        "active_plan_root_cid",
        "active_plan_revision",
        "event_cursor",
        "semantic_root_cid",
        "population_cid",
        "task_population_cid",
        "dependency_population_cid",
        "protected_tasks_root_cid",
        "frontier_cid",
        "frontier_task_cids",
        "captured_at_ms",
        "authority_mutated",
        "process_started",
        "observation_cid",
    }
)


class ExternalAgentPlanR2Error(RuntimeError):
    """A Plan-R2 authorization or transition failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExternalAgentPlanR2Error("value is not canonical JSON") from exc


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _positive_int(value: object) -> bool:
    return type(value) is int and int(value) > 0


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and int(value) >= 0


def _require_sha(value: object, noun: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise ExternalAgentPlanR2Error(f"{noun} is not a full sha256 identity")
    return text


def _require_safe_id(value: object, noun: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise ExternalAgentPlanR2Error(f"{noun} is not a bounded identifier")
    return text


def _require_ed25519_did(value: object, noun: str) -> str:
    """Decode one Ed25519 ``did:key`` and keep profile errors inside Plan R2."""

    if not isinstance(value, str):
        raise ExternalAgentPlanR2Error(f"{noun} is not a valid Ed25519 did:key")
    try:
        ed25519_public_key_from_did(value)
    except LocalProfileTampered as exc:
        raise ExternalAgentPlanR2Error(
            f"{noun} is not a valid Ed25519 did:key"
        ) from exc
    return value


def _closed_mapping(value: object, fields: frozenset[str], noun: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ExternalAgentPlanR2Error(f"{noun} shape is not canonical")
    return dict(value)


def _bounded_sequence(value: object, *, maximum: int, noun: str) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) > maximum
    ):
        raise ExternalAgentPlanR2Error(f"{noun} is outside its population bound")
    return list(value)


def _scope(task: Mapping[str, Any], field: str) -> frozenset[str]:
    body = task.get("body")
    if not isinstance(body, Mapping):
        raise ExternalAgentPlanR2Error("task body is not canonical")
    value = body.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ExternalAgentPlanR2Error(f"frontier task {field} is not explicit")
    return frozenset(value)


def _validate_population(
    *,
    new_plan: object,
    tasks: object,
    dependencies: object,
    protected_tasks: object,
    frontier_task_cids: object,
) -> dict[str, Any]:
    plan = _closed_mapping(new_plan, _PLAN_FIELDS, "new plan")
    for field in ("plan_cid", "plan_root_cid", "semantic_root_cid"):
        _require_sha(plan.get(field), f"new_plan.{field}")
    if (
        plan.get("plan_alias") != "EAAEF-PLAN-R2"
        or plan.get("status") != "active"
        or not _positive_int(plan.get("revision"))
        or int(plan["revision"]) < 2
        or not isinstance(plan.get("body"), Mapping)
    ):
        raise ExternalAgentPlanR2Error("new plan is not an active Plan R2 revision")

    raw_tasks = _bounded_sequence(tasks, maximum=_MAX_TASKS, noun="tasks")
    if not raw_tasks:
        raise ExternalAgentPlanR2Error("Plan R2 task population is empty")
    normalized_tasks: list[dict[str, Any]] = []
    by_cid: dict[str, dict[str, Any]] = {}
    aliases: set[str] = set()
    ordinals: list[int] = []
    for index, raw in enumerate(raw_tasks):
        task = _closed_mapping(raw, _TASK_FIELDS, f"tasks[{index}]")
        task_cid = _require_sha(task.get("task_cid"), f"tasks[{index}].task_cid")
        _require_sha(task.get("goal_cid"), f"tasks[{index}].goal_cid")
        _require_sha(task.get("plan_cid"), f"tasks[{index}].plan_cid")
        alias = _require_safe_id(task.get("task_alias"), f"tasks[{index}].task_alias")
        _require_safe_id(task.get("objective_id"), f"tasks[{index}].objective_id")
        if (
            task_cid in by_cid
            or alias in aliases
            or not _positive_int(task.get("ordinal"))
            or not _nonnegative_int(task.get("revision"))
            or task.get("status") not in _TASK_STATUSES
            or not isinstance(task.get("identity"), Mapping)
            or task["identity"].get("task_cid") != task_cid
            or not isinstance(task.get("body"), Mapping)
        ):
            raise ExternalAgentPlanR2Error("Plan R2 task identity/status is invalid")
        aliases.add(alias)
        ordinals.append(int(task["ordinal"]))
        normalized = dict(task)
        normalized_tasks.append(normalized)
        by_cid[task_cid] = normalized
    if normalized_tasks != sorted(normalized_tasks, key=lambda item: item["task_cid"]):
        raise ExternalAgentPlanR2Error("Plan R2 tasks are not sorted by task_cid")
    if len(ordinals) != len(set(ordinals)):
        raise ExternalAgentPlanR2Error("Plan R2 task ordinals are not unique")

    raw_dependencies = _bounded_sequence(
        dependencies,
        maximum=_MAX_DEPENDENCIES,
        noun="dependencies",
    )
    normalized_dependencies: list[dict[str, Any]] = []
    dependency_keys: list[tuple[str, str, str]] = []
    for index, raw in enumerate(raw_dependencies):
        dependency = _closed_mapping(
            raw, _DEPENDENCY_FIELDS, f"dependencies[{index}]"
        )
        task_cid = _require_sha(
            dependency.get("task_cid"), f"dependencies[{index}].task_cid"
        )
        dependency_cid = _require_sha(
            dependency.get("dependency_task_cid"),
            f"dependencies[{index}].dependency_task_cid",
        )
        kind = _require_safe_id(
            dependency.get("kind"), f"dependencies[{index}].kind"
        )
        if (
            task_cid == dependency_cid
            or task_cid not in by_cid
            or dependency_cid not in by_cid
        ):
            raise ExternalAgentPlanR2Error("Plan R2 dependency endpoints are invalid")
        dependency_keys.append((task_cid, dependency_cid, kind))
        normalized_dependencies.append(dict(dependency))
    if dependency_keys != sorted(set(dependency_keys)):
        raise ExternalAgentPlanR2Error("Plan R2 dependencies are not uniquely sorted")

    raw_protected = _bounded_sequence(
        protected_tasks,
        maximum=_MAX_TASKS,
        noun="protected_tasks",
    )
    normalized_protected: list[dict[str, Any]] = []
    protected_cids: list[str] = []
    for index, raw in enumerate(raw_protected):
        protected = _closed_mapping(
            raw, _PROTECTED_FIELDS, f"protected_tasks[{index}]"
        )
        task_cid = _require_sha(
            protected.get("task_cid"), f"protected_tasks[{index}].task_cid"
        )
        row = _closed_mapping(
            protected.get("task_row"), _TASK_FIELDS, f"protected_tasks[{index}].task_row"
        )
        if (
            task_cid not in by_cid
            or row != by_cid[task_cid]
            or protected.get("status") not in _PROTECTED_STATUSES
            or protected.get("status") != row.get("status")
            or protected.get("revision") != row.get("revision")
            or protected.get("task_row_cid") != _cid(row)
        ):
            raise ExternalAgentPlanR2Error(
                "protected task does not bind its complete canonical row"
            )
        protected_cids.append(task_cid)
        normalized_protected.append(dict(protected))
    if protected_cids != sorted(set(protected_cids)):
        raise ExternalAgentPlanR2Error("protected tasks are not uniquely sorted")
    required_protected = {
        task_cid
        for task_cid, task in by_cid.items()
        if task["status"] in _PROTECTED_STATUSES
    }
    if set(protected_cids) != required_protected:
        raise ExternalAgentPlanR2Error(
            "every in-flight/terminal protected task must bind its full row"
        )

    raw_frontier = _bounded_sequence(
        frontier_task_cids,
        maximum=_MAX_FRONTIER,
        noun="frontier_task_cids",
    )
    frontier = [_require_sha(item, "frontier_task_cids[]") for item in raw_frontier]
    if (
        not frontier
        or frontier != sorted(set(frontier))
        or any(item not in by_cid for item in frontier)
        or any(by_cid[item]["status"] != "todo" for item in frontier)
    ):
        raise ExternalAgentPlanR2Error("Plan R2 frontier is not a sorted ready population")
    frontier_set = set(frontier)
    for task_cid, dependency_cid, _kind in dependency_keys:
        if task_cid in frontier_set and by_cid[dependency_cid]["status"] not in {
            "completed",
            "accepted",
        }:
            raise ExternalAgentPlanR2Error("Plan R2 frontier has unresolved dependencies")
    for index, left_cid in enumerate(frontier):
        left = by_cid[left_cid]
        left_reads = _scope(left, "read_scope")
        left_writes = _scope(left, "write_scope")
        left_effects = _scope(left, "effect_scope")
        for right_cid in frontier[index + 1 :]:
            right = by_cid[right_cid]
            right_reads = _scope(right, "read_scope")
            right_writes = _scope(right, "write_scope")
            right_effects = _scope(right, "effect_scope")
            if (
                left_writes & (right_reads | right_writes)
                or right_writes & left_reads
                or left_effects & right_effects
            ):
                raise ExternalAgentPlanR2Error(
                    "Plan R2 frontier contains overlapping mutations/effects"
                )

    population = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-population@1",
        "new_plan": plan,
        "tasks": normalized_tasks,
        "dependencies": normalized_dependencies,
        "protected_tasks": normalized_protected,
        "frontier_task_cids": frontier,
    }
    return {
        "population": population,
        "population_cid": _cid(population),
        "plan_root_cid": str(plan["plan_root_cid"]),
        "task_population_cid": _cid(normalized_tasks),
        "dependency_population_cid": _cid(normalized_dependencies),
        "protected_tasks_root_cid": _cid(normalized_protected),
        "frontier_cid": _cid(
            {
                "schema": "EAAEFConflictFreeFrontier@1",
                "tasks": frontier,
            }
        ),
    }


def prepare_plan_r2_transition_authorization(
    *,
    board_namespace: str,
    source_head: str,
    source_tree: str,
    source_generation_cid: str,
    bootstrap_admission_cid: str,
    r1_launch_capsule_cid: str,
    quack_owner_qualification_cid: str,
    quack_command_fabric_qualification_cid: str,
    owner_principal_did: str,
    shard_id: str,
    store_id: str,
    owner_generation: int,
    expected_epoch: int,
    fencing_token: int,
    lease_id: str,
    expected_version: int,
    expected_active_plan_cid: str,
    expected_active_plan_root_cid: str,
    expected_active_plan_revision: int,
    expected_event_cursor: str,
    expected_semantic_root_cid: str,
    new_plan: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    dependencies: Sequence[Mapping[str, Any]],
    protected_tasks: Sequence[Mapping[str, Any]],
    frontier_task_cids: Sequence[str],
    delta_cid: str,
    request_id: str,
    idempotency_key: str,
    deadline_ms: int,
    issued_at_ms: int,
    expires_at_ms: int,
    one_use_nonce: str,
) -> dict[str, Any]:
    """Prepare an unsigned authorization statement without changing state."""

    population = _validate_population(
        new_plan=new_plan,
        tasks=tasks,
        dependencies=dependencies,
        protected_tasks=protected_tasks,
        frontier_task_cids=frontier_task_cids,
    )
    if (
        board_namespace != "external-agent-autonomous-execution-fabric-v1"
        or not _GIT_OBJECT.fullmatch(source_head)
        or not _GIT_OBJECT.fullmatch(source_tree)
        or not owner_principal_did.startswith("did:key:z")
        or any(
            not _SHA256.fullmatch(value)
            for value in (
                source_generation_cid,
                bootstrap_admission_cid,
                r1_launch_capsule_cid,
                quack_owner_qualification_cid,
                quack_command_fabric_qualification_cid,
                expected_active_plan_cid,
                expected_active_plan_root_cid,
                expected_semantic_root_cid,
                delta_cid,
            )
        )
        or not all(
            _positive_int(value)
            for value in (
                owner_generation,
                expected_epoch,
                fencing_token,
                expected_active_plan_revision,
                deadline_ms,
                issued_at_ms,
                expires_at_ms,
            )
        )
        or not _nonnegative_int(expected_version)
        or issued_at_ms >= expires_at_ms
        or deadline_ms < issued_at_ms
        or deadline_ms > expires_at_ms
    ):
        raise ExternalAgentPlanR2Error("Plan R2 transition identity/lifetime is invalid")
    for value, noun in (
        (shard_id, "shard_id"),
        (store_id, "store_id"),
        (lease_id, "lease_id"),
        (expected_event_cursor, "expected_event_cursor"),
        (request_id, "request_id"),
        (idempotency_key, "idempotency_key"),
        (one_use_nonce, "one_use_nonce"),
    ):
        _require_safe_id(value, noun)
    statement: dict[str, Any] = {
        "schema": PLAN_R2_TRANSITION_STATEMENT_SCHEMA,
        "board_namespace": board_namespace,
        "source_head": source_head,
        "source_tree": source_tree,
        "source_generation_cid": source_generation_cid,
        "bootstrap_admission_cid": bootstrap_admission_cid,
        "r1_launch_capsule_cid": r1_launch_capsule_cid,
        "quack_owner_qualification_cid": quack_owner_qualification_cid,
        "quack_command_fabric_qualification_cid": quack_command_fabric_qualification_cid,
        "owner_principal_did": owner_principal_did,
        "shard_id": shard_id,
        "store_id": store_id,
        "owner_generation": owner_generation,
        "expected_epoch": expected_epoch,
        "fencing_token": fencing_token,
        "lease_id": lease_id,
        "expected_version": expected_version,
        "expected_active_plan_cid": expected_active_plan_cid,
        "expected_active_plan_root_cid": expected_active_plan_root_cid,
        "expected_active_plan_revision": expected_active_plan_revision,
        "expected_event_cursor": expected_event_cursor,
        "expected_semantic_root_cid": expected_semantic_root_cid,
        "new_plan": population["population"]["new_plan"],
        "tasks": population["population"]["tasks"],
        "dependencies": population["population"]["dependencies"],
        "protected_tasks": population["population"]["protected_tasks"],
        "frontier_task_cids": population["population"]["frontier_task_cids"],
        "population_cid": population["population_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "task_population_cid": population["task_population_cid"],
        "dependency_population_cid": population["dependency_population_cid"],
        "protected_tasks_root_cid": population["protected_tasks_root_cid"],
        "frontier_cid": population["frontier_cid"],
        "delta_cid": delta_cid,
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "deadline_ms": deadline_ms,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
        "one_use_nonce": one_use_nonce,
        "authority": dict(_AUTHORITY),
    }
    statement["statement_cid"] = _cid(statement)
    return statement


def _validate_statement(statement: Mapping[str, Any]) -> None:
    body = dict(statement)
    statement_cid = str(body.pop("statement_cid", ""))
    if (
        set(statement) != _STATEMENT_FIELDS
        or statement.get("schema") != PLAN_R2_TRANSITION_STATEMENT_SCHEMA
        or statement_cid != _cid(body)
        or statement.get("authority") != _AUTHORITY
    ):
        raise ExternalAgentPlanR2Error("Plan R2 authorization statement is invalid")
    rebuilt = prepare_plan_r2_transition_authorization(
        **{
            key: statement[key]
            for key in _STATEMENT_FIELDS
            if key not in {
                "schema",
                "statement_cid",
                "population_cid",
                "plan_root_cid",
                "task_population_cid",
                "dependency_population_cid",
                "protected_tasks_root_cid",
                "frontier_cid",
                "authority",
            }
        }
    )
    if rebuilt != dict(statement):
        raise ExternalAgentPlanR2Error("Plan R2 population identity is invalid")


def prepare_plan_r2_transition_approval(
    statement: Mapping[str, Any],
    *,
    role: str,
    identity_did: str,
    issued_at_ms: int,
    expires_at_ms: int,
) -> dict[str, Any]:
    _validate_statement(statement)
    identity = _require_ed25519_did(
        identity_did,
        "Plan R2 transition approval identity",
    )
    if (
        role not in {"independent_operator", "independent_security_reviewer"}
        or identity == statement["owner_principal_did"]
        or not _positive_int(issued_at_ms)
        or not _positive_int(expires_at_ms)
        or issued_at_ms < int(statement["issued_at_ms"])
        or issued_at_ms >= expires_at_ms
        or expires_at_ms > int(statement["expires_at_ms"])
    ):
        raise ExternalAgentPlanR2Error("Plan R2 transition approval is invalid")
    return {
        "schema": PLAN_R2_TRANSITION_APPROVAL_SCHEMA,
        "role": role,
        "identity_did": identity,
        "statement_cid": statement["statement_cid"],
        "one_use_nonce": statement["one_use_nonce"],
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }


def seal_plan_r2_transition_approval(
    statement: Mapping[str, Any],
    prepared_approval: Mapping[str, Any],
    *,
    signature: str,
) -> dict[str, Any]:
    """Attach an externally produced approval signature without loading a key."""

    value = _closed_mapping(
        prepared_approval,
        _APPROVAL_SIGNING_FIELDS,
        "Plan R2 transition approval signing payload",
    )
    expected = prepare_plan_r2_transition_approval(
        statement,
        role=str(value.get("role") or ""),
        identity_did=str(value.get("identity_did") or ""),
        issued_at_ms=value.get("issued_at_ms"),
        expires_at_ms=value.get("expires_at_ms"),
    )
    if value != expected:
        raise ExternalAgentPlanR2Error(
            "Plan R2 transition approval signing payload differs from its statement"
        )
    if not isinstance(signature, str) or not signature:
        raise ExternalAgentPlanR2Error("Plan R2 transition approval signature is absent")
    sealed = {**value, "signature": signature}
    _canonical_bytes(sealed)
    return sealed


def _verify_approval(
    approval: object,
    *,
    statement: Mapping[str, Any],
    role: str,
    trusted_dids: Sequence[str],
    now_ms: int,
) -> str:
    value = _closed_mapping(approval, _APPROVAL_FIELDS, f"{role} approval")
    identity = str(value.get("identity_did") or "")
    if (
        value.get("schema") != PLAN_R2_TRANSITION_APPROVAL_SCHEMA
        or value.get("role") != role
        or identity not in frozenset(trusted_dids)
        or identity == statement["owner_principal_did"]
        or value.get("statement_cid") != statement["statement_cid"]
        or value.get("one_use_nonce") != statement["one_use_nonce"]
        or not _positive_int(value.get("issued_at_ms"))
        or not _positive_int(value.get("expires_at_ms"))
        or int(value["issued_at_ms"]) > now_ms
        or now_ms >= int(value["expires_at_ms"])
        or int(value["expires_at_ms"]) > int(statement["expires_at_ms"])
    ):
        raise ExternalAgentPlanR2Error(f"{role} approval is invalid")
    payload = dict(value)
    signature = payload.pop("signature", None)
    if not isinstance(signature, str) or not signature:
        raise ExternalAgentPlanR2Error(f"{role} approval is unsigned")
    try:
        verify_did_key_signature(
            identity_did=identity,
            payload=payload,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise ExternalAgentPlanR2Error(f"{role} approval signature is invalid") from exc
    return identity


def assemble_plan_r2_transition_authorization(
    statement: Mapping[str, Any],
    *,
    operator_approval: Mapping[str, Any],
    security_approval: Mapping[str, Any],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    _validate_statement(statement)
    operator = _verify_approval(
        operator_approval,
        statement=statement,
        role="independent_operator",
        trusted_dids=trusted_operator_dids,
        now_ms=now_ms,
    )
    security = _verify_approval(
        security_approval,
        statement=statement,
        role="independent_security_reviewer",
        trusted_dids=trusted_security_reviewer_dids,
        now_ms=now_ms,
    )
    if operator == security:
        raise ExternalAgentPlanR2Error("operator and security reviewer must differ")
    authorization = {
        **dict(statement),
        "schema": PLAN_R2_TRANSITION_AUTHORIZATION_SCHEMA,
        "operator_approval": dict(operator_approval),
        "security_approval": dict(security_approval),
    }
    authorization["authorization_cid"] = _cid(authorization)
    return authorization


def verify_plan_r2_transition_authorization(
    authorization: object,
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    value = _closed_mapping(
        authorization, _AUTHORIZATION_FIELDS, "Plan R2 transition authorization"
    )
    body = dict(value)
    authorization_cid = str(body.pop("authorization_cid", ""))
    operator_approval = body.pop("operator_approval", None)
    security_approval = body.pop("security_approval", None)
    if (
        value.get("schema") != PLAN_R2_TRANSITION_AUTHORIZATION_SCHEMA
        or authorization_cid
        != _cid({**body, "operator_approval": operator_approval, "security_approval": security_approval})
    ):
        raise ExternalAgentPlanR2Error("Plan R2 authorization self-address is invalid")
    statement = dict(body)
    statement["schema"] = PLAN_R2_TRANSITION_STATEMENT_SCHEMA
    _validate_statement(statement)
    operator = _verify_approval(
        operator_approval,
        statement=statement,
        role="independent_operator",
        trusted_dids=trusted_operator_dids,
        now_ms=now_ms,
    )
    security = _verify_approval(
        security_approval,
        statement=statement,
        role="independent_security_reviewer",
        trusted_dids=trusted_security_reviewer_dids,
        now_ms=now_ms,
    )
    if operator == security or now_ms >= int(statement["expires_at_ms"]):
        raise ExternalAgentPlanR2Error("Plan R2 authorization separation/lifetime failed")
    report = {
        "schema": PLAN_R2_TRANSITION_VERIFICATION_SCHEMA,
        "valid": True,
        "authorization_cid": authorization_cid,
        "statement_cid": statement["statement_cid"],
        "source_head": statement["source_head"],
        "source_tree": statement["source_tree"],
        "plan_root_cid": statement["plan_root_cid"],
        "population_cid": statement["population_cid"],
        "frontier_cid": statement["frontier_cid"],
        "operator_identity_did": operator,
        "security_reviewer_identity_did": security,
        "owner_principal_did": statement["owner_principal_did"],
        "expires_at_ms": statement["expires_at_ms"],
        "authority_mutated": False,
        "process_started": False,
    }
    report["verification_cid"] = _cid(report)
    return report


def _validate_plan_r2_operational_capability_signing_payload(
    payload: object,
) -> dict[str, Any]:
    value = _closed_mapping(
        payload,
        _CAPABILITY_SIGNING_FIELDS,
        "Plan R2 operational capability signing payload",
    )
    owner_principal_did = _require_ed25519_did(
        value.get("owner_principal_did"),
        "Plan R2 operational capability owner principal",
    )
    reviewer_identity_did = _require_ed25519_did(
        value.get("reviewer_identity_did"),
        "Plan R2 operational capability reviewer",
    )
    if (
        value.get("schema") != PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA
        or value.get("allowed") is not True
        or value.get("blockers") != []
        or value.get("duckdb_version") != "1.5.5"
        or value.get("quack_build") != "quack@1.5.5+core"
        or value.get("authorized_state_command_schema")
        != "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        or any(value.get(field) is not True for field in _CAPABILITY_REQUIRED_TRUE)
        or not _GIT_OBJECT.fullmatch(str(value.get("source_head") or ""))
        or not _GIT_OBJECT.fullmatch(str(value.get("source_tree") or ""))
        or any(
            not _SHA256.fullmatch(str(value.get(field) or ""))
            for field in (
                "bootstrap_admission_cid",
                "quack_owner_qualification_cid",
                "quack_command_fabric_qualification_cid",
            )
        )
        or reviewer_identity_did == owner_principal_did
        or not _SAFE_ID.fullmatch(str(value.get("shard_id") or ""))
        or not all(
            _positive_int(value.get(field))
            for field in (
                "owner_generation",
                "epoch",
                "fence",
                "issued_at_ms",
                "expires_at_ms",
            )
        )
        or int(value["issued_at_ms"]) >= int(value["expires_at_ms"])
    ):
        raise ExternalAgentPlanR2Error(
            "Plan R2 operational capability signing payload is invalid"
        )
    _canonical_bytes(value)
    return value


def _plan_r2_statement_binding(transition: Mapping[str, Any]) -> dict[str, Any]:
    if transition.get("schema") == PLAN_R2_TRANSITION_STATEMENT_SCHEMA:
        statement = dict(transition)
        _validate_statement(statement)
        return statement
    value = _closed_mapping(
        transition,
        _AUTHORIZATION_FIELDS,
        "Plan R2 transition authorization signing source",
    )
    if value.get("schema") != PLAN_R2_TRANSITION_AUTHORIZATION_SCHEMA:
        raise ExternalAgentPlanR2Error(
            "Plan R2 operational capability has no exact transition binding"
        )
    body = dict(value)
    authorization_cid = str(body.pop("authorization_cid", ""))
    if authorization_cid != _cid(body):
        raise ExternalAgentPlanR2Error(
            "Plan R2 transition authorization signing source is not self-addressed"
        )
    statement = {key: value[key] for key in _STATEMENT_FIELDS}
    statement["schema"] = PLAN_R2_TRANSITION_STATEMENT_SCHEMA
    _validate_statement(statement)
    return statement


def plan_r2_operational_capability_signing_payload(
    transition: Mapping[str, Any],
    *,
    reviewer_identity_did: str,
    issued_at_ms: int,
    expires_at_ms: int,
) -> dict[str, Any]:
    """Build public Plan-R2 capability claims for an independent reviewer."""

    statement = _plan_r2_statement_binding(transition)
    owner_principal_did = _require_ed25519_did(
        statement.get("owner_principal_did"),
        "Plan R2 transition owner principal",
    )
    reviewer_identity = _require_ed25519_did(
        reviewer_identity_did,
        "Plan R2 operational capability reviewer",
    )
    if (
        reviewer_identity == owner_principal_did
        or not _positive_int(issued_at_ms)
        or not _positive_int(expires_at_ms)
        or issued_at_ms < int(statement["issued_at_ms"])
        or issued_at_ms >= expires_at_ms
        or expires_at_ms > int(statement["expires_at_ms"])
    ):
        raise ExternalAgentPlanR2Error(
            "Plan R2 operational capability reviewer/lifetime is invalid"
        )
    value: dict[str, Any] = {
        "schema": PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA,
        "allowed": True,
        "blockers": [],
        "source_head": statement["source_head"],
        "source_tree": statement["source_tree"],
        "bootstrap_admission_cid": statement["bootstrap_admission_cid"],
        "quack_owner_qualification_cid": statement["quack_owner_qualification_cid"],
        "quack_command_fabric_qualification_cid": statement[
            "quack_command_fabric_qualification_cid"
        ],
        "owner_principal_did": statement["owner_principal_did"],
        "shard_id": statement["shard_id"],
        "owner_generation": statement["owner_generation"],
        "epoch": statement["expected_epoch"],
        "fence": statement["fencing_token"],
        "duckdb_version": "1.5.5",
        "quack_build": "quack@1.5.5+core",
        "authorized_state_command_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        **{field: True for field in _CAPABILITY_REQUIRED_TRUE},
        "reviewer_identity_did": reviewer_identity,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }
    return _validate_plan_r2_operational_capability_signing_payload(value)


def seal_plan_r2_operational_capability(
    prepared_payload: Mapping[str, Any],
    *,
    reviewer_signature: str,
) -> dict[str, Any]:
    """Seal externally produced review evidence into a self-addressed capability."""

    value = _validate_plan_r2_operational_capability_signing_payload(prepared_payload)
    if not isinstance(reviewer_signature, str) or not reviewer_signature:
        raise ExternalAgentPlanR2Error(
            "Plan R2 operational capability reviewer signature is absent"
        )
    signed = {**value, "reviewer_signature": reviewer_signature}
    capability = {**signed, "capability_cid": _cid(signed)}
    _canonical_bytes(capability)
    return capability


def verify_plan_r2_operational_capability(
    capability: object,
    *,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    value = _closed_mapping(
        capability, _CAPABILITY_FIELDS, "Plan R2 operational capability"
    )
    body = dict(value)
    capability_cid = str(body.pop("capability_cid", ""))
    signature = body.pop("reviewer_signature", None)
    reviewer = str(value.get("reviewer_identity_did") or "")
    if (
        value.get("schema") != PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA
        or capability_cid != _cid({**body, "reviewer_signature": signature})
        or value.get("allowed") is not True
        or value.get("blockers") != []
        or value.get("duckdb_version") != "1.5.5"
        or value.get("quack_build") != "quack@1.5.5+core"
        or value.get("authorized_state_command_schema")
        != "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        or any(value.get(field) is not True for field in _CAPABILITY_REQUIRED_TRUE)
        or reviewer not in frozenset(trusted_reviewer_dids)
        or reviewer == value.get("owner_principal_did")
        or not _GIT_OBJECT.fullmatch(str(value.get("source_head") or ""))
        or not _GIT_OBJECT.fullmatch(str(value.get("source_tree") or ""))
        or any(
            not _SHA256.fullmatch(str(value.get(field) or ""))
            for field in (
                "bootstrap_admission_cid",
                "quack_owner_qualification_cid",
                "quack_command_fabric_qualification_cid",
            )
        )
        or not all(
            _positive_int(value.get(field))
            for field in (
                "owner_generation",
                "epoch",
                "fence",
                "issued_at_ms",
                "expires_at_ms",
            )
        )
        or int(value["issued_at_ms"]) > now_ms
        or now_ms >= int(value["expires_at_ms"])
        or not isinstance(signature, str)
        or not signature
    ):
        raise ExternalAgentPlanR2Error("typed_quack_plan_transition_unavailable")
    signed_payload = dict(value)
    signed_payload.pop("capability_cid", None)
    signed_payload.pop("reviewer_signature", None)
    try:
        verify_did_key_signature(
            identity_did=reviewer,
            payload=signed_payload,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise ExternalAgentPlanR2Error("typed_quack_plan_transition_unavailable") from exc
    return value


def assess_plan_r2_transition(
    authorization: object,
    capability: object,
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    trusted_capability_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    verified_authorization: dict[str, Any] = {}
    verified_capability: dict[str, Any] = {}
    try:
        verified_authorization = verify_plan_r2_transition_authorization(
            authorization,
            trusted_operator_dids=trusted_operator_dids,
            trusted_security_reviewer_dids=trusted_security_reviewer_dids,
            now_ms=now_ms,
        )
    except ExternalAgentPlanR2Error as exc:
        blockers.append(f"transition_authorization_invalid:{exc}")
    try:
        verified_capability = verify_plan_r2_operational_capability(
            capability,
            trusted_reviewer_dids=trusted_capability_reviewer_dids,
            now_ms=now_ms,
        )
    except ExternalAgentPlanR2Error:
        blockers.append("typed_quack_plan_transition_unavailable")
    if verified_authorization and verified_capability:
        auth = dict(authorization) if isinstance(authorization, Mapping) else {}
        comparisons = {
            "source_head": "source_head",
            "source_tree": "source_tree",
            "bootstrap_admission_cid": "bootstrap_admission_cid",
            "quack_owner_qualification_cid": "quack_owner_qualification_cid",
            "quack_command_fabric_qualification_cid": "quack_command_fabric_qualification_cid",
            "owner_principal_did": "owner_principal_did",
            "shard_id": "shard_id",
            "owner_generation": "owner_generation",
            "expected_epoch": "epoch",
            "fencing_token": "fence",
        }
        if any(auth[left] != verified_capability[right] for left, right in comparisons.items()):
            blockers.append("transition_capability_identity_mismatch")
    blockers = list(dict.fromkeys(blockers))
    report = {
        "schema": PLAN_R2_TRANSITION_DECISION_SCHEMA,
        "allowed": not blockers,
        "blockers": blockers,
        "authorization_cid": verified_authorization.get("authorization_cid", ""),
        "capability_cid": verified_capability.get("capability_cid", ""),
        "authority_mutated": False,
        "process_started": False,
    }
    report["decision_cid"] = _cid(report)
    return report


@runtime_checkable
class AuthorizedPlanR2TransitionRepository(Protocol):
    INTERFACE: str

    def prepare_authorized_plan_r2_transition(
        self, authorization: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def apply_authorized_plan_r2_transition(
        self,
        authorization: Mapping[str, Any],
        prepared_projection: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def observe_authorized_plan_r2_transition(
        self,
        authorization: Mapping[str, Any],
        transition_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _validate_prepared(
    value: object,
    *,
    authorization: Mapping[str, Any],
    capability: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    projection = _closed_mapping(value, _PREPARED_FIELDS, "Plan R2 prepared projection")
    body = dict(projection)
    projection_cid = str(body.pop("projection_cid", ""))
    if (
        projection.get("schema") != PLAN_R2_PREPARED_PROJECTION_SCHEMA
        or projection_cid != _cid(body)
        or projection.get("authorization_cid") != authorization["authorization_cid"]
        or projection.get("statement_cid") != authorization["statement_cid"]
        or projection.get("capability_cid") != capability["capability_cid"]
        or not _SHA256.fullmatch(
            str(projection.get("authorized_prepare_command_cid") or "")
        )
        or projection.get("source_head") != authorization["source_head"]
        or projection.get("source_tree") != authorization["source_tree"]
        or projection.get("shard_id") != authorization["shard_id"]
        or projection.get("owner_generation") != authorization["owner_generation"]
        or projection.get("epoch") != authorization["expected_epoch"]
        or projection.get("fence") != authorization["fencing_token"]
        or projection.get("before_plan_cid") != authorization["expected_active_plan_cid"]
        or projection.get("before_plan_root_cid")
        != authorization["expected_active_plan_root_cid"]
        or projection.get("before_plan_revision")
        != authorization["expected_active_plan_revision"]
        or projection.get("before_version") != authorization["expected_version"]
        or projection.get("before_event_cursor") != authorization["expected_event_cursor"]
        or projection.get("before_semantic_root_cid")
        != authorization["expected_semantic_root_cid"]
        or projection.get("population_cid") != authorization["population_cid"]
        or projection.get("plan_root_cid") != authorization["plan_root_cid"]
        or projection.get("protected_tasks_root_cid")
        != authorization["protected_tasks_root_cid"]
        or projection.get("frontier_cid") != authorization["frontier_cid"]
        or projection.get("authority_mutated") is not False
        or projection.get("process_started") is not False
        or not _positive_int(projection.get("prepared_at_ms"))
        or not _positive_int(projection.get("expires_at_ms"))
        or int(projection["prepared_at_ms"]) > now_ms
        or now_ms >= int(projection["expires_at_ms"])
        or int(projection["expires_at_ms"]) > int(authorization["expires_at_ms"])
    ):
        raise ExternalAgentPlanR2Error("prepared Plan R2 projection is invalid")
    return projection


def prepare_authorized_plan_r2_transition(
    repository: object,
    authorization: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    trusted_capability_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    decision = assess_plan_r2_transition(
        authorization,
        capability,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        trusted_capability_reviewer_dids=trusted_capability_reviewer_dids,
        now_ms=now_ms,
    )
    if decision["allowed"] is not True:
        raise ExternalAgentPlanR2Error(",".join(decision["blockers"]))
    if (
        getattr(repository, "INTERFACE", "") != AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE
        or not isinstance(repository, AuthorizedPlanR2TransitionRepository)
    ):
        raise ExternalAgentPlanR2Error(
            "typed_quack_plan_transition_owner_adapter_unavailable"
        )
    projection = repository.prepare_authorized_plan_r2_transition(authorization)
    return _validate_prepared(
        projection,
        authorization=authorization,
        capability=capability,
        now_ms=now_ms,
    )


def _validate_receipt(
    value: object,
    *,
    authorization: Mapping[str, Any],
    capability: Mapping[str, Any],
    prepared: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    receipt = _closed_mapping(value, _RECEIPT_FIELDS, "Plan R2 transition receipt")
    body = dict(receipt)
    receipt_cid = str(body.pop("receipt_cid", ""))
    new_plan = authorization["new_plan"]
    if (
        receipt.get("schema") != PLAN_R2_TRANSITION_RECEIPT_SCHEMA
        or receipt_cid != _cid(body)
        or receipt.get("authorization_cid") != authorization["authorization_cid"]
        or receipt.get("statement_cid") != authorization["statement_cid"]
        or receipt.get("capability_cid") != capability["capability_cid"]
        or not _SHA256.fullmatch(
            str(receipt.get("authorized_apply_command_cid") or "")
        )
        or receipt.get("prepared_projection_cid") != prepared["projection_cid"]
        or receipt.get("authorized_prepare_command_cid")
        != prepared["authorized_prepare_command_cid"]
        or receipt.get("source_head") != authorization["source_head"]
        or receipt.get("source_tree") != authorization["source_tree"]
        or receipt.get("shard_id") != authorization["shard_id"]
        or receipt.get("owner_generation") != authorization["owner_generation"]
        or receipt.get("epoch") != authorization["expected_epoch"]
        or receipt.get("fence") != authorization["fencing_token"]
        or receipt.get("before_plan_cid") != authorization["expected_active_plan_cid"]
        or receipt.get("after_plan_cid") != new_plan["plan_cid"]
        or receipt.get("before_plan_root_cid")
        != authorization["expected_active_plan_root_cid"]
        or receipt.get("after_plan_root_cid") != authorization["plan_root_cid"]
        or receipt.get("before_plan_revision")
        != authorization["expected_active_plan_revision"]
        or receipt.get("after_plan_revision") != new_plan["revision"]
        or receipt.get("before_version") != authorization["expected_version"]
        or not _positive_int(receipt.get("after_version"))
        or int(receipt["after_version"]) <= int(receipt["before_version"])
        or receipt.get("before_event_cursor") != authorization["expected_event_cursor"]
        or not _SAFE_ID.fullmatch(str(receipt.get("after_event_cursor") or ""))
        or receipt.get("before_semantic_root_cid")
        != authorization["expected_semantic_root_cid"]
        or receipt.get("after_semantic_root_cid") != new_plan["semantic_root_cid"]
        or receipt.get("population_cid") != authorization["population_cid"]
        or receipt.get("task_population_cid") != authorization["task_population_cid"]
        or receipt.get("dependency_population_cid")
        != authorization["dependency_population_cid"]
        or receipt.get("protected_tasks_root_cid")
        != authorization["protected_tasks_root_cid"]
        or receipt.get("frontier_cid") != authorization["frontier_cid"]
        or receipt.get("frontier_task_cids") != authorization["frontier_task_cids"]
        or receipt.get("protected_tasks_unchanged") is not True
        or not _SHA256.fullmatch(str(receipt.get("transaction_cid") or ""))
        or receipt.get("replayed") is not False
        or not _positive_int(receipt.get("committed_at_ms"))
        or int(receipt["committed_at_ms"]) < int(prepared["prepared_at_ms"])
        or int(receipt["committed_at_ms"]) > now_ms
    ):
        raise ExternalAgentPlanR2Error("atomic Plan R2 transition receipt is invalid")
    return receipt


def apply_authorized_plan_r2_transition(
    repository: object,
    authorization: Mapping[str, Any],
    capability: Mapping[str, Any],
    prepared_projection: Mapping[str, Any],
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    trusted_capability_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    decision = assess_plan_r2_transition(
        authorization,
        capability,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        trusted_capability_reviewer_dids=trusted_capability_reviewer_dids,
        now_ms=now_ms,
    )
    if decision["allowed"] is not True:
        raise ExternalAgentPlanR2Error(",".join(decision["blockers"]))
    prepared = _validate_prepared(
        prepared_projection,
        authorization=authorization,
        capability=capability,
        now_ms=now_ms,
    )
    if (
        getattr(repository, "INTERFACE", "") != AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE
        or not isinstance(repository, AuthorizedPlanR2TransitionRepository)
    ):
        raise ExternalAgentPlanR2Error(
            "typed_quack_plan_transition_owner_adapter_unavailable"
        )
    receipt = repository.apply_authorized_plan_r2_transition(
        authorization, prepared
    )
    return _validate_receipt(
        receipt,
        authorization=authorization,
        capability=capability,
        prepared=prepared,
        now_ms=now_ms,
    )


def _validate_transition_receipt_for_launch(
    value: object,
    *,
    authorization: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    """Recheck every immutable/CAS binding needed by a later launch."""

    receipt = _closed_mapping(value, _RECEIPT_FIELDS, "Plan R2 transition receipt")
    body = dict(receipt)
    receipt_cid = str(body.pop("receipt_cid", ""))
    new_plan = authorization["new_plan"]
    if (
        receipt.get("schema") != PLAN_R2_TRANSITION_RECEIPT_SCHEMA
        or receipt_cid != _cid(body)
        or receipt.get("authorization_cid") != authorization["authorization_cid"]
        or receipt.get("statement_cid") != authorization["statement_cid"]
        or not _SHA256.fullmatch(str(receipt.get("capability_cid") or ""))
        or not _SHA256.fullmatch(
            str(receipt.get("authorized_prepare_command_cid") or "")
        )
        or not _SHA256.fullmatch(
            str(receipt.get("authorized_apply_command_cid") or "")
        )
        or not _SHA256.fullmatch(
            str(receipt.get("prepared_projection_cid") or "")
        )
        or receipt.get("source_head") != authorization["source_head"]
        or receipt.get("source_tree") != authorization["source_tree"]
        or receipt.get("shard_id") != authorization["shard_id"]
        or receipt.get("owner_generation") != authorization["owner_generation"]
        or receipt.get("epoch") != authorization["expected_epoch"]
        or receipt.get("fence") != authorization["fencing_token"]
        or receipt.get("before_plan_cid")
        != authorization["expected_active_plan_cid"]
        or receipt.get("after_plan_cid") != new_plan["plan_cid"]
        or receipt.get("before_plan_root_cid")
        != authorization["expected_active_plan_root_cid"]
        or receipt.get("after_plan_root_cid") != authorization["plan_root_cid"]
        or receipt.get("before_plan_revision")
        != authorization["expected_active_plan_revision"]
        or receipt.get("after_plan_revision") != new_plan["revision"]
        or receipt.get("before_version") != authorization["expected_version"]
        or not _positive_int(receipt.get("after_version"))
        or int(receipt["after_version"]) <= int(receipt["before_version"])
        or receipt.get("before_event_cursor")
        != authorization["expected_event_cursor"]
        or not _SAFE_ID.fullmatch(str(receipt.get("after_event_cursor") or ""))
        or receipt.get("before_semantic_root_cid")
        != authorization["expected_semantic_root_cid"]
        or receipt.get("after_semantic_root_cid") != new_plan["semantic_root_cid"]
        or receipt.get("population_cid") != authorization["population_cid"]
        or receipt.get("task_population_cid")
        != authorization["task_population_cid"]
        or receipt.get("dependency_population_cid")
        != authorization["dependency_population_cid"]
        or receipt.get("protected_tasks_root_cid")
        != authorization["protected_tasks_root_cid"]
        or receipt.get("frontier_cid") != authorization["frontier_cid"]
        or receipt.get("frontier_task_cids")
        != authorization["frontier_task_cids"]
        or receipt.get("protected_tasks_unchanged") is not True
        or not _SHA256.fullmatch(str(receipt.get("transaction_cid") or ""))
        or receipt.get("replayed") is not False
        or not _positive_int(receipt.get("committed_at_ms"))
        or int(receipt["committed_at_ms"]) > now_ms
    ):
        raise ExternalAgentPlanR2Error(
            "atomic Plan R2 transition receipt is invalid for launch"
        )
    return receipt


def _validate_state_observation(
    value: object,
    *,
    authorization: Mapping[str, Any],
    receipt: Mapping[str, Any],
    now_ms: int,
) -> dict[str, Any]:
    observation = _closed_mapping(
        value, _OBSERVATION_FIELDS, "Plan R2 state observation"
    )
    body = dict(observation)
    observation_cid = str(body.pop("observation_cid", ""))
    if (
        observation.get("schema") != PLAN_R2_STATE_OBSERVATION_SCHEMA
        or observation_cid != _cid(body)
        or observation.get("authorization_cid")
        != authorization["authorization_cid"]
        or observation.get("transition_receipt_cid") != receipt["receipt_cid"]
        or observation.get("transaction_cid") != receipt["transaction_cid"]
        or observation.get("authorized_prepare_command_cid")
        != receipt["authorized_prepare_command_cid"]
        or observation.get("authorized_apply_command_cid")
        != receipt["authorized_apply_command_cid"]
        or observation.get("quack_command_fabric_qualification_cid")
        != authorization["quack_command_fabric_qualification_cid"]
        or observation.get("source_head") != authorization["source_head"]
        or observation.get("source_tree") != authorization["source_tree"]
        or observation.get("owner_principal_did")
        != authorization["owner_principal_did"]
        or observation.get("shard_id") != authorization["shard_id"]
        or observation.get("owner_generation") != authorization["owner_generation"]
        or observation.get("epoch") != authorization["expected_epoch"]
        or observation.get("fence") != authorization["fencing_token"]
        or observation.get("store_version") != receipt["after_version"]
        or observation.get("active_plan_cid") != receipt["after_plan_cid"]
        or observation.get("active_plan_root_cid")
        != receipt["after_plan_root_cid"]
        or observation.get("active_plan_revision")
        != receipt["after_plan_revision"]
        or observation.get("event_cursor") != receipt["after_event_cursor"]
        or observation.get("semantic_root_cid")
        != receipt["after_semantic_root_cid"]
        or observation.get("population_cid") != receipt["population_cid"]
        or observation.get("task_population_cid")
        != receipt["task_population_cid"]
        or observation.get("dependency_population_cid")
        != receipt["dependency_population_cid"]
        or observation.get("protected_tasks_root_cid")
        != receipt["protected_tasks_root_cid"]
        or observation.get("frontier_cid") != receipt["frontier_cid"]
        or observation.get("frontier_task_cids")
        != receipt["frontier_task_cids"]
        or observation.get("authority_mutated") is not False
        or observation.get("process_started") is not False
        or not _positive_int(observation.get("captured_at_ms"))
        or int(observation["captured_at_ms"]) < int(receipt["committed_at_ms"])
        or int(observation["captured_at_ms"]) > now_ms
    ):
        raise ExternalAgentPlanR2Error(
            "Plan R2 launch state was not independently re-observed"
        )
    return observation


def observe_authorized_plan_r2_transition(
    repository: object,
    authorization: Mapping[str, Any],
    capability: Mapping[str, Any],
    transition_receipt: Mapping[str, Any],
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    trusted_capability_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    """Read back the committed R2 state without granting process authority."""

    decision = assess_plan_r2_transition(
        authorization,
        capability,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        trusted_capability_reviewer_dids=trusted_capability_reviewer_dids,
        now_ms=now_ms,
    )
    if decision["allowed"] is not True:
        raise ExternalAgentPlanR2Error(",".join(decision["blockers"]))
    receipt = _validate_transition_receipt_for_launch(
        transition_receipt,
        authorization=authorization,
        now_ms=now_ms,
    )
    if receipt["capability_cid"] != capability["capability_cid"]:
        raise ExternalAgentPlanR2Error(
            "Plan R2 transition capability differs from the read-back authority"
        )
    if (
        getattr(repository, "INTERFACE", "")
        != AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE
        or not isinstance(repository, AuthorizedPlanR2TransitionRepository)
    ):
        raise ExternalAgentPlanR2Error(
            "typed_quack_plan_transition_owner_adapter_unavailable"
        )
    observation = repository.observe_authorized_plan_r2_transition(
        authorization, receipt
    )
    return _validate_state_observation(
        observation,
        authorization=authorization,
        receipt=receipt,
        now_ms=now_ms,
    )


def validate_plan_r2_launch_transition(
    *,
    repository: object | None = None,
    authorization: Mapping[str, Any],
    transition_receipt: Mapping[str, Any],
    state_observation: Mapping[str, Any],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    """Join artifacts to a fresh readback from the authenticated owner.

    Content hashes authenticate immutable bytes, not their truth or current
    control-plane state.  Launch validation therefore requires the typed owner
    repository to perform another signed, fenced read transaction and compares
    that result with the supplied historical observation.
    """

    verify_plan_r2_transition_authorization(
        authorization,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        now_ms=now_ms,
    )
    receipt = _validate_transition_receipt_for_launch(
        transition_receipt,
        authorization=authorization,
        now_ms=now_ms,
    )
    observation = _validate_state_observation(
        state_observation,
        authorization=authorization,
        receipt=receipt,
        now_ms=now_ms,
    )
    if (
        repository is None
        or getattr(repository, "INTERFACE", "")
        != AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE
        or not isinstance(repository, AuthorizedPlanR2TransitionRepository)
    ):
        raise ExternalAgentPlanR2Error(
            "owner_authenticated_plan_r2_live_readback_unavailable"
        )
    owner_bindings = {
        "capability_cid": receipt["capability_cid"],
        "command_fabric_qualification_cid": authorization[
            "quack_command_fabric_qualification_cid"
        ],
        "owner_principal_did": authorization["owner_principal_did"],
        "shard_id": authorization["shard_id"],
        "store_id": authorization["store_id"],
        "owner_generation": authorization["owner_generation"],
        "owner_epoch": authorization["expected_epoch"],
        "fence_epoch": authorization["fencing_token"],
    }
    mismatched_owner_bindings = sorted(
        field
        for field, expected in owner_bindings.items()
        if getattr(repository, field, None) != expected
    )
    if mismatched_owner_bindings:
        raise ExternalAgentPlanR2Error(
            "owner_authenticated_plan_r2_repository_binding_mismatch:"
            + ",".join(mismatched_owner_bindings)
        )
    try:
        live_value = repository.observe_authorized_plan_r2_transition(
            authorization, receipt
        )
    except Exception as exc:
        raise ExternalAgentPlanR2Error(
            "owner_authenticated_plan_r2_live_readback_failed"
        ) from exc
    live_observation = _validate_state_observation(
        live_value,
        authorization=authorization,
        receipt=receipt,
        now_ms=now_ms,
    )
    if now_ms - int(live_observation["captured_at_ms"]) > _MAX_LAUNCH_READBACK_AGE_MS:
        raise ExternalAgentPlanR2Error(
            "owner_authenticated_plan_r2_live_readback_is_stale"
        )
    stable_observation_fields = _OBSERVATION_FIELDS - {
        "captured_at_ms",
        "observation_cid",
    }
    if any(
        live_observation[field] != observation[field]
        for field in stable_observation_fields
    ):
        raise ExternalAgentPlanR2Error(
            "Plan R2 artifact differs from current owner live readback"
        )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-plan-r2-launch-transition-verification@1"
        ),
        "valid": True,
        "authorization_cid": authorization["authorization_cid"],
        "transition_receipt_cid": receipt["receipt_cid"],
        "state_observation_cid": observation["observation_cid"],
        "active_plan_cid": receipt["after_plan_cid"],
        "active_plan_root_cid": receipt["after_plan_root_cid"],
        "active_plan_revision": receipt["after_plan_revision"],
        "event_cursor": receipt["after_event_cursor"],
        "semantic_root_cid": receipt["after_semantic_root_cid"],
        "frontier_cid": receipt["frontier_cid"],
        "frontier_task_cids": list(receipt["frontier_task_cids"]),
        "authority_mutated": False,
        "process_started": False,
    }


def _plan_r2_artifact_relative_path(
    kind: str,
    source_head: str,
    plan_root_cid: str,
    *,
    registry_prefix: str = EAAEF_AUTHORITY_REGISTRY_PREFIX,
) -> Path:
    if (
        registry_prefix != EAAEF_AUTHORITY_REGISTRY_PREFIX
        or not _GIT_OBJECT.fullmatch(str(source_head or ""))
        or not _SHA256.fullmatch(str(plan_root_cid or ""))
        or kind not in {
            "plan-r2-transition-authorization",
            "plan-r2-transition-receipt",
            "plan-r2-state-observation",
        }
    ):
        raise ExternalAgentPlanR2Error("Plan R2 authority artifact path is invalid")
    return Path(registry_prefix) / (
        f"{kind}--{source_head}--{plan_root_cid.removeprefix('sha256:')}.json"
    )


def plan_r2_transition_authorization_relative_path(
    source_head: str,
    plan_root_cid: str,
    *,
    registry_prefix: str = EAAEF_AUTHORITY_REGISTRY_PREFIX,
) -> Path:
    return _plan_r2_artifact_relative_path(
        "plan-r2-transition-authorization",
        source_head,
        plan_root_cid,
        registry_prefix=registry_prefix,
    )


def plan_r2_transition_receipt_relative_path(
    source_head: str,
    plan_root_cid: str,
    *,
    registry_prefix: str = EAAEF_AUTHORITY_REGISTRY_PREFIX,
) -> Path:
    return _plan_r2_artifact_relative_path(
        "plan-r2-transition-receipt",
        source_head,
        plan_root_cid,
        registry_prefix=registry_prefix,
    )


def plan_r2_state_observation_relative_path(
    source_head: str,
    plan_root_cid: str,
    *,
    registry_prefix: str = EAAEF_AUTHORITY_REGISTRY_PREFIX,
) -> Path:
    return _plan_r2_artifact_relative_path(
        "plan-r2-state-observation",
        source_head,
        plan_root_cid,
        registry_prefix=registry_prefix,
    )


def _publish_or_confirm_plan_r2_artifact(
    repo_root: str | Path,
    relative_path: Path,
    value: Mapping[str, Any],
    *,
    noun: str,
) -> None:
    """Create once, or confirm an exact immutable replay after a crash."""

    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
        ExternalAgentBootstrapAdmissionError,
        _publish_create_once_repo_json,
    )

    try:
        _publish_create_once_repo_json(
            repo_root,
            relative_path,
            value,
            noun=noun,
        )
        return
    except ExternalAgentBootstrapAdmissionError as exc:
        if "refusing to overwrite immutable" not in str(exc):
            raise ExternalAgentPlanR2Error(str(exc)) from exc
    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_configured_board_capsule import (
        ExternalAgentConfiguredBoardCapsuleError,
        _read_stable_repo_json,
    )

    try:
        observed, _file_cid = _read_stable_repo_json(
            Path(repo_root),
            relative_path.as_posix(),
            noun=noun,
        )
    except ExternalAgentConfiguredBoardCapsuleError as exc:
        raise ExternalAgentPlanR2Error(str(exc)) from exc
    if _canonical_bytes(observed) != _canonical_bytes(dict(value)):
        raise ExternalAgentPlanR2Error(
            f"immutable {noun} conflicts with the requested replay"
        )


def publish_plan_r2_transition_authorization(
    repo_root: str | Path,
    authorization: Mapping[str, Any],
    *,
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    verification = verify_plan_r2_transition_authorization(
        authorization,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        now_ms=now_ms,
    )
    _publish_or_confirm_plan_r2_artifact(
        repo_root,
        plan_r2_transition_authorization_relative_path(
            str(authorization["source_head"]),
            str(authorization["plan_root_cid"]),
        ),
        authorization,
        noun="Plan R2 transition authorization",
    )
    return verification


def publish_plan_r2_transition_result(
    repo_root: str | Path,
    *,
    repository: object | None = None,
    authorization: Mapping[str, Any],
    transition_receipt: Mapping[str, Any],
    state_observation: Mapping[str, Any],
    trusted_operator_dids: Sequence[str],
    trusted_security_reviewer_dids: Sequence[str],
    now_ms: int,
) -> dict[str, Any]:
    verification = validate_plan_r2_launch_transition(
        repository=repository,
        authorization=authorization,
        transition_receipt=transition_receipt,
        state_observation=state_observation,
        trusted_operator_dids=trusted_operator_dids,
        trusted_security_reviewer_dids=trusted_security_reviewer_dids,
        now_ms=now_ms,
    )
    source_head = str(authorization["source_head"])
    plan_root_cid = str(authorization["plan_root_cid"])
    _publish_or_confirm_plan_r2_artifact(
        repo_root,
        plan_r2_transition_receipt_relative_path(source_head, plan_root_cid),
        transition_receipt,
        noun="Plan R2 transition receipt",
    )
    _publish_or_confirm_plan_r2_artifact(
        repo_root,
        plan_r2_state_observation_relative_path(source_head, plan_root_cid),
        state_observation,
        noun="Plan R2 state observation",
    )
    return verification


__all__ = (
    "AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE",
    "ExternalAgentPlanR2Error",
    "PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA",
    "PLAN_R2_PREPARED_PROJECTION_SCHEMA",
    "PLAN_R2_STATE_OBSERVATION_SCHEMA",
    "PLAN_R2_TRANSITION_APPROVAL_SCHEMA",
    "PLAN_R2_TRANSITION_AUTHORIZATION_SCHEMA",
    "PLAN_R2_TRANSITION_RECEIPT_SCHEMA",
    "PLAN_R2_TRANSITION_STATEMENT_SCHEMA",
    "apply_authorized_plan_r2_transition",
    "assemble_plan_r2_transition_authorization",
    "assess_plan_r2_transition",
    "observe_authorized_plan_r2_transition",
    "plan_r2_operational_capability_signing_payload",
    "prepare_authorized_plan_r2_transition",
    "plan_r2_state_observation_relative_path",
    "plan_r2_transition_authorization_relative_path",
    "plan_r2_transition_receipt_relative_path",
    "prepare_plan_r2_transition_approval",
    "prepare_plan_r2_transition_authorization",
    "publish_plan_r2_transition_authorization",
    "publish_plan_r2_transition_result",
    "seal_plan_r2_operational_capability",
    "seal_plan_r2_transition_approval",
    "validate_plan_r2_launch_transition",
    "verify_plan_r2_operational_capability",
    "verify_plan_r2_transition_authorization",
)
