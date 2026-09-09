"""Closed historical denial window for retained callback reconciliation.

History identifies the original source, never completion authority. Callers must
also prove the physical failed attempts, expired current fence, preserved Portal
request, candidate ancestry and fresh target validation before native CAS.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import TASK_REVISION_HISTORY_PROJECTION_SCHEMA
from ..task_sources.typed_state_owner import (
    TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TypedStateOwnerAuthorizationError,
    _validated_database_claim_process_attestation,
)

SOURCE_REASON = "Portal completion source canonical task key mismatches"
GUARD_REASON = "retained callback recovery requires exact source seed before dispatch"
PREFLIGHT_REASON = "validation_project_dependency_preflight_failed"
IDENTITY = frozenset(
    {
        "attempt_id",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
    }
)
ROUTE = frozenset(
    {
        "execution_route_binding",
        "execution_route_policy_id",
        "execution_route_origin_revision",
    }
)
EXECUTION = frozenset(
    {"execution_phase", "execution_revision", "execution_finished_at_ms"}
)
CONTROL = frozenset({"control_expected_revision", "control_expected_status"})
TERMINAL = (
    IDENTITY
    | ROUTE
    | EXECUTION
    | CONTROL
    | {"operation", "reason", "retryable", "coordination"}
)
RETRY = (
    IDENTITY
    | ROUTE
    | EXECUTION
    | CONTROL
    | {
        "operation",
        "reason",
        "backoff_ms",
        "backoff_seconds",
        "retry_not_before_ms",
        "coordination",
        "evidence_source",
        "queue_reason",
        "queue_receipt",
        "queue_reused",
    }
)
CLAIM = (
    IDENTITY
    | ROUTE
    | {
        "operation",
        "claimed_from_revision",
        "claim_phase_schema",
        "claim_process_attestation",
        "strict_task_sharding",
        "idle_lane_work_stealing",
        "task_prefix",
        "task_shard_count",
        "task_shard_index",
    }
)


def verified_suffix(
    history: Any, *, task_cid: str, task_alias: str, control_revision: int
) -> dict[str, Any] | None:
    """Match exactly one eight-revision suffix, including both admissions.

    Additional history is allowed for verification of an already issued seed;
    callers still bind their current task snapshot and CAS independently.
    """
    if type(control_revision) is not int or control_revision < 8:
        return None
    if not isinstance(history, Mapping):
        return None
    value = dict(history)
    digest = value.pop("projection_cid", None)
    rows = value.get("revisions")
    if (
        set(history) != {"schema", "task_cid", "revisions", "projection_cid"}
        or value.get("schema") != TASK_REVISION_HISTORY_PROJECTION_SCHEMA
        or value.get("task_cid") != task_cid
        or digest != content_identity(value)
        or not isinstance(rows, list)
        or len(rows) < control_revision
        or any(
            not isinstance(x, Mapping) or set(x) != {"revision", "status", "body"}
            for x in rows
        )
        or [x["revision"] for x in rows] != list(range(1, len(rows) + 1))
    ):
        return None
    chain = rows[control_revision - 8 : control_revision]
    if [x["status"] for x in chain] != [
        "blocked",
        "retrying",
        "in_progress",
        "in_progress",
        "retrying",
        "in_progress",
        "in_progress",
        "blocked",
    ]:
        return None
    if any(not isinstance(x["body"], Mapping) for x in chain):
        return None
    bodies = [dict(x["body"]) for x in chain]
    receipts = [x.pop("completion_receipt", None) for x in bodies]
    if any(x != bodies[0] for x in bodies[1:]) or any(
        not isinstance(x, Mapping) for x in receipts
    ):
        return None
    source, rearm, middle, middle_admitted, retry, current, admitted, terminal = (
        receipts
    )
    source_revision = control_revision - 7
    if (
        set(source) != TERMINAL
        or set(terminal) != TERMINAL
        or set(rearm) != RETRY
        or set(retry) != RETRY
        or set(middle) != CLAIM
        or set(current) != CLAIM
        or source["operation"] != "database_portal_terminal_failure"
        or source["reason"] != SOURCE_REASON
        or source["retryable"] is not False
        or terminal["operation"] != "database_portal_terminal_failure"
        or terminal["reason"] != GUARD_REASON
        or terminal["retryable"] is not False
        or rearm["operation"] != "database_portal_validation_retry_recovery"
        or rearm["reason"] != "portal_completion_handshake_retry"
        or rearm["evidence_source"] != "portal_completion_handshake_reclassified"
        or rearm["backoff_ms"] != 0
        or rearm["backoff_seconds"] != 0
        or retry["operation"] != "database_portal_retry"
        or retry["reason"] != PREFLIGHT_REASON
        or retry["evidence_source"] != "typed_portal_deferral"
        or any(source[k] != rearm[k] for k in IDENTITY | EXECUTION)
        or any(middle[k] != retry[k] for k in IDENTITY)
        or any(current[k] != terminal[k] for k in IDENTITY)
    ):
        return None
    from ..task_sources.task_execution_route_policy import TaskExecutionRouteBinding

    route = source["execution_route_binding"]
    if (
        not isinstance(route, Mapping)
        or route.get("task_cid") != task_cid
        or route.get("task_alias") != task_alias
        or any(any(r[k] != source[k] for k in ROUTE) for r in receipts)
    ):
        return None
    try:
        if dict(route) != TaskExecutionRouteBinding.from_dict(route).to_dict():
            return None
    except (ValueError, TypeError):
        return None
    if any(
        middle[k] != current[k]
        for k in (
            "strict_task_sharding",
            "idle_lane_work_stealing",
            "task_prefix",
            "task_shard_count",
            "task_shard_index",
        )
    ):
        return None
    for receipt, revision, prior_status in (
        (source, source_revision, "in_progress"),
        (rearm, source_revision + 1, "blocked"),
        (retry, source_revision + 4, "in_progress"),
        (terminal, control_revision, "in_progress"),
    ):
        if (
            receipt["control_expected_revision"] != revision - 1
            or receipt["control_expected_status"] != prior_status
            or receipt["execution_phase"] != "failed"
            or receipt["execution_revision"] != 3
            or type(receipt["execution_finished_at_ms"]) is not int
            or receipt["execution_finished_at_ms"] <= 0
            or not isinstance(receipt["coordination"], Mapping)
        ):
            return None
    for receipt in (rearm, retry):
        if (
            receipt["queue_reason"]
            != "database_portal_retry:"
            + receipt["attempt_id"]
            + ":"
            + receipt["reason"]
            or not isinstance(receipt["queue_receipt"], Mapping)
            or type(receipt["queue_reused"]) is not bool
            or type(receipt["retry_not_before_ms"]) is not int
            or type(receipt["backoff_ms"]) is not int
            or receipt["backoff_ms"] < 0
            or receipt["backoff_ms"] != receipt["backoff_seconds"] * 1000
        ):
            return None
    for claim, admission, revision in (
        (middle, middle_admitted, source_revision + 2),
        (current, admitted, source_revision + 5),
    ):
        try:
            _validated_database_claim_process_attestation(claim)
        except (TypedStateOwnerAuthorizationError, ValueError, TypeError):
            return None
        if (
            claim["operation"] != "database_claim"
            or claim["claim_phase_schema"] != TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
            or claim["claimed_from_revision"] != revision - 1
            or claim["strict_task_sharding"] is not True
            or claim["idle_lane_work_stealing"] != ""
            or type(claim["task_prefix"]) is not str
            or not task_alias.startswith(claim["task_prefix"])
            or type(claim["task_shard_count"]) is not int
            or type(claim["task_shard_index"]) is not int
            or not 0 <= claim["task_shard_index"] < claim["task_shard_count"]
            or dict(admission)
            != {
                **dict(claim),
                "operation": "database_attempt_admitted",
                "claim_phase_schema": TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
                "admitted_from_revision": revision,
                "attempt_execution_phase": "claimed",
                "attempt_execution_revision": 1,
            }
        ):
            return None
    for index, receipt in enumerate((source, middle, current)):
        if (
            any(
                type(receipt[k]) is not str or not receipt[k]
                for k in ("attempt_id", "claim_id", "lease_id", "owner_session_id")
            )
            or any(
                type(receipt[k]) is not int or receipt[k] < 1
                for k in ("attempt_number", "fencing_token", "fence_epoch")
            )
            or receipt["attempt_number"] != source["attempt_number"] + index
            or receipt["fencing_token"] != source["fencing_token"] + index
            or receipt["fence_epoch"] != source["fence_epoch"] + index
            or receipt["owner_session_id"] != source["owner_session_id"]
        ):
            return None
    if any(
        len({r[k] for r in (source, middle, current)}) != 3
        for k in ("attempt_id", "claim_id", "lease_id")
    ):
        return None
    result = {
        "source_task_revision": source_revision,
        "recovery_control_revision": control_revision,
        "source_receipt": dict(source),
        "middle_receipt": dict(retry),
        "current_receipt": dict(terminal),
        "middle_claim": dict(middle),
        "current_claim": dict(current),
        "semantic_body": bodies[0],
    }
    result["context_id"] = content_identity(result)
    return result


def verified_seed_predecessor(
    history: Any,
    *,
    task_cid: str,
    task_alias: str,
    seed: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> bool:
    """Admit the ordinary typed overlay only for this exact retained seed."""
    from .implementation_daemon import (
        _DATABASE_POST_MERGE_CALLBACK_INTEGRATION_RECOVERY_RECEIPT_FIELDS,
        DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2,
    )

    context = verified_suffix(
        history,
        task_cid=task_cid,
        task_alias=task_alias,
        control_revision=seed.get("recovery_control_revision"),
    )
    if context is None:
        return False
    source = context["source_receipt"]
    revision = context["recovery_control_revision"]
    if (
        seed.get("schema") != DATABASE_POST_MERGE_COMPLETION_RECOVERY_SEED_SCHEMA_V2
        or seed.get("terminal_reason") != SOURCE_REASON
        or seed.get("task_cid") != task_cid
        or seed.get("task_alias") != task_alias
        or seed.get("source_task_revision") != context["source_task_revision"]
        or seed.get("qualification_kind") != "callback_integration"
        or any(seed.get(k) != source[k] for k in IDENTITY)
        or set(predecessor)
        != _DATABASE_POST_MERGE_CALLBACK_INTEGRATION_RECOVERY_RECEIPT_FIELDS
        | ROUTE
        | {"backoff_ms", "retry_not_before_ms"}
        or type(predecessor.get("backoff_ms")) is not int
        or predecessor["backoff_ms"] != 0
        or type(predecessor.get("retry_not_before_ms")) is not int
        or predecessor["retry_not_before_ms"] < 0
        or predecessor.get("operation")
        != "database_post_merge_declared_outputs_callback_integration_recovery"
        or predecessor.get("control_expected_revision") != revision
        or predecessor.get("control_expected_status") != "blocked"
        or predecessor.get("post_merge_completion_recovery_seed") != dict(seed)
        or any(predecessor.get(k) != source[k] for k in IDENTITY | EXECUTION | ROUTE)
        or any(
            predecessor.get(k) != seed.get(k)
            for k in ("request_id", "candidate_commit", "qualified_target_commit")
        )
        or predecessor.get("callback_requalification_receipt_id")
        != seed.get("qualification_receipt_id")
        or predecessor.get("callback_reconciliation_evidence_id")
        != seed.get("recovery_evidence_id")
        or predecessor.get("source_binding_id") != seed.get("queue_source_binding_id")
        or predecessor.get("source_projection_immutable_digest")
        != seed.get("queue_source_projection_immutable_digest")
        or not isinstance(predecessor.get("coordination"), Mapping)
        or any(
            predecessor["coordination"].get(k) != source[k]
            for k in ("attempt_id", "claim_id", "attempt_number")
        )
    ):
        return False
    rows = history["revisions"]
    return len(rows) > revision and rows[revision] == {
        "revision": revision + 1,
        "status": "retrying",
        "body": {**context["semantic_body"], "completion_receipt": dict(predecessor)},
    }
