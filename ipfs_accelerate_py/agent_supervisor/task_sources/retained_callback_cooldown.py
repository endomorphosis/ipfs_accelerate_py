"""Forward-only cooldown binding for a previously admitted retained callback.

This never accepts a candidate or rewrites its seed. The owner independently
replays the canonical history before replacing a released scheduling row.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ..proof.formal_verification_contracts import content_identity
from .control_plane_contracts import canonical_json_bytes

SCHEMA = "ipfs_accelerate_py/agent-supervisor/retained-callback-cooldown-binding@1"
FIELD = "retained_callback_binding"
SEED_FIELDS = frozenset(
    {
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
        "recovery_control_revision",
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
)


def build_binding(
    *,
    task: Mapping[str, Any],
    history: Mapping[str, Any],
    prior_queue: Mapping[str, Any],
) -> dict[str, Any]:
    from ..todo_daemon.retained_callback_suffix import (
        EXECUTION,
        IDENTITY,
        ROUTE,
        SOURCE_REASON,
        verified_suffix,
    )
    from .typed_state_owner import (
        TypedStateOwnerAuthorizationError as Error,
    )
    from .typed_state_owner import (
        _validated_stored_retry_cooldown,
    )

    revision = task.get("revision")
    body = task.get("body")
    receipt = body.get("completion_receipt") if isinstance(body, Mapping) else None
    seed = (
        receipt.get("post_merge_completion_recovery_seed")
        if isinstance(receipt, Mapping)
        else None
    )
    if (
        task.get("status") != "retrying"
        or type(revision) is not int
        or not isinstance(seed, Mapping)
        or revision < 9
        or set(seed) != SEED_FIELDS
        or seed.get("task_cid") != task.get("task_cid")
        or seed.get("task_alias") != task.get("task_alias")
    ):
        raise Error("retained callback cooldown requires an exact retrying seed")
    context = verified_suffix(
        history,
        task_cid=task["task_cid"],
        task_alias=task["task_alias"],
        control_revision=revision - 1,
    )
    if context is None or len(history["revisions"]) != revision:
        raise Error("retained callback cooldown history does not reproduce")
    if history["revisions"][-1] != {
        "revision": revision,
        "status": "retrying",
        "body": body,
    }:
        raise Error("retained callback cooldown current task changed")
    semantic = dict(body)
    semantic.pop("completion_receipt")
    source = context["source_receipt"]
    middle = context["middle_receipt"]
    latest = context["current_receipt"]
    seed_material = dict(seed)
    seed_id = seed_material.pop("seed_id")
    common = IDENTITY | EXECUTION | ROUTE
    expected_fields = common | {
        "operation",
        "request_id",
        "candidate_commit",
        "source_binding_id",
        "source_projection_immutable_digest",
        "queue_reason",
        "queue_receipt",
        "coordination",
        "control_expected_status",
        "control_expected_revision",
        "source_integration_commit",
        "source_train_receipt_id",
        "qualified_target_commit",
        "callback_requalification_receipt_id",
        "callback_reconciliation_evidence_id",
        "post_merge_completion_recovery_seed",
        "backoff_ms",
        "retry_not_before_ms",
    }
    if (
        semantic != context["semantic_body"]
        or set(receipt) != expected_fields
        or seed.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/database-post-merge-completion-recovery-seed@2"
        or seed_id
        != "sha256:" + hashlib.sha256(canonical_json_bytes(seed_material)).hexdigest()
        or seed.get("source_task_revision") != context["source_task_revision"]
        or seed.get("recovery_control_revision") != revision - 1
        or seed.get("terminal_reason") != SOURCE_REASON
        or seed.get("qualification_kind") != "callback_integration"
        or any(seed.get(k) != source[k] for k in IDENTITY)
        or any(receipt.get(k) != source[k] for k in common)
        or receipt.get("operation")
        != "database_post_merge_declared_outputs_callback_integration_recovery"
        or receipt.get("control_expected_status") != "blocked"
        or receipt.get("control_expected_revision") != revision - 1
        or type(receipt.get("backoff_ms")) is not int
        or receipt["backoff_ms"] != 0
        or type(receipt.get("retry_not_before_ms")) is not int
        or receipt["retry_not_before_ms"] < 0
        or any(
            receipt.get(k) != seed.get(k)
            for k in ("request_id", "candidate_commit", "qualified_target_commit")
        )
        or receipt.get("callback_requalification_receipt_id")
        != seed.get("qualification_receipt_id")
        or receipt.get("callback_reconciliation_evidence_id")
        != seed.get("recovery_evidence_id")
        or receipt.get("source_binding_id") != seed.get("queue_source_binding_id")
        or receipt.get("source_projection_immutable_digest")
        != seed.get("queue_source_projection_immutable_digest")
        or receipt.get("queue_reason")
        != "database_post_merge_declared_outputs_callback_integration:"
        + seed["request_id"]
        + ":"
        + seed["qualification_receipt_id"]
        or not isinstance(receipt.get("coordination"), Mapping)
        or any(
            receipt["coordination"].get(k) != source[k]
            for k in ("attempt_id", "claim_id", "attempt_number")
        )
    ):
        raise Error("retained callback cooldown source receipt or seed changed")
    if FIELD in json_extension(prior_queue):
        raise Error("retained callback cooldown cannot recursively migrate")
    queue = _validated_stored_retry_cooldown(prior_queue, task_cid=task["task_cid"])
    extension = queue["extension"]
    if FIELD in extension:
        raise Error("retained callback cooldown cannot recursively migrate")
    expected = {
        **{k: middle[k] for k in IDENTITY},
        "expected_task_revision": middle["control_expected_revision"],
        "reason": middle["queue_reason"],
        "delay_ms": middle["backoff_ms"],
        "retry_not_before_ms": middle["retry_not_before_ms"],
    }
    if any(
        type(extension.get(k)) is not type(v) or extension.get(k) != v
        for k, v in expected.items()
    ):
        raise Error(
            "retained callback cooldown prior queue differs from its historical retry"
        )
    material = {
        "schema": SCHEMA,
        "task_cid": task["task_cid"],
        "task_alias": task["task_alias"],
        "task_revision": revision,
        "source_receipt_cid": content_identity(dict(receipt)),
        "source_seed_id": seed_id,
        "source_receipt": dict(receipt),
        "task_body_cid": content_identity(dict(body)),
        "history_projection_cid": history["projection_cid"],
        "prior_queue": {k: v for k, v in queue.items() if k != "extension"},
        "identity": {k: latest[k] for k in sorted(IDENTITY)},
    }
    return {**material, "binding_id": content_identity(material)}


def validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact witness; live history is rederived by the owner."""
    from ..todo_daemon.retained_callback_suffix import IDENTITY
    from .typed_state_owner import TypedStateOwnerAuthorizationError as Error
    from .typed_state_owner import _validated_stored_retry_cooldown

    try:
        value = dict(binding)
        binding_id = value.pop("binding_id")
        if (
            set(value)
            != {
                "schema",
                "task_cid",
                "task_alias",
                "task_revision",
                "source_receipt_cid",
                "source_seed_id",
                "source_receipt",
                "task_body_cid",
                "history_projection_cid",
                "prior_queue",
                "identity",
            }
            or value["schema"] != SCHEMA
        ):
            raise Error("retained callback cooldown binding schema changed")
        source = value["source_receipt"]
        seed = source["post_merge_completion_recovery_seed"]
        identity = value["identity"]
        prior = value["prior_queue"]
        if FIELD in json_extension(prior):
            raise Error("retained callback cooldown cannot recursively migrate")
        prior = _validated_stored_retry_cooldown(prior, task_cid=value["task_cid"])
        if (
            binding_id != content_identity(value)
            or value["source_receipt_cid"] != content_identity(source)
            or value["source_seed_id"] != seed["seed_id"]
            or seed["task_cid"] != value["task_cid"]
            or seed["task_alias"] != value["task_alias"]
            or type(value["task_revision"]) is not int
            or value["task_revision"] < 9
            or source["control_expected_revision"] != value["task_revision"] - 1
            or set(identity) != IDENTITY
            or any(
                type(identity[k]) is not int
                or identity[k] != source[k] + 2
                or identity[k] != prior["extension"][k] + 1
                for k in ("attempt_number", "fencing_token", "fence_epoch")
            )
            or identity["owner_session_id"] != source["owner_session_id"]
            or identity["owner_session_id"] != prior["owner_session_id"]
            or any(
                type(identity[k]) is not str
                or not identity[k]
                or identity[k] in {source[k], prior["extension"][k]}
                for k in ("attempt_id", "claim_id", "lease_id")
            )
        ):
            raise Error("retained callback cooldown binding changed")
        return {**value, "binding_id": binding_id}
    except (KeyError, TypeError, IndexError, ValueError, RecursionError) as exc:
        raise Error("retained callback cooldown binding is malformed") from exc


def payload_from_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_binding(binding)
    receipt = value["source_receipt"]
    return {
        "task_cid": value["task_cid"],
        "expected_task_revision": value["task_revision"] - 1,
        "expected_task_status": "retrying",
        **value["identity"],
        "delay_ms": receipt["backoff_ms"],
        "reason": receipt["queue_reason"],
        "now_ms": receipt["retry_not_before_ms"] - receipt["backoff_ms"],
        "selection_penalty": json_extension(value["prior_queue"])["selection_penalty"],
        FIELD: value,
    }


def json_extension(queue: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(queue["extension_json"])


def require_claim_binding(
    connection: Any, *, task: Mapping[str, Any], next_receipt: Mapping[str, Any]
) -> None:
    """Bind a fresh owner reservation to both source seed and forward floor."""
    from .typed_state_owner import TypedStateOwnerAuthorizationError as Error
    from .typed_state_owner import _validated_stored_retry_cooldown

    receipt = task["body"].get("completion_receipt", {})
    seed = receipt.get("post_merge_completion_recovery_seed", {})
    if not (
        task["status"] == "retrying"
        and isinstance(seed, Mapping)
        and seed.get("terminal_reason")
        == "Portal completion source canonical task key mismatches"
        and seed.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/database-post-merge-completion-recovery-seed@2"
    ):
        return
    rows = connection.execute(
        "SELECT task_cid, claim_cid, resolution_cid, claimant_did, logical_epoch, fencing_token, expires_at_ms, attempt, state, started_at_ms, release_reason, retry_not_before_ms, owner_session_id, fence_epoch, revision, extension_schema, extension_json FROM leases WHERE task_cid = ? LIMIT 2",
        [task["task_cid"]],
    ).fetchall()
    if len(rows) != 1:
        raise Error("retained callback reservation has no unique cooldown")
    queue = _validated_stored_retry_cooldown(rows[0], task_cid=task["task_cid"])
    binding = queue["extension"].get(FIELD)
    if not isinstance(binding, Mapping):
        raise Error("retained callback reservation requires its forward cooldown")
    import json

    rows = connection.execute(
        "SELECT revision, status, body_json FROM task_revisions WHERE task_cid = ? ORDER BY revision LIMIT 25",
        [task["task_cid"]],
    ).fetchall()
    history = {
        "schema": "ipfs_accelerate_py/agent-supervisor/task-revision-history-projection@1",
        "task_cid": task["task_cid"],
        "revisions": [
            {"revision": int(r[0]), "status": str(r[1]), "body": json.loads(r[2])}
            for r in rows
        ],
    }
    history["projection_cid"] = content_identity(history)
    expected = build_binding(
        task=task, history=history, prior_queue=binding["prior_queue"]
    )
    if (
        expected != binding
        or next_receipt.get("post_merge_completion_recovery_seed") != seed
        or next_receipt.get("post_merge_completion_recovery_source_attempt_id")
        != seed["attempt_id"]
        or any(
            type(next_receipt.get(k)) is not int
            or next_receipt[k] <= binding["identity"][k]
            for k in ("attempt_number", "fencing_token", "fence_epoch")
        )
        or any(
            next_receipt.get(k) in {seed[k], binding["identity"][k]}
            for k in ("attempt_id", "claim_id", "lease_id")
        )
    ):
        raise Error(
            "retained callback reservation changed source or regressed its floor"
        )
