"""Closed transport material for an exhausted retained-callback recovery.

This module performs no I/O. The owner must rebuild parameters from its own
task, history and lease rows inside the transaction before authorizing writes.
The daemon separately verifies physical attempts, coordination and fresh Git
qualification. A digest or caller-provided history is not that authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from ..proof.formal_verification_contracts import content_identity
from .control_plane_contracts import canonical_json_bytes

COMMAND = "task.post_merge.exhausted_retry.recover"
SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-exhausted-post-merge-recovery@1"
)
QUEUE_RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/typed-database-exhausted-post-merge-queue-receipt@1"

_IDENTITY = frozenset(
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
_COOLDOWN_FIELDS = frozenset(
    {
        "task_cid",
        "expected_task_revision",
        "expected_task_status",
        *_IDENTITY,
        "delay_ms",
        "started_at_ms",
        "retry_not_before_ms",
        "selection_penalty",
        "consecutive_failures",
        "reason",
        "resolution_cid",
        "expected_queue_revision",
        "expected_queue_attempt",
        "extension_schema",
        "extension_json",
    }
)
_FIELDS = _COOLDOWN_FIELDS | {
    "schema",
    "operation",
    "status",
    "task_alias",
    "history_projection_cid",
    "context_id",
    "expected_task_body_cid",
    "expected_control_receipt_json",
    "transition_operation",
    "transition_receipt_json",
    "final_transition_receipt_cid",
    "expected_prior_queue_json",
}


def _deny(message: str) -> None:
    from .typed_state_owner import TypedStateOwnerAuthorizationError

    raise TypedStateOwnerAuthorizationError("exhausted post-merge recovery " + message)


def _same(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _object(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not str:
        _deny(name + " is not canonical JSON")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        _deny(name + " is malformed")
    if not isinstance(decoded, dict) or _json(decoded) != value:
        _deny(name + " is not a canonical object")
    return decoded


def _identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: receipt.get(key) for key in _IDENTITY}
    for key, value in result.items():
        if key in {"attempt_number", "fencing_token", "fence_epoch"}:
            if type(value) is not int or value < 1:
                _deny("has an invalid identity counter")
        elif type(value) is not str or not value.strip() or value != value.strip():
            _deny("has an invalid identity string")
    return result


def _transition(
    value: Any, *, task_cid: str, task_alias: str, revision: int
) -> dict[str, Any]:
    from .typed_state_owner import _validated_post_merge_retry_transition
    from ..todo_daemon.implementation_daemon import (
        DatabaseImplementationAuthorityError,
        DatabaseImplementationDaemon,
    )

    if not isinstance(value, Mapping):
        _deny("transition is not an object")
    transition = dict(value)
    if (
        transition.get("operation")
        != "database_post_merge_declared_outputs_callback_integration_recovery"
        or type(transition.get("backoff_ms")) is not int
        or transition["backoff_ms"] != 0
        or type(transition.get("retry_not_before_ms")) is not int
        or transition["retry_not_before_ms"] != 0
    ):
        _deny("requires the exact immediate retained callback transition")
    # This existing verifier is pure and needs only its static digest method.
    verifier = object.__new__(DatabaseImplementationDaemon)
    try:
        seed = verifier._verified_post_merge_completion_recovery_seed(
            transition.get("post_merge_completion_recovery_seed")
        )
    except DatabaseImplementationAuthorityError:
        _deny("seed is invalid")
    if (
        seed.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/database-post-merge-completion-recovery-seed@2"
        or seed.get("task_alias") != task_alias
        or seed.get("terminal_reason")
        != "Portal callback reconciliation binding is invalid"
        or any(
            seed.get("queue_source_" + key) != seed.get(key)
            for key in (
                "attempt_id",
                "claim_id",
                "lease_id",
                "fencing_token",
                "fence_epoch",
            )
        )
    ):
        _deny("seed does not bind the original retained callback")
    ordinary = {
        key: item
        for key, item in transition.items()
        if key not in {"backoff_ms", "retry_not_before_ms"}
    }
    _validated_post_merge_retry_transition(
        ordinary,
        task_cid=task_cid,
        expected_task_revision=revision,
        queue_reason=str(transition.get("queue_reason") or ""),
        attempt_identity=_identity(transition),
        recovery_seed_json=_json(seed),
    )
    return transition


def _prior_row(prior_queue: Any, *, task_cid: str) -> dict[str, Any]:
    from .typed_state_owner import (
        _RETRY_COOLDOWN_ROW_FIELDS,
        _validated_stored_retry_cooldown,
    )

    if not isinstance(prior_queue, Mapping):
        _deny("requires the exact prior released cooldown")
    # Accept a validated-row convenience member, but never silently discard
    # arbitrary caller fields when sealing the stored byte representation.
    if set(prior_queue) not in (
        set(_RETRY_COOLDOWN_ROW_FIELDS),
        set(_RETRY_COOLDOWN_ROW_FIELDS) | {"extension"},
    ):
        _deny("prior cooldown has a foreign field")
    row = {key: prior_queue[key] for key in _RETRY_COOLDOWN_ROW_FIELDS}
    validated = _validated_stored_retry_cooldown(row, task_cid=task_cid)
    if "extension" in prior_queue and not _same(
        prior_queue["extension"], validated["extension"]
    ):
        _deny("prior cooldown parsed bytes differ")
    return row


def _bind_middle(
    row: Mapping[str, Any], middle: Mapping[str, Any], *, task_cid: str
) -> None:
    from .typed_state_owner import (
        TYPED_RETRY_COOLDOWN_SCHEMA,
        _validated_stored_retry_cooldown,
    )

    prior = _validated_stored_retry_cooldown(row, task_cid=task_cid)
    extension = prior["extension"]
    expected = {
        **_identity(middle),
        "task_cid": task_cid,
        "expected_task_revision": middle["control_expected_revision"],
        "delay_ms": middle["backoff_ms"],
        "reason": middle["queue_reason"],
        "retry_not_before_ms": middle["retry_not_before_ms"],
        "selection_penalty": 0,
    }
    if any(not _same(extension.get(key), value) for key, value in expected.items()):
        _deny("prior cooldown does not reproduce the middle retry")
    receipt = middle.get("queue_receipt")
    details = receipt.get("details") if isinstance(receipt, Mapping) else None
    expected_details = {
        "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "operation": "task.retry.cooldown.record",
        "task_cid": task_cid,
        "expected_task_revision": middle["control_expected_revision"],
        "attempt_id": middle["attempt_id"],
        "claim_id": middle["claim_id"],
        "attempt_number": middle["attempt_number"],
        "queue_revision": row["revision"],
        "retry_not_before_ms": middle["retry_not_before_ms"],
        "reason": middle["queue_reason"],
    }
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(details, Mapping)
        or set(receipt)
        != {
            "schema",
            "event_id",
            "event_type",
            "global_sequence",
            "recorded_at",
            "subject_id",
            "revision",
            "changed",
            "details",
        }
        or set(details) != set(expected_details) | {"store_revision_before"}
        or any(
            not _same(details.get(key), value)
            for key, value in expected_details.items()
        )
        or receipt.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/intent-receipt@1"
        or receipt.get("event_type") != "TASK_RETRY_COOLDOWN_RECORDED"
        or receipt.get("recorded_at") != "typed-state-owner"
        or receipt.get("subject_id") != task_cid
        or not _same(receipt.get("revision"), row["revision"])
        or not _same(receipt.get("global_sequence"), 0)
        or type(receipt.get("changed")) is not bool
        or type(details.get("store_revision_before")) is not int
        or details["store_revision_before"] < 0
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("event_id") or ""))
        is None
    ):
        _deny("middle retry receipt does not bind its exact cooldown")


def validate_recovery_context(
    *,
    task: Any,
    history: Any,
    expected_control_receipt: Any,
    transition_receipt: Any,
    prior_queue: Any,
) -> dict[str, Any]:
    from ..todo_daemon.retained_callback_suffix import EXECUTION, ROUTE, verified_suffix

    get = (
        task.get if isinstance(task, Mapping) else lambda key: getattr(task, key, None)
    )
    task_cid, task_alias, revision, body = (
        get(key) for key in ("task_cid", "task_alias", "revision", "body")
    )
    if (
        type(revision) is not int
        or get("status") != "blocked"
        or not isinstance(body, Mapping)
    ):
        _deny("has no exact blocked task")
    context = verified_suffix(
        history, task_cid=task_cid, task_alias=task_alias, control_revision=revision
    )
    if (
        context is None
        or context.get("preflight_exhaustion") is not True
        or len(history["revisions"]) != revision
        or not _same(
            history["revisions"][-1],
            {"revision": revision, "status": "blocked", "body": dict(body)},
        )
        or not _same(body.get("completion_receipt"), expected_control_receipt)
        or not _same(context["current_receipt"], expected_control_receipt)
    ):
        _deny("canonical exhausted history does not reproduce")
    transition = _transition(
        transition_receipt, task_cid=task_cid, task_alias=task_alias, revision=revision
    )
    source, current = context["source_receipt"], context["current_receipt"]
    seed = transition["post_merge_completion_recovery_seed"]
    if seed["source_task_revision"] != context["source_task_revision"] or any(
        not _same(transition.get(key), source[key])
        for key in _IDENTITY | EXECUTION | ROUTE
    ):
        _deny("transition is not bound to the original source")
    prior = _prior_row(prior_queue, task_cid=task_cid)
    _bind_middle(prior, context["middle_receipt"], task_cid=task_cid)
    return {
        **context,
        "task_cid": task_cid,
        "task_alias": task_alias,
        "history_projection_cid": history["projection_cid"],
        "expected_task_body_cid": content_identity(dict(body)),
        "source_identity": _identity(source),
        "current_identity": _identity(current),
        "transition_receipt": transition,
        "expected_prior_queue": prior,
    }


def _queue_receipt(
    parameters: Mapping[str, Any],
    transition: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": QUEUE_RECEIPT_SCHEMA,
        "operation": "task.post_merge.exhausted_retry.cooldown",
        "task_cid": parameters["task_cid"],
        "task_alias": parameters["task_alias"],
        "expected_task_revision": parameters["expected_task_revision"],
        "source_identity": _identity(transition),
        "current_identity": _identity(parameters),
        "history_projection_cid": parameters["history_projection_cid"],
        "context_id": parameters["context_id"],
        "expected_task_body_cid": parameters["expected_task_body_cid"],
        "expected_control_receipt": _object(
            parameters["expected_control_receipt_json"], "control"
        ),
        "prior_queue": dict(prior),
        "prior_queue_cid": content_identity(dict(prior)),
        "transition_receipt_cid": content_identity(dict(transition)),
        "queue_reason": parameters["reason"],
        "retry_not_before_ms": parameters["retry_not_before_ms"],
        "expected_queue_revision": parameters["expected_queue_revision"],
        "queue_revision": parameters["expected_queue_revision"] + 1,
        "resolution_cid": parameters["resolution_cid"],
    }
    return {**body, "receipt_id": content_identity(body)}


def build_parameters(
    *,
    task: Any,
    history: Any,
    expected_control_receipt: Any,
    transition_receipt: Any,
    prior_queue: Any,
    now_ms: int,
) -> dict[str, Any]:
    from .typed_state_owner import TYPED_RETRY_COOLDOWN_SCHEMA

    if type(now_ms) is not int or now_ms < 0:
        _deny("clock is invalid")
    context = validate_recovery_context(
        task=task,
        history=history,
        expected_control_receipt=expected_control_receipt,
        transition_receipt=transition_receipt,
        prior_queue=prior_queue,
    )
    prior, transition = context["expected_prior_queue"], context["transition_receipt"]
    extension = {
        "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "task_cid": context["task_cid"],
        "expected_task_revision": context["recovery_control_revision"],
        **context["current_identity"],
        "delay_ms": 0,
        "started_at_ms": now_ms,
        "retry_not_before_ms": now_ms,
        "selection_penalty": 0,
        "consecutive_failures": context["current_identity"]["attempt_number"],
        "reason": transition["queue_reason"],
        "expected_queue_revision": prior["revision"],
        "expected_queue_attempt": prior["attempt"],
    }
    parameters = {
        **extension,
        "schema": SCHEMA,
        "operation": COMMAND,
        "expected_task_status": "blocked",
        "status": "retrying",
        "task_alias": context["task_alias"],
        "history_projection_cid": context["history_projection_cid"],
        "context_id": context["context_id"],
        "expected_task_body_cid": context["expected_task_body_cid"],
        "expected_control_receipt_json": _json(expected_control_receipt),
        "transition_operation": transition["operation"],
        "transition_receipt_json": _json(transition),
        "expected_prior_queue_json": _json(prior),
        "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "extension_json": _json(extension),
        "resolution_cid": content_identity(
            {"typed_retry_cooldown": extension, "started_at_ms": now_ms}
        ),
    }
    queue_receipt = _queue_receipt(parameters, transition, prior)
    parameters["final_transition_receipt_cid"] = content_identity(
        {**transition, "queue_receipt": queue_receipt}
    )
    validate_parameters(parameters)
    return parameters


def validate_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate transport bytes; owner-derived canonical history is still required."""
    from .typed_state_owner import (
        TYPED_RETRY_COOLDOWN_SCHEMA,
        _validated_retry_cooldown_parameters,
    )

    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        _deny("parameters differ from the closed schema")
    parameters = dict(value)
    if (
        parameters["schema"] != SCHEMA
        or parameters["operation"] != COMMAND
        or parameters["expected_task_status"] != "blocked"
        or parameters["status"] != "retrying"
        or type(parameters["expected_task_revision"]) is not int
        or parameters["expected_task_revision"] < 8
        or any(
            type(parameters[key]) is not str or not parameters[key].strip()
            for key in (
                "task_alias",
                "history_projection_cid",
                "context_id",
                "expected_task_body_cid",
            )
        )
        or not _same(parameters["delay_ms"], 0)
        or not _same(parameters["selection_penalty"], 0)
    ):
        _deny("parameters have invalid control or queue policy")
    transition = _transition(
        _object(parameters["transition_receipt_json"], "transition"),
        task_cid=parameters["task_cid"],
        task_alias=parameters["task_alias"],
        revision=parameters["expected_task_revision"],
    )
    control = _object(parameters["expected_control_receipt_json"], "control")
    if (
        control.get("operation") != "database_portal_typed_deferral_budget_exhausted"
        or control.get("reason") != "typed_portal_deferral_budget_exhausted"
        or not _same(_identity(control), _identity(parameters))
        or any(
            control[key] != transition[key] + 2
            for key in ("attempt_number", "fencing_token", "fence_epoch")
        )
        or control.get("owner_session_id") != transition.get("owner_session_id")
    ):
        _deny("source and current identities do not reproduce")
    prior = _prior_row(
        _object(parameters["expected_prior_queue_json"], "prior queue"),
        task_cid=parameters["task_cid"],
    )
    if (
        not _same(parameters["expected_queue_revision"], prior["revision"])
        or not _same(parameters["expected_queue_attempt"], prior["attempt"])
        or prior["attempt"] + 1 != control["attempt_number"]
        or prior["fencing_token"] + 1 != control["fencing_token"]
        or prior["fence_epoch"] + 1 != control["fence_epoch"]
        or prior["owner_session_id"] != control["owner_session_id"]
    ):
        _deny("current scheduling fence does not follow the exact prior cooldown")
    cooldown = {key: parameters[key] for key in _COOLDOWN_FIELDS}
    cooldown.update(
        schema=TYPED_RETRY_COOLDOWN_SCHEMA,
        operation="task.retry.cooldown.record",
        expected_task_status="in_progress",
    )
    _validated_retry_cooldown_parameters(cooldown)
    queue_receipt = _queue_receipt(parameters, transition, prior)
    final = {**transition, "queue_receipt": queue_receipt}
    if (
        parameters["transition_operation"] != transition["operation"]
        or parameters["reason"] != transition["queue_reason"]
        or parameters["final_transition_receipt_cid"] != content_identity(final)
    ):
        _deny("final transition differs from its owner-derived queue receipt")
    return {
        **parameters,
        "cooldown_parameters": cooldown,
        "expected_control_receipt": control,
        "transition_receipt": transition,
        "final_transition_receipt": final,
        "final_transition_receipt_json": _json(final),
        "queue_receipt": queue_receipt,
        "expected_prior_queue": prior,
    }


def command_digest(value: Mapping[str, Any]) -> str:
    validated = validate_parameters(value)
    return hashlib.sha256(
        canonical_json_bytes({key: validated[key] for key in _FIELDS})
    ).hexdigest()


def validate_successor_cooldown(*, task: Any, cooldown: Any) -> dict[str, Any]:
    """Reproduce the complete durable task/lease binding, without history I/O.

    Historical authority is independently checked by the owner and by the
    adapter's replay path. This check rejects partial or foreign queue receipts
    during ordinary task selection and cannot authorize a new recovery.
    """
    get = (
        task.get if isinstance(task, Mapping) else lambda key: getattr(task, key, None)
    )
    task_cid, task_alias, revision, body = (
        get(key) for key in ("task_cid", "task_alias", "revision", "body")
    )
    if (
        get("status") != "retrying"
        or type(revision) is not int
        or not isinstance(body, Mapping)
    ):
        _deny("successor has no exact retrying task")
    final = body.get("completion_receipt")
    queue = final.get("queue_receipt") if isinstance(final, Mapping) else None
    if not isinstance(queue, Mapping) or queue.get("schema") != QUEUE_RECEIPT_SCHEMA:
        _deny("successor has no closed queue receipt")
    required = {
        "schema",
        "operation",
        "task_cid",
        "task_alias",
        "expected_task_revision",
        "source_identity",
        "current_identity",
        "history_projection_cid",
        "context_id",
        "expected_task_body_cid",
        "expected_control_receipt",
        "prior_queue",
        "prior_queue_cid",
        "transition_receipt_cid",
        "queue_reason",
        "retry_not_before_ms",
        "expected_queue_revision",
        "queue_revision",
        "resolution_cid",
        "receipt_id",
    }
    if set(queue) != required or not isinstance(
        queue["expected_control_receipt"], Mapping
    ):
        _deny("successor queue receipt has an incomplete shape")
    current = _prior_row(cooldown, task_cid=task_cid)
    extension = _object(current["extension_json"], "successor extension")
    transition = {**dict(final), "queue_receipt": {}}
    parameters = {
        **extension,
        "schema": SCHEMA,
        "operation": COMMAND,
        "expected_task_status": "blocked",
        "status": "retrying",
        "task_alias": task_alias,
        "history_projection_cid": queue["history_projection_cid"],
        "context_id": queue["context_id"],
        "expected_task_body_cid": queue["expected_task_body_cid"],
        "expected_control_receipt_json": _json(queue["expected_control_receipt"]),
        "transition_operation": transition.get("operation"),
        "transition_receipt_json": _json(transition),
        "final_transition_receipt_cid": content_identity(dict(final)),
        "expected_prior_queue_json": _json(queue["prior_queue"]),
        "extension_schema": current["extension_schema"],
        "extension_json": current["extension_json"],
        "resolution_cid": current["resolution_cid"],
    }
    validated = validate_parameters(parameters)
    expected_current = {
        "task_cid": parameters["task_cid"],
        "claim_cid": parameters["claim_id"],
        "resolution_cid": parameters["resolution_cid"],
        "claimant_did": parameters["owner_session_id"],
        "logical_epoch": parameters["fence_epoch"],
        "fencing_token": parameters["fencing_token"],
        "expires_at_ms": 0,
        "attempt": parameters["attempt_number"],
        "state": "released",
        "started_at_ms": parameters["started_at_ms"],
        "release_reason": parameters["reason"],
        "retry_not_before_ms": parameters["retry_not_before_ms"],
        "owner_session_id": parameters["owner_session_id"],
        "fence_epoch": parameters["fence_epoch"],
        "revision": parameters["expected_queue_revision"] + 1,
        "extension_schema": parameters["extension_schema"],
        "extension_json": parameters["extension_json"],
    }
    if (
        task_cid != parameters["task_cid"]
        or task_alias != queue["task_alias"]
        or revision != parameters["expected_task_revision"] + 1
        or not _same(current, expected_current)
        or content_identity(
            {**dict(body), "completion_receipt": queue["expected_control_receipt"]}
        )
        != parameters["expected_task_body_cid"]
    ):
        _deny("successor task or cooldown differs from its exact binding")
    return validated
