"""Exact retained exhaustion transport; fixtures are disposable historical copies."""

import copy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.exhausted_post_merge_recovery import (
    QUEUE_RECEIPT_SCHEMA,
    build_parameters,
    command_digest,
    validate_parameters,
    validate_recovery_context,
    validate_successor_cooldown,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_RETRY_COOLDOWN_SCHEMA,
    TypedStateOwnerAuthorizationError,
    _validated_stored_retry_cooldown,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_callback_suffix import (
    IDENTITY,
)
from test.api.test_retained_callback_suffix import _seeded_dispatch_fixture


def exhausted_fixture():
    """Return task, history, prior stored lease and native-shaped transition."""
    _daemon, _bridge, _attempt, _record, _seed, transition, history = (
        _seeded_dispatch_fixture(preflight_exhaustion=True)
    )
    history["revisions"] = history["revisions"][:11]
    history.pop("projection_cid")
    history["projection_cid"] = content_identity(history)
    transition["retry_not_before_ms"] = 0
    task = {
        "task_cid": history["task_cid"],
        "task_alias": "DOEP-053",
        "revision": 11,
        "status": "blocked",
        "body": copy.deepcopy(history["revisions"][-1]["body"]),
    }
    middle = history["revisions"][7]["body"]["completion_receipt"]
    extension = {
        "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "task_cid": task["task_cid"],
        "expected_task_revision": middle["control_expected_revision"],
        **{key: middle[key] for key in IDENTITY},
        "delay_ms": middle["backoff_ms"],
        "started_at_ms": middle["retry_not_before_ms"] - middle["backoff_ms"],
        "retry_not_before_ms": middle["retry_not_before_ms"],
        "selection_penalty": 0,
        "consecutive_failures": middle["attempt_number"],
        "reason": middle["queue_reason"],
        "expected_queue_revision": 1,
        "expected_queue_attempt": 1,
    }
    prior = {
        "task_cid": task["task_cid"],
        "claim_cid": middle["claim_id"],
        "resolution_cid": content_identity(
            {
                "typed_retry_cooldown": extension,
                "started_at_ms": extension["started_at_ms"],
            }
        ),
        "claimant_did": middle["owner_session_id"],
        "logical_epoch": middle["fence_epoch"],
        "fencing_token": middle["fencing_token"],
        "expires_at_ms": 0,
        "attempt": middle["attempt_number"],
        "state": "released",
        "started_at_ms": extension["started_at_ms"],
        "release_reason": middle["queue_reason"],
        "retry_not_before_ms": middle["retry_not_before_ms"],
        "owner_session_id": middle["owner_session_id"],
        "fence_epoch": middle["fence_epoch"],
        "revision": 2,
        "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "extension_json": canonical_json_bytes(extension).decode(),
    }
    _validated_stored_retry_cooldown(prior, task_cid=task["task_cid"])
    return task, history, prior, transition


def _arguments():
    task, history, prior, transition = exhausted_fixture()
    return {
        "task": task,
        "history": history,
        "prior_queue": prior,
        "expected_control_receipt": copy.deepcopy(task["body"]["completion_receipt"]),
        "transition_receipt": transition,
    }


def _rehash(arguments):
    history = arguments["history"]
    history.pop("projection_cid", None)
    history["projection_cid"] = content_identity(history)


def test_exhausted_contract_preserves_source_and_schedules_current_fence():
    args = _arguments()
    before = copy.deepcopy(args)
    parameters = build_parameters(**args, now_ms=1_789_320_000_000)
    validated = validate_parameters(parameters)
    assert args == before
    assert validated["transition_receipt"]["attempt_number"] == 1
    assert validated["cooldown_parameters"]["attempt_number"] == 3
    assert validated["cooldown_parameters"]["expected_queue_attempt"] == 2
    receipt = validated["queue_receipt"]
    assert receipt["schema"] == QUEUE_RECEIPT_SCHEMA
    assert receipt["source_identity"]["attempt_number"] == 1
    assert receipt["current_identity"]["attempt_number"] == 3
    assert receipt["prior_queue"] == args["prior_queue"]
    assert receipt["prior_queue_cid"] == content_identity(args["prior_queue"])
    assert validated["final_transition_receipt"] == {
        **args["transition_receipt"],
        "queue_receipt": receipt,
    }
    assert command_digest(parameters) == command_digest(copy.deepcopy(parameters))


@pytest.mark.parametrize(
    "mutation",
    [
        "history_digest",
        "history_gap",
        "history_extra",
        "semantic_body",
        "task_revision_bool",
        "task_snapshot",
        "control_receipt",
        "source_reason",
        "budget_count",
        "budget_bool",
        "seed_digest",
        "seed_unknown",
        "seed_source_revision",
        "seed_current_identity",
        "seed_queue_source",
        "seed_target",
        "transition_route",
        "transition_execution",
        "transition_extra",
        "transition_backoff_bool",
        "transition_deadline",
        "prior_missing",
        "prior_active",
        "prior_newer",
        "prior_foreign_claim",
        "prior_unknown",
        "prior_bytes",
        "prior_middle_deadline",
        "middle_receipt_revision",
        "middle_receipt_bool",
    ],
)
def test_exhausted_contract_denies_unproven_history_seed_or_queue(mutation):
    args = _arguments()
    history, task, transition, prior = (
        args[key] for key in ("history", "task", "transition_receipt", "prior_queue")
    )
    seed = transition["post_merge_completion_recovery_seed"]
    rows = history["revisions"]
    if mutation == "history_digest":
        history["projection_cid"] = "foreign"
    elif mutation == "history_gap":
        rows.pop(3)
    elif mutation == "history_extra":
        rows.append(copy.deepcopy(rows[-1]))
    elif mutation == "semantic_body":
        rows[7]["body"]["title"] = "foreign"
    elif mutation == "task_revision_bool":
        task["revision"] = True
    elif mutation == "task_snapshot":
        task["body"]["title"] = "foreign"
    elif mutation == "control_receipt":
        args["expected_control_receipt"]["execution_revision"] = 999
    elif mutation == "source_reason":
        rows[3]["body"]["completion_receipt"]["reason"] = "foreign"
    elif mutation == "budget_count":
        rows[-1]["body"]["completion_receipt"]["retry_budget"][
            "typed_deferral_count"
        ] = 1
    elif mutation == "budget_bool":
        rows[-1]["body"]["completion_receipt"]["attempt_consumed"] = 0
    elif mutation == "seed_digest":
        seed["seed_id"] = "sha256:" + "0" * 64
    elif mutation == "seed_unknown":
        seed["unverified"] = True
    elif mutation == "seed_source_revision":
        seed["source_task_revision"] += 1
    elif mutation == "seed_current_identity":
        seed["attempt_number"] = 3
    elif mutation == "seed_queue_source":
        seed["queue_source_claim_id"] = "claim:foreign"
    elif mutation == "seed_target":
        seed["qualified_target_commit"] = "f" * 40
    elif mutation == "transition_route":
        transition["execution_route_origin_revision"] += 1
    elif mutation == "transition_execution":
        transition["execution_revision"] += 1
    elif mutation == "transition_extra":
        transition["unverified"] = True
    elif mutation == "transition_backoff_bool":
        transition["backoff_ms"] = False
    elif mutation == "transition_deadline":
        transition["retry_not_before_ms"] = 1
    elif mutation == "prior_missing":
        args["prior_queue"] = None
    elif mutation == "prior_active":
        prior["state"] = "accepted"
    elif mutation == "prior_newer":
        prior["attempt"] = 3
    elif mutation == "prior_foreign_claim":
        prior["claim_cid"] = "claim:foreign"
    elif mutation == "prior_unknown":
        prior["unverified"] = True
    elif mutation == "prior_bytes":
        prior["extension_json"] = "{}"
    elif mutation == "prior_middle_deadline":
        extension = json.loads(prior["extension_json"])
        extension["started_at_ms"] += 1
        extension["retry_not_before_ms"] += 1
        prior.update(
            started_at_ms=extension["started_at_ms"],
            retry_not_before_ms=extension["retry_not_before_ms"],
            extension_json=canonical_json_bytes(extension).decode(),
            resolution_cid=content_identity(
                {
                    "typed_retry_cooldown": extension,
                    "started_at_ms": extension["started_at_ms"],
                }
            ),
        )
    elif mutation == "middle_receipt_revision":
        rows[7]["body"]["completion_receipt"]["queue_receipt"]["revision"] = 1
    elif mutation == "middle_receipt_bool":
        rows[7]["body"]["completion_receipt"]["queue_receipt"]["changed"] = 1
    if mutation != "history_digest":
        _rehash(args)
    if mutation.startswith("seed_") and mutation != "seed_digest":
        seed.pop("seed_id")
        seed["seed_id"] = (
            "sha256:"
            + __import__("hashlib").sha256(canonical_json_bytes(seed)).hexdigest()
        )
    with pytest.raises(
        (TypedStateOwnerAuthorizationError, DatabaseImplementationAuthorityError)
    ):
        validate_recovery_context(**args)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "operation",
        "status",
        "control_json",
        "extension_json",
        "prior_json",
        "queue_receipt",
        "current_identity",
        "counter_bool",
        "delay",
        "penalty",
        "history_binding",
    ],
)
def test_closed_parameters_deny_malformed_or_rebound_transport(mutation):
    parameters = build_parameters(**_arguments(), now_ms=1_789_320_000_000)
    if mutation == "unknown":
        parameters["allow_unverified"] = True
    elif mutation == "operation":
        parameters["operation"] = "task.post_merge.retry.recover"
    elif mutation == "status":
        parameters["status"] = "ready"
    elif mutation == "control_json":
        parameters["expected_control_receipt_json"] += " "
    elif mutation == "extension_json":
        parameters["extension_json"] += " "
    elif mutation == "prior_json":
        parameters["expected_prior_queue_json"] += " "
    elif mutation == "queue_receipt":
        parameters["final_transition_receipt_cid"] = "foreign"
    elif mutation == "current_identity":
        parameters["attempt_id"] = "attempt:foreign"
    elif mutation == "counter_bool":
        parameters["expected_queue_attempt"] = True
    elif mutation == "delay":
        parameters["delay_ms"] = 1
    elif mutation == "penalty":
        parameters["selection_penalty"] = 1
    elif mutation == "history_binding":
        parameters["history_projection_cid"] = "foreign"
    with pytest.raises(TypedStateOwnerAuthorizationError):
        validate_parameters(parameters)


def test_exact_stored_prior_json_bytes_are_preserved_and_bound():
    args = _arguments()
    canonical = build_parameters(**args, now_ms=1_789_320_000_000)
    args["prior_queue"]["extension_json"] = json.dumps(
        json.loads(args["prior_queue"]["extension_json"]), indent=2
    )
    whitespace = build_parameters(**args, now_ms=1_789_320_000_000)
    assert command_digest(canonical) != command_digest(whitespace)
    assert (
        validate_parameters(whitespace)["queue_receipt"]["prior_queue"]
        == args["prior_queue"]
    )


def _successor_fixture():
    args = _arguments()
    parameters = build_parameters(**args, now_ms=1_789_320_000_000)
    validated = validate_parameters(parameters)
    successor = copy.deepcopy(args["task"])
    successor.update(status="retrying", revision=12)
    successor["body"]["completion_receipt"] = validated["final_transition_receipt"]
    current = {
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
        "revision": 3,
        "extension_schema": parameters["extension_schema"],
        "extension_json": parameters["extension_json"],
    }
    return successor, current, parameters


def test_successor_reconstructs_exact_original_transport_without_history_io():
    successor, cooldown, parameters = _successor_fixture()
    result = validate_successor_cooldown(task=successor, cooldown=cooldown)
    assert {key: result[key] for key in parameters} == parameters
    assert result["transition_receipt"]["attempt_number"] == 1
    assert result["cooldown_parameters"]["attempt_number"] == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "task_status",
        "task_revision",
        "task_body",
        "queue_missing",
        "queue_unknown",
        "queue_source",
        "queue_current",
        "queue_history",
        "queue_control",
        "queue_prior",
        "queue_digest",
        "lease_revision",
        "lease_claim",
        "lease_json",
        "lease_active",
        "deadline",
    ],
)
def test_successor_denies_corrupted_task_or_queue_binding(mutation):
    task, cooldown, _parameters = _successor_fixture()
    receipt = task["body"]["completion_receipt"]
    queue = receipt["queue_receipt"]
    if mutation == "task_status":
        task["status"] = "blocked"
    elif mutation == "task_revision":
        task["revision"] += 1
    elif mutation == "task_body":
        task["body"]["title"] = "foreign"
    elif mutation == "queue_missing":
        receipt.pop("queue_receipt")
    elif mutation == "queue_unknown":
        queue["allow_unverified"] = True
    elif mutation == "queue_source":
        queue["source_identity"]["attempt_id"] = "attempt:foreign"
    elif mutation == "queue_current":
        queue["current_identity"]["attempt_id"] = "attempt:foreign"
    elif mutation == "queue_history":
        queue["history_projection_cid"] = "foreign"
    elif mutation == "queue_control":
        queue["expected_control_receipt"]["claim_id"] = "claim:foreign"
    elif mutation == "queue_prior":
        queue["prior_queue"]["revision"] = 1
    elif mutation == "queue_digest":
        queue["receipt_id"] = "foreign"
    elif mutation == "lease_revision":
        cooldown["revision"] += 1
    elif mutation == "lease_claim":
        cooldown["claim_cid"] = "claim:foreign"
    elif mutation == "lease_json":
        cooldown["extension_json"] += " "
    elif mutation == "lease_active":
        cooldown["state"] = "accepted"
    elif mutation == "deadline":
        queue["retry_not_before_ms"] += 1
    with pytest.raises(TypedStateOwnerAuthorizationError):
        validate_successor_cooldown(task=task, cooldown=cooldown)
