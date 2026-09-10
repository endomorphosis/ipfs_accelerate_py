"""An exact stale callback consumer may requalify without regressing its fence."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.retained_callback_cooldown import (
    FIELD,
    build_binding,
    payload_from_binding,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TypedStateOwnerError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_callback_suffix import (
    EVIDENCE_REASON,
    IDENTITY,
    verified_suffix,
)
from test.api.test_retained_callback_cooldown import fixture, forward
from test.api.test_retained_callback_suffix import _rehash, _seeded_dispatch_fixture


def generation_fixture():
    d, b, a, record, seed, predecessor, h = _seeded_dispatch_fixture()
    old = h["revisions"][14]["body"]["completion_receipt"]
    receipt = {
        **old,
        **{k: getattr(a, k) for k in IDENTITY},
        "control_expected_revision": 18,
        "reason": EVIDENCE_REASON,
        "coordination": {},
        "execution_finished_at_ms": old["execution_finished_at_ms"] + 1000,
    }
    h["revisions"].append(
        {
            "revision": 19,
            "status": "blocked",
            "body": {**record.body, "completion_receipt": receipt},
        }
    )
    _rehash(h)
    return d, a, seed, h


@pytest.mark.parametrize(
    "mutation",
    [None, "seed", "admission", "route", "fence", "reason", "semantic", "hash", "phase_revision"],
)
def test_generation_history_is_exact(mutation):
    d, a, seed, h = generation_fixture()
    if mutation == "seed":
        seed["qualified_target_commit"] = "f" * 40
    elif mutation == "admission":
        h["revisions"][17]["body"]["completion_receipt"]["admitted_from_revision"] = 16
    elif mutation == "route":
        h["revisions"][18]["body"]["completion_receipt"]["execution_route_policy_id"] = "foreign"
    elif mutation == "fence":
        h["revisions"][18]["body"]["completion_receipt"]["fencing_token"] += 1
    elif mutation == "reason":
        h["revisions"][18]["body"]["completion_receipt"]["reason"] = "arbitrary failure"
    elif mutation == "semantic":
        h["revisions"][18]["body"]["title"] = "foreign"
    elif mutation == "phase_revision":
        h["revisions"][18]["body"]["completion_receipt"]["execution_revision"] = 4
    _rehash(h)
    if mutation == "hash":
        h["projection_cid"] = "foreign"
    result = verified_suffix(h, task_cid=a.task_cid, task_alias=a.task_alias, control_revision=19)
    assert (result is not None) == (mutation is None)
    if result:
        assert result["source_seed"] == seed
        assert result["current_receipt"]["attempt_number"] == 5
        assert result["source_receipt"]["attempt_number"] == 2


@pytest.mark.parametrize(
    "mutation", [None, "provider_phase", "seed", "latest", "target", "live_fence"]
)
def test_generation_physical_context(mutation):
    d, a, seed, h = generation_fixture()
    receipt = h["revisions"][-1]["body"]["completion_receipt"]
    attempts = d.get_attempt.__self__
    old = list(attempts.values())[-1]
    current = replace(
        old,
        **{k: getattr(a, k) for k in IDENTITY},
        finished_at_ms=receipt["execution_finished_at_ms"],
        body={
            "post_merge_completion_recovery_seed": copy.deepcopy(seed),
            "post_merge_completion_recovery_source_attempt_id": seed["attempt_id"],
        },
    )
    attempts[current.attempt_id] = current
    phase_map = d.phase_history.__self__
    phases = copy.deepcopy(phase_map[old.attempt_id])
    for p in phases:
        p.update(
            attempt_id=current.attempt_id,
            fencing_token=current.fencing_token,
            fence_epoch=current.fence_epoch,
        )
    phases[-1]["committed_at_ms"] = current.finished_at_ms
    phases[-1]["body"]["reason"] = EVIDENCE_REASON
    phase_map[current.attempt_id] = phases
    d._post_merge_completion_target_advanced = lambda *args, **kwargs: mutation != "target"
    if mutation == "provider_phase":
        phases[1]["phase"] = "provider"
    elif mutation == "seed":
        current.body["post_merge_completion_recovery_seed"]["seed_id"] = "foreign"
    elif mutation == "latest":
        d._local_attempt_is_exact_latest = lambda a: False
    elif mutation == "live_fence":
        d._terminal_coordination_reproduces_read_only = lambda *args, **kwargs: False
    task = SimpleNamespace(task_cid=a.task_cid, task_alias=a.task_alias, **h["revisions"][-1])
    result = d._retained_callback_suffix_context(task)
    assert (result is not None) == (mutation is None)


def refresh_queue_fixture():
    old_task, old_history, original_queue = fixture()
    _, prior_queue = forward(old_task, old_history, original_queue)
    d, a, seed, h = generation_fixture()
    # Use the actual predecessor of the forward-bound queue in the next claim.
    predecessor = old_task["body"]["completion_receipt"]
    seed = predecessor["post_merge_completion_recovery_seed"]
    h["revisions"][15] = copy.deepcopy(old_history["revisions"][15])
    for index in (16, 17):
        h["revisions"][index]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ] = seed
    _rehash(h)
    new_seed = {
        **seed,
        "recovery_control_revision": 19,
        "qualified_target_commit": "e" * 40,
        "qualification_receipt_id": "receipt:fresh",
        "recovery_evidence_id": "sha256:" + "f" * 64,
    }
    new_seed.pop("seed_id")
    new_seed["seed_id"] = d._database_portal_evidence_digest(new_seed)
    new_receipt = {
        **predecessor,
        "control_expected_revision": 19,
        "post_merge_completion_recovery_seed": new_seed,
        "qualified_target_commit": new_seed["qualified_target_commit"],
        "callback_requalification_receipt_id": new_seed["qualification_receipt_id"],
        "callback_reconciliation_evidence_id": new_seed["recovery_evidence_id"],
        "queue_reason": "database_post_merge_declared_outputs_callback_integration:"
        + seed["request_id"]
        + ":receipt:fresh",
    }
    h["revisions"].append(
        {
            "revision": 20,
            "status": "retrying",
            "body": {**old_task["body"], "completion_receipt": new_receipt},
        }
    )
    _rehash(h)
    return {**old_task, **h["revisions"][-1]}, h, prior_queue


@pytest.mark.parametrize(
    "mutation", [None, "prior_binding", "same_target", "same_receipt", "floor"]
)
def test_refresh_cooldown_preserves_old_receipt_and_latest_floor(mutation):
    task, history, prior = refresh_queue_fixture()
    if mutation == "prior_binding":
        ext = __import__("json").loads(prior["extension_json"])
        ext[FIELD]["source_seed_id"] = "foreign"
        prior["extension_json"] = canonical_json_bytes(ext).decode()
    elif mutation in {"same_target", "same_receipt"}:
        receipt = task["body"]["completion_receipt"]
        seed = receipt["post_merge_completion_recovery_seed"]
        old = history["revisions"][15]["body"]["completion_receipt"][
            "post_merge_completion_recovery_seed"
        ]
        key = "qualified_target_commit" if mutation == "same_target" else "qualification_receipt_id"
        seed[key] = old[key]
        seed.pop("seed_id")
        seed["seed_id"] = (
            "sha256:" + __import__("hashlib").sha256(canonical_json_bytes(seed)).hexdigest()
        )
        receipt["qualified_target_commit"] = seed["qualified_target_commit"]
        receipt["callback_requalification_receipt_id"] = seed["qualification_receipt_id"]
        receipt["queue_reason"] = (
            "database_post_merge_declared_outputs_callback_integration:"
            + seed["request_id"]
            + ":"
            + seed["qualification_receipt_id"]
        )
        _rehash(history)
    elif mutation == "floor":
        prior["attempt"] = 5
    if mutation:
        with pytest.raises(TypedStateOwnerError):
            build_binding(task=task, history=history, prior_queue=prior)
    else:
        before = copy.deepcopy(prior)
        binding = build_binding(task=task, history=history, prior_queue=prior)
        payload = payload_from_binding(binding)
        assert payload["attempt_number"] == payload["fencing_token"] == payload["fence_epoch"] == 5
        assert binding["source_receipt"]["attempt_number"] == 2
        assert binding["prior_queue"] == before == prior
        assert binding["schema"].endswith("@2")


@pytest.mark.parametrize("changed_target", [False, True])
def test_callback_seed_distinguishes_target_generation_from_other_evidence(changed_target):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalBridgeError,
    )

    d, bridge, attempt, record, seed, predecessor, h = _seeded_dispatch_fixture()
    request = SimpleNamespace(
        task_id=attempt.task_alias, commit_sha=seed["candidate_commit"], metadata={}
    )
    binding = {
        key.removeprefix("queue_source_"): value
        for key, value in seed.items()
        if key.startswith("queue_source_")
    }
    binding["task_cid"] = attempt.task_cid
    bridge.merge_queue = SimpleNamespace(get=lambda key: request)
    bridge._owned_post_merge_recovery_projection = lambda *args, **kwargs: SimpleNamespace(
        binding=binding
    )
    bridge._post_merge_recovery_evidence = lambda *args, **kwargs: {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-post-merge-callback-integration-recovery@1",
        "request_id": seed["request_id"],
        "candidate_commit": seed["candidate_commit"],
        "qualified_target_commit": "f" * 40 if changed_target else seed["qualified_target_commit"],
        "callback_requalification_receipt": {},
        "evidence_id": "changed",
        "callback_requalification_receipt_id": "changed",
    }
    message = "target generation changed" if changed_target else "seed evidence changed"
    with pytest.raises(DatabasePortalBridgeError, match=message):
        bridge._post_merge_completion_recovery_seed_from_record(attempt=attempt, record=record)


@pytest.mark.parametrize("field", ["extension_json", "retained_callback_binding"])
def test_cooldown_large_records_keep_command_and_unrelated_text_bounds(field):
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        MAX_COMMAND_BYTES,
        MAX_TEXT_BYTES,
        ControlPlaneBoundsError,
    )
    from test.api.test_agent_supervisor_control_plane_contracts import _command

    value = "x" * (MAX_TEXT_BYTES + 1)
    command = _command(parameters={"operation": "task.retry.cooldown.record", field: value})
    assert command.parameters[field] == value
    with pytest.raises(ControlPlaneBoundsError):
        _command(parameters={"operation": "unrelated", field: value})
    with pytest.raises(ControlPlaneBoundsError):
        _command(parameters={"operation": "task.retry.cooldown.record", "unrelated": value})
    with pytest.raises(ControlPlaneBoundsError):
        _command(
            parameters={
                "operation": "task.retry.cooldown.record",
                field: "x" * (MAX_COMMAND_BYTES + 1),
            }
        )
