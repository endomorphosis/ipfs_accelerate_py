"""Replay the preserved native suffix and reject unsafe claim/dispatch mutations."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_callback_suffix import (
    EXECUTION,
    IDENTITY,
    ROUTE,
    SOURCE_REASON,
    verified_suffix,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _callback_integration_recovery_evidence,
)


def _fixture():
    f = json.loads(
        (Path(__file__).parent / "fixtures/retained_callback_suffix.json").read_text()
    )
    history = {
        "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
        "task_cid": f["task_cid"],
        "revisions": f["revisions"],
    }
    _rehash(history)
    return f, history


def _rehash(history):
    history.pop("projection_cid", None)
    history["projection_cid"] = content_identity(history)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "gap",
        "hash",
        "foreign_task",
        "semantic",
        "foreign_source",
        "foreign_claim",
        "foreign_admission",
        "foreign_route",
        "active",
        "extra_field",
        "wrong_guard",
        "unbound_deferral",
        "wrong_fence",
        "wrong_source_reason",
    ],
)
def test_closed_native_suffix(mutation):
    f, h = _fixture()
    c = h["revisions"]
    if mutation == "gap":
        c.pop(3)
    elif mutation == "foreign_task":
        h["task_cid"] = "foreign"
    elif mutation == "semantic":
        c[9]["body"]["title"] = "changed"
    elif mutation == "foreign_source":
        c[7]["body"]["completion_receipt"]["claim_id"] = "foreign"
    elif mutation == "foreign_claim":
        c[9]["body"]["completion_receipt"]["claim_id"] = "foreign"
    elif mutation == "foreign_admission":
        c[10]["body"]["completion_receipt"]["claim_id"] = "foreign"
    elif mutation == "foreign_route":
        c[12]["body"]["completion_receipt"]["execution_route_binding"]["task_cid"] = (
            "foreign"
        )
    elif mutation == "active":
        c[-1]["status"] = "in_progress"
    elif mutation == "extra_field":
        c[13]["body"]["completion_receipt"]["authority"] = True
    elif mutation == "wrong_guard":
        c[-1]["body"]["completion_receipt"]["reason"] = "arbitrary"
    elif mutation == "unbound_deferral":
        c[11]["body"]["completion_receipt"]["evidence_source"] = "operator"
    elif mutation == "wrong_fence":
        c[-1]["body"]["completion_receipt"]["fence_epoch"] += 1
    elif mutation == "wrong_source_reason":
        c[7]["body"]["completion_receipt"]["reason"] = "arbitrary"
    _rehash(h)
    if mutation == "hash":
        h["projection_cid"] = "foreign"
    result = verified_suffix(
        h, task_cid=f["task_cid"], task_alias=f["task_alias"], control_revision=15
    )
    assert (result is not None) == (mutation is None)


def _physical_fixture():
    f, h = _fixture()
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.open = lambda: daemon
    attempts = {a["attempt_id"]: daemon._attempt_from_mapping(a) for a in f["attempts"]}
    phases = {
        key: [
            {**p, "body": json.loads(p["body_json"])}
            for p in f["phases"]
            if p["attempt_id"] == key
        ]
        for key in attempts
    }
    daemon.get_attempt = attempts.get
    daemon.phase_history = phases.__getitem__
    daemon._local_attempt_is_exact_latest = lambda a: True
    daemon._failed_attempt_coordination_successor = lambda a: None
    daemon._terminal_coordination_reproduces_read_only = lambda a, **kw: True
    daemon.list_running_attempts = list
    daemon._coordinator = SimpleNamespace(get_prepared_task_completion=lambda cid: None)
    task = SimpleNamespace(
        task_cid=f["task_cid"],
        task_alias=f["task_alias"],
        revision=15,
        status="blocked",
        body=h["revisions"][-1]["body"],
    )
    daemon._task_source = SimpleNamespace(
        task_revision_history_projection=lambda cid: h
    )
    return daemon, task, attempts, phases, h


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "provider_phase",
        "provider_true",
        "missing_source",
        "changed_terminal",
        "active_attempt",
        "prepared",
        "newer_cursor",
        "live_fence",
    ],
)
def test_physical_no_provider_suffix(mutation):
    d, task, attempts, phases, _h = _physical_fixture()
    source, middle, current = list(attempts.values())
    if mutation == "provider_phase":
        phases[current.attempt_id][1]["phase"] = "provider"
    elif mutation == "provider_true":
        phases[middle.attempt_id][-1]["body"]["provider_dispatched"] = True
    elif mutation == "missing_source":
        attempts.pop(source.attempt_id)
    elif mutation == "changed_terminal":
        phases[current.attempt_id][-1]["body"]["reason"] = "foreign"
    elif mutation == "active_attempt":
        d.list_running_attempts = lambda: [current]
    elif mutation == "prepared":
        d.coordinator.get_prepared_task_completion = lambda cid: {"prepared": True}
    elif mutation == "newer_cursor":
        d._local_attempt_is_exact_latest = lambda a: False
    elif mutation == "live_fence":
        d._terminal_coordination_reproduces_read_only = lambda a, **kw: False
    try:
        result = d._retained_callback_suffix_context(task)
    except DatabaseImplementationAuthorityError:
        result = None
    assert (result is not None) == (mutation is None)
    if result:
        assert result["source_attempt"] == source
        assert result["current_attempt"] == current


def _seeded_dispatch_fixture():
    d, task, attempts, _phases, h = _physical_fixture()
    source, _middle, _current = list(attempts.values())
    evidence = _callback_integration_recovery_evidence(d, source)
    seed = d._build_post_merge_completion_recovery_seed(
        attempt=source,
        task_revision=8,
        recovery_control_revision=15,
        evidence=evidence,
        qualified_target_commit=evidence["qualified_target_commit"],
        qualification_kind="callback_integration",
        qualification_receipt_id=evidence["callback_requalification_receipt_id"],
        recovery_evidence_id=evidence["evidence_id"],
        terminal_reason=SOURCE_REASON,
    )
    old = h["revisions"][7]["body"]["completion_receipt"]
    predecessor = {k: old[k] for k in IDENTITY | EXECUTION | ROUTE}
    predecessor.update(
        {
            "operation": "database_post_merge_declared_outputs_callback_integration_recovery",
            "control_expected_revision": 15,
            "control_expected_status": "blocked",
            "post_merge_completion_recovery_seed": seed,
            "queue_receipt": {},
            "queue_reason": "database_post_merge_declared_outputs_callback_integration:"
            + seed["request_id"]
            + ":"
            + seed["qualification_receipt_id"],
            "coordination": {
                k: old[k] for k in ("attempt_id", "claim_id", "attempt_number")
            },
            "request_id": seed["request_id"],
            "candidate_commit": seed["candidate_commit"],
            "qualified_target_commit": seed["qualified_target_commit"],
            "source_binding_id": seed["queue_source_binding_id"],
            "source_projection_immutable_digest": seed[
                "queue_source_projection_immutable_digest"
            ],
            "callback_requalification_receipt_id": seed["qualification_receipt_id"],
            "callback_reconciliation_evidence_id": seed["recovery_evidence_id"],
            "source_integration_commit": "b" * 40,
            "source_train_receipt_id": "sha256:" + "4" * 64,
        }
    )
    claim = copy.deepcopy(h["revisions"][12]["body"]["completion_receipt"])
    claim.update(
        {
            "attempt_id": "attempt:seed-consumer",
            "claim_id": "claim:seed-consumer",
            "lease_id": "lease:seed-consumer",
            "attempt_number": 5,
            "fencing_token": 5,
            "fence_epoch": 5,
            "claimed_from_revision": 16,
            "post_merge_completion_recovery_seed": seed,
            "post_merge_completion_recovery_source_attempt_id": source.attempt_id,
        }
    )
    admitted = {
        **claim,
        "operation": "database_attempt_admitted",
        "claim_phase_schema": "ipfs_accelerate_py/agent-supervisor/typed-database-attempt-admission@1",
        "admitted_from_revision": 17,
        "attempt_execution_phase": "claimed",
        "attempt_execution_revision": 1,
    }
    semantic = {k: v for k, v in task.body.items() if k != "completion_receipt"}
    for rev, status, r in (
        (16, "retrying", predecessor),
        (17, "in_progress", claim),
        (18, "in_progress", admitted),
    ):
        h["revisions"].append(
            {
                "revision": rev,
                "status": status,
                "body": {**semantic, "completion_receipt": r},
            }
        )
    _rehash(h)
    record = SimpleNamespace(
        task_cid=task.task_cid,
        task_alias=task.task_alias,
        status="in_progress",
        revision=18,
        body=h["revisions"][-1]["body"],
    )
    attempt = SimpleNamespace(
        task_cid=task.task_cid,
        task_alias=task.task_alias,
        **{k: claim[k] for k in IDENTITY},
    )
    bridge = object.__new__(DatabasePortalExecutionBridge)
    bridge.task_source = d.task_source
    bridge.implementation_timeout = 1
    return d, bridge, attempt, record, seed, predecessor, h


@pytest.mark.parametrize(
    "mutation",
    [None, "foreign_source", "foreign_receipt", "gap", "foreign_admission", "no_seed"],
)
def test_native_seed_admission_and_actual_dispatch_boundary(mutation):
    _d, b, a, r, seed, predecessor, h = _seeded_dispatch_fixture()
    if mutation == "foreign_source":
        seed["attempt_id"] = "foreign"
    elif mutation == "foreign_receipt":
        predecessor["candidate_commit"] = "d" * 40
    elif mutation == "gap":
        h["revisions"].pop(5)
    elif mutation == "foreign_admission":
        r.body["completion_receipt"]["lease_id"] = "foreign"
    elif mutation == "no_seed":
        r.body["completion_receipt"].pop("post_merge_completion_recovery_seed")
    _rehash(h)
    claim = b._post_merge_completion_claim_receipt(
        attempt=a,
        record=r,
        status_receipt=r.body["completion_receipt"],
        seed=seed,
        recovery_control_revision=15,
    )
    assert (claim is not None) == (mutation is None)
    calls = []
    b._record_for_attempt = lambda *args: r
    b._execution_route_binding = lambda **kwargs: calls.append("route") or {}
    b._protected_preservation_seed_from_record = lambda **kwargs: None
    b._post_commit_candidate_seed_from_record = lambda **kwargs: None
    b._ensure_attempt_projection = lambda *args: (None, None)

    def verify(**kwargs):
        # Keep the actual native claim/admission verifier at dispatch. Expensive
        # immutable Git/Portal qualification has independent real-receipt tests.
        if (
            b._post_merge_completion_claim_receipt(
                attempt=a,
                record=r,
                status_receipt=r.body["completion_receipt"],
                seed=seed,
                recovery_control_revision=15,
            )
            is None
        ):
            raise DatabasePortalBridgeError("suffix claim rejected")
        return seed

    b._post_merge_completion_recovery_seed_from_record = verify
    b._accept_post_merge_completion_recovery_seed = lambda **kwargs: {
        "retained": True,
        "provider_dispatched": False,
    }
    b._verify_projection = lambda *args: pytest.fail("ordinary provider path reached")
    if mutation is None:
        assert b.run_provider(a) == {"retained": True, "provider_dispatched": False}
    else:
        with pytest.raises(DatabasePortalBridgeError):
            b.run_provider(a)
    if mutation == "no_seed":
        assert calls == []


def test_retained_suffix_uses_current_real_claim_for_atomic_cas(tmp_path, monkeypatch):
    """The original source cannot replace the latest claim as CAS authority.

    The matcher has independent captured-native tests above. This test injects
    its result while retaining real DuckDB claims, expiry, phase state and CAS.
    """
    from test.api.test_agent_supervisor_database_implementation_daemon import (
        _open_daemon,
        _population,
    )

    now = {"ms": 1000}
    calls = []
    d = _open_daemon(
        tmp_path,
        clock_ms=lambda: now["ms"],
        lease_ms=5000,
        provider_calls=calls,
        effect_calls=calls,
    )
    try:
        d.materialize_population(_population(1))
        source = d.claim_next()
        source = d.commit_phase(source, "context")
        source = d.commit_phase(
            source,
            "failed",
            body={
                "reason": SOURCE_REASON,
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        now["ms"] += 6000
        source_coordination = d._reconcile_failed_attempt_coordination(source)
        d._persist_terminal_portal_failure(
            source, reason=SOURCE_REASON, coordination_evidence=source_coordination
        )
        original = d.task_source.get(source.task_cid)
        d._persist_task_retry_state(
            source,
            reason="portal_completion_handshake_retry",
            backoff_ms=0,
            evidence_source="portal_completion_handshake_reclassified",
            coordination_evidence=source_coordination,
            allow_blocked_recovery=True,
        )
        current = d.claim_next()
        assert current and current.attempt_id != source.attempt_id
        current = d.commit_phase(current, "context")
        current = d.commit_phase(
            current,
            "failed",
            body={
                "reason": "retained callback recovery requires exact source seed before dispatch",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )
        now["ms"] += 6000
        d._persist_terminal_portal_failure(
            current,
            reason="retained callback recovery requires exact source seed before dispatch",
            coordination_evidence=d._reconcile_failed_attempt_coordination(current),
        )
        before = d.task_source.get(current.task_cid)
        context = {
            "receiver_suffix": True,
            "source_attempt": source,
            "current_attempt": current,
            "source_task_revision": original.revision,
            "source_coordination": source_coordination,
            "source_receipt": dict(original.body["completion_receipt"]),
            "current_receipt": dict(before.body["completion_receipt"]),
            "context_id": "test:independently-verified-window",
            "portable_coordination_authority": False,
        }
        d._post_merge_completion_crash_recovery_context = lambda task, **kw: context
        def reproduce_before_fence(task):
            # Production preauthorization reads coordination. Re-entering this
            # from the atomic callback must fail in the real coordinator.
            d.coordinator.get_task_claim(current.claim_id)
            return context

        d._retained_callback_suffix_context = reproduce_before_fence
        d._retained_callback_suffix_physical_context = lambda task: context
        monkeypatch.setattr(
            d,
            "_verified_post_merge_callback_integration_receipt",
            lambda raw, **kwargs: dict(raw),
        )
        evidence = _callback_integration_recovery_evidence(d, source)
        result = d.recover_blocked_post_merge_declared_outputs(evidence)
        assert result["recovered"] and result["changed"]
        after = d.task_source.get(source.task_cid)
        receipt = after.body["completion_receipt"]
        assert after.status == "retrying" and after.revision == before.revision + 1
        assert receipt["attempt_id"] == source.attempt_id
        assert receipt["control_expected_revision"] == before.revision
        assert (
            receipt["post_merge_completion_recovery_seed"]["source_task_revision"]
            == original.revision
        )
        assert (
            receipt["post_merge_completion_recovery_seed"]["recovery_control_revision"]
            == before.revision
        )
        assert d.coordinator.get_task_claim(current.claim_id).state.value == "expired"
        assert not d.list_running_attempts()
        assert calls == []
    finally:
        d.close()
