"""Opt-in packet admission: real native DBs and registered Ed25519 profiles.

These fixtures exercise the authority protocol with synthetic metadata results;
they do not claim actual programme execution, proof, or legal acceptance.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority as profiles
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_database_coordinator
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import LeaseCoordinator
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import _repository
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import EAAEFBorrowedTransactionAdapter
from ipfs_accelerate_py.agent_supervisor.task_sources.execution_packet_admission import (
    EVIDENCE_KIND, POLICY_SCHEMA, ExecutionPacketAdmissionError,
    NativeExecutionPacketAuthority, contract_payload_on, observed_validation_payload,
    require_execution_packet_on,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentCompletionError, IntentRepositoryConflictError, open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError, DatabaseImplementationDaemon, PortalImplementationDaemon,
)


def _descriptor(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    workspace = tmp_path / "repository"
    workspace.mkdir()
    for name, text in (("producer.py", "pass\n"), ("verifier.py", "pass\n"), ("input.json", "{}\n")):
        (workspace / name).write_text(text)
    for argv in (("init", "-q"), ("add", "."), ("-c", "user.name=Packet test", "-c", "user.email=packet@example.invalid", "commit", "-qm", "fixture")):
        subprocess.run(["git", "-C", str(workspace), *argv], check=True, capture_output=True)
    # Incidental fixture-authoring mirrors carry no admission authority.
    # Keep them local; admission reads must never invoke this writer.
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index
    mirrors = []
    monkeypatch.setattr(supervisor_meta_index, "mirror_work_record",
        lambda **record: mirrors.append(record) or {"status": "skip", "reason": "private_protocol_fixture"})
    monkeypatch.setattr(profiles, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "registered-profile-roots")
    profiles.initialize_local_profile_lifecycle_root()
    _, head, repository_cid = _repository(workspace)
    references = {}
    for role in ("owner", "verifier"):
        directory, lifecycle = tmp_path / role, tmp_path / (role + "-lifecycle")
        profile = profiles.initialize_local_profile(repository_cid=repository_cid, baseline_commit=head,
            capabilities=["read", "test"], profile_dir=directory, lifecycle_dir=lifecycle)
        references[role] = {"profile_dir": str(directory), "lifecycle_dir": str(lifecycle),
            "repository_cid": repository_cid, "profile_id": profile.profile_id,
            "identity_did": profile.identity_did, "content_id": profile.content_id}
    body = {"execution_packet_required": True, "completion": "explicit_validation",
        "code_edit_allowlist": [], "exclusive_keys": ["packet:metadata"],
        "execution_packet_policy": {"schema": POLICY_SCHEMA, "purpose": "control_metadata_analysis",
            **references, "max_age_seconds": 3600}}
    packet = {"purpose": "control_metadata_analysis", "cwd": str(workspace),
        "argv": [sys.executable, "-B", str(workspace / "producer.py")],
        "validation_argv": [sys.executable, "-B", str(workspace / "verifier.py")],
        "source_descriptors": [_descriptor(workspace / p) for p in ("producer.py", "verifier.py")],
        "inputs": [{"role": "source_metadata", "use": "verification", "descriptor": _descriptor(workspace / "input.json")}],
        "outputs": ["report.json"], "code_allowlist": [], "test_allowlist": [str(workspace / "verifier.py")],
        "exclusive_keys": ["packet:metadata"], "retry_key": "metadata:attempt1", "invalidation_key": "metadata:source-v1",
        "budgets": {"cpu_threads": 1, "memory_mib": 128, "storage_bytes": 1024 * 1024,
            "max_children": 2, "max_tokens": 0, "deadline_seconds": 30, "max_attempts": 1}}
    intent = open_intent_repository(tmp_path / "intent.duckdb", owner_id="owner:packet-test")
    intent.upsert_objective(objective_id="objective:packet", objective_alias="PACKET-O", title="Metadata gate")
    intent.upsert_goal(goal_cid="goal:packet", goal_alias="PACKET-G", title="Metadata gate", objective_id="objective:packet")
    intent.upsert_plan(plan_cid="plan:packet", plan_alias="PACKET-P", goal_cid="goal:packet", status="active")
    value = SimpleNamespace(intent=intent, workspace=workspace, references=references, body=body, packet=packet,
        cid="task:packet", tmp_path=tmp_path, evidence_id="", mirrors=mirrors)
    _upsert(value)
    try:
        yield value
    finally:
        intent.close()


def _upsert(s, **changes):
    arguments = {"task_cid": s.cid, "task_alias": "PACKET-T", "goal_cid": "goal:packet",
        "objective_id": "objective:packet", "plan_cid": "plan:packet", "status": "blocked",
        "body": deepcopy(s.body), "outputs": [{"path": "report.json"}],
        "acceptance": ["Current independently checked metadata evidence"], "validations": []}
    arguments.update(changes)
    return s.intent.upsert_task(**arguments)


def _sign(s, payload, role):
    ref = s.references[role]
    return {"payload": deepcopy(payload), "binding": profiles.sign_profile_binding(
        profile_dir=Path(ref["profile_dir"]), lifecycle_dir=Path(ref["lifecycle_dir"]), payload=payload)}


def _accept(s, *, owner_role="owner", validation_role="verifier", record_validation=True, outcome="passed", parent=""):
    with s.intent._connection(write=False) as connection:
        payload = contract_payload_on(connection, s.cid, deepcopy(s.packet))
    envelope = _sign(s, payload, owner_role)
    receipt = s.intent.record_evidence(task_cid=s.cid, evidence_kind=EVIDENCE_KIND, digest=content_identity(envelope),
        body={"execution_packet_contract": envelope}, parent_evidence_id=parent)
    s.evidence_id = receipt.subject_id
    if record_validation:
        observed = _sign(s, observed_validation_payload(envelope, outcome=outcome), validation_role)
        s.intent.record_validation_result(task_cid=s.cid, outcome=outcome, evidence_digest=content_identity(observed),
            argv=s.packet["validation_argv"], body={"execution_packet_validation": observed})
    return envelope


def _check(s, phase="dispatch"):
    with s.intent._connection(write=False) as connection:
        return require_execution_packet_on(connection, s.cid, phase=phase)


def _snapshot(s):
    with s.intent._connection(write=False) as connection:
        tables = ("tasks", "task_revisions", "domain_events", "task_claims", "task_attempts", "leases", "provider_invocations", "effect_claims")
        return {table: connection.execute("SELECT * FROM " + table + " ORDER BY 1").fetchall() for table in tables}


def _cas(s, status):
    current = s.intent.get_task(s.cid)
    return s.intent.cas_task_status(task_cid=s.cid, expected_revision=current["revision"], new_status=status,
        receipt={"operation": "packet_test_lifecycle"})


@pytest.mark.parametrize("route", ["upsert", "status", "retry", "claimed", "running"])
def test_missing_packet_rejects_raw_dispatch_without_effects(scenario, route):
    s = scenario
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError, match="packet_current_native_contract_required"):
        if route == "upsert":
            _upsert(s, status="ready")
        else:
            _cas(s, "retrying" if route == "retry" else "ready" if route == "status" else route)
    assert _snapshot(s) == before


def test_unsigned_planner_fields_are_not_native_admission(scenario):
    s = scenario
    changed = deepcopy(s.body)
    changed.update({"packet_accepted": True, "admitted": True, "validation_passed": True})
    _upsert(s, body=changed)
    with pytest.raises(ExecutionPacketAdmissionError, match="current_native_contract"):
        _check(s)


@pytest.mark.parametrize("flag", [False, None, "true", 1, "missing"])
def test_requirement_is_sticky_and_bool_only(scenario, flag):
    s = scenario
    body = deepcopy(s.body)
    if flag == "missing":
        body.pop("execution_packet_required")
    else:
        body["execution_packet_required"] = flag
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError, match="packet_requirement"):
        _upsert(s, body=body)
    assert _snapshot(s) == before


@pytest.mark.parametrize("owner_role,validation_role,record_validation,outcome", [
    ("verifier", "verifier", True, "passed"),
    ("owner", "owner", True, "passed"),
    ("owner", "verifier", False, "passed"),
    ("owner", "verifier", True, "failed"),
])
def test_real_independent_profile_and_native_pass_are_required(scenario, owner_role, validation_role, record_validation, outcome):
    s = scenario
    _accept(s, owner_role=owner_role, validation_role=validation_role, record_validation=record_validation, outcome=outcome)
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError):
        _cas(s, "ready")
    assert _snapshot(s) == before


def test_signature_forgery_rejected(scenario):
    s = scenario
    envelope = _accept(s)
    envelope["binding"]["signature"] = "00" * 64
    s.intent.record_evidence(task_cid=s.cid, evidence_kind=EVIDENCE_KIND, digest=content_identity(envelope),
        evidence_id=s.evidence_id, body={"execution_packet_contract": envelope})
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError):
        _cas(s, "ready")
    assert _snapshot(s) == before


def test_lifecycle_revisions_do_not_self_invalidate_contract(scenario):
    s = scenario
    _accept(s)
    accepted = _check(s)
    initial_revision = s.intent.get_task(s.cid)["revision"]
    _cas(s, "ready")
    _cas(s, "in_progress")
    assert s.intent.get_task(s.cid)["revision"] == initial_revision + 2
    assert _check(s) == accepted
    _upsert(s, status="in_progress", body=dict(s.intent.get_task(s.cid)["body"]))
    s.intent.set_task_dependencies(s.cid, [])
    assert _check(s) == accepted


@pytest.mark.parametrize("field", ["body", "plan", "outputs", "acceptance", "validations"])
def test_semantic_amendment_invalidates_accepted_packet(scenario, field):
    s = scenario
    _accept(s)
    changes = {}
    if field == "body":
        body = deepcopy(s.body)
        body["input_version"] = 2
        changes["body"] = body
    elif field == "plan":
        s.intent.upsert_plan(plan_cid="plan:other", plan_alias="OTHER-P", goal_cid="goal:packet", status="active")
        changes["plan_cid"] = "plan:other"
    elif field == "outputs":
        changes["outputs"] = [{"path": "other.json"}]
    elif field == "acceptance":
        changes["acceptance"] = ["Different acceptance"]
    else:
        changes["validations"] = [[sys.executable, "new_check.py"]]
    _upsert(s, **changes)
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError):
        _cas(s, "ready")
    assert _snapshot(s) == before


def test_reverting_semantic_body_does_not_replay_old_acceptance(scenario):
    s = scenario
    _accept(s)
    body = deepcopy(s.body)
    body["input_version"] = 2
    _upsert(s, body=body)
    _upsert(s)
    with pytest.raises(ExecutionPacketAdmissionError, match="native_spec_changed"):
        _check(s)


@pytest.mark.parametrize("name", ["producer.py", "input.json"])
def test_current_source_and_input_bytes_checked_before_ready(scenario, name):
    s = scenario
    _accept(s)
    (s.workspace / name).write_text("changed\n")
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError, match="packet_input"):
        _cas(s, "ready")
    assert _snapshot(s) == before


def test_dependency_completion_revision_requires_fresh_rebind(scenario):
    s = scenario
    s.intent.upsert_task(task_cid="task:dependency", task_alias="DEPENDENCY", goal_cid="goal:packet", status="blocked")
    s.intent.record_evidence(task_cid="task:dependency", evidence_kind="metadata", digest="digest:dependency")
    s.intent.cas_task_status(task_cid="task:dependency", expected_revision=1, new_status="completed")
    _upsert(s, dependencies=["task:dependency"])
    _accept(s)
    assert _check(s)
    first = s.evidence_id
    s.intent.cas_task_status(task_cid="task:dependency", expected_revision=2, new_status="blocked")
    s.intent.cas_task_status(task_cid="task:dependency", expected_revision=3, new_status="completed")
    with pytest.raises(ExecutionPacketAdmissionError, match="dependency_changed"):
        _check(s)
    _accept(s, parent=first)
    _cas(s, "ready")
    assert _check(s)


def test_ready_dependency_amendment_fails_without_mutating_edges(scenario):
    s = scenario
    _accept(s)
    _cas(s, "ready")
    s.intent.upsert_task(task_cid="task:dependency", task_alias="DEPENDENCY", goal_cid="goal:packet", status="blocked")
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError):
        s.intent.set_task_dependencies(s.cid, ["task:dependency"])
    assert s.intent.get_task(s.cid)["dependencies"] == ()
    assert _snapshot(s) == before


def test_raw_embedded_claim_requires_live_native_owner(scenario):
    s = scenario
    _accept(s)
    _cas(s, "ready")
    with open_database_coordinator(s.tmp_path / "coordination.duckdb") as coordinator:
        coordinator.register_task(task_cid=s.cid, body={"execution_packet_required": True})
        for fake in (None, True, lambda: True):
            with pytest.raises(ExecutionPacketAdmissionError, match="live_native_authority"):
                coordinator.claim_task(task_cid=s.cid, owner_session_id="owner:claim", execution_packet_authority=fake)
        connection = coordinator._require()
        assert connection.execute("SELECT COUNT(*) FROM task_claims").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM task_attempts").fetchone()[0] == 0
        assert coordinator.claim_task(task_cid=s.cid, owner_session_id="owner:claim",
            execution_packet_authority=NativeExecutionPacketAuthority(s.intent)).task_cid == s.cid


def test_stale_embedded_claim_creates_no_attempt(scenario):
    s = scenario
    _accept(s)
    _cas(s, "ready")
    (s.workspace / "input.json").write_text("changed\n")
    with open_database_coordinator(s.tmp_path / "coordination.duckdb") as coordinator:
        coordinator.register_task(task_cid=s.cid, body={"execution_packet_required": True})
        authority = NativeExecutionPacketAuthority(s.intent)
        assert coordinator.claim_ready_task(owner_session_id="owner:claim", execution_packet_authority=authority) is None
        connection = coordinator._require()
        for table in ("task_claims", "task_attempts", "fenced_leases"):
            assert connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] == 0


@pytest.mark.parametrize("fault", ["missing", "stale", "forged"])
def test_eaaef_owned_claim_rejects_invalid_packet_before_claim_effects(scenario, fault):
    s = scenario
    envelope = _accept(s)
    _cas(s, "ready")
    if fault == "stale":
        (s.workspace / "input.json").write_text("changed\n")
    elif fault == "forged":
        envelope["binding"]["signature"] = "00" * 64
        s.intent.record_evidence(task_cid=s.cid, evidence_kind=EVIDENCE_KIND, digest=content_identity(envelope),
            evidence_id=s.evidence_id, body={"execution_packet_contract": envelope})
    else:
        # Model a pre-existing/migrated ready projection whose retained contract is missing.
        with s.intent._connection(write=True) as connection:
            connection.execute("DELETE FROM evidence_nodes WHERE evidence_id=?", [s.evidence_id])
    before = _snapshot(s)
    adapter = object.__new__(EAAEFBorrowedTransactionAdapter)
    with s.intent._connection(write=True) as connection:
        assert adapter._claim_ready(connection, {"owner_session_id": "owner:eaaef", "lease_ms": 60000,
            "exclude_task_cids": [], "now_ms": int(time.time() * 1000)}, lease={}, idempotency_key="claim:test", fence_epoch=1) is None
    assert _snapshot(s) == before


def test_metadata_provider_denied_for_daemon_and_eaaef_even_with_valid_packet(scenario):
    s = scenario
    _accept(s)
    _cas(s, "ready")
    before = _snapshot(s)
    with pytest.raises(ExecutionPacketAdmissionError, match="no_provider_authority"):
        _check(s, "provider")
    current = SimpleNamespace(body=s.intent.get_task(s.cid)["body"])
    daemon = SimpleNamespace(task_source=SimpleNamespace(get=lambda cid: current), _typed_quack_authority_binding=None)
    with pytest.raises(DatabaseImplementationAuthorityError, match="no provider authority"):
        DatabaseImplementationDaemon._require_provider_admission(daemon, SimpleNamespace(task_cid=s.cid))
    args = {name: "unused" for name in ("kind", "record_id", "attempt_id", "operation_key", "idempotency_key", "owner_session_id")}
    args.update(task_cid=s.cid, recorded_at_ms=1, fencing_token=1, fence_epoch=1, claim={})
    with s.intent._connection(write=True) as connection:
        with pytest.raises(ExecutionPacketAdmissionError, match="no_provider_authority"):
            EAAEFBorrowedTransactionAdapter._container_dispatch_idempotent(SimpleNamespace(), connection, "effect.reserve", args)
        connection.execute("INSERT INTO task_attempts VALUES ('attempt:packet', ?, 1, 'owner:test', 1, 1, '2026-10-28T00:00:00+00:00', NULL, 'running', 1)", [s.cid])
        with pytest.raises(ExecutionPacketAdmissionError, match="no_provider_authority"):
            EAAEFBorrowedTransactionAdapter._idempotent(SimpleNamespace(), connection, "provider.reserve",
                {"kind": "provider", "attempt_id": "attempt:packet", "idempotency_key": "provider:test"})
        assert connection.execute("SELECT status FROM task_attempts WHERE attempt_id='attempt:packet'").fetchone()[0] == "running"
        assert connection.execute("SELECT COUNT(*) FROM provider_invocations").fetchone()[0] == 0
        connection.execute("DELETE FROM task_attempts WHERE attempt_id='attempt:packet'")
    assert _snapshot(s) == before


def test_unflagged_legacy_claim_and_manual_metadata_completion_preserved(scenario):
    s = scenario
    s.intent.upsert_task(task_cid="task:legacy", task_alias="LEGACY", goal_cid="goal:packet", status="ready")
    with open_database_coordinator(s.tmp_path / "coordination.duckdb") as coordinator:
        coordinator.register_task(task_cid="task:legacy")
        assert coordinator.claim_task(task_cid="task:legacy", owner_session_id="owner:legacy").task_cid == "task:legacy"
    with pytest.raises(IntentCompletionError):
        _cas(s, "completed")
    s.intent.record_evidence(task_cid=s.cid, evidence_kind="metadata", digest="digest:observed-metadata")
    _cas(s, "completed")
    assert s.intent.get_task(s.cid)["status"] == "completed"


def test_revoked_registered_profile_cannot_dispatch(scenario):
    s = scenario
    _accept(s)
    reference = s.references["verifier"]
    profiles.revoke_local_profile(profile_dir=Path(reference["profile_dir"]), lifecycle_dir=Path(reference["lifecycle_dir"]))
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError):
        _cas(s, "ready")
    assert _snapshot(s) == before


@pytest.mark.parametrize("fault", ["model_tokens", "purpose", "unbound_test", "edit_scope"])
def test_metadata_packet_purpose_and_effect_limits_are_closed(scenario, fault):
    s = scenario
    if fault == "model_tokens":
        s.packet["budgets"]["max_tokens"] = 1
    elif fault == "purpose":
        s.packet["purpose"] = "training"
    elif fault == "unbound_test":
        s.packet["test_allowlist"] = ["unbound.py"]
    else:
        s.packet["code_allowlist"] = ["producer.py"]
    before = _snapshot(s)
    with pytest.raises(ExecutionPacketAdmissionError):
        _accept(s)
    assert _snapshot(s) == before


@pytest.mark.parametrize("status", ["ready", "claimed", "running"])
def test_eaaef_direct_status_gate_precedes_claim_and_status_effects(scenario, status):
    s = scenario
    before = _snapshot(s)
    adapter = object.__new__(EAAEFBorrowedTransactionAdapter)
    with s.intent._connection(write=True) as connection:
        with pytest.raises(ExecutionPacketAdmissionError, match="current_native_contract"):
            adapter._task_cas(connection, {"task_cid": s.cid, "expected_revision": 1,
                "status": status, "receipt": {}, "evidence_digests": []}, authorized_lease={})
    assert _snapshot(s) == before


def test_eaaef_registration_requires_exact_canonical_packet_marker(scenario):
    s = scenario
    adapter = object.__new__(EAAEFBorrowedTransactionAdapter)
    arguments = {"task_cid": s.cid, "task_id": "PACKET-T", "dependency_task_cids": [],
        "body": {"task_alias": "PACKET-T", "status": "blocked", "execution_packet_required": True}}
    with s.intent._connection(write=False) as connection:
        assert adapter._register_task(connection, arguments)["canonical_task_store_reused"] is True
        arguments["body"].pop("execution_packet_required")
        with pytest.raises(RuntimeError, match="canonical task projection"):
            adapter._register_task(connection, arguments)


def test_embedded_projection_flag_cannot_be_downgraded(scenario):
    s = scenario
    with open_database_coordinator(s.tmp_path / "coordination.duckdb") as coordinator:
        coordinator.register_task(task_cid=s.cid)
        coordinator.register_task(task_cid=s.cid, body={"execution_packet_required": True})
        coordinator.register_task(task_cid=s.cid, body={"execution_packet_required": False})
        with pytest.raises(ExecutionPacketAdmissionError, match="live_native_authority"):
            coordinator.claim_task(task_cid=s.cid, owner_session_id="owner:test")
        assert coordinator._require().execute("SELECT COUNT(*) FROM task_attempts").fetchone()[0] == 0


def test_legacy_source_converter_preserves_nested_boolean_and_provider_refuses():
    source = SimpleNamespace(body={"authored_task": {"execution_packet_required": True}},
        status="ready", task_cid="task:native", goal_id="goal", goal_cid="goal:native",
        dependency_task_ids=(), board_namespace="native", task_id="NATIVE-T", title="Metadata",
        source_line=1)
    task = PortalImplementationDaemon._portal_task_from_source_task(source)
    assert task.metadata["execution packet required"] == "true"
    gate = PortalImplementationDaemon._evaluate_pre_implementation_provider_gate(SimpleNamespace(),
        task=task, attempt=1, worktree_path=Path("/unused"))
    assert gate["provider_authorized"] is False
    assert gate["skip_provider"] is True


def test_legacy_bundle_claim_fails_closed_and_marker_is_sticky(tmp_path):
    bundle = {"bundle_key": "metadata/packet", "source_todo": "native.todo.md",
        "tasks": [{"task_id": "PACKET-T", "metadata": {"execution packet required": "true"}}]}
    with LeaseCoordinator(tmp_path / "legacy.sqlite3") as coordinator:
        registered = coordinator.register_bundle(bundle)
        cid = registered["task_cid"]
        with pytest.raises(ExecutionPacketAdmissionError, match="no_native_owner_bridge"):
            coordinator.claim(cid, "did:web:packet.example")
        assert coordinator.claim_ready("did:web:packet.example") is None
        changed = deepcopy(bundle)
        changed["profile_g"] = deepcopy(registered)
        changed["tasks"][0]["metadata"].pop("execution packet required")
        with pytest.raises(ExecutionPacketAdmissionError, match="requirement_cannot_be_removed"):
            coordinator.register_bundle(changed)
        assert coordinator._connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0


def test_flagged_denial_does_not_charge_attempt_or_stop_eligible_peer(scenario):
    s = scenario
    with open_database_coordinator(s.tmp_path / "coordination.duckdb") as coordinator:
        coordinator.register_task(task_cid=s.cid, body={"execution_packet_required": True}, now_ms=1)
        coordinator.register_task(task_cid="task:legacy-peer", now_ms=2)
        claim = coordinator.claim_ready_task(owner_session_id="owner:mixed")
        assert claim.task_cid == "task:legacy-peer"
        connection = coordinator._require()
        assert connection.execute("SELECT COUNT(*) FROM task_attempts WHERE task_cid=?", [s.cid]).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM task_claims WHERE task_cid=?", [s.cid]).fetchone()[0] == 0


def test_gate_profile_and_signature_reads_do_not_mirror_metadata(scenario):
    s = scenario
    _accept(s)
    before = list(s.mirrors)
    assert _check(s)
    assert s.mirrors == before


def test_nested_authored_requirement_cannot_be_removed_under_true_top_flag(scenario):
    s = scenario
    body = deepcopy(s.body)
    body["authored_task"] = {"execution_packet_required": True}
    _upsert(s, body=body)
    body["authored_task"] = {}
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError, match="requirement_cannot_be_removed"):
        _upsert(s, body=body)
    assert _snapshot(s) == before
