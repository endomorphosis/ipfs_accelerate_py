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
def scenario(tmp_path, monkeypatch, request):
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
    database = tmp_path / "intent.duckdb"
    operational = request.node.originalname == "test_eaaef_owned_claim_rejects_invalid_packet_before_claim_effects"
    if operational:
        from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import install_eaaef_operational_schema
        install_eaaef_operational_schema(database, application_version="packet-gate-test",
            tool_version="packet-gate-test", owner_id="owner:packet-test-materializer")
    intent = open_intent_repository(database, owner_id="owner:packet-test", install_schema=not operational)
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
        with coordinator._database_operation():
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


def test_plan_reversion_does_not_replay_old_acceptance(scenario):
    s = scenario
    _accept(s)
    s.intent.upsert_plan(plan_cid="plan:other", plan_alias="OTHER-P", goal_cid="goal:packet", status="active")
    _upsert(s, plan_cid="plan:other")
    _upsert(s, plan_cid="plan:packet")
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError, match="native_spec_changed"):
        _cas(s, "ready")
    assert _snapshot(s) == before


def test_retained_validation_pair_for_foreign_task_cannot_admit(scenario):
    s = scenario
    _accept(s)
    s.intent.upsert_task(task_cid="task:foreign", task_alias="FOREIGN-T", goal_cid="goal:packet", status="blocked")
    # The public writer keeps these keys consistent. Model a retained native
    # inconsistency without changing the genuine signed reviewer payload.
    with s.intent._connection(write=True) as connection:
        connection.execute("UPDATE validation_runs SET task_cid='task:foreign' WHERE task_cid=?", [s.cid])
    before = _snapshot(s)
    with pytest.raises(IntentRepositoryConflictError, match="independent_native_validation_required"):
        _cas(s, "ready")
    assert _snapshot(s) == before
    assert s.intent.get_task(s.cid)["status"] == "blocked"


def test_readonly_repository_identity_matches_existing_bridge(scenario):
    s = scenario
    observed = _repository(s.workspace)
    before = list(s.mirrors)
    assert _repository(s.workspace, mirror=False) == observed
    assert s.mirrors == before
    with pytest.raises(ValueError, match="exact repository metadata mirroring control"):
        _repository(s.workspace, mirror=1)


@pytest.mark.parametrize("location", ["body", "authored_task"])
@pytest.mark.parametrize("value", [True, 0, 1, "true", None],
                         ids=["required", "integer-zero", "integer-one", "string", "null"])
def test_metadata_contract_cannot_admit_unbound_implementation_scope(scenario, location, value):
    s = scenario
    body = deepcopy(s.body)
    body["requires_product_change_scope"] = False
    target = body if location == "body" else body.setdefault("authored_task", {})
    target["requires_product_change_scope"] = value
    _upsert(s, body=body)
    before = _snapshot(s)
    with pytest.raises(ExecutionPacketAdmissionError, match="packet_implementation_scope_"):
        _accept(s)
    assert _snapshot(s) == before


@pytest.mark.parametrize("location", ["body", "authored_task"])
def test_explicit_false_implementation_requirement_preserves_metadata_admission(scenario, location):
    s = scenario
    body = deepcopy(s.body)
    target = body if location == "body" else body.setdefault("authored_task", {})
    target["requires_product_change_scope"] = False
    _upsert(s, body=body)
    _accept(s)
    assert _check(s)


def _portal_packet_selector_probe(*, forbid_queue=False):
    """Only external collaborators are probes; the real selector stays intact."""
    effects = []
    queue_calls = []
    retained = {"spent_attempts": {"FLAGGED": 5, "ELIGIBLE": 7},
                "cooldowns": {"retained-other-task": "original-deadline"}}

    def forbidden(label):
        def fail(*_args, **_kwargs):
            effects.append(label)
            pytest.fail("selector attempted unsafe admission/repair effect: " + label)
        return fail

    def queue_call(label, result):
        def call(identity):
            if forbid_queue:
                return forbidden(label)(identity)
            queue_calls.append((label, identity))
            return result
        return call

    queue = SimpleNamespace(
        is_cooled_down=queue_call("cooldown", False),
        get_penalty=queue_call("penalty", 0),
        record_selection=queue_call("selection", None),
        dirty=False,
        save=forbidden("unexpected_queue_save"),
    )
    probe = SimpleNamespace(
        manual_completion_authority_revalidation_only=False,
        _manual_completion_authority_revalidation_only_task=lambda _task: False,
        _task_has_blocking_pending_merge=lambda _task: False,
        _strict_off_mission_deprioritized_task_ids=lambda _strategy: set(),
        degradation_state=SimpleNamespace(degraded_submodules=lambda: []),
        _inflight_submodule_paths=lambda: set(),
        _canonical_ref=lambda task: "cid:" + task.task_id,
        task_queue=queue,
        _todo_vector_selection_context=lambda _tasks, _ready: {},
        _todo_vector_selection_rank=lambda _task, _context: (),
        _task_work_surface_rank=lambda _task: (),
        _task_has_recent_no_change_outcome=lambda _task_id, _outcomes: False,
        _plan_runtime_enabled=lambda: False,
        _record_event=forbidden("unexpected_event"),
        _run_implementation=forbidden("provider_dispatch"),
        _require_provider_task_contract=forbidden("provider_contract"),
    )
    # These four repairs are intentionally absent from the primary caller.
    # Transplanting the whole older qualified selector must fail this seam test.
    for name in (
        "_release_repaired_dependency_preflight_cooldowns",
        "_release_repaired_read_only_validation_cooldowns",
        "_release_obsolete_provider_repair_cooldowns",
        "_release_obsolete_venv_launcher_cooldowns",
    ):
        setattr(probe, name, forbidden(name))
    return probe, effects, queue_calls, retained


def test_portal_packet_selection_flagged_only_has_no_queue_or_repair_effects(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import execution_packet_admission as packet_gate
    observed = []
    actual_predicate = packet_gate.execution_packet_required_metadata

    def observe(metadata):
        observed.append(dict(metadata))
        return actual_predicate(metadata)

    monkeypatch.setattr(packet_gate, "execution_packet_required_metadata", observe)
    task = SimpleNamespace(task_id="FLAGGED", metadata={"execution packet required": "true"},
        title="Flagged metadata work", acceptance="Preserve current authority", priority="P0",
        track="ops", depends_on=())
    probe, effects, queue_calls, retained = _portal_packet_selector_probe(forbid_queue=True)
    before = deepcopy(retained)
    statuses = {task.task_id: "ready"}
    selected = PortalImplementationDaemon._select_next_task(
        probe, [task], statuses, {}, {}, {}, downstream_unlock_by_task={})
    assert selected is None
    assert observed == [task.metadata]
    assert effects == queue_calls == []
    assert retained == before
    assert statuses == {task.task_id: "ready"}


def test_portal_packet_selection_preserves_eligible_peer_without_cooldown_repairs(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import execution_packet_admission as packet_gate
    observed = []
    actual_predicate = packet_gate.execution_packet_required_metadata

    def observe(metadata):
        observed.append(dict(metadata))
        return actual_predicate(metadata)

    monkeypatch.setattr(packet_gate, "execution_packet_required_metadata", observe)
    flagged = SimpleNamespace(task_id="FLAGGED", metadata={"execution_packet_required": True},
        title="Higher-priority flagged work", acceptance="Preserve authority", priority="P0",
        track="ops", depends_on=())
    eligible = SimpleNamespace(task_id="ELIGIBLE", metadata={}, title="Unflagged legacy work",
        acceptance="Existing selection is allowed", priority="P1", track="ops", depends_on=())
    probe, effects, queue_calls, retained = _portal_packet_selector_probe()
    before = deepcopy(retained)
    statuses = {"FLAGGED": "ready", "ELIGIBLE": "ready"}
    selected = PortalImplementationDaemon._select_next_task(
        probe, [flagged, eligible], statuses, {}, {}, {}, downstream_unlock_by_task={})
    assert selected is eligible
    assert observed == [flagged.metadata, eligible.metadata]
    assert effects == []
    assert queue_calls == [("cooldown", "cid:ELIGIBLE"),
                           ("penalty", "cid:ELIGIBLE"),
                           ("selection", "cid:ELIGIBLE")]
    assert retained == before
    assert statuses == {"FLAGGED": "ready", "ELIGIBLE": "ready"}
# Dedicated native execution-only route. These cases use the original fixture
# above: real temporary intent/coordinator databases, registered distinct local
# profiles and genuine native signatures. Positive cases execute the declared
# producer and verifier through the original validation scheduler.
def _metadata_execution_policy(s):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import EXECUTION_POLICY_SCHEMA
    return {"schema": EXECUTION_POLICY_SCHEMA, "task_cid": s.cid,
        "purpose": "control_metadata_analysis", "execution_mode": "deterministic-only",
        "completion_mode": "manual", "max_attempts": 1,
        "provider_authority": False, "completion_authority": False}


def _pure_metadata_task(*, policy_changes=None, body_changes=None):
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import TaskRecord
    # Pure original typed snapshot: no repository, profile, signature, database,
    # scheduler, claim or execution is constructed for these early guard cases.
    policy = _metadata_execution_policy(SimpleNamespace(cid="task:packet"))
    policy.update(policy_changes or {})
    body = {"completion": "manual", "provider_role": "deterministic-only",
            "context_budget_tokens": "0", "execution_packet_required": True,
            "code_edit_allowlist": [], "execution_packet_execution_policy": policy}
    body.update(body_changes or {})
    return TaskRecord(task_cid="task:packet", task_alias="PACKET-T", goal_cid="goal:packet",
                      ordinal=0, status="ready", revision=1, body=body)


def _prepare_metadata_execution(s, *, policy_changes=None, body_changes=None,
                                producer=None, verifier=None, validations=None,
                                accept=True, ready=True):
    s.body.update(completion="manual", provider_role="deterministic-only",
                  context_budget_tokens="0", execution_packet_execution_policy=_metadata_execution_policy(s))
    s.body["execution_packet_execution_policy"].update(policy_changes or {})
    s.body.update(body_changes or {})
    (s.workspace / "producer.py").write_text(producer or (
        "from pathlib import Path\n"
        "Path('report.json').write_text('{\"metadata\":true}\\n')\n"
        "Path('execution-order.txt').write_text('producer\\n')\n"))
    (s.workspace / "verifier.py").write_text(verifier or (
        "from pathlib import Path\nimport json\n"
        "assert json.loads(Path('report.json').read_text()) == {'metadata': True}\n"
        "assert Path('execution-order.txt').read_text() == 'producer\\n'\n"
        "with Path('execution-order.txt').open('a') as f: f.write('verifier\\n')\n"))
    (s.workspace / "precheck.py").write_text("raise SystemExit('precheck must not replace post-output verifier')\n")
    post_verifier = [sys.executable, "-B", str(s.workspace / "verifier.py")]
    s.packet["validation_argv"] = [sys.executable, "-B", str(s.workspace / "precheck.py")]
    s.packet["test_allowlist"] = [str(s.workspace / "precheck.py"), str(s.workspace / "verifier.py")]
    s.packet["source_descriptors"] = [_descriptor(s.workspace / name)
                                      for name in ("producer.py", "verifier.py", "precheck.py")]
    rows = validations if validations is not None else [
        {"argv": s.packet["argv"], "purpose": "control_metadata_analysis", "role": "producer", "provider_allowed": False},
        {"argv": post_verifier, "purpose": "control_metadata_analysis", "role": "independent_verifier", "provider_allowed": False}]
    s.execution_argv = [s.packet["argv"], post_verifier]
    _upsert(s, body=deepcopy(s.body), validations=rows)
    if accept:
        _accept(s)
    if ready:
        _cas(s, "ready")
    return s.intent.get_task(s.cid)["revision"]


def _metadata_daemon(s, **changes):
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    state = s.tmp_path / "metadata-executor-state"
    state.mkdir(mode=0o700, exist_ok=True)
    source = DatabaseTaskSource(s.intent.database_path, install_schema=False)
    options = dict(database_path=s.intent.database_path,
        coordination_path=s.tmp_path / "metadata-coordination.duckdb",
        state_dir=state, state_prefix="single-metadata", owner_session_id="owner:metadata-execution-test",
        authority_mode="embedded_exclusive", task_source_kind="duckdb",
        task_source=source, close_task_source=True, max_task_attempts=1,
        execution_slice_task_cids=(s.cid,), require_real_execution=True)
    options.update(changes)
    return DatabaseImplementationDaemon(**options)


def test_native_metadata_execution_runs_declared_producer_and_distinct_verifier(scenario):
    s = scenario
    revision = _prepare_metadata_execution(s)
    before = s.intent.get_task(s.cid)
    with _metadata_daemon(s) as daemon:
        result = daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert result["passed"] is result["claim_released"] is True
        assert result["provider_dispatched"] is result["completion_authority"] is False
        assert result["automatic_retry_admitted"] is False
        assert [row["argv"] for row in result["commands"]] == s.execution_argv
        assert s.execution_argv[1] != s.packet["validation_argv"]
        assert (s.workspace / "execution-order.txt").read_text() == "producer\nverifier\n"
        assert len(result["outputs"]) == 1 and result["outputs"][0]["path"] == str(s.workspace / "report.json")
        assert Path(result["durable_result"]["path"]).is_file()
        assert Path(result["release_closeout"]["path"]).is_file()
        assert daemon.coordinator.get_lease(result["claim"]["lease_id"]).state.value == "released"
        assert daemon.list_running_attempts() == []
        with pytest.raises(DatabaseImplementationAuthorityError, match="no provider authority"):
            daemon._require_provider_admission(SimpleNamespace(task_cid=s.cid))
    after = s.intent.get_task(s.cid)
    assert after["status"] == "blocked" and after["body"]["completion"] == "manual"
    assert after["acceptance"] == before["acceptance"]
    assert after["outputs"] == before["outputs"]
    assert after["body"]["completion_receipt"]["completion_authority"] is False
    assert after["body"]["completion_receipt"]["operation"] == "native_metadata_execution_awaiting_manual_acceptance"
    snapshot = _snapshot(s)
    assert snapshot["provider_invocations"] == snapshot["effect_claims"] == []
    assert all(row["raw_observation"]["sha256"] for row in result["commands"])
    # Exact native summaries contain no floats; full genuine scheduler timings
    # remain preserved in their descriptor-bound ordinary result artifacts.
    assert content_identity(result["commands"])
    import json
    for row in result["commands"]:
        full = json.loads(Path(row["raw_observation"]["path"]).read_text())
        assert type(full["throughput"]["elapsed_seconds"]) is float
        assert full["results"][0]["validation_process_custody"]["children_reaped"] is True



@pytest.mark.parametrize("change", [
    {"max_attempts": 2}, {"max_attempts": True}, {"task_cid": "task:foreign"},
    {"provider_authority": True}, {"completion_authority": True},
    {"completion_mode": "auto"}, {"unreviewed_permission": True},
])
def test_native_metadata_execution_rejects_forged_execution_policy(change):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import NativeMetadataExecutionError, _policy
    task = _pure_metadata_task(policy_changes=change)
    before = deepcopy(task.to_dict())
    with pytest.raises(NativeMetadataExecutionError, match="metadata_execution_policy"):
        _policy(task)
    assert task.to_dict() == before


def test_native_metadata_execution_default_manual_remains_unclaimable(scenario):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import NativeMetadataExecutionError
    s = scenario
    revision = _prepare_metadata_execution(s, body_changes={"execution_packet_execution_policy": None})
    before = _snapshot(s)
    with _metadata_daemon(s) as daemon:
        assert daemon.claim_next() is None
        with pytest.raises(NativeMetadataExecutionError, match="policy_shape"):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
    assert _snapshot(s) == before


@pytest.mark.parametrize("changes", [
    {"context_budget_tokens": 0}, {"context_budget_tokens": "1"},
    {"provider_role": "grok-implement"}, {"completion": "auto"},
])
def test_native_metadata_execution_rejects_model_or_automatic_modes(changes):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import NativeMetadataExecutionError, _policy
    task = _pure_metadata_task(body_changes=changes)
    before = deepcopy(task.to_dict())
    with pytest.raises(NativeMetadataExecutionError, match="task_mode"):
        _policy(task)
    assert task.to_dict() == before


def test_native_metadata_execution_rejects_different_signed_purpose(scenario):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import NativeMetadataExecutionError
    s = scenario
    revision = _prepare_metadata_execution(s, policy_changes={"purpose": "metadata_validation"})
    before = _snapshot(s)
    with _metadata_daemon(s) as daemon:
        with pytest.raises(NativeMetadataExecutionError, match="packet_purpose_changed"):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
    assert _snapshot(s) == before


def test_native_metadata_execution_missing_native_acceptance_creates_no_claim(scenario):
    s = scenario
    revision = _prepare_metadata_execution(s, accept=False, ready=False)
    before = _snapshot(s)
    with _metadata_daemon(s) as daemon:
        with pytest.raises(ExecutionPacketAdmissionError):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
    assert _snapshot(s) == before and not (s.workspace / "report.json").exists()


@pytest.mark.parametrize("fault", ["revision", "source", "signature"])
def test_native_metadata_execution_stale_or_forged_native_acceptance_creates_no_claim(scenario, fault):
    s = scenario
    revision = _prepare_metadata_execution(s)
    if fault == "revision":
        revision += 1
    elif fault == "source":
        (s.workspace / "input.json").write_text("changed\n")
    else:
        envelope = _accept(s, parent=s.evidence_id)
        envelope["binding"]["signature"] = "00" * 64
        s.intent.record_evidence(task_cid=s.cid, evidence_kind=EVIDENCE_KIND,
            digest=content_identity(envelope), evidence_id=s.evidence_id,
            body={"execution_packet_contract": envelope})
    before = _snapshot(s)
    with _metadata_daemon(s) as daemon:
        with pytest.raises((ExecutionPacketAdmissionError, ValueError)):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
    assert _snapshot(s) == before and not (s.workspace / "report.json").exists()


def test_native_metadata_execution_rejects_unbound_declared_argv(scenario):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import NativeMetadataExecutionError
    s = scenario
    revision = _prepare_metadata_execution(s, validations=[{"argv": s.packet["validation_argv"]}])
    before = _snapshot(s)
    with _metadata_daemon(s) as daemon:
        with pytest.raises(NativeMetadataExecutionError, match="declared_commands_changed"):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
    assert _snapshot(s) == before


@pytest.mark.parametrize("configuration", [
    {"execution_slice_task_cids": ()}, {"execution_slice_task_cids": ("task:foreign",)},
    {"max_task_attempts": 2},
])
def test_native_metadata_execution_requires_exact_one_task_budget(configuration):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import NativeMetadataExecutionError, _invocation
    args = dict(authority_mode="embedded_exclusive", max_task_attempts=1,
                execution_slice_task_cids=frozenset({"task:packet"}), task_cid="task:packet")
    args.update(configuration)
    args["execution_slice_task_cids"] = frozenset(args["execution_slice_task_cids"])
    with pytest.raises(NativeMetadataExecutionError, match="single_owner_slice"):
        _invocation(**args)


@pytest.mark.parametrize("failed_role", ["producer", "verifier"])
def test_native_metadata_execution_real_failure_releases_without_completion(scenario, failed_role):
    s = scenario
    options = {failed_role: "raise SystemExit(7)\n"}
    revision = _prepare_metadata_execution(s, **options)
    with _metadata_daemon(s) as daemon:
        result = daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert result["passed"] is False and result["claim_released"] is True
        assert len(result["commands"]) == (1 if failed_role == "producer" else 2)
        assert daemon.coordinator.get_lease(result["claim"]["lease_id"]).state.value == "released"
    task = s.intent.get_task(s.cid)
    assert task["status"] == "blocked" and task["body"]["completion"] == "manual"
    assert task["body"]["completion_receipt"]["passed"] is False
    assert _snapshot(s)["provider_invocations"] == []


def test_native_metadata_execution_unknown_scheduler_outcome_retains_claim(scenario, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_metadata_execution as module
    s = scenario
    revision = _prepare_metadata_execution(s)
    def lost_return(*args, **kwargs):
        raise RuntimeError("lost scheduler outcome")
    monkeypatch.setattr(module.ValidationScheduler, "run", lost_return)
    with _metadata_daemon(s) as daemon:
        with pytest.raises(RuntimeError, match="lost scheduler outcome"):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        current = s.intent.get_task(s.cid)
        assert current["status"] == "in_progress"
        lease = daemon.coordinator.get_lease(current["body"]["completion_receipt"]["lease_id"])
        assert lease.state.value == "accepted"
        assert any(r["evidence_kind"] == "native_metadata_execution"
                   for r in s.intent.current_evidence_for_task(s.cid))
        # This fixture's replacement never launches; the test author can close
        # its exact synthetic claim after proving the runtime kept it unknown.
        daemon.coordinator.release(lease, reason="fixture_lost_return_without_launch")
    assert not (s.workspace / "report.json").exists()


def test_native_metadata_execution_drift_between_commands_stops_verifier(scenario):
    s = scenario
    revision = _prepare_metadata_execution(s, producer=(
        "from pathlib import Path\n"
        "Path('report.json').write_text('{}\\n')\n"
        "Path('input.json').write_text('drift\\n')\n"))
    with _metadata_daemon(s) as daemon:
        result = daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert result["passed"] is False and result["claim_released"] is True
        assert len(result["commands"]) == 1
    assert s.intent.get_task(s.cid)["status"] == "blocked"
    assert not (s.workspace / "execution-order.txt").exists()


def test_native_metadata_execution_candidate_cannot_automatically_retry_or_complete(scenario):
    s = scenario
    revision = _prepare_metadata_execution(s)
    with _metadata_daemon(s) as daemon:
        result = daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert result["passed"] is True
        current = s.intent.get_task(s.cid)
        before = _snapshot(s)
        with pytest.raises((ExecutionPacketAdmissionError, ValueError)):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=current["revision"])
        assert _snapshot(s) == before
        assert daemon.claim_next() is None
        assert s.intent.get_task(s.cid)["status"] == "blocked"
def test_native_metadata_execution_existing_native_history_cannot_create_attempt_two(scenario):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_metadata_execution import NativeMetadataExecutionError
    s = scenario
    revision = _prepare_metadata_execution(s)
    with _metadata_daemon(s) as daemon:
        daemon.coordinator.register_task(task_cid=s.cid, body={"execution_packet_required": True})
        claim = daemon.coordinator.claim_task(task_cid=s.cid, owner_session_id="owner:prior-manual",
            execution_packet_authority=NativeExecutionPacketAuthority(daemon.task_source.intent))
        lease = daemon.coordinator.get_lease(claim.lease_id)
        daemon.coordinator.release(lease, reason="fixture_prior_execution")
        before = daemon.coordinator.lease_events(scope_key=lease.scope_key, limit=100)
        with pytest.raises(NativeMetadataExecutionError, match="prior_native_claim"):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert daemon.coordinator.lease_events(scope_key=lease.scope_key, limit=100) == before
        assert daemon.coordinator.get_task_attempt(claim.attempt_id).attempt_number == 1
    assert s.intent.get_task(s.cid)["status"] == "ready"
    assert not (s.workspace / "report.json").exists()


def test_native_metadata_execution_lost_claim_return_never_calls_claim_twice(scenario, monkeypatch):
    s = scenario
    revision = _prepare_metadata_execution(s)
    with _metadata_daemon(s) as daemon:
        original = daemon.coordinator.claim_task
        issued = []
        def lost_return(**kwargs):
            claim = original(**kwargs)
            issued.append(claim)
            raise RuntimeError("lost original claim response")
        with monkeypatch.context() as local:
            local.setattr(daemon.coordinator, "claim_task", lost_return)
            with pytest.raises(RuntimeError, match="lost original claim response"):
                daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
            with pytest.raises(ValueError):
                daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert len(issued) == 1
        lease = daemon.coordinator.get_lease(issued[0].lease_id)
        assert lease.state.value == "accepted"
        assert s.intent.get_task(s.cid)["status"] == "ready"
        # The fixture author knows no executor was called, and releases only
        # this exact synthetic claim after proving the runtime retained it.
        daemon.coordinator.release(lease, reason="fixture_claim_response_lost_without_launch")


def test_native_metadata_execution_native_evidence_error_cannot_reclaim_even_new_contract(scenario, monkeypatch):
    s = scenario
    revision = _prepare_metadata_execution(s)
    with _metadata_daemon(s) as daemon:
        def refused_evidence(**kwargs):
            raise OSError("native evidence write refused")
        with monkeypatch.context() as local:
            local.setattr(daemon.task_source.intent, "record_evidence", refused_evidence)
            with pytest.raises(OSError, match="native evidence write refused"):
                daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        # A fresh signature does not reset the stable task journal or budget.
        _accept(s, parent=s.evidence_id)
        with pytest.raises(FileExistsError):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert daemon.coordinator.lease_events(limit=100) == []
    assert not (s.workspace / "report.json").exists()


def test_native_metadata_execution_lost_cas_return_reconciles_exact_original_receipt(scenario, monkeypatch):
    s = scenario
    revision = _prepare_metadata_execution(s)
    with _metadata_daemon(s) as daemon:
        original = daemon.task_source.compare_and_set_status
        calls = []
        def lost_response(cid, expected_revision, status, receipt=None, **kwargs):
            value = original(cid, expected_revision, status, receipt=receipt, **kwargs)
            calls.append(status)
            if status == "in_progress":
                raise RuntimeError("lost committed CAS response")
            return value
        monkeypatch.setattr(daemon.task_source, "compare_and_set_status", lost_response)
        result = daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        assert result["passed"] is result["claim_released"] is True
        assert calls == ["in_progress", "blocked"]
    assert s.intent.get_task(s.cid)["status"] == "blocked"
    assert (s.workspace / "execution-order.txt").read_text() == "producer\nverifier\n"


def test_native_metadata_execution_failed_claim_readback_retains_control_and_lease(scenario, monkeypatch):
    s = scenario
    revision = _prepare_metadata_execution(s)
    with _metadata_daemon(s) as daemon:
        original = daemon.task_source.get
        def unavailable_after_claim(cid):
            value = original(cid)
            if value.status == "in_progress":
                raise OSError("claimed readback unavailable")
            return value
        with monkeypatch.context() as local:
            local.setattr(daemon.task_source, "get", unavailable_after_claim)
            with pytest.raises(OSError, match="claimed readback unavailable"):
                daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        current = s.intent.get_task(s.cid)
        assert current["status"] == "in_progress"
        lease = daemon.coordinator.get_lease(current["body"]["completion_receipt"]["lease_id"])
        assert lease.state.value == "accepted"
        assert not (s.workspace / "report.json").exists()
        daemon.coordinator.release(lease, reason="fixture_readback_failed_without_executor_launch")


@pytest.mark.parametrize("custody", [None, {"supported": True, "children_reaped": False}])
def test_native_metadata_execution_returned_scheduler_unknown_does_not_release(scenario, monkeypatch, custody):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_metadata_execution as module
    s = scenario
    revision = _prepare_metadata_execution(s)
    def unknown_result(self, commands, **kwargs):
        row = {"command": commands[0].command, "cache_hit": False, "returncode": 75}
        if custody is not None:
            row["validation_process_custody"] = custody
        return {"attempted": True, "passed": False, "elapsed_seconds": 0.25, "results": [row]}
    monkeypatch.setattr(module.ValidationScheduler, "run", unknown_result)
    with _metadata_daemon(s) as daemon:
        with pytest.raises(module.NativeMetadataExecutionError, match="stop_witness_missing"):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        current = s.intent.get_task(s.cid)
        assert current["status"] == "in_progress"
        lease = daemon.coordinator.get_lease(current["body"]["completion_receipt"]["lease_id"])
        assert lease.state.value == "accepted"
        assert len(list(Path(daemon.execution_state_dir).glob("*.command-0.json"))) == 1
        daemon.coordinator.release(lease, reason="fixture_returned_unknown_without_launch")


def test_native_metadata_execution_late_observation_write_error_cannot_release(scenario, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_metadata_execution as module
    s = scenario
    revision = _prepare_metadata_execution(s)
    original = module._durable_json
    observations = []
    def unavailable_observation(path, value):
        if path.name.endswith(".command-0.json"):
            observations.append(value)
            raise OSError("raw observation fsync unavailable")
        return original(path, value)
    monkeypatch.setattr(module, "_durable_json", unavailable_observation)
    with _metadata_daemon(s) as daemon:
        with pytest.raises(OSError, match="raw observation fsync unavailable"):
            daemon.run_native_metadata_validation(task_cid=s.cid, expected_revision=revision)
        current = s.intent.get_task(s.cid)
        assert current["status"] == "in_progress"
        lease = daemon.coordinator.get_lease(current["body"]["completion_receipt"]["lease_id"])
        assert lease.state.value == "accepted"
        assert (s.workspace / "execution-order.txt").read_text() == "producer\n"
        # This test's actual original runner returned a genuine stop witness;
        # the runtime had deliberately not accepted it before fsync failed.
        module._stopped_observation(observations[0], observations[0]["results"][0]["command"])
        daemon.coordinator.release(lease, reason="fixture_real_stop_proven_after_failed_observation_write")
