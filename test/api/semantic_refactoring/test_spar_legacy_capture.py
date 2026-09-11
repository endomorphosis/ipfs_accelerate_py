"""Disposable pidfds, queue files and locks; no host unit or native credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.ops.agent_supervisor import spar_legacy_capture as capture
from scripts.ops.agent_supervisor import spar_merge_owner as role
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import read_process_birth
from test.api.semantic_refactoring.test_spar_legacy_import_plan import (
    prepare, preserved as preserved_fixture, bytes_before,
)


@pytest.fixture
def armed(tmp_path, monkeypatch):
    original = preserved_fixture.__wrapped__(tmp_path)
    queue, context, unknown, receipt_path, receipt_body = prepare(original)
    context["store_id"] = str(queue / "merge_queue.duckdb")
    for name in (".merge_queue.duckdb.rebuild.lock", "train/consumer.lock"):
        path = queue / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    owner = tmp_path / "owner"
    owner.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    context["scope_bindings"][0].update(board_namespace="spar", config_cid=role._cid({}),
        attempt_root=str(tmp_path / "state/lane-0/spar_lane_0_database_portal_attempts"))
    bootstrap = {"plan_root_cid": context["scope_bindings"][0]["plan_cid"]}
    bootstrap["bootstrap_receipt_id"] = role._cid(bootstrap)
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(json.dumps(bootstrap))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    dead_child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    birth = read_process_birth(process.pid).to_dict()
    lane_birth = read_process_birth(dead_child.pid).to_dict()
    dead_child.terminate()
    dead_child.wait(timeout=10)
    identity = {"server_id": "native-test", "process_birth_id": "native-birth",
                "process_birth": birth, "store_id": "native-task-test", "database_uuid": "native-uuid",
                "generation": 101, "fence_epoch": 101}
    owner_path = owner / "quack-state-server.status.json"
    owner_path.write_text(json.dumps({"lifecycle": "ready", "identity": identity}))
    broker_path = owner / "spar-bootstrap-broker.json"
    broker = {"schema": "spar/state-owner-bootstrap-broker@1", "controller_pid": process.pid,
              "failure": "", "current_births": {"spar-0": {
                  "supervisor_process_birth": lane_birth, "daemon_process_birth": lane_birth}}}
    broker_path.write_text(json.dumps(broker))
    board = SimpleNamespace(repo_root=tmp_path, board_namespace="spar", max_lanes=1,
                            task_prefix="SPAR-", runtime_paths={"root": str(tmp_path)},
                            payload={"merge_target_branch": "main", "runtime_paths": {
                                "root": str(tmp_path), "state": str(tmp_path / "state"), "merge_queue": str(queue)}})
    board.path = lambda name: Path(name)
    source = {"head": context["source_commit"], "tree": context["source_tree"]}
    native = SimpleNamespace(
        _load_config=lambda path: (board, {}),
        _runtime_paths=lambda board: {"owner": owner, "bootstrap_receipt": bootstrap_path},
        _identity=role._cid,
        _assert_clean_current_tree=lambda _: (source["head"], source["tree"]),
        _source_forest=lambda _, head: {"head": head},
        source_binding=lambda _: {"head": source["head"], "tree": source["tree"],
                    "forest": {"head": source["head"]}, "config_sha256": role._cid({}).removeprefix("sha256:")},
        authoritative_status=lambda _: {
            "authoritative_task_observation": True, "board_namespace": "spar",
            "owner_identity": identity, "leases": [], "closeout_snapshot": {
                "closeout_facts": {"truncated": False, "all_relations_available": True}}},
    )
    unit = {"Id": capture.UNIT, "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
            "MainPID": str(process.pid), "ControlGroup": "/disposable-test", "WorkingDirectory": str(tmp_path),
            "ExecStart": [["/usr/bin/python3", ["/usr/bin/python3",
                "scripts/materialize_semantic_preserving_remodularization_program.py", "supervise", "--implement"], False]],
            "Restart": "on-failure", "SendSIGKILL": "yes", "TimeoutStopUSec": "2min",
            "RefuseManualStart": "no", "Conditions": [], "NeedDaemonReload": "no",
            "Delegate": "no", "ExitType": "main"}
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_text(str(process.pid) + "\n")
    (cgroup / "cgroup.events").write_text("populated 1\nfrozen 0\n")
    original_directory = role._open_directory
    monkeypatch.setattr(capture, "_native_operator", lambda _: native)
    monkeypatch.setattr(capture, "_repository_id", lambda _: "repo:spar")
    monkeypatch.setattr(capture, "configured_queue_root", lambda _: queue)
    monkeypatch.setattr(capture, "_unit", lambda: json.loads(json.dumps(unit)))
    monkeypatch.setattr(capture, "_namespaces", lambda _: {"mnt": {"state": "unknown", "error_class": "PermissionError"}})
    monkeypatch.setattr(role, "_open_directory", lambda path: original_directory(
        cgroup if str(path) == "/sys/fs/cgroup/disposable-test" else path))
    session = None
    try:
        session = capture.RetainedNativeLegacySession(repository_root=tmp_path, config_path=config_path,
                    expected_source_commit=source["head"], expected_source_tree=source["tree"])

        def close_native():
            unit.update(Restart="no", SendSIGKILL="no", TimeoutStopUSec="infinity", RefuseManualStart="yes",
                        Conditions=[["ConditionPathExists", False, True, str(tmp_path / "HOLD")]])
            (tmp_path / "HOLD").write_text("disposable capture hold")
            process.terminate()
            process.wait(timeout=10)
            unit.update(MainPID="0", ActiveState="inactive", SubState="dead")
            owner_path.write_text(json.dumps({"lifecycle": "stopped", "identity": identity}))
            (cgroup / "cgroup.procs").write_text("")
            (cgroup / "cgroup.events").write_text("populated 0\nfrozen 0\n")

        yield SimpleNamespace(session=session, close_native=close_native, unit=unit, source=source,
                              context=context, queue=queue, cgroup=cgroup, owner_path=owner_path,
                              process=process, tmp=tmp_path, unknown=unknown, receipt_path=receipt_path,
                              receipt_body=receipt_body, broker_path=broker_path)
    finally:
        if session:
            session.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


def do_capture(armed):
    return armed.session.capture(destination=armed.tmp / "captured",
                                 inspection_destination=armed.tmp / "inspection", context=armed.context)


def test_retained_real_pidfd_complete_capture_preserves_unknown(armed):
    armed.close_native()
    before = bytes_before(armed.queue)
    result = do_capture(armed)
    receipt = result.require_current()
    assert receipt["capture_coherent"] is True
    assert receipt["consumer_processes_closed"] is True
    assert receipt["pre_stop_namespaces"]["mnt"]["state"] == "unknown"
    for key in ("callback_settled", "signing_authority", "source_admitted", "completion_authority"):
        assert receipt[key] is False
    assert bytes_before(result.path) == before == bytes_before(armed.queue)
    assert len(receipt["manifest"]["receipt_imports"]) == 1
    # The OFD writer lock remains held after the producer read/closed the same inode.
    code = "import duckdb,sys; duckdb.connect(sys.argv[1], config={'threads': 1})"
    denied = subprocess.run([sys.executable, "-c", code, str(armed.queue / "merge_queue.duckdb")],
                            capture_output=True, text=True, timeout=10)
    assert denied.returncode != 0 and "lock" in denied.stderr.lower()
    armed.session.close()
    with pytest.raises(role.SparMergeOwnerError, match="already closed"):
        result.require_current()


@pytest.mark.parametrize("field,value", [("Restart", "on-failure"), ("SendSIGKILL", "yes"),
    ("TimeoutStopUSec", "2min"), ("RefuseManualStart", "no"), ("Conditions", []), ("MainPID", "123")])
def test_capture_refuses_missing_inhibition_or_relaunch(armed, field, value):
    armed.close_native()
    armed.unit[field] = value
    with pytest.raises(role.SparMergeOwnerError):
        do_capture(armed)
    assert not (armed.tmp / "captured").exists()


def test_capture_refuses_live_retained_pidfd_even_if_projection_says_closed(armed):
    armed.unit.update(Restart="no", SendSIGKILL="no", TimeoutStopUSec="infinity", RefuseManualStart="yes",
                      MainPID="0", ActiveState="inactive", SubState="dead",
                      Conditions=[["ConditionPathExists", False, True, str(armed.tmp / "HOLD")]])
    (armed.tmp / "HOLD").write_text("disposable")
    with pytest.raises(role.SparMergeOwnerError, match="pidfd has not exited"):
        do_capture(armed)
    assert armed.process.poll() is None


@pytest.mark.parametrize("kind", ["missing", "unreadable", "populated"])
def test_capture_requires_positive_cgroup_empty_observation(armed, monkeypatch, kind):
    armed.close_native()
    if kind == "missing":
        (armed.cgroup / "cgroup.events").unlink()
    elif kind == "populated":
        (armed.cgroup / "cgroup.events").write_text("populated 1\n")
    else:
        monkeypatch.setattr(capture, "_cgroup_population", lambda _: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises((role.SparMergeOwnerError, OSError)):
        do_capture(armed)
    assert not (armed.tmp / "captured").exists()


@pytest.mark.parametrize("kind", ["owner", "source", "broker"])
def test_capture_refuses_changed_current_native_identity(armed, kind):
    armed.close_native()
    if kind == "owner":
        body = json.loads(armed.owner_path.read_text())
        body["identity"]["generation"] += 1
        armed.owner_path.write_text(json.dumps(body))
    elif kind == "source":
        armed.source["head"] = "e" * 40
    else:
        body = json.loads(armed.broker_path.read_text())
        body["failure"] = "unknown"
        armed.broker_path.write_text(json.dumps(body))
    with pytest.raises(role.SparMergeOwnerError):
        do_capture(armed)


def test_capture_rejects_receipt_substitution_after_copy(armed):
    armed.close_native()
    result = do_capture(armed)
    (result.path / "private/callback-signing-material").write_bytes(b"substituted")
    with pytest.raises(role.SparMergeOwnerError):
        result.require_current()


def test_audit_json_cannot_recreate_retained_capture(armed):
    armed.close_native()
    result = do_capture(armed)
    forged = capture.CoherentLegacyCapture(armed.session, result.path, result.receipt, result._identities)
    with pytest.raises(role.SparMergeOwnerError, match="not retained by its producer"):
        forged.require_current()


def test_capture_cannot_be_changed_through_audit_export(armed):
    armed.close_native()
    result = do_capture(armed)
    exported = result.receipt
    exported["callback_settled"] = True
    exported["manifest"]["source_commit"] = "f" * 40
    assert result.require_current()["callback_settled"] is False
    assert result.receipt["manifest"]["source_commit"] == armed.context["source_commit"]


@pytest.mark.parametrize("lock", ["consumer", "database"])
def test_capture_does_not_steal_an_existing_lock(armed, lock):
    armed.close_native()
    path = armed.queue / ("train/consumer.lock" if lock == "consumer" else "merge_queue.duckdb")
    code = ("import sys,fcntl,time; f=open(sys.argv[1]); fcntl.flock(f,fcntl.LOCK_EX); print('ready',flush=True); time.sleep(60)"
            if lock == "consumer" else
            "import sys,duckdb,time; c=duckdb.connect(sys.argv[1],config={'threads':1}); print('ready',flush=True); time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises((role.SparMergeOwnerError, BlockingIOError)):
            do_capture(armed)
        assert not (armed.tmp / "captured").exists()
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=10)
        child.stdout.close()


def installed_origin(armed):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
            destination=armed.tmp / "prepared", manifest=captured.receipt["manifest"])
    result = origin.install_captured_queue(captured, prepared)
    return captured, prepared, result


def test_installed_origin_survives_two_native_owner_generations(armed):
    from scripts.ops.agent_supervisor import spar_merge_owner_handoff as handoff
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    from test.api.semantic_refactoring.test_spar_merge_owner_bootstrap import start, attach, attach_recovery

    before = bytes_before(armed.queue)
    captured, prepared, result = installed_origin(armed)
    assert result["installed"] is True and result["callback_settled"] is False
    assert bytes_before(captured.path) == before
    assert (armed.queue / origin.REQUIRED_MARKER).exists()
    for name, value in before.items():
        if name != "merge_queue.duckdb":
            assert (armed.queue / name).read_bytes() == value
    armed.session.close()
    generations = []
    for _ in range(2):
        resumed = handoff._load_origin(armed.queue / "merge_queue.duckdb", profile=handoff.LEGACY_PROFILE,
                    repository_id=armed.context["repository_id"], target_branch=armed.context["target_branch"],
                    store_id=armed.context["store_id"], scopes=armed.context["scope_bindings"])
        assert resumed.database_uuid == prepared.database_uuid
        server = start(resumed, armed.tmp / "successor")
        connection = recovery_connection = None
        try:
            generations.append(server.identity.generation)
            connection, queue = attach(server)
            row = json.loads(queue.call("get", request_id=armed.unknown.request_id)["request_json"])
            assert row["status"] == "processing"
            assert row["claim_token"] == armed.unknown.claim_token
            assert row["claim_generation"] == armed.unknown.claim_generation
            assert row["consumer_id"] == "retained-old-consumer"
            recovery_connection, recovery = attach_recovery(server, resumed.manifest)
            receipt = recovery.get_receipt(armed.receipt_path.stem, revision=1)
            assert receipt["receipt"] == armed.receipt_body
        finally:
            if connection:
                connection.close()
            if recovery_connection:
                recovery_connection.close()
            server.stop()
    assert generations[1] == generations[0] + 1


def test_origin_install_refuses_a_closed_session_before_canonical_effect(armed):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
            destination=armed.tmp / "prepared", manifest=captured.receipt["manifest"])
    before = bytes_before(armed.queue)
    armed.session.close()
    with pytest.raises(role.SparMergeOwnerError, match="already closed"):
        origin.install_captured_queue(captured, prepared)
    assert bytes_before(armed.queue) == before


def test_interrupted_origin_install_keeps_native_requirement_and_old_bytes(armed, monkeypatch):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    from scripts.ops.agent_supervisor import spar_merge_owner_handoff as handoff
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
            destination=armed.tmp / "prepared", manifest=captured.receipt["manifest"])
    old_database = (armed.queue / "merge_queue.duckdb").read_bytes()
    replace = os.replace

    def crash(source, destination):
        if Path(source).name == ".native-legacy-database.prepared":
            raise OSError("disposable crash before replacement")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="disposable crash"):
        origin.install_captured_queue(captured, prepared)
    assert (armed.queue / origin.REQUIRED_MARKER).exists()
    assert (armed.queue / "merge_queue.duckdb").read_bytes() == old_database
    assert (captured.path / "merge_queue.duckdb").read_bytes() == old_database
    armed.session.close()
    with pytest.raises(role.SparMergeOwnerError, match="no native fresh origin"):
        handoff._load_origin(armed.queue / "merge_queue.duckdb", profile=handoff.LEGACY_PROFILE,
                    repository_id=armed.context["repository_id"], target_branch=armed.context["target_branch"],
                    store_id=armed.context["store_id"], scopes=armed.context["scope_bindings"])


def attach_disposable_sentinel(armed, monkeypatch):
    armed.unit.update(Restart="no", SendSIGKILL="no", TimeoutStopUSec="infinity", RefuseManualStart="yes",
                      Delegate="yes", ExitType="cgroup",
                      Conditions=[["ConditionPathExists", False, True, str(armed.tmp / "HOLD")]])
    original_run = subprocess.run

    def attach(argv, **kwargs):
        if argv[:3] == ["busctl", "--user", "call"]:
            assert argv[6:11] == ["AttachProcessesToUnit", "ssau", capture.UNIT, "", "1"]
            pid = int(argv[11])
            (armed.cgroup / "cgroup.procs").write_text(f"{armed.process.pid}\n{pid}\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return original_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", attach)
    sentinel = armed.session.retain_workflow_sentinel()
    pid = sentinel["birth"]["pid"]
    assert os.readlink(f"/proc/{pid}/cwd") == "/"
    assert set(os.listdir(f"/proc/{pid}/fd")) == {"0", "1", "2"}
    armed.close_native()
    armed.unit.update(ActiveState="active", SubState="running")
    (armed.cgroup / "cgroup.procs").write_text(str(pid) + "\n")
    (armed.cgroup / "cgroup.events").write_text("populated 1\nfrozen 0\n")
    return sentinel


def test_only_reviewed_workflow_sentinel_can_remain_during_capture(armed, monkeypatch):
    sentinel = attach_disposable_sentinel(armed, monkeypatch)
    captured = do_capture(armed)
    closure = captured.require_current()["closure"]
    assert closure["native_actors_closed"] is True
    assert closure["cgroup_empty_observed"] is False
    assert closure["workflow_sentinel"]["birth"] == sentinel["birth"]
    assert closure["workflow_sentinel"]["observed_cgroup_members"] == [sentinel["birth"]["pid"]]
    armed.session.close()
    assert armed.session.sentinel_close_receipt == {
        "birth": sentinel["birth"], "pidfd_exit_observed": True, "returncode": 0}


def test_unregistered_actor_beside_sentinel_blocks_copy(armed, monkeypatch):
    sentinel = attach_disposable_sentinel(armed, monkeypatch)
    (armed.cgroup / "cgroup.procs").write_text(str(sentinel["birth"]["pid"]) + "\n" + str(os.getpid()) + "\n")
    with pytest.raises(role.SparMergeOwnerError, match="unknown actor"):
        do_capture(armed)
    assert not (armed.tmp / "captured").exists()


def test_sentinel_cannot_be_attached_after_stop_marker(armed):
    armed.close_native()
    with pytest.raises(role.SparMergeOwnerError, match="before the hold"):
        armed.session.retain_workflow_sentinel()


@pytest.mark.parametrize("field", ["repository_id", "target_branch", "source_commit", "store_id", "scope_bindings"])
def test_import_context_cannot_substitute_the_native_namespace(armed, field):
    armed.close_native()
    armed.context[field] = [] if field == "scope_bindings" else "foreign"
    with pytest.raises(role.SparMergeOwnerError, match="sealed namespace"):
        do_capture(armed)
    assert not (armed.tmp / "captured").exists()


def test_native_legacy_entry_requires_installed_origin_and_current_coordinates(armed, monkeypatch):
    from dataclasses import replace
    from scripts.ops.agent_supervisor import spar_merge_owner_handoff as handoff
    from ipfs_accelerate_py.agent_supervisor.merge import checkout_lock
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport
    from test.api.test_agent_supervisor_quack_state_server import _compatible_report
    from test.api.semantic_refactoring.test_launch_source_amendment_task_source import _fixture

    _, _, amendment, *_ = _fixture()
    amendment = replace(amendment, board_namespace="spar", launch_config_cid=role._cid({}),
                         bootstrap_plan_root_cid=armed.context["scope_bindings"][0]["plan_cid"], amendment_id="")
    monkeypatch.setattr(checkout_lock, "checkout_repository_id", lambda _: "repo:spar")
    armed.queue.chmod(0o700)
    with pytest.raises(role.SparMergeOwnerError, match="no native fresh origin"):
        handoff.start_native_queue_for_launch(board=armed.session.board, paths={}, amendment=amendment,
                                              profile=handoff.LEGACY_PROFILE)
    _, prepared, _ = installed_origin(armed)
    armed.session.close()
    owner = handoff.start_native_queue_for_launch(board=armed.session.board, paths={}, amendment=amendment,
                profile=handoff.LEGACY_PROFILE, transport=FakeQuackTransport(), capability_probe=_compatible_report)
    try:
        assert owner.server.ready()["ready"] is True
        assert owner.prepared.database_uuid == prepared.database_uuid
        assert owner.prepared.receipt_imports
    finally:
        owner.close()
    foreign = replace(amendment, launch_config_cid="sha256:" + "f" * 64, amendment_id="")
    with pytest.raises(role.SparMergeOwnerError, match="namespace differs"):
        handoff.start_native_queue_for_launch(board=armed.session.board, paths={}, amendment=foreign,
                                              profile=handoff.LEGACY_PROFILE)


def test_durable_requirement_selects_native_validation_even_after_interruption(tmp_path):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    from scripts.ops.agent_supervisor.spar_merge_owner_handoff import LEGACY_PROFILE
    assert origin.required_profile(tmp_path, "") == ""
    marker = tmp_path / origin.REQUIRED_MARKER
    marker.write_text("incomplete marker is still a veto on legacy fallback")
    assert origin.required_profile(tmp_path, "") == LEGACY_PROFILE
    marker.unlink()
    marker.symlink_to(tmp_path / "absent")
    assert origin.required_profile(tmp_path, "") == LEGACY_PROFILE


def test_origin_install_refuses_candidate_changed_after_logical_validation(armed, monkeypatch):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
            destination=armed.tmp / "prepared", manifest=captured.receipt["manifest"])
    before = bytes_before(armed.queue)
    sync = origin._sync_directory
    changed = False

    def mutate_after_validation(directory):
        nonlocal changed
        if not changed:
            changed = True
            with role.open_duckdb_connection(prepared.database_path, prefer_quack=False) as connection:
                connection.execute("DELETE FROM merge_requests WHERE request_id=?", [armed.unknown.request_id])
                connection.execute("CHECKPOINT")
        return sync(directory)

    monkeypatch.setattr(origin, "_sync_directory", mutate_after_validation)
    with pytest.raises(role.SparMergeOwnerError, match="prepared native database changed"):
        origin.install_captured_queue(captured, prepared)
    assert changed
    assert (armed.queue / "merge_queue.duckdb").read_bytes() == before["merge_queue.duckdb"]
    assert bytes_before(captured.path) == before

def test_origin_install_refuses_staging_changed_before_replacement(armed, monkeypatch):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
            destination=armed.tmp / "prepared", manifest=captured.receipt["manifest"])
    before = bytes_before(armed.queue)
    copy = role.copy_entry
    changed = False

    def mutate_staging(root, entry, destination, **kwargs):
        nonlocal changed
        result = copy(root, entry, destination, **kwargs)
        if destination is not None and Path(destination).name == ".native-legacy-database.prepared":
            changed = True
            Path(destination).write_bytes(b"substituted after staging copy")
        return result

    monkeypatch.setattr(role, "copy_entry", mutate_staging)
    with pytest.raises(role.SparMergeOwnerError):
        origin.install_captured_queue(captured, prepared)
    assert changed
    assert (armed.queue / "merge_queue.duckdb").read_bytes() == before["merge_queue.duckdb"]
    assert bytes_before(captured.path) == before


def test_path_substitution_while_validated_database_handle_is_open_must_be_denied(armed, monkeypatch):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    from scripts.ops.agent_supervisor.spar_merge_owner_handoff import ORIGIN_TABLE
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
        destination=armed.tmp / 'prepared', manifest=captured.receipt['manifest'])
    canonical_before = (armed.queue / 'merge_queue.duckdb').read_bytes()
    captured_before = bytes_before(captured.path)
    native_inventory = role.inventory
    replacement = armed.tmp / 'not-a-database'
    substituted = b'UNVALIDATED PATHNAME SUBSTITUTION DURING OPEN DATABASE HANDLE\n'
    replacement.write_bytes(substituted)
    swapped = False

    def swap_after_logical_inventory(connection):
        nonlocal swapped
        value = native_inventory(connection)
        if not swapped and ORIGIN_TABLE in value:
            swapped = True
            os.replace(replacement, prepared.database_path)
        return value

    monkeypatch.setattr(role, 'inventory', swap_after_logical_inventory)
    rejected = False
    try:
        origin.install_captured_queue(captured, prepared)
    except Exception:
        rejected = True
    assert swapped
    assert bytes_before(captured.path) == captured_before
    actual = (armed.queue / 'merge_queue.duckdb').read_bytes()
    assert rejected and actual == canonical_before, (
        f'installation_rejected={rejected}, canonical_contains_unvalidated_substitution={actual == substituted}'
    )


def test_prepared_cursor_substitution_cannot_reset_captured_positions(armed):
    import hashlib
    from dataclasses import replace
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    from scripts.ops.agent_supervisor import spar_merge_owner_handoff as handoff
    from test.api.semantic_refactoring.test_spar_merge_owner_bootstrap import start, attach_recovery
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import _POST_MERGE_RECOVERY_CURSOR_SCHEMA, _canonical_json
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    coordinates={'target_repository_id':armed.context['repository_id'],
                 'target_branch':armed.context['target_branch'],
                 'attempt_root':armed.context['scope_bindings'][0]['attempt_root']}
    body={'schema':_POST_MERGE_RECOVERY_CURSOR_SCHEMA, **coordinates,
          'cursors':{stage:'retained:'+stage for stage in role.STAGES}}
    body['state_id']=content_identity(body)
    path=armed.queue/'train/post-merge-recovery-cursors'/(hashlib.sha256(_canonical_json(coordinates)).hexdigest()+'.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(body))
    armed.close_native()
    captured=do_capture(armed)
    prepared=role.prepare_offline_clone(offline_root=captured.path,
        destination=armed.tmp/'prepared',manifest=captured.receipt['manifest'])
    assert len(prepared.cursor_imports)==1
    reset={stage:'' for stage in role.STAGES}
    substituted={**prepared.cursor_imports[0], 'cursors':reset, 'state_cid':role._cid(reset)}
    try:
        origin.install_captured_queue(captured,replace(prepared,cursor_imports=(substituted,)))
    except role.SparMergeOwnerError:
        return
    armed.session.close()
    resumed=handoff._load_origin(armed.queue/'merge_queue.duckdb',profile=handoff.LEGACY_PROFILE,
        repository_id=armed.context['repository_id'],target_branch=armed.context['target_branch'],
        store_id=armed.context['store_id'],scopes=armed.context['scope_bindings'])
    server=start(resumed,armed.tmp/'successor')
    connection=None
    try:
        connection,api=attach_recovery(server,resumed.manifest)
        actual=api.load_cursors()['cursors']
        assert actual==body['cursors'], f'captured_cursor_positions_reset={actual == reset}'
    finally:
        if connection:connection.close()
        server.stop()


def test_prepared_inventory_cannot_redefine_original_population(armed):
    from dataclasses import replace
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
        destination=armed.tmp / "prepared", manifest=captured.receipt["manifest"])
    original = (armed.queue / "merge_queue.duckdb").read_bytes()
    with role.open_duckdb_connection(prepared.database_path, prefer_quack=False) as connection:
        connection.execute("DELETE FROM merge_requests WHERE request_id=?", [armed.unknown.request_id])
        replacement_baseline = role.inventory(connection)
    with pytest.raises(role.SparMergeOwnerError, match="preservation inventory differs"):
        origin.install_captured_queue(captured, replace(prepared, preserved_inventory=replacement_baseline))
    assert (armed.queue / "merge_queue.duckdb").read_bytes() == original


def test_locked_candidate_veto_does_not_drop_existing_posix_writer_lock(armed):
    from scripts.ops.agent_supervisor import spar_legacy_origin as origin
    armed.close_native()
    captured = do_capture(armed)
    prepared = role.prepare_offline_clone(offline_root=captured.path,
        destination=armed.tmp / 'prepared', manifest=captured.receipt['manifest'])
    with role.open_duckdb_connection(prepared.database_path, prefer_quack=False):
        assert role._writer_lock_held(prepared.database_path)
        with pytest.raises(role.SparMergeOwnerError, match='observed kernel lock'):
            origin.install_captured_queue(captured, prepared)
        assert role._writer_lock_held(prepared.database_path)
