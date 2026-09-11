"""Independent lane admission must coexist with current retained native owners."""

from __future__ import annotations

import shutil
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import owner_task_quarantine as local
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_custody_prelaunch import native_independent_launch_scope
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import _read_control_plane_source_snapshot
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.owner_task_quarantine import (
    QuarantineDenied,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
from test.api.test_owner_task_quarantine_custody import entered
from test.api.test_agent_supervisor_quack_owner_mutation import (
    _admitted_observation, _isolation_receipt, _isolation_server_kwargs,
)


@pytest.fixture
def four_native_lanes(tmp_path, monkeypatch):
    repo, retained, _, attempt, _ = entered(tmp_path)
    server = None
    consumer = None
    stopping = threading.Event()
    lanes = {}
    try:
        retained.task_shard_count = 4
        retained.strict_task_sharding = True
        home = retained._task_home_shard_index(attempt.task_alias)
        retained.task_shard_index = home
        aliases = {}
        for number in range(100, 300):
            alias = f"PCTDD-{number}"
            aliases.setdefault(retained._task_home_shard_index(alias), alias)
        assert set(aliases) == set(range(4))
        for lane, alias in aliases.items():
            retained.task_source.intent.upsert_task(
                task_cid=f"task:independent:{lane}", task_alias=alias,
                goal_cid="goal:pctdd", status="ready",
            )
        control = repo / "control" / "control.duckdb"
        control.parent.mkdir()
        shutil.copyfile(repo / "control.duckdb", control)
        receipt_path, receipt = _isolation_receipt(repo)
        server = build_server(
            database_path=control, state_dir=receipt_path.parent,
            **_isolation_server_kwargs(receipt), store_id="control/control.duckdb",
            repository_id="repository:test", isolation_receipt_path=receipt_path,
            isolation_observer=_admitted_observation,
            capability_probe=lambda **_: probe_quack_capabilities(),
        )
        identity = server.start()
        for key, value in {
            "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": identity.secret_handle,
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": "control/control.duckdb",
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": "test-quarantine",
            "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION": "test-quarantine",
            "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": str(identity.generation),
            "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": str(identity.schema_revision),
            "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(repo),
        }.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)

        def consume():
            while not stopping.is_set():
                server.service_mutation_inbox(max_requests=8)
                stopping.wait(0.01)

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        retained._task_source.close()
        retained._task_source = DatabaseTaskSource(
            identity.listen_uri, install_schema=False, owner_id=retained.owner_session_id,
        )
        retained._database_portal_bridge.task_source = retained._task_source
        retained.authority_mode = "quack"
        before = local.capture(retained, attempt)
        retained.acknowledge_owner_task_quarantine(attempt.attempt_id)
        lanes[home] = retained
        for lane in range(4):
            if lane == home:
                continue
            root = repo / "lanes" / f"lane-{lane}"
            root.mkdir(parents=True)
            source = DatabaseTaskSource(
                identity.listen_uri, install_schema=False, owner_id=f"owner:lane:{lane}",
            )
            peer = DatabaseImplementationDaemon(
                database_path=root / "quack-lane-control.duckdb",
                coordination_path=root / "quack-lane-coordination.duckdb",
                execution_path=root / "quack-lane-control.execution.duckdb",
                owner_session_id=f"owner:lane:{lane}", authority_mode="quack",
                task_source_kind="duckdb", quack_uri=identity.listen_uri,
                task_source=source, task_prefix="PCTDD-", require_real_execution=True,
                task_shard_count=4, task_shard_index=lane, strict_task_sharding=True,
            )
            lanes[lane] = peer
            peer.open()
            retained_bridge = retained._database_portal_bridge
            bridge = DatabasePortalExecutionBridge(
                task_source=source, attempt_root=root / "portal-attempts",
                portal_factory=retained_bridge.portal_factory,
                workspace_repository_root=retained_bridge.workspace_repository_root,
                workspace_root=retained_bridge.workspace_root,
            )
            peer.bind_execution_callbacks(provider_fn=bridge.run_provider,
                effect_fn=bridge.apply_effect, validation_fn=bridge.validate_effect)
            peer.bind_database_portal_bridge(bridge)
        yield repo, retained, attempt, lanes, before
    finally:
        for daemon in reversed(list(lanes.values())):
            daemon.close()
        if retained not in lanes.values():
            retained.close()
        stopping.set()
        if consumer is not None:
            consumer.join(5)
        if server is not None:
            server.stop()


def test_four_native_lane_writers_can_admit_independent_claims_without_peer_reopen(four_native_lanes):
    repo, retained, attempt, lanes, before = four_native_lanes
    home = retained.task_shard_index
    assert len(lanes) == 4
    assert all(daemon._retained_embedded_writer_fence_current() for daemon in lanes.values())
    claimed = []
    imported_source = _read_control_plane_source_snapshot()
    for lane, daemon in sorted(lanes.items()):
        reconciliation = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="four-current-native-lanes", force=True,
        )
        assert reconciliation["blocked"] is False
        if lane == home:
            assert reconciliation["independent_work_admitted"] is True
            assert reconciliation["safe_to_restart"] is False
            assert reconciliation["quiesced"] is False
        with native_independent_launch_scope(
            daemon=daemon, binding={"lane": lane}, imported_source=imported_source,
            source_probe=_read_control_plane_source_snapshot,
        ) as admission:
            status = admission.diagnostic()
            assert status["independent_work_admitted"] is True
            assert status["safe_to_restart"] is False
            assert status["quiesced"] is False
            assert status["completion_authorized"] is False
        with pytest.raises(QuarantineDenied, match="scope_expired"):
            admission.consume({"lane": lane}, lambda: pytest.fail("expired scope consumed"))
        current = daemon.claim_next()
        assert current is not None and current.task_cid == f"task:independent:{lane}"
        claimed.append(current)
    assert len({item.claim_id for item in claimed}) == 4
    assert len({item.task_cid for item in claimed}) == 4
    assert local.capture(retained, attempt) == before
    with pytest.raises(QuarantineDenied):
        retained.resume_attempt(attempt, provider_fn=lambda _: pytest.fail("retained callback replayed"))
    assert local.capture(retained, attempt) == before


def test_native_scope_preserves_unknown_custody_on_source_launch_ack_and_owner_changes(four_native_lanes):
    import copy
    repo, retained, attempt, _lanes, before = four_native_lanes
    source = _read_control_plane_source_snapshot()
    current_source = copy.deepcopy(source)
    binding = {"command": ["native-child"], "owner": retained.owner_session_id}
    with native_independent_launch_scope(
        daemon=retained, binding=binding, imported_source=source,
        source_probe=lambda: current_source,
    ) as admission:
        with pytest.raises(QuarantineDenied, match="child_binding_changed"):
            admission.consume({**binding, "command": ["different-child"]},
                              lambda: pytest.fail("substituted launch"))
        current_source["source_id"] = "source:changed-after-audit"
        with pytest.raises(QuarantineDenied, match="loaded_source_changed"):
            admission.consume(binding, lambda: pytest.fail("changed source launched"))
        current_source["source_id"] = source["source_id"]
        current_source["repository_revision"] = "1" * 40
        assert admission.diagnostic()["independent_work_admitted"] is True
        connection = retained._require_connection()
        key = local.ACK_KEY + ":" + attempt.attempt_id
        ack = connection.execute("SELECT value FROM daemon_execution_metadata WHERE key=?", [key]).fetchone()[0]
        connection.execute("UPDATE daemon_execution_metadata SET value='{}' WHERE key=?", [key])
        try:
            with pytest.raises(QuarantineDenied, match="local_ack_changed"):
                admission.consume(binding, lambda: pytest.fail("partial ack launched"))
        finally:
            connection.execute("UPDATE daemon_execution_metadata SET value=? WHERE key=?", [ack, key])
        old_owner = retained.owner_session_id
        retained.owner_session_id = "owner:substituted"
        try:
            with pytest.raises(QuarantineDenied, match="execution_owner_changed"):
                admission.consume(binding, lambda: pytest.fail("substituted owner launched"))
        finally:
            retained.owner_session_id = old_owner
        from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine as workspace
        registry_record = next(workspace.registry(repo).glob("*.json"))
        original_record = registry_record.read_bytes()
        registry_record.write_text("{}")
        try:
            with pytest.raises(QuarantineDenied):
                admission.consume(binding, lambda: pytest.fail("incomplete root custody launched"))
        finally:
            registry_record.write_bytes(original_record)
        assert admission.diagnostic()["safe_to_restart"] is False
        assert local.capture(retained, attempt) == before
        assert retained._retained_embedded_writer_fence_current()
    assert local.capture(retained, attempt) == before


def test_consumed_native_scope_launches_once_then_new_native_owner_claims_independent_work(four_native_lanes):
    import sys
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
        SupervisedChildSpec, launch_supervised_child,
    )
    repo, retained, attempt, lanes, before = four_native_lanes
    lane = retained.task_shard_index
    source = _read_control_plane_source_snapshot()
    old_bridge = retained._database_portal_bridge
    uri = retained._database_portal_bridge.task_source.intent.database_path
    spec = SupervisedChildSpec(
        repo_root=repo, command=(sys.executable, "-c", "print('isolated-native-launch')"),
        log_path=repo / "child.log", child_pid_path=repo / "child.pid",
    )
    binding = {"command": list(spec.command), "owner": retained.owner_session_id}
    children = []
    def start():
        assert retained._closed
        assert all(peer._retained_embedded_writer_fence_current()
                   for index, peer in lanes.items() if index != lane)
        child = launch_supervised_child(spec)
        children.append(child)
        return child
    with native_independent_launch_scope(
        daemon=retained, binding=binding, imported_source=source,
        source_probe=_read_control_plane_source_snapshot,
    ) as admission:
        child = admission.consume(binding, start)
        with pytest.raises(QuarantineDenied, match="scope_expired"):
            admission.consume(binding, lambda: pytest.fail("launch scope replayed"))
    assert len(children) == 1
    # Wait only for the disposable print-only child; no provider is launched.
    import os
    _, status = os.waitpid(child.pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert "isolated-native-launch" in (repo / "child.log").read_text()
    fresh_source = DatabaseTaskSource(uri, install_schema=False, owner_id=retained.owner_session_id)
    successor = DatabaseImplementationDaemon(
        database_path=repo / "control.duckdb",
        coordination_path=retained.coordination_path, execution_path=retained.execution_path,
        owner_session_id=retained.owner_session_id, authority_mode="quack",
        task_source_kind="duckdb", quack_uri=uri, task_source=fresh_source,
        task_prefix="PCTDD-", require_real_execution=True,
        task_shard_count=4, task_shard_index=lane, strict_task_sharding=True,
    )
    try:
        successor.open()
        bridge = DatabasePortalExecutionBridge(
            task_source=fresh_source, attempt_root=old_bridge.attempt_root,
            portal_factory=old_bridge.portal_factory,
            workspace_repository_root=old_bridge.workspace_repository_root,
            workspace_root=old_bridge.workspace_root,
        )
        successor.bind_execution_callbacks(provider_fn=bridge.run_provider,
            effect_fn=bridge.apply_effect, validation_fn=bridge.validate_effect)
        successor.bind_database_portal_bridge(bridge)
        assert successor.process_instance_id != retained.process_instance_id
        # Merely reopening cannot inherit the old controller's acknowledgement.
        with pytest.raises(QuarantineDenied, match="local_ack_changed"):
            local.current(successor)
        local.refresh(successor)  # The production child does this before its startup audit.
        result = successor.reconcile_quiesced_database_portal_attempts(trigger="new-native-child", force=True)
        assert result["independent_work_admitted"] is True
        assert result["safe_to_restart"] is False
        assert result["quiesced"] is False
        claimed = successor.claim_next()
        assert claimed.task_cid == f"task:independent:{lane}"
        assert local.capture(successor, successor.get_attempt(attempt.attempt_id)) == before
        with pytest.raises(QuarantineDenied):
            successor.resume_attempt(successor.get_attempt(attempt.attempt_id),
                provider_fn=lambda _: pytest.fail("retained callback replayed"))
    finally:
        successor.close()


@pytest.mark.parametrize("admission_case", ["native", "diagnostic_only", "changed_source", "unknown_child"])
def test_real_controller_spawn_reaudits_native_lane_under_owner_and_launch_fences(four_native_lanes, monkeypatch, admission_case):
    import os
    import sys
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import DatabaseProgramConfig
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor, PortalSupervisorConfig,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        exclusive_file_lock, quack_owner_mutation_write_lock_path,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import serialized_lock_update
    repo, retained, attempt, lanes, before = four_native_lanes
    lane = next(index for index in lanes if index != retained.task_shard_index)
    prior = lanes[lane]
    state_dir = prior.execution_path.parent
    uri = prior._quack_uri
    prior.close()
    program = DatabaseProgramConfig(
        authority_mode="quack", task_source_kind="duckdb", quack_endpoint=uri,
        endpoint_secret_handle=os.environ["IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE"],
        store_id=os.environ["IPFS_ACCELERATE_AGENT_STATE_STORE_ID"],
        store_generation="test-quarantine", schema_revision="test-quarantine",
        failover_policy="fail_closed",
    )
    config = PortalSupervisorConfig(
        todo_path=repo / "tasks.duckdb", state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json", events_path=state_dir / "events.jsonl",
        state_dir=state_dir, state_prefix=f"lane-{lane}", task_prefix="PCTDD-",
        database_program=program, database_owner_session_id=prior.owner_session_id,
        repo_root=repo, worktree_root=retained._database_portal_bridge.workspace_root,
        merge_target_branch="main", task_shard_count=4, task_shard_index=lane,
        strict_task_sharding=True, implement=True,
    )
    supervisor = PortalImplementationSupervisor(config)
    command = (sys.executable, "-c", "print('native-controller-launch')")
    monkeypatch.setattr(supervisor, "_build_daemon_command", lambda: list(command))
    monkeypatch.setattr(supervisor, "_effective_managed_daemon_sharding", lambda: (4, lane, True))
    # The generic disposable owner has no production historical manifest pins.
    # All native Quack/owner/custody/launch work below remains real.
    monkeypatch.setattr(supervisor, "_retained_fenced_provider_program_applicable", lambda _: True)
    monkeypatch.setattr(supervisor, "_database_managed_daemon_identity_required", lambda: True)
    loop = SupervisorLoop(supervisor.build_supervisor_loop_config())
    assert loop.config.child_launcher == supervisor._launch_database_child_with_current_custody
    monkeypatch.setattr(supervisor, "_retained_fenced_provider_program_applicable", lambda _: False)
    launched = []
    real_launch = supervisor_runtime.launch_supervised_child
    def launch(spec):
        owner_lock = quack_owner_mutation_write_lock_path(program.store_id)
        with pytest.raises(TimeoutError):
            with exclusive_file_lock(owner_lock, timeout_seconds=0.02):
                pytest.fail("owner fence released before actual spawn")
        with pytest.raises(TimeoutError):
            with serialized_lock_update(supervisor._managed_daemon_launch_lock_path(), timeout_seconds=0.02):
                pytest.fail("launch fence released before actual spawn")
        assert retained._retained_embedded_writer_fence_current()
        assert local.capture(retained, attempt) == before
        launched.append(real_launch(spec))
        return launched[-1]
    monkeypatch.setattr(supervisor_runtime, "launch_supervised_child", launch)
    if admission_case == "diagnostic_only":
        monkeypatch.setattr(supervisor, "_reconcile_interrupted_database_portal_attempts_bound",
            lambda *_, **__: {"reason": "database_portal_attempts_reconciled", "blocked": False,
                "reconciled": False, "quiesced": False, "safe_to_restart": False,
                "completion_authorized": False, "independent_work_admitted": True,
                "owner_task_quarantines": [attempt.attempt_id]})
    elif admission_case == "changed_source":
        source = supervisor._control_plane_source_snapshot()
        monkeypatch.setattr(supervisor, "_control_plane_source_snapshot",
            lambda: {**source, "source_id": "changed:after-config"})
    elif admission_case == "unknown_child":
        monkeypatch.setattr(supervisor, "_terminate_managed_daemon_tree",
            lambda **_: {"quiesced": False, "safe_to_restart": False})
    outcome = loop.run()
    assert outcome.restart_count == 1
    if admission_case != "native":
        assert outcome.last_exit_code == 127
        assert outcome.last_recycle_reason == "launch_failed"
        assert launched == []
    else:
        assert outcome.last_exit_code == 0
        assert len(launched) == 1
        assert "native-controller-launch" in launched[0].log_path.read_text()
    assert local.capture(retained, attempt) == before


def test_source_probe_rejects_fifo_without_stalling_child(tmp_path):
    import multiprocessing
    import os
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as module
    os.mkfifo(tmp_path / "fifo.py")
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    def probe():
        module.__file__ = str(tmp_path / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py")
        module.CONTROL_PLANE_SOURCE_PATHS = ("fifo.py",)
        writer.send(module._read_control_plane_source_snapshot()["sources"])
    child = context.Process(target=probe)
    child.start()
    try:
        child.join(5)
        assert not child.is_alive(), "FIFO stalled the source gate before native admission"
        assert child.exitcode == 0
        assert reader.poll(1)
        assert reader.recv()[0]["available"] is False
    finally:
        if child.is_alive():
            child.kill()
            child.join(5)
        reader.close()
        writer.close()
