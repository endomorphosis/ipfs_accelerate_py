"""Database supervisor maintenance never treats the live owner file as Markdown."""
from __future__ import annotations

import hashlib
from pathlib import Path
import secrets
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    InProcessQuackTransport, _allocate_loopback_port,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import DatabaseProgramConfig
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as module


@pytest.fixture
def observed_owner(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "control.duckdb"
    owner = duckdb.connect(str(path), config={"autoinstall_known_extensions": False})
    try:
        owner.execute("LOAD quack")
    except Exception as exc:
        owner.close()
        pytest.skip(f"installed Quack unavailable: {type(exc).__name__}")
    owner.execute("CREATE TABLE tasks(task_alias VARCHAR, status VARCHAR)")
    owner.execute("INSERT INTO tasks VALUES ('AF-001', 'ready'), ('AF-002', 'blocked')")
    owner.execute("CREATE TABLE goals(goal_cid VARCHAR)")
    owner.execute("INSERT INTO goals VALUES ('goal-1')")
    owner.execute("CREATE TABLE domain_events(global_sequence BIGINT)")
    owner.execute("INSERT INTO domain_events VALUES (1), (2)")
    owner.execute("CREATE TABLE control_plane_metadata(key VARCHAR, value VARCHAR)")
    owner.execute("INSERT INTO control_plane_metadata VALUES ('database_uuid', 'test-database')")
    owner.execute("CREATE TABLE state_servers(store_id VARCHAR, database_uuid VARCHAR, schema_revision VARCHAR, generation BIGINT)")
    owner.execute("INSERT INTO state_servers VALUES ('test-store', 'test-database', '1', 1)")
    owner.execute("CHECKPOINT")
    token, port = secrets.token_urlsafe(32), _allocate_loopback_port()
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    identity = SimpleNamespace(server_id="observation-test", store_id="test-store",
        database_uuid="test-database", schema_revision=1, schema_fingerprint="test-schema",
        generation=1, process_birth_id="test-birth")
    transport = InProcessQuackTransport()
    try:
        transport.start(owner, host="127.0.0.1", port=port, token=token, identity=identity)
        program = DatabaseProgramConfig(authority_mode="quack", task_source_kind="duckdb",
            endpoint_secret_handle="handle:observation-test", quack_endpoint=f"quack:127.0.0.1:{port}",
            store_id="test-store", store_generation="1", schema_revision="1")
        args = module.parse_args(["--todo-path", str(path), "--state-dir", str(tmp_path / "state"),
            "--task-prefix", "AF-", "--state-prefix", "test-observer", "--implement",
            *program.cli_args()])
        config = module.supervisor_config_from_args(args, repo_root=tmp_path)
        # Legacy producer defaults deliberately remain enabled: authority must
        # protect the file even if callers forget campaign-specific flags.
        assert config.reconciliation_guardrail_enabled
        monkeypatch.setattr(module, "isolate_board_runtime", lambda **kwargs: pytest.fail("DB constructor tried board import"))
        supervisor = module.PortalImplementationSupervisor(config)
        yield SimpleNamespace(owner=owner, path=path, supervisor=supervisor)
    finally:
        try:
            transport.stop(owner)
        finally:
            owner.close()


def forbid_legacy(supervisor, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("database supervisor entered legacy maintenance")
    for name in ("_begin_supervisor_maintenance_heartbeat", "_acquire_implementation_maintenance_lease",
                 "ensure_todo_board_for_refill", "repair_main_checkout_merge_state",
                 "_build_worktree_reconciliation_daemon", "is_stuck", "ensure_state_file",
                 "ensure_strategy_file", "record_dependency_guardrails", "record_retry_budget_guardrails",
                 "reconcile_objective_task_janitor", "_run_once_with_maintenance_under_lease"):
        monkeypatch.setattr(supervisor, name, forbidden)


def test_real_remote_observation_preserves_binary_and_needs_no_legacy_json(observed_owner, monkeypatch):
    test = observed_owner
    supervisor = test.supervisor
    forbid_legacy(supervisor, monkeypatch)
    before = hashlib.sha256(test.path.read_bytes()).hexdigest(), test.path.stat().st_ino
    original_read, original_write, original_rename = Path.read_text, Path.write_text, Path.rename
    def guard_read(path, *args, **kwargs):
        assert path != test.path, "binary control file was read as text"
        return original_read(path, *args, **kwargs)
    def guard_write(path, *args, **kwargs):
        assert path != test.path, "binary control file was overwritten"
        return original_write(path, *args, **kwargs)
    def guard_rename(path, *args, **kwargs):
        assert path != test.path, "binary control file was renamed"
        return original_rename(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guard_read)
    monkeypatch.setattr(Path, "write_text", guard_write)
    monkeypatch.setattr(Path, "rename", guard_rename)
    result = supervisor.run_once()
    assert result["transport"] == "quack"
    assert result["task_status_counts"] == {"blocked": 1, "ready": 1}
    assert result["goal_count"] == 1 and result["event_watermark"] == 2
    assert result["legacy_maintenance_skipped"] is True
    assert result["completion_authority"] is False
    assert result["provider_progress_observed"] is False
    assert not supervisor.config.state_path.exists()
    assert not supervisor.config.strategy_path.exists()
    assert supervisor._run_once_with_maintenance(lambda phase: None)["task_count"] == 2
    assert supervisor._reconcile_interrupted_implementation_after_shutdown()["completion_authority"] is False
    after = hashlib.sha256(test.path.read_bytes()).hexdigest(), test.path.stat().st_ino
    assert before == after


def test_database_forever_preflight_and_watchdog_preserve_native_child(observed_owner, monkeypatch):
    supervisor = observed_owner.supervisor
    forbid_legacy(supervisor, monkeypatch)
    captured = []
    class NoLaunchLoop:
        def __init__(self, config, watchdog_hook):
            self.config = config
            captured.append(config)
        def run(self):
            return SimpleNamespace(status="stopped", restart_count=0, last_exit_code=0,
                last_recycle_reason="test-no-launch", last_run_id="test", last_log_path="")
    monkeypatch.setattr(supervisor, "shared_supervisor_loop_class", NoLaunchLoop)
    supervisor._run_forever_loop()
    config = captured[0]
    assert "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon" in config.command
    assert "--implement" in config.command and "--authority-mode" in config.command
    assert config.spec.status_path == supervisor._database_observation_path()
    assert config.spec.launch_env["IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT"] == supervisor.config.database_program.quack_endpoint
    supervisor._last_supervisor_maintenance_at = 0
    loop = SimpleNamespace(config=SimpleNamespace(status_extra_fields={}))
    decision = supervisor._supervisor_loop_watchdog_decision(loop, SimpleNamespace(pid=123), {})
    assert decision.action == "continue"
    assert loop.config.status_extra_fields["database_owner_available"] is True
    # An observed owner generation change fails closed without falling back to
    # the control file, creating a legacy board, or launching a replacement.
    observed_owner.owner.execute("UPDATE state_servers SET generation = 2")
    supervisor._last_supervisor_maintenance_at = 0
    decision = supervisor._supervisor_loop_watchdog_decision(loop, SimpleNamespace(pid=123), {})
    assert decision.action == "stop"
    assert decision.reason == "database_owner_unavailable"
    assert loop.config.status_extra_fields["database_owner_available"] is False
