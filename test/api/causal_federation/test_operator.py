"""Hermetic qualification for the narrow one-lane CASF bootstrap operator."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_causal_event_federation.py"
CONFIG = ROOT / "config/agent_supervisor_causal_event_federation_scheduler.json"


def _operator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("casf_bootstrap_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _authority(**updates):
    value = {
        "available": True,
        "task_count": 44,
        "ready_count": 1,
        "active_count": 0,
        "blocked_count": 0,
        "completed_count": 12,
        "terminal_count": 12,
        "event_cursor": 20,
    }
    value.update(updates)
    return value


def _runtime(**updates):
    value = {
        "supervisor_status": {"status": "running"},
        "task_state": {
            "task_count": 44,
            "completed_count": 12,
            "eligible_ready_count": 1,
            "blocked_count": 0,
            "external_reserved_count": 0,
            "active_task_id": "",
            "implementation_in_progress": False,
        },
        "supervisor_fresh": True,
        "task_state_fresh": True,
        "supervisor_after_launch": True,
        "task_state_after_launch": True,
        "outbox_worker": {
            "healthy": True,
            "available": True,
            "thread_alive": True,
            "server_owned": True,
            "polling": False,
            "watermark": 21,
            "committed_sequence": 21,
            "caught_up": True,
            "commit_observer_bound": True,
            "observer_error_type": "",
            "last_error_type": "",
        },
    }
    value.update(updates)
    return value


BOOTSTRAP_EVENT_ID = "event:casf-bootstrap"
BOOTSTRAP_ACKNOWLEDGEMENT_ID = "ack:casf-bootstrap"
BOOTSTRAP_DELIVERY_ATTEMPT_ID = "delivery-attempt:casf-bootstrap"


def _first_tranche_authority(**runtime_health_updates):
    runtime_health = {
        "available": True,
        "lifecycle_state": "IDLE",
        "current_runtime_lease": True,
        "process_bound": True,
        "bootstrap_event_acknowledged": True,
        "consumer_cursor_advanced": True,
        "pending_required_deliveries": 0,
        "acknowledged_event_id": BOOTSTRAP_EVENT_ID,
        "acknowledgement_id": BOOTSTRAP_ACKNOWLEDGEMENT_ID,
        "delivery_attempt_id": BOOTSTRAP_DELIVERY_ATTEMPT_ID,
    }
    runtime_health.update(runtime_health_updates)
    return _authority(runtime_health=runtime_health)


def _first_tranche_runtime(*, runtime_updates=None, **supervisor_updates):
    supervisor = {
        "status": "idle",
        "execution_scope": "first_tranche_event_coordination_only",
        "task_execution_admitted": False,
        "server_owned_event_wait": True,
        "event_wait_transport": "typed_state_owner_bounded_long_wait",
        "event_wait_qualified": True,
        "event_wait_adaptive_polling": False,
        "first_event_id": BOOTSTRAP_EVENT_ID,
        "first_acknowledgement_id": BOOTSTRAP_ACKNOWLEDGEMENT_ID,
        "first_delivery_attempt_id": BOOTSTRAP_DELIVERY_ATTEMPT_ID,
    }
    supervisor.update(supervisor_updates)
    runtime = _runtime(
        supervisor_status=supervisor,
        supervisor_process_bound=True,
    )
    runtime.update(runtime_updates or {})
    return runtime


def test_native_launch_plan_admits_only_one_event_wait_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    raw_token = "raw-token-material-for-test"
    monkeypatch.setenv(operator.STATE_TOKEN_ENV, raw_token)

    plan = operator._launch_plan(board, stamp="20000101T000000Z")

    assert plan["lanes"] == 1
    assert plan["admitted_lanes"] == 1
    assert plan["runtime"] == "CASFEventSupervisorRuntime@1"
    assert plan["registered_logical_subagents"] == 1
    assert plan["maximum_active_subagents"] == 0
    assert plan["strict_task_sharding"] is True
    assert plan["work_stealing"] is False
    assert plan["credential_transport"] == "private_inherited_pipe"
    assert plan["credential_in_argv"] is False
    assert plan["credential_in_environment"] is False
    assert plan["state_transport"] == "typed_quack_state_owner"
    assert plan["server_owned_event_wait"] is True
    assert plan["event_wait_qualified"] is True
    assert plan["task_execution_admitted"] is False
    assert plan["execution_scope"] == "first_tranche_event_coordination_only"
    assert plan["provider_route_preflight"]["provider_execution_admitted"] is False
    assert plan["event_driven_federation_qualified"] is False
    assert plan["multi_supervisor_qualified"] is False
    assert plan["parallel_execution_qualified"] is False
    assert plan["high_concurrency_qualified"] is False
    assert raw_token not in json.dumps(plan)
    assert operator.STATE_TOKEN_ENV not in json.dumps(plan)
    assert "argv" not in plan
    assert "environment" not in plan


def test_population_uses_production_parser_and_current_configured_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    board, config = operator._load_config(CONFIG)
    monkeypatch.setattr(
        operator,
        "_assert_clean_current_tree",
        lambda _config: ("a" * 40, "b" * 40),
    )
    monkeypatch.setattr(
        operator,
        "_tracked_bytes",
        lambda path, head: Path(path).read_bytes(),
    )

    population = operator._population(board, config)

    assert population["program_id"] == operator.PROGRAM_ID
    assert len(population["objectives"]) == 17
    assert len(population["tasks"]) == 44
    assert [item["task_id"] for item in population["tasks"]] == [
        f"CASF-{index:03d}" for index in range(44)
    ]
    expected_completed = config["initial_projection"]["completed_task_ids"]
    assert [
        item["task_id"]
        for item in population["tasks"]
        if item["status"] in operator.COMPLETED_STATUSES
    ] == expected_completed
    assert sum(len(item["dependencies"]) for item in population["tasks"]) == 191


def test_population_materializes_through_canonical_migrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_causal_event_federation_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    operator = _operator()
    board, config = operator._load_config(CONFIG)
    monkeypatch.setattr(
        operator,
        "_assert_clean_current_tree",
        lambda _config: ("c" * 40, "d" * 40),
    )
    monkeypatch.setattr(
        operator,
        "_tracked_bytes",
        lambda path, head: Path(path).read_bytes(),
    )
    population = operator._population(board, config)
    database = tmp_path / "control.duckdb"

    with DatabaseTaskSource(
        database,
        owner_id="casf-operator-hermetic-materializer",
        repository_tree_id=population["repository_tree_id"],
        plan_root_cid=population["plan_root_cid"],
    ) as source:
        source.materialize(population)
        snapshot = source.snapshot().to_dict()
        ready = [item.task_alias for item in source.ready_tasks(limit=100).tasks]

    schema = verify_causal_event_federation_schema(database)
    assert snapshot["task_count"] == 44
    assert snapshot["goal_count"] == 17
    assert snapshot["dependency_count"] == 191
    assert ready == config["initial_projection"]["ready_task_ids"]
    assert schema["schema_revision"] == 2
    assert schema["migration_id"] == "0002_causal_event_federation_core"


def test_port_collision_fails_closed_without_touching_listener() -> None:
    operator = _operator()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()

        assert operator._port_is_free(host, port) is False
        with pytest.raises(operator.OperatorError, match="occupied"):
            operator._require_free_port(host, port)

        # The existing listener remains owned by this test and usable.
        assert listener.fileno() >= 0


def test_state_owner_socket_is_short_server_derived_and_private(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )

    operator = _operator()
    board, _config = operator._load_config(CONFIG)

    socket_path = operator._runtime_paths(board)["owner_socket"]

    assert socket_path.is_absolute()
    assert len(os.fsencode(socket_path)) <= operator.UNIX_SOCKET_PATH_CEILING
    assert socket_path.parent.name == f"ipfs-accelerate-casf-{os.geteuid()}"
    assert socket_path.name.startswith("owner-")
    assert str(operator.ROOT) not in str(socket_path)

    private_socket = tmp_path / "private" / "owner.sock"
    operator._prepare_private_socket_parent(private_socket)
    metadata = os.lstat(private_socket.parent)
    assert stat.S_ISDIR(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert metadata.st_uid == os.geteuid()
    server = build_server(
        database_path=tmp_path / "control.duckdb",
        state_dir=tmp_path / "deliberately-long-owner-state-directory",
        typed_command_socket_path=private_socket,
    )
    assert server.typed_command_socket_path() == private_socket

    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir(mode=0o755)
    unsafe_parent.chmod(0o755)
    with pytest.raises(operator.OperatorError, match="custody is unsafe"):
        operator._prepare_private_socket_parent(unsafe_parent / "owner.sock")

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(operator.OperatorError, match="custody is unsafe"):
        operator._prepare_private_socket_parent(linked_parent / "owner.sock")


def test_receipts_are_private_content_addressed_and_tamper_evident(
    tmp_path: Path,
) -> None:
    operator = _operator()
    paths = {
        "operator_evidence": tmp_path / "evidence",
        "status_receipt": tmp_path / "status-current.json",
    }
    receipt = operator._persist_receipt(
        paths,
        "status",
        {"schema": operator.STATUS_SCHEMA, "classification": "progressing"},
    )

    operator._verify_receipt(receipt, kind="status")
    assert stat.S_IMODE(paths["status_receipt"].stat().st_mode) == 0o600
    immutable = list((tmp_path / "evidence" / "receipts").glob("status-*.json"))
    assert len(immutable) == 1
    tampered = dict(receipt)
    tampered["classification"] = "safely_idle"
    with pytest.raises(operator.OperatorError, match="invalid"):
        operator._verify_receipt(tampered, kind="status")


def test_owner_argv_and_environment_exclude_raw_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setenv(operator.STATE_TOKEN_ENV, "raw-state-token-test")
    monkeypatch.setenv("SOME_PASSWORD", "raw-password-test")

    argv = operator._operator_command(CONFIG, "state-owner")
    supervisor_argv = operator._supervisor_runtime_command(CONFIG, 7)
    environment = operator._state_owner_environment()

    assert "raw-state-token-test" not in " ".join(argv)
    assert "raw-state-token-test" not in " ".join(supervisor_argv)
    assert "raw-password-test" not in json.dumps(environment)
    assert operator.STATE_TOKEN_ENV not in environment
    assert "SOME_PASSWORD" not in environment
    assert supervisor_argv[-2:] == ["--credential-fd", "7"]
    assert "supervisor-runtime" in supervisor_argv


def test_route_preflight_uses_complete_configured_tuple_without_self_authority() -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)

    route = operator._route_preflight(board)

    assert route["primary_provider_id"] == "grok_cli"
    assert route["primary_model_id"] == "grok-4.6"
    assert route["fallback_provider_id"] == "codex"
    assert route["fallback_model_id"] == "gpt-5.6-terra"
    assert route["fallback_trigger"] == "primary_quota_exhausted"
    assert route["fallback_reasoning_effort"] == "high"
    assert route["authorization_required"] is False
    assert route["authorization_present"] is False
    assert route["canonical_route_resolver_passed"] is True
    assert route["operator_created_authority"] is False
    assert route["provider_execution_admitted"] is False


def test_native_launch_plan_does_not_promote_inherited_route_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    plan = operator._launch_plan(board, stamp="20000101T000000Z")
    foreign = {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_PATH": "foreign.json",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_SHA256": "sha256:" + "f" * 64,
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_ID": "sha256:" + "e" * 64,
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_KIND": "foreign",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE": "foreign-board",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD": "a" * 40,
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_TREE": "b" * 40,
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_ID": "foreign-route",
    }
    for name, value in foreign.items():
        monkeypatch.setenv(name, value)
    rendered = json.dumps(plan, sort_keys=True)
    assert plan["provider_route_preflight"]["authorization_present"] is False
    assert plan["provider_route_preflight"]["provider_execution_admitted"] is False
    assert plan["task_execution_admitted"] is False
    assert "foreign-route" not in rendered
    assert "foreign-board" not in rendered
    for name, value in foreign.items():
        assert os.environ[name] == value


def test_native_runtime_rejects_non_private_credential_descriptor() -> None:
    operator = _operator()

    with pytest.raises(operator.OperatorError, match="credential descriptor"):
        operator._supervisor_runtime_command(CONFIG, 2)


def test_health_does_not_promote_active_rows_without_an_execution_profile() -> None:
    operator = _operator()
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_authority(active_count=1),
        runtime=_runtime(),
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )
    assert result["classification"] == "progress_unqualified"
    assert result["healthy"] is False
    assert result["blocked_or_stuck"] is True
    assert "active_task_or_attempt_observed" in result["progress_evidence"]


def test_health_does_not_promote_ready_rows_without_an_execution_profile() -> None:
    operator = _operator()
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_authority(),
        runtime=_runtime(),
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )
    assert result["classification"] == "progress_unqualified"
    assert result["healthy"] is False
    assert result["blocked_or_stuck"] is True
    assert "fresh_supervisor_cycle_observed_ready_work" in result["progress_evidence"]


def test_health_separates_coordinator_readiness_from_unadmitted_ready_work() -> None:
    operator = _operator()
    runtime = _first_tranche_runtime()
    assert runtime["supervisor_status"]["event_wait_transport"] == (
        "typed_state_owner_bounded_long_wait"
    )
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_first_tranche_authority(),
        runtime=runtime,
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )

    assert result["classification"] == "coordinator_ready"
    assert result["healthy"] is False
    assert result["plan_work_healthy"] is False
    assert result["plan_work_blocked"] is True
    assert result["plan_execution_status"] == "unadmitted"
    assert result["coordinator_ready"] is True
    assert result["coordinator_transport_healthy"] is True
    assert result["coordinator_blocked_or_stuck"] is False
    assert result["blocked_or_stuck"] is True
    assert result["reason_codes"] == [
        "ready_work_present_but_task_execution_unadmitted"
    ]
    assert result["progress_evidence"] == []
    assert result["safe_idle_evidence"] == []
    assert result["coordinator_evidence"] == [
        "exact_process_birth_and_current_runtime_lease",
        "bootstrap_event_durably_acknowledged",
        "state_owner_outbox_worker_live",
        "typed_server_owned_event_wait_qualified",
    ]
    assert (
        operator._launch_success_mode(result, allow_coordinator_only=False) == ""
    )
    assert operator._launch_success_mode(
        result, allow_coordinator_only=True
    ) == "coordinator_transport_only"


def test_cli_keeps_plan_health_and_coordinator_transport_requirements_distinct() -> None:
    operator = _operator()
    launch = operator._parser().parse_args(
        ["launch", "--allow-coordinator-only"]
    )
    assert launch.allow_coordinator_only is True
    status = operator._parser().parse_args(
        ["status", "--require-coordinator-ready"]
    )
    assert status.require_coordinator_ready is True
    assert status.require_healthy is False
    with pytest.raises(SystemExit):
        operator._parser().parse_args(
            ["status", "--require-healthy", "--require-coordinator-ready"]
        )


@pytest.mark.parametrize(
    "supervisor_update",
    [
        {"task_execution_admitted": True},
        {"server_owned_event_wait": False},
        {"event_wait_qualified": False},
        {"event_wait_adaptive_polling": True},
    ],
)
def test_health_does_not_call_unqualified_first_tranche_safely_idle(
    supervisor_update,
) -> None:
    operator = _operator()
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_first_tranche_authority(),
        runtime=_first_tranche_runtime(**supervisor_update),
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )

    assert result["classification"] != "safely_idle"
    assert result.get("coordinator_ready") is not True


@pytest.mark.parametrize(
    ("runtime_health_update", "runtime_update", "reason"),
    [
        (
            {"bootstrap_event_acknowledged": False},
            {},
            "authoritative_bootstrap_event_or_runtime_evidence_incomplete",
        ),
        (
            {},
            {"supervisor_fresh": False},
            "supervisor_heartbeat_missing_or_stale",
        ),
        (
            {},
            {"supervisor_process_bound": False},
            "authoritative_bootstrap_event_or_runtime_evidence_incomplete",
        ),
        (
            {"process_bound": False},
            {},
            "authoritative_bootstrap_event_or_runtime_evidence_incomplete",
        ),
    ],
    ids=(
        "missing-bootstrap-ack",
        "stale-or-missed-heartbeat",
        "observed-process-binding-mismatch",
        "authoritative-process-binding-mismatch",
    ),
)
def test_health_rejects_incomplete_authoritative_first_tranche_evidence(
    runtime_health_update, runtime_update, reason
) -> None:
    operator = _operator()
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_first_tranche_authority(**runtime_health_update),
        runtime=_first_tranche_runtime(runtime_updates=runtime_update),
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )

    assert result["classification"] == "stuck"
    assert result["healthy"] is False
    assert result["blocked_or_stuck"] is True
    assert reason in result["reason_codes"]
    assert result["safe_idle_evidence"] == []


def test_terminal_quiescence_requires_live_process_bound_runtime_authority() -> None:
    operator = _operator()
    terminal_state = _first_tranche_runtime(
        runtime_updates={
            "task_state": {
                "task_count": 44,
                "completed_count": 44,
                "eligible_ready_count": 0,
                "blocked_count": 0,
                "external_reserved_count": 0,
                "active_task_id": "",
                "implementation_in_progress": False,
            }
        }
    )
    terminal_authority = _first_tranche_authority()
    terminal_authority.update(
        {
            "ready_count": 0,
            "active_count": 0,
            "completed_count": 44,
            "terminal_count": 44,
        }
    )
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=terminal_authority,
        runtime=terminal_state,
        baseline={"event_cursor": 20, "completed_count": 43},
        within_startup_grace=False,
    )
    assert result["classification"] == "completion_unqualified"
    assert result["healthy"] is False
    assert result["plan_work_healthy"] is False
    assert result["plan_work_blocked"] is True
    assert result["coordinator_transport_healthy"] is True
    assert result["safe_idle_evidence"] == []
    assert "exact_process_birth_and_current_runtime_lease" in result[
        "coordinator_evidence"
    ]

    dead = operator.classify_health(
        owner_liveness="alive",
        master_liveness="dead",
        task_authority=terminal_authority,
        runtime=terminal_state,
        baseline={"event_cursor": 20, "completed_count": 43},
        within_startup_grace=False,
    )
    assert dead["classification"] == "stuck"
    assert dead["healthy"] is False

    missing_runtime = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_authority(
            ready_count=0,
            active_count=0,
            completed_count=44,
            terminal_count=44,
        ),
        runtime=terminal_state,
        baseline={"event_cursor": 20, "completed_count": 43},
        within_startup_grace=False,
    )
    assert missing_runtime["classification"] == "stuck"
    assert missing_runtime["healthy"] is False

    stale = dict(terminal_state)
    stale["task_state_after_launch"] = False
    rejected = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=terminal_authority,
        runtime=stale,
        baseline={},
        within_startup_grace=False,
    )
    assert rejected["classification"] == "stuck"


@pytest.mark.parametrize(
    ("supervisor_updates", "runtime_health_updates"),
    [
        ({"execution_scope": "foreign-runtime"}, {}),
        ({"task_execution_admitted": True}, {}),
        (
            {
                "server_owned_event_wait": False,
                "event_wait_qualified": False,
                "event_wait_adaptive_polling": True,
            },
            {},
        ),
        ({}, {"lifecycle_state": "ACTIVE"}),
    ],
    ids=("foreign-scope", "task-execution", "polling-wait", "non-idle-authority"),
)
def test_terminal_rows_cannot_promote_unqualified_coordinator_transport(
    supervisor_updates,
    runtime_health_updates,
) -> None:
    operator = _operator()
    terminal_runtime = _first_tranche_runtime(
        runtime_updates={
            "task_state": {
                "task_count": 44,
                "completed_count": 44,
                "eligible_ready_count": 0,
                "blocked_count": 0,
                "external_reserved_count": 0,
                "active_task_id": "",
                "implementation_in_progress": False,
            }
        },
        **supervisor_updates,
    )
    terminal_authority = _first_tranche_authority(**runtime_health_updates)
    terminal_authority.update(
        {
            "ready_count": 0,
            "active_count": 0,
            "completed_count": 44,
            "terminal_count": 44,
        }
    )

    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=terminal_authority,
        runtime=terminal_runtime,
        baseline={"event_cursor": 20, "completed_count": 44},
        within_startup_grace=False,
    )

    assert result.get("coordinator_ready") is not True
    assert result.get("coordinator_transport_healthy") is not True
    assert operator._launch_success_mode(
        result, allow_coordinator_only=True
    ) == ""


def test_terminal_quiescence_rejects_later_pending_delivery_or_outbox_lag() -> None:
    operator = _operator()
    terminal_state = _first_tranche_runtime(
        runtime_updates={
            "task_state": {
                "task_count": 44,
                "completed_count": 44,
                "eligible_ready_count": 0,
                "blocked_count": 0,
                "external_reserved_count": 0,
                "active_task_id": "",
                "implementation_in_progress": False,
            }
        }
    )
    terminal_authority = _first_tranche_authority()
    terminal_authority.update(
        {
            "ready_count": 0,
            "active_count": 0,
            "completed_count": 44,
            "terminal_count": 44,
        }
    )
    pending = dict(terminal_authority)
    pending["runtime_health"] = dict(terminal_authority["runtime_health"])
    pending["runtime_health"]["pending_required_deliveries"] = 1
    pending_result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=pending,
        runtime=terminal_state,
        baseline={},
        within_startup_grace=False,
    )
    assert pending_result["classification"] != "safely_idle"
    assert pending_result["healthy"] is False

    lagging_worker = operator._outbox_worker_health(
        {
            "outbox_worker": {
                "available": True,
                "thread_alive": True,
                "server_owned": True,
                "polling": False,
                "watermark": 40,
                "committed_sequence": 41,
                "drain_count": 2,
                "last_error_type": "",
            },
            "typed_command_gateway": {
                "commit_observer_bound": True,
                "last_observer_error_type": "",
            },
        }
    )
    assert lagging_worker["caught_up"] is False
    assert lagging_worker["healthy"] is False
    lagging_state = dict(terminal_state)
    lagging_state["outbox_worker"] = lagging_worker
    lagging_result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=terminal_authority,
        runtime=lagging_state,
        baseline={},
        within_startup_grace=False,
    )
    assert lagging_result["classification"] == "stuck"
    assert lagging_result["healthy"] is False


def test_health_fails_closed_when_owner_outbox_worker_exits() -> None:
    operator = _operator()
    failed_worker = {
        "healthy": False,
        "available": False,
        "thread_alive": False,
        "server_owned": True,
        "polling": False,
        "last_error_type": "RuntimeError",
    }
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_first_tranche_authority(),
        runtime=_first_tranche_runtime(
            runtime_updates={"outbox_worker": failed_worker}
        ),
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )
    assert result == {
        "classification": "stuck",
        "healthy": False,
        "blocked_or_stuck": True,
        "reason_codes": ["state_owner_outbox_worker_unavailable"],
        "progress_evidence": [],
        "safe_idle_evidence": [],
    }


def test_outbox_worker_health_rejects_missing_malformed_or_polling_status() -> None:
    operator = _operator()
    assert operator._outbox_worker_health({})["healthy"] is False
    malformed = operator._outbox_worker_health(
        {
            "outbox_worker": {
                "available": True,
                "thread_alive": True,
                "server_owned": True,
                "polling": False,
                "watermark": "not-an-integer",
                "last_error_type": "",
            },
            "typed_command_gateway": {
                "commit_observer_bound": True,
                "last_observer_error_type": "",
            },
        }
    )
    assert malformed["malformed"] is True
    assert malformed["healthy"] is False
    polling = operator._outbox_worker_health(
        {
            "outbox_worker": {
                "available": True,
                "thread_alive": True,
                "server_owned": True,
                "polling": True,
                "last_error_type": "",
            }
        }
    )
    assert polling["healthy"] is False

    suppressed_observer_failure = operator._outbox_worker_health(
        {
            "outbox_worker": {
                "available": True,
                "thread_alive": True,
                "server_owned": True,
                "polling": False,
                "watermark": 21,
                "committed_sequence": 21,
                "last_error_type": "",
            },
            "typed_command_gateway": {
                "commit_observer_bound": True,
                "last_observer_error_type": "RuntimeError",
            },
        }
    )
    assert suppressed_observer_failure["caught_up"] is True
    assert suppressed_observer_failure["observer_error_type"] == "RuntimeError"
    assert suppressed_observer_failure["healthy"] is False


def test_state_owner_outbox_health_uses_full_canonical_status() -> None:
    operator = _operator()

    class Server:
        status_calls = 0

        def status(self):
            self.status_calls += 1
            return {
                "outbox_worker": {
                    "available": True,
                    "thread_alive": True,
                    "server_owned": True,
                    "polling": False,
                    "watermark": 99,
                    "committed_sequence": 99,
                    "drain_count": 1,
                    "last_error_type": "",
                },
                "typed_command_gateway": {
                    "commit_observer_bound": True,
                    "last_observer_error_type": "",
                },
            }

        def outbox_worker_capability(self):
            raise AssertionError("worker-only status is not an admission witness")

    server = Server()
    health = operator._state_owner_outbox_health(server)

    assert health["healthy"] is True
    assert health["caught_up"] is True
    assert health["commit_observer_bound"] is True
    assert server.status_calls == 1


@pytest.mark.parametrize(
    ("authority", "runtime", "master", "expected"),
    [
        (_authority(blocked_count=1), _runtime(), "alive", "blocked"),
        (_authority(), _runtime(supervisor_fresh=False), "alive", "stuck"),
        (_authority(), _runtime(), "dead", "stuck"),
    ],
)
def test_health_fails_closed_for_blocked_stale_or_dead_runner(
    authority, runtime, master, expected
) -> None:
    operator = _operator()
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness=master,
        task_authority=authority,
        runtime=runtime,
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )
    assert result["classification"] == expected
    assert result["healthy"] is False
    assert result["blocked_or_stuck"] is True


def test_health_distinguishes_bounded_startup_from_stuck() -> None:
    operator = _operator()
    runtime = _runtime(supervisor_status={}, supervisor_fresh=False)
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_authority(),
        runtime=runtime,
        baseline={},
        within_startup_grace=True,
    )
    assert result["classification"] == "starting"
    assert result["healthy"] is False
    assert result["blocked_or_stuck"] is False


def test_launch_generation_is_single_use_until_recovery_is_implemented(
    tmp_path: Path,
) -> None:
    operator = _operator()
    paths = {
        "operator_evidence": tmp_path / "evidence",
        "launch_receipt": tmp_path / "launch.json",
        "stop_receipt": tmp_path / "stop.json",
    }
    operator._require_unused_launch_generation(paths)

    launch = operator._persist_receipt(
        paths,
        "launch",
        {
            "schema": operator.LAUNCH_SCHEMA,
            "program_id": operator.PROGRAM_ID,
            "marker": "test",
        },
    )
    with pytest.raises(operator.OperatorError, match="no complete matching stop"):
        operator._require_unused_launch_generation(paths)

    operator._persist_receipt(
        paths,
        "stop",
        {
            "schema": operator.STOP_SCHEMA,
            "program_id": operator.PROGRAM_ID,
            "complete": True,
            "launch_receipt_id": launch["launch_receipt_id"],
        },
    )
    with pytest.raises(operator.OperatorError, match="CASF-029"):
        operator._require_unused_launch_generation(paths)

    tampered = json.loads(paths["stop_receipt"].read_text(encoding="utf-8"))
    tampered["complete"] = False
    operator._atomic_json(paths["stop_receipt"], tampered)
    with pytest.raises(operator.OperatorError, match="identity is absent or invalid"):
        operator._require_unused_launch_generation(paths)


def test_exact_process_birth_stop_leaves_no_process_running() -> None:
    operator = _operator()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        birth = operator._process_birth(process.pid)
        assert operator._birth_liveness(birth) == "alive"
        result = operator._terminate_birth(birth, grace_seconds=2.0)
        assert result in {"terminated", "killed"}
        assert operator._birth_liveness(birth) == "dead"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_stop_ignores_stale_pid_observations_and_signals_only_sealed_births(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    paths = {
        "operator_evidence": tmp_path / "evidence",
        "launch_receipt": tmp_path / "launch.json",
        "stop_receipt": tmp_path / "stop.json",
        "owner_status": tmp_path / "owner-status.json",
        "owner": tmp_path / "owner",
        "master_pid": tmp_path / "master.pid",
        "supervisor_pid": tmp_path / "supervisor.pid",
        "daemon_pid": tmp_path / "daemon.pid",
        "supervisor_status": tmp_path / "supervisor-status.json",
    }
    master_birth = {"pid": 777_001, "start_time_ticks": 11, "boot_id": "sealed"}
    owner_birth = {"pid": 777_002, "start_time_ticks": 12, "boot_id": "sealed"}
    operator._persist_receipt(
        paths,
        "launch",
        {
            "schema": operator.LAUNCH_SCHEMA,
            "program_id": operator.PROGRAM_ID,
            "master_process_birth": master_birth,
            "supervisor_process_birth": master_birth,
            "owner_identity": {"process_birth": owner_birth},
        },
    )

    # These observations deliberately name this live pytest process.  They are
    # not launch authority and must never be recaptured as a fresh birth.
    stale_pid = os.getpid()
    for key in ("master_pid", "supervisor_pid", "daemon_pid"):
        paths[key].write_text(f"{stale_pid}\n", encoding="utf-8")
    paths["supervisor_status"].write_text(
        json.dumps(
            {
                "supervisor_pid": stale_pid,
                "daemon_pid": stale_pid,
                "active_worker_pids": [stale_pid],
            }
        ),
        encoding="utf-8",
    )

    signaled: list[dict] = []
    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, {}))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(operator, "_owner_liveness", lambda _status: "dead")
    monkeypatch.setattr(operator, "_birth_liveness", lambda _birth: "dead")
    monkeypatch.setattr(operator, "_port_is_free", lambda _host, _port: True)
    monkeypatch.setattr(
        operator,
        "_process_birth",
        lambda _pid: (_ for _ in ()).throw(AssertionError("PID recaptured")),
    )

    def terminate(birth, *, grace_seconds):
        del grace_seconds
        signaled.append(dict(birth))
        return "terminated"

    monkeypatch.setattr(operator, "_terminate_birth", terminate)

    receipt = operator.stop(CONFIG)

    assert receipt["complete"] is True
    assert signaled == [master_birth, owner_birth]
    assert all(item["birth"]["pid"] != stale_pid for item in receipt["process_results"])


def test_stale_owner_identity_is_rejected_before_any_stop() -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        StateServerIdentity,
    )

    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    identity = StateServerIdentity(
        server_id="server:test",
        store_id="foreign/control.duckdb",
        database_uuid="db-test",
        schema_revision=2,
        schema_fingerprint="sha256:" + "0" * 64,
        generation=1,
        fence_epoch=1,
        revision=1,
        process_birth=current_process_birth(),
        listen_uri="quack:127.0.0.1:41417",
        extension_fingerprint="sha256:" + "1" * 64,
        credential_generation=1,
        secret_handle="handle:casf-v1",
        repository_id="repository:ipfs_accelerate_py",
        startup_epoch=1,
        started_at="2000-01-01T00:00:00+00:00",
        status="ready",
    )

    with pytest.raises(operator.OperatorError, match="store identity"):
        operator._owner_identity(
            board,
            {"lifecycle": "ready", "identity": identity.to_dict()},
            expected_pid=os.getpid(),
        )
