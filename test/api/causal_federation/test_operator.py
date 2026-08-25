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
from types import ModuleType, SimpleNamespace

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


def _execution_route_summary(
    *,
    deterministic_task_count: int = 44,
    model_task_count: int = 0,
    policy_id: str = "route-policy:test",
) -> dict:
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        TASK_EXECUTION_ROUTE_SUMMARY_SCHEMA,
    )

    return {
        "schema": TASK_EXECUTION_ROUTE_SUMMARY_SCHEMA,
        "policy_id": policy_id,
        "plan_root_cid": "plan:test",
        "repository_tree_id": "tree:test",
        "source_revision": 7,
        "task_count": 44,
        "deterministic_task_count": deterministic_task_count,
        "model_task_count": model_task_count,
    }


def _install_admitted_generation(
    operator: ModuleType,
    paths: dict[str, Path],
    *,
    launch_route: dict,
    current_route: dict | None = None,
) -> dict[str, dict]:
    births = {
        "master": {
            "pid": 777_101,
            "start_time_ticks": 11,
            "boot_id": "boot:test",
            "parent_pid": 1,
        },
        "owner": {
            "pid": 777_102,
            "start_time_ticks": 12,
            "boot_id": "boot:test",
            "parent_pid": 1,
        },
        "executor_supervisor": {
            "pid": 777_103,
            "start_time_ticks": 13,
            "boot_id": "boot:test",
            "parent_pid": 777_102,
        },
        "executor": {
            "pid": 777_104,
            "start_time_ticks": 14,
            "boot_id": "boot:test",
            "parent_pid": 777_103,
        },
    }
    operator._persist_receipt(
        paths,
        "launch",
        {
            "schema": operator.LAUNCH_SCHEMA,
            "program_id": operator.PROGRAM_ID,
            "launched_at_ns": 1,
            "source_head": "source:test",
            "repository_tree_id": "tree:test",
            "master_process_birth": births["master"],
            "supervisor_process_birth": births["master"],
            "owner_identity": {"process_birth": births["owner"]},
            "executor_supervisor_process_birth": births["executor_supervisor"],
            "executor_process_birth_at_launch": births["executor"],
            "task_execution_admitted": True,
            "execution_route_policy": launch_route,
        },
    )
    sealed_current_route = current_route or launch_route
    operator._atomic_json(
        paths["executor_current"],
        {
            "supervisor_process_birth": births["executor_supervisor"],
            "executor_process_birth": births["executor"],
            "execution_route_policy": sealed_current_route,
            "execution_route_policy_id": sealed_current_route["policy_id"],
            "execution_route_plan_root_cid": sealed_current_route["plan_root_cid"],
            "execution_route_source_revision": sealed_current_route[
                "source_revision"
            ],
        },
    )
    return births


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


class _QuiescenceClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.value += float(seconds)


class _GenerationObservation:
    def __init__(self, content_id: str) -> None:
        self.content_id = content_id


class _QuiescenceClient:
    def __init__(self, *, generations: list[str], rows: list[list[dict]]) -> None:
        self._generations = list(generations)
        self._rows = list(rows)
        self.generation_calls = 0
        self.routing_calls: list[dict] = []

    @staticmethod
    def _next(values: list):
        assert values
        return values.pop(0) if len(values) > 1 else values[0]

    def load_generation(self) -> _GenerationObservation:
        self.generation_calls += 1
        return _GenerationObservation(self._next(self._generations))

    def execute(self, operation: str, parameters: dict) -> list[dict]:
        assert operation == "casf_select_subscription_routing_state"
        assert parameters["tenant_id"] == "tenant:test"
        assert parameters["federation_id"] == "federation:test"
        assert parameters["subscription_id"] == "subscription:test"
        assert parameters["observed_at"]
        self.routing_calls.append(dict(parameters))
        return self._next(self._rows)


class _RouteProjectionFactory:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.closes = 0

    def __call__(self, _client, *, owns_client: bool):
        assert owns_client is False
        factory = self
        self.calls += 1

        class _Projection:
            def seal_execution_route_policy(self, modes: dict[str, str]):
                assert len(modes) == 44
                outcome = _QuiescenceClient._next(factory._outcomes)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            def close(self) -> None:
                factory.closes += 1

        return _Projection()


def _quiescence_admission() -> SimpleNamespace:
    return SimpleNamespace(
        federation_identity=SimpleNamespace(
            record_id="federation:test",
            binding=SimpleNamespace(tenant_id="tenant:test"),
        ),
        subscription=SimpleNamespace(
            subscription_id="subscription:test",
            tenant_id="tenant:test",
            federation_id="federation:test",
            revision=3,
            maximum_pending=64,
            maximum_fanout=16,
        ),
    )


def _routing_row(*, pending_deliveries: int) -> dict:
    return {
        "subscription_id": "subscription:test",
        "revision": 3,
        "maximum_pending": 64,
        "maximum_fanout": 16,
        "pending_deliveries": pending_deliveries,
    }


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


def test_admitted_launch_route_remains_current_44_to_0_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    monkeypatch.setenv(operator.STATE_TOKEN_ENV, "raw-token-material-for-test")

    plan = operator._launch_plan(board, admit_task_execution=True)
    modes = operator._casf_mixed_execution_modes()
    current = _execution_route_summary()
    preceding = _execution_route_summary(
        deterministic_task_count=43,
        model_task_count=1,
    )
    historical = _execution_route_summary(
        deterministic_task_count=41,
        model_task_count=3,
    )

    assert plan["execution_route_expected_counts"] == {
        "task_count": 44,
        "deterministic_task_count": 44,
        "model_task_count": 0,
    }
    assert len(operator.CASF_DETERMINISTIC_TASK_ALIASES) == 44
    assert sum(mode == "deterministic-only" for mode in modes.values()) == 44
    assert sum(mode != "deterministic-only" for mode in modes.values()) == 0
    assert (
        operator._validated_execution_route_summary(
            current,
            require_casf_population=True,
        )
        == current
    )
    for obsolete in (preceding, historical):
        with pytest.raises(operator.OperatorError, match="exact CASF population"):
            operator._validated_execution_route_summary(
                obsolete,
                require_casf_population=True,
            )


def test_scheduler_schema_revision_must_match_canonical_migration_head() -> None:
    operator = _operator()

    assert operator._require_canonical_schema_revision("3") == 3
    with pytest.raises(
        operator.OperatorError,
        match=r"canonical migration head \(configured=2, latest=3\)",
    ):
        operator._require_canonical_schema_revision("2")


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


def test_executor_state_directory_is_private_and_fails_closed(
    tmp_path: Path,
) -> None:
    operator = _operator()
    private_state = tmp_path / "runtime" / "state" / "executor"
    operator._prepare_private_executor_state(private_state)
    metadata = os.lstat(private_state)
    assert stat.S_ISDIR(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert metadata.st_uid == os.geteuid()

    unsafe_state = tmp_path / "unsafe-executor"
    unsafe_state.mkdir(mode=0o770)
    unsafe_state.chmod(0o770)
    with pytest.raises(operator.OperatorError, match="custody is unsafe"):
        operator._prepare_private_executor_state(unsafe_state)

    inaccessible_state = tmp_path / "inaccessible-executor"
    inaccessible_state.mkdir(mode=0o600)
    inaccessible_state.chmod(0o600)
    with pytest.raises(operator.OperatorError, match="custody is unsafe"):
        operator._prepare_private_executor_state(inaccessible_state)

    target = tmp_path / "executor-target"
    target.mkdir(mode=0o700)
    linked_state = tmp_path / "linked-executor"
    linked_state.symlink_to(target, target_is_directory=True)
    with pytest.raises(operator.OperatorError, match="custody is unsafe"):
        operator._prepare_private_executor_state(linked_state)


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
    assert environment[operator.LEGACY_BOARD_UNSTALL_POLICY_ENV] == "disabled"
    assert supervisor_argv[-2:] == ["--credential-fd", "7"]
    assert "supervisor-runtime" in supervisor_argv


def test_executor_environment_disables_legacy_board_unstall() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        LEGACY_BOARD_UNSTALL_POLICY_ENV,
    )

    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    route = operator._route_preflight(board)

    environment = operator._executor_environment(
        board,
        route,
        owner_identity={"generation": 3, "schema_revision": 3},
    )

    assert (
        operator.LEGACY_BOARD_UNSTALL_POLICY_ENV
        == LEGACY_BOARD_UNSTALL_POLICY_ENV
    )
    assert environment[operator.LEGACY_BOARD_UNSTALL_POLICY_ENV] == "disabled"


def test_state_owner_build_disables_legacy_board_unstall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        quack_state_server as server_module,
    )

    operator = _operator()
    board, config = operator._load_config(CONFIG)
    paths = {
        "database": tmp_path / "control.duckdb",
        "bootstrap_receipt": tmp_path / "bootstrap.json",
        "owner": tmp_path / "owner",
        "owner_socket": tmp_path / "owner" / "typed-owner.sock",
    }
    paths["database"].write_bytes(b"sealed-test-placeholder")
    paths["bootstrap_receipt"].write_text("{}\n", encoding="utf-8")
    bootstrap = {
        "schema": operator.BOOTSTRAP_SCHEMA,
        "program_id": operator.PROGRAM_ID,
    }
    observed: dict[str, object] = {}

    class _BuildObserved(Exception):
        pass

    def observe_build(**kwargs):
        observed.update(kwargs)
        raise _BuildObserved

    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, config))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(operator, "_json_object", lambda _path: bootstrap)
    monkeypatch.setattr(operator, "_verify_receipt", lambda *_a, **_k: None)
    monkeypatch.setattr(operator, "_require_free_port", lambda *_a, **_k: None)
    monkeypatch.setattr(operator, "_quack_capability", lambda: None)
    monkeypatch.setattr(operator, "_prepare_private_socket_parent", lambda _p: None)
    monkeypatch.setattr(server_module, "build_server", observe_build)
    monkeypatch.setenv(operator.LEGACY_BOARD_UNSTALL_POLICY_ENV, "enabled")

    with pytest.raises(_BuildObserved):
        operator.state_owner(CONFIG)

    assert observed["allow_legacy_board_unstall"] is False
    assert os.environ[operator.LEGACY_BOARD_UNSTALL_POLICY_ENV] == "disabled"


def test_state_owner_internal_grant_uses_maximum_bounded_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        quack_state_server as server_module,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        DEFAULT_GRANT_TTL_SECONDS,
        MAX_GRANT_TTL_SECONDS,
    )

    operator = _operator()
    board, config = operator._load_config(CONFIG)
    paths = {
        "database": tmp_path / "control.duckdb",
        "bootstrap_receipt": tmp_path / "bootstrap.json",
        "owner": tmp_path / "owner",
        "owner_socket": tmp_path / "owner" / "typed-owner.sock",
    }
    paths["database"].write_bytes(b"sealed-test-placeholder")
    paths["bootstrap_receipt"].write_text("{}\n", encoding="utf-8")
    bootstrap = {
        "schema": operator.BOOTSTRAP_SCHEMA,
        "program_id": operator.PROGRAM_ID,
    }
    captured: dict[str, object] = {}

    class _GrantCaptured(Exception):
        pass

    class _Server:
        def typed_command_socket_path(self):
            return paths["owner_socket"]

        def start(self):
            return SimpleNamespace(process_birth_id="process-birth:state-owner")

        def issue_typed_client_grant(self, **kwargs):
            captured.update(kwargs)
            raise _GrantCaptured

    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, config))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(operator, "_json_object", lambda _path: bootstrap)
    monkeypatch.setattr(operator, "_verify_receipt", lambda *_a, **_k: None)
    monkeypatch.setattr(operator, "_require_free_port", lambda *_a, **_k: None)
    monkeypatch.setattr(operator, "_quack_capability", lambda: None)
    monkeypatch.setattr(operator, "_prepare_private_socket_parent", lambda _p: None)
    monkeypatch.setattr(server_module, "build_server", lambda **_kwargs: _Server())

    with pytest.raises(_GrantCaptured):
        operator.state_owner(CONFIG)

    assert operator.INTERNAL_CLIENT_GRANT_TTL_SECONDS == MAX_GRANT_TTL_SECONDS
    assert captured["ttl_seconds"] == MAX_GRANT_TTL_SECONDS
    assert captured["ttl_seconds"] != DEFAULT_GRANT_TTL_SECONDS
    assert captured["process_birth_id"] == "process-birth:state-owner"
    assert captured["client_id"] == "casf-state-owner:federation-runtime"


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
    assert lagging_result["reason_codes"] == [
        "state_owner_outbox_catching_up"
    ]


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


def test_quiescent_route_seal_returns_one_generation_stable_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    policy = object()
    client = _QuiescenceClient(
        generations=["generation:a", "generation:a"],
        rows=[[_routing_row(pending_deliveries=0)]],
    )
    factory = _RouteProjectionFactory([policy])
    clock = _QuiescenceClock()

    result = operator._seal_quiescent_execution_route_policy(
        server=object(),
        owner_client=client,
        admission=_quiescence_admission(),
        timeout_seconds=1.0,
        monotonic=clock,
        sleeper=clock.sleep,
        task_source_factory=factory,
    )

    assert result is policy
    assert client.generation_calls == 2
    assert len(client.routing_calls) == 1
    assert factory.calls == factory.closes == 1
    assert clock.sleeps == []


def test_quiescent_route_seal_waits_for_pending_delivery_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    policy = object()
    client = _QuiescenceClient(
        generations=["generation:a", "generation:a", "generation:a"],
        rows=[
            [_routing_row(pending_deliveries=1)],
            [_routing_row(pending_deliveries=0)],
        ],
    )
    factory = _RouteProjectionFactory([policy])
    clock = _QuiescenceClock()

    result = operator._seal_quiescent_execution_route_policy(
        server=object(),
        owner_client=client,
        admission=_quiescence_admission(),
        timeout_seconds=1.0,
        monotonic=clock,
        sleeper=clock.sleep,
        task_source_factory=factory,
    )

    assert result is policy
    assert client.generation_calls == 3
    assert len(client.routing_calls) == 2
    assert factory.calls == factory.closes == 1
    assert clock.sleeps == [operator.EXECUTION_ROUTE_QUIESCENCE_RETRY_SECONDS]


def test_quiescent_route_seal_retries_a_generation_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    stale_policy = object()
    stable_policy = object()
    client = _QuiescenceClient(
        generations=[
            "generation:a",
            "generation:b",
            "generation:b",
            "generation:b",
        ],
        rows=[
            [_routing_row(pending_deliveries=0)],
            [_routing_row(pending_deliveries=0)],
        ],
    )
    factory = _RouteProjectionFactory([stale_policy, stable_policy])
    clock = _QuiescenceClock()

    result = operator._seal_quiescent_execution_route_policy(
        server=object(),
        owner_client=client,
        admission=_quiescence_admission(),
        timeout_seconds=1.0,
        monotonic=clock,
        sleeper=clock.sleep,
        task_source_factory=factory,
    )

    assert result is stable_policy
    assert client.generation_calls == 4
    assert factory.calls == factory.closes == 2
    assert clock.sleeps == [operator.EXECUTION_ROUTE_QUIESCENCE_RETRY_SECONDS]


def test_quiescent_route_seal_retries_only_a_typed_snapshot_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        TaskSourceConflictError,
    )

    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    policy = object()
    client = _QuiescenceClient(
        generations=["generation:a", "generation:a", "generation:a"],
        rows=[
            [_routing_row(pending_deliveries=0)],
            [_routing_row(pending_deliveries=0)],
        ],
    )
    factory = _RouteProjectionFactory(
        [TaskSourceConflictError("snapshot raced"), policy]
    )
    clock = _QuiescenceClock()

    result = operator._seal_quiescent_execution_route_policy(
        server=object(),
        owner_client=client,
        admission=_quiescence_admission(),
        timeout_seconds=1.0,
        monotonic=clock,
        sleeper=clock.sleep,
        task_source_factory=factory,
    )

    assert result is policy
    assert client.generation_calls == 3
    assert factory.calls == factory.closes == 2
    assert clock.sleeps == [operator.EXECUTION_ROUTE_QUIESCENCE_RETRY_SECONDS]


def test_quiescent_route_seal_times_out_while_deliveries_remain_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    client = _QuiescenceClient(
        generations=["generation:a"],
        rows=[[_routing_row(pending_deliveries=1)]],
    )
    factory = _RouteProjectionFactory([object()])
    clock = _QuiescenceClock()

    with pytest.raises(
        operator.OperatorError,
        match="execution-route quiescence timed out: subscription_deliveries_pending",
    ):
        operator._seal_quiescent_execution_route_policy(
            server=object(),
            owner_client=client,
            admission=_quiescence_admission(),
            timeout_seconds=0.1,
            monotonic=clock,
            sleeper=clock.sleep,
            task_source_factory=factory,
        )

    assert client.generation_calls == 3
    assert len(client.routing_calls) == 3
    assert factory.calls == factory.closes == 0
    assert clock.sleeps == [0.05, 0.05]


def test_quiescent_route_seal_propagates_a_non_conflict_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    client = _QuiescenceClient(
        generations=["generation:a"],
        rows=[[_routing_row(pending_deliveries=0)]],
    )
    failure = PermissionError("typed transport denied")
    factory = _RouteProjectionFactory([failure])
    clock = _QuiescenceClock()

    with pytest.raises(PermissionError) as observed:
        operator._seal_quiescent_execution_route_policy(
            server=object(),
            owner_client=client,
            admission=_quiescence_admission(),
            timeout_seconds=1.0,
            monotonic=clock,
            sleeper=clock.sleep,
            task_source_factory=factory,
        )

    assert observed.value is failure
    assert factory.calls == factory.closes == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "routing_rows",
    [
        [],
        [
            _routing_row(pending_deliveries=0),
            _routing_row(pending_deliveries=0),
        ],
    ],
    ids=("absent", "ambiguous"),
)
def test_quiescent_route_seal_rejects_noncanonical_subscription_authority(
    monkeypatch: pytest.MonkeyPatch,
    routing_rows: list[dict],
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    client = _QuiescenceClient(
        generations=["generation:a"],
        rows=[routing_rows],
    )
    factory = _RouteProjectionFactory([object()])
    clock = _QuiescenceClock()

    with pytest.raises(
        operator.OperatorError,
        match="quiescence requires one active subscription",
    ):
        operator._seal_quiescent_execution_route_policy(
            server=object(),
            owner_client=client,
            admission=_quiescence_admission(),
            timeout_seconds=1.0,
            monotonic=clock,
            sleeper=clock.sleep,
            task_source_factory=factory,
        )

    assert client.generation_calls == 1
    assert len(client.routing_calls) == 1
    assert factory.calls == factory.closes == 0
    assert clock.sleeps == []


def test_quiescent_route_seal_rejects_admitted_fanout_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        operator,
        "_state_owner_outbox_health",
        lambda _server: {"healthy": True},
    )
    row = _routing_row(pending_deliveries=0)
    row["maximum_fanout"] = 17
    client = _QuiescenceClient(
        generations=["generation:a"],
        rows=[[row]],
    )
    factory = _RouteProjectionFactory([object()])
    clock = _QuiescenceClock()

    with pytest.raises(
        operator.OperatorError,
        match="subscription fanout authority drifted",
    ):
        operator._seal_quiescent_execution_route_policy(
            server=object(),
            owner_client=client,
            admission=_quiescence_admission(),
            timeout_seconds=1.0,
            monotonic=clock,
            sleeper=clock.sleep,
            task_source_factory=factory,
        )

    assert factory.calls == factory.closes == 0
    assert clock.sleeps == []


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


def test_steady_state_outbox_guard_survives_gated_drain_and_resets_on_catchup() -> None:
    operator = _operator()
    clock = {"now": 100.0}
    guard = operator._SteadyStateOutboxHealth(
        grace_seconds=30.0,
        monotonic=lambda: clock["now"],
    )
    owner_status = {
        "outbox_worker": {
            "available": True,
            "thread_alive": True,
            "server_owned": True,
            "polling": False,
            "watermark": 126,
            "committed_sequence": 127,
            "drain_count": 59,
            "last_error_type": "",
        },
        "typed_command_gateway": {
            "commit_observer_bound": True,
            "last_observer_error_type": "",
        },
    }
    gated = operator._outbox_worker_health(owner_status)
    assert gated["structural_healthy"] is True
    assert gated["caught_up"] is False
    assert gated["healthy"] is False
    assert gated["classification"] == "state_owner_outbox_catching_up"

    first = guard.observe(gated)
    assert first["continue_running"] is True
    assert first["classification"] == "state_owner_outbox_catching_up"
    clock["now"] = 129.999
    assert guard.observe(gated)["continue_running"] is True

    owner_status["outbox_worker"]["watermark"] = 127
    caught_up = operator._outbox_worker_health(owner_status)
    recovered = guard.observe(caught_up)
    assert recovered["continue_running"] is True
    assert recovered["classification"] == "state_owner_outbox_healthy"

    # A later independent gap receives a fresh bounded grace window.
    clock["now"] = 200.0
    owner_status["outbox_worker"]["committed_sequence"] = 128
    later_gap = operator._outbox_worker_health(owner_status)
    restarted = guard.observe(later_gap)
    assert restarted["continue_running"] is True
    assert restarted["lag_seconds"] == 0.0


def test_steady_state_outbox_guard_fails_closed_after_persistent_lag() -> None:
    operator = _operator()
    clock = {"now": 500.0}
    guard = operator._SteadyStateOutboxHealth(
        grace_seconds=30.0,
        monotonic=lambda: clock["now"],
    )
    lagging = operator._outbox_worker_health(
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

    assert guard.observe(lagging)["continue_running"] is True
    clock["now"] = 529.999
    assert guard.observe(lagging)["continue_running"] is True
    clock["now"] = 530.0
    expired = guard.observe(lagging)
    assert expired["continue_running"] is False
    assert expired["classification"] == "state_owner_outbox_lag_timeout"
    assert expired["reason_code"] == "state_owner_outbox_lag_timeout"
    assert expired["lag_seconds"] == 30.0


def test_steady_state_outbox_guard_rejects_structural_failure_immediately() -> None:
    operator = _operator()
    guard = operator._SteadyStateOutboxHealth(monotonic=lambda: 1.0)
    failed = operator._outbox_worker_health(
        {
            "outbox_worker": {
                "available": False,
                "thread_alive": False,
                "server_owned": True,
                "polling": False,
                "watermark": 127,
                "committed_sequence": 127,
                "drain_count": 59,
                "last_error_type": "",
            },
            "typed_command_gateway": {
                "commit_observer_bound": True,
                "last_observer_error_type": "",
            },
        }
    )

    decision = guard.observe(failed)
    assert decision["continue_running"] is False
    assert decision["classification"] == "state_owner_outbox_unavailable"
    assert decision["reason_code"] == "state_owner_outbox_unavailable"


def test_steady_state_outbox_guard_retains_bounded_transport_reason() -> None:
    operator = _operator()
    guard = operator._SteadyStateOutboxHealth(monotonic=lambda: 1.0)
    failed = operator._outbox_worker_health(
        {
            "outbox_worker": {
                "available": False,
                "thread_alive": False,
                "server_owned": True,
                "polling": False,
                "watermark": 127,
                "committed_sequence": 127,
                "drain_count": 59,
                "last_error_type": "QuackClientTransportError",
                "last_error_message": "must-not-escape",
            },
            "typed_command_gateway": {
                "commit_observer_bound": True,
                "last_observer_error_type": "",
            },
        }
    )

    decision = guard.observe(failed)

    assert decision["continue_running"] is False
    assert decision["reason_code"] == "state_owner_outbox_transport_failure"
    assert "QuackClientTransportError" not in json.dumps(decision)
    assert "must-not-escape" not in json.dumps(decision)


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


def test_launch_generation_admits_fresh_identity_after_complete_stop(
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
    operator._require_unused_launch_generation(paths)


def test_complete_stop_retires_consumed_control_plane(tmp_path: Path) -> None:
    operator = _operator()
    runtime = tmp_path / "runtime"
    database = runtime / "control.duckdb"
    evidence = runtime / "evidence" / "bootstrap-operator"
    owner = runtime / "quack-owner"
    state = runtime / "state"
    evidence.mkdir(parents=True)
    owner.mkdir()
    state.mkdir()
    database.write_text("duckdb", encoding="utf-8")
    (runtime / ".control.duckdb.lock").write_text("", encoding="utf-8")
    paths = {
        "runtime": runtime,
        "database": database,
        "owner": owner,
        "state": state,
        "operator_evidence": evidence,
        "launch_receipt": evidence / "launch-current.json",
        "stop_receipt": evidence / "stop-current.json",
    }
    launch = operator._persist_receipt(
        paths,
        "launch",
        {
            "schema": operator.LAUNCH_SCHEMA,
            "program_id": operator.PROGRAM_ID,
            "marker": "retire",
        },
    )
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
    operator._require_unused_launch_generation(paths)
    assert not database.exists()
    archived = list((runtime / "quarantine").glob("consumed-*"))
    assert len(archived) == 1
    assert (archived[0] / "control-plane" / "control.duckdb").is_file()
    assert (archived[0] / "evidence" / "bootstrap-operator").is_dir()


def test_tampered_stop_receipt_still_fails_closed(tmp_path: Path) -> None:
    operator = _operator()
    paths = {
        "operator_evidence": tmp_path / "evidence",
        "launch_receipt": tmp_path / "launch.json",
        "stop_receipt": tmp_path / "stop.json",
    }
    launch = operator._persist_receipt(
        paths,
        "launch",
        {
            "schema": operator.LAUNCH_SCHEMA,
            "program_id": operator.PROGRAM_ID,
            "marker": "tamper",
        },
    )
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


@pytest.mark.parametrize(
    ("deterministic_task_count", "model_task_count"),
    [(41, 3), (43, 1)],
)
def test_stop_retires_exact_historical_execution_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deterministic_task_count: int,
    model_task_count: int,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    paths = {
        "operator_evidence": tmp_path / "evidence",
        "launch_receipt": tmp_path / "launch.json",
        "stop_receipt": tmp_path / "stop.json",
        "executor_current": tmp_path / "executor-current.json",
        "owner_status": tmp_path / "owner-status.json",
        "owner": tmp_path / "owner",
    }
    historical_route = _execution_route_summary(
        deterministic_task_count=deterministic_task_count,
        model_task_count=model_task_count,
        policy_id=f"route-policy:historical-{deterministic_task_count}-{model_task_count}",
    )
    births = _install_admitted_generation(
        operator,
        paths,
        launch_route=historical_route,
    )
    retired: list[dict] = []

    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, {}))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(operator, "_owner_liveness", lambda _status: "dead")
    monkeypatch.setattr(operator, "_birth_liveness", lambda _birth: "dead")
    monkeypatch.setattr(operator, "_port_is_free", lambda _host, _port: True)

    def retire_executor(*, paths, supervisor_birth, fallback_executor_birth, grace_seconds):
        del paths, grace_seconds
        retired.extend([dict(supervisor_birth), dict(fallback_executor_birth)])
        return (
            [
                {
                    "role": "executor_supervisor",
                    "birth": dict(supervisor_birth),
                    "result": "already_dead",
                },
                {
                    "role": "executor_daemon",
                    "birth": dict(fallback_executor_birth),
                    "result": "already_dead",
                },
            ],
            [],
        )

    monkeypatch.setattr(operator, "_retire_configured_executor", retire_executor)
    monkeypatch.setattr(
        operator,
        "_terminate_birth",
        lambda _birth, *, grace_seconds: "already_dead",
    )

    receipt = operator.stop(CONFIG)

    assert receipt["complete"] is True
    assert receipt["execution_route_policy"] == historical_route
    assert retired == [births["executor_supervisor"], births["executor"]]


@pytest.mark.parametrize(
    "tamper",
    ["executor_route_mismatch", "executor_binding_mismatch", "launch_receipt"],
)
def test_historical_stop_rejects_mismatched_or_tampered_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    paths = {
        "operator_evidence": tmp_path / "evidence",
        "launch_receipt": tmp_path / "launch.json",
        "stop_receipt": tmp_path / "stop.json",
        "executor_current": tmp_path / "executor-current.json",
        "owner_status": tmp_path / "owner-status.json",
        "owner": tmp_path / "owner",
    }
    historical_route = _execution_route_summary(
        deterministic_task_count=41,
        model_task_count=3,
        policy_id="route-policy:historical",
    )
    current_route = (
        {**historical_route, "policy_id": "route-policy:foreign"}
        if tamper == "executor_route_mismatch"
        else historical_route
    )
    _install_admitted_generation(
        operator,
        paths,
        launch_route=historical_route,
        current_route=current_route,
    )
    if tamper == "executor_binding_mismatch":
        executor_current = operator._json_object(paths["executor_current"])
        executor_current["execution_route_policy_id"] = "route-policy:foreign"
        operator._atomic_json(paths["executor_current"], executor_current)
    elif tamper == "launch_receipt":
        launch = operator._json_object(paths["launch_receipt"])
        launch["execution_route_policy"]["policy_id"] = "route-policy:tampered"
        operator._atomic_json(paths["launch_receipt"], launch)

    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, {}))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(
        operator,
        "_retire_configured_executor",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched route reached executor retirement")
        ),
    )

    expected = (
        "identity is absent or invalid"
        if tamper == "launch_receipt"
        else "executor runtime is not bound"
    )
    with pytest.raises(operator.OperatorError, match=expected):
        operator.stop(CONFIG)


def test_status_reports_obsolete_route_only_for_fully_dead_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    paths = {
        "operator_evidence": tmp_path / "evidence",
        "launch_receipt": tmp_path / "launch.json",
        "status_receipt": tmp_path / "status.json",
        "executor_current": tmp_path / "executor-current.json",
        "owner_status": tmp_path / "owner-status.json",
    }
    historical_route = _execution_route_summary(
        deterministic_task_count=41,
        model_task_count=3,
        policy_id="route-policy:historical",
    )
    births = _install_admitted_generation(
        operator,
        paths,
        launch_route=historical_route,
    )
    executor_current = operator._json_object(paths["executor_current"])

    monkeypatch.setattr(operator, "_load_config", lambda _path: (board, {}))
    monkeypatch.setattr(operator, "_runtime_paths", lambda _board: paths)
    monkeypatch.setattr(operator, "_owner_liveness", lambda _status: "dead")
    monkeypatch.setattr(operator, "_birth_liveness", lambda _birth: "dead")
    monkeypatch.setattr(
        operator,
        "_runtime_projection",
        lambda _paths, *, launched_at_ns, expected_supervisor_birth: {
            "supervisor_status": {},
            "task_state": {},
        },
    )
    monkeypatch.setattr(
        operator,
        "_executor_runtime_projection",
        lambda _paths, *, expected_supervisor_birth: {
            "current": executor_current,
            "execution_route_policy": historical_route,
            "supervisor_liveness": "dead",
            "executor_liveness": "dead",
        },
    )

    snapshot = operator._status_snapshot(CONFIG, persist=False)

    assert snapshot["execution_route_policy"] == historical_route
    assert snapshot["execution_route_population"] == "obsolete"
    assert snapshot["classification"] == "unavailable"
    assert snapshot["healthy"] is False
    assert snapshot["blocked_or_stuck"] is True

    monkeypatch.setattr(
        operator,
        "_birth_liveness",
        lambda birth: (
            "alive" if dict(birth) == births["master"] else "dead"
        ),
    )
    with pytest.raises(operator.OperatorError, match="runtime that is not fully dead"):
        operator._status_snapshot(CONFIG, persist=False)


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
        schema_revision=3,
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
