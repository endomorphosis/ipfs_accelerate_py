from __future__ import annotations

import importlib.util
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    MigrationDriftError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    load_datasets_authoritative_operational_catalog,
    verify_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)

ROOT = Path(__file__).resolve().parents[2]
OPERATOR_PATH = ROOT / (
    "scripts/run_logic_governed_compositional_verification_fabric_quack.py"
)


def _operator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lgcvf_quack_successor", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_datasets_profile(database: Path) -> None:
    operator = _operator()
    operator.datasets_profile_migration(database)
    repository = open_intent_repository(
        database,
        owner_id="lgcvf-quack-successor-test:seed",
        install_schema=False,
    )
    try:
        repository.upsert_objective(
            objective_id="objective:test",
            objective_alias="O",
            title="Synthetic objective",
        )
        repository.upsert_goal(
            goal_cid="goal:test",
            goal_alias="G",
            title="Synthetic goal",
            objective_id="objective:test",
        )
        repository.upsert_plan(
            plan_cid="plan:test",
            plan_alias="P",
            goal_cid="goal:test",
        )
        repository.upsert_task(
            task_cid="task:test",
            task_alias="LGCVF-TEST",
            goal_cid="goal:test",
            plan_cid="plan:test",
            objective_id="objective:test",
            ordinal=1,
            status="ready",
        )
    finally:
        repository.close()


def test_datasets_profile_callback_rejects_default_catalog_drift(tmp_path: Path) -> None:
    operator = _operator()
    database = tmp_path / "control.duckdb"

    report = operator.datasets_profile_migration(database)
    verification = verify_datasets_authoritative_operational_schema(database)
    expected_catalog = load_datasets_authoritative_operational_catalog().fingerprint()

    assert verification["valid"] is True
    assert report.schema_fingerprint == verification["schema_fingerprint"]
    assert report.catalog_fingerprint == expected_catalog
    assert verification["catalog_fingerprint"] == expected_catalog
    with pytest.raises(MigrationDriftError):
        install_control_plane_schema(database)


def test_successor_bootstrap_invokes_protected_recovery_verifier_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    observed: dict[str, object] = {}
    report = {
        "valid": True,
        "target_generation": "lgcvf-run-v17",
        "stores_unchanged": True,
        "source_database_statuses_read": False,
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "ready_task_ids": ["LGCVF-081"],
    }

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(report),
            stderr="",
        )

    monkeypatch.setattr(operator.subprocess, "run", run)

    assert operator._canonical_recovery_verification(tmp_path) == report
    command = observed["command"]
    assert isinstance(command, list)
    assert command[:4] == [sys.executable, "-I", "-S", "-B"]
    assert command[-1] == "recovery-verify"


def test_verified_run_v17_clone_is_no_overwrite_and_content_addressed(
    tmp_path: Path,
) -> None:
    operator = _operator()
    source = tmp_path / "run-v17" / "control.duckdb"
    target = tmp_path / "run-v18" / "control.duckdb"
    provenance = tmp_path / "run-v18" / "evidence" / "provenance.json"
    source.parent.mkdir(parents=True)
    _seed_datasets_profile(source)
    source_digest = operator._sha256_regular_file(source)
    recovery = {
        "valid": True,
        "target_generation": "lgcvf-run-v17",
        "stores_unchanged": True,
        "source_database_statuses_read": False,
        "verification_root": "sha256:" + ("ab" * 32),
        "receipt_cid": "baguqeera-test-recovery",
    }

    receipt = operator.clone_verified_successor(
        source,
        target,
        provenance,
        recovery_verification=recovery,
    )

    assert source_digest == operator._sha256_regular_file(source)
    assert source_digest == operator._sha256_regular_file(target)
    assert receipt["source_generation"] == "lgcvf-run-v17"
    assert receipt["target_generation"] == "lgcvf-run-v18"
    assert receipt["source_database_statuses_read"] is False
    assert operator._strict_json(
        provenance,
        expected_schema=operator.PROVENANCE_SCHEMA,
    ) == receipt
    with pytest.raises(operator.SuccessorOperatorError, match="overwrite"):
        operator.clone_verified_successor(
            source,
            target,
            provenance,
            recovery_verification=recovery,
        )


def test_owner_socket_and_ducklake_projection_are_physically_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    production_paths = operator._paths(ROOT)
    socket_path = production_paths["owner_socket"]

    assert socket_path.parent.name == f"ipfs-accelerate-lgcvf-{os.geteuid()}"
    assert len(os.fsencode(socket_path)) <= operator.UNIX_SOCKET_PATH_CEILING
    assert str(ROOT) not in str(socket_path)
    operator._prepare_private_owner_socket(socket_path)
    metadata = os.lstat(socket_path.parent)
    assert oct(metadata.st_mode & 0o777) == "0o700"
    assert metadata.st_uid == os.geteuid()

    monkeypatch.setattr(
        operator,
        "_extension_preflight",
        lambda: {
            "available": True,
            "extensions": {"quack": "test", "ducklake": "test"},
            "automatic_installation_permitted": False,
        },
    )
    preflight = operator.projection_preflight(tmp_path)
    runtime_paths = operator._paths(tmp_path)
    assert preflight["valid"] is False
    assert preflight["capability"]["available"] is True
    assert preflight["source_database_present"] is False
    assert preflight["provenance_receipt_present"] is False
    assert preflight["source_admitted"] is False
    assert Path(preflight["projection_root"]) == runtime_paths["projection_root"]
    assert runtime_paths["projection_root"] / "control.duckdb" != runtime_paths[
        "successor_database"
    ]
    assert preflight["authoritative"] is False
    assert preflight["scheduling_authority"] is False
    assert preflight["completion_authority"] is False
    assert preflight["read_by_scheduler"] is False
    assert preflight["requires_stopped_checkpoint"] is True


def test_projection_extension_policy_is_load_only_and_never_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        board_control_plane as board_control_plane_module,
    )

    operator = _operator()

    class LoadFailureConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str) -> None:
            self.statements.append(statement)
            raise RuntimeError("injected local LOAD failure")

    connection = LoadFailureConnection()
    error = board_control_plane_module._try_load_extension(
        connection,
        "quack",
        allow_install=False,
    )
    assert connection.statements == ["LOAD quack"]
    assert "INSTALL disabled by policy" in error

    observed: dict[str, object] = {}
    sentinel = object()

    def open_projection(
        repo_root: Path,
        *,
        root: Path,
        allow_extension_install: bool,
    ) -> object:
        observed.update(
            {
                "repo_root": repo_root,
                "root": root,
                "allow_extension_install": allow_extension_install,
            }
        )
        return sentinel

    monkeypatch.setattr(
        board_control_plane_module,
        "open_board_control_plane",
        open_projection,
    )
    projection_root = tmp_path / "projection"
    assert operator._open_projection_plane(tmp_path, projection_root) is sentinel
    assert observed == {
        "repo_root": tmp_path,
        "root": projection_root,
        "allow_extension_install": False,
    }

    identity = type(
        "Identity",
        (),
        {"generation": 1, "schema_revision": "schema-v1"},
    )()
    owner_state = tmp_path / "owner"
    owner_state.mkdir()
    environment = operator._child_environment(
        token="test_token_value",
        identity=identity,
        owner_state=owner_state,
        root=tmp_path,
    )
    assert (
        environment[board_control_plane_module.BOARD_EXTENSION_INSTALL_POLICY_ENV]
        == board_control_plane_module.BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
        == operator.BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
    )


OWNER_PROCESS = r"""
import importlib.util
import json
import os
import select
import sys
import time
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server


def emit(payload):
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


configuration = json.loads(sys.stdin.readline())
spec = importlib.util.spec_from_file_location(
    "lgcvf_quack_successor_owner", configuration["operator_path"]
)
if spec is None or spec.loader is None:
    raise RuntimeError("successor operator module is unavailable")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
os.environ["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] = configuration[
    "store_generation"
]
server = None
try:
    server = build_server(
        database_path=Path(configuration["database"]),
        state_dir=Path(configuration["owner_state"]),
        host="127.0.0.1",
        port=0,
        store_id=configuration["store_id"],
        repository_id="repository:lgcvf-quack-successor-test",
        secret_handle=configuration["secret_handle"],
        migrate=operator.datasets_profile_migration,
        typed_command_socket_path=Path(configuration["typed_socket"]),
    )
    identity = server.start()
    if server._vault is None:
        raise RuntimeError("owner token vault is unavailable")
    token = server._vault.resolve(identity.secret_handle)
    emit(
        {
            "event": "ready",
            "identity": identity.to_dict(),
            "mutation_inbox": str(server.mutation_inbox_path()),
            "token": token,
            "typed_socket": str(server.typed_command_socket_path()),
        }
    )
    stop_requested = False
    while not stop_requested:
        server.service_mutation_inbox(max_requests=32)
        readable, _, _ = select.select([sys.stdin], [], [], 0.005)
        if not readable:
            continue
        line = sys.stdin.readline()
        if not line:
            stop_requested = True
            continue
        request = json.loads(line)
        action = request.get("action")
        if action == "observe":
            row = server._connection.execute(
                "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
            ).fetchone()
            emit(
                {
                    "event": "observed",
                    "lifecycle": server.lifecycle.value,
                    "task": [str(row[0]), int(row[1])],
                }
            )
        elif action == "stop":
            stop_requested = True
        else:
            raise RuntimeError("owner control action is invalid")
    server.stop()
    emit({"event": "stopped"})
except BaseException as exc:
    emit(
        {
            "event": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )
    raise
"""


WORKER = r"""
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    resolve_database_implementation_paths,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)

endpoint, lane_text, lane_root_text, ready_text, gate_text, result_text = sys.argv[1:]
lane = int(lane_text)
lane_root = Path(lane_root_text)
ready = Path(ready_text)
gate = Path(gate_text)
result_path = Path(result_text)
sidecars = resolve_database_implementation_paths(
    Namespace(
        state_dir=lane_root,
        state_prefix=f"lgcvf_lane_{lane}",
        authority_mode="quack",
        database_path=None,
        todo_path=None,
        coordination_path=None,
    ),
    authority_mode="quack",
)
source = None
daemon = None
payload = {
    "lane": lane,
    "database_path": str(sidecars["database_path"]),
    "coordination_path": str(sidecars["coordination_path"]),
    "execution_path": str(sidecars["execution_path"]),
}
try:
    source = DatabaseTaskSource(
        endpoint,
        install_schema=False,
        owner_id=f"lgcvf-quack-test:lane:{lane}",
    )
    daemon = DatabaseImplementationDaemon(
        database_path=sidecars["database_path"],
        coordination_path=sidecars["coordination_path"],
        execution_path=sidecars["execution_path"],
        owner_session_id=f"lgcvf-quack-test:lane:{lane}",
        authority_mode="quack",
        task_source_kind="duckdb",
        quack_uri=endpoint,
        task_source=source,
        task_shard_count=4,
        task_shard_index=lane,
        strict_task_sharding=True,
    )
    task = source.get_task("LGCVF-TEST")
    if task is None or task.status != "ready" or task.revision != 1:
        raise RuntimeError("lane did not observe the exact initial task head")
    ready.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 30.0
    while not gate.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("CAS start gate timed out")
        time.sleep(0.005)
    try:
        result = source.compare_and_set_status(
            "LGCVF-TEST",
            expected_revision=1,
            status="in_progress",
        )
    except TaskSourceConflictError:
        payload["outcome"] = "conflict"
    else:
        payload["outcome"] = "success"
        payload["revision"] = result.revision
except BaseException as exc:
    payload["outcome"] = "error"
    payload["error_type"] = type(exc).__name__
    payload["error"] = str(exc)
finally:
    if daemon is not None:
        daemon.close()
    if source is not None:
        source.close()
    result_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
"""


def _wait_for_paths(paths: list[Path], processes: list[subprocess.Popen[str]]) -> None:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            return
        exited = [process.returncode for process in processes if process.poll() is not None]
        if exited:
            raise AssertionError(f"worker exited before the CAS gate: {exited}")
        time.sleep(0.02)
    raise AssertionError("workers did not reach the CAS gate")


def _read_owner_event(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("timed out waiting for owner control response")
        readable, _, _ = select.select([process.stdout], [], [], remaining)
        if not readable:
            continue
        line = process.stdout.readline()
        if not line:
            raise AssertionError(
                f"owner control pipe closed with returncode={process.poll()}"
            )
        payload = json.loads(line)
        assert isinstance(payload, dict)
        return payload


def _capture_process_birth(process: subprocess.Popen[str]) -> ProcessBirthIdentity:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("owner exited before process birth was captured")
        try:
            birth = read_process_birth(process.pid)
        except OSError:
            birth = None
        if birth is not None:
            return birth
        time.sleep(0.01)
    raise AssertionError("owner process birth could not be captured")


def _signal_exact_process(
    process: subprocess.Popen[str],
    birth: ProcessBirthIdentity,
    signum: int,
) -> None:
    if process.poll() is not None:
        return
    if owner_liveness(birth) is not OwnerLiveness.ALIVE:
        return
    try:
        process_group = os.getpgid(birth.pid)
        if process_group == birth.pid:
            os.killpg(process_group, signum)
        else:
            os.kill(birth.pid, signum)
    except ProcessLookupError:
        return


def _bounded_stop_owner(
    process: subprocess.Popen[str],
    birth: ProcessBirthIdentity,
) -> dict[str, object]:
    event: dict[str, object] = {}
    if process.poll() is None:
        assert process.stdin is not None
        try:
            process.stdin.write('{"action":"stop"}\n')
            process.stdin.flush()
            event = _read_owner_event(process, timeout_seconds=5.0)
        except (AssertionError, BrokenPipeError, OSError):
            event = {}
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        _signal_exact_process(process, birth, signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _signal_exact_process(process, birth, signal.SIGKILL)
            process.wait(timeout=2.0)
    stdout_tail, stderr = process.communicate(timeout=1.0)
    return {
        "event": event,
        "returncode": process.returncode,
        "stdout_tail": stdout_tail,
        "stderr": stderr,
    }


def _assert_secret_absent_from_regular_files(root: Path, secret: str) -> None:
    needle = secret.encode("ascii")
    for candidate in root.rglob("*"):
        try:
            candidate.lstat()
        except OSError:
            continue
        if not candidate.is_file() or candidate.is_symlink():
            continue
        with candidate.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                assert needle not in block, f"raw Quack token persisted in {candidate}"


def test_real_four_process_quack_cas_has_one_winner_and_private_sidecars(
    tmp_path: Path,
) -> None:
    capability = probe_quack_capabilities()
    if not capability.passes_health_check:
        pytest.skip(
            "preinstalled pinned Quack capability unavailable: "
            f"{capability.status.value}/{capability.reason_code}"
        )

    operator = _operator()
    runtime = tmp_path / "run-v18"
    database = runtime / "control.duckdb"
    owner_state = runtime / "quack-owner"
    logical_generation = "lgcvf-synthetic-v1"
    _seed_datasets_profile(database)
    test_paths = operator._paths(tmp_path)
    operator._prepare_private_owner_socket(test_paths["owner_socket"])
    owner_environment = {
        name: os.environ[name]
        for name in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
        if name in os.environ
    }
    owner_environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    owner = subprocess.Popen(
        [sys.executable, "-c", OWNER_PROCESS],
        cwd=ROOT,
        env=owner_environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    owner_birth = _capture_process_birth(owner)
    owner_shutdown: dict[str, object] | None = None
    processes: list[subprocess.Popen[str]] = []
    try:
        owner_configuration = {
            "database": str(database),
            "operator_path": str(OPERATOR_PATH),
            "owner_state": str(owner_state),
            "secret_handle": operator.SECRET_HANDLE,
            "store_generation": logical_generation,
            "store_id": str(database),
            "typed_socket": str(test_paths["owner_socket"]),
        }
        assert owner.stdin is not None
        owner.stdin.write(
            json.dumps(owner_configuration, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        owner.stdin.flush()
        ready = _read_owner_event(owner, timeout_seconds=30.0)
        assert ready.get("event") == "ready", ready
        identity = ready.get("identity")
        assert isinstance(identity, dict)
        token = ready.get("token")
        assert isinstance(token, str) and token
        endpoint = str(identity["listen_uri"])
        typed_socket = Path(str(ready["typed_socket"]))
        mutation_inbox = Path(str(ready["mutation_inbox"]))
        assert typed_socket == test_paths["owner_socket"]
        assert len(os.fsencode(typed_socket)) <= operator.UNIX_SOCKET_PATH_CEILING
        assert token.encode("ascii") not in Path(
            f"/proc/{owner.pid}/cmdline"
        ).read_bytes()

        sink = operator._token_sink(owner_state)
        gate = tmp_path / "cas.go"
        lane_roots = [runtime / "state" / f"lane-{index}" for index in range(4)]
        ready_paths = [tmp_path / f"lane-{index}.ready" for index in range(4)]
        result_paths = [tmp_path / f"lane-{index}.result.json" for index in range(4)]
        for lane_root in lane_roots:
            lane_root.mkdir(parents=True)
        environment = dict(owner_environment)
        environment.update(
            {
                operator.TOKEN_ENV: token,
                operator.TOKEN_FILE_ENV: str(sink),
                "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE": "quack",
                "IPFS_ACCELERATE_AGENT_TASK_SOURCE_KIND": "duckdb",
                "IPFS_ACCELERATE_AGENT_STATE_FAILOVER_POLICY": "fail_closed",
                "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": str(
                    identity["secret_handle"]
                ),
                "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT": endpoint,
                "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": str(identity["store_id"]),
                "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": logical_generation,
                "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION": (
                    "datasets-authoritative-operational-v1"
                ),
                "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": str(
                    identity["generation"]
                ),
                "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": str(
                    identity["schema_revision"]
                ),
                "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH": str(owner_state),
                "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR": str(mutation_inbox),
                "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(tmp_path),
            }
        )
        for lane in range(4):
            command = [
                sys.executable,
                "-c",
                WORKER,
                endpoint,
                str(lane),
                str(lane_roots[lane]),
                str(ready_paths[lane]),
                str(gate),
                str(result_paths[lane]),
            ]
            assert all(token not in item for item in command)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            )
        _wait_for_paths(ready_paths, processes)
        for process in processes:
            cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
            assert token.encode("ascii") not in cmdline
        gate.write_text("go\n", encoding="utf-8")
        outputs = [process.communicate(timeout=45.0) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
        assert sorted(result["outcome"] for result in results) == [
            "conflict",
            "conflict",
            "conflict",
            "success",
        ]
        assert {result.get("revision") for result in results} == {None, 2}
        for field in ("database_path", "coordination_path", "execution_path"):
            assert len({result[field] for result in results}) == 4
        assert len(
            {
                result[field]
                for result in results
                for field in (
                    "database_path",
                    "coordination_path",
                    "execution_path",
                )
            }
        ) == 12
        for result in results:
            assert Path(result["coordination_path"]).is_file()
            assert Path(result["execution_path"]).is_file()
        owner.stdin.write('{"action":"observe"}\n')
        owner.stdin.flush()
        observed = _read_owner_event(owner, timeout_seconds=10.0)
        assert observed == {
            "event": "observed",
            "lifecycle": "ready",
            "task": ["in_progress", 2],
        }
        assert not tuple(runtime.rglob("*.quack-token"))
        _assert_secret_absent_from_regular_files(tmp_path, token)
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3.0)
        owner_shutdown = _bounded_stop_owner(owner, owner_birth)
    assert owner_shutdown["event"] == {"event": "stopped"}, owner_shutdown
    assert owner_shutdown["returncode"] == 0, owner_shutdown
    assert not (owner_state / "typed-state-owner.token").exists()
