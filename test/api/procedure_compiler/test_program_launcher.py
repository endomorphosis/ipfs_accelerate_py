from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/ops/agent_supervisor/procedure_compiler_program.py"
CONFIG = REPO_ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pcpc_program_launcher_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(module: ModuleType) -> Any:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    return module.parse_program_config(payload, repo_root=REPO_ROOT, config_path=CONFIG)


def _fake_inspect(
    module: ModuleType,
    config: Any,
    *,
    head: str = "1" * 40,
    tree: str = "2" * 40,
    container_id: str = "3" * 64,
    container_name: str | None = None,
    running: bool = True,
) -> dict[str, Any]:
    exact_name = container_name or module.OWNER_CONTAINER_NAME
    labels = module.owner_labels(config, head=head, tree=tree)
    owner_argv = module.build_owner_create_argv(config, head=head, tree=tree)
    command = owner_argv[owner_argv.index(config.image_id) + 1 :]
    port_key = f"{config.container_port}/tcp"
    binding = [{"HostIp": config.host, "HostPort": str(config.port)}]
    mounts = [
        {
            "Type": "bind",
            "Source": str(config.repo_root),
            "Destination": str(config.repo_root),
            "RW": False,
        },
        {
            "Type": "bind",
            "Source": str(config.owner_write_root),
            "Destination": str(config.owner_write_root),
            "RW": True,
        },
    ]
    mounts.extend(
        {
            "Type": "bind",
            "Source": str(config.extension_directory / name),
            "Destination": f"{module.OWNER_EXTENSION_TARGET}/{name}",
            "RW": False,
        }
        for name in config.extension_hashes
    )
    return {
        "Id": container_id,
        "Name": f"/{exact_name}",
        "Image": config.image_id,
        "Config": {
            "Image": config.image_id,
            "User": "0:0",
            "Hostname": module.OWNER_CONTAINER_HOSTNAME,
            "Entrypoint": ["/usr/bin/env"],
            "Env": ["PATH=/opt/pcpc-runtime/bin:/usr/local/bin:/usr/bin:/bin"],
            "Cmd": command,
            "Labels": labels,
            "ExposedPorts": {port_key: {}},
        },
        "HostConfig": {
            "ReadonlyRootfs": True,
            "NetworkMode": "bridge",
            "PortBindings": {port_key: binding},
            "PublishAllPorts": False,
            "IpcMode": "private",
            "PidMode": "",
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "PidsLimit": config.pids_limit,
            "Memory": config.memory_bytes,
            "NanoCpus": config.cpus * 1_000_000_000,
        },
        "State": {"Running": running},
        "NetworkSettings": {
            "Ports": {port_key: binding} if running else {},
            "Networks": (
                {
                    "bridge": {
                        "NetworkID": "4" * 64,
                        "EndpointID": "5" * 64,
                        "IPAddress": "172.17.0.2",
                    }
                }
                if running
                else {}
            ),
        },
        "Mounts": mounts,
    }


def test_owner_config_is_closed_and_rejects_unknown_normative_field() -> None:
    module = _load()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["quack_owner_isolation"]["surprise_authority"] = True

    with pytest.raises(module.ProgramLaunchError, match="unknown or missing") as raised:
        module.parse_program_config(payload, repo_root=REPO_ROOT, config_path=CONFIG)

    assert raised.value.code == "config_unknown_field"


def test_owner_argv_has_exact_isolation_and_no_host_secret_or_socket_mount() -> None:
    module = _load()
    config = _config(module)
    argv = module.build_owner_create_argv(config, head="1" * 40, tree="2" * 40)
    joined = "\n".join(argv)

    assert argv[:5] == [
        config.runtime_executable,
        "--host",
        config.runtime_endpoint,
        "container",
        "create",
    ]
    assert "--read-only" in argv
    assert argv[argv.index("--network") + 1] == "bridge"
    assert argv[argv.index("--publish") + 1] == (
        f"{config.host}:{config.port}:{config.container_port}/tcp"
    )
    assert argv[argv.index("--ipc") + 1] == "private"
    assert "--pid" not in argv
    assert argv[argv.index("--user") + 1] == "0:0"
    assert argv[argv.index("--hostname") + 1] == module.OWNER_CONTAINER_HOSTNAME
    assert argv[argv.index("--entrypoint") + 1] == "/usr/bin/env"
    assert argv[argv.index("--container-bind-host") + 1] == "0.0.0.0"
    assert argv[argv.index("--container-port") + 1] == str(config.container_port)
    assert "-i" in argv
    assert f"src={config.repo_root},dst={config.repo_root},readonly" in joined
    assert f"src={config.owner_write_root},dst={config.owner_write_root}" in joined
    for name in config.extension_hashes:
        assert (
            f"src={config.extension_directory / name},"
            f"dst={module.OWNER_EXTENSION_TARGET}/{name},readonly"
        ) in joined
    assert joined.count(".duckdb_extension") == 8
    assert "/var/run/docker.sock" not in joined
    assert str(Path.home() / ".codex") not in joined
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in joined
    assert config.secret_handle in argv


def test_isolation_receipt_is_canonical_cidv1_and_forgery_is_rejected(tmp_path: Path) -> None:
    module = _load()
    config = replace(
        _config(module),
        owner_write_root=tmp_path / "control",
        database_path=tmp_path / "control" / "control.duckdb",
        state_dir=tmp_path / "control" / "quack-owner",
    )
    receipt = module.build_owner_isolation_receipt(
        config,
        container_id="3" * 64,
        issued_at="2026-08-20T23:00:00Z",
    )

    module.validate_owner_isolation_receipt(receipt, config=config, container_id="3" * 64)
    assert receipt["receipt_cid"].startswith("b")
    assert json.loads(module.canonical_json_bytes(receipt)) == receipt

    forged = dict(receipt)
    forged["allowed_rw_mount_targets"] = [str(tmp_path)]
    with pytest.raises(module.ProgramLaunchError) as raised:
        module.validate_owner_isolation_receipt(forged, config=config, container_id="3" * 64)
    assert raised.value.code == "receipt_forged"

    unknown = dict(receipt)
    unknown["authority"] = True
    with pytest.raises(module.ProgramLaunchError) as raised:
        module.validate_owner_isolation_receipt(unknown, config=config, container_id="3" * 64)
    assert raised.value.code == "receipt_unknown_field"


def test_atomic_receipt_creation_never_replaces_unknown_existing_file(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "receipt.json"
    payload = {"schema": "test@1", "value": 1}
    module._atomic_create(path, payload)
    first = path.read_bytes()
    module._atomic_create(path, payload)
    assert path.read_bytes() == first == module.canonical_json_bytes(payload)
    assert path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(module.ProgramLaunchError) as raised:
        module._atomic_create(path, {"schema": "test@1", "value": 2})
    assert raised.value.code == "existing_artifact_conflict"
    assert path.read_bytes() == first


def test_forged_container_inspect_and_scope_escape_are_rejected() -> None:
    module = _load()
    config = _config(module)
    inspect = _fake_inspect(module, config)

    assert (
        module.validate_owner_container_inspect(
            inspect,
            config=config,
            head="1" * 40,
            tree="2" * 40,
            require_running=True,
        )
        == "3" * 64
    )

    weakened = copy.deepcopy(inspect)
    weakened["HostConfig"]["ReadonlyRootfs"] = False
    with pytest.raises(module.ProgramLaunchError) as raised:
        module.validate_owner_container_inspect(
            weakened,
            config=config,
            head="1" * 40,
            tree="2" * 40,
        )
    assert raised.value.code == "container_isolation_weakened"

    escaped = copy.deepcopy(inspect)
    escaped["Mounts"].append({"Type": "bind", "Source": "/", "Destination": "/host", "RW": True})
    with pytest.raises(module.ProgramLaunchError) as raised:
        module.validate_owner_container_inspect(
            escaped,
            config=config,
            head="1" * 40,
            tree="2" * 40,
        )
    assert raised.value.code == "container_mount_mismatch"

    forged = copy.deepcopy(inspect)
    forged["Config"]["Labels"]["org.ipfs-accelerate.pcpc.head"] = "f" * 40
    with pytest.raises(module.ProgramLaunchError) as raised:
        module.validate_owner_container_inspect(
            forged,
            config=config,
            head="1" * 40,
            tree="2" * 40,
        )
    assert raised.value.code == "container_label_mismatch"

    extra_network = copy.deepcopy(inspect)
    extra_network["NetworkSettings"]["Networks"]["unexpected"] = {
        "NetworkID": "6" * 64,
        "EndpointID": "7" * 64,
        "IPAddress": "172.18.0.2",
    }
    with pytest.raises(module.ProgramLaunchError) as raised:
        module.validate_owner_container_inspect(
            extra_network,
            config=config,
            head="1" * 40,
            tree="2" * 40,
            require_running=True,
        )
    assert raised.value.code == "container_network_mismatch"


def test_materialization_verification_uses_injected_runner_and_rejects_extra_field() -> None:
    module = _load()
    head = "1" * 40
    tree = "2" * 40
    valid = {
        "schema": module.MATERIALIZATION_VERIFICATION_SCHEMA,
        "valid": True,
        "repository_commit": head,
        "repository_tree": tree,
        "database_path": _config(module).store_id,
        "projection_cid": "bafy-projection",
        "task_count": 32,
        "completed_task_ids": [f"PCPC-{index:03d}" for index in range(9)],
        "ready_task_ids": ["PCPC-009", "PCPC-011", "PCPC-013"],
        "blocked_task_ids": [],
        "projection_matches_events": True,
        "plan_current": True,
        "tasks_current": True,
        "qualification_current": True,
        "freshly_qualified": True,
        "fresh_qualification_cid": "bafy-qualification",
    }
    calls: list[tuple[str, ...]] = []

    def runner(argv, *, cwd, env=None, timeout=60):
        del cwd, env, timeout
        calls.append(tuple(argv))
        return module.CommandResult(0, json.dumps(valid), "")

    launcher = module.ProcedureCompilerProgramLauncher(repo_root=REPO_ROOT, runner=runner)
    assert launcher._materialization_verify(head=head, tree=tree)["valid"] is True
    assert calls[0][-1] == "--verify"

    valid["forged_completion"] = True
    with pytest.raises(module.ProgramLaunchError) as raised:
        launcher._materialization_verify(head=head, tree=tree)
    assert raised.value.code == "materialization_invalid"


@pytest.mark.parametrize("forgery", ["unknown_field", "log_path_escape"])
def test_supervisor_launch_rejects_unknown_receipt_and_path_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forgery: str
) -> None:
    module = _load()
    head = "1" * 40
    tree = "2" * 40
    base = _config(module)
    config = replace(
        base,
        state_root=tmp_path / "state",
        evidence_root=tmp_path / "evidence",
    )
    launch = {
        "coordinator_pid": 424242,
        "coordinator_pid_path": str(config.state_root / "configured-board-master.pid"),
        "coordinator_log": str(tmp_path / "outside" / "coordinator.log"),
    }
    if forgery == "unknown_field":
        launch["self_authorized"] = True

    def runner(argv, *, cwd, env=None, timeout=60):
        del argv, cwd, env, timeout
        return module.CommandResult(0, json.dumps(launch), "")

    launcher = module.ProcedureCompilerProgramLauncher(repo_root=REPO_ROOT, runner=runner)
    launcher.config = config
    monkeypatch.setattr(launcher, "repository_identity", lambda *, require_clean: (head, tree))
    monkeypatch.setattr(
        launcher,
        "owner_status",
        lambda: {
            "schema": module.OWNER_STATUS_RECEIPT_SCHEMA,
            "program": module.PROGRAM,
            "repository_commit": head,
            "repository_tree": tree,
            "container_id": "9" * 64,
            "running": True,
            "ready": True,
            "remote_probe": {"blocked_task_ids": []},
            "receipt_cid": "identity-is-not-authority",
        },
    )

    with pytest.raises(module.ProgramLaunchError) as raised:
        launcher.supervisor_start()

    assert raised.value.code == "supervisor_receipt_invalid"


@pytest.mark.parametrize(
    ("failure_stage", "expected_code", "receipt_expected"),
    [
        ("post_create", "container_label_mismatch", False),
        ("post_receipt", "container_start_failed", True),
        ("readiness_timeout", "owner_readiness_timeout", True),
    ],
)
def test_failed_owner_attempt_is_exactly_removed_quarantined_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_code: str,
    receipt_expected: bool,
) -> None:
    module = _load()
    base = _config(module)
    owner_root = tmp_path / "control"
    owner_root.mkdir(mode=0o700)
    database_path = owner_root / "control.duckdb"
    database_path.touch(mode=0o600)
    state_dir = owner_root / "quack-owner"
    config = replace(
        base,
        owner_write_root=owner_root,
        state_dir=state_dir,
        database_path=database_path,
        store_id="state/test/control.duckdb",
        host="127.0.0.1",
        port=45679,
        container_port=45679,
        endpoint="quack:127.0.0.1:45679",
        evidence_root=tmp_path / "evidence",
        state_root=tmp_path / "state",
    )
    container_id = "8" * 64
    runtime_state = {"running": False, "removed": False, "inspect_calls": 0}
    command_calls: list[tuple[str, ...]] = []

    def runner(argv, *, cwd, env=None, timeout=60):
        del cwd, env, timeout
        call = tuple(argv)
        command_calls.append(call)
        action = call[4] if len(call) > 4 and call[3] == "container" else ""
        if action == "create":
            return module.CommandResult(0, f"{container_id}\n", "")
        if action == "start":
            if failure_stage == "post_receipt":
                return module.CommandResult(1, "", "bounded start failure")
            runtime_state["running"] = True
            return module.CommandResult(0, f"{container_id}\n", "")
        if action == "stop":
            runtime_state["running"] = False
            return module.CommandResult(0, f"{container_id}\n", "")
        if action == "rm":
            runtime_state["removed"] = True
            return module.CommandResult(0, f"{container_id}\n", "")
        raise AssertionError(f"unexpected runner call: {call}")

    launcher = module.ProcedureCompilerProgramLauncher(
        repo_root=REPO_ROOT,
        runner=runner,
        clock=lambda: "2026-08-20T23:00:00Z",
        sleeper=lambda _seconds: None,
    )
    launcher.config = config
    monkeypatch.setattr(
        launcher, "repository_identity", lambda *, require_clean: ("1" * 40, "2" * 40)
    )
    monkeypatch.setattr(launcher, "_verify_runtime", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_materialization_verify",
        lambda *, head, tree: {"repository_commit": head, "repository_tree": tree},
    )

    def inspect_container(*, allow_absent: bool):
        del allow_absent
        runtime_state["inspect_calls"] += 1
        if runtime_state["removed"] or runtime_state["inspect_calls"] == 1:
            return None
        inspect = _fake_inspect(
            module,
            config,
            running=bool(runtime_state["running"]),
            container_id=container_id,
        )
        if failure_stage == "post_create" and runtime_state["inspect_calls"] == 2:
            inspect["Config"]["Labels"]["org.ipfs-accelerate.pcpc.head"] = "f" * 40
        return inspect

    monkeypatch.setattr(launcher, "_inspect_container", inspect_container)
    if failure_stage == "readiness_timeout":
        monkeypatch.setattr(module, "MAX_OWNER_WAIT_SECONDS", 0)

    with pytest.raises(module.ProgramLaunchError) as raised:
        launcher.owner_start()

    assert raised.value.code == expected_code
    assert runtime_state["removed"] is True
    assert not state_dir.exists()
    quarantine = owner_root / "quack-owner-quarantine" / container_id
    assert quarantine.is_dir()
    assert (quarantine / module.OWNER_ISOLATION_FILENAME).exists() is receipt_expected
    assert any(call[3:5] == ("container", "rm") for call in command_calls)
    if failure_stage == "readiness_timeout":
        assert any(call[3:5] == ("container", "stop") for call in command_calls)
    # The fixed live state path is available for a bounded retry; evidence is
    # preserved under the content-bound failed-container identity.
    state_dir.mkdir(mode=0o700)
    assert state_dir.is_dir()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("IPFS_ACCELERATE_PCPC_RUN_DOCKER_E2E") != "1",
    reason="set IPFS_ACCELERATE_PCPC_RUN_DOCKER_E2E=1 for rootless-Docker owner E2E",
)
def test_live_owner_create_inspect_receipt_start_handle_ready_stop() -> None:
    """Exercise the real container contract in disposable program-owned state."""

    module = _load()
    base = _config(module)
    state_parent = REPO_ROOT / "state"
    state_parent.mkdir(exist_ok=True)
    container_name = f"pcpc-owner-e2e-{uuid.uuid4().hex}"
    container_id = ""
    with tempfile.TemporaryDirectory(prefix=".pcpc-owner-e2e-", dir=state_parent) as raw:
        root = Path(raw).resolve()
        owner_root = root / "control"
        owner_state = owner_root / "quack-owner"
        database_path = owner_root / "control.duckdb"
        owner_state.mkdir(parents=True, mode=0o700)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        relative_database = database_path.relative_to(REPO_ROOT).as_posix()
        config = replace(
            base,
            owner_write_root=owner_root,
            state_dir=owner_state,
            database_path=database_path,
            store_id=relative_database,
            secret_handle=f"handle:pcpc-owner-e2e-{uuid.uuid4().hex}",
            port=port,
            container_port=port,
            endpoint=f"quack:127.0.0.1:{port}",
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )

        population = {
            "objectives": [{"goal_cid": "goal:e2e", "goal_alias": "G-E2E"}],
            "plans": [{"plan_cid": "plan:e2e", "goal_cid": "goal:e2e", "status": "active"}],
            "tasks": [
                {
                    "task_cid": "task:e2e",
                    "task_alias": "E2E-001",
                    "goal_cid": "goal:e2e",
                    "plan_cid": "plan:e2e",
                    "title": "Disposable owner readiness task",
                    "status": "ready",
                }
            ],
        }
        with DatabaseTaskSource(database_path) as source:
            source.materialize(population, repository_tree_id="2" * 40, plan_root_cid="plan:e2e")
        argv = module.build_owner_create_argv(
            config,
            head="1" * 40,
            tree="2" * 40,
            container_name=container_name,
        )
        try:
            created = subprocess.run(
                argv,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
            )
            assert created.returncode == 0, created.stderr
            container_id = created.stdout.strip()
            inspected = subprocess.run(
                [
                    config.runtime_executable,
                    "--host",
                    config.runtime_endpoint,
                    "container",
                    "inspect",
                    container_id,
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            inspect = json.loads(inspected.stdout)[0]
            module.validate_owner_container_inspect(
                inspect,
                config=config,
                head="1" * 40,
                tree="2" * 40,
                container_id=container_id,
                container_name=container_name,
            )
            isolation = module.build_owner_isolation_receipt(
                config,
                container_id=container_id,
                issued_at="2026-08-20T23:00:00Z",
            )
            module._atomic_create(config.isolation_receipt_path, isolation)
            subprocess.run(
                [
                    config.runtime_executable,
                    "--host",
                    config.runtime_endpoint,
                    "container",
                    "start",
                    container_id,
                ],
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            deadline = time.monotonic() + 90
            last_error = "status was not published"
            while time.monotonic() < deadline:
                try:
                    status = module._safe_read_json(
                        owner_state / "quack-state-server.status.json",
                        exact_fields=module._STATUS_FIELDS,
                        noun="owner status",
                    )
                    module.validate_owner_status_payload(
                        status,
                        config=config,
                        head="1" * 40,
                        tree="2" * 40,
                    )
                    probe = module._default_remote_probe(config, tree="2" * 40)
                    if probe["authenticated"] and probe["task_count"] == 1:
                        break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    time.sleep(0.25)
            else:
                logs = subprocess.run(
                    [
                        config.runtime_executable,
                        "--host",
                        config.runtime_endpoint,
                        "logs",
                        container_id,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                pytest.fail(
                    f"owner did not become ready ({last_error}): "
                    f"{logs.stderr[-2000:]} {logs.stdout[-2000:]}"
                )
            running_inspect = subprocess.run(
                [
                    config.runtime_executable,
                    "--host",
                    config.runtime_endpoint,
                    "container",
                    "inspect",
                    container_id,
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            module.validate_owner_container_inspect(
                json.loads(running_inspect.stdout)[0],
                config=config,
                head="1" * 40,
                tree="2" * 40,
                container_id=container_id,
                container_name=container_name,
                require_running=True,
            )
        finally:
            if container_id:
                subprocess.run(
                    [
                        config.runtime_executable,
                        "--host",
                        config.runtime_endpoint,
                        "stop",
                        "--time",
                        "5",
                        container_id,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
                subprocess.run(
                    [
                        config.runtime_executable,
                        "--host",
                        config.runtime_endpoint,
                        "rm",
                        container_id,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
