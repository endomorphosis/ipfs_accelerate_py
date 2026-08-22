from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import signal
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


def _with_hermetic_extensions(module: ModuleType, config: Any, root: Path) -> Any:
    extension_directory = root / "source-extensions"
    extension_directory.mkdir()
    qualification_hashes: dict[str, str] = {}
    for index, name in enumerate(config.qualification_extension_hashes):
        content = f"hermetic-extension-{index}".encode()
        (extension_directory / name).write_bytes(content)
        qualification_hashes[name] = hashlib.sha256(content).hexdigest()
    return replace(
        config,
        extension_directory=extension_directory,
        extension_hashes={
            name: qualification_hashes[name] for name in config.extension_hashes
        },
        projection_extension_hashes={
            name: qualification_hashes[name]
            for name in config.projection_extension_hashes
        },
        state_root=root / "state",
    )


def _write_private(path: Path, payload: str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def _coordinator_launch_fixture(
    module: ModuleType,
    config: Any,
    *,
    head: str,
    tree: str,
    pid: int = 424242,
    include_lane_heartbeats: bool,
) -> tuple[dict[str, Any], Any, Any, dict[str, Any], tuple[int, ...]]:
    session = "3" * 64
    configuration_revision = module.content_identity(
        {
            "path": config.config_path.relative_to(config.repo_root).as_posix(),
            "bytes_sha256": config.config_sha256,
        }
    )
    scheduler_payload = json.loads(config.config_path.read_text(encoding="utf-8"))
    state_relative = Path(scheduler_payload["runtime_paths"]["state"])
    taskboard_path = config.repo_root / scheduler_payload["taskboard_path"]
    launch_attestation_max_age_ms = int(
        min(
            float(scheduler_payload["watchdog_startup_grace_seconds"]),
            module.COORDINATOR_READY_TIMEOUT_MAX_SECONDS,
        )
        * 1_000
    )
    log_root = config.state_root.parent / "logs"
    status_path = config.state_root / f"configured-board-{session}.status.json"
    lane_paths = tuple(
        config.state_root
        / f"lane-{lane_index}"
        / f"pcpc_lane_{lane_index}_supervisor_status.json"
        for lane_index in range(4)
    )
    bootstrap = "raise SystemExit(99)"
    pin_json = "{}"
    argv = (
        sys.executable,
        "-I",
        "-c",
        bootstrap,
        "9",
        pin_json,
        "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
        "sha256:" + hashlib.sha256(bootstrap.encode("utf-8")).hexdigest(),
        "sha256:" + "4" * 64,
        "--repo-root",
        str(config.repo_root),
        "--config",
        str(config.config_path),
        "--accepted-tree-root",
        str(config.repo_root),
        "--accepted-control-plane-pin-json",
        pin_json,
        "--accepted-control-plane-fd",
        "9",
        "--accepted-control-plane-capsule-parent",
        str(Path(tempfile.gettempdir()) / "asref-configured-control-plane-test"),
        "--coordinator-launch-session",
        session,
        "--coordinator-status-path",
        str(status_path),
        "launch",
        "--foreground",
        "--duration-seconds",
        str(float(module.MAX_SUPERVISOR_DURATION_SECONDS)),
        "--implement",
    )
    profile = module.LifecycleProfile(
        target_id=f"configured-board-coordinator:{module.PROGRAM}",
        run_id=f"configured-board:{module.PROGRAM}:{session}",
        configuration_root=configuration_revision,
        repository_root=str(config.repo_root),
        state_root=str(config.state_root),
        run_root=str(config.state_root),
        argv=argv,
        cwd=str(config.repo_root),
        health_path=str(status_path),
        health_stale_ms=launch_attestation_max_age_ms,
    )
    identity = module.ProcessIdentity(
        pid=pid,
        start_time_ticks=987654,
        parent_pid=1234,
        process_group_id=pid,
        session_id=pid,
        boot_id="test-boot-id",
        argv=argv,
        cwd=str(config.repo_root),
        executable=str(Path(sys.executable).resolve()),
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=0,
        configuration_root=profile.configuration_root,
    )
    now_ms = int(time.time() * 1000)
    unsigned_status = {
        "schema": module._COORDINATOR_STATUS_SCHEMA,
        "repository_commit": head,
        "repository_tree": tree,
        "configuration_revision": configuration_revision,
        "board_namespace": module.PROGRAM,
        "launch_session_id": session,
        "lifecycle_profile_id": profile.profile_id,
        "coordinator_pid": pid,
        "coordinator_process_start_ticks": identity.start_time_ticks,
        "coordinator_argv_cid": module.content_identity({"argv": list(argv)}),
        "started_at_ms": now_ms,
        "attested_at_ms": now_ms,
        "phase": "launch_attested",
        "lane_status_paths": [str(path) for path in lane_paths],
    }
    status = {
        **unsigned_status,
        "receipt_cid": module.content_identity(unsigned_status),
    }
    pid_path = config.state_root / "configured-board-master.pid"
    log_path = log_root / f"configured-board-20260821T000000Z-{session}.log"
    _write_private(pid_path, f"{pid}\n")
    _write_private(log_path, "coordinator started\n")
    _write_private(status_path, status)
    lane_pids = tuple(500000 + lane_index for lane_index in range(4))
    if include_lane_heartbeats:
        updated_at = module.datetime.now(module.UTC).isoformat().replace("+00:00", "Z")
        for lane_index, (lane_path, lane_pid) in enumerate(
            zip(lane_paths, lane_pids, strict=True)
        ):
            _write_private(
                lane_path,
                {
                    "schema": module._LANE_STATUS_SCHEMA,
                    "status": "running",
                    "updated_at": updated_at,
                    "supervisor_pid": lane_pid,
                    "repo_root": str(config.repo_root),
                    "task_prefix": "## PCPC-",
                    "state_prefix": f"pcpc_lane_{lane_index}",
                },
            )
    unsigned_launch = {
        "schema": module._COORDINATOR_LAUNCH_SCHEMA,
        "repository_commit": head,
        "repository_tree": tree,
        "configuration_revision": configuration_revision,
        "board_namespace": module.PROGRAM,
        "launch_session_id": session,
        "coordinator_pid": pid,
        "coordinator_pid_path": str(pid_path),
        "coordinator_log": str(log_path),
        "coordinator_status_path": str(status_path),
        "coordinator_status_cid": status["receipt_cid"],
        "coordinator_profile": profile.to_dict(),
        "coordinator_process_identity": identity.to_dict(),
        "coordinator_argv_cid": module.content_identity({"argv": list(argv)}),
    }
    launch = {
        **unsigned_launch,
        "receipt_cid": module.content_identity(unsigned_launch),
    }
    bindings = {
        "configuration_revision": configuration_revision,
        "state_root": config.state_root,
        "log_root": log_root,
        "lane_status_paths": lane_paths,
        "state_relative": state_relative,
        "taskboard_path": taskboard_path,
        "task_prefix": "PCPC-",
        "task_header_prefix": "## PCPC-",
        "max_lanes": 4,
        "launch_attestation_max_age_ms": launch_attestation_max_age_ms,
    }
    return launch, profile, identity, bindings, lane_pids


def _admit_test_owner(
    module: ModuleType,
    launcher: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    head: str,
    tree: str,
) -> None:
    monkeypatch.setattr(
        launcher, "repository_identity", lambda *, require_clean: (head, tree)
    )
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
    monkeypatch.setattr(
        launcher, "_probe", lambda *, tree: {"blocked_task_ids": []}
    )


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


def test_ducklake_projection_config_is_closed_and_pinned() -> None:
    module = _load()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["ducklake_projection_program"]["surprise_install_policy"] = "allowed"

    with pytest.raises(module.ProgramLaunchError, match="unknown or missing") as raised:
        module.parse_program_config(payload, repo_root=REPO_ROOT, config_path=CONFIG)

    assert raised.value.code == "config_unknown_field"

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["ducklake_projection_program"]["extension_files_sha256"][
        "ducklake.duckdb_extension"
    ] = "0" * 64
    with pytest.raises(module.ProgramLaunchError, match="allowlist") as raised:
        module.parse_program_config(payload, repo_root=REPO_ROOT, config_path=CONFIG)
    assert raised.value.code == "config_invalid"


@pytest.mark.parametrize(
    ("catalog_path", "data_path"),
    (
        (
            "docs/redteam.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/data",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            ".",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/data",
        ),
        (
            "state/agent_supervisor_proof_carrying_procedure_compiler/history/catalog.ducklake",
            "state/agent_supervisor_proof_carrying_procedure_compiler/history",
        ),
    ),
)
def test_ducklake_projection_paths_are_exact_and_non_overlapping(
    catalog_path: str, data_path: str
) -> None:
    module = _load()
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["ducklake_projection_program"]["catalog_path"] = catalog_path
    payload["ducklake_projection_program"]["data_path"] = data_path

    with pytest.raises(module.ProgramLaunchError) as raised:
        module.parse_program_config(payload, repo_root=REPO_ROOT, config_path=CONFIG)

    assert raised.value.code == "config_invalid"


def test_qualification_refuses_missing_pinned_ducklake_extension(tmp_path: Path) -> None:
    module = _load()
    config = _with_hermetic_extensions(module, _config(module), tmp_path)
    (config.extension_directory / "ducklake.duckdb_extension").unlink()

    with pytest.raises(module.ProgramLaunchError, match="ducklake") as raised:
        module._qualification_environment(config)

    assert raised.value.code == "extension_missing"
    assert not config.state_root.exists()


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
    for name in config.projection_extension_hashes:
        assert name not in joined
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


def test_materialization_verification_uses_isolated_extension_home_and_rejects_extra_field(
    tmp_path: Path,
) -> None:
    module = _load()
    head = "1" * 40
    tree = "2" * 40
    config = _with_hermetic_extensions(module, _config(module), tmp_path)
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
    qualification_homes: list[Path] = []
    contaminate_home = {"enabled": False}

    def runner(argv, *, cwd, env=None, timeout=60):
        del cwd, timeout
        assert env is not None
        assert "HOME" in env
        assert set(env) <= {
            "PATH",
            "HOME",
            "PYTHONUSERBASE",
            "PYTHONDONTWRITEBYTECODE",
            "CUDA_CACHE_DISABLE",
            "CUDA_CACHE_PATH",
            "XDG_CACHE_HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TZ",
            module.TRUSTED_DUCKDB_HOME_ENV,
        }
        assert not any(
            fragment in name.upper()
            for name in env
            for fragment in ("CODEX", "OPENAI", "ANTHROPIC", "GITHUB_TOKEN", "HF_TOKEN")
        )
        home = Path(env["HOME"])
        assert env[module.TRUSTED_DUCKDB_HOME_ENV] == str(home)
        assert home != Path.home()
        assert home.stat().st_mode & 0o777 == 0o500
        assert {item.name for item in home.iterdir()} == {".cache", ".duckdb"}
        cache = home / ".cache"
        assert cache.stat().st_mode & 0o777 == 0o700
        assert {item.name for item in cache.iterdir()} == {
            "cuda",
            "ipfs_accelerate",
            "xdg",
        }
        assert env["CUDA_CACHE_PATH"] == str(cache / "cuda")
        assert env["XDG_CACHE_HOME"] == str(cache / "xdg")
        assert env["CUDA_CACHE_DISABLE"] == "1"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        isolated_extensions = home / ".duckdb/extensions/v1.5.5/linux_arm64"
        assert {
            name: hashlib.sha256((isolated_extensions / name).read_bytes()).hexdigest()
            for name in config.qualification_extension_hashes
        } == config.qualification_extension_hashes
        assert all(
            (isolated_extensions / name).stat().st_mode & 0o777 == 0o400
            for name in config.qualification_extension_hashes
        )
        (cache / "ipfs_accelerate" / "qualified.txt").write_text(
            "non-authoritative cache\n", encoding="utf-8"
        )
        (cache / "cuda" / "qualified.bin").write_bytes(b"cache")
        if contaminate_home["enabled"]:
            home.chmod(0o700)
            try:
                (home / ".codex").mkdir()
            finally:
                home.chmod(0o500)
        qualification_homes.append(home)
        calls.append(tuple(argv))
        return module.CommandResult(0, json.dumps(valid), "")

    launcher = module.ProcedureCompilerProgramLauncher(repo_root=REPO_ROOT, runner=runner)
    launcher.config = config
    assert launcher._materialization_verify(head=head, tree=tree)["valid"] is True
    assert calls[0][-1] == "--verify"
    assert qualification_homes and qualification_homes[0].is_dir()
    first_file_inode = (
        qualification_homes[0]
        / ".duckdb/extensions/v1.5.5/linux_arm64/quack.duckdb_extension"
    ).stat().st_ino

    valid["forged_completion"] = True
    with pytest.raises(module.ProgramLaunchError) as raised:
        launcher._materialization_verify(head=head, tree=tree)
    assert raised.value.code == "materialization_invalid"
    assert set(qualification_homes) == {config.qualification_home}
    assert (
        config.qualification_home
        / ".duckdb/extensions/v1.5.5/linux_arm64/quack.duckdb_extension"
    ).stat().st_ino == first_file_inode

    valid.pop("forged_completion")
    contaminate_home["enabled"] = True
    with pytest.raises(module.ProgramLaunchError) as raised:
        launcher._materialization_verify(head=head, tree=tree)
    assert raised.value.code == "qualification_home_invalid"
    contaminate_home["enabled"] = False
    recovered = module._qualification_environment(config)
    assert recovered["HOME"] == str(config.qualification_home)
    assert {item.name for item in config.qualification_home.iterdir()} == {
        ".cache",
        ".duckdb",
    }
    assert not (
        config.qualification_home / ".cache/ipfs_accelerate/qualified.txt"
    ).exists()
    quarantines = tuple((config.state_root / "qualification-home-quarantine").iterdir())
    assert len(quarantines) == 1
    assert (quarantines[0] / ".codex").is_dir()
    assert not any(".staging-" in item.name for item in config.qualification_home.parent.iterdir())


@pytest.mark.parametrize(
    "forgery", ["unknown_field", "log_path_escape", "stdout_contamination"]
)
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
    config = _with_hermetic_extensions(module, config, tmp_path)
    launch = {
        "coordinator_pid": 424242,
        "coordinator_pid_path": str(config.state_root / "configured-board-master.pid"),
        "coordinator_log": str(tmp_path / "outside" / "coordinator.log"),
    }
    if forgery == "unknown_field":
        launch["self_authorized"] = True

    def runner(argv, *, cwd, env=None, timeout=60):
        del cwd, timeout
        assert argv[-1] == "--launch-receipt-only"
        assert env is not None
        assert env["HOME"] == str(config.qualification_home)
        assert env[module.TRUSTED_DUCKDB_HOME_ENV] == str(config.qualification_home)
        assert config.qualification_home.is_dir()
        stdout = json.dumps(launch)
        if forgery == "stdout_contamination":
            stdout = "untrusted provider diagnostic\n" + stdout
        return module.CommandResult(0, stdout, "")

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

    assert raised.value.code == (
        "json_invalid"
        if forgery == "stdout_contamination"
        else "supervisor_receipt_invalid"
    )


def test_supervisor_launch_rejects_forged_unrelated_live_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    head = "1" * 40
    tree = "2" * 40
    config = replace(
        _with_hermetic_extensions(module, _config(module), tmp_path),
        evidence_root=tmp_path / "evidence",
    )
    observed: dict[str, Any] = {}

    def runner(argv, *, cwd, env=None, timeout=60):
        del cwd, env, timeout
        assert argv[-1] == "--launch-receipt-only"
        launch, profile, identity, bindings, _lane_pids = _coordinator_launch_fixture(
            module,
            config,
            head=head,
            tree=tree,
            include_lane_heartbeats=False,
        )
        observed.update(profile=profile, identity=identity, bindings=bindings)
        return module.CommandResult(0, json.dumps(launch), "")

    launcher = module.ProcedureCompilerProgramLauncher(repo_root=REPO_ROOT, runner=runner)
    launcher.config = config
    launcher.sleeper = lambda _seconds: None
    _admit_test_owner(module, launcher, monkeypatch, head=head, tree=tree)
    monkeypatch.setattr(
        module,
        "_scheduler_launch_bindings",
        lambda _config: observed["bindings"],
    )

    class UnrelatedLiveProcessAdapter:
        def _identity(self, pid: int, profile: Any) -> Any:
            del pid, profile
            payload = observed["identity"].to_dict()
            payload.pop("identity_id")
            payload["start_time_ticks"] += 1
            return module.ProcessIdentity.from_dict(payload)

        def _stat(self, pid: int) -> tuple[int, int, int, int]:
            identity = observed["identity"]
            return 1, pid, pid, identity.start_time_ticks + 1

    monkeypatch.setattr(module, "LinuxProcessAdapter", UnrelatedLiveProcessAdapter)

    with pytest.raises(module.ProgramLaunchError) as raised:
        launcher.supervisor_start()

    assert raised.value.code == "coordinator_identity_mismatch"
    assert not config.evidence_root.exists()


def test_supervisor_launch_rejects_coordinator_without_all_lane_heartbeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    head = "1" * 40
    tree = "2" * 40
    config = replace(
        _with_hermetic_extensions(module, _config(module), tmp_path),
        evidence_root=tmp_path / "evidence",
    )
    observed: dict[str, Any] = {}
    process_state = {"alive": False}
    runner_calls = 0

    def runner(argv, *, cwd, env=None, timeout=60):
        nonlocal runner_calls
        del cwd, env, timeout
        runner_calls += 1
        process_state["alive"] = True
        launch, profile, identity, bindings, _lane_pids = _coordinator_launch_fixture(
            module,
            config,
            head=head,
            tree=tree,
            include_lane_heartbeats=False,
        )
        observed.update(profile=profile, identity=identity, bindings=bindings)
        return module.CommandResult(0, json.dumps(launch), "")

    launcher = module.ProcedureCompilerProgramLauncher(repo_root=REPO_ROOT, runner=runner)
    launcher.config = config
    sleeps: list[float] = []
    launcher.sleeper = sleeps.append
    _admit_test_owner(module, launcher, monkeypatch, head=head, tree=tree)
    monkeypatch.setattr(
        module,
        "_scheduler_launch_bindings",
        lambda _config: observed["bindings"],
    )

    class ExactProcessAdapter:
        def _identity(self, pid: int, profile: Any) -> Any:
            if not process_state["alive"]:
                raise ProcessLookupError(pid)
            assert pid == observed["identity"].pid
            assert profile == observed["profile"]
            return observed["identity"]

        def _stat(self, pid: int) -> tuple[int, int, int, int]:
            identity = observed["identity"]
            assert pid == identity.pid
            if not process_state["alive"]:
                raise ProcessLookupError(pid)
            return (
                identity.parent_pid,
                identity.process_group_id,
                identity.session_id,
                identity.start_time_ticks,
            )

        def _signal_exact(self, identity: Any, signum: int) -> None:
            assert identity == observed["identity"]
            assert signum == signal.SIGTERM
            process_state["alive"] = False

    monkeypatch.setattr(module, "LinuxProcessAdapter", ExactProcessAdapter)

    for expected_runner_calls in (1, 2):
        with pytest.raises(module.ProgramLaunchError) as raised:
            launcher.supervisor_start()

        assert raised.value.code == "coordinator_not_ready"
        assert runner_calls == expected_runner_calls
        assert not (config.state_root / "configured-board-master.pid").exists()
        assert not any(config.state_root.glob("configured-board-*.status.json"))

    assert len(sleeps) == 2 * (module.MAX_SUPERVISOR_READINESS_ATTEMPTS - 1)
    assert not config.evidence_root.exists()


@pytest.mark.parametrize("process_kind", ["unrelated", "zombie"])
def test_supervisor_launch_rejects_unrelated_or_zombie_lane_process(
    tmp_path: Path,
    process_kind: str,
) -> None:
    module = _load()
    config = _with_hermetic_extensions(module, _config(module), tmp_path)
    payload = json.loads(config.config_path.read_text(encoding="utf-8"))
    bindings = {
        "state_relative": Path(payload["runtime_paths"]["state"]),
        "taskboard_path": config.repo_root / payload["taskboard_path"],
        "task_header_prefix": "## PCPC-",
    }
    command = (
        [sys.executable, "-c", "pass"]
        if process_kind == "zombie"
        else [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    process = subprocess.Popen(
        command,
        cwd=config.repo_root,
        start_new_session=True,
    )
    try:
        if process_kind == "zombie":
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                raw = Path(f"/proc/{process.pid}/stat").read_text(encoding="utf-8")
                if raw[raw.rfind(")") + 2 :].split()[0] == "Z":
                    break
                time.sleep(0.01)
        coordinator_ticks = module.LinuxProcessAdapter._stat(os.getpid())[3]
        assert not module._configured_lane_process_ready(
            config=config,
            bindings=bindings,
            lane_index=0,
            supervisor_pid=process.pid,
            coordinator_pid=os.getpid(),
            coordinator_start_ticks=coordinator_ticks,
            repository_commit="1" * 40,
            repository_tree="2" * 40,
        )
    finally:
        if process_kind != "zombie":
            process.terminate()
        process.wait(timeout=5.0)


def test_supervisor_launch_admits_exact_lifecycle_marked_lane_process(
    tmp_path: Path,
) -> None:
    module = _load()
    config = replace(
        _with_hermetic_extensions(module, _config(module), tmp_path),
        repo_root=tmp_path,
    )
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    state_relative = Path(payload["runtime_paths"]["state"])
    entry = tmp_path / module.IMPLEMENTATION_ENTRY_RELATIVE
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    bindings = {
        "state_relative": state_relative,
        "taskboard_path": config.repo_root / payload["taskboard_path"],
        "task_header_prefix": "## PCPC-",
        "max_lanes": 4,
    }
    lane_name = f"{module.PROGRAM}-0"
    state_dir = (config.repo_root / state_relative / "lane-0").resolve()
    command = [
        sys.executable,
        str(entry),
        "--todo-path",
        str(bindings["taskboard_path"]),
        "--task-prefix",
        "## PCPC-",
        "--state-dir",
        str(state_dir),
        "--state-prefix",
        "pcpc_lane_0",
        "--task-shard-count",
        "4",
        "--task-shard-index",
        "0",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            module.RUN_ID_ENV: (
                "multi-supervisor:"
                + hashlib.sha256(
                    f"{config.repo_root.resolve()}:{lane_name}".encode()
                ).hexdigest()
            ),
            module.PROFILE_ID_ENV: "sha256:" + "5" * 64,
            module.TARGET_ID_ENV: f"supervisor-track:{lane_name}",
            module.REPOSITORY_ROOT_ENV: str(config.repo_root.resolve()),
            module.STATE_ROOT_ENV: str(state_dir),
            module.RUN_ROOT_ENV: str(state_dir / "lifecycle-runs" / lane_name),
            module.FENCING_EPOCH_ENV: "0",
            module.CONFIGURATION_ROOT_ENV: "sha256:" + "6" * 64,
        }
    )
    coordinator_ticks = module.LinuxProcessAdapter._stat(os.getpid())[3]
    process = subprocess.Popen(
        command,
        cwd=config.repo_root,
        env=environment,
        start_new_session=True,
    )
    try:
        assert module._configured_lane_process_ready(
            config=config,
            bindings=bindings,
            lane_index=0,
            supervisor_pid=process.pid,
            coordinator_pid=os.getpid(),
            coordinator_start_ticks=coordinator_ticks,
            repository_commit="1" * 40,
            repository_tree="2" * 40,
        )
    finally:
        process.terminate()
        process.wait(timeout=5.0)


def test_supervisor_launch_admits_exact_identity_and_all_lane_heartbeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    head = "1" * 40
    tree = "2" * 40
    config = replace(
        _with_hermetic_extensions(module, _config(module), tmp_path),
        evidence_root=tmp_path / "evidence",
    )
    observed: dict[str, Any] = {}

    def runner(argv, *, cwd, env=None, timeout=60):
        del cwd, env, timeout
        launch, profile, identity, bindings, lane_pids = _coordinator_launch_fixture(
            module,
            config,
            head=head,
            tree=tree,
            include_lane_heartbeats=True,
        )
        observed.update(
            profile=profile,
            identity=identity,
            bindings=bindings,
            lane_pids=lane_pids,
            launch=launch,
        )
        return module.CommandResult(0, json.dumps(launch), "")

    launcher = module.ProcedureCompilerProgramLauncher(repo_root=REPO_ROOT, runner=runner)
    launcher.config = config
    _admit_test_owner(module, launcher, monkeypatch, head=head, tree=tree)
    monkeypatch.setattr(
        module,
        "_scheduler_launch_bindings",
        lambda _config: observed["bindings"],
    )

    class ExactProcessAdapter:
        def _identity(self, pid: int, profile: Any) -> Any:
            assert pid == observed["identity"].pid
            assert profile == observed["profile"]
            return observed["identity"]

    monkeypatch.setattr(module, "LinuxProcessAdapter", ExactProcessAdapter)
    observed_lane_pids: list[int] = []

    def lane_process_ready(**kwargs: Any) -> bool:
        assert kwargs["coordinator_pid"] == observed["identity"].pid
        assert kwargs["coordinator_start_ticks"] == observed["identity"].start_time_ticks
        assert kwargs["supervisor_pid"] in observed["lane_pids"]
        observed_lane_pids.append(kwargs["supervisor_pid"])
        return True

    monkeypatch.setattr(module, "_configured_lane_process_ready", lane_process_ready)

    receipt = launcher.supervisor_start()

    assert receipt["coordinator_launch_receipt_cid"] == observed["launch"]["receipt_cid"]
    assert receipt["coordinator_profile_id"] == observed["profile"].profile_id
    assert set(observed_lane_pids) == set(observed["lane_pids"])
    assert len(tuple(config.evidence_root.joinpath("program-launcher").iterdir())) == 1


def test_post_receipt_blocked_probe_fences_coordinator_and_retry_is_unblocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    head = "1" * 40
    tree = "2" * 40
    config = replace(
        _with_hermetic_extensions(module, _config(module), tmp_path),
        evidence_root=tmp_path / "evidence",
    )
    observed: dict[str, Any] = {}
    process_state = {"alive": False}
    runner_calls = 0

    def runner(argv, *, cwd, env=None, timeout=60):
        nonlocal runner_calls
        del argv, cwd, env, timeout
        runner_calls += 1
        process_state["alive"] = True
        launch, profile, identity, bindings, lane_pids = _coordinator_launch_fixture(
            module,
            config,
            head=head,
            tree=tree,
            include_lane_heartbeats=True,
        )
        observed.update(
            profile=profile,
            identity=identity,
            bindings=bindings,
            lane_pids=lane_pids,
        )
        return module.CommandResult(0, json.dumps(launch), "")

    launcher = module.ProcedureCompilerProgramLauncher(repo_root=REPO_ROOT, runner=runner)
    launcher.config = config
    launcher.sleeper = lambda _seconds: None
    _admit_test_owner(module, launcher, monkeypatch, head=head, tree=tree)
    monkeypatch.setattr(
        module,
        "_scheduler_launch_bindings",
        lambda _config: observed["bindings"],
    )
    monkeypatch.setattr(
        module,
        "_configured_lane_process_ready",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        launcher,
        "_probe",
        lambda *, tree: {"blocked_task_ids": ["PCPC-999"]},
    )

    class ExactProcessAdapter:
        def _identity(self, pid: int, profile: Any) -> Any:
            if not process_state["alive"]:
                raise ProcessLookupError(pid)
            assert pid == observed["identity"].pid
            assert profile == observed["profile"]
            return observed["identity"]

        def _stat(self, pid: int) -> tuple[int, int, int, int]:
            identity = observed["identity"]
            assert pid == identity.pid
            if not process_state["alive"]:
                raise ProcessLookupError(pid)
            return (
                identity.parent_pid,
                identity.process_group_id,
                identity.session_id,
                identity.start_time_ticks,
            )

        def _signal_exact(self, identity: Any, signum: int) -> None:
            assert identity == observed["identity"]
            assert signum == signal.SIGTERM
            process_state["alive"] = False

    monkeypatch.setattr(module, "LinuxProcessAdapter", ExactProcessAdapter)

    for expected_runner_calls in (1, 2):
        with pytest.raises(module.ProgramLaunchError) as raised:
            launcher.supervisor_start()

        assert raised.value.code == "supervisor_blocked"
        assert runner_calls == expected_runner_calls
        assert not process_state["alive"]
        assert not (config.state_root / "configured-board-master.pid").exists()
        assert not any(config.state_root.glob("configured-board-*.status.json"))
    assert not config.evidence_root.exists()


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


def test_bootstrap_owner_state_dir_admits_materialize_lock(
    tmp_path: Path,
) -> None:
    module = _load()
    state_dir = tmp_path / "quack-owner"
    module._admit_bootstrap_owner_state_dir(state_dir)
    assert state_dir.is_dir()

    lock = state_dir / "write-transaction.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    mutations = state_dir / "mutations"
    mutations.mkdir(mode=0o700)
    module._admit_bootstrap_owner_state_dir(state_dir)

    (state_dir / "isolation.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.ProgramLaunchError) as raised:
        module._admit_bootstrap_owner_state_dir(state_dir)
    assert raised.value.code == "orphaned_owner_state"


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
