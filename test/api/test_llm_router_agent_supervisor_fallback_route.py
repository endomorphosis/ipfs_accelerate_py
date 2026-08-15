"""Focused fail-closed contracts for the canonical implementation route."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    local_profile as local_profile_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints import (
    provider_attempt_store as provider_attempt_store_module,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
    export_local_profile_lifecycle_witness,
    initialize_local_profile,
    lifecycle_root_identity_did,
    revoke_local_profile,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.provider_attempt_store import (
    DurableProviderAttemptCAS,
    ProviderAttemptStoreError,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    grok_cli_runner as grok_cli_runner_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner import (
    _validate_quota_evidence_in_accepted_child,
    build_grok_quota_routed_agent_command,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    render_grok_failure_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


def _protected_effect_launch_context(
    *, workspace: str = "/tmp/work"
) -> dict[str, object]:
    def identity(value: object) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    image_id = llm_router.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID
    container_id = "sha256:" + "6" * 64
    container_name = "ipfs-accelerate-codex-123-" + "a" * 32
    docker_config = "/tmp/asref-codex-container-fixture/docker-config"
    cidfile = "/tmp/asref-codex-container-fixture/container.cid"
    provider_argv = [
        "/usr/local/bin/codex",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        workspace,
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="high"',
        "-",
    ]
    runtime_receipt = {
        "path": "/usr/bin/docker",
        "device": 1,
        "inode": 2,
        "mode": 0o100755,
        "uid": 0,
        "size": 3,
        "mtime_ns": 4,
        "ctime_ns": 5,
    }
    mount_receipt = [
        "type=bind,src=/usr,dst=/usr,readonly",
        "type=bind,src=/etc/ssl/certs,dst=/etc/ssl/certs,readonly",
        (
            "type=bind,src=/usr/bin/python3.12,"
            "dst=/opt/ipfs-task-tools/bin/python,readonly"
        ),
        f"type=bind,src={workspace},dst={workspace}",
        (
            "type=bind,src=/tmp/auth.json,"
            "dst=/opt/codex-home/auth.json,readonly"
        ),
    ]
    container_environment = {
        "BASH_ENV": "",
        "CODEX_HOME": "/opt/codex-home",
        "ENV": "",
        "HOME": "/opt/codex-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/opt/ipfs-task-tools/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "/opt/ipfs-validation-site-packages",
        "TERM": "dumb",
    }
    environment_receipt = {
        "docker_cli": dict(container_environment),
        "container": dict(container_environment),
    }
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    uid_text, gid_text = uid_gid.split(":", 1)
    overrides = [
        "BASH_ENV=",
        "CUDA_VISIBLE_DEVICES=-1",
        "ENV=",
        "LD_LIBRARY_PATH=",
        "LD_PRELOAD=",
        "LIBRARY_PATH=",
        "NVIDIA_DRIVER_CAPABILITIES=",
        "NVIDIA_REQUIRE_CUDA=",
        "NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS=",
        "NVIDIA_VISIBLE_DEVICES=void",
    ]
    create_argv = [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        docker_config,
        "create",
        "--pull=never",
        "--interactive",
        "--read-only",
        "--network=bridge",
        "--runtime=runc",
        "--entrypoint=/usr/bin/env",
        "--tmpfs",
        f"/tmp:rw,nosuid,nodev,noexec,mode=0700,uid={uid_text},gid={gid_text}",
        "--tmpfs",
        f"/var/tmp:rw,nosuid,nodev,noexec,mode=0700,uid={uid_text},gid={gid_text}",
        "--tmpfs",
        f"/opt/codex-home:rw,nosuid,nodev,noexec,mode=0700,uid={uid_text},gid={gid_text}",
        "--name",
        container_name,
        "--cidfile",
        cidfile,
        "--label",
        "ipfs_accelerate.codex_fallback_isolation=true",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=1024",
        "--user",
        uid_gid,
        "--workdir",
        workspace,
    ]
    for override in overrides:
        create_argv.extend(["--env", override])
    for mount in mount_receipt:
        create_argv.extend(["--mount", mount])
    inner = list(provider_argv)
    inner[6] = "danger-full-access"
    create_argv.extend(
        [
            image_id,
            "-i",
            *[
                f"{name}={value}"
                for name, value in sorted(container_environment.items())
            ],
            *inner,
        ]
    )
    start_argv = [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        docker_config,
        "start",
        "--attach",
        "--interactive",
        container_id.removeprefix("sha256:"),
    ]
    command_receipt = {
        "create_argv": create_argv,
        "start_argv": start_argv,
        "provider_argv": provider_argv,
    }
    cleanup_receipt = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-effect-cleanup@1"
        ),
        "lease_root": "/tmp/asref-codex-container-fixture",
        "docker_config": docker_config,
        "cidfile": cidfile,
        "provider_home": "/tmp/asref-codex-home-fixture",
        "prompt_path": "/tmp/asref-grok-prompt-fixture.txt",
        "watchdog_pid": 123,
        "watchdog_start_ticks": 456,
    }
    cleanup_receipt["receipt_id"] = identity(cleanup_receipt)
    return {
        "provider_id": "codex",
        "command_id": identity(command_receipt),
        "runtime_id": identity(runtime_receipt),
        "image_id": image_id,
        "mount_id": identity(mount_receipt),
        "environment_id": identity(environment_receipt),
        "cleanup_id": cleanup_receipt["receipt_id"],
        "container_name": container_name,
        "container_id": container_id,
        "runtime_receipt": runtime_receipt,
        "image_receipt": {
            "image_id": image_id,
            "image_label": llm_router.AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL,
        },
        "command_receipt": command_receipt,
        "mount_receipt": mount_receipt,
        "environment_receipt": environment_receipt,
        "cleanup_receipt": cleanup_receipt,
    }


def _effect_detail_identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _refresh_effect_detail_identities(
    context: dict[str, object],
) -> None:
    for identity_name, receipt_name in (
        ("runtime_id", "runtime_receipt"),
        ("command_id", "command_receipt"),
        ("mount_id", "mount_receipt"),
        ("environment_id", "environment_receipt"),
    ):
        context[identity_name] = _effect_detail_identity(context[receipt_name])


def _legacy_effect_launch_context(tag: str) -> dict[str, str]:
    def identity(value: str) -> str:
        return "sha256:" + hashlib.sha256(value.encode()).hexdigest()

    return {
        "provider_id": "codex",
        "command_id": identity(f"command:{tag}"),
        "runtime_id": identity(f"runtime:{tag}"),
        "image_id": identity(f"image:{tag}"),
        "mount_id": identity(f"mount:{tag}"),
        "environment_id": identity(f"environment:{tag}"),
        "container_name": f"test-container-{tag}",
        "container_id": identity(f"container:{tag}"),
    }


def _recorded_effect_inspection(
    launch: dict[str, object],
    timestamp: int,
    *,
    status: str,
    returncode: int = 17,
) -> dict[str, object]:
    present = status != "absent"
    return {
        "status": status,
        "inspection_runtime_id": launch["runtime_id"],
        "inspection_command_id": "sha256:" + "d" * 64,
        "observed_at_ms": timestamp,
        "provider_id": launch["provider_id"],
        "command_id": launch["command_id"],
        "runtime_id": launch["runtime_id"],
        "image_id": launch["image_id"],
        "mount_id": launch["mount_id"],
        "environment_id": launch["environment_id"],
        "container_name": launch["container_name"],
        "container_id": launch["container_id"] if present else "",
        "returncode": returncode if status == "exited" else None,
    }


def _quarantine_after_eight_dead_adopters(
    store: DurableProviderAttemptCAS,
    *,
    launch_context: dict[str, object] | dict[str, str],
) -> tuple[object, str, dict[str, object]]:
    values: dict[str, object] = {
        "logical_attempt_id": "attempt:eight-dead-adopters",
        "route_id": "route:quarantine",
        "decision_id": "decision:quarantine",
        "task_id": "task:quarantine",
        "worktree_id": "worktree:quarantine",
        "authorized": True,
    }
    started = store.reserve_or_adopt(
        **values,
        launch_context=launch_context,
        effect_owner_id="sha256:" + "0" * 64,
        now_ms=1_000,
    )
    assert started.launch_authorized
    current = started.reservation
    stale_capability = started.completion_capability
    for generation in range(1, 9):
        adopted = store.adopt_effect(
            current,
            effect_owner_id="sha256:" + format(generation, "x") * 64,
            now_ms=1_000 + generation,
        )
        assert adopted.adoption_authorized
        assert not adopted.launch_authorized
        assert adopted.reservation.effect_adoption_generation == generation
        current = adopted.reservation
        stale_capability = adopted.completion_capability

    quarantined = store.adopt_effect(
        current,
        effect_owner_id="sha256:" + "9" * 64,
        now_ms=1_009,
    )
    assert quarantined.reservation.state == "quarantined"
    assert quarantined.reservation.effect_adoption_generation == 8
    assert not quarantined.adoption_authorized
    assert not quarantined.launch_authorized
    return quarantined.reservation, stale_capability, values


def _live_cleanup_launch_context() -> tuple[
    dict[str, object], dict[str, Path]
]:
    lease_root = Path(tempfile.mkdtemp(prefix="asref-codex-container-"))
    lease_root.chmod(0o700)
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(mode=0o700)
    cidfile = lease_root / "container.cid"
    cidfile.write_text("6" * 64, encoding="ascii")
    cidfile.chmod(0o600)
    provider_home = Path(tempfile.mkdtemp(prefix="asref-codex-home-"))
    provider_home.chmod(0o700)
    prompt_descriptor, prompt_name = tempfile.mkstemp(
        prefix="asref-grok-prompt-", suffix=".txt"
    )
    os.close(prompt_descriptor)
    prompt_path = Path(prompt_name)
    prompt_path.chmod(0o600)

    context = _protected_effect_launch_context()
    command = context["command_receipt"]
    assert isinstance(command, dict)
    for name in ("create_argv", "start_argv"):
        argv = command[name]
        assert isinstance(argv, list)
        command[name] = [
            str(docker_config)
            if item == "/tmp/asref-codex-container-fixture/docker-config"
            else (
                str(cidfile)
                if item == "/tmp/asref-codex-container-fixture/container.cid"
                else item
            )
            for item in argv
        ]
    cleanup_receipt: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-effect-cleanup@1"
        ),
        "lease_root": str(lease_root),
        "docker_config": str(docker_config),
        "cidfile": str(cidfile),
        "provider_home": str(provider_home),
        "prompt_path": str(prompt_path),
        "watchdog_pid": 999_999_999,
        "watchdog_start_ticks": 1,
    }
    cleanup_receipt["receipt_id"] = _effect_detail_identity(
        cleanup_receipt
    )
    context["cleanup_receipt"] = cleanup_receipt
    context["cleanup_id"] = cleanup_receipt["receipt_id"]
    _refresh_effect_detail_identities(context)
    assert llm_router._agent_effect_launch_details_valid(context)
    return context, {
        "lease_root": lease_root,
        "docker_config": docker_config,
        "cidfile": cidfile,
        "provider_home": provider_home,
        "prompt_path": prompt_path,
    }


def _discard_live_cleanup_inputs(paths: dict[str, Path]) -> None:
    shutil.rmtree(paths["lease_root"], ignore_errors=True)
    shutil.rmtree(paths["provider_home"], ignore_errors=True)
    try:
        paths["prompt_path"].unlink()
    except FileNotFoundError:
        pass


@pytest.mark.parametrize(
    "attack",
    [
        "privileged_flag",
        "arbitrary_required_mount_source",
        "writable_auth_mount",
        "duplicate_provider_model",
        "duplicate_docker_config",
        "alternate_start",
        "alternate_environment",
    ],
)
def test_protected_docker_receipt_rejects_semantic_argv_tampering(
    attack: str,
) -> None:
    workspace = "/tmp/protected-workspace"
    baseline = _protected_effect_launch_context(workspace=workspace)
    assert llm_router._agent_effect_launch_details_valid(
        baseline,
        workspace_path=workspace,
    )
    tampered = json.loads(json.dumps(baseline))
    command = tampered["command_receipt"]
    mounts = tampered["mount_receipt"]
    environment = tampered["environment_receipt"]
    assert isinstance(command, dict)
    assert isinstance(mounts, list)
    assert isinstance(environment, dict)
    create_argv = command["create_argv"]
    start_argv = command["start_argv"]
    provider_argv = command["provider_argv"]
    assert isinstance(create_argv, list)
    assert isinstance(start_argv, list)
    assert isinstance(provider_argv, list)

    if attack == "privileged_flag":
        image_index = create_argv.index(
            llm_router.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID
        )
        create_argv.insert(image_index, "--privileged")
    elif attack == "arbitrary_required_mount_source":
        original = next(
            item
            for item in mounts
            if item == "type=bind,src=/usr,dst=/usr,readonly"
        )
        substituted = (
            "type=bind,src=/tmp/attacker-usr,dst=/usr,readonly"
        )
        mounts[mounts.index(original)] = substituted
        create_argv[create_argv.index(original)] = substituted
    elif attack == "writable_auth_mount":
        original = next(
            item
            for item in mounts
            if "dst=/opt/codex-home/auth.json" in item
        )
        substituted = original.removesuffix(",readonly")
        mounts[mounts.index(original)] = substituted
        create_argv[create_argv.index(original)] = substituted
    elif attack == "duplicate_provider_model":
        provider_argv[-1:-1] = ["-m", "gpt-5.6-terra"]
        create_argv[-1:-1] = ["-m", "gpt-5.6-terra"]
    elif attack == "duplicate_docker_config":
        config_path = create_argv[create_argv.index("--config") + 1]
        create_index = create_argv.index("create")
        create_argv[create_index:create_index] = ["--config", config_path]
    elif attack == "alternate_start":
        start_argv[start_argv.index("--attach")] = "--detach"
    else:
        docker_cli = environment["docker_cli"]
        container = environment["container"]
        assert isinstance(docker_cli, dict)
        assert isinstance(container, dict)
        docker_cli["TERM"] = "xterm"
        container["TERM"] = "xterm"
        create_argv[create_argv.index("TERM=dumb")] = "TERM=xterm"

    _refresh_effect_detail_identities(tampered)
    assert not llm_router._agent_effect_launch_details_valid(
        tampered,
        workspace_path=workspace,
    )


def test_effect_owner_and_two_dead_adopters_terminalize_one_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def legacy_launch_context(tag: str) -> dict[str, str]:
        def identity(value: str) -> str:
            return "sha256:" + hashlib.sha256(value.encode()).hexdigest()

        return {
            "provider_id": "codex",
            "command_id": identity(f"command:{tag}"),
            "runtime_id": identity(f"runtime:{tag}"),
            "image_id": identity(f"image:{tag}"),
            "mount_id": identity(f"mount:{tag}"),
            "environment_id": identity(f"environment:{tag}"),
            "container_name": f"test-container-{tag}",
            "container_id": identity(f"container:{tag}"),
        }

    values = {
        "logical_attempt_id": "attempt:two-dead-adopters",
        "route_id": "route:one",
        "decision_id": "decision:one",
        "task_id": "task:one",
        "worktree_id": "worktree:one",
        "authorized": True,
    }
    owner_ids = ["sha256:" + character * 64 for character in "abc"]
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    started = store.reserve_or_adopt(
        **values,
        launch_context=legacy_launch_context("same-container"),
        effect_owner_id=owner_ids[0],
        now_ms=1_000,
    )
    assert started.launch_authorized
    original_launch = dict(started.reservation.effect_launch_receipt)

    monkeypatch.setattr(
        provider_attempt_store_module,
        "_process_identity_alive",
        lambda _pid, _start: False,
    )
    inspection_statuses = iter(("running", "exited"))
    inspected_container_ids: list[object] = []

    def inspect_same_container(
        launch: dict[str, object], timestamp: int
    ) -> dict[str, object]:
        status = next(inspection_statuses)
        inspected_container_ids.append(launch["container_id"])
        return {
            "status": status,
            "inspection_runtime_id": launch["runtime_id"],
            "inspection_command_id": "sha256:" + "d" * 64,
            "observed_at_ms": timestamp,
            "provider_id": launch["provider_id"],
            "command_id": launch["command_id"],
            "runtime_id": launch["runtime_id"],
            "image_id": launch["image_id"],
            "mount_id": launch["mount_id"],
            "environment_id": launch["environment_id"],
            "container_name": launch["container_name"],
            "container_id": launch["container_id"],
            "returncode": 17 if status == "exited" else None,
        }

    monkeypatch.setattr(
        provider_attempt_store_module,
        "_inspect_recorded_docker_effect",
        inspect_same_container,
    )
    adopter_one = store.adopt_effect(
        started.reservation,
        effect_owner_id=owner_ids[1],
        now_ms=1_001,
    )
    adopter_two = store.adopt_effect(
        adopter_one.reservation,
        effect_owner_id=owner_ids[2],
        now_ms=1_002,
    )
    assert adopter_one.adoption_authorized
    assert adopter_two.adoption_authorized
    assert not adopter_one.launch_authorized
    assert not adopter_two.launch_authorized
    assert adopter_two.reservation.effect_adoption_generation == 2
    assert adopter_two.reservation.reservation_id == (
        started.reservation.reservation_id
    )
    assert dict(adopter_two.reservation.effect_launch_receipt) == original_launch
    assert inspected_container_ids == [
        original_launch["container_id"],
        original_launch["container_id"],
    ]

    adoption_receipt = dict(adopter_two.reservation.effect_adoption_receipt)
    assert llm_router._agent_effect_adoption_receipt_valid(
        adoption_receipt,
        launch=original_launch,
        logical_attempt_id=started.reservation.logical_attempt_id,
        reservation_id=started.reservation.reservation_id,
    )
    forged = json.loads(json.dumps(adoption_receipt))
    forged_prior = forged["prior_adoption_receipts"][0]
    forged_prior["effect_owner_id"] = "sha256:" + "e" * 64
    forged_prior["receipt_id"] = _effect_detail_identity(
        {
            key: item
            for key, item in forged_prior.items()
            if key != "receipt_id"
        }
    )
    forged["receipt_id"] = _effect_detail_identity(
        {key: item for key, item in forged.items() if key != "receipt_id"}
    )
    assert not llm_router._agent_effect_adoption_receipt_valid(
        forged,
        launch=original_launch,
        logical_attempt_id=started.reservation.logical_attempt_id,
        reservation_id=started.reservation.reservation_id,
    )
    assert not provider_attempt_store_module._valid_effect_adoption_receipt(
        replace(
            adopter_two.reservation,
            effect_adoption_receipt=forged,
        )
    )

    terminal = store.complete(
        adopter_two.reservation,
        returncode=17,
        outcome={"result": "failed", "returncode": 17},
        completion_capability=adopter_two.completion_capability,
        effect_owner_id=owner_ids[2],
        now_ms=1_003,
    )
    replay = store.reserve_or_adopt(**values)
    assert terminal.terminal
    assert replay.reservation == terminal
    assert not replay.launch_authorized
    assert len(tuple((tmp_path / "attempts").glob("*.json"))) == 1


def test_recorded_effect_cleanup_is_reestablished_without_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "ipfs-accelerate-codex-123-" + "a" * 32
    removed: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_remove_exact_docker_container",
        lambda *, docker_bin, docker_config, container_name, settle_for_creation: (
            removed.append((docker_bin, str(docker_config), container_name))
        ),
    )
    with tempfile.TemporaryDirectory(
        prefix="asref-codex-container-"
    ) as raw_lease_root:
        lease_root = Path(raw_lease_root)
        lease_root.chmod(0o700)
        docker_config = lease_root / "docker-config"
        docker_config.mkdir(mode=0o700)
        cidfile = lease_root / "container.cid"
        cidfile.write_text("6" * 64, encoding="ascii")
        provider_home = Path(
            tempfile.mkdtemp(prefix="asref-codex-home-")
        )
        provider_home.chmod(0o700)
        prompt_fd, prompt_name = tempfile.mkstemp(
            prefix="asref-grok-prompt-", suffix=".txt"
        )
        os.close(prompt_fd)
        prompt_path = Path(prompt_name)
        prompt_path.chmod(0o600)
        cleanup_receipt = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "provider-effect-cleanup@1"
            ),
            "lease_root": str(lease_root),
            "docker_config": str(docker_config),
            "cidfile": str(cidfile),
            "provider_home": str(provider_home),
            "prompt_path": str(prompt_path),
            "watchdog_pid": 999_999_999,
            "watchdog_start_ticks": 1,
        }
        cleanup_receipt["receipt_id"] = _effect_detail_identity(
            cleanup_receipt
        )
        launch_receipt = {
            "container_name": container_name,
            "cleanup_id": cleanup_receipt["receipt_id"],
            "cleanup_receipt": cleanup_receipt,
            "runtime_receipt": {"path": "/usr/bin/docker"},
            "command_receipt": {
                "create_argv": [
                    "/usr/bin/docker",
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "create",
                    "--name",
                    container_name,
                    "--cidfile",
                    str(cidfile),
                ]
            },
        }
        terminal_marker = lease_root / "cas-terminal"
        assert not terminal_marker.exists()

        grok_cli_runner_module._release_recorded_codex_effect_cleanup(
            launch_receipt
        )
        assert removed == [
            ("/usr/bin/docker", str(docker_config), container_name)
        ]
        assert not lease_root.exists()
        assert not provider_home.exists()
        assert not prompt_path.exists()

        # A later terminal replay observes already-complete cleanup.
        grok_cli_runner_module._release_recorded_codex_effect_cleanup(
            launch_receipt
        )
        assert len(removed) == 1


@pytest.mark.parametrize(
    ("terminal_status", "terminal_returncode", "decision", "dispatched"),
    [
        ("exited", 17, "fallback_failed", True),
        ("absent", 125, "effect_not_created", False),
    ],
)
def test_eight_adoption_cap_can_only_terminalize_after_exact_reinspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    terminal_returncode: int,
    decision: str,
    dispatched: bool,
) -> None:
    launch_context, cleanup_paths = _live_cleanup_launch_context()
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    inspection_status = ["running"]
    inspected: list[tuple[object, int, str]] = []

    def inspect(
        launch: dict[str, object], timestamp: int
    ) -> dict[str, object]:
        inspected.append(
            (launch["container_id"], timestamp, inspection_status[0])
        )
        return _recorded_effect_inspection(
            launch,
            timestamp,
            status=inspection_status[0],
            returncode=17,
        )

    monkeypatch.setattr(
        provider_attempt_store_module,
        "_process_identity_alive",
        lambda _pid, _start: False,
    )
    monkeypatch.setattr(
        provider_attempt_store_module,
        "_inspect_recorded_docker_effect",
        inspect,
    )
    monkeypatch.setattr(
        grok_cli_runner_module,
        "_start_recorded_codex_effect",
        lambda *_args, **_kwargs: pytest.fail(
            "quarantine repair must never relaunch the provider"
        ),
    )
    removed: list[tuple[str, Path, str]] = []

    def remove_exact(
        *,
        docker_bin: str,
        docker_config: Path,
        container_name: str,
        settle_for_creation: bool,
    ) -> None:
        del settle_for_creation
        assert cleanup_paths["lease_root"].exists()
        assert cleanup_paths["provider_home"].exists()
        assert cleanup_paths["prompt_path"].exists()
        removed.append((docker_bin, docker_config, container_name))

    monkeypatch.setattr(
        grok_cli_runner_module,
        "_remove_exact_docker_container",
        remove_exact,
    )
    try:
        quarantined, stale_capability, _values = (
            _quarantine_after_eight_dead_adopters(
                store,
                launch_context=launch_context,
            )
        )
        assert quarantined.quarantine_receipt["inspection_status"] == "running"
        assert quarantined.quarantine_terminalization_receipt == {}
        assert len(inspected) == 9
        assert all(
            container_id
            == quarantined.effect_launch_receipt["container_id"]
            for container_id, _timestamp, _status in inspected
        )

        restarted = DurableProviderAttemptCAS(tmp_path / "attempts")
        durable = restarted.read(quarantined.logical_attempt_id)
        assert durable == quarantined
        inspection_status[0] = terminal_status
        repair_owner_id = "sha256:" + "a" * 64
        repair = restarted.claim_quarantined_terminalization(
            durable,
            effect_owner_id=repair_owner_id,
            now_ms=1_010,
        )
        assert repair.adoption_authorized
        assert not repair.launch_authorized
        assert repair.completion_capability
        assert repair.reservation.state == "quarantined"
        repair_receipt = (
            repair.reservation.quarantine_terminalization_receipt
        )
        assert repair_receipt["inspection_status"] == terminal_status
        assert repair_receipt["terminal_returncode"] == terminal_returncode
        assert repair_receipt["outcome_decision"] == decision
        assert repair_receipt["fallback_dispatched"] is dispatched
        assert inspected[-1] == (
            quarantined.effect_launch_receipt["container_id"],
            1_010,
            terminal_status,
        )
        outcome = {
            "decision": decision,
            "fallback_dispatched": dispatched,
            "fallback_returncode": terminal_returncode,
            "reservation_id": repair.reservation.reservation_id,
            "effect_launch_receipt": (
                repair.reservation.effect_launch_receipt
            ),
            "effect_adoption_receipt": (
                repair.reservation.effect_adoption_receipt
            ),
            "effect_quarantine_receipt": (
                repair.reservation.quarantine_receipt
            ),
            "effect_quarantine_terminalization_receipt": (
                repair.reservation.quarantine_terminalization_receipt
            ),
        }

        for invalid_capability in (stale_capability, "f" * 64):
            with pytest.raises(ProviderAttemptStoreError, match="effect winner"):
                restarted.complete(
                    repair.reservation,
                    returncode=terminal_returncode,
                    outcome=outcome,
                    completion_capability=invalid_capability,
                    effect_owner_id=repair_owner_id,
                    now_ms=1_011,
                )
            still_quarantined = restarted.read(
                quarantined.logical_attempt_id
            )
            assert still_quarantined == repair.reservation

        terminal = restarted.complete(
            repair.reservation,
            returncode=terminal_returncode,
            outcome=outcome,
            completion_capability=repair.completion_capability,
            effect_owner_id=repair_owner_id,
            now_ms=1_011,
        )
        assert terminal.terminal
        assert terminal.terminal_outcome == outcome
        grok_cli_runner_module._release_recorded_codex_effect_cleanup(
            terminal.effect_launch_receipt
        )
        assert len(removed) == 1
        assert not cleanup_paths["lease_root"].exists()
        assert not cleanup_paths["provider_home"].exists()
        assert not cleanup_paths["prompt_path"].exists()
    finally:
        _discard_live_cleanup_inputs(cleanup_paths)


@pytest.mark.parametrize("observed_status", ["created", "running"])
def test_live_quarantined_effect_remains_durably_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_status: str,
) -> None:
    status = ["running"]

    def inspect(
        launch: dict[str, object], timestamp: int
    ) -> dict[str, object]:
        return _recorded_effect_inspection(
            launch,
            timestamp,
            status=status[0],
        )

    monkeypatch.setattr(
        provider_attempt_store_module,
        "_process_identity_alive",
        lambda _pid, _start: False,
    )
    monkeypatch.setattr(
        provider_attempt_store_module,
        "_inspect_recorded_docker_effect",
        inspect,
    )
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    quarantined, stale_capability, _values = (
        _quarantine_after_eight_dead_adopters(
            store,
            launch_context=_legacy_effect_launch_context("quarantined-live"),
        )
    )
    status[0] = observed_status
    result = store.claim_quarantined_terminalization(
        quarantined,
        effect_owner_id="sha256:" + "a" * 64,
        now_ms=1_010,
    )
    assert not result.adoption_authorized
    assert not result.launch_authorized
    assert not result.completion_capability
    assert result.reservation == quarantined
    assert result.reservation.quarantine_terminalization_receipt == {}
    assert store.read(quarantined.logical_attempt_id) == quarantined
    with pytest.raises(ProviderAttemptStoreError):
        store.complete(
            quarantined,
            returncode=17,
            outcome={"returncode": 17},
            completion_capability=stale_capability,
            effect_owner_id="sha256:" + "8" * 64,
            now_ms=1_011,
        )
    assert store.read(quarantined.logical_attempt_id) == quarantined


def _terminal_watchdog_arguments(
    context: dict[str, object], paths: dict[str, Path]
) -> list[str]:
    return [
        "--provider",
        "codex",
        "--docker-bin",
        "/usr/bin/docker",
        "--container-name",
        str(context["container_name"]),
        "--cidfile",
        str(paths["cidfile"]),
        "--lease-root",
        str(paths["lease_root"]),
        "--provider-home",
        str(paths["provider_home"]),
        "--prompt-path",
        str(paths["prompt_path"]),
    ]


def _arm_terminal_watchdog_signal(
    context: dict[str, object],
    paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("cas-owned", "cas-terminal"):
        marker = paths["lease_root"] / name
        marker.write_text(str(context["container_name"]), encoding="ascii")
        marker.chmod(0o600)
    handlers: dict[int, object] = {}

    def capture_handler(signum: int, handler: object) -> None:
        handlers[signum] = handler

    class SignalBeforePoll:
        @property
        def buffer(self) -> SignalBeforePoll:
            return self

        def read(self, _maximum: int) -> bytes:
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)
            raise AssertionError("terminal signal handler did not exit")

    monkeypatch.setattr(
        grok_cli_runner_module.signal,
        "signal",
        capture_handler,
    )
    monkeypatch.setattr(
        grok_cli_runner_module.sys,
        "stdin",
        SignalBeforePoll(),
    )


def test_terminal_watchdog_signal_reaps_before_removing_cleanup_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths = _live_cleanup_launch_context()
    observed: list[tuple[bool, bool, bool, bool]] = []
    _arm_terminal_watchdog_signal(context, paths, monkeypatch)

    def remove_exact(**_kwargs: object) -> None:
        observed.append(
            (
                paths["docker_config"].exists(),
                paths["provider_home"].exists(),
                paths["prompt_path"].exists(),
                (paths["lease_root"] / "cas-terminal").exists(),
            )
        )

    monkeypatch.setattr(
        grok_cli_runner_module,
        "_remove_exact_docker_container",
        remove_exact,
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            grok_cli_runner_module._docker_cleanup_watchdog_main(
                _terminal_watchdog_arguments(context, paths)
            )
        assert stopped.value.code == 128 + signal.SIGTERM
        assert observed == [(True, True, True, True)]
        assert not paths["lease_root"].exists()
        assert not paths["provider_home"].exists()
        assert not paths["prompt_path"].exists()
    finally:
        _discard_live_cleanup_inputs(paths)


def test_terminal_watchdog_signal_preserves_inputs_when_docker_rm_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, paths = _live_cleanup_launch_context()
    observed: list[bool] = []
    _arm_terminal_watchdog_signal(context, paths, monkeypatch)

    def failed_remove(**_kwargs: object) -> None:
        observed.append(paths["docker_config"].exists())
        raise ValueError("exact Docker rm returned nonzero")

    monkeypatch.setattr(
        grok_cli_runner_module,
        "_remove_exact_docker_container",
        failed_remove,
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            grok_cli_runner_module._docker_cleanup_watchdog_main(
                _terminal_watchdog_arguments(context, paths)
            )
        assert stopped.value.code == 125
        assert observed == [True]
        assert paths["lease_root"].exists()
        assert paths["docker_config"].exists()
        assert paths["provider_home"].exists()
        assert paths["prompt_path"].exists()
        assert (paths["lease_root"] / "cas-owned").exists()
        assert (paths["lease_root"] / "cas-terminal").exists()
    finally:
        _discard_live_cleanup_inputs(paths)

_BOARD = "agent-supervisor-prompt-only-self-improvement-v3"
_ARTIFACT = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "provider_fallback_policy_authorization_20260808.json"
)
_ROUTE_ID = (
    "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
)
_ROOT_PIN = llm_router._V3_AGENT_LIFECYCLE_ROOT_PIN_PATH
_WITNESS = (
    "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
    "local_profile_lifecycle_witness_20260808.json"
)


@pytest.fixture(autouse=True)
def _isolated_lifecycle_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_profile_module,
        "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE",
        tmp_path / "root-registry",
    )


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sign(key: Ed25519PrivateKey, payload: dict[str, Any]) -> str:
    return base64.b64encode(key.sign(_canonical(payload))).decode("ascii")


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _test_control_plane_capsule(
    tmp_path: Path,
    *,
    source_head: str,
    source_tree: str,
) -> llm_router.AgentImplementationControlPlanePin:
    """Build a private inert capsule for signature/equality unit tests."""

    root = tmp_path / "accepted-control-plane"
    payloads = {
        relative: (f"# inert accepted test source: {relative}\n").encode()
        for relative in llm_router._AGENT_CONTROL_PLANE_RELATIVE_FILES
    }
    digests: dict[str, str] = {}
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o400)
        digests[relative] = "sha256:" + hashlib.sha256(payload).hexdigest()
    manifest: dict[str, Any] = {
        "schema": llm_router._AGENT_CONTROL_PLANE_MANIFEST_SCHEMA,
        "source_head": source_head,
        "source_tree": source_tree,
        "files": dict(sorted(digests.items())),
    }
    manifest["capsule_id"] = llm_router._content_addressed_mapping(
        manifest,
        identity_field="capsule_id",
    )
    manifest_path = root / llm_router._AGENT_CONTROL_PLANE_MANIFEST_FILENAME
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    manifest_path.chmod(0o400)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o500)
    root.chmod(0o500)
    return llm_router.build_agent_implementation_control_plane_pin(
        runner_path=(
            root
            / "ipfs_accelerate_py"
            / "agent_supervisor"
            / "runtime"
            / "grok_cli_runner.py"
        ),
        capsule_root=root,
    )


def _reviewed_route(
    tmp_path: Path,
    *,
    reviewer_provider: str = "local_operator",
    corrupt_static_signature: bool = False,
) -> tuple[
    Path,
    Ed25519PrivateKey,
    llm_router.AgentImplementationRoutePlan,
    llm_router.AgentImplementationInvocationBinding,
]:
    repository = tmp_path / "candidate"
    repository.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Route Test")
    (repository / "README").write_text("accepted source\n", encoding="utf-8")
    _git(repository, "add", "README")
    _git(repository, "-c", "commit.gpgsign=false", "commit", "-qm", "source")
    source_head = _git(repository, "rev-parse", "HEAD^{commit}")
    source_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_identity = ed25519_did_key(reviewer_key.public_key())
    route_fields: dict[str, Any] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.6",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
        "route_id": _ROUTE_ID,
        "allowed_trigger_classes": [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ],
    }
    profile_dir = tmp_path / "reviewer-profile"
    lifecycle_dir = tmp_path / "reviewer-lifecycle"
    profile = initialize_local_profile(
        repository_cid="repository:one",
        baseline_commit=source_head,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        signing_key=reviewer_key.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        ),
        effect_bounds=("edit", "isolated_worktree", "test"),
        budget_cid="budget:one",
        resource_cid="resource:one",
        route_id=_ROUTE_ID,
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_reasoning_effort="high",
    )
    root_identity_did = lifecycle_root_identity_did()
    pinned_at_ms = int(time.time()) * 1000
    root_pin: dict[str, Any] = {
        "schema": llm_router._AGENT_LIFECYCLE_ROOT_PIN_SCHEMA,
        "board_namespace": _BOARD,
        "base_head": source_head,
        "base_tree": source_tree,
        "root_identity_did": root_identity_did,
        "pinned_at_ms": pinned_at_ms,
    }
    root_pin["pin_id"] = llm_router._content_addressed_mapping(
        root_pin,
        identity_field="pin_id",
    )
    root_pin_path = repository / _ROOT_PIN
    root_pin_path.parent.mkdir(parents=True, exist_ok=True)
    root_pin_path.write_bytes(_canonical(root_pin))
    root_pin_path.chmod(0o644)
    _git(repository, "add", _ROOT_PIN)
    _git(
        repository,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "pin lifecycle root",
    )
    root_pin_sha256 = "sha256:" + hashlib.sha256(
        root_pin_path.read_bytes()
    ).hexdigest()
    witness_nonce = "witness:" + hashlib.sha256(
        str(repository).encode("utf-8")
    ).hexdigest()
    authorized_at_ms = int(time.time()) * 1000
    witness = export_local_profile_lifecycle_witness(
        repository_cid="repository:one",
        board_namespace=_BOARD,
        base_head=source_head,
        base_tree=source_tree,
        nonce=witness_nonce,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        observed_at_ms=authorized_at_ms,
        expires_at_ms=authorized_at_ms + 10 * 60 * 1000,
    )
    witness_path = repository / _WITNESS
    witness_path.write_bytes(_canonical(witness))
    witness_path.chmod(0o644)
    witness_sha256 = "sha256:" + hashlib.sha256(
        witness_path.read_bytes()
    ).hexdigest()
    authority_bounds: dict[str, Any] = {
        "repository_cid": "repository:one",
        "baseline_commit": source_head,
        "effects": ["edit", "isolated_worktree", "test"],
        "budget_cid": "budget:one",
        "resource_cid": "resource:one",
        "authority_cid": profile.content_id,
    }
    review_payload = llm_router.agent_implementation_route_review_payload(
        board_namespace=_BOARD,
        authorization_kind="explicit_operator_override",
        source_head=source_head,
        source_tree=source_tree,
        route=route_fields,
        authority_bounds=authority_bounds,
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        reviewer_profile_id=profile.profile_id,
        reviewer_profile_content_id=profile.content_id,
        reviewer_lifecycle_anchor_id=profile.lifecycle_anchor_id,
        reviewer_lifecycle_generation=profile.lifecycle_generation,
        reviewer_witness_path=_WITNESS,
        reviewer_witness_sha256=witness_sha256,
        lifecycle_root_identity_did=root_identity_did,
        lifecycle_witness_nonce=witness_nonce,
        lifecycle_root_pin_path=_ROOT_PIN,
        lifecycle_root_pin_sha256=root_pin_sha256,
        authorized_at_ms=authorized_at_ms,
        fallback_implementer_identity="codex",
    )
    signature = _sign(reviewer_key, review_payload)
    if corrupt_static_signature:
        signature = signature[:-2] + "AA"
    artifact = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-policy-authorization@2"
        ),
        "board_namespace": _BOARD,
        "authorization_source": {
            "kind": "explicit_operator_override",
            "source_head": source_head,
            "source_tree": source_tree,
            "prospective_only": True,
            "requires_descendant_tree": True,
        },
        "route": route_fields,
        "ownership_contract": {
            "canonical_route_plan_owner": "ipfs_accelerate_py.llm_router",
            "typed_fallback_decision_owner": "ipfs_accelerate_py.llm_router",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed": False,
        },
        "bootstrap_route_guarantees": {
            "explicit_codex_review_conflict_denied": True,
        },
        "reviewer": {
            "identity": reviewer_identity,
            "provider": reviewer_provider,
            "profile_id": profile.profile_id,
            "profile_content_id": profile.content_id,
            "lifecycle_anchor_id": profile.lifecycle_anchor_id,
            "generation": profile.lifecycle_generation,
            "witness_path": _WITNESS,
            "witness_sha256": witness_sha256,
            "signature": signature,
        },
        "authority_bounds": authority_bounds,
        "fallback_implementer_identity": "codex",
        "lifecycle_root_identity_did": root_identity_did,
        "lifecycle_witness_nonce": witness_nonce,
        "lifecycle_root_pin_path": _ROOT_PIN,
        "lifecycle_root_pin_sha256": root_pin_sha256,
        "authorized_at_ms": authorized_at_ms,
    }
    artifact_path = repository / _ARTIFACT
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(_canonical(artifact))
    artifact_path.chmod(0o644)
    _git(repository, "add", _ARTIFACT, _WITNESS)
    _git(
        repository,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "reviewed route",
    )
    digest = "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    authorization = llm_router.load_agent_implementation_route_authorization(
        repo_root=repository,
        artifact_path=_ARTIFACT,
        board_namespace=_BOARD,
        expected_sha256=digest,
    )
    route = llm_router.resolve_agent_implementation_route(
        primary_provider_id="grok_cli",
        primary_model_id="grok-4.6",
        fallback_provider_id="codex",
        fallback_model_id="gpt-5.6-terra",
        fallback_trigger="primary_quota_or_auth_unavailable",
        fallback_reasoning_effort="high",
        authorization=authorization,
    )

    control_plane = _test_control_plane_capsule(
        tmp_path,
        source_head=source_head,
        source_tree=source_tree,
    )
    baseline = _git(repository, "rev-parse", "HEAD^{commit}")
    attempt = 1
    task_id = "task:one"
    task_revision_cid = "task-revision:one"
    prompt_cid = "prompt:one"
    worktree_id = content_identity(
        {
            "workspace_path": str(repository.resolve()),
            "baseline_commit": baseline,
        }
    )
    logical_body = {
        "task_id": task_id,
        "task_revision_cid": task_revision_cid,
        "attempt": attempt,
        "prompt_cid": prompt_cid,
        "worktree_id": worktree_id,
        "route_id": route.route_id,
    }
    logical_attempt_id = content_identity(logical_body)
    invocation_id = content_identity(
        {**logical_body, "logical_attempt_id": logical_attempt_id}
    )
    issued_at_ms = int(time.time() * 1000)
    attempt_store, attempt_store_identity = (
        llm_router.bind_agent_implementation_attempt_store(
            tmp_path / "attempt-state",
            create=True,
        )
    )
    unsigned = llm_router.AgentImplementationInvocationBinding(
        schema=(
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-invocation@2"
        ),
        invocation_id=invocation_id,
        logical_attempt_id=logical_attempt_id,
        task_id=task_id,
        attempt=attempt,
        task_revision_cid=task_revision_cid,
        prompt_cid=prompt_cid,
        worktree_id=worktree_id,
        workspace_path=str(repository.resolve()),
        repository_cid="repository:one",
        baseline_commit=baseline,
        effects=("edit", "isolated_worktree", "test"),
        scope_cid="scope:one",
        budget_cid="budget:one",
        resource_cid="resource:one",
        authority_cid=profile.content_id,
        route_id=route.route_id,
        primary_provider_id=route.primary_provider_id,
        primary_model_id=route.primary_model_id,
        fallback_provider_id=route.fallback_provider_id,
        fallback_model_id=route.fallback_model_id,
        fallback_reasoning_effort=route.fallback_reasoning_effort,
        fallback_implementer_identity=route.fallback_implementer_identity,
        reviewer_identity=reviewer_identity,
        reviewer_provider=reviewer_provider,
        profile_id=profile.profile_id,
        profile_identity_did=profile.identity_did,
        profile_lifecycle_anchor_id=profile.lifecycle_anchor_id,
        profile_lifecycle_generation=profile.lifecycle_generation,
        profile_dir=str(profile_dir.resolve()),
        profile_lifecycle_dir=str(lifecycle_dir.resolve()),
        issued_at_ms=issued_at_ms,
        expires_at_ms=issued_at_ms + 60_000,
        provider_attempt_store=str(attempt_store),
        provider_attempt_store_identity=attempt_store_identity,
        control_plane=control_plane,
        reviewer_signature="pending",
    )
    invocation = replace(
        unsigned,
        reviewer_signature=_sign(reviewer_key, unsigned.signed_payload()),
    )
    bound = llm_router.bind_agent_implementation_route_invocation(
        route,
        invocation,
        repo_root=repository,
        workspace=repository,
        expected_binding=invocation.signed_payload(),
        now_ms=issued_at_ms,
        max_age_ms=60_000,
    )
    return repository, reviewer_key, bound, invocation


def test_high_route_never_accepts_an_ambient_six_field_profile() -> None:
    with pytest.raises(ValueError, match="authorization"):
        llm_router.resolve_agent_implementation_route(
            primary_provider_id="grok_cli",
            primary_model_id="grok-4.6",
            fallback_provider_id="codex",
            fallback_model_id="gpt-5.6-terra",
            fallback_trigger="primary_quota_or_auth_unavailable",
            fallback_reasoning_effort="high",
        )


def test_protected_bootstrap_artifact_is_denied_until_it_binds_a_reviewer() -> None:
    with pytest.raises(ValueError):
        llm_router.load_agent_implementation_route_authorization(
            repo_root=Path.cwd(),
            artifact_path=(
                "data/agent_supervisor/prompt_only_self_improvement_v3/"
                "convergence/provider_fallback_policy_authorization_20260808.json"
            ),
            board_namespace="agent-supervisor-prompt-only-self-improvement-v3",
        )


def test_real_non_codex_signature_binds_the_complete_scoped_invocation(
    tmp_path: Path,
) -> None:
    repository, _key, route, invocation = _reviewed_route(tmp_path)
    nonce = "a" * 64
    receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="not signed in",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=1,
    )
    decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=receipt,
        expected_nonce=nonce,
        expected_model="grok-4.6",
        expected_probe_returncode=1,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=invocation.issued_at_ms,
        max_age_ms=60_000,
    )
    assert decision.authorized
    assert decision.fallback_model_id == "gpt-5.6-terra"
    assert decision.fallback_reasoning_effort == "high"
    assert decision.reviewer_provider == "local_operator"
    assert decision.invocation_binding_id == invocation.content_id
    assert decision.control_plane_id == invocation.control_plane.capsule_id
    outcome_route = route.as_outcome_dict()
    assert outcome_route["invocation_binding_id"] == invocation.content_id
    assert outcome_route["accepted_control_plane"] == (
        invocation.control_plane.as_dict()
    )
    assert len(_canonical(outcome_route)) < 4096


def test_scoped_native_quota_requires_live_lifecycle_signed_evidence(
    tmp_path: Path,
) -> None:
    repository, _key, route, invocation = _reviewed_route(tmp_path)
    now_ms = invocation.issued_at_ms
    nonce = "b" * 64
    receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text=(
            llm_router._AGENT_IMPLEMENTATION_SPENDING_LIMIT_MESSAGE
        ),
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        observed_at_ms=now_ms,
    )
    initial = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=receipt,
        expected_nonce=nonce,
        expected_model="grok-4.6",
        expected_probe_returncode=41,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=now_ms,
        max_age_ms=60_000,
    )
    assert not initial.authorized
    assert initial.requires_independent_quota_verification

    session_id = "f159e13e-462f-43bc-9da2-01bd0c1f5761"
    home = tmp_path / "native-verifier-home"
    session = home / "sessions" / session_id
    session.mkdir(parents=True, mode=0o700)

    def update(value: dict[str, object]) -> dict[str, object]:
        return {
            "method": "session/update",
            "params": {"sessionId": session_id, "update": value},
        }

    message = llm_router._AGENT_IMPLEMENTATION_SPENDING_LIMIT_MESSAGE
    events = [
        update(
            {
                "sessionUpdate": "retry_state",
                "type": "failed",
                "error_type": "api",
                "message": message,
                "_meta": {"modelId": "grok-4.6"},
            }
        ),
        update(
            {
                "sessionUpdate": "turn_completed",
                "stop_reason": "error",
                "agent_result": message,
                "_meta": {"modelId": "grok-4.6"},
            }
        ),
    ]
    transcript = session / "updates.jsonl"
    transcript.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events),
        encoding="utf-8",
    )
    transcript.chmod(0o600)
    summary = session / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "info": {"id": session_id},
                "current_model_id": "grok-4.6",
                "grok_home": str(home),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary.chmod(0o600)
    verifier_root = tmp_path / "verifier"
    verifier_workspace = verifier_root / "workspace"
    verifier_workspace.mkdir(parents=True, mode=0o700)
    prompt = verifier_root / "prompt.txt"
    prompt.write_text("Reply with exactly the single word OK.\n")
    prompt.chmod(0o600)
    grok = tmp_path / "grok"
    grok.write_text("#!/bin/sh\nexit 41\n")
    grok.chmod(0o700)
    command = [
        str(grok.resolve()),
        "--model",
        "grok-4.6",
        "--max-turns",
        "1",
        "--cwd",
        str(verifier_workspace.resolve()),
        "--permission-mode",
        "dontAsk",
        "--output-format",
        "streaming-json",
        "--no-plan",
        "--no-subagents",
        "--disable-web-search",
        "--no-memory",
        "--verbatim",
        "--tools",
        "",
        "--prompt-file",
        str(prompt.resolve()),
        "--session-id",
        session_id,
        "--disallowed-tools",
        llm_router.AGENT_IMPLEMENTATION_QUOTA_VERIFIER_DISALLOWED_TOOLS,
    ]
    evidence = _validate_quota_evidence_in_accepted_child(
        grok_home=home,
        expected_session_id=session_id,
        verifier_returncode=41,
        failure_receipt=receipt,
        invocation_binding=invocation,
        verifier_command=command,
        verifier_workspace=verifier_workspace,
        verifier_prompt_path=prompt,
        observed_at_ms=now_ms,
    )
    assert isinstance(evidence, llm_router.AgentImplementationQuotaEvidence)
    assert evidence.signer_process_pid != os.getpid()
    assert evidence.signer_parent_pid == os.getpid()
    assert evidence.signer_identity_did == invocation.profile_identity_did
    assert evidence.signer_signature
    authorized = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=receipt,
        expected_nonce=nonce,
        expected_model="grok-4.6",
        expected_probe_returncode=41,
        independent_quota_evidence=evidence,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=now_ms,
        max_age_ms=60_000,
    )
    assert authorized.authorized
    assert authorized.verifier_status == "confirmed_quota"

    store = DurableProviderAttemptCAS(
        invocation.provider_attempt_store,
        expected_directory_identity=(
            invocation.provider_attempt_store_identity
        ),
    )
    authorization_context = (
        llm_router.build_agent_implementation_effect_authorization_context(
            route=route,
            repo_root=repository,
            failure_receipt=receipt,
            decision=authorized,
            expected_nonce=nonce,
            expected_model="grok-4.6",
            expected_probe_returncode=41,
            quota_evidence=evidence,
        )
    )
    reserved = store.reserve_or_adopt(
        logical_attempt_id=invocation.logical_attempt_id,
        route_id=route.route_id,
        decision_id=authorized.content_id,
        task_id=invocation.task_id,
        worktree_id=invocation.worktree_id,
        authorized=True,
        authorization_context=authorization_context,
        now_ms=now_ms,
    )
    claim = store.claim_effect(
        reserved.reservation,
        launch_context=_protected_effect_launch_context(
            workspace=invocation.workspace_path
        ),
        now_ms=now_ms,
    )
    outcome = llm_router.build_agent_implementation_route_outcome(
        receipt=receipt,
        route=route,
        decision="fallback_failed",
        verifier_status="confirmed_quota",
        fallback_dispatched=True,
        fallback_returncode=17,
        decision_id=authorized.content_id,
        quota_evidence=evidence,
        reservation_id=claim.reservation.reservation_id,
        effect_launch_receipt=claim.reservation.effect_launch_receipt,
    )
    assert llm_router.valid_agent_implementation_route_outcome(
        outcome,
        receipt=receipt,
        route=route,
        runner_returncode=17,
    )
    terminal = store.complete(
        claim.reservation,
        returncode=17,
        outcome=outcome,
        completion_capability=claim.completion_capability,
        now_ms=now_ms + 1,
    )
    assert terminal.terminal_outcome == outcome

    command = [
        "python",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(
            route.as_binding_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    log_path = tmp_path / "protected-terminal.log"
    log_path.write_text(
        render_grok_failure_receipt(receipt)
        + "\n"
        + llm_router.render_agent_implementation_route_outcome(outcome)
        + "\n",
        encoding="utf-8",
    )
    log_path.chmod(0o600)
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = repository
    daemon.state_path = tmp_path / "missing-daemon-state.json"
    accepted_audit = daemon._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    )
    assert accepted_audit["route_outcome_id"] == outcome["outcome_id"]

    mismatched_terminal = llm_router.build_agent_implementation_route_outcome(
        receipt=receipt,
        route=route,
        decision="fallback_failed",
        verifier_status="confirmed_quota",
        fallback_dispatched=True,
        fallback_returncode=18,
        decision_id=authorized.content_id,
        quota_evidence=evidence,
        reservation_id=claim.reservation.reservation_id,
        effect_launch_receipt=claim.reservation.effect_launch_receipt,
    )
    log_path.write_text(
        render_grok_failure_receipt(receipt)
        + "\n"
        + llm_router.render_agent_implementation_route_outcome(
            mismatched_terminal
        )
        + "\n",
        encoding="utf-8",
    )
    assert not daemon._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=18,
    ).get("route_outcome_id")

    forged_outcome = dict(outcome)
    forged_evidence = dict(outcome["quota_evidence"])
    forged_evidence["signer_signature"] = "A" * 88
    forged_evidence["evidence_id"] = llm_router._content_addressed_mapping(
        forged_evidence,
        identity_field="evidence_id",
    )
    forged_outcome["quota_evidence"] = forged_evidence
    forged_outcome["quota_evidence_id"] = forged_evidence["evidence_id"]
    forged_outcome["outcome_id"] = llm_router._content_addressed_mapping(
        forged_outcome,
        identity_field="outcome_id",
    )
    assert not llm_router.valid_agent_implementation_route_outcome(
        forged_outcome,
        receipt=receipt,
        route=route,
        runner_returncode=17,
    )

    forged = replace(evidence, signer_signature="A" * 88)
    denied = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=receipt,
        expected_nonce=nonce,
        expected_model="grok-4.6",
        expected_probe_returncode=41,
        independent_quota_evidence=forged,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=now_ms,
        max_age_ms=60_000,
    )
    assert not denied.authorized

    revoke_local_profile(
        profile_dir=Path(invocation.profile_dir),
        lifecycle_dir=Path(invocation.profile_lifecycle_dir),
    )
    with pytest.raises(ValueError):
        llm_router.decide_agent_implementation_fallback(
            route,
            repo_root=repository,
            failure_receipt=receipt,
            expected_nonce=nonce,
            expected_model="grok-4.6",
            expected_probe_returncode=41,
            independent_quota_evidence=evidence,
            expected_invocation_binding=invocation.signed_payload(),
            now_ms=now_ms,
            max_age_ms=60_000,
        )


def test_router_outcome_binds_exact_quarantine_terminalization_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _key, route, invocation = _reviewed_route(tmp_path)
    now_ms = invocation.issued_at_ms
    nonce = "c" * 64
    receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="not signed in",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=1,
        observed_at_ms=now_ms,
    )
    decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=receipt,
        expected_nonce=nonce,
        expected_model="grok-4.6",
        expected_probe_returncode=1,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=now_ms,
        max_age_ms=60_000,
    )
    assert decision.authorized
    authorization_context = (
        llm_router.build_agent_implementation_effect_authorization_context(
            route=route,
            repo_root=repository,
            failure_receipt=receipt,
            decision=decision,
            expected_nonce=nonce,
            expected_model="grok-4.6",
            expected_probe_returncode=1,
        )
    )
    store = DurableProviderAttemptCAS(
        invocation.provider_attempt_store,
        expected_directory_identity=(
            invocation.provider_attempt_store_identity
        ),
    )
    status = ["running"]
    monkeypatch.setattr(
        provider_attempt_store_module,
        "_process_identity_alive",
        lambda _pid, _start: False,
    )
    monkeypatch.setattr(
        provider_attempt_store_module,
        "_inspect_recorded_docker_effect",
        lambda launch, timestamp: _recorded_effect_inspection(
            launch,
            timestamp,
            status=status[0],
            returncode=17,
        ),
    )
    started = store.reserve_or_adopt(
        logical_attempt_id=invocation.logical_attempt_id,
        route_id=route.route_id,
        decision_id=decision.content_id,
        task_id=invocation.task_id,
        worktree_id=invocation.worktree_id,
        authorized=True,
        authorization_context=authorization_context,
        launch_context=_protected_effect_launch_context(
            workspace=invocation.workspace_path
        ),
        effect_owner_id="sha256:" + "0" * 64,
        now_ms=now_ms,
    )
    current = started.reservation
    for generation in range(1, 9):
        adopted = store.adopt_effect(
            current,
            effect_owner_id=(
                "sha256:" + format(generation, "x") * 64
            ),
            now_ms=now_ms + generation,
        )
        assert adopted.adoption_authorized
        current = adopted.reservation
    quarantined = store.adopt_effect(
        current,
        effect_owner_id="sha256:" + "9" * 64,
        now_ms=now_ms + 9,
    ).reservation
    assert quarantined.state == "quarantined"
    status[0] = "exited"
    repair_owner = "sha256:" + "a" * 64
    repair = store.claim_quarantined_terminalization(
        quarantined,
        effect_owner_id=repair_owner,
        now_ms=now_ms + 10,
    )
    assert repair.adoption_authorized
    outcome = llm_router.build_agent_implementation_route_outcome(
        receipt=receipt,
        route=route,
        decision="fallback_failed",
        verifier_status=decision.verifier_status,
        fallback_dispatched=True,
        fallback_returncode=17,
        decision_id=decision.content_id,
        reservation_id=repair.reservation.reservation_id,
        effect_launch_receipt=repair.reservation.effect_launch_receipt,
        effect_adoption_receipt=repair.reservation.effect_adoption_receipt,
        effect_quarantine_receipt=repair.reservation.quarantine_receipt,
        effect_quarantine_terminalization_receipt=(
            repair.reservation.quarantine_terminalization_receipt
        ),
    )
    assert llm_router.valid_agent_implementation_route_outcome(
        outcome,
        receipt=receipt,
        route=route,
        runner_returncode=17,
    )
    rendered = llm_router.render_agent_implementation_route_outcome(outcome)
    assert len(rendered.encode("utf-8")) <= (
        llm_router._AGENT_IMPLEMENTATION_ROUTE_OUTCOME_MAX_BYTES
        + len(llm_router.AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX)
    )
    assert llm_router.extract_agent_implementation_route_outcomes(rendered) == (
        outcome,
    )

    changed_returncode = json.loads(json.dumps(outcome))
    changed_returncode["fallback_returncode"] = 18
    changed_returncode["outcome_id"] = llm_router._content_addressed_mapping(
        changed_returncode,
        identity_field="outcome_id",
    )
    assert not llm_router.valid_agent_implementation_route_outcome(
        changed_returncode,
        receipt=receipt,
        route=route,
        runner_returncode=18,
    )
    tampered_repair = json.loads(json.dumps(outcome))
    tampered_repair_receipt = tampered_repair[
        "effect_quarantine_terminalization_receipt"
    ]
    tampered_repair_receipt["container_returncode"] = 18
    tampered_repair_receipt["receipt_id"] = _effect_detail_identity(
        {
            key: item
            for key, item in tampered_repair_receipt.items()
            if key != "receipt_id"
        }
    )
    tampered_repair["outcome_id"] = llm_router._content_addressed_mapping(
        tampered_repair,
        identity_field="outcome_id",
    )
    assert not llm_router.valid_agent_implementation_route_outcome(
        tampered_repair,
        receipt=receipt,
        route=route,
        runner_returncode=17,
    )
    omitted_lineage = json.loads(json.dumps(outcome))
    omitted_lineage["effect_quarantine_receipt"] = {}
    omitted_lineage["effect_quarantine_terminalization_receipt"] = {}
    omitted_lineage["outcome_id"] = llm_router._content_addressed_mapping(
        omitted_lineage,
        identity_field="outcome_id",
    )
    with pytest.raises(ProviderAttemptStoreError, match="repair receipt"):
        store.complete(
            repair.reservation,
            returncode=17,
            outcome=omitted_lineage,
            completion_capability=repair.completion_capability,
            effect_owner_id=repair_owner,
            now_ms=now_ms + 11,
        )
    terminal = store.complete(
        repair.reservation,
        returncode=17,
        outcome=outcome,
        completion_capability=repair.completion_capability,
        effect_owner_id=repair_owner,
        # Completion may legitimately occur long after the once-only start
        # authority expired.  Historical validation is anchored at the CAS
        # claim, not at terminal wall clock.
        now_ms=now_ms + 10 * 60 * 1000,
    )
    assert terminal.terminal_outcome == outcome
    command = [
        "python",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(
            route.as_binding_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    log_path = tmp_path / "quarantine-terminal.log"
    log_path.write_text(
        render_grok_failure_receipt(receipt)
        + "\n"
        + llm_router.render_agent_implementation_route_outcome(outcome)
        + "\n",
        encoding="utf-8",
    )
    log_path.chmod(0o600)
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.repo_root = repository
    daemon.state_path = tmp_path / "missing-quarantine-daemon-state.json"
    audit = daemon._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    )
    assert audit["route_outcome_id"] == outcome["outcome_id"]
    # Accounting authority is self-contained in the CAS.  Removing the live
    # authorization generation after the effect cannot erase/reclassify a
    # completed attempt, while any new dispatch still fails closed.
    for relative in (_ARTIFACT, _WITNESS, _ROOT_PIN):
        (repository / relative).unlink()
    historical_audit = daemon._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    )
    assert historical_audit["route_outcome_id"] == outcome["outcome_id"]
    with pytest.raises(ValueError):
        llm_router.decide_agent_implementation_fallback(
            route,
            repo_root=repository,
            failure_receipt=receipt,
            expected_nonce=nonce,
            expected_model="grok-4.6",
            expected_probe_returncode=1,
            expected_invocation_binding=invocation.signed_payload(),
            now_ms=now_ms + 10 * 60 * 1000,
            max_age_ms=60_000,
        )
    log_path.write_text(
        render_grok_failure_receipt(receipt)
        + "\n"
        + llm_router.render_agent_implementation_route_outcome(
            tampered_repair
        )
        + "\n",
        encoding="utf-8",
    )
    assert not daemon._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    ).get("route_outcome_id")


def _capacity_observation(
    provider_id: str,
    *,
    observed_at_ms: int,
    healthy: bool = True,
    max_concurrency: int = 4,
    active_requests: int = 0,
) -> dict[str, object]:
    return {
        "provider_id": provider_id,
        "healthy": healthy,
        "quota_remaining": 100,
        "latency_ms": 25,
        "context_window_tokens": 131_072,
        "token_budget_remaining": 1_000_000,
        "max_concurrency": max_concurrency,
        "active_requests": active_requests,
        "capabilities": ["implementation"],
        "observed_at_ms": observed_at_ms,
        "retry_after_ms": 0,
        "available_concurrency": max(
            0, max_concurrency - active_requests
        ),
    }


def test_route_capacity_projection_selects_one_lane_without_dispatch_authority(
    tmp_path: Path,
) -> None:
    _repository, _key, route, invocation = _reviewed_route(tmp_path)
    now_ms = invocation.issued_at_ms
    primary = _capacity_observation(
        "grok",
        observed_at_ms=now_ms,
        max_concurrency=3,
        active_requests=1,
    )
    fallback = _capacity_observation(
        "codex_cli",
        observed_at_ms=now_ms,
        max_concurrency=8,
    )
    profile = llm_router.project_agent_implementation_route_capacity(
        route,
        observations=[fallback, primary],
        now_ms=now_ms,
        max_age_ms=60_000,
    )
    assert profile.provider_id == route.route_id
    assert profile.max_concurrency == 3
    assert profile.active_requests == 1
    assert profile.available_concurrency == 2
    assert profile.schedulable
    assert all(not lane.dispatch_authorized for lane in profile.lanes)
    snapshot = profile.as_compiler_snapshot()
    assert snapshot["profile_id"] == profile.profile_id
    assert snapshot["route_id"] == route.route_id
    assert snapshot["max_age_ms"] == 60_000
    assert snapshot["fresh_until_ms"] == now_ms + 60_000
    assert len(snapshot["lanes"]) == 2

    primary["healthy"] = False
    fallback_profile = llm_router.project_agent_implementation_route_capacity(
        route,
        observations=[primary, fallback],
        now_ms=now_ms,
        max_age_ms=60_000,
    )
    assert fallback_profile.max_concurrency == 8
    assert fallback_profile.available_concurrency == 8
    fallback_lane = next(
        lane
        for lane in fallback_profile.lanes
        if lane.role == "typed_fallback_capacity_only"
    )
    assert fallback_lane.capacity_available
    assert not fallback_lane.dispatch_authorized


def test_route_capacity_projection_rejects_malformed_raw_observations(
    tmp_path: Path,
) -> None:
    _repository, _key, route, invocation = _reviewed_route(tmp_path)
    now_ms = invocation.issued_at_ms
    primary = _capacity_observation("grok_cli", observed_at_ms=now_ms)
    fallback = _capacity_observation("codex", observed_at_ms=now_ms)
    malformed_outer: tuple[object, ...] = (primary, fallback)
    with pytest.raises(ValueError):
        llm_router.project_agent_implementation_route_capacity(
            route,
            observations=malformed_outer,  # type: ignore[arg-type]
            now_ms=now_ms,
            max_age_ms=60_000,
        )
    with pytest.raises(ValueError):
        llm_router.project_agent_implementation_route_capacity(
            route,
            observations=[primary],
            now_ms=now_ms,
            max_age_ms=60_000,
        )
    duplicate = dict(fallback)
    duplicate["provider_id"] = "grok"
    with pytest.raises(ValueError):
        llm_router.project_agent_implementation_route_capacity(
            route,
            observations=[primary, duplicate],
            now_ms=now_ms,
            max_age_ms=60_000,
        )
    tuple_capabilities = dict(fallback)
    tuple_capabilities["capabilities"] = ("implementation",)
    with pytest.raises(ValueError):
        llm_router.project_agent_implementation_route_capacity(
            route,
            observations=[primary, tuple_capabilities],
            now_ms=now_ms,
            max_age_ms=60_000,
        )
    epoch = dict(fallback)
    epoch["observed_at_ms"] = 0
    with pytest.raises(ValueError):
        llm_router.project_agent_implementation_route_capacity(
            route,
            observations=[primary, epoch],
            now_ms=now_ms,
            max_age_ms=60_000,
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("invocation_id", "invocation:other"),
        ("task_id", "task:other"),
        ("task_revision_cid", "task-revision:other"),
        ("prompt_cid", "prompt:other"),
        ("worktree_id", "worktree:other"),
        ("repository_cid", "repository:other"),
        ("baseline_commit", "b" * 40),
        ("effects", ["edit"]),
        ("scope_cid", "scope:other"),
        ("budget_cid", "budget:other"),
        ("resource_cid", "resource:other"),
        ("authority_cid", "authority:other"),
        ("route_id", "route:other"),
        ("primary_provider_id", "other-primary"),
        ("fallback_provider_id", "other-fallback"),
        ("fallback_model_id", "other-model"),
        ("fallback_reasoning_effort", "medium"),
        ("reviewer_identity", "did:key:zOther"),
        ("reviewer_provider", "other-reviewer"),
        ("provider_attempt_store", "/tmp/other-attempt-store"),
    ],
)
def test_router_denies_every_live_invocation_equality_mismatch(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    repository, _key, route, invocation = _reviewed_route(tmp_path)
    expected = invocation.signed_payload()
    expected[field] = replacement
    nonce = "b" * 64
    receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="not signed in",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=1,
    )
    decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=receipt,
        expected_nonce=nonce,
        expected_model="grok-4.6",
        expected_probe_returncode=1,
        expected_invocation_binding=expected,
        now_ms=invocation.issued_at_ms,
        max_age_ms=60_000,
    )
    assert not decision.authorized
    assert decision.reason_code == "signed_invocation_mismatch"


def test_reviewer_signature_and_role_cannot_be_forged(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="signature"):
        _reviewed_route(tmp_path / "bad-signature", corrupt_static_signature=True)
    with pytest.raises(ValueError, match="authority bounds|exact scoped route"):
        _reviewed_route(tmp_path / "codex-reviewer", reviewer_provider="codex")


def test_valid_signature_still_cannot_select_a_candidate_control_plane(
    tmp_path: Path,
) -> None:
    repository, reviewer_key, route, invocation = _reviewed_route(tmp_path)
    candidate_runner = repository / "grok_cli_runner.py"
    candidate_runner.write_text("raise RuntimeError('candidate')\n", encoding="utf-8")
    candidate_pin = replace(
        invocation.control_plane,
        runner_path=str(candidate_runner),
        runner_sha256=(
            "sha256:" + hashlib.sha256(candidate_runner.read_bytes()).hexdigest()
        ),
    )
    unsigned = replace(
        invocation,
        control_plane=candidate_pin,
        reviewer_signature="pending",
    )
    forged = replace(
        unsigned,
        reviewer_signature=_sign(reviewer_key, unsigned.signed_payload()),
    )
    with pytest.raises(ValueError, match="control-plane"):
        llm_router.bind_agent_implementation_route_invocation(
            replace(route, invocation_binding=None),
            forged,
            repo_root=repository,
            workspace=repository,
            expected_binding=forged.signed_payload(),
            now_ms=forged.issued_at_ms,
            max_age_ms=60_000,
        )


def test_signed_invocation_cannot_hide_a_symlinked_attempt_store(
    tmp_path: Path,
) -> None:
    repository, reviewer_key, route, invocation = _reviewed_route(tmp_path)
    actual_store = tmp_path / "actual-attempt-store"
    actual_store.mkdir(mode=0o700)
    linked_store = tmp_path / "linked-attempt-store"
    linked_store.symlink_to(actual_store, target_is_directory=True)
    unsigned = replace(
        invocation,
        provider_attempt_store=str(linked_store),
        reviewer_signature="pending",
    )
    linked = replace(
        unsigned,
        reviewer_signature=_sign(reviewer_key, unsigned.signed_payload()),
    )
    with pytest.raises(ValueError, match="state path|workspace/state"):
        llm_router.bind_agent_implementation_route_invocation(
            replace(route, invocation_binding=None),
            linked,
            repo_root=repository,
            workspace=repository,
            expected_binding=linked.signed_payload(),
            now_ms=linked.issued_at_ms,
            max_age_ms=60_000,
        )


def test_control_plane_capsule_binds_every_scoped_effect_dependency() -> None:
    dependencies = set(llm_router._AGENT_CONTROL_PLANE_RELATIVE_FILES)
    assert {
        "ipfs_accelerate_py/llm_router.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_environment.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_failure_policy.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    }.issubset(dependencies)
    accepted_root = Path(llm_router.__file__).resolve().parents[1]
    complete_tree = {
        str(path.relative_to(accepted_root))
        for path in llm_router._agent_control_plane_source_files(accepted_root)
    }
    assert {
        "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
        "ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_failure_policy.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_timeout.py",
    }.issubset(complete_tree)


def test_control_plane_digest_rejects_a_path_swap_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "accepted.py"
    replacement = tmp_path / "replacement.py"
    backup = tmp_path / "opened.py"
    target.write_bytes(b"accepted\n")
    replacement.write_bytes(b"attacker\n")
    real_read = os.read
    swapped = False

    def swapping_read(descriptor: int, maximum: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, maximum)
        if chunk and not swapped:
            swapped = True
            target.replace(backup)
            replacement.replace(target)
        return chunk

    monkeypatch.setattr(llm_router.os, "read", swapping_read)
    with pytest.raises(ValueError, match="immutable|unavailable"):
        llm_router._agent_file_digest(target)


def test_absolute_accepted_runner_ignores_incompatible_candidate_imports(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    package = candidate / "ipfs_accelerate_py"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('candidate package imported')\n",
        encoding="utf-8",
    )
    marker = tmp_path / "sitecustomize-ran"
    (candidate / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    accepted_root = Path(llm_router.__file__).resolve().parents[1]
    accepted_runner = (
        accepted_root
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "runtime"
        / "grok_cli_runner.py"
    )
    command = build_grok_quota_routed_agent_command(
        workspace=candidate,
        python_executable=sys.executable,
        enable_codex_fallback=False,
        accepted_runner_path=accepted_runner,
    )
    assert command[:3] == [sys.executable, "-I", str(accepted_runner)]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(candidate)
    completed = subprocess.run(
        [*command[:3], "--help"],
        cwd=candidate,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_sealed_control_plane_archive_survives_disk_substitution(
    tmp_path: Path,
) -> None:
    _repository, _key, _route, invocation = _reviewed_route(tmp_path)
    pin = invocation.control_plane
    sealed = llm_router.seal_agent_implementation_control_plane_capsule(pin)
    runner = Path(pin.runner_path)
    runner.chmod(0o600)
    runner.write_text("raise RuntimeError('substituted disk runner')\n")
    completed = subprocess.run(
        [sys.executable, "-I", sealed.executable_path, "--help"],
        pass_fds=(sealed.descriptor,),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    try:
        assert completed.returncode == 0, completed.stderr
        # The fixture capsule deliberately contains inert module bodies.  A
        # successful, silent launch proves the interpreter consumed the
        # sealed archive rather than the now-hostile pathname on disk.
        assert completed.stdout == ""
        assert "substituted disk runner" not in completed.stderr
        with pytest.raises(OSError):
            os.write(sealed.descriptor, b"attacker")
    finally:
        os.close(sealed.descriptor)
