from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    connect_allowlist_proxy,
    grok_cli_runner,
    worker_network,
)
from ipfs_accelerate_py.agent_supervisor.runtime.connect_allowlist_proxy import (
    parse_connect_authority,
    serve_connect_client,
)
from ipfs_accelerate_py.agent_supervisor.runtime.worker_network import (
    PROVIDER_HOSTNAME_ALLOWLISTS,
    WORKER_NETWORK_AUTHORIZATION_SCHEMA,
    WorkerNetworkProfile,
    derived_worker_network_name,
    diagnostic_network_arguments,
    load_worker_network_authorization,
    validate_provider_hostname,
    validate_provider_worker_command,
    validate_worker_network_inspection,
    worker_network_approval_cid,
    worker_network_authorization_relative_path,
)


def _cid(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("ascii")).hexdigest()


def _profile(
    *,
    provider: str,
    workspace: Path,
    container_name: str,
    lease_root: Path,
    index: int = 0,
) -> WorkerNetworkProfile:
    values = {
        "provider": provider,
        "docker_network": f"eaaef-worker-{index}",
        "proxy_endpoint": f"http://172.28.{index}.2:3128",
        "approval_identity": f"eaaef-network-approval:test-{index}",
        "effect_cid": _cid(f"effect-{index}"),
        "workspace": workspace,
        "container_name": container_name,
        "lease_id": lease_root.name,
        "lease_root": lease_root,
    }
    return WorkerNetworkProfile(
        **values,
        allowed_hostnames=PROVIDER_HOSTNAME_ALLOWLISTS[provider],
        approval_cid=worker_network_approval_cid(**values),
    )


def _verified_profile(
    tmp_path: Path,
    *,
    provider: str,
    workspace: Path,
    container_name: str,
    lease_root: Path,
) -> tuple[SimpleNamespace, WorkerNetworkProfile]:
    """Materialize and verify one real signed test authorization."""

    profile_dir = tmp_path / f"reviewer-profile-{provider}"
    profile_dir.mkdir(mode=0o700, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    reviewer_did = ed25519_did_key(key.public_key())
    worker_did = ed25519_did_key(Ed25519PrivateKey.generate().public_key())
    provider_did = ed25519_did_key(Ed25519PrivateKey.generate().public_key())
    now_ms = int(time.time() * 1000)
    invocation = SimpleNamespace(
        invocation_id=_cid(f"invocation-{provider}-{container_name}"),
        content_id=_cid(f"binding-{provider}-{container_name}"),
        logical_attempt_id=_cid(f"attempt-{provider}-{container_name}"),
        task_id="EAAEF-NETWORK-TEST",
        worktree_id=_cid(f"worktree-{workspace}"),
        route_id=_cid(f"route-{provider}"),
        profile_dir=str(profile_dir),
        reviewer_identity=reviewer_did,
        profile_identity_did=reviewer_did,
        expected_worker_principal_did=worker_did,
        expected_provider_principal_did=provider_did,
        primary_provider_id="grok_cli",
        fallback_provider_id="codex",
        expires_at_ms=now_ms + 120_000,
        control_plane=SimpleNamespace(capsule_id=_cid("capsule")),
    )
    unsigned = {
        "schema": WORKER_NETWORK_AUTHORIZATION_SCHEMA,
        "invocation_binding_id": invocation.content_id,
        "logical_attempt_id": invocation.logical_attempt_id,
        "task_id": invocation.task_id,
        "worktree_id": invocation.worktree_id,
        "control_plane_capsule_id": invocation.control_plane.capsule_id,
        "effect_cid": invocation.content_id,
        "provider": provider,
        "route_id": invocation.route_id,
        "workspace": str(workspace),
        "container_name": container_name,
        "lease_id": lease_root.name,
        "lease_root": str(lease_root),
        "docker_network": derived_worker_network_name(invocation.worktree_id),
        "docker_network_id": "b" * 64,
        "docker_network_internal": True,
        "proxy_endpoint": "http://172.28.0.2:3128",
        "proxy_container_id": "c" * 64,
        "proxy_image_id": "sha256:" + "d" * 64,
        "allowed_hostnames": list(PROVIDER_HOSTNAME_ALLOWLISTS[provider]),
        "issued_at_ms": now_ms - 1_000,
        "expires_at_ms": now_ms + 60_000,
        "one_use_nonce": "network-nonce:0123456789abcdef",
        "signer_did": reviewer_did,
        "worker_principal_did": worker_did,
        "provider_principal_did": provider_did,
    }
    authorization_id = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    signed = {**unsigned, "authorization_id": authorization_id}
    record = {
        **signed,
        "signature": base64.b64encode(
            key.sign(
                json.dumps(
                    signed,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
        ).decode("ascii"),
    }
    path = profile_dir / worker_network_authorization_relative_path(
        invocation.invocation_id,
        provider,
    )
    path.parent.mkdir(parents=True)
    path.parent.parent.chmod(0o700)
    path.parent.chmod(0o700)
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)
    profile = grok_cli_runner._signed_worker_network_profile(
        invocation_binding=invocation,
        provider=provider,
        workspace=workspace,
    )
    return invocation, profile


def _fallback_command(codex: Path, workspace: Path) -> list[str]:
    return [
        str(codex),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]


def test_codex_worker_uses_only_internal_network_and_exact_proxy(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "worktree"
    lease_root = tmp_path / "asref-codex-container-lease0"
    docker_config = lease_root / "docker-config"
    workspace.mkdir()
    docker_config.mkdir(parents=True)
    auth = tmp_path / "auth.json"
    auth.write_text("not-read-by-test\n", encoding="utf-8")
    auth.chmod(0o600)
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    container_name = "ipfs-accelerate-codex-1-" + "a" * 32
    profile = _profile(
        provider="codex",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )

    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=_fallback_command(codex, workspace),
        workspace=workspace,
        source_auth=auth,
        child_env=grok_cli_runner._codex_task_container_environment(),
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
        network_profile=profile,
    )

    assert "--network=bridge" not in command
    assert "--network=host" not in command
    assert "--network=none" not in command
    assert command.count("--network=eaaef-worker-0") == 1
    assert "--dns=127.0.0.1" in command
    for name, value in profile.proxy_environment().items():
        assert f"{name}={value}" in command
    mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
    assert all("docker.sock" not in value for value in mounts)
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert f"{os.getuid()}:{os.getgid()}" in command


def test_grok_worker_replaces_ambient_proxy_and_has_no_default_bridge(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "worktree"
    lease_root = tmp_path / "asref-grok-container-lease0"
    docker_config = lease_root / "docker-config"
    grok_home = tmp_path / "grok-home"
    mask_root = lease_root / "masks"
    workspace.mkdir()
    docker_config.mkdir(parents=True)
    grok_home.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("task\n", encoding="utf-8")
    grok = tmp_path / "grok"
    grok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    grok.chmod(0o700)
    container_name = "ipfs-accelerate-grok-1-" + "e" * 32
    profile = _profile(
        provider="grok",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )

    command = grok_cli_runner._docker_grok_command(
        grok_command=[str(grok), "--prompt-file", str(prompt)],
        grok_bin=grok,
        workspace=workspace,
        prompt_path=prompt,
        grok_home=grok_home,
        base_env={},
        child_env={
            "PATH": "/usr/bin:/bin",
            "HTTP_PROXY": "http://attacker.invalid:9999",
            "FTP_PROXY": "http://attacker.invalid:9999",
        },
        denied_paths=(),
        mask_root=mask_root,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image="sha256:" + "f" * 64,
        network_profile=profile,
    )

    assert command.count("--network=eaaef-worker-0") == 1
    assert not any(value in command for value in ("--network=bridge", "--network=host"))
    assert not any("attacker.invalid" in value for value in command)
    assert "--env" in command
    assert "PATH" in command
    for name, value in profile.proxy_environment().items():
        assert f"{name}={value}" in command
    image = "sha256:" + "f" * 64
    image_index = command.index(image)
    with pytest.raises(ValueError, match="proxy environment is not exact"):
        validate_provider_worker_command(
            [
                *command[:image_index],
                "--env",
                "ALL_PROXY=http://172.28.0.9:9000",
                *command[image_index:],
            ],
            profile=profile,
            expected_image=image,
        )


def test_legacy_provider_commands_fail_closed_and_diagnostics_default_none(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "worktree"
    lease_root = tmp_path / "asref-codex-container-lease0"
    workspace.mkdir()
    (lease_root / "docker-config").mkdir(parents=True)
    auth = tmp_path / "auth.json"
    auth.write_text("unread\n", encoding="utf-8")
    auth.chmod(0o600)
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\n", encoding="utf-8")
    codex.chmod(0o700)

    with pytest.raises(ValueError, match="approved worker network profile"):
        grok_cli_runner._docker_codex_fallback_command(
            codex_command=_fallback_command(codex, workspace),
            workspace=workspace,
            source_auth=auth,
            child_env=grok_cli_runner._codex_task_container_environment(),
            docker_config=lease_root / "docker-config",
            container_name="ipfs-accelerate-codex-1-" + "b" * 32,
            cidfile=lease_root / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
        )
    assert diagnostic_network_arguments() == ("--network=none",)


@pytest.mark.parametrize("network", ["bridge", "host", "none", "default"])
def test_provider_profiles_reject_default_docker_networks(
    tmp_path: Path,
    network: str,
) -> None:
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    lease_root = tmp_path / "asref-grok-container-lease0"
    lease_root.mkdir()
    values = {
        "provider": "grok",
        "docker_network": network,
        "proxy_endpoint": "http://172.29.0.2:3128",
        "approval_identity": "eaaef-network-approval:test",
        "effect_cid": _cid("effect"),
        "workspace": workspace,
        "container_name": "ipfs-accelerate-grok-1-" + "c" * 32,
        "lease_id": lease_root.name,
        "lease_root": lease_root,
    }
    with pytest.raises(ValueError, match="dedicated Docker network"):
        WorkerNetworkProfile(
            **values,
            allowed_hostnames=PROVIDER_HOSTNAME_ALLOWLISTS["grok"],
            approval_cid=worker_network_approval_cid(**values),
        )


def test_profiles_reject_destination_and_proxy_substitution(tmp_path: Path) -> None:
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    lease_root = tmp_path / "asref-grok-container-lease0"
    lease_root.mkdir()
    profile = _profile(
        provider="grok",
        workspace=workspace,
        container_name="ipfs-accelerate-grok-1-" + "d" * 32,
        lease_root=lease_root,
    )
    assert validate_provider_hostname(profile, "api.x.ai") == "api.x.ai"
    for destination in ("example.com", "API.X.AI", "api.x.ai.", "1.1.1.1"):
        with pytest.raises(ValueError):
            validate_provider_hostname(profile, destination)

    values = {
        "provider": "grok",
        "docker_network": "eaaef-worker-0",
        "proxy_endpoint": "http://proxy.internal:3128",
        "approval_identity": "eaaef-network-approval:test",
        "effect_cid": _cid("effect"),
        "workspace": workspace,
        "container_name": "ipfs-accelerate-grok-1-" + "d" * 32,
        "lease_id": lease_root.name,
        "lease_root": lease_root,
    }
    with pytest.raises(ValueError, match="literal RFC1918"):
        worker_network_approval_cid(**values)


def test_five_worker_profiles_bind_disjoint_effects(tmp_path: Path) -> None:
    profiles: list[WorkerNetworkProfile] = []
    for index in range(5):
        workspace = tmp_path / f"worktree-{index}"
        lease_root = tmp_path / f"asref-codex-container-lease{index}"
        workspace.mkdir()
        lease_root.mkdir()
        profiles.append(
            _profile(
                provider="codex",
                workspace=workspace,
                container_name=(f"ipfs-accelerate-codex-{index + 1}-" + f"{index + 1:x}" * 32),
                lease_root=lease_root,
                index=index,
            )
        )

    assert len({profile.workspace for profile in profiles}) == 5
    assert len({profile.container_name for profile in profiles}) == 5
    assert len({profile.lease_id for profile in profiles}) == 5
    assert len({profile.lease_root for profile in profiles}) == 5
    assert len({profile.approval_cid for profile in profiles}) == 5
    for index, profile in enumerate(profiles):
        command = [
            "docker",
            "create",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=1024",
            "--cpus=4",
            "--memory=16g",
            "--memory-swap=16g",
            "--user",
            "1000:1000",
            "--name",
            profile.container_name,
            "--cidfile",
            str(profile.lease_root / "container.cid"),
            "--workdir",
            str(profile.workspace),
            "--label",
            "ipfs_accelerate.codex_fallback_isolation=true",
            *profile.docker_arguments(),
        ]
        for name, value in sorted(profile.proxy_environment().items()):
            command.extend(["--env", f"{name}={value}"])
        command.append("worker-image")
        validate_provider_worker_command(
            command, profile=profile, expected_image="worker-image"
        )
        assert f"--network=eaaef-worker-{index}" in command
        with pytest.raises(ValueError, match="does not bind"):
            profile.validate_effect_binding(
                provider="codex",
                workspace=profiles[(index + 1) % 5].workspace,
                container_name=profile.container_name,
                lease_root=tmp_path / profile.lease_id,
            )


def test_connect_proxy_enforces_allowlist_and_relays_without_internet() -> None:
    assert parse_connect_authority("api.x.ai:443", allowed_hostnames=("api.x.ai",)) == (
        "api.x.ai",
        443,
    )
    for authority in ("1.1.1.1:443", "example.com:443", "api.x.ai:80"):
        with pytest.raises(ValueError):
            parse_connect_authority(authority, allowed_hostnames=("api.x.ai",))

    client, proxy_side = socket.socketpair()
    upstream_proxy, upstream_server = socket.socketpair()

    def connector(hostname: str, port: int) -> socket.socket:
        assert (hostname, port) == ("api.x.ai", 443)
        return upstream_proxy

    thread = threading.Thread(
        target=serve_connect_client,
        kwargs={
            "client": proxy_side,
            "allowed_hostnames": ("api.x.ai",),
            "connector": connector,
        },
    )
    thread.start()
    client.sendall(b"CONNECT api.x.ai:443 HTTP/1.1\r\nHost: api.x.ai\r\n\r\n")
    assert client.recv(4096) == b"HTTP/1.1 200 Connection Established\r\n\r\n"
    client.sendall(b"local-only-test")
    assert upstream_server.recv(4096) == b"local-only-test"
    client.shutdown(socket.SHUT_WR)
    upstream_server.close()
    thread.join(timeout=2.0)
    client.close()
    assert not thread.is_alive()


def test_connect_proxy_rejects_multicast_and_bounds_idle_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        connect_allowlist_proxy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("224.0.0.1", 443),
            )
        ],
    )
    with pytest.raises(OSError, match="public address"):
        connect_allowlist_proxy._connect_global_upstream("api.x.ai", 443)

    client, proxy_side = socket.socketpair()
    monkeypatch.setattr(
        connect_allowlist_proxy, "CONNECT_HEADER_TIMEOUT_SECONDS", 0.01
    )
    with pytest.raises((OSError, TimeoutError)):
        connect_allowlist_proxy._read_connect_header(proxy_side)
    client.close()
    proxy_side.close()


def test_proxy_cli_has_no_arbitrary_production_allowlist() -> None:
    with pytest.raises(SystemExit):
        connect_allowlist_proxy.main(
            [
                "--listen-host", "0.0.0.0", "--listen-port", "3128",
                "--allow-host", "example.com",
            ]
        )


@pytest.mark.parametrize(
    "injected",
    [
        ("--user", "root:1000"),
        ("--user", "00:1000"),
        ("--net=host",),
        ("--memory=0",),
        ("--pids-limit=-1",),
        ("--volume=/var/run/docker.sock:/x",),
        ("--privileged",),
        ("--device=/dev/kvm",),
        ("--cap-add=NET_ADMIN",),
        ("--name", "ipfs-accelerate-codex-9-" + "f" * 32),
        ("--mount", "type=volume,src=credentials,dst=/credentials"),
    ],
)
def test_docker_boundary_rejects_alias_duplicate_and_privilege_bypasses(
    tmp_path: Path,
    injected: tuple[str, ...],
) -> None:
    workspace = tmp_path / "worktree"
    lease_root = tmp_path / "asref-codex-container-bypass"
    workspace.mkdir()
    lease_root.mkdir()
    profile = _profile(
        provider="codex",
        workspace=workspace,
        container_name="ipfs-accelerate-codex-1-" + "a" * 32,
        lease_root=lease_root,
    )
    command = [
        "docker", "create", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--pids-limit=1024", "--cpus=4",
        "--memory=16g", "--memory-swap=16g", "--user", "1000:1000",
        "--name", profile.container_name,
        "--cidfile", str(profile.lease_root / "container.cid"),
        "--workdir", str(profile.workspace),
        "--label", "ipfs_accelerate.codex_fallback_isolation=true",
        *profile.docker_arguments(),
    ]
    for name, value in sorted(profile.proxy_environment().items()):
        command.extend(["--env", f"{name}={value}"])
    command.extend([*injected, "worker-image"])
    with pytest.raises(ValueError):
        validate_provider_worker_command(
            command, profile=profile, expected_image="worker-image"
        )


def test_lane_name_derivation_and_network_inspection_reject_collisions() -> None:
    names = {derived_worker_network_name(f"worktree:{index}") for index in range(5)}
    assert len(names) == 5
    assert derived_worker_network_name("worktree:0") != derived_worker_network_name(
        "worktree:1"
    )
    proxy_id = "a" * 64
    authorization = SimpleNamespace(
        docker_network=derived_worker_network_name("worktree:0"),
        docker_network_id="b" * 64,
        proxy_endpoint="http://172.28.0.2:3128",
        proxy_container_id=proxy_id,
    )
    inspection = [
        {
            "Name": authorization.docker_network,
            "Id": authorization.docker_network_id,
            "Internal": True,
            "Ingress": False,
            "Driver": "bridge",
            "Scope": "local",
            "Attachable": False,
            "IPAM": {"Config": [{"Subnet": "172.28.0.0/24"}]},
            "Containers": {
                proxy_id: {"IPv4Address": "172.28.0.2/24"},
                "c" * 64: {"IPv4Address": "172.28.0.3/24"},
            },
        }
    ]
    with pytest.raises(ValueError, match="unexpected peer"):
        validate_worker_network_inspection(
            inspection, authorization=authorization
        )


def test_network_authorization_requires_fresh_trusted_signature_and_stable_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "worktree"
    profile_dir = tmp_path / "reviewer-profile"
    workspace.mkdir()
    profile_dir.mkdir(mode=0o700)
    key = Ed25519PrivateKey.generate()
    reviewer_did = ed25519_did_key(key.public_key())
    worker_did = ed25519_did_key(Ed25519PrivateKey.generate().public_key())
    provider_did = ed25519_did_key(Ed25519PrivateKey.generate().public_key())
    invocation_id = _cid("invocation")
    invocation_content_id = _cid("invocation-binding")
    lease_root = tmp_path / "asref-codex-container-signed"
    invocation = SimpleNamespace(
        invocation_id=invocation_id,
        content_id=invocation_content_id,
        logical_attempt_id=_cid("attempt"),
        task_id="EAAEF-101",
        worktree_id=_cid("worktree"),
        route_id=_cid("route"),
        profile_dir=str(profile_dir),
        reviewer_identity=reviewer_did,
        profile_identity_did=reviewer_did,
        primary_provider_id="grok_cli",
        fallback_provider_id="codex",
        expires_at_ms=2_000_000,
        control_plane=SimpleNamespace(capsule_id=_cid("capsule")),
    )
    unsigned = {
        "schema": WORKER_NETWORK_AUTHORIZATION_SCHEMA,
        "invocation_binding_id": invocation_content_id,
        "logical_attempt_id": invocation.logical_attempt_id,
        "task_id": invocation.task_id,
        "worktree_id": invocation.worktree_id,
        "control_plane_capsule_id": invocation.control_plane.capsule_id,
        "effect_cid": invocation_content_id,
        "provider": "codex",
        "route_id": invocation.route_id,
        "workspace": str(workspace),
        "container_name": "ipfs-accelerate-codex-1-" + "a" * 32,
        "lease_id": lease_root.name,
        "lease_root": str(lease_root),
        "docker_network": derived_worker_network_name(invocation.worktree_id),
        "docker_network_id": "b" * 64,
        "docker_network_internal": True,
        "proxy_endpoint": "http://172.28.0.2:3128",
        "proxy_container_id": "c" * 64,
        "proxy_image_id": "sha256:" + "d" * 64,
        "allowed_hostnames": list(PROVIDER_HOSTNAME_ALLOWLISTS["codex"]),
        "issued_at_ms": 1_000_000,
        "expires_at_ms": 1_100_000,
        "one_use_nonce": "network-nonce:0123456789abcdef",
        "signer_did": reviewer_did,
        "worker_principal_did": worker_did,
        "provider_principal_did": provider_did,
    }
    authorization_id = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    signed = {**unsigned, "authorization_id": authorization_id}
    signature = base64.b64encode(
        key.sign(
            json.dumps(
                signed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
    ).decode("ascii")
    record = {**signed, "signature": signature}
    path = profile_dir / worker_network_authorization_relative_path(
        invocation.invocation_id, "codex"
    )
    path.parent.mkdir(parents=True)
    path.parent.parent.chmod(0o700)
    path.parent.chmod(0o700)
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)

    verified = load_worker_network_authorization(
        invocation_binding=invocation,
        provider="codex",
        workspace=workspace,
        now_ms=1_050_000,
        expected_worker_principal_did=worker_did,
        expected_provider_principal_did=provider_did,
    )
    assert verified.authorization_id == authorization_id
    assert verified.signer_did == reviewer_did

    record["signature"] = base64.b64encode(b"0" * 64).decode("ascii")
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)
    with pytest.raises(ValueError, match="signature"):
        load_worker_network_authorization(
            invocation_binding=invocation,
            provider="codex",
            workspace=workspace,
            now_ms=1_050_000,
            expected_worker_principal_did=worker_did,
            expected_provider_principal_did=provider_did,
        )


def test_network_authorization_rejects_writable_or_renamed_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "worktree"
    lease_root = tmp_path / "asref-codex-container-parent-test"
    workspace.mkdir()
    invocation, profile = _verified_profile(
        tmp_path,
        provider="codex",
        workspace=workspace,
        container_name="ipfs-accelerate-codex-1-" + "a" * 32,
        lease_root=lease_root,
    )
    authorization = profile.authorization
    assert authorization is not None
    authorization_parent = authorization.source_path.parent
    authorization_parent.chmod(0o770)
    with pytest.raises(ValueError, match="parent is not trusted"):
        load_worker_network_authorization(
            invocation_binding=invocation,
            provider="codex",
            workspace=workspace,
            expected_worker_principal_did=(
                authorization.worker_principal_did
            ),
            expected_provider_principal_did=(
                authorization.provider_principal_did
            ),
        )
    authorization_parent.chmod(0o700)

    original_read = worker_network.os.read
    swapped = False

    def rename_parent_during_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, size)
        if not swapped:
            swapped = True
            moved = authorization_parent.with_name(
                authorization_parent.name + "-moved"
            )
            authorization_parent.rename(moved)
            authorization_parent.mkdir(mode=0o700)
        return data

    monkeypatch.setattr(worker_network.os, "read", rename_parent_during_read)
    with pytest.raises(ValueError, match="changed during read"):
        load_worker_network_authorization(
            invocation_binding=invocation,
            provider="codex",
            workspace=workspace,
            expected_worker_principal_did=(
                authorization.worker_principal_did
            ),
            expected_provider_principal_did=(
                authorization.provider_principal_did
            ),
        )
