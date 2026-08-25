from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control import provider_attempt_store
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.runtime import (
    worker_container_execution_profile as execution_profiles,
)
from ipfs_accelerate_py.agent_supervisor.runtime import worker_network_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime.worker_network import (
    PROVIDER_HOSTNAME_ALLOWLISTS,
    WorkerNetworkProfile,
    worker_network_approval_cid,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    implementation_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    external_agent_fabric_bootstrap as bootstrap,
)

from ipfs_accelerate_py import llm_router


def _cid(character: str) -> str:
    return "sha256:" + character * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _pin() -> SimpleNamespace:
    values = {
        "schema": "control-plane-pin@1",
        "runner_path": "/accepted/runner.py",
        "runner_sha256": "a" * 64,
        "capsule_root": "/accepted/capsule",
        "capsule_id": _cid("b"),
        "source_head": "c" * 40,
        "source_tree": "d" * 40,
        "archive_sha256": "e" * 64,
    }
    return SimpleNamespace(**values, as_dict=lambda: dict(values))


def _policy(*, status: str = "admitted") -> dict[str, object]:
    return {
        **dispatch._POLICY_SEMANTICS,  # noqa: SLF001 - exact protocol fixture
        "child_propagation_status": status,
    }


def _launch(*, status: str = "admitted") -> dict[str, object]:
    pin = _pin()
    return dispatch.build_worker_network_launch_authority(
        {
            "configured_board_capsule_cid": _cid("1"),
            "verification_cid": _cid("2"),
            "source_head": pin.source_head,
            "source_tree": pin.source_tree,
            "provider_worker_principal_did": "did:key:zworker",
            "provider_principal_did": "did:key:zprovider",
            "qualified_worker_image_digest": _cid("3"),
            "qualified_worker_container_profile_cid": _cid("4"),
            "worker_network_authorization_policy": _policy(status=status),
        },
        accepted_control_plane_pin=pin,
        require_admitted=status == "admitted",
    )


def _signed_execution_profile(
    tmp_path: Path,
    *,
    now_ms: int | None = None,
    grok_sources: tuple[Path, Path, Path] | None = None,
    grok_mount_override: tuple[str, str, object] | None = None,
) -> tuple[
    dict[str, object],
    SimpleNamespace,
    execution_profiles.WorkerContainerExecutionProfile,
    dict[str, object],
]:
    now_ms = now_ms or int(time.time() * 1000)
    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_did = ed25519_did_key(reviewer_key.public_key())
    server = {
        "Platform": {"Name": "synthetic-rootless-engine"},
        "Version": "27.5.1",
    }
    profile_schema = (
        bootstrap.EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2
        if grok_sources is not None
        else bootstrap.EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA
    )
    profile: dict[str, object] = {
        "schema": profile_schema,
        "runtime": "oci",
        "workload_class": "agent_worker",
        "task_dispatch_admitted": True,
        "execution_mode": "rootless_engine",
        "rootless_supported": True,
        "daemon_identity_cid": grok_cli_runner._qualified_worker_daemon_identity(  # noqa: SLF001
            runtime="docker",
            server=server,
        ),
        "daemon_policy_cid": _cid("a"),
        "bootstrap_policy_cid": bootstrap.EAAEF_BOOTSTRAP_POLICY_CID,
        "rootful_fallback_admitted": False,
        "image_digest": _cid("3"),
        "rootless": True,
        "nonroot_user": "65532:65532",
        "read_only_base": True,
        "network_mode": "policy_proxy_only",
        "cap_drop": ["ALL"],
        "no_new_privileges": True,
        "pids_limit": 256,
        "cpu_limit": 2,
        "memory_limit_bytes": 4 * 1024**3,
        "disk_limit_bytes": 16 * 1024**3,
        "maximum_parallel_workers": 2,
        "maximum_parallel_containers": 3,
        "gpu": {"mode": "none", "device_ids": [], "memory_limit_bytes": 0},
        "privileged": False,
        "host_pid": False,
        "host_ipc": False,
        "devices": [],
        "docker_socket_mounted": False,
        "inherit_host_environment": False,
        "environment": dict(bootstrap._EXPECTED_CONTAINER_ENV),  # noqa: SLF001
        "mounts": [
            {
                "source_identity": _cid("8"),
                "target": "/workspace",
                "read_only": False,
                "kind": "worktree",
            },
            *(
                [
                    {
                        "source_identity": _cid("9"),
                        "target": "/opt/codex-home/auth.json",
                        "read_only": True,
                        "kind": "provider_auth",
                    }
                ]
                if grok_sources is None
                else []
            ),
        ],
        "image_qualification_cid": _cid("6"),
        "sbom_digest": _cid("7"),
        "toolchain_versions": {
            "python": "3.12.3",
            "codex": "0.147.0",
            "grok": "1.0.5",
        },
        "network_policy_cid": _cid("d"),
        "worker_principal_did": "did:key:zworker",
        "provider_principal_did": "did:key:zprovider",
        "reviewer_identity_did": reviewer_did,
        "reviewer_role": (
            bootstrap.EAAEF_WORKER_CONTAINER_PROFILE_REVIEWER_ROLE_V2
            if grok_sources is not None
            else "independent_container_security_reviewer"
        ),
        "reviewed_at_ms": now_ms - 1_000,
        "expires_at_ms": now_ms + 60_000,
    }
    if grok_sources is not None:
        prompt_path, policy_path, provider_home = grok_sources
        mounts = profile["mounts"]
        assert isinstance(mounts, list)
        mounts.extend(
            [
                {
                    "source_identity": (
                        execution_profiles.worker_container_execution_file_source_identity(
                            (provider_home / "auth.json").resolve(strict=True)
                        )
                    ),
                    "target": "/opt/codex-home/auth.json",
                    "read_only": True,
                    "kind": "provider_auth",
                },
                {
                    "source_identity": (
                        execution_profiles.worker_container_execution_file_source_identity(
                            prompt_path
                        )
                    ),
                    "target": execution_profiles.EAAEF_GROK_PROMPT_MOUNT_TARGET,
                    "read_only": True,
                    "kind": "grok_prompt",
                },
                {
                    "source_identity": (
                        execution_profiles.worker_container_execution_file_source_identity(
                            policy_path
                        )
                    ),
                    "target": execution_profiles.EAAEF_GROK_POLICY_MOUNT_TARGET,
                    "read_only": True,
                    "kind": "grok_policy",
                },
                {
                    "source_identity": (
                        execution_profiles.worker_container_execution_grok_provider_home_source_identity(
                            provider_home
                        )
                    ),
                    "target": (
                        execution_profiles.EAAEF_GROK_PROVIDER_HOME_MOUNT_TARGET
                    ),
                    "read_only": False,
                    "kind": "grok_provider_home",
                },
            ]
        )
        if grok_mount_override is not None:
            kind, field, override = grok_mount_override
            selected = next(item for item in mounts if item.get("kind") == kind)
            selected[field] = override
    resource = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "external-agent-worker-resource-profile@1"
        ),
        "pids_limit": profile["pids_limit"],
        "cpu_limit": profile["cpu_limit"],
        "memory_limit_bytes": profile["memory_limit_bytes"],
        "disk_limit_bytes": profile["disk_limit_bytes"],
        "maximum_parallel_workers": profile["maximum_parallel_workers"],
        "maximum_parallel_containers": profile["maximum_parallel_containers"],
        "gpu": profile["gpu"],
    }
    profile["resource_profile_cid"] = _content_id(resource)
    profile["reviewer_signature"] = base64.b64encode(
        reviewer_key.sign(
            bootstrap.eaaef_worker_container_profile_signing_bytes(profile)
        )
    ).decode("ascii")
    profile["profile_cid"] = _content_id(profile)

    pin = _pin()
    launch = dispatch.build_worker_network_launch_authority(
        {
            "configured_board_capsule_cid": _cid("1"),
            "verification_cid": _cid("2"),
            "source_head": pin.source_head,
            "source_tree": pin.source_tree,
            "provider_worker_principal_did": "did:key:zworker",
            "provider_principal_did": "did:key:zprovider",
            "qualified_worker_image_digest": profile["image_digest"],
            "qualified_worker_container_profile_cid": profile["profile_cid"],
            "worker_network_authorization_policy": _policy(),
        },
        accepted_control_plane_pin=pin,
    )
    profile_root = tmp_path / "reviewer-profile"
    artifact_path = profile_root / (
        execution_profiles.worker_container_execution_profile_relative_path(
            str(profile["profile_cid"])
        )
    )
    artifact_path.parent.mkdir(parents=True, mode=0o700)
    profile_root.chmod(0o700)
    artifact_path.parent.chmod(0o700)
    artifact: dict[str, object] = {
        "schema": (
            execution_profiles.EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2
            if grok_sources is not None
            else execution_profiles.EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA
        ),
        "source_head": launch["source_head"],
        "source_tree": launch["source_tree"],
        "accepted_control_plane_capsule_id": launch[
            "accepted_control_plane_capsule_id"
        ],
        "qualified_worker_image_digest": profile["image_digest"],
        "qualified_worker_container_profile_cid": profile["profile_cid"],
        "engine_endpoint": f"unix:///run/user/{os.geteuid()}/docker.sock",
        "profile": profile,
        "issued_at_ms": now_ms - 500,
        "expires_at_ms": now_ms + 30_000,
        "signer_identity_did": reviewer_did,
        "signer_role": (
            execution_profiles.EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2
            if grok_sources is not None
            else execution_profiles.EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE
        ),
    }
    artifact["signature"] = base64.b64encode(
        reviewer_key.sign(
            execution_profiles.worker_container_execution_profile_signing_bytes(
                artifact
            )
        )
    ).decode("ascii")
    artifact["artifact_cid"] = _content_id(artifact)
    artifact_path.write_bytes(_canonical(artifact))
    artifact_path.chmod(0o600)
    invocation = SimpleNamespace(
        control_plane=pin,
        profile_dir=str(profile_root),
        resource_cid=profile["profile_cid"],
        worktree_id=_cid("8"),
        expires_at_ms=now_ms + 45_000,
    )
    loaded = execution_profiles.load_worker_container_execution_profile(
        launch_authority=launch,
        invocation_binding=invocation,
        now_ms=now_ms,
    )
    return launch, invocation, loaded, server


def _grok_mount_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    prompt_path = tmp_path / "asref-grok-prompt-signed.txt"
    prompt_path.write_text("implement the signed task\n", encoding="utf-8")
    prompt_path.chmod(0o600)
    provider_home = tmp_path / "asref-grok-home-signed"
    provider_home.mkdir(mode=0o700)
    controls = {
        "alternate-provider-deny-sentinel": "provider isolation sentinel\n",
        "config.toml": "[cli]\nuse_leader = false\n",
        "sandbox.toml": "[profiles.ipfs_accelerate_isolated]\n",
    }
    for name, payload in controls.items():
        path = provider_home / name
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)
    source_home = tmp_path / "signed-source-grok-home"
    source_home.mkdir(mode=0o700)
    source_auth = source_home / "auth.json"
    source_auth.write_text('{"signed":"authority"}\n', encoding="utf-8")
    source_auth.chmod(0o600)
    (provider_home / "auth.json").symlink_to(source_auth)
    return prompt_path, provider_home / "sandbox.toml", provider_home


def _invocation(workspace: Path) -> SimpleNamespace:
    return SimpleNamespace(
        content_id=_cid("5"),
        invocation_id=_cid("6"),
        logical_attempt_id=_cid("7"),
        task_id="EAAEF-TASK",
        worktree_id=_cid("8"),
        route_id="grok-auth-or-quota-codex-terra",
        workspace_path=str(workspace),
        control_plane=_pin(),
    )


def _network_profile(
    tmp_path: Path,
    *,
    provider: str,
    workspace: Path,
    container_name: str,
    lease_root: Path,
) -> WorkerNetworkProfile:
    values = {
        "provider": provider,
        "docker_network": "eaaef-worker-test",
        "proxy_endpoint": "http://172.30.0.2:3128",
        "approval_identity": "eaaef-network-approval:test",
        "effect_cid": _cid("e"),
        "workspace": workspace,
        "container_name": container_name,
        "lease_id": lease_root.name,
        "lease_root": lease_root,
    }
    del tmp_path
    return WorkerNetworkProfile(
        **values,
        allowed_hostnames=PROVIDER_HOSTNAME_ALLOWLISTS[provider],
        approval_cid=worker_network_approval_cid(**values),
    )


def test_launch_authority_is_path_free_and_binds_capsule_worker_profile() -> None:
    launch = _launch()

    assert launch["worker_principal_did"] == "did:key:zworker"
    assert launch["provider_principal_did"] == "did:key:zprovider"
    assert launch["qualified_worker_image_digest"] == _cid("3")
    assert launch["qualified_worker_container_profile_cid"] == _cid("4")
    encoded = dispatch.canonical_worker_network_launch_authority_json(
        launch,
        accepted_control_plane_pin=_pin(),
    )
    assert "/network-authorizations/" not in encoded
    assert '"profile_dir":' not in encoded


def test_unadmitted_launch_authority_stays_fail_closed() -> None:
    launch = _launch(status="unavailable_fail_closed")

    with pytest.raises(ValueError, match="not admitted"):
        dispatch.parse_worker_network_launch_authority(
            launch,
            accepted_control_plane_pin=_pin(),
            require_admitted=True,
        )


def test_attempt_uses_only_deterministic_relative_artifacts_and_exact_cids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invocation = _invocation(workspace)
    calls: list[dict[str, object]] = []

    def fake_load(**kwargs):
        calls.append(dict(kwargs))
        provider = str(kwargs["provider"])
        return SimpleNamespace(
            artifact_cid=_cid("a" if provider == "codex" else "b"),
            authorization_id=_cid("c" if provider == "codex" else "d"),
        )

    monkeypatch.setattr(dispatch, "load_worker_network_authorization", fake_load)
    attempt = dispatch.build_worker_network_attempt_authority(
        _launch(),
        invocation_binding=invocation,
        workspace=workspace,
        providers=("grok", "codex"),
    )
    entries = attempt["providers"]
    assert [item["provider"] for item in entries] == ["codex", "grok"]
    assert all(not Path(item["artifact_relative_path"]).is_absolute() for item in entries)
    assert all(
        item["artifact_relative_path"].startswith("network-authorizations/")
        for item in entries
    )
    assert all(call["expected_worker_principal_did"] == "did:key:zworker" for call in calls)
    assert all(call["expected_provider_principal_did"] == "did:key:zprovider" for call in calls)
    assert any(call.get("expected_artifact_cid") == _cid("a") for call in calls)
    assert any(call.get("expected_artifact_cid") == _cid("b") for call in calls)

    tampered = json.loads(json.dumps(attempt))
    tampered["providers"][0]["artifact_relative_path"] = "../../authority.json"
    body = {key: value for key, value in tampered.items() if key != "authority_cid"}
    tampered["authority_cid"] = dispatch._content_id(body)  # noqa: SLF001
    with pytest.raises(ValueError, match="source binding"):
        dispatch.verify_worker_network_attempt_authority(
            tampered,
            invocation_binding=invocation,
            workspace=workspace,
        )


def test_daemon_command_propagates_attempt_authority_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/opt/grok")
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        grok_cli_runner,
        "build_grok_quota_routed_agent_command",
        lambda **_kwargs: ["python", "runner.py", "--workspace", str(tmp_path)],
    )
    route = SimpleNamespace(
        invocation_binding=SimpleNamespace(content_id=_cid("9")),
        permits_authentication_unavailable=False,
        as_binding_dict=lambda: {"route": "bound"},
    )
    raw = '{"exact":"attempt"}'
    command = implementation_daemon._grok_cli_command(  # noqa: SLF001
        workspace_path=tmp_path,
        route_plan=route,
        enable_codex_fallback=False,
        worker_network_attempt_authority_json=raw,
    )

    index = command.index(dispatch.EAAEF_WORKER_NETWORK_ATTEMPT_AUTHORITY_FLAG)
    assert command[index + 1] == raw
    assert command.count(dispatch.EAAEF_WORKER_NETWORK_ATTEMPT_AUTHORITY_FLAG) == 1


@pytest.mark.parametrize("enable_fallback", [False, True])
def test_eaaef_command_construction_uses_only_in_image_provider_binaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enable_fallback: bool,
) -> None:
    sealed_runner = tmp_path / "sealed-runner.py"
    sealed_runner.write_text("# accepted generation\n", encoding="utf-8")
    route = SimpleNamespace(
        authorization=SimpleNamespace(board_namespace=dispatch.EAAEF_BOARD_NAMESPACE),
        invocation_binding=SimpleNamespace(content_id=_cid("9")),
        permits_authentication_unavailable=True,
        as_binding_dict=lambda: {"route": "bound"},
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: pytest.fail("EAAEF must not resolve a host Grok binary"),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: pytest.fail("EAAEF must not probe host Grok authentication"),
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda _name: pytest.fail("EAAEF must not resolve a host provider"),
    )

    command = implementation_daemon._grok_cli_command(  # noqa: SLF001
        workspace_path=tmp_path,
        route_plan=route,
        sealed_runner_path=str(sealed_runner),
        enable_codex_fallback=enable_fallback,
        allow_auth_unavailable_fallback=enable_fallback,
        worker_network_attempt_authority_json='{"exact":"attempt"}',
    )

    assert command[:3] == [
        implementation_daemon.sys.executable,
        "-I",
        str(sealed_runner),
    ]
    assert command[command.index("--grok-bin") + 1] == "/opt/eaaef/bin/grok"
    assert command[command.index("--agent-implementation-route-json") + 1]
    assert (
        dispatch.EAAEF_WORKER_NETWORK_ATTEMPT_AUTHORITY_FLAG in command
    )
    if enable_fallback:
        fallback = json.loads(
            command[command.index("--codex-fallback-command-json") + 1]
        )
        assert fallback[0] == "/opt/eaaef/bin/codex"
    else:
        assert "--codex-fallback-command-json" not in command


@pytest.mark.parametrize(
    ("grok_bin", "codex_bin", "expected"),
    [
        ("/host/bin/grok", "", "EAAEF Grok must use"),
        ("/opt/eaaef/bin/grok", "/host/bin/codex", "EAAEF Codex must use"),
    ],
)
def test_eaaef_command_builder_rejects_host_provider_paths(
    tmp_path: Path,
    grok_bin: str,
    codex_bin: str,
    expected: str,
) -> None:
    runner = tmp_path / "sealed-runner.py"
    runner.write_text("# accepted generation\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        grok_cli_runner.build_grok_quota_routed_agent_command(
            workspace=tmp_path,
            grok_bin=grok_bin,
            codex_bin=codex_bin,
            enable_codex_fallback=bool(codex_bin),
            accepted_runner_path=runner,
            eaaef_in_image_provider_binaries=True,
        )


@pytest.mark.parametrize(
    ("provider_role", "requires_preflight"),
    [("grok-only", False), ("grok-implement", True)],
)
def test_eaaef_capability_blocks_before_invocation_or_latch_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_role: str,
    requires_preflight: bool,
) -> None:
    route = SimpleNamespace(
        authorization=SimpleNamespace(board_namespace=dispatch.EAAEF_BOARD_NAMESPACE),
        invocation_binding=None,
        permits_authentication_unavailable=True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_configured_agent_implementation_route_plan",
        lambda _repo: route,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = object.__new__(implementation_daemon.PortalImplementationDaemon)
    daemon.repo_root = repo
    daemon.state_path = repo / "state/task_state.json"
    daemon.implementation_command = None
    daemon._launch_task_execution_route_binding = None
    daemon.manual_completion_authority_revalidation_only = False
    monkeypatch.setattr(
        daemon,
        "_require_implement_workspace_not_merge_target",
        lambda workspace_path, **_kwargs: workspace_path,
    )
    monkeypatch.setattr(
        daemon,
        "_protected_effect_recovery_command",
        lambda **_kwargs: None,
    )
    task = implementation_daemon.PortalTask(
        task_id="EAAEF-NO-LATCH-" + provider_role.upper(),
        title="Exercise protected primary",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["src/provider.py"],
        metadata={"Provider role": provider_role},
        canonical_task_cid=_cid("a"),
        board_namespace=dispatch.EAAEF_BOARD_NAMESPACE,
    )
    state = implementation_daemon.PortalTaskState()
    monkeypatch.setattr(
        daemon,
        "_bind_scoped_provider_invocation",
        lambda *_args, **_kwargs: pytest.fail(
            "unqualified primary capability must block before invocation signing"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_latch_protected_attempt",
        lambda *_args, **_kwargs: pytest.fail(
            "unqualified primary capability must not publish a latch"
        ),
    )

    for _retry in range(2):
        with pytest.raises(
            implementation_daemon.ImplementationRetryDeferred,
        ) as raised:
            daemon._build_implementation_command(  # noqa: SLF001
                repo,
                task=task,
                prompt="signed task prompt",
                attempt=1,
                state=state,
            )
        rendered = str(raised.value)
        assert "source-addressed-container-execution-profile-launch@1" not in rendered
        assert (
            grok_cli_runner.EAAEF_CONTAINER_EXECUTION_PROFILE_LAUNCH_STATUS
            == "implemented_unqualified_fail_closed"
        )
        assert "provider-neutral-effect-cas@1=unavailable_fail_closed" in rendered
        assert (
            "containerized-grok-preflight-receipt@1=unavailable_fail_closed"
            in rendered
        ) is requires_preflight
        assert state.protected_implementation_attempts == {}
        assert not daemon.state_path.exists()


def test_runner_parses_eaaef_primary_without_codex_then_blocks_before_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invocation = SimpleNamespace(
        prompt_cid=grok_cli_runner._agent_prompt_cid("signed prompt"),  # noqa: SLF001
        control_plane=_pin(),
        provider_attempt_store=str(tmp_path / "attempt-store"),
        provider_attempt_store_identity="",
        logical_attempt_id=_cid("7"),
    )
    route = SimpleNamespace(
        authorization=SimpleNamespace(board_namespace=dispatch.EAAEF_BOARD_NAMESPACE),
        invocation_binding=invocation,
    )
    observed: list[str] = []

    class EmptyAttemptStore:
        def __init__(self, *_args, **_kwargs) -> None:
            observed.append("attempt-store")

        def read(self, _logical_attempt_id: str) -> None:
            return None

    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: (observed.append("route") or route),
    )
    monkeypatch.setattr(
        llm_router,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin_value, _descriptor: "/proc/self/fd/71",
    )
    monkeypatch.setattr(
        provider_attempt_store,
        "DurableProviderAttemptCAS",
        EmptyAttemptStore,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "c" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_verified_worker_network_attempt",
        lambda *_args, **_kwargs: (
            observed.append("network") or {"grok": object()},
            _launch(),
        ),
    )
    monkeypatch.setattr(grok_cli_runner.sys, "argv", ["/proc/self/fd/71"])
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("signed prompt"))
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("provider process started"),
    )

    args = argparse.Namespace(
        codex_fallback_command_json="",
        codex_fallback_reasoning_effort="high",
        outer_runner_command=[],
        canonical_legacy_preflight_route=False,
        workspace=workspace,
        receipt_fd_declared=False,
        agent_implementation_recovery_json="",
        grok_failure_receipt_nonce="",
        agent_implementation_route_json="{}",
        grok_bin="/opt/eaaef/bin/grok",
        model="grok-4.6",
        require_command=[],
        worker_network_attempt_authority_json='{"bound":true}',
    )

    assert grok_cli_runner._run(args, -1) == 2  # noqa: SLF001
    assert observed == ["route", "attempt-store", "network"]
    rendered = capsys.readouterr().err
    assert (
        "source-addressed-container-execution-profile-launch@1="
        "artifact_missing_or_invalid"
    ) in rendered


def test_rootless_profile_cid_cannot_authorize_hardcoded_rootful_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _launch()
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "a profile CID must not select the hardcoded rootful Docker socket"
        ),
    )

    for inspect in (
        lambda: grok_cli_runner._inspect_qualified_worker_image(  # noqa: SLF001
            docker_bin="/usr/bin/docker",
            docker_config=tmp_path,
            launch_authority=launch,
        ),
        lambda: grok_cli_runner._inspect_qualified_worker_container(  # noqa: SLF001
            docker_bin="/usr/bin/docker",
            docker_config=tmp_path,
            container_id="f" * 64,
            launch_authority=launch,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="source-addressed-container-execution-profile-launch@1",
        ):
            inspect()


def test_source_addressed_execution_profile_loads_exact_signed_semantics(
    tmp_path: Path,
) -> None:
    launch, _invocation_binding, profile, _server = _signed_execution_profile(
        tmp_path
    )

    assert profile.profile_cid == launch["qualified_worker_container_profile_cid"]
    assert profile.image_digest == launch["qualified_worker_image_digest"]
    assert profile.engine_endpoint == (
        f"unix:///run/user/{os.geteuid()}/docker.sock"
    )
    assert profile.execution_mode == "rootless_engine"
    assert profile.pids_limit == 256
    assert profile.cpu_limit == 2
    assert profile.memory_limit_bytes == 4 * 1024**3
    assert profile.disk_limit_bytes == 16 * 1024**3
    assert profile.cap_drop == ("ALL",)
    assert profile.mount_for_kind("worktree").target == "/workspace"
    assert profile.mount_for_kind("provider_auth").read_only is True


def test_signed_grok_execution_profile_v2_projects_exact_mounts_without_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path, policy_path, provider_home = _grok_mount_sources(tmp_path)
    launch, _invocation_binding, profile, _server = _signed_execution_profile(
        tmp_path,
        grok_sources=(prompt_path, policy_path, provider_home),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-grok-container-signed"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    container_name = "ipfs-accelerate-grok-1-" + "a" * 32
    network_profile = _network_profile(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("container command ran"),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("provider process started"),
    )

    command = grok_cli_runner._docker_grok_command(  # noqa: SLF001
        grok_command=[
            "/opt/eaaef/bin/grok",
            "--cwd",
            str(workspace),
            "--prompt-file",
            str(prompt_path),
        ],
        grok_bin=Path("/opt/eaaef/bin/grok"),
        workspace=workspace,
        prompt_path=prompt_path,
        grok_home=provider_home,
        base_env={
            "GROK_HOME": str(
                (provider_home / "auth.json").resolve(strict=True).parent
            )
        },
        child_env=profile.container_environment(),
        denied_paths=(),
        mask_root=lease_root / "masks",
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=profile.image_digest,
        network_profile=network_profile,
        qualified_worker_launch_authority=launch,
        qualified_worker_execution_profile=profile,
    )

    assert profile.schema == (
        execution_profiles.EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2
    )
    assert profile.profile_schema == (
        bootstrap.EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2
    )
    observed_mounts = {
        value.split("dst=", maxsplit=1)[1].split(",", maxsplit=1)[0]: value
        for index, value in enumerate(command)
        if index > 0 and command[index - 1] == "--mount"
    }
    assert set(observed_mounts) == {
        "/workspace",
        "/opt/codex-home/auth.json",
        execution_profiles.EAAEF_GROK_PROMPT_MOUNT_TARGET,
        execution_profiles.EAAEF_GROK_POLICY_MOUNT_TARGET,
        execution_profiles.EAAEF_GROK_PROVIDER_HOME_MOUNT_TARGET,
    }
    assert observed_mounts[execution_profiles.EAAEF_GROK_PROMPT_MOUNT_TARGET].endswith(
        ",readonly"
    )
    assert observed_mounts[execution_profiles.EAAEF_GROK_POLICY_MOUNT_TARGET].endswith(
        ",readonly"
    )
    assert not observed_mounts[
        execution_profiles.EAAEF_GROK_PROVIDER_HOME_MOUNT_TARGET
    ].endswith(",readonly")
    image_index = command.index(profile.image_digest)
    inner = command[image_index + 1 :]
    assert inner[inner.index("--cwd") + 1] == "/workspace"
    assert inner[inner.index("--prompt-file") + 1] == (
        execution_profiles.EAAEF_GROK_PROMPT_MOUNT_TARGET
    )
    assert not (lease_root / "masks").exists()


def test_signed_grok_execution_profile_rejects_source_and_path_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path, policy_path, provider_home = _grok_mount_sources(tmp_path)
    launch, _invocation_binding, profile, _server = _signed_execution_profile(
        tmp_path,
        grok_sources=(prompt_path, policy_path, provider_home),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("container inspection ran"),
    )
    prompt_path.write_text("tampered\n", encoding="utf-8")
    prompt_path.chmod(0o600)
    with pytest.raises(ValueError, match="grok_prompt source identity drifted"):
        execution_profiles.reverify_worker_container_execution_grok_mounts(
            profile,
            launch_authority=launch,
            workspace=workspace,
            prompt_path=prompt_path,
            policy_path=policy_path,
            provider_home=provider_home,
        )

    prompt_path.write_text("implement the signed task\n", encoding="utf-8")
    prompt_path.chmod(0o600)
    prompt_alias = tmp_path / "prompt-alias.txt"
    prompt_alias.symlink_to(prompt_path)
    with pytest.raises(ValueError, match="source path is invalid"):
        execution_profiles.reverify_worker_container_execution_grok_mounts(
            profile,
            launch_authority=launch,
            workspace=workspace,
            prompt_path=prompt_alias,
            policy_path=policy_path,
            provider_home=provider_home,
        )


def test_signed_grok_restart_rejects_unrecorded_provider_home_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path, policy_path, provider_home = _grok_mount_sources(tmp_path)
    launch, _invocation_binding, profile, _server = _signed_execution_profile(
        tmp_path,
        grok_sources=(prompt_path, policy_path, provider_home),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("restart inspection ran"),
    )
    session_dir = provider_home / "sessions"
    session_dir.mkdir(mode=0o700)
    poisoned = session_dir / "unrecorded.json"
    poisoned.write_text('{"injected":true}\n', encoding="utf-8")
    poisoned.chmod(0o600)

    with pytest.raises(ValueError, match="grok_provider_home source identity drifted"):
        execution_profiles.reverify_worker_container_execution_grok_mounts(
            profile,
            launch_authority=launch,
            workspace=workspace,
            prompt_path=prompt_path,
            policy_path=policy_path,
            provider_home=provider_home,
        )


def test_signed_grok_mount_command_tamper_fails_before_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path, policy_path, provider_home = _grok_mount_sources(tmp_path)
    launch, _invocation_binding, profile, _server = _signed_execution_profile(
        tmp_path,
        grok_sources=(prompt_path, policy_path, provider_home),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-grok-container-signed"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    container_name = "ipfs-accelerate-grok-1-" + "b" * 32
    network_profile = _network_profile(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )
    command = grok_cli_runner._docker_grok_command(  # noqa: SLF001
        grok_command=[
            "/opt/eaaef/bin/grok",
            "--cwd",
            str(workspace),
            "--prompt-file",
            str(prompt_path),
        ],
        grok_bin=Path("/opt/eaaef/bin/grok"),
        workspace=workspace,
        prompt_path=prompt_path,
        grok_home=provider_home,
        base_env={
            "GROK_HOME": str(
                (provider_home / "auth.json").resolve(strict=True).parent
            )
        },
        child_env=profile.container_environment(),
        denied_paths=(),
        mask_root=lease_root / "masks",
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=profile.image_digest,
        network_profile=network_profile,
        qualified_worker_launch_authority=launch,
        qualified_worker_execution_profile=profile,
    )
    policy_mount_index = next(
        index
        for index, item in enumerate(command)
        if item == "--mount"
        and execution_profiles.EAAEF_GROK_POLICY_MOUNT_TARGET
        in command[index + 1]
    )
    command[policy_mount_index + 1] = command[policy_mount_index + 1].replace(
        execution_profiles.EAAEF_GROK_POLICY_MOUNT_TARGET,
        "/run/eaaef/tampered-policy.toml",
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("container command ran"),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("provider process started"),
    )

    with pytest.raises(ValueError, match="signed mount projection drifted"):
        grok_cli_runner._reverify_qualified_grok_mount_boundary(  # noqa: SLF001
            launch_authority=launch,
            execution_profile=profile,
            create_command=command,
            workspace=workspace,
        )


@pytest.mark.parametrize(
    ("kind", "field", "override"),
    (
        ("grok_prompt", "target", "/run/eaaef/grok/alternate.txt"),
        ("grok_policy", "read_only", False),
        ("grok_provider_home", "target", "/home/shared-grok"),
    ),
)
def test_signed_grok_profile_rejects_resigned_mount_widening(
    tmp_path: Path,
    kind: str,
    field: str,
    override: object,
) -> None:
    sources = _grok_mount_sources(tmp_path)

    with pytest.raises(ValueError, match="container_profile_invalid"):
        _signed_execution_profile(
            tmp_path,
            grok_sources=sources,
            grok_mount_override=(kind, field, override),
        )


def test_execution_profile_reverification_rejects_resigned_rootful_endpoint(
    tmp_path: Path,
) -> None:
    launch, _invocation_binding, profile, _server = _signed_execution_profile(
        tmp_path
    )
    artifact = json.loads(profile.source_path.read_text(encoding="utf-8"))
    artifact["engine_endpoint"] = "unix:///var/run/docker.sock"
    # Even without a valid replacement signature, endpoint drift is diagnosed
    # before a caller-selected rootful engine can be contacted.
    artifact.pop("artifact_cid")
    artifact["artifact_cid"] = _content_id(artifact)
    profile.source_path.write_bytes(_canonical(artifact))
    profile.source_path.chmod(0o600)

    with pytest.raises(ValueError, match="artifact CID drifted|binding is invalid"):
        execution_profiles.reverify_worker_container_execution_profile(
            profile,
            launch_authority=launch,
            now_ms=1_800_000_000_000,
        )


def test_qualified_inspection_reverifies_profile_and_uses_only_signed_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch, _invocation_binding, profile, server = _signed_execution_profile(
        tmp_path
    )
    calls: list[list[str]] = []

    def completed(command, **_kwargs):
        argv = [str(item) for item in command]
        calls.append(argv)
        assert argv[1] == f"--host={profile.engine_endpoint}"
        assert "unix:///var/run/docker.sock" not in argv
        if "version" in argv:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"Server": server}))
        if "info" in argv:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(["name=rootless"]),
            )
        if "image" in argv:
            return SimpleNamespace(returncode=0, stdout=profile.image_digest + "\n")
        endpoint_cid = "sha256:" + hashlib.sha256(
            profile.engine_endpoint.encode("utf-8")
        ).hexdigest()
        return SimpleNamespace(
            returncode=0,
            stdout=(
                f"{profile.image_digest}|{profile.image_digest}|"
                f"{profile.profile_cid}|{profile.artifact_cid}|"
                f"{profile.daemon_identity_cid}|{endpoint_cid}\n"
            ),
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", completed)

    assert (
        grok_cli_runner._inspect_qualified_worker_image(  # noqa: SLF001
            docker_bin="/usr/bin/docker",
            docker_config=tmp_path / "docker-config",
            launch_authority=launch,
            execution_profile=profile,
        )
        == profile.image_digest
    )
    grok_cli_runner._inspect_qualified_worker_container(  # noqa: SLF001
        docker_bin="/usr/bin/docker",
        docker_config=tmp_path / "docker-config",
        container_id="f" * 64,
        launch_authority=launch,
        execution_profile=profile,
    )
    assert [argv[4] for argv in calls] == [
        "version",
        "info",
        "image",
        "version",
        "info",
        "container",
    ]


def test_profile_labels_include_artifact_daemon_and_endpoint_identities(
    tmp_path: Path,
) -> None:
    launch, _invocation_binding, profile, _server = _signed_execution_profile(
        tmp_path
    )
    labels = grok_cli_runner._qualified_worker_container_label_arguments(  # noqa: SLF001
        launch,
        profile,
    )
    rendered = " ".join(labels)

    assert profile.profile_cid in rendered
    assert profile.artifact_cid in rendered
    assert profile.daemon_identity_cid in rendered
    assert grok_cli_runner._QUALIFIED_WORKER_EXECUTION_PROFILE_LABEL in rendered  # noqa: SLF001


def test_signed_network_inspection_uses_execution_profile_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch, _invocation_binding, execution_profile, _server = (
        _signed_execution_profile(tmp_path)
    )
    authorization = SimpleNamespace(
        docker_network_id="b" * 64,
        proxy_container_id="c" * 64,
        proxy_image_id=_cid("d"),
    )
    network_profile = SimpleNamespace(authorization=authorization)
    calls: list[list[str]] = []

    def completed(command, **_kwargs):
        argv = [str(item) for item in command]
        calls.append(argv)
        assert argv[1] == f"--host={execution_profile.engine_endpoint}"
        if "network" in argv:
            return SimpleNamespace(returncode=0, stdout=b"[]", stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=(authorization.proxy_image_id + "\n").encode("ascii"),
            stderr=b"",
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", completed)
    monkeypatch.setattr(
        grok_cli_runner,
        "validate_worker_network_inspection",
        lambda *_args, **_kwargs: None,
    )

    grok_cli_runner._inspect_signed_worker_network(  # noqa: SLF001
        docker_bin="/usr/bin/docker",
        docker_config=tmp_path / "docker-config",
        profile=network_profile,
        launch_authority=launch,
        execution_profile=execution_profile,
    )

    assert len(calls) == 2


def test_caller_selected_container_engine_endpoint_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="endpoint is not locally admitted"):
        grok_cli_runner._DockerContainerLease.create(  # noqa: SLF001
            "/caller/docker",
            provider="grok",
            provider_home=tmp_path / "home",
            prompt_path=tmp_path / "prompt",
            engine_endpoint="tcp://attacker.invalid:2375",
        )


def test_effect_receipt_label_binds_exact_launch_network_and_principals(
    tmp_path: Path,
) -> None:
    launch, _invocation_binding, execution_profile, _server = (
        _signed_execution_profile(tmp_path)
    )
    authorization = SimpleNamespace(
        artifact_cid=_cid("a"),
        authorization_id=_cid("b"),
        worker_principal_did="did:key:zworker",
        provider_principal_did="did:key:zprovider",
    )
    profile = SimpleNamespace(
        authorization=authorization,
        approval_cid=_cid("c"),
    )

    label = grok_cli_runner._eaaef_worker_effect_image_label(  # noqa: SLF001
        launch_authority=launch,
        network_profile=profile,
        execution_profile=execution_profile,
    )
    decoded = json.loads(label)
    assert decoded["launch_authority"]["authority_cid"] == launch["authority_cid"]
    assert decoded["network_authorization_artifact_cid"] == _cid("a")
    assert decoded["network_authorization_id"] == _cid("b")
    assert decoded["network_approval_cid"] == _cid("c")
    assert (
        decoded["container_execution_profile_artifact_cid"]
        == execution_profile.artifact_cid
    )
    assert (
        decoded["container_daemon_identity_cid"]
        == execution_profile.daemon_identity_cid
    )

    authorization.provider_principal_did = "did:key:zattacker"
    with pytest.raises(ValueError, match="principal binding"):
        grok_cli_runner._eaaef_worker_effect_image_label(  # noqa: SLF001
            launch_authority=launch,
            network_profile=profile,
            execution_profile=execution_profile,
        )


def test_mutable_qualified_worker_image_is_rejected() -> None:
    launch = _launch()
    launch["qualified_worker_image_digest"] = "worker:latest"

    with pytest.raises(ValueError, match="image/profile identity"):
        grok_cli_runner._qualified_worker_bounds(launch)  # noqa: SLF001


def test_receipt_fd_route_fails_before_any_host_provider_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("host provider process started"),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("host provider command ran"),
    )
    args = argparse.Namespace(
        codex_fallback_command_json="",
        codex_fallback_reasoning_effort="medium",
        outer_runner_command=[],
        canonical_legacy_preflight_route=False,
        grok_failure_receipt_nonce="",
        agent_implementation_route_json="",
        workspace=tmp_path,
        receipt_fd_declared=True,
    )

    assert grok_cli_runner._run(args, 9) == 2  # noqa: SLF001


def test_signed_execution_profile_projects_exact_codex_create_boundary(
    tmp_path: Path,
) -> None:
    launch, _invocation_binding, execution_profile, _server = (
        _signed_execution_profile(tmp_path)
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-codex-container-qualified"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    network_profile = _network_profile(
        tmp_path,
        provider="codex",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)

    command = grok_cli_runner._docker_codex_fallback_command(  # noqa: SLF001
        codex_command=[
            "/opt/eaaef/bin/codex",
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
            'model_reasoning_effort="high"',
            "-",
        ],
        workspace=workspace,
        source_auth=source_auth,
        child_env=execution_profile.container_environment(),
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=execution_profile.image_digest,
        network_profile=network_profile,
        qualified_worker_launch_authority=launch,
        qualified_worker_execution_profile=execution_profile,
    )

    assert command[1] == f"--host={execution_profile.engine_endpoint}"
    assert "--pids-limit=256" in command
    assert "--cpus=2" in command
    assert f"--memory={4 * 1024**3}" in command
    assert f"--memory-swap={4 * 1024**3}" in command
    assert f"--storage-opt=size={16 * 1024**3}" in command
    assert command[command.index("--user") + 1] == "65532:65532"
    assert command[command.index("--workdir") + 1] == "/workspace"
    mounts = {
        value.split("dst=", maxsplit=1)[1].split(",", maxsplit=1)[0]
        for index, value in enumerate(command)
        if index > 0 and command[index - 1] == "--mount"
    }
    assert mounts == {"/workspace", "/opt/codex-home/auth.json"}
    image_index = command.index(execution_profile.image_digest)
    provider_argv = command[image_index + 1 :]
    assert provider_argv[provider_argv.index("-C") + 1] == "/workspace"


def test_signed_execution_profile_resource_override_is_rejected(
    tmp_path: Path,
) -> None:
    launch, _invocation_binding, execution_profile, _server = (
        _signed_execution_profile(tmp_path)
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-codex-container-qualified"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    container_name = "ipfs-accelerate-codex-1-" + "c" * 32
    network_profile = _network_profile(
        tmp_path,
        provider="codex",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    command = grok_cli_runner._docker_codex_fallback_command(  # noqa: SLF001
        codex_command=[
            "/opt/eaaef/bin/codex",
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
            'model_reasoning_effort="high"',
            "-",
        ],
        workspace=workspace,
        source_auth=source_auth,
        child_env=execution_profile.container_environment(),
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=execution_profile.image_digest,
        network_profile=network_profile,
        qualified_worker_launch_authority=launch,
        qualified_worker_execution_profile=execution_profile,
    )
    command[command.index("--memory=4294967296")] = "--memory=8589934592"

    with pytest.raises(ValueError, match="unapproved Docker option|override"):
        grok_cli_runner.validate_provider_worker_command(
            command,
            profile=network_profile,
            expected_image=execution_profile.image_digest,
            additional_labels=(
                grok_cli_runner._qualified_worker_container_label_arguments(  # noqa: SLF001
                    launch,
                    execution_profile,
                )[1::2]
            ),
            container_execution_profile=execution_profile,
        )


def test_qualified_commands_require_full_source_addressed_profile_semantics(
    tmp_path: Path,
) -> None:
    launch = _launch()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("implement", encoding="utf-8")

    grok_lease = tmp_path / "asref-grok-container-qualified"
    grok_config = grok_lease / "docker-config"
    grok_config.mkdir(parents=True)
    grok_home = tmp_path / "asref-grok-home-qualified"
    grok_home.mkdir()
    grok_name = "ipfs-accelerate-grok-1-" + "a" * 32
    grok_profile = _network_profile(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name=grok_name,
        lease_root=grok_lease,
    )
    with pytest.raises(
        ValueError,
        match="source-addressed-container-execution-profile-launch@2",
    ):
        grok_cli_runner._docker_grok_command(  # noqa: SLF001
            grok_command=["/caller/chosen/grok", "--version"],
            grok_bin=Path("/caller/chosen/grok"),
            workspace=workspace,
            prompt_path=prompt,
            grok_home=grok_home,
            base_env={},
            child_env={"PATH": "/opt/eaaef/bin:/usr/bin:/bin"},
            denied_paths=(),
            mask_root=grok_lease / "masks",
            docker_config=grok_config,
            container_name=grok_name,
            cidfile=grok_lease / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=str(launch["qualified_worker_image_digest"]),
            network_profile=grok_profile,
            qualified_worker_launch_authority=launch,
        )
    assert not (grok_lease / "masks").exists()

    codex_lease = tmp_path / "asref-codex-container-qualified"
    codex_config = codex_lease / "docker-config"
    codex_config.mkdir(parents=True)
    codex_name = "ipfs-accelerate-codex-1-" + "b" * 32
    codex_profile = _network_profile(
        tmp_path,
        provider="codex",
        workspace=workspace,
        container_name=codex_name,
        lease_root=codex_lease,
    )
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    with pytest.raises(
        ValueError,
        match="source-addressed-container-execution-profile-launch@1",
    ):
        grok_cli_runner._docker_codex_fallback_command(  # noqa: SLF001
            codex_command=[
                "/opt/eaaef/bin/codex",
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
                'model_reasoning_effort="high"',
                "-",
            ],
            workspace=workspace,
            source_auth=source_auth,
            child_env=(
                grok_cli_runner._qualified_worker_container_environment()  # noqa: SLF001
            ),
            docker_config=codex_config,
            container_name=codex_name,
            cidfile=codex_lease / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=str(launch["qualified_worker_image_digest"]),
            network_profile=codex_profile,
            qualified_worker_launch_authority=launch,
        )


def test_eaaef_plan_bound_child_cannot_demote_to_portal_or_local_sidecar() -> None:
    with pytest.raises(
        implementation_supervisor.PlanBoundDispatchError,
        match="local Markdown/JSON/DuckDB",
    ):
        implementation_supervisor._reject_unsealed_eaaef_daemon_gateway(  # noqa: SLF001
            daemon_class=type("NotPortal", (), {}),
            local_authority_paths=("/host/state.json",),
        )

    with pytest.raises(
        implementation_supervisor.PlanBoundDispatchError,
        match="cannot construct PortalImplementationDaemon",
    ):
        implementation_supervisor._reject_unsealed_eaaef_daemon_gateway(  # noqa: SLF001
            daemon_class=implementation_daemon.PortalImplementationDaemon,
            local_authority_paths=(),
        )


def test_recorded_eaaef_restart_requires_profile_before_docker_or_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _launch()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-codex-container-restart"
    docker_config = lease_root / "docker-config"
    container_name = "ipfs-accelerate-codex-1-" + "c" * 32
    container_id = "d" * 64
    authorization = SimpleNamespace(
        artifact_cid=_cid("a"),
        authorization_id=_cid("b"),
        worker_principal_did="did:key:zworker",
        provider_principal_did="did:key:zprovider",
    )
    profile = SimpleNamespace(
        authorization=authorization,
        approval_cid=_cid("c"),
    )
    expected_label = "unreachable-without-signed-execution-profile"
    start = [
        "/usr/bin/docker",
        f"--host={grok_cli_runner._DOCKER_LOCAL_HOST}",  # noqa: SLF001
        "--config",
        str(docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]
    receipt = {
        "runtime_id": _cid("f"),
        "container_name": container_name,
        "container_id": "sha256:" + container_id,
        "image_id": launch["qualified_worker_image_digest"],
        "image_receipt": {
            "image_id": launch["qualified_worker_image_digest"],
            "image_label": expected_label,
        },
        "command_receipt": {"start_argv": start},
    }
    observed: list[str] = []

    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: pytest.fail("EAAEF restart resolved a host Docker binary"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_runtime_receipt_identity",
        lambda _docker: _cid("f"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_recorded_codex_lease_root",
        lambda _receipt: (lease_root, docker_config, container_name),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_verified_worker_network_attempt",
        lambda *_args, **_kwargs: ({"codex": authorization}, launch),
    )

    def signed_profile(**kwargs):
        assert kwargs["expected_artifact_cid"] == authorization.artifact_cid
        assert kwargs["expected_container_name"] == container_name
        assert kwargs["expected_lease_root"] == lease_root
        assert kwargs["expected_worker_principal_did"] == "did:key:zworker"
        assert kwargs["expected_provider_principal_did"] == "did:key:zprovider"
        observed.append("authorization")
        return profile

    monkeypatch.setattr(
        grok_cli_runner,
        "_signed_worker_network_profile",
        signed_profile,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_qualified_worker_image",
        lambda **_kwargs: observed.append("image"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_signed_worker_network",
        lambda **_kwargs: observed.append("network"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_qualified_worker_container",
        lambda **_kwargs: observed.append("container"),
    )

    class Process:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("restart process started"),
    )

    with pytest.raises(
        ValueError,
        match="source-addressed-container-execution-profile-launch@1",
    ):
        grok_cli_runner._start_recorded_codex_effect(  # noqa: SLF001
            receipt,
            prompt="continue",
            invocation_binding=_invocation(workspace),
            workspace=workspace,
            worker_network_attempt_authority_json='{"exact":"attempt"}',
        )


def test_recorded_restart_authority_failure_is_pre_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_runtime_receipt_identity",
        lambda _docker: _cid("f"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_recorded_codex_lease_root",
        lambda _receipt: (
            tmp_path / "asref-codex-container-restart",
            tmp_path / "asref-codex-container-restart/docker-config",
            "ipfs-accelerate-codex-1-" + "c" * 32,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_verified_worker_network_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("authorization CID drifted")
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("restart process started"),
    )
    receipt = {
        "runtime_id": _cid("f"),
        "container_name": "ipfs-accelerate-codex-1-" + "c" * 32,
        "container_id": "sha256:" + "d" * 64,
    }

    with pytest.raises(ValueError, match="authorization CID drifted"):
        grok_cli_runner._start_recorded_codex_effect(  # noqa: SLF001
            receipt,
            prompt="continue",
            invocation_binding=_invocation(workspace),
            workspace=workspace,
            worker_network_attempt_authority_json='{"tampered":true}',
        )
