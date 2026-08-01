from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.provider_command_environment import (
    PROVIDER_COMMAND_ENV_DIGEST_ENV,
    PROVIDER_COMMAND_ENV_WRAPPER_ENV,
    PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV,
    project_provider_command_environment,
    sealed_provider_command_environment,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV,
    FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV,
    VALIDATION_PATH_ENV,
    ValidationRuntimeError,
    build_validation_environment,
    canonical_validation_environment_contract,
    formal_toolchain_deployment_manifest,
)

def test_projection_is_explicit_and_rejects_missing_managed_root(
    tmp_path: Path,
) -> None:
    environment = {
        "PATH": str(tmp_path / "hostile-provider-bin"),
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT": "/opt",
        "XAI_API_KEY": "must-not-enter-command-contract",
        "UNRELATED_SECRET": "must-not-enter-command-contract",
    }

    projected = project_provider_command_environment(environment)

    assert projected["PATH"] != environment["PATH"]
    assert projected["IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT"] == str(
        Path("/opt").resolve()
    )
    assert "HOME" not in projected
    assert "XAI_API_KEY" not in projected
    assert "UNRELATED_SECRET" not in projected

    environment["IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT"] = str(
        tmp_path / "missing"
    )
    with pytest.raises(
        ValidationRuntimeError,
        match="deployed toolchain root is unavailable",
    ):
        project_provider_command_environment(environment)


def test_sealed_wrapper_restores_declared_path_and_uses_minimal_environment(
    tmp_path: Path,
) -> None:
    environment = {
        "PATH": str(tmp_path / "sealed-provider-path"),
        VALIDATION_PATH_ENV: "/usr/bin:/bin",
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT": "/opt",
        "UNRELATED_SECRET": "not-approved",
    }
    projected = project_provider_command_environment(environment)
    declared_path = projected["PATH"]
    with sealed_provider_command_environment(environment) as contract:
        preflight = subprocess.run(
            [contract.wrapper_path, "--preflight", "sh"],
            env={"PATH": str(tmp_path / "sealed-shell-path")},
            text=True,
            capture_output=True,
            check=False,
        )
        executed = subprocess.run(
            [
                contract.wrapper_path,
                "--",
                "/bin/sh",
                "-c",
                """\
printf '%s\\n' "$PATH"
printf '%s\\n' "$IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT"
printf '%s\\n' "$IPFS_ACCELERATE_AGENT_FORMAL_TOOLCHAIN_CONTRACT_SHA256"
printf '%s\\n' "${UNRELATED_SECRET-unset}"
""",
            ],
            env={
                "PATH": str(tmp_path / "sealed-shell-path"),
                "UNRELATED_SECRET": "sealed-shell-secret",
            },
            text=True,
            capture_output=True,
            check=False,
        )

    assert preflight.returncode == 0
    assert preflight.stdout.strip() == "provider command available: sh"
    assert executed.returncode == 0
    assert executed.stdout.splitlines() == [
        declared_path,
        str(Path("/opt").resolve()),
        contract.formal_toolchain_contract_sha256,
        "unset",
    ]
    assert contract.sealed is True
    assert contract.environment_names == (
        "IPFS_ACCELERATE_AGENT_FORMAL_TOOLCHAIN_CONTRACT_SHA256",
        "IPFS_ACCELERATE_AGENT_VALIDATION_PATH",
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
        "PATH",
    )
    assert len(contract.contract_sha256) == 64


def test_required_command_preflight_fails_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dispatched: list[list[str]] = []
    sentinel = tmp_path / "provider-write"

    def fake_run(command, **_kwargs):
        dispatched.append(list(command))
        sentinel.write_text("provider was dispatched", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("work"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setenv(
        PROVIDER_COMMAND_REQUIRED_COMMANDS_ENV,
        "missing-provider-proof-command",
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--mode",
            "agent",
        ]
    )

    assert result == 2
    assert dispatched == []
    assert not sentinel.exists()
    assert (
        "required formal toolchain command is unavailable"
        in capsys.readouterr().err
    )


def test_grok_runner_exposes_sealed_wrapper_and_contract_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_run = subprocess.run
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        environment = dict(kwargs["env"])
        wrapper = environment[PROVIDER_COMMAND_ENV_WRAPPER_ENV]
        probe = real_run(
            [
                wrapper,
                "--",
                "python3",
                "-c",
                "print('managed-identity')",
            ],
            env={"PATH": str(tmp_path / "sealed-shell-path")},
            text=True,
            capture_output=True,
            check=False,
        )
        captured["probe"] = probe
        captured["env"] = environment
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("work"))
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setenv(
        VALIDATION_PATH_ENV,
        "/usr/bin:/bin",
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--mode",
            "agent",
        ]
    )

    assert result == 0
    probe = captured["probe"]
    assert isinstance(probe, subprocess.CompletedProcess)
    assert probe.returncode == 0
    assert probe.stdout == "managed-identity\n"
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert len(str(environment[PROVIDER_COMMAND_ENV_DIGEST_ENV])) == 64
    assert (
        environment[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV]
        == canonical_validation_environment_contract(
            {VALIDATION_PATH_ENV: "/usr/bin:/bin"}
        )["formal_toolchain_contract_sha256"]
    )


def test_formal_toolchain_contract_mismatch_blocks_before_grok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dispatched: list[list[str]] = []

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("work"))
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda command, **_kwargs: dispatched.append(list(command)),
    )
    monkeypatch.setenv(
        FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV,
        "0" * 64,
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--mode",
            "agent",
        ]
    )

    assert result == 2
    assert dispatched == []
    assert "formal toolchain deployment contract identity mismatch" in (
        capsys.readouterr().err
    )


def test_validation_consumes_same_digest_bound_formal_toolchain_manifest() -> None:
    source = {
        VALIDATION_PATH_ENV: "/usr/bin:/bin",
        FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV: "python3,sh",
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT": "/opt",
    }
    manifest = formal_toolchain_deployment_manifest(source)
    source[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV] = str(
        manifest["manifest_sha256"]
    )

    validation_environment = build_validation_environment(source)
    rebuilt_environment = build_validation_environment(
        validation_environment
    )
    contract = canonical_validation_environment_contract(source)

    assert (
        validation_environment[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV]
        == manifest["manifest_sha256"]
        == contract["formal_toolchain_contract_sha256"]
    )
    assert (
        rebuilt_environment[FORMAL_TOOLCHAIN_CONTRACT_SHA256_ENV]
        == manifest["manifest_sha256"]
    )
    assert rebuilt_environment["PATH"] == validation_environment["PATH"]
    assert validation_environment[
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT"
    ] == str(Path("/opt").resolve())
    assert set(contract["formal_toolchain_required_executables"]) == {
        "python3",
        "sh",
    }
    assert all(
        len(identity) == 64
        for identity in contract[
            "formal_toolchain_required_executables"
        ].values()
    )


def test_validation_rejects_user_writable_formal_toolchain_root(
    tmp_path: Path,
) -> None:
    writable_root = tmp_path / "profile-toolchain"
    writable_root.mkdir()

    with pytest.raises(
        ValidationRuntimeError,
        match="root-owned/read-only root",
    ):
        build_validation_environment(
            {
                "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT": str(
                    writable_root
                )
            }
        )


def test_provider_prompt_publishes_exact_validation_toolchain_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VALIDATION_PATH_ENV, "/usr/bin:/bin")
    monkeypatch.setenv(
        FORMAL_TOOLCHAIN_REQUIRED_COMMANDS_ENV,
        "python3,sh",
    )
    monkeypatch.setenv(
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
        "/opt",
    )
    contract = canonical_validation_environment_contract()

    guidance = (
        TodoImplementationDaemon
        ._authoritative_validation_environment_guidance()
    )

    assert contract["formal_toolchain_contract_sha256"] in guidance
    assert f"${PROVIDER_COMMAND_ENV_WRAPPER_ENV}" in guidance
    assert "root-owned/read-only root" in guidance
    assert "does not bypass or replace authoritative validation" in guidance


def test_provider_wrapper_never_expands_authoritative_validation_path(
    tmp_path: Path,
) -> None:
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    provider_path = os.pathsep.join((str(provider_bin), "/usr/bin", "/bin"))
    provider_environment = {
        "PATH": provider_path,
        "HOME": str(tmp_path),
    }

    projected = project_provider_command_environment(provider_environment)
    validation = canonical_validation_environment_contract(
        provider_environment
    )

    assert projected["PATH"] == validation["path"]
    assert provider_path != validation["path"]
    assert str(provider_bin) not in validation["path_entries"]
    assert validation["inherited_path_ignored"] is True
