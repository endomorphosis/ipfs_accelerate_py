"""Fail-closed coverage for the board-pinned authority validation backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
    AUTHORITY_VALIDATION_CONTAINER_BACKEND,
    VALIDATION_BACKEND_ENV,
    VALIDATION_CONTAINER_IMAGE_ENV,
    apply_board_validation_runtime,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    build_validation_commands,
)


IMAGE = (
    "sha256:"
    "fbe85c882cbad09dcef78841b5c7cabc1ec0541aca2a8884d018d34c9f1732ae"
)


def test_checked_in_board_binds_exact_container_runtime_over_ambient() -> None:
    root = Path(__file__).resolve().parents[2]
    canonical = json.loads(
        (
            root
            / "config/agent_supervisor_logic_governed_compositional_"
            "verification_fabric_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (
            root
            / "config/agent_supervisor_logic_governed_compositional_"
            "verification_fabric_quack_candidate_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    assert candidate["validation_runtime"] == canonical["validation_runtime"]
    assert canonical["validation_runtime"] == {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "authority-validation-container-runtime@1"
        ),
        "backend": AUTHORITY_VALIDATION_CONTAINER_BACKEND,
        "container_image": IMAGE,
        "required_modules": ["pytest", "z3", "cvc5"],
    }
    environment = {
        VALIDATION_BACKEND_ENV: "hostile-backend",
        VALIDATION_CONTAINER_IMAGE_ENV: "sha256:" + "0" * 64,
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH": "/hostile",
    }

    result = apply_board_validation_runtime(
        root,
        root
        / "docs/architecture/"
        "logic_governed_compositional_verification_fabric.todo.md",
        environ=environment,
    )

    assert result["applied"] is True
    assert result["backend"] == AUTHORITY_VALIDATION_CONTAINER_BACKEND
    assert result["container_image"] == IMAGE
    assert environment[VALIDATION_BACKEND_ENV] == (
        AUTHORITY_VALIDATION_CONTAINER_BACKEND
    )
    assert environment[VALIDATION_CONTAINER_IMAGE_ENV] == IMAGE
    assert "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH" not in environment
    assert environment["IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON_MODULES"] == (
        "pytest,z3,cvc5"
    )


def test_sealed_child_projection_preserves_validation_runtime() -> None:
    environment = {
        VALIDATION_BACKEND_ENV: AUTHORITY_VALIDATION_CONTAINER_BACKEND,
        VALIDATION_CONTAINER_IMAGE_ENV: IMAGE,
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON_MODULES": (
            "pytest,z3,cvc5"
        ),
        "HOSTILE_AMBIENT_VALUE": "must-not-cross",
    }

    projected = multi_runner_module._plan_bound_positive_child_environment(
        environment
    )

    assert projected[VALIDATION_BACKEND_ENV] == (
        AUTHORITY_VALIDATION_CONTAINER_BACKEND
    )
    assert projected[VALIDATION_CONTAINER_IMAGE_ENV] == IMAGE
    assert projected[
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON_MODULES"
    ] == "pytest,z3,cvc5"
    assert "HOSTILE_AMBIENT_VALUE" not in projected


def test_live_child_projection_preserves_validation_runtime(
    tmp_path: Path,
) -> None:
    marker = tmp_path / ".ephemeral-token-persistence-disabled"
    marker.write_text(
        "trusted controller keeps the Quack attach credential in memory\n",
        encoding="utf-8",
    )
    marker.chmod(0o400)
    environment = {
        VALIDATION_BACKEND_ENV: AUTHORITY_VALIDATION_CONTAINER_BACKEND,
        VALIDATION_CONTAINER_IMAGE_ENV: IMAGE,
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON_MODULES": (
            "pytest,z3,cvc5"
        ),
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "opaque-test-token",
        multi_runner_module.QUACK_TOKEN_FILE_ENV: str(marker / "unavailable"),
        multi_runner_module.BOARD_EXTENSION_INSTALL_POLICY_ENV: "load_only",
    }

    projected = (
        multi_runner_module._lgcvf_configured_board_live_positive_child_environment(
            environment,
            common_args=(
                "--endpoint-secret-handle",
                "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
            ),
        )
    )

    assert projected[VALIDATION_BACKEND_ENV] == (
        AUTHORITY_VALIDATION_CONTAINER_BACKEND
    )
    assert projected[VALIDATION_CONTAINER_IMAGE_ENV] == IMAGE
    assert projected[
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON_MODULES"
    ] == "pytest,z3,cvc5"


def test_lgcvf_live_runtime_admission_fails_closed() -> None:
    supervisor_module._require_lgcvf_live_validation_runtime(
        {
            "applied": True,
            "backend": AUTHORITY_VALIDATION_CONTAINER_BACKEND,
            "container_image": IMAGE,
            "required_modules": ["pytest", "z3", "cvc5"],
        }
    )

    with pytest.raises(
        supervisor_module.SupervisorSchedulerConfigError,
        match="validation runtime is not admitted",
    ):
        supervisor_module._require_lgcvf_live_validation_runtime(
            {
                "applied": False,
                "reason": "validation_runtime_invalid",
            }
        )


def test_authority_dependency_probe_uses_bounded_networkless_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"print('bounded source')\n"
    source_sha256 = hashlib.sha256(source).hexdigest()
    captured: dict[str, object] = {}
    contract = {
        "available": True,
        "contract_id": "contract-id",
        "docker_path": "/usr/bin/docker",
        "docker_endpoint": "unix:///run/user/1000/docker.sock",
        "image_id": IMAGE,
        "rootless": True,
    }
    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_authority_validation_isolation_contract",
        staticmethod(lambda: contract),
    )
    monkeypatch.setattr(
        daemon_module,
        "_active_module_source_bytes",
        lambda: source,
    )

    def bounded(command, *, input_payload, environment, inherited_fds=()):
        captured.update(
            {
                "command": list(command),
                "input_payload": input_payload,
                "environment": dict(environment),
                "inherited_fds": tuple(inherited_fds),
            }
        )
        result = {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "passed": True,
            "reason": "project_dependencies_satisfied",
            "projects": [],
            "probe_source_sha256": source_sha256,
        }
        return 0, json.dumps(result).encode("utf-8"), {}

    monkeypatch.setattr(daemon_module, "_run_bounded_probe_process", bounded)
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    payload = {"schema": PROJECT_DEPENDENCY_PROBE_SCHEMA, "projects": []}

    result = PortalImplementationDaemon._authority_validation_dependency_probe(
        payload,
        workspace_path=tmp_path,
    )

    assert result["passed"] is True
    command = captured["command"]
    assert "--pull=never" in command
    assert "--interactive" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--entrypoint=/usr/bin/python" in command
    assert IMAGE in command
    assert json.loads(captured["input_payload"]) == payload
    assert result["authority_validation_isolation"]["container_removed"] is True
    assert result["validation_python_launcher"]["sealed"] is True


def test_explicit_container_backend_never_falls_back_when_socket_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        daemon_module.AUTHORITY_VALIDATION_BACKEND_ENV,
        daemon_module.AUTHORITY_VALIDATION_CONTAINER_BACKEND,
    )
    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_authority_validation_isolation_contract",
        staticmethod(lambda: {"available": True}),
    )
    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_unix_stream_socket_permitted",
        staticmethod(lambda: False),
    )
    monkeypatch.setattr(
        PortalImplementationDaemon,
        "_validation_command_runner",
        staticmethod(
            lambda **_kwargs: pytest.fail("host validation fallback was used")
        ),
    )
    spec = build_validation_commands(["python -m pytest -q test/example.py"])[
        0
    ]

    result = PortalImplementationDaemon._authority_validation_command_runner(
        spec=spec,
        workspace_path=tmp_path,
        timeout_seconds=30,
        environment={"PYTHON": "/usr/bin/python"},
    )

    assert result["returncode"] == 75
    assert result["reason"] == "authority_validation_docker_socket_denied"
    assert result["infrastructure_failure"] is True
