"""Hermetic execution bounds for the EAAEF-180..183 early frontier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.validation import eaaef_host_admission

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "scripts/collect_eaaef_host_admission_receipts.py"
RUNNER = ROOT / "scripts/run_eaaef_host_admission_supervisor.py"
EARLY = ("EAAEF-180", "EAAEF-181", "EAAEF-182", "EAAEF-183")
LATER = tuple(f"EAAEF-{number}" for number in range(184, 192))


def _load_script(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _forbidden(name: str):
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(f"early frontier reached forbidden effect {name}")

    return fail


def _typed_launch_plan() -> dict[str, Any]:
    blocker = "rootless engine evidence pending"
    return {
        "schema": eaaef_host_admission.EARLY_FRONTIER_LAUNCH_PLAN_SCHEMA,
        "allowed": False,
        "blockers": [blocker],
        "blocker_classes": {
            blocker: eaaef_host_admission.classify_blocker(blocker)
        },
        "argv": [],
        "candidate_executable_withheld": True,
        "execution_prohibited": True,
        "materialization_receipt_cid": "sha256:" + "a" * 64,
        "bootstrap_admission_statement": None,
        "bootstrap_admission_published": False,
        "process_started": False,
    }


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "early_frontier"),
        (["--early-frontier"], "early_frontier"),
        (["--full-host-evidence"], "full_host_evidence"),
    ],
)
def test_collector_cli_keeps_full_evidence_behind_explicit_scope(
    argv: list[str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(COLLECTOR, f"tested_eaaef_collector_{expected}_{len(argv)}")
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_collect_host_admission",
        lambda: calls.append("early_frontier") or {"scope": "early_frontier"},
    )
    monkeypatch.setattr(
        module,
        "_collect_full_host_admission",
        lambda: calls.append("full_host_evidence")
        or {"scope": "full_host_evidence"},
    )

    assert module.main(argv) == 0

    assert calls == [expected]
    assert json.loads(capsys.readouterr().out) == {"scope": expected}


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "immutable_observation"),
        (["--early-frontier"], "early_frontier"),
        (["--full-bootstrap"], "full_bootstrap"),
    ],
)
def test_supervisor_cli_defaults_to_immutable_observation(
    argv: list[str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(RUNNER, f"tested_eaaef_runner_{expected}_{len(argv)}")
    calls: list[str] = []

    def run_once(*, scope: str) -> dict[str, str]:
        calls.append(scope)
        return {"execution_scope": scope}

    monkeypatch.setattr(module, "run_once", run_once)

    assert module.main(argv) == 0

    assert calls == [expected]
    assert json.loads(capsys.readouterr().out) == {"execution_scope": expected}


def test_early_frontier_collector_never_probes_or_writes_later_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    sentinels: dict[Path, bytes] = {}
    for task_id in LATER:
        path = receipt_dir / eaaef_host_admission.RECEIPT_FILES[task_id]
        sentinels[path] = f"preserve:{task_id}\n".encode()
        path.write_bytes(sentinels[path])

    monkeypatch.setattr(eaaef_host_admission, "ROOT", tmp_path)
    monkeypatch.setattr(eaaef_host_admission, "RECEIPT_DIR", receipt_dir)
    monkeypatch.setattr(
        eaaef_host_admission,
        "load_isolated_launch_plan",
        lambda **_kwargs: _typed_launch_plan(),
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "_source_identity",
        lambda: {
            "source_head": "1" * 40,
            "source_tree": "2" * 40,
            "board_cid": "sha256:" + "3" * 64,
            "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        },
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "bind_runtime_principals",
        lambda: {
            "principals": [
                {"role": role, "did": f"did:key:test-{role}", "admitted_authority": False}
                for role in ("worker", "provider", "quack_owner")
            ],
            "secret_material_exported": False,
            "admitted_authority": False,
        },
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "probe_duckdb_quack",
        lambda: {"decision": "admitted", "network_install_attempted": False},
    )
    monkeypatch.setattr(
        eaaef_host_admission,
        "probe_engine_mode",
        lambda: {
            "decision": "admitted",
            "docker_host": "unix:///run/user/1000/docker.sock",
            "docker_socket_mounted": False,
            "supervisor_started": False,
        },
    )
    for name in (
        "materialize_host_evidence",
        "probe_provider_authorization",
        "probe_worker_image",
        "probe_container_profile",
        "probe_worker_network",
        "probe_command_fabric",
        "probe_native_lane",
        "probe_plan_r2",
        "load_admission_bundle_signatures",
    ):
        monkeypatch.setattr(eaaef_host_admission, name, _forbidden(name))

    result = eaaef_host_admission.collect_early_frontier_and_write()

    assert tuple(result["decisions"]) == EARLY
    assert result["scope"] == "early_frontier_180_183"
    assert {
        path.name for path in receipt_dir.iterdir() if path.read_bytes().startswith(b"{")
    } == {
        eaaef_host_admission.RECEIPT_FILES[task_id] for task_id in EARLY
    }
    for path, expected in sentinels.items():
        assert path.read_bytes() == expected


def test_invalid_lifecycle_preflight_fails_before_host_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_dir = tmp_path / "receipts"
    monkeypatch.setattr(eaaef_host_admission, "RECEIPT_DIR", receipt_dir)
    monkeypatch.setattr(
        eaaef_host_admission,
        "load_isolated_launch_plan",
        lambda **_kwargs: {
            "valid": False,
            "error": "datasets planning pin differs from the qualified source",
        },
    )
    for name in (
        "bind_runtime_principals",
        "probe_duckdb_quack",
        "probe_engine_mode",
        "write_early_frontier_host_admission_receipts",
    ):
        monkeypatch.setattr(eaaef_host_admission, name, _forbidden(name))

    with pytest.raises(
        eaaef_host_admission.EarlyFrontierPreflightBlocked,
        match=eaaef_host_admission.EARLY_FRONTIER_PREFLIGHT_BLOCKER,
    ):
        eaaef_host_admission.collect_early_frontier_and_write()

    assert not receipt_dir.exists()


def test_isolated_launch_failure_reports_typed_source_lifecycle_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {
                "valid": False,
                "error": "bootstrap launcher path is not the reviewed repository launcher",
            }
        ),
        stderr="",
    )
    monkeypatch.setattr(
        eaaef_host_admission.subprocess,
        "run",
        lambda *_args, **_kwargs: completed,
    )

    with pytest.raises(
        eaaef_host_admission.EarlyFrontierPreflightBlocked,
        match=(
            eaaef_host_admission.EARLY_FRONTIER_PREFLIGHT_BLOCKER
            + ".*bootstrap launcher path"
        ),
    ):
        eaaef_host_admission.load_isolated_launch_plan()


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "tcp://127.0.0.1:2375",
        "ssh://docker@example.invalid",
        "http://127.0.0.1:2375",
        "unix://relative/docker.sock",
    ],
)
def test_docker_info_rejects_non_local_unix_before_subprocess(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eaaef_host_admission.subprocess,
        "run",
        _forbidden("docker subprocess"),
    )

    with pytest.raises(ValueError, match="local absolute unix"):
        eaaef_host_admission._docker_info(endpoint)


def test_engine_probe_rejects_tcp_environment_and_uses_only_unix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EAAEF_DOCKER_HOST", "tcp://127.0.0.1:2375")
    monkeypatch.setenv("DOCKER_HOST", "ssh://user:secret@example.invalid")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")
    observed: list[str] = []

    def probe(host: str) -> tuple[int, dict[str, Any]]:
        observed.append(host)
        assert host.startswith("unix:///")
        if host == "unix:///run/user/4242/docker.sock":
            return 0, {
                "DockerRootDir": "/run/user/4242/.local/share/docker",
                "SecurityOptions": ["name=rootless"],
                "ServerVersion": "test",
            }
        return 1, {}

    monkeypatch.setattr(eaaef_host_admission, "_docker_info", probe)

    evidence = eaaef_host_admission.probe_engine_mode()

    assert evidence["decision"] == "admitted"
    assert evidence["docker_host"] == "unix:///run/user/4242/docker.sock"
    assert observed == ["unix:///run/user/4242/docker.sock"]
    assert {item["source"] for item in evidence["endpoint_rejections"]} == {
        "EAAEF_DOCKER_HOST",
        "DOCKER_HOST",
    }
    assert "secret" not in json.dumps(evidence)


def test_missing_docker_cli_is_typed_missing_without_endpoint_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EAAEF_DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")

    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(eaaef_host_admission.subprocess, "run", missing)

    evidence = eaaef_host_admission.probe_engine_mode()

    assert evidence["decision"] == "typed_missing"
    assert evidence["mode"] == "engine_unavailable"
    assert evidence["docker_host"] == ""
    assert evidence["docker_info_returncode"] == 127
    assert {item["docker_host"] for item in evidence["probes"]} == {
        "unix:///run/user/4242/docker.sock",
        f"unix://{Path.home()}/.docker/run/docker.sock",
        "unix:///var/run/docker.sock",
    }


@pytest.mark.parametrize("scope", ("early_frontier", "full_bootstrap"))
def test_legacy_mutable_supervisor_scopes_fail_before_any_effect(
    scope: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script(
        RUNNER,
        f"tested_eaaef_disabled_legacy_runner_{scope}",
    )
    database = tmp_path / "run-v-test/control.duckdb"
    status = tmp_path / "status.json"

    for name in (
        "_active_control_db",
        "_acquire_state_owner_lease",
        "_collect_host_admission",
        "_collect_full_host_admission",
        "_current_host_admission_identity",
        "_database_task_source_class",
        "_write_status",
    ):
        monkeypatch.setattr(runner, name, _forbidden(name))
    monkeypatch.setattr(runner, "STATUS_PATH", status)

    with pytest.raises(
        RuntimeError,
        match="legacy mutable EAAEF host-admission scope is disabled",
    ):
        runner.run_once(scope=scope)

    assert not database.exists()
    assert not status.exists()
