from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner


ORDERED_ROUTE = dict(
    multi_supervisor_runner.ORDERED_IMPLEMENTATION_PROVIDER_ROUTE
)


def _clear_ordered_route(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ORDERED_ROUTE:
        monkeypatch.delenv(name, raising=False)


def _detached_implementation_args(repo_root: Path) -> list[str]:
    return [
        "--repo-root",
        str(repo_root),
        "--implementation-track",
        "POLICY|worker.py|state|policy",
        "--detach",
    ]


def _fake_detached_payload() -> dict[str, object]:
    return {
        "stamp": "POLICY",
        "master_pid": 1234,
        "master_log": "master.log",
        "master_pid_file": "master.pid",
    }


def test_direct_multi_supervisor_defaults_complete_route_before_detach(
    tmp_path, monkeypatch
) -> None:
    _clear_ordered_route(monkeypatch)
    captured: dict[str, str] = {}

    def fake_launch(_args, _argv):
        captured.update(
            {name: os.environ.get(name, "") for name in ORDERED_ROUTE}
        )
        return _fake_detached_payload()

    monkeypatch.setattr(multi_supervisor_runner, "launch_detached", fake_launch)

    assert (
        multi_supervisor_runner.main(_detached_implementation_args(tmp_path))
        == 0
    )
    assert captured == ORDERED_ROUTE
    assert captured[
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
    ] == "primary_quota_exhausted"
    assert captured["IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"] == "medium"


def test_direct_multi_supervisor_canonicalizes_compatible_grok_alias(
    tmp_path, monkeypatch
) -> None:
    _clear_ordered_route(monkeypatch)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER",
        "grok",
    )
    captured: dict[str, str] = {}

    def fake_launch(_args, _argv):
        captured.update(
            {name: os.environ.get(name, "") for name in ORDERED_ROUTE}
        )
        return _fake_detached_payload()

    monkeypatch.setattr(multi_supervisor_runner, "launch_detached", fake_launch)

    assert (
        multi_supervisor_runner.main(_detached_implementation_args(tmp_path))
        == 0
    )
    assert captured == ORDERED_ROUTE


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER", "auto"),
        ("IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER", "codex"),
        ("IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER", "copilot"),
        ("IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER", "always"),
        (
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER",
            "primary_unavailable",
        ),
        (
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER",
            "primary_unavailable_or_quota_exhausted",
        ),
        ("IPFS_ACCELERATE_AGENT_GROK_MODEL", "grok-4.6"),
        ("IPFS_ACCELERATE_AGENT_CODEX_MODEL", "gpt-5.6-codex"),
        ("IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT", "low"),
    ),
)
def test_direct_multi_supervisor_rejects_incompatible_partial_route_atomically(
    tmp_path, monkeypatch, name, value
) -> None:
    _clear_ordered_route(monkeypatch)
    monkeypatch.setenv(name, value)
    before = {key: os.environ.get(key) for key in ORDERED_ROUTE}
    monkeypatch.setattr(
        multi_supervisor_runner,
        "launch_detached",
        lambda *_args, **_kwargs: pytest.fail(
            "incompatible route reached detached launch"
        ),
    )

    with pytest.raises(SystemExit) as raised:
        multi_supervisor_runner.main(_detached_implementation_args(tmp_path))

    assert raised.value.code == 2
    assert {key: os.environ.get(key) for key in ORDERED_ROUTE} == before


@pytest.mark.parametrize(
    ("fallback_trigger", "reasoning_effort"),
    (
        ("primary_quota_exhausted", "medium"),
        ("primary_quota_exhausted", "high"),
        ("primary_unavailable_or_quota_exhausted", "high"),
    ),
)
def test_direct_multi_supervisor_preserves_closed_configured_trigger(
    tmp_path, monkeypatch, fallback_trigger, reasoning_effort
) -> None:
    _clear_ordered_route(monkeypatch)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER",
        fallback_trigger,
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT",
        reasoning_effort,
    )
    captured: dict[str, str] = {}

    def fake_launch(_args, _argv):
        captured.update(
            {name: os.environ.get(name, "") for name in ORDERED_ROUTE}
        )
        return _fake_detached_payload()

    monkeypatch.setattr(multi_supervisor_runner, "launch_detached", fake_launch)

    assert (
        multi_supervisor_runner.main(_detached_implementation_args(tmp_path))
        == 0
    )
    assert captured[
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
    ] == fallback_trigger
    assert captured["IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"] == (
        reasoning_effort
    )


def test_generic_raw_tracks_do_not_receive_implementation_provider_policy(
    tmp_path, monkeypatch
) -> None:
    _clear_ordered_route(monkeypatch)
    captured: dict[str, str | None] = {}

    def fake_launch(_args, _argv):
        captured.update({name: os.environ.get(name) for name in ORDERED_ROUTE})
        return _fake_detached_payload()

    monkeypatch.setattr(multi_supervisor_runner, "launch_detached", fake_launch)

    result = multi_supervisor_runner.main(
        [
            "--repo-root",
            str(tmp_path),
            "--track",
            "RAW|worker.py|worker.log|supervisor.pid|daemon.pid",
            "--detach",
        ]
    )

    assert result == 0
    assert captured == {name: None for name in ORDERED_ROUTE}


def test_direct_multi_supervisor_preserves_reviewed_high_reasoning(
    tmp_path, monkeypatch
) -> None:
    _clear_ordered_route(monkeypatch)
    effort_env = "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
    monkeypatch.setenv(effort_env, "high")
    captured: dict[str, str] = {}

    def fake_launch(_args, _argv):
        captured.update(
            {name: os.environ.get(name, "") for name in ORDERED_ROUTE}
        )
        return _fake_detached_payload()

    monkeypatch.setattr(multi_supervisor_runner, "launch_detached", fake_launch)

    assert (
        multi_supervisor_runner.main(_detached_implementation_args(tmp_path))
        == 0
    )
    assert captured == {**ORDERED_ROUTE, effort_env: "high"}


def test_detached_master_inherits_exact_runtime_package_root(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class RunningProcess:
        pid = os.getpid()

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return RunningProcess()

    monkeypatch.setattr(multi_supervisor_runner.subprocess, "Popen", fake_popen)
    argv = [
        "--repo-root",
        str(tmp_path),
        "--master-dir",
        str(tmp_path / "state"),
        "--master-pid-path",
        str(tmp_path / "state/master.pid"),
        "--track",
        "T|worker.py|child.log|supervisor.pid|daemon.pid",
        "--detach",
    ]
    args = multi_supervisor_runner.build_arg_parser().parse_args(argv)

    result = multi_supervisor_runner.launch_detached(args, argv)

    package_root = str(
        Path(multi_supervisor_runner.__file__).resolve().parents[3]
    )
    assert captured["command"][:3] == [
        multi_supervisor_runner.sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
    ]
    assert str(captured["env"]["PYTHONPATH"]).split(os.pathsep)[0] == package_root
    assert result["master_pid"] == os.getpid()


def test_repo_launcher_rejects_incompatible_caller_route_defaults(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="incompatible explicit configuration"):
        multi_supervisor_runner.build_repo_implementation_multi_supervisor_launcher(
            repo_root=tmp_path,
            implementation_track_configs=(
                multi_supervisor_runner.ImplementationSupervisorTrackConfig(
                    name="POLICY",
                    script_path="worker.py",
                    state_dir="state",
                    state_prefix="policy",
                ),
            ),
            runtime_package_names=None,
            env_defaults={
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "codex",
            },
        )


def test_repo_launcher_canonicalizes_compatible_partial_route_defaults(
    tmp_path,
) -> None:
    launcher = (
        multi_supervisor_runner.build_repo_implementation_multi_supervisor_launcher(
            repo_root=tmp_path,
            implementation_track_configs=(
                multi_supervisor_runner.ImplementationSupervisorTrackConfig(
                    name="POLICY",
                    script_path="worker.py",
                    state_dir="state",
                    state_prefix="policy",
                ),
            ),
            runtime_package_names=None,
            env_defaults={
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok",
                "CUSTOM_LAUNCH_DEFAULT": "preserved",
            },
        )
    )

    defaults = dict(launcher.env_defaults)
    assert {
        name: defaults[name]
        for name in ORDERED_ROUTE
    } == ORDERED_ROUTE
    assert defaults["CUSTOM_LAUNCH_DEFAULT"] == "preserved"


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/tactician_hammer_logic_repair_supervisor.sh",
        "scripts/proof_gated_contract_repair_supervisor.sh",
    ),
)
def test_direct_shell_launchers_export_complete_ordered_route(relative_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / relative_path).read_text(encoding="utf-8")

    for name, value in ORDERED_ROUTE.items():
        assert f'export {name}="${{{name}:-{value}}}"' in text
