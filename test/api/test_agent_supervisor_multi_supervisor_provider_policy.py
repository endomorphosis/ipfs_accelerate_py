from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

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


def test_repo_launcher_rejects_incompatible_caller_route_defaults(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="reviewed legacy"):
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


def test_route_seal_accepts_explicit_auth_or_quota_high_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_id = (
        "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
    )
    expected = {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
            "primary_quota_or_auth_unavailable"
        ),
        "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.5",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE": "board",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_PATH": "policy.json",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_SHA256": (
            "sha256:" + "a" * 64
        ),
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_ID": (
            "sha256:" + "b" * 64
        ),
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_KIND": "explicit_operator_override",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD": "c" * 40,
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_TREE": "d" * 40,
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_ID": route_id,
    }
    environment = dict(expected)
    authorization = SimpleNamespace(
        authorization_kind="explicit_operator_override",
        source_head="c" * 40,
        source_tree="d" * 40,
    )
    plan = SimpleNamespace(
        route_id=route_id,
        as_environment=lambda: dict(expected),
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "load_agent_implementation_route_authorization",
        lambda **_kwargs: authorization,
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "resolve_agent_implementation_route",
        lambda **_kwargs: plan,
    )

    assert (
        multi_supervisor_runner.seal_ordered_implementation_provider_route(
            environment,
            repo_root=tmp_path,
        )
        == expected
    )
    assert environment == expected


def test_route_seal_rejects_hybrid_trigger_and_effort_atomically() -> None:
    environment = {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
            "primary_quota_or_auth_unavailable"
        ),
        "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.5",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
    }
    environment["IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"] = "medium"
    before = dict(environment)

    with pytest.raises(ValueError, match="reviewed legacy"):
        multi_supervisor_runner.seal_ordered_implementation_provider_route(
            environment
        )

    assert environment == before


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
