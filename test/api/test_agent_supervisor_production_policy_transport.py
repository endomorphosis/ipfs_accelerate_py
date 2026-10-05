"""Operator policy transport and native managed-child adoption fences."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_attestation import (
    production_provider_review_key_path,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    PRODUCTION_CLI_POLICY_NAME,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SupervisedChildIdentity,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    ProcessBirthIdentity,
)


POLICY_OPTIONS = (
    "--production-provider-policy",
    "--production-provider-context-budget-tokens",
    "--production-provider-timeout-seconds",
    "--production-provider-review-authority-key-path",
    "--production-provider-launch-authority-receipt-path",
    "--production-provider-launch-authority-receipt-content-id",
)


def _config(tmp_path: Path, *options: str):
    args = supervisor_module.parse_args(
        [
            "--todo-path", str(tmp_path / "todo.md"),
            "--state-dir", str(tmp_path / "state"),
            "--implement",
            *options,
        ]
    )
    return supervisor_module.supervisor_config_from_args(args, repo_root=tmp_path)


def _policy_supervisor(tmp_path: Path):
    config = _config(
        tmp_path,
        "--production-provider-policy", PRODUCTION_CLI_POLICY_NAME,
        "--production-provider-context-budget-tokens", "3072",
        "--production-provider-timeout-seconds", "240",
        "--production-provider-review-authority-key-path",
        str(tmp_path / "bundle authority" / "review.ed25519"),
        "--production-provider-launch-authority-receipt-path",
        str(tmp_path / "launch authority" / "receipt.json"),
        "--production-provider-launch-authority-receipt-content-id",
        "sha256:" + "a" * 64,
    )
    return supervisor_module.PortalImplementationSupervisor(config)


def test_policy_defaults_match_current_policy_without_creating_authority(tmp_path):
    config = _config(tmp_path, "--production-provider-policy", PRODUCTION_CLI_POLICY_NAME)
    assert config.production_provider_context_budget_tokens == DEFAULT_CONTEXT_BUDGET_TOKENS
    assert config.production_provider_timeout_seconds == DEFAULT_PROVIDER_TIMEOUT_SECONDS
    assert config.production_provider_review_authority_key_path == production_provider_review_key_path(config.state_path)
    assert not config.production_provider_review_authority_key_path.exists()
    supervisor = supervisor_module.PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()
    assert supervisor._managed_daemon_matches_command_line(shlex.join(command))
    assert not config.production_provider_review_authority_key_path.exists()


def test_operator_bounds_and_both_authority_bindings_survive_native_command(tmp_path):
    supervisor = _policy_supervisor(tmp_path)
    command = supervisor._build_daemon_command()
    assert supervisor._managed_daemon_matches_command_line(shlex.join(command))
    for option in POLICY_OPTIONS:
        assert command.count(option) == 1
    assert command[command.index(POLICY_OPTIONS[1]) + 1] == "3072"
    assert command[command.index(POLICY_OPTIONS[2]) + 1] == "240.0"
    assert command[command.index(POLICY_OPTIONS[3]) + 1] == str(supervisor.config.production_provider_review_authority_key_path)
    assert command[command.index(POLICY_OPTIONS[4]) + 1] == str(supervisor.config.production_provider_launch_authority_receipt_path)
    assert command[command.index(POLICY_OPTIONS[5]) + 1] == "sha256:" + "a" * 64


@pytest.mark.parametrize("option", POLICY_OPTIONS)
@pytest.mark.parametrize("mutation", ("remove", "change", "duplicate", "equals"))
def test_adoption_requires_one_exact_complete_operator_policy(tmp_path, option, mutation):
    supervisor = _policy_supervisor(tmp_path)
    command = supervisor._build_daemon_command()
    index = command.index(option)
    value = command[index + 1]
    if mutation == "remove":
        del command[index:index + 2]
    elif mutation == "change":
        command[index + 1] = value + "-other"
    elif mutation == "duplicate":
        command.extend((option, value))
    else:
        command[index:index + 2] = [option + "=" + value]
    assert not supervisor._managed_daemon_matches_command_line(shlex.join(command))


@pytest.mark.parametrize("option,value", (
    (POLICY_OPTIONS[1], "2048"),
    (POLICY_OPTIONS[1], "0"),
    (POLICY_OPTIONS[2], "240"),
    (POLICY_OPTIONS[2], "0"),
    (POLICY_OPTIONS[3], "shared-review.ed25519"),
    (POLICY_OPTIONS[4], "authority.json"),
    (POLICY_OPTIONS[5], "sha256:" + "a" * 64),
))
def test_each_operator_option_requires_an_explicit_policy(tmp_path, option, value):
    with pytest.raises(ValueError, match="require a production provider policy"):
        _config(tmp_path, option, value)


@pytest.mark.parametrize("option,value", (
    (POLICY_OPTIONS[1], "0"),
    (POLICY_OPTIONS[1], "-1"),
    (POLICY_OPTIONS[1], "100000000"),
    (POLICY_OPTIONS[2], "0"),
    (POLICY_OPTIONS[2], "-1"),
    (POLICY_OPTIONS[2], "601"),
    (POLICY_OPTIONS[2], "nan"),
    (POLICY_OPTIONS[2], "inf"),
))
def test_operator_policy_rejects_out_of_bounds_values(tmp_path, option, value):
    with pytest.raises(ValueError, match="must be"):
        _config(tmp_path, POLICY_OPTIONS[0], PRODUCTION_CLI_POLICY_NAME, option, value)


def test_unknown_policy_rejected_for_cli_and_embedded_config(tmp_path):
    with pytest.raises(SystemExit):
        _config(tmp_path, POLICY_OPTIONS[0], "invented-policy")
    with pytest.raises(ValueError, match="production CLI policy"):
        supervisor_module.PortalSupervisorConfig(
            todo_path=tmp_path / "todo.md",
            state_path=tmp_path / "state.json",
            strategy_path=tmp_path / "strategy.json",
            events_path=tmp_path / "events.jsonl",
            state_dir=tmp_path,
            production_provider_policy="invented-policy",
        )


@pytest.mark.parametrize("field", (
    "production_provider_context_budget_tokens",
    "production_provider_timeout_seconds",
))
def test_embedded_config_rejects_boolean_numeric_bounds(tmp_path, field):
    with pytest.raises(ValueError, match="must be"):
        supervisor_module.PortalSupervisorConfig(
            todo_path=tmp_path / "todo.md",
            state_path=tmp_path / "state.json",
            strategy_path=tmp_path / "strategy.json",
            events_path=tmp_path / "events.jsonl",
            state_dir=tmp_path,
            production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
            **{field: True},
        )


@pytest.mark.parametrize("option", POLICY_OPTIONS)
@pytest.mark.parametrize("style", ("pairs", "equals", "abbreviation"))
def test_cli_rejects_duplicate_operator_options(tmp_path, option, style):
    values = {
        POLICY_OPTIONS[0]: PRODUCTION_CLI_POLICY_NAME,
        POLICY_OPTIONS[1]: "3072",
        POLICY_OPTIONS[2]: "240",
        POLICY_OPTIONS[3]: "shared-review.ed25519",
        POLICY_OPTIONS[4]: "launch-receipt.json",
        POLICY_OPTIONS[5]: "sha256:" + "a" * 64,
    }
    value = values[option]
    if style == "pairs":
        repeated = [option, value]
    elif style == "equals":
        repeated = [option + "=" + value]
    else:
        repeated = [option[:-1], value]
    with pytest.raises(SystemExit):
        supervisor_module.parse_args([option, value, *repeated])


@pytest.mark.parametrize("options", (
    (POLICY_OPTIONS[4], "receipt.json"),
    (POLICY_OPTIONS[5], "sha256:" + "a" * 64),
    (POLICY_OPTIONS[4], "receipt.json", POLICY_OPTIONS[5], "not-a-content-id"),
))
def test_launch_authority_is_a_bound_pair(tmp_path, options):
    with pytest.raises(ValueError, match="launch authority"):
        _config(tmp_path, POLICY_OPTIONS[0], PRODUCTION_CLI_POLICY_NAME, *options)


def test_disabled_policy_does_not_adopt_policy_bearing_daemon(tmp_path):
    supervisor = supervisor_module.PortalImplementationSupervisor(_config(tmp_path))
    command = supervisor._build_daemon_command()
    assert not any(option in command for option in POLICY_OPTIONS)
    assert supervisor._managed_daemon_matches_command_line(shlex.join(command))
    command.extend((POLICY_OPTIONS[0], PRODUCTION_CLI_POLICY_NAME))
    assert not supervisor._managed_daemon_matches_command_line(shlex.join(command))


def _policy_values(tmp_path):
    return dict(zip(POLICY_OPTIONS, (
        PRODUCTION_CLI_POLICY_NAME,
        "3072",
        "240",
        str(tmp_path / "shared authority" / "review.ed25519"),
        str(tmp_path / "launch authority" / "receipt.json"),
        "sha256:" + "a" * 64,
    )))


def _abbreviated_argument(option, value, style):
    abbreviated = option[:-1]
    return [abbreviated, value] if style == "pair" else [abbreviated + "=" + value]


@pytest.mark.parametrize("option", POLICY_OPTIONS)
@pytest.mark.parametrize("style", ("pair", "equals"))
def test_disabled_policy_rejects_native_argparse_abbreviations(tmp_path, option, style):
    supervisor = supervisor_module.PortalImplementationSupervisor(_config(tmp_path))
    command = supervisor._build_daemon_command()
    addition = _abbreviated_argument(option, _policy_values(tmp_path)[option], style)
    marker = "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    parsed = daemon_module.parse_args([*command[command.index(marker) + 1:], *addition])
    field = option[2:].replace("-", "_")
    expected = "240.0" if option == POLICY_OPTIONS[2] else _policy_values(tmp_path)[option]
    assert str(getattr(parsed, field)) == expected
    assert not supervisor._managed_daemon_matches_command_line(shlex.join([*command, *addition]))


@pytest.mark.parametrize("style", ("pair", "equals"))
def test_disabled_policy_rejects_complete_native_abbreviated_policy(tmp_path, style):
    supervisor = supervisor_module.PortalImplementationSupervisor(_config(tmp_path))
    command = supervisor._build_daemon_command()
    additions = [
        token
        for option, value in _policy_values(tmp_path).items()
        for token in _abbreviated_argument(option, value, style)
    ]
    marker = "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    parsed = daemon_module.parse_args([*command[command.index(marker) + 1:], *additions])
    assert parsed.implement is True
    assert parsed.production_provider_policy == PRODUCTION_CLI_POLICY_NAME
    assert parsed.production_provider_context_budget_tokens == 3072
    assert parsed.production_provider_timeout_seconds == 240.0
    assert parsed.production_provider_review_authority_key_path == tmp_path / "shared authority" / "review.ed25519"
    assert parsed.production_provider_launch_authority_receipt_path == tmp_path / "launch authority" / "receipt.json"
    assert parsed.production_provider_launch_authority_receipt_content_id == "sha256:" + "a" * 64
    assert not supervisor._managed_daemon_matches_command_line(shlex.join([*command, *additions]))


@pytest.mark.parametrize("option", POLICY_OPTIONS)
@pytest.mark.parametrize("style", ("pair", "equals"))
def test_configured_policy_rejects_extra_abbreviated_selection(tmp_path, option, style):
    supervisor = _policy_supervisor(tmp_path)
    command = supervisor._build_daemon_command()
    assert supervisor._managed_daemon_matches_command_line(shlex.join(command))
    additions = _abbreviated_argument(option, _policy_values(tmp_path)[option], style)
    assert not supervisor._managed_daemon_matches_command_line(shlex.join([*command, *additions]))


def test_malformed_quoted_command_never_adopted(tmp_path):
    supervisor = _policy_supervisor(tmp_path)
    command_line = shlex.join(supervisor._build_daemon_command()) + " '"
    assert not supervisor._managed_daemon_matches_command_line(command_line)


def test_obsolete_policy_not_adopted_or_signalled(tmp_path, monkeypatch):
    supervisor = _policy_supervisor(tmp_path)
    pid = 111
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    command = supervisor._build_daemon_command()
    command[command.index(POLICY_OPTIONS[2]) + 1] = "300.0"
    old_command = tuple(command)
    identity = SupervisedChildIdentity(
        process_birth=ProcessBirthIdentity(
            pid=pid, start_time_ticks=1234, boot_id="test-boot", parent_pid=17,
        ),
        command=old_command,
        owner_scope=supervisor._managed_daemon_owner_scope(),
        created_at="2026-10-05T00:00:00+00:00",
    )
    identity_path = supervisor._managed_daemon_identity_path()
    identity_path.write_text(json.dumps(identity.to_dict()) + "\n", encoding="utf-8")
    monkeypatch.setattr(supervisor_module, "process_is_running", lambda value: int(value) == pid)
    monkeypatch.setattr(supervisor_module, "process_command_line", lambda value: shlex.join(old_command))
    monkeypatch.setattr(supervisor_module, "read_process_command_argv", lambda value: old_command)
    monkeypatch.setattr(supervisor, "_list_process_details", lambda: [])
    fence_calls = []
    monkeypatch.setattr(supervisor_module, "terminate_pid_tree", lambda value, **kwargs: fence_calls.append((value, kwargs)) or True)

    result = supervisor._adopt_existing_daemon()

    assert result is None
    assert fence_calls == []
    assert not pid_path.exists()
    assert identity_path.exists()
