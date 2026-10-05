"""Operator selection cannot be enabled or weakened by omitted/default flags."""

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, PortalTask, parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    PRODUCTION_CLI_POLICY_NAME,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    MAX_PROVIDER_PROMPT_TOKENS,
)


def _construct(tmp_path: Path, **options):
    return PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md", state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path, **options,
    )


@pytest.mark.parametrize("options", [
    {"production_provider_context_budget_tokens": 0},
    {"production_provider_timeout_seconds": 0.0},
    {"production_provider_review_authority_key_path": Path("review.key")},
    {"production_provider_launch_authority_receipt_path": Path("launch.json")},
    {"production_provider_launch_authority_receipt_content_id": "sha256:" + "a" * 64},
])
def test_daemon_requires_explicit_operator_policy_for_every_option(tmp_path, options):
    with pytest.raises(ValueError, match="require a production provider policy"):
        _construct(tmp_path, **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [
    {"production_provider_context_budget_tokens": 0},
    {"production_provider_context_budget_tokens": -1},
    {"production_provider_context_budget_tokens": True},
    {"production_provider_context_budget_tokens": MAX_PROVIDER_PROMPT_TOKENS + 1},
    {"production_provider_timeout_seconds": 0.0},
    {"production_provider_timeout_seconds": -1.0},
    {"production_provider_timeout_seconds": True},
    {"production_provider_timeout_seconds": float("nan")},
    {"production_provider_timeout_seconds": float("inf")},
    {"production_provider_timeout_seconds": 1201.0},
])
def test_daemon_does_not_treat_invalid_explicit_bounds_as_defaults(tmp_path, options):
    with pytest.raises(ValueError):
        _construct(tmp_path, production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
                   **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("flag,value", [
    ("--production-provider-policy", PRODUCTION_CLI_POLICY_NAME),
    ("--production-provider-context-budget-tokens", "4096"),
    ("--production-provider-timeout-seconds", "240"),
    ("--production-provider-review-authority-key-path", "/tmp/review.key"),
    ("--production-provider-launch-authority-receipt-path", "/tmp/launch.json"),
    ("--production-provider-launch-authority-receipt-content-id", "sha256:" + "a" * 64),
])
@pytest.mark.parametrize("equals", [False, True])
def test_daemon_rejects_duplicate_operator_options(flag, value, equals):
    repeated = [flag + "=" + value] if equals else [flag, value]
    with pytest.raises(SystemExit) as failure:
        parse_args([flag, value, *repeated])
    assert failure.value.code == 2


def test_task_metadata_cannot_enable_the_production_operator_route(tmp_path):
    daemon = _construct(tmp_path)
    task = PortalTask(
        task_id="TEST-1", title="metadata is not operator authority", status="ready",
        completion="manual", priority="P1", track="test", outputs=["module.py"],
        metadata={"production provider policy": PRODUCTION_CLI_POLICY_NAME},
    )
    assert daemon.production_provider_policy is None
    with pytest.raises(ValueError, match="operator provider policy"):
        daemon.run_production_model_assisted_route(
            task, attempt=1, workspace_path=tmp_path, apply=False,
        )


def test_unconfigured_operator_bound_options_remain_absent():
    args = parse_args([])
    assert args.production_provider_context_budget_tokens is None
    assert args.production_provider_timeout_seconds is None
