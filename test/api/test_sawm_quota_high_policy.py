"""Prospective SAWM policy and its real scheduler-to-provider route.

No live owner, database, provider, or supervisor is accessed by these tests.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import configured_board_scheduler
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"


def _validator():
    name = "_sawm_high_policy_validator"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/validate_semantic_addressed_world_model_board.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_sawm_current_and_historical_provider_policies_are_distinct():
    validator = _validator()
    current = _config()
    assert validator._provider_policy_errors(current, multi_lane_selected=True) == []
    historical = copy.deepcopy(current)
    historical.pop("provider_policy_amendment")
    historical["provider"]["fallback_reasoning_effort"] = "medium"
    assert validator._provider_policy_errors(historical, multi_lane_selected=True) == []
    # An arbitrary high edit cannot acquire the prospective operator policy.
    historical["provider"]["fallback_reasoning_effort"] = "high"
    assert validator._provider_policy_errors(historical, multi_lane_selected=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "auth_trigger", "medium", "model", "lineage", "prior_source",
        "board", "integer_boolean", "extra_policy_field", "legacy_lineage",
    ],
)
def test_sawm_policy_amendment_rejects_drift(mutation):
    validator = _validator()
    config = _config()
    amendment = config["provider_policy_amendment"]
    if mutation == "auth_trigger":
        config["provider"]["fallback_trigger"] = "primary_quota_or_auth_unavailable"
    elif mutation == "medium":
        config["provider"]["fallback_reasoning_effort"] = "medium"
    elif mutation == "model":
        config["provider"]["fallback_model_id"] = "another-model"
    elif mutation == "lineage":
        config[validator._M70_SUCCESSOR_KEY]["authority_cid"] = "sha256:" + "a" * 64
    elif mutation == "prior_source":
        amendment["prior_source_head"] = "b" * 40
    elif mutation == "board":
        config["board_namespace"] = "another-board"
    elif mutation == "integer_boolean":
        amendment["prospective_only"] = 1
    elif mutation == "extra_policy_field":
        amendment["allow_authentication_fallback"] = True
    else:
        config.pop(validator._M70_SUCCESSOR_KEY)
    assert validator._provider_policy_errors(config, multi_lane_selected=True)


def _configured_route():
    board = configured_board_scheduler.load_configured_board(CONFIG, repo_root=ROOT)
    launch = configured_board_scheduler.configured_board_launch_plan(
        board, implement=True, detach=True, stamp="20260911T000000Z"
    )
    return launch, configured_board_scheduler._resolved_ordered_provider_route(
        board.payload["provider"], repo_root=ROOT, board_namespace=board.board_namespace
    )


def test_sawm_configured_launch_reaches_exact_daemon_terra_high_argv(tmp_path, monkeypatch):
    launch, route = _configured_route()
    environment = launch["environment"]
    assert environment[implementation_daemon._CODEX_REASONING_EFFORT_ENV] == "high"
    assert environment[implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV] == (
        "primary_quota_exhausted"
    )
    for name in configured_board_scheduler.SCHEDULER_PROVIDER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        if name in configured_board_scheduler.SCHEDULER_PROVIDER_ENV_NAMES:
            monkeypatch.setenv(name, value)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.delenv(implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV, raising=False)
    monkeypatch.delenv(implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV, raising=False)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok")
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/opt/providers/grok")
    monkeypatch.setattr(implementation_daemon, "_goose_meta_spark_available", lambda: False)
    monkeypatch.setattr(
        grok_cli_runner, "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )
    board_path = tmp_path / "tasks.todo.md"
    board_path.write_text("# Tasks\n")
    daemon = implementation_daemon.TodoImplementationDaemon(
        todo_path=board_path, state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path, worktree_root=tmp_path,
    )
    command = daemon._build_implementation_command(tmp_path)
    assert command[command.index("--model") + 1] == "grok-4.6"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    assert 'model_reasoning_effort="medium"' not in fallback
    binding = json.loads(command[command.index("--agent-implementation-route-json") + 1])
    assert binding["route_id"] == route.route_id
    assert binding["fallback_trigger"] == "primary_quota_exhausted"
    assert binding["fallback_reasoning_effort"] == "high"
    assert route.permits_authentication_unavailable is False


@pytest.mark.parametrize(
    ("diagnostic", "reason", "requires_quota"),
    [
        ("not signed in", "authentication_fallback_not_in_route", False),
        ("Grok Build usage balance exhausted", "independent_quota_verification_required", True),
        ("network timeout", "failure_class_not_authorized", False),
    ],
)
def test_sawm_configured_high_route_remains_quota_only(diagnostic, reason, requires_quota):
    _launch, route = _configured_route()
    nonce = "a" * 64
    receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text=diagnostic, nonce=nonce, model="grok-4.6", probe_returncode=41,
    )
    decision = llm_router.decide_agent_implementation_fallback(
        route, repo_root=ROOT, failure_receipt=receipt, expected_nonce=nonce,
        expected_model="grok-4.6", expected_probe_returncode=41,
    )
    assert decision.authorized is False
    assert decision.requires_independent_quota_verification is requires_quota
    assert decision.reason_code == reason


def test_sawm_high_route_accepts_independently_verified_native_quota(tmp_path):
    _launch, route = _configured_route()
    nonce = "b" * 64
    message = (
        "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
        "You have run out of credits or need a Grok subscription. Add credits at "
        "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
    )
    receipt = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text=message, nonce=nonce, model="grok-4.6", probe_returncode=41,
    )
    session_id = "00000000-0000-4000-8000-000000000001"
    home = tmp_path / "independent-grok-home"
    session = home / "sessions" / session_id
    session.mkdir(parents=True)
    updates = [
        {"sessionUpdate": "retry_state", "type": "failed", "error_type": "api", "message": message},
        {"sessionUpdate": "turn_completed", "stop_reason": "error", "agent_result": message},
    ]
    (session / "updates.jsonl").write_text("".join(
        json.dumps({"method": "_x.ai/session/update", "params": {
            "sessionId": session_id, "update": update,
        }}) + "\n" for update in updates
    ))
    (session / "summary.json").write_text(json.dumps({
        "info": {"id": session_id}, "current_model_id": "grok-4.6", "grok_home": str(home),
    }))
    evidence = llm_router.validate_agent_implementation_quota_evidence(
        grok_home=home, expected_session_id=session_id, verifier_returncode=41,
        failure_receipt=receipt,
    )
    decision = llm_router.decide_agent_implementation_fallback(
        route, repo_root=ROOT, failure_receipt=receipt, expected_nonce=nonce,
        expected_model="grok-4.6", expected_probe_returncode=41,
        independent_quota_evidence=evidence,
    )
    assert decision.authorized is True
    assert decision.verifier_status == "confirmed_quota"
