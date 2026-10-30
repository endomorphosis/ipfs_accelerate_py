"""Canonical grok_cli/grok-4.7 emergency Codex route for the campaign adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search import (
    CANONICAL_ROUTING,
    FALLBACK_PROVIDER,
    PRIMARY_MODEL,
    PRIMARY_PROVIDER,
    ROUTE_POLICY,
    ROUTER_NAME,
    WorldwideRouteError,
    admitted_tiny_canary,
    bind_explicit_agent_route,
    evaluate_emergency_fallback,
    require_canonical_routing,
    run_admitted_tiny_canary,
    snapshot_workspace,
    trusted_pre_effect_receipt,
)

RECEIPT = Path("artifacts/worldwide-legal-search/receipts/WWL-003.json")
BINDINGS = Path("docs/worldwide/provider-bindings.md")


def plan(routing=None):
    body = json.loads(json.dumps(CANONICAL_ROUTING))
    if routing:
        body.update(routing)
    return {"schema": "worldwide-legal-search.plan@1", "routing": body}


def configuration(tmp_path: Path):
    workspace = tmp_path / "worktrees" / "task-1"
    workspace.mkdir(parents=True)
    (workspace / "seed.txt").write_text("seed\n", encoding="utf-8")
    config = {
        "state_root": str(tmp_path),
        "provider": {
            "grok_bin": "/usr/bin/grok",
            "codex_bin": "/usr/bin/codex",
            "codex_model": "deployment-configured-model",
            "codex_reasoning_effort": "high",
        },
    }
    return config, workspace


def _result(returncode, *, launched=True, activity="unknown"):
    return llm_router.AgentCLIProviderResult(
        returncode,
        launched=launched,
        activity_state=llm_router.AgentCLIActivityState(activity),
    )


def test_canonical_route_matches_the_llm_router_policy():
    routing = require_canonical_routing(plan())
    assert routing["router"] == ROUTER_NAME == "ipfs_accelerate_py.llm_router"
    assert routing["primary_provider"] == PRIMARY_PROVIDER == "grok_cli"
    assert routing["primary_model"] == PRIMARY_MODEL == "grok-4.7"
    assert routing["fallback_provider"] == FALLBACK_PROVIDER == "codex_cli"
    assert routing["policy"] == ROUTE_POLICY == llm_router.GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY
    assert routing["unchanged_workspace_required"] is True
    assert routing["unknown_effects_require_reconciliation"] is True
    assert routing["mandatory_routine_codex_review"] is False
    drifted = plan({"primary_model": "grok-4"})
    with pytest.raises(WorldwideRouteError, match="canonical"):
        require_canonical_routing(drifted)
    review = plan({"mandatory_routine_codex_review": True})
    with pytest.raises(WorldwideRouteError, match="canonical"):
        require_canonical_routing(review)


def test_bind_builds_the_explicit_agent_argv_without_launching(tmp_path):
    config, workspace = configuration(tmp_path)
    before = snapshot_workspace(workspace)
    binding = bind_explicit_agent_route(
        plan(), config, workspace=workspace, task_id="WWL-003", attempt=1, python_executable="/usr/bin/python3",
    )
    argv = binding["argv"]
    primary = json.loads(argv[argv.index("--primary-command-json") + 1])
    fallback = json.loads(argv[argv.index("--fallback-command-json") + 1])
    assert binding["launched"] is False
    assert binding["hosted_model_invocations"] == 0
    assert binding["paid_provider_calls"] == 0
    assert binding["completion_authority"] is False
    assert Path(argv[1]).name == "provider_fallback_runner.py"
    assert Path(primary[1]).name == "grok_cli_runner.py"
    assert primary[primary.index("--model") + 1] == "grok-4.7"
    assert primary[primary.index("--mode") + 1] == "agent"
    assert argv[argv.index("--fallback-policy") + 1] == ROUTE_POLICY
    assert argv[argv.index("--probe-grok-model") + 1] == "grok-4.7"
    assert "--probe-route-readiness" in argv
    assert "--primary-unavailable-kind" not in argv
    assert fallback[fallback.index("-m") + 1] == "deployment-configured-model"
    assert "model_reasoning_effort=\"high\"" in fallback
    assert fallback[fallback.index("--sandbox") + 1] == "workspace-write"
    assert "--dangerously-bypass-approvals-and-sandbox" not in fallback
    assert snapshot_workspace(workspace).digest == before.digest
    config["provider"]["grok_model"] = "grok-other"
    with pytest.raises(WorldwideRouteError, match="exact grok-4.7"):
        bind_explicit_agent_route(plan(), config, workspace=workspace, task_id="WWL-003", attempt=1)


@pytest.mark.parametrize(
    ("kind", "reason", "activity"),
    (
        ("grok_quota_exhausted", "grok_provider_insufficient_quota", "no_activity"),
        ("authentication_failure", "grok_authentication_failure", "no_activity"),
        ("launch_failure", "grok_process_did_not_launch", "no_activity"),
    ),
)
def test_typed_pre_effect_failures_fallback_and_record_telemetry(tmp_path, kind, reason, activity):
    _config, workspace = configuration(tmp_path)
    before = snapshot_workspace(workspace)
    receipt = trusted_pre_effect_receipt(kind, returncode=19, activity=activity, reason=reason)
    telemetry = tmp_path / "telemetry.json"
    decision = evaluate_emergency_fallback(
        workspace,
        task_id="WWL-003",
        attempt=1,
        stage="implementation",
        primary_result=_result(19, activity=activity),
        workspace_before=before,
        telemetry_path=telemetry,
        trusted_failure_receipt=receipt,
    )
    stored = json.loads(telemetry.read_text(encoding="utf-8"))
    assert stored == decision
    assert decision["decision"] == "fallback"
    assert decision["workspace_unchanged"] is True
    assert decision["effects_ambiguous"] is False
    assert decision["failure_kind"] == kind
    assert decision["primary_provider"] == "grok_cli"
    assert decision["primary_model"] == "grok-4.7"
    assert decision["fallback_policy"] == ROUTE_POLICY
    assert decision["paid_provider_calls"] == 0
    assert decision["hosted_model_invocations"] == 0
    assert decision["completion_authority"] is False
    record = decision["route_record"]
    assert record["schema"] == "ipfs_accelerate_py/provider-route@1"
    assert record["route"] == "fallback"
    assert record["failure_kind"] == kind
    assert record["side_effects_started"] is False
    assert record["task_id"] == "WWL-003"
    assert record["attempt"] == 1
    assert record["stage"] == "implementation"
    assert record["primary_provider"] == "grok"
    assert record["fallback_provider"] == "codex"
    assert _CANARY_PROMPT_ABSENT(decision)
    assert snapshot_workspace(workspace).digest == before.digest


def _CANARY_PROMPT_ABSENT(payload) -> bool:
    encoded = json.dumps(payload)
    return "admitted tiny route canary" not in encoded and "api_key" not in encoded and "secret" not in encoded


def test_pre_dispatch_launch_failure_requires_an_unchanged_workspace(tmp_path):
    _config, workspace = configuration(tmp_path)
    before = snapshot_workspace(workspace)
    allowed = evaluate_emergency_fallback(
        workspace,
        task_id="WWL-003",
        attempt=1,
        stage="implementation",
        primary_result=_result(None, launched=False, activity="pre_dispatch"),
        workspace_before=before,
        telemetry_path=tmp_path / "allowed.json",
        primary_unavailable_kind="launch_failure",
    )
    assert allowed["decision"] == "fallback"
    assert allowed["failure_kind"] == "launch_failure"
    assert allowed["workspace_unchanged"] is True
    (workspace / "seed.txt").write_text("mutated\n", encoding="utf-8")
    denied = evaluate_emergency_fallback(
        workspace,
        task_id="WWL-003",
        attempt=1,
        stage="implementation",
        primary_result=_result(None, launched=False, activity="pre_dispatch"),
        workspace_before=before,
        telemetry_path=tmp_path / "denied.json",
        primary_unavailable_kind="launch_failure",
    )
    assert denied["decision"] == "reconcile"
    assert denied["terminal_reason"] == "side_effects_started"
    assert denied["effects_ambiguous"] is True
    assert denied["route_record"] is None
    assert denied["workspace_unchanged"] is False


@pytest.mark.parametrize(
    ("kind", "reason", "activity"),
    (
        ("grok_quota_exhausted", "grok_provider_insufficient_quota", "started"),
        ("grok_quota_exhausted", "grok_provider_insufficient_quota", "unknown"),
        ("timeout", "grok_timeout", "no_activity"),
        ("transport_failure", "grok_transport_failure", "no_activity"),
        ("malformed_output", "grok_output_not_valid_text", "no_activity"),
        ("generic_nonzero_exit", "grok_non_routeable_failure", "no_activity"),
        ("task_failure", "grok_task_failure", "no_activity"),
    ),
)
def test_ambiguous_effects_do_not_fallback(tmp_path, kind, reason, activity):
    _config, workspace = configuration(tmp_path)
    before = snapshot_workspace(workspace)
    if activity == "no_activity" and kind == "grok_quota_exhausted":
        (workspace / "seed.txt").write_text("changed-before-decision\n", encoding="utf-8")
    receipt = trusted_pre_effect_receipt(kind, returncode=19, activity=activity, reason=reason)
    decision = evaluate_emergency_fallback(
        workspace,
        task_id="WWL-003",
        attempt=1,
        stage="implementation",
        primary_result=_result(19, activity=activity),
        workspace_before=before,
        telemetry_path=tmp_path / f"{kind}-{activity}.json",
        trusted_failure_receipt=receipt,
    )
    assert decision["decision"] == "reconcile"
    assert decision["route_record"] is None
    assert decision["paid_provider_calls"] == 0
    if activity in {"started", "unknown"} or kind == "grok_quota_exhausted":
        assert decision["terminal_reason"] == "side_effects_started"
        assert decision["effects_ambiguous"] is True
    else:
        assert decision["terminal_reason"] == "failure_not_fallback_eligible"
        assert decision["effects_ambiguous"] is False


def test_forged_receipt_and_unreliable_snapshot_do_not_fallback(tmp_path):
    _config, workspace = configuration(tmp_path)
    before = snapshot_workspace(workspace)
    receipt = trusted_pre_effect_receipt(
        "authentication_failure", returncode=19, activity="no_activity", reason="grok_authentication_failure",
    )
    forged = json.dumps({**json.loads(receipt), "provider_body": "authentication_failure"})
    forged_decision = evaluate_emergency_fallback(
        workspace,
        task_id="WWL-003",
        attempt=1,
        stage="implementation",
        primary_result=_result(19, activity="no_activity"),
        workspace_before=before,
        telemetry_path=tmp_path / "forged.json",
        trusted_failure_receipt=forged,
    )
    assert forged_decision["decision"] == "reconcile"
    assert forged_decision["failure_kind"] == "malformed_output"
    assert forged_decision["terminal_reason"] == "failure_not_fallback_eligible"
    unreliable = llm_router.AgentCLIWorkspaceSnapshot("", False, "workspace_snapshot_unavailable")
    denied = evaluate_emergency_fallback(
        workspace,
        task_id="WWL-003",
        attempt=1,
        stage="implementation",
        primary_result=_result(None, launched=False, activity="pre_dispatch"),
        workspace_before=unreliable,
        telemetry_path=tmp_path / "unreliable.json",
        primary_unavailable_kind="launch_failure",
    )
    assert denied["decision"] == "reconcile"
    assert denied["terminal_reason"] == "workspace_snapshot_unavailable"
    assert denied["effects_ambiguous"] is True
    assert denied["route_record"] is None


def test_admitted_tiny_live_canary_is_the_only_provider_process(tmp_path):
    workspace = tmp_path / "candidate"
    state = tmp_path / "state"
    workspace.mkdir()
    state.mkdir()
    (workspace / "seed.txt").write_text("seed\n", encoding="utf-8")
    before = snapshot_workspace(workspace)
    result = run_admitted_tiny_canary(admitted_tiny_canary(), workspace=workspace, state_root=state)
    assert result["schema"] == "worldwide-legal-search.tiny-live-canary@1"
    assert result["decision"] == "fallback"
    assert result["failure_kind"] == "launch_failure"
    assert result["workspace_unchanged"] is True
    assert result["workspace_digest_before"] == result["workspace_digest_after"] == before.digest
    assert result["probe_route_readiness"] is False
    assert result["fallback_executable"] == "admitted-local-canary"
    assert result["hosted_model_invocations"] == 0
    assert result["paid_provider_calls"] == 0
    assert result["completion_authority"] is False
    assert result["route_record"]["schema"] == "ipfs_accelerate_py/provider-route@1"
    assert result["route_record"]["side_effects_started"] is False
    assert result["primary_model"] == "grok-4.7"
    assert snapshot_workspace(workspace).digest == before.digest
    with pytest.raises(WorldwideRouteError, match="already ran"):
        run_admitted_tiny_canary(admitted_tiny_canary(), workspace=workspace, state_root=state)
    other = admitted_tiny_canary()
    other["paid_provider_calls"] = 1
    fresh_workspace = tmp_path / "other-candidate"
    fresh_state = tmp_path / "other-state"
    fresh_workspace.mkdir()
    fresh_state.mkdir()
    with pytest.raises(WorldwideRouteError, match="admission"):
        run_admitted_tiny_canary(other, workspace=fresh_workspace, state_root=fresh_state)
    assert list(fresh_state.rglob("*")) == []


def test_receipt_and_bindings_document_the_admitted_route():
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    document = BINDINGS.read_text(encoding="utf-8")
    assert payload["task_id"] == "WWL-003"
    assert payload["router"] == ROUTER_NAME
    assert payload["primary_provider"] == "grok_cli"
    assert payload["primary_model"] == "grok-4.7"
    assert payload["fallback_provider"] == "codex_cli"
    assert payload["fallback_policy"] == ROUTE_POLICY
    assert payload["unchanged_workspace_required"] is True
    assert payload["unknown_effects_require_reconciliation"] is True
    assert payload["mandatory_routine_codex_review"] is False
    assert payload["typed_pre_effect_fallback"] is True
    assert payload["route_telemetry"] is True
    assert payload["ambiguous_effects_block_fallback"] is True
    assert payload["live_canary"] == "admitted-tiny"
    assert payload["hosted_model_invocations"] == 0
    assert payload["paid_provider_calls"] == 0
    assert payload["execution_completed"] is False
    assert payload["completion_authority"] is False
    assert payload["self_certified_completion"] is False
    assert payload["validation_argv"] == ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_provider_route.py", "-q"]
    assert "worldwide_legal_search.py" in "".join(payload["outputs"])
    for phrase in (
        "ipfs_accelerate_py.llm_router",
        "grok_cli",
        "grok-4.7",
        "grok_quota_auth_or_unavailable",
        "unchanged workspace",
        "ambiguous effects",
        "admitted tiny live canary",
        "does not grant campaign completion",
    ):
        assert phrase in document
