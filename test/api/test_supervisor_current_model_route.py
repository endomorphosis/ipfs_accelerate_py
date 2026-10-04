"""Fresh commands and evidence must bind the operator-selected model pair."""
import json

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.entrypoints.provider_route import (
    ProviderRoutePolicy, ProviderRouteError, QuotaExhaustionEvidence,
    evaluate_quota_fallback,
)
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    build_grok_failure_receipt, valid_grok_failure_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon


def test_current_route_roundtrips_without_authorizing_authentication_fallback():
    route = llm_router.resolve_agent_implementation_route(default_route='legacy')
    assert route.primary_model_id == 'grok-4.7'
    assert route.fallback_model_id == 'gpt-6.1-sol'
    assert not route.permits_authentication_unavailable
    assert llm_router.resolve_agent_implementation_route(**route.as_dict()) == route
    assert 'grok47-sol61' in route.route_id


@pytest.mark.parametrize('field, model', [
    ('primary_model_id', 'grok-4.6'),
    ('fallback_model_id', 'gpt-5.6-sol'),
    ('fallback_model_id', 'gpt-5.6-terra'),
])
def test_stale_model_tuple_cannot_be_reinterpreted_as_current_route(field, model):
    values = llm_router.resolve_agent_implementation_route(default_route='legacy').as_dict()
    values[field] = model
    with pytest.raises(ValueError):
        llm_router.resolve_agent_implementation_route(**values)


@pytest.mark.parametrize('reasoning', ['medium', 'high'])
def test_daemon_emits_and_validates_exact_primary_and_backup(tmp_path, monkeypatch, reasoning):
    workspace = tmp_path / 'work'
    workspace.mkdir()
    executable = tmp_path / 'codex'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o755)
    for key in ('IPFS_ACCELERATE_AGENT_GROK_MODEL', 'GROK_CLI_MODEL', 'GROK_MODEL',
                'ipfs_accelerate_py_GROK_CLI_MODEL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(daemon, '_grok_binary', lambda: '/opt/providers/grok')
    monkeypatch.setattr(daemon, '_grok_cli_available', lambda: True)
    monkeypatch.setattr(runner, 'resolve_codex_quota_fallback_executable', lambda **_: str(executable))
    command = daemon._grok_cli_command(workspace_path=workspace, fallback_reasoning_effort=reasoning)
    assert command[command.index('--model') + 1] == 'grok-4.7'
    fallback = json.loads(command[command.index('--codex-fallback-command-json') + 1])
    assert fallback[fallback.index('-m') + 1] == 'gpt-6.1-sol'
    runner._validate_codex_quota_fallback_command(fallback, workspace=workspace,
                                                expected_fallback_reasoning_effort=reasoning)
    fallback[fallback.index('-m') + 1] = 'gpt-5.6-sol'
    with pytest.raises(ValueError, match='model'):
        runner._validate_codex_quota_fallback_command(fallback, workspace=workspace)


def test_probe_receipt_is_bound_to_current_primary_without_reusing_old_evidence():
    nonce = 'a' * 64
    receipt = build_grok_failure_receipt(probe_stderr_text='Error: max turns reached\n',
        nonce=nonce, model='grok-4.7', probe_returncode=41, primary_dispatched=False)
    assert valid_grok_failure_receipt(receipt, nonce=nonce, model='grok-4.7', returncode=41)
    assert not valid_grok_failure_receipt(receipt, nonce=nonce, model='grok-4.6', returncode=41)


def test_quota_policy_retains_gates_while_selecting_new_backup():
    policy = ProviderRoutePolicy()
    evidence = QuotaExhaustionEvidence(preferred_provider='grok', preferred_model_id='grok-4.7',
        usage_evidence_cid='usage:fixture', observed_capability_cid='capability:fixture', observed_at_ms=1000)
    decision = evaluate_quota_fallback(policy, quota_evidence=evidence, now_ms=1001)
    assert decision.admitted and decision.selected_model_id == 'gpt-6.1-sol'
    for change in ({'repository_effect_observed': True}, {'prior_fallback_dispatches': 1},
                   {'prompt_selected_fallback': True}, {'fallback_model_id': 'gpt-5.6-sol'}):
        with pytest.raises(ProviderRouteError):
            evaluate_quota_fallback(policy, quota_evidence=evidence, **change)
