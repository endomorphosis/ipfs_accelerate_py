from __future__ import annotations

import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import importlib.metadata

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_provider as provider


def configuration(tmp_path):
    workspace = tmp_path / "worktrees" / "task-1"
    workspace.mkdir(parents=True)
    return {"state_root": str(tmp_path), "provider": {"grok_bin": "/usr/bin/grok", "codex_bin": "/usr/bin/codex"}}, workspace


def test_exact_canonical_route_binds_task_and_workspace(tmp_path):
    config, workspace = configuration(tmp_path)
    command = provider.build_provider_argv(config, workspace=workspace, task_id="WWL-000", attempt=1, python_executable="/usr/bin/python3")
    assert Path(command[1]).name == "provider_fallback_runner.py"
    assert command[command.index("--fallback-policy") + 1] == "grok_quota_auth_or_unavailable"
    primary = json.loads(command[command.index("--primary-command-json") + 1])
    fallback = json.loads(command[command.index("--fallback-command-json") + 1])
    assert Path(primary[1]).name == "grok_cli_runner.py"
    assert primary[primary.index("--model") + 1] == "grok-4.7"
    assert "--codex-fallback-command-json" not in primary
    assert fallback[fallback.index("--sandbox") + 1] == "workspace-write"
    assert "--probe-route-readiness" in command
    assert command[command.index("--route-task-id") + 1] == "WWL-000"
    assert command[command.index("--route-attempt") + 1] == "1"
    assert "--dangerously-bypass-approvals-and-sandbox" not in fallback


def test_provider_rejects_wrong_model_scope_or_missing_task_identity(tmp_path):
    config, workspace = configuration(tmp_path)
    config["provider"]["grok_model"] = "grok-other"
    with pytest.raises(ValueError, match="exact grok-4.7"):
        provider.build_provider_argv(config, workspace=workspace, task_id="WWL-000", attempt=1)
    del config["provider"]["grok_model"]
    with pytest.raises(ValueError, match="native task identity"):
        provider.build_provider_argv(config, workspace=workspace, task_id="", attempt=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside the admitted"):
        provider.build_provider_argv(config, workspace=outside, task_id="WWL-000", attempt=1)


def test_portal_implementation_command_preserves_config_path(tmp_path):
    path = tmp_path / "config with spaces.json"
    command = shlex.split(provider.implementation_command("/usr/bin/python3", path))
    assert command == ["/usr/bin/python3", "-m", provider.MODULE, "--config", str(path)]


def test_operational_repair_identity_must_be_in_the_admitted_plan(tmp_path):
    config, workspace = configuration(tmp_path)
    identity = "WWL-REPAIR-0123456789abcdef"
    plan_path = tmp_path / "repair-plan.json"
    plan = {"routing": {"primary_provider": "grok_cli", "primary_model": "grok-4.7", "policy": "grok_quota_auth_or_unavailable", "fallback_provider": "codex_cli"}, "tasks": [{"id": identity}]}
    plan_path.write_text(json.dumps(plan))
    config["plan_path"] = str(plan_path)
    command = provider.build_provider_argv(config, workspace=workspace, task_id=identity, attempt=1)
    assert command[command.index("--route-task-id") + 1] == identity
    with pytest.raises(ValueError, match="absent from the admitted"):
        provider.build_provider_argv(config, workspace=workspace, task_id="WWL-REPAIR-fedcba9876543210", attempt=1)
    with pytest.raises(ValueError, match="native task identity"):
        provider.build_provider_argv(config, workspace=workspace, task_id="WWL-REPAIR-arbitrary-path", attempt=1)


def test_native_protected_boundary_receives_its_required_route_receipt(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor import provider_fallback_runner as runner
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import PROVIDER_PROTECTED_STATE_ROOT_ENV
    config, workspace = configuration(tmp_path)
    monkeypatch.setenv(PROVIDER_PROTECTED_STATE_ROOT_ENV, str(tmp_path))
    monkeypatch.delenv("IPFS_PROOF_REUSE_STATE_ROOT", raising=False)
    command = provider.build_provider_argv(config, workspace=workspace, task_id="WWL-000", attempt=1)
    receipt_path = Path(command[command.index("--route-receipt-path") + 1])
    assert receipt_path.relative_to(tmp_path).parts == ("state", "ptr_lane_worldwide_legal_search", "provider_route_receipts", "WWL-000", "provider-route-attempt-1.json")
    # Exercise the actual runner's receipt admission, stopping before any
    # provider process or auth lookup. The mock is only the filesystem-boundary
    # construction and deliberately unavailable Codex probe.
    boundary = SimpleNamespace(receipt=SimpleNamespace(to_dict=lambda **kw: {"fixture": True, **kw}), state_root=tmp_path, command_prefix=(), environment={})
    monkeypatch.setattr(runner, "_prepare_provider_boundary_runtime", lambda **kw: boundary)
    monkeypatch.setattr(runner, "probe_grok_codex_agent_route_readiness", lambda **kw: SimpleNamespace(codex_ready=False))
    assert runner.main(command[2:]) == 2  # probe refusal, not missing receipt 75
    without_receipt = list(command[2:])
    index = without_receipt.index("--route-receipt-path")
    del without_receipt[index:index + 2]
    assert runner.main(without_receipt) == 75


def test_default_pytest_loads_each_native_governance_plugin_once(pytestconfig):
    manager = pytestconfig.pluginmanager
    assert manager.get_plugin("ipfs_accelerate_py.testing.pytest_ast_seal") is not None
    assert manager.get_plugin("ipfs_accelerate_py.testing.proof_reuse.plugin") is not None
    assert manager.get_plugin("ipfs-pytest-ast-seal") is None
    assert manager.get_plugin("ipfs-proof-reuse") is None
    assert not pytestconfig.getoption("disable_plugin_autoload")
    installed = {entry.name for entry in importlib.metadata.entry_points(group="pytest11")}
    if "anyio" in installed:
        assert manager.get_plugin("anyio") is not None


def test_missing_grok_binary_reaches_native_emergency_decision(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor import provider_fallback_runner as runner
    from ipfs_accelerate_py.llm_router import AgentCLIProviderFailureKind
    config, workspace = configuration(tmp_path)
    del config["provider"]["grok_bin"]
    monkeypatch.setattr(provider, "find_grok_cli", lambda: None)
    command = provider.build_provider_argv(config, workspace=workspace, task_id="WWL-000", attempt=1)
    assert command[command.index("--probe-grok-bin") + 1] == "grok"
    monkeypatch.setattr(runner, "_prepare_provider_boundary_runtime", lambda **kw: SimpleNamespace(receipt=None, command_prefix=(), environment={}))
    monkeypatch.setattr(runner, "probe_grok_codex_agent_route_readiness", lambda **kw: SimpleNamespace(codex_ready=True, grok_ready=False, failure_kind=AgentCLIProviderFailureKind.LAUNCH_FAILURE, reason_code="grok_cli_unavailable"))
    providers = []
    monkeypatch.setattr(runner, "_run_provider", lambda command, **kw: (providers.append(kw["provider_name"]) or SimpleNamespace(result=SimpleNamespace(returncode=0), boundary_applied=True)))
    monkeypatch.setattr(runner.sys, "stdin", __import__("io").StringIO("bounded fixture task"))
    assert runner.main(command[2:]) == 0
    assert providers == ["codex"]
