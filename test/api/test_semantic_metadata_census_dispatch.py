"""Current native preparation census stops before any provider allocation."""
import hashlib
import io
import json
import subprocess

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_metadata_dispatch import dispatch_case, invoke  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner


def forbid_provider(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    def forbidden(*args, **kwargs):
        pytest.fail("census reached a provider operation")

    for name in ("discover_available_providers", "select_efficient_route"):
        monkeypatch.setattr(intelligence_index, name, forbidden)
    for name in ("get_llm_provider", "generate_text"):
        monkeypatch.setattr(llm_router, name, forbidden)


def census(case, **kwargs):
    options = dict(semantic_metadata_view="common-bindings@1", census_only=True)
    options.update(kwargs)
    return invoke(case, **options)


def test_census_binds_actual_model_input_and_never_allocates_provider(dispatch_case, monkeypatch, capsys):
    case = dispatch_case
    _, actual = invoke(case, semantic_metadata_view="common-bindings@1")
    actual_input = case["calls"][-1][0]
    capsys.readouterr()
    forbid_provider(monkeypatch)
    before = subprocess.check_output(["git", "status", "--porcelain"], cwd=case["workspace"])
    output, receipt = census(case)
    assert json.loads(output) == receipt
    assert capsys.readouterr().out == ""
    assert receipt["schema"] == "router-semantic-metadata-census@1"
    assert receipt["census"]["selected_complete_sha256"] == hashlib.sha256(actual_input.encode()).hexdigest()
    assert receipt["census"]["selected_complete_bytes"] == actual["model_prompt_bytes"]
    assert receipt["provider_calls"] == receipt["router_calls"] == 0
    assert receipt["provider_dispatch_observed"] is False
    assert receipt["planning_performed_by_census"] is False
    assert receipt["provider_token_usage"] is None and receipt["total_token_savings_measured"] is False
    assert all(receipt[k] is False for k in ("proof_authority", "execution_authority",
        "completion_authority", "settlement_authority", "publication_authority",
        "source_freshness_authority", "required_fact_omission_authority"))
    assert str(case["repository"]) not in output and str(case["workspace"]) not in output
    assert case["prompt"] not in output and "authored-model" not in output
    assert "def " not in output and "model_prompt" not in receipt
    assert len(case["calls"]) == 1  # Only the earlier fake ordinary coding dispatch.
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=case["workspace"]) == before


@pytest.mark.parametrize("options", [
    {"census_only": 1}, {"census_only": "true"},
    {"semantic_metadata_view": "legacy"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"purpose": "planning"},
])
def test_census_requires_explicit_original_semantic_coding_view(dispatch_case, monkeypatch, options):
    forbid_provider(monkeypatch)
    with pytest.raises(ValueError):
        census(dispatch_case, **options)
    assert dispatch_case["calls"] == []


def test_census_refuses_current_source_drift(dispatch_case, monkeypatch):
    case = dispatch_case
    forbid_provider(monkeypatch)
    source = case["repository"] / "mod.py"
    source.write_text(source.read_text() + "\n# changed current source\n")
    with pytest.raises(ValueError):
        census(case)
    assert case["calls"] == []


def test_census_refuses_workspace_from_a_foreign_git_owner(dispatch_case, tmp_path, monkeypatch):
    case = dispatch_case
    forbid_provider(monkeypatch)
    other = tmp_path / "foreign-owner"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True, capture_output=True)
    monkeypatch.chdir(other)
    with pytest.raises(ValueError, match="source repository differs"):
        census(case)
    assert case["calls"] == []


def test_census_refuses_inherited_database_authority(dispatch_case, monkeypatch):
    forbid_provider(monkeypatch)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_OWNER_TOKEN", "authored-secret-must-not-export")
    with pytest.raises(ValueError, match="inherited database authority"):
        census(dispatch_case)
    assert dispatch_case["calls"] == []


def test_census_retains_exact_wrapper_binding_checks(dispatch_case, tmp_path, monkeypatch):
    forbid_provider(monkeypatch)
    artifact = tmp_path / "foreign-wrapper.json"
    artifact.write_text('{"secret":"untrusted-wrapper"}')
    with pytest.raises(ValueError):
        census(dispatch_case, public_instruction_artifact=artifact,
            public_instruction_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            public_instruction_task_cid="cid:foreign-task")
    with pytest.raises(ValueError):
        census(dispatch_case, doctor_residual_artifact=artifact,
            doctor_residual_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            doctor_residual_task_cid="cid:foreign-task")
    assert dispatch_case["calls"] == []


def test_cli_census_emits_one_closed_observation(dispatch_case, monkeypatch, capsys):
    case = dispatch_case
    forbid_provider(monkeypatch)
    monkeypatch.setattr("sys.argv", ["router", "--provider", "codex_cli", "--model", "authored-model",
        "--semantic-repository", str(case["repository"]), "--semantic-metadata-view", "common-bindings@1",
        "--semantic-metadata-census-only", "--coding-reply-mode", "ordinary-completion@1"])
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(case["prompt"].encode())))
    assert runner.main() == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1 and captured.err == ""
    receipt = json.loads(lines[0])
    assert receipt["schema"] == "router-semantic-metadata-census@1" and receipt["provider_calls"] == 0
    assert receipt["current_semantic_encoder_used"] is True and receipt["credential_isolation_checked"] is True
    assert case["calls"] == []


def test_cli_drift_failure_exports_only_closed_diagnostic(dispatch_case, monkeypatch, capsys):
    case = dispatch_case
    forbid_provider(monkeypatch)
    source = case["repository"] / "mod.py"
    source.write_text(source.read_text() + "\n# source-private-secret\n")
    monkeypatch.setattr("sys.argv", ["router", "--model", "authored-model",
        "--semantic-repository", str(case["repository"]), "--semantic-metadata-view", "common-bindings@1",
        "--semantic-metadata-census-only"])
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(case["prompt"].encode())))
    assert runner.main() == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "source-private-secret" not in captured.err
    diagnostic = json.loads(captured.err)
    assert diagnostic["schema"] == "router-implementation-error@2"
    assert diagnostic["diagnostic"]["phase"] == "semantic_context"
    assert diagnostic["diagnostic"]["provider_dispatch_observed"] is None
    assert case["calls"] == []
