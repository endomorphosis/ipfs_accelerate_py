"""Explicit output guidance survives actual dispatch and exact prompt replay."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import tool_output_artifact as artifacts
from ipfs_accelerate_py.agent_supervisor.runtime import tool_output_policy as policy
from benchmarks.agent_supervisor.container_coding.terminal_context_audit import _model_projection, _receipt


@pytest.fixture
def route(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    workspace = tmp_path / "allocated" / "work"
    workspace.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "private-native-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kw:
        SimpleNamespace(provider="codex_cli", model_name="authored", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *a, **kw: object())
    seen = []
    stores = []
    original = policy.create_binding

    def binding(**kwargs):
        result = original(**kwargs)
        stores.append(Path(result["store_path"]))
        return result

    monkeypatch.setattr(policy, "create_binding", binding)

    def dispatch(*, selected="legacy", action=None, **kwargs):
        def generate(prompt, **options):
            seen.append((prompt, options))
            if action:
                action(stores[-1])
            set_last_cli_observation("codex_cli", {"exit_code": 0})
            return "authored provider result"
        monkeypatch.setattr(llm_router, "generate_text", generate)
        return runner.run(prompt="Preserve literal instruction bytes: λ $(not-a-command)",
            provider="codex_cli", model="authored", timeout=30, max_output_tokens=512,
            tool_output_policy=selected, **kwargs)

    yield SimpleNamespace(workspace=workspace, dispatch=dispatch, seen=seen, stores=stores)
    for store in stores:
        if store.exists():
            shutil.rmtree(store)


def test_legacy_control_has_unchanged_prompt_and_no_store(route):
    output, receipt = route.dispatch()
    expected, advisory = runner.render_model_prompt(
        prompt="Preserve literal instruction bytes: λ $(not-a-command)", purpose="coding", workspace=route.workspace)
    assert route.seen[0][0] == expected
    assert "tool_output_policy" not in receipt and route.stores == []
    assert receipt["workspace_advisory_bytes"] == len(advisory.encode())
    assert output == "authored provider result"


def test_selected_guidance_is_audited_but_nonadoption_stays_zero(route):
    output, receipt = route.dispatch(selected="artifact-preview@1")
    binding = receipt["tool_output_policy"]
    assert binding["usage"]["capture_count"] == binding["usage"]["read_count"] == 0
    assert binding["helper_adoption_guaranteed"] is False
    assert binding["usage"]["helper_operations_observed"] is False
    assert binding["usage"]["store_cleanup_complete"] is True
    assert not route.stores[-1].exists()
    assert "--preview-bytes 2048" in route.seen[0][0]
    assert output == "authored provider result"
    _receipt({**receipt, "phase": "coding"}, route.workspace.parent)
    reconstructed, advisory, checks = _model_projection(
        rendered="Preserve literal instruction bytes: λ $(not-a-command)",
        receipt=receipt, repository=None)
    assert reconstructed == route.seen[0][0] and all(checks.values())
    assert checks["tool_output_helper_source"] is True
    assert hashlib.sha256(advisory.encode()).hexdigest() == receipt["workspace_advisory_sha256"]


def test_real_capture_and_middle_retrieval_survive_provider_receipt(route):
    marker = b"IMPORTANT_MIDDLE_EVIDENCE\n"
    evidence = b"BEGIN\n" + b"x" * 6000 + marker + b"y" * 6000 + b"END\n"
    captured = []

    def action(store):
        result = artifacts.run(store=store, argv=[sys.executable, "-c",
            "import sys;sys.stdout.buffer.write(b'BEGIN\\n'+b'x'*6000+b'IMPORTANT_MIDDLE_EVIDENCE\\n'+b'y'*6000+b'END\\n')"], preview_bytes=512)
        assert result["complete"] is True and result["exit_code"] == 0
        descriptor = result["streams"]["stdout"]
        page = artifacts.read(store=store, sha256=descriptor["sha256"], offset=6006, length=len(marker))
        assert page["text"] == marker.decode()
        captured.append(result)

    _, receipt = route.dispatch(selected="artifact-preview@1", action=action)
    usage = receipt["tool_output_policy"]["usage"]
    assert usage["capture_count"] == usage["complete_capture_count"] == usage["read_count"] == 1
    assert usage["read_bytes"] == len(marker) and usage["observed_bytes"] == len(evidence)
    assert "IMPORTANT_MIDDLE_EVIDENCE" not in json.dumps(receipt)
    assert usage["raw_content_exported"] is False
    assert usage["helper_operations_observed"] is True
    assert usage["store_cleanup_complete"] is True and not route.stores[-1].exists()


def test_provider_failure_keeps_usage_without_masking_original_error(route, capsys):
    def fail(store):
        artifacts.run(store=store, argv=[sys.executable, "-c", "print('small')"])
        raise TimeoutError("authored provider failure")
    with pytest.raises(TimeoutError, match="authored provider failure"):
        route.dispatch(selected="artifact-preview@1", action=fail)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    receipt = [row for row in rows if row.get('schema') == 'router-implementation-invocation@1'][-1]
    assert receipt["status"] == "failed" and receipt["tool_output_policy"]["usage"]["capture_count"] == 1
    assert receipt["tool_output_policy"]["usage"]["store_cleanup_complete"] is True


def test_unavailable_store_summary_preserves_provider_result(route, monkeypatch):
    monkeypatch.setattr(artifacts, "summarize_store", lambda _: (_ for _ in ()).throw(ValueError("invalid worker metadata")))
    output, receipt = route.dispatch(selected="artifact-preview@1")
    assert output == "authored provider result"
    assert receipt["tool_output_policy"]["usage"] == {"status": "unavailable", "helper_operations_observed": None,
        "store_cleanup_complete": True, "artifacts_retained_after_invocation": False}


def test_cleanup_failure_is_explicit_and_does_not_mask_provider_result(route, monkeypatch):
    monkeypatch.setattr(artifacts, "remove_store", lambda *a, **kw: (_ for _ in ()).throw(ValueError("changed store")))
    output, receipt = route.dispatch(selected="artifact-preview@1")
    assert output == "authored provider result"
    usage = receipt["tool_output_policy"]["usage"]
    assert usage["store_cleanup_complete"] is False
    assert usage["artifacts_retained_after_invocation"] is None
    assert usage["store_cleanup_error"] == "cleanup_unavailable"


@pytest.mark.parametrize("kwargs", [{"purpose": "planning"}, {"census_only": True}])
def test_incompatible_route_rejected_before_dispatch(route, kwargs):
    with pytest.raises(ValueError):
        route.dispatch(selected="artifact-preview@1", **kwargs)
    assert not route.seen and not route.stores


def test_unsupported_platform_refuses_opt_in_before_dispatch(route, monkeypatch):
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux"):
        route.dispatch(selected="artifact-preview@1")
    assert not route.seen and not route.stores
    assert policy.validate_policy("legacy", purpose="coding", provider="codex_cli") == "legacy"


@pytest.mark.parametrize("mutation", ["instruction", "helper_digest", "store_escape", "authority", "extra"])
def test_binding_tampering_cannot_pass_exact_replay(route, mutation):
    _, receipt = route.dispatch(selected="artifact-preview@1")
    changed = copy.deepcopy(receipt)
    binding = changed["tool_output_policy"]
    if mutation == "instruction":
        binding["instruction_sha256"] = '0' * 64
    elif mutation == "helper_digest":
        binding["helper_sha256"] = '0' * 64
    elif mutation == "store_escape":
        binding["store_path"] = "/tmp/../elsewhere"
    elif mutation == "authority":
        binding["execution_authority"] = True
    else:
        binding["arbitrary_instructions"] = "injected"
    if mutation == "helper_digest":
        _, _, checks = _model_projection(rendered="Preserve literal instruction bytes: λ $(not-a-command)",
            receipt=changed, repository=None)
        assert checks["tool_output_helper_source"] is False
    else:
        with pytest.raises(ValueError):
            _model_projection(rendered="Preserve literal instruction bytes: λ $(not-a-command)",
                receipt=changed, repository=None)
