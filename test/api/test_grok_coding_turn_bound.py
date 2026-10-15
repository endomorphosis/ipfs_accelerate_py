"""Exact runner option dispatch with authored provider observations only."""
from pathlib import Path

import pytest

from test.api.test_grok_container_router import routed_workspace, envelope
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner


@pytest.mark.parametrize("turns", [1, 16, 128])
def test_actual_router_sends_exact_bound_and_records_policy(routed_workspace, monkeypatch, turns):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", lambda **_: {"fixture": True})
    calls = []
    def generate(prompt, **kwargs):
        calls.append(kwargs)
        assert kwargs["grok_max_turns"] == turns
        assert kwargs["grok_stream_progress"] is True
        set_last_cli_observation("grok_cli", {"exit_code": 0,
            **native_grok_observation(envelope(), exit_code=0)})
        return "done"
    monkeypatch.setattr(routed_workspace, "generate_text", generate)
    _, receipt = runner.run(prompt="Authored task", provider="grok_cli", model="grok-4.7",
        timeout=30, max_output_tokens=10, purpose="coding", grok_coding_max_turns=turns,
        container_boundary=Path("/fixture/boundary"), container_boundary_sha256="a" * 64)
    assert len(calls) == 1
    assert receipt["provider_invocation_policy"]["max_turns"] == turns
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 33
    assert receipt["completion_authority"] is False


@pytest.mark.parametrize("options", [dict(provider="codex_cli"), dict(purpose="planning"),
    dict(census_only=True), dict(grok_coding_max_turns=True), dict(grok_coding_max_turns=0),
    dict(grok_coding_max_turns=129)])
def test_incompatible_bound_rejected_before_provider_or_workspace(routed_workspace, monkeypatch, options):
    monkeypatch.setattr(routed_workspace, "generate_text", lambda *_a, **_k: pytest.fail("provider reached"))
    args = dict(prompt="Authored", provider="grok_cli", model="grok-4.7", timeout=30,
        max_output_tokens=10, purpose="coding", grok_coding_max_turns=16)
    args.update(options)
    with pytest.raises(ValueError, match="coding turns|coding maximum turns"):
        runner.run(**args)
