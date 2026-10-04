"""Current model defaults join signed profiles without rewriting older lanes."""
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import agent_implementation_route as routes
from ipfs_accelerate_py.agent_supervisor.control import profile_authority as authority
from ipfs_accelerate_py.agent_supervisor import provider_fallback_runner as runner


def _profile(tmp_path, **options):
    return authority.initialize_local_profile(
        repository_cid="sha256:" + "a" * 64,
        baseline_commit="b" * 40,
        profile_dir=tmp_path / "profile",
        lifecycle_dir=tmp_path / "lifecycle",
        **options,
    )


def test_current_default_profile_matches_current_route_and_reloads(tmp_path):
    profile = _profile(tmp_path)
    assert profile.route_id == routes._V3_AGENT_IMPLEMENTATION_ROUTE_ID
    assert profile.fallback_model_id == "gpt-6.1-sol"
    assert profile.fallback_provider_id == "codex"
    assert profile.fallback_reasoning_effort == "high"
    assert authority.load_local_profile(
        repository_cid=profile.repository_cid,
        profile_dir=tmp_path / "profile",
        lifecycle_dir=tmp_path / "lifecycle",
    ) == profile


def test_separately_authorized_external_lane_keeps_its_original_model(tmp_path):
    profile = _profile(tmp_path, route_id=authority.EAAEF_SCOPED_ROUTE_ID)
    assert profile.fallback_model_id == "gpt-5.6-terra"
    assert authority.load_local_profile(
        repository_cid=profile.repository_cid,
        profile_dir=tmp_path / "profile",
        lifecycle_dir=tmp_path / "lifecycle",
    ) == profile


@pytest.mark.parametrize("external", [False, True])
def test_profile_cannot_substitute_a_model_from_another_route(tmp_path, external):
    options = {"route_id": authority.EAAEF_SCOPED_ROUTE_ID} if external else {}
    profile = _profile(tmp_path, **options)
    raw = profile.to_dict()
    raw["fallback_model_id"] = "gpt-6.1-sol" if external else "gpt-5.6-terra"
    with pytest.raises(authority.LocalProfileTampered, match="authority bounds"):
        authority._profile_from_raw(raw)


def test_readiness_probe_receives_current_default_pair_before_dispatch(tmp_path, monkeypatch):
    observed = []

    def stop_at_probe(**options):
        observed.append(options)
        raise RuntimeError("diagnostic stop before provider dispatch")

    monkeypatch.setattr(runner, "probe_grok_codex_agent_route_readiness", stop_at_probe)
    monkeypatch.setattr(runner, "_prepare_provider_boundary_runtime", lambda **options:
        SimpleNamespace(receipt=None, command_prefix=(), environment={}))
    result = runner.main([
        "--workspace", str(tmp_path), "--primary-provider", "grok",
        "--fallback-provider", "codex", "--primary-command-json", '["/bin/true"]',
        "--fallback-command-json", '["/bin/true"]', "--probe-route-readiness",
    ])
    assert result == 2
    assert len(observed) == 1
    assert observed[0]["grok_model"] == "grok-4.7"
    assert observed[0]["codex_model"] == "gpt-6.1-sol"
    assert observed[0]["codex_reasoning_effort"] == "high"
