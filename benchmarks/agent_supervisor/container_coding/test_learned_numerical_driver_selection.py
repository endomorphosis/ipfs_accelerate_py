"""Container selection controls; no model, training or provider execution."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as profiles
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver


@pytest.fixture
def capture(tmp_path, monkeypatch):
    calls = []
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda *args: None)
    monkeypatch.setattr(driver.time, "monotonic", lambda: 1000.)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: {"intent_preplanning": {}})
    def stop_at_initial_context(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("captured initial-context boundary")
    monkeypatch.setattr(driver.preparation, "initial_context", stop_at_initial_context)
    monkeypatch.setattr(driver.preparation, "plan", lambda **kwargs: pytest.fail("selection test reached a provider"))
    state = tmp_path / "state/run"
    state.mkdir(parents=True)
    return dict(instruction=tmp_path / "instruction", state=state, arm="full",
                numerical_selection_config=tmp_path / "learned-config.json"), calls


@pytest.mark.parametrize("resource_profile", [None, profiles.EXTENDED_SOURCE384_PROFILE,
                                               profiles.FORMALIZATION_SOURCE384_PROFILE])
@pytest.mark.parametrize("frozen", [False, True])
def test_learned_request_forwards_budget_without_implicit_training(capture, monkeypatch, resource_profile, frozen):
    arguments, contexts = capture
    for key, value in profiles.admission_environment(resource_profile).items():
        monkeypatch.setenv(key, value)
    legacy_calls = []
    def legacy(**kwargs):
        legacy_calls.append(kwargs)
        assert kwargs["security_checkpoint"] == Path("/authored/frozen-checkpoint")
        return dict(train_autoencoder=False, security_checkpoint={"authored_frozen_selection": True})
    monkeypatch.setattr(driver, "_security_runtime_inputs", legacy)
    if frozen:
        arguments.update(security_checkpoint=Path("/authored/frozen-checkpoint"),
                         security_checkpoint_manifest_sha256="a" * 64)
    result = driver.run(**arguments, resource_profile=resource_profile)
    assert result["error_phase"] == "initial_context"
    assert result["error"]["message"] == "captured initial-context boundary"
    assert len(contexts) == 1
    options = contexts[0]
    assert options["numerical_selection_config"] == arguments["numerical_selection_config"]
    assert options["train_autoencoder"] is False
    assert bool(legacy_calls) is frozen
    assert ("security_checkpoint" in options) is frozen
    assert options["numerical_selection_timeout_seconds"] == profiles.execution_budget(resource_profile)["source384_seconds"]
    if resource_profile == profiles.FORMALIZATION_SOURCE384_PROFILE:
        assert options["numerical_selection_budget_profile"] == resource_profile
    else:
        assert "numerical_selection_budget_profile" not in options
    assert result["provider_invocations"] == []


@pytest.mark.parametrize("field", ["security_initializer", "canonical_cve_export", "canonical_cve_manifest_sha256"])
def test_learned_request_rejects_explicit_training_before_preparation(capture, monkeypatch, field):
    arguments, contexts = capture
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: pytest.fail("conflicting request started preparation"))
    arguments[field] = "a" * 64 if field.endswith("sha256") else Path("/authored/training-input")
    with pytest.raises(ValueError, match="excludes legacy training"):
        driver.run(**arguments)
    assert contexts == []


@pytest.mark.parametrize("field", ["security_checkpoint_manifest_sha256", "security_checkpoint_hub_descriptor",
                                   "formula_decoder_descriptor", "header_protocol_descriptor"])
def test_learned_request_cannot_silently_drop_orphan_frozen_assets(capture, monkeypatch, field):
    arguments, contexts = capture
    monkeypatch.setattr(driver.preparation, "prepare", lambda **kwargs: pytest.fail("orphan selection started preparation"))
    arguments[field] = "a" * 64 if field.endswith("sha256") else Path("/authored/decoder-asset")
    with pytest.raises(ValueError, match="explicit frozen checkpoint"):
        driver.run(**arguments)
    assert contexts == []
