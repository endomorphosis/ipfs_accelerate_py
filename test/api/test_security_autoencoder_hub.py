"""Supervisor ModelManager consumer integration for datasets-owned weights."""
from dataclasses import replace
import json
import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import security_autoencoder_hub as hub
from tests.unit.logic.formalization.autoencoder.test_security_autoencoder_hub import (
    teacher, fork, joint_inputs, package, _descriptor,
)

def test_modelmanager_registers_only_real_security_inference_and_rechecks_weights(package, tmp_path):
    from ipfs_accelerate_py.model_manager import ModelManager
    from ipfs_accelerate_py.model_catalog.schema import Operation
    manager = ModelManager(storage_path=str(tmp_path / "model-registry.json"), use_database=False,
        enable_ipfs=False, project_legacy_models=False)
    result = hub.register_security_checkpoint(manager=manager, package=package[0],
        expected_manifest_sha256=package[1]["manifest_sha256"], hub_descriptor=_descriptor(package))
    assert result["inference_probe"]["executed"] and result["operation"] == "security.advise"
    found = manager.get_model_descriptor(result["model_id"]).record
    assert {op for c in found.capabilities for op in c.operations} == {Operation.SECURITY_ADVISE}
    assert result["text_generation"] is result["general_text_embedding"] is result["proof_authority"] is False
    fixture = json.loads((package[0] / "inference-fixture.json").read_text())
    scored = hub.score_registered_security_observations(manager=manager, model_id=result["model_id"],
        observations=fixture["observations"])
    assert scored["rows"] == fixture["results"]["rows"]
    path = package[0] / "checkpoint.json"
    path.chmod(0o644); path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        hub.score_registered_security_observations(manager=manager, model_id=result["model_id"],
            observations=fixture["observations"])


@pytest.mark.parametrize("level", ["model", "provider"])
@pytest.mark.parametrize("disabled", ["lifecycle", "authorized", "healthy", "routable", "configured_unknown"])
def test_registered_inference_rejects_disabled_catalog_records_before_loading(package, tmp_path, monkeypatch, level, disabled):
    from ipfs_accelerate_py.model_manager import ModelManager
    from ipfs_accelerate_py.model_catalog.schema import LifecycleState
    manager = ModelManager(storage_path=str(tmp_path / "model-registry.json"), use_database=False,
        enable_ipfs=False, project_legacy_models=False)
    registered = hub.register_security_checkpoint(manager=manager, package=package[0],
        expected_manifest_sha256=package[1]["manifest_sha256"])
    model = manager.get_model_descriptor(registered["model_id"]).record
    provider = manager.get_service(model.provider_id).record
    selected = model if level == "model" else provider
    if disabled == "lifecycle":
        selected = replace(selected, lifecycle=LifecycleState.STOPPED)
    else:
        field, value = ("configured", None) if disabled == "configured_unknown" else (disabled, False)
        selected = replace(selected, state=replace(selected.state, **{field: value}))
    model, provider = (selected, provider) if level == "model" else (model, selected)
    source = hub._RegisteredSecuritySource("security.autoencoder." + package[1]["manifest_sha256"],
        model, provider, package[1]["manifest_sha256"])
    manager.catalog.register_source(source.source, source, load=True, strict=True, side_effecting=False)
    assert manager.get_model_descriptor(model.model_id).record == model
    assert manager.get_service(provider.provider_id).record == provider
    monkeypatch.setattr(hub, "_loaded", lambda *_: pytest.fail("disabled model must not load or score weights"))
    with pytest.raises(ValueError, match="ready and operational"):
        hub.score_registered_security_observations(manager=manager, model_id=model.model_id, observations=[])
