"""Real offline checkpoint inference joined to native indexed planning."""
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from test.api.test_codebase_autoencoder_transfer import teacher, fork, joint_inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _security_runtime_inputs
from ipfs_accelerate_py.agent_supervisor.runtime import security_autoencoder_advisor as advisor


@pytest.fixture
def checkpoint(joint_inputs, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import codebase_autoencoder as trainer
    from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_checkpoint import export_security_checkpoint
    trained = trainer.train_codebase_autoencoder(**joint_inputs)
    package = export_security_checkpoint(repository=joint_inputs["repository"], expected_receipt=trained,
        output=tmp_path / "portable-security-checkpoint")
    monkeypatch.setattr(trainer, "train_codebase_autoencoder", lambda **_: pytest.fail("frozen runtime trained"))
    monkeypatch.setattr(trainer, "validate_codebase_autoencoder", lambda **_: pytest.fail("frozen runtime depended on training state"))
    return package


def test_frozen_runtime_selection_is_explicit_and_excludes_training(checkpoint):
    selected = _security_runtime_inputs(security_checkpoint=Path(checkpoint["output"]),
        security_checkpoint_manifest_sha256=checkpoint["manifest_sha256"], security_initializer=None,
        canonical_cve_export=None, canonical_cve_manifest_sha256=None)
    assert selected == {"train_autoencoder": False, "security_checkpoint": checkpoint}
    with pytest.raises(ValueError, match="excludes local training"):
        _security_runtime_inputs(security_checkpoint=Path(checkpoint["output"]),
            security_checkpoint_manifest_sha256=checkpoint["manifest_sha256"], security_initializer=Path("/initializer"),
            canonical_cve_export=None, canonical_cve_manifest_sha256=None)


def test_actual_frozen_advice_reaches_planner_and_admission_without_training(original, checkpoint, monkeypatch):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    receipt = prep.initial_context(state=state, security_checkpoint=checkpoint)
    advice = receipt["security_autoencoder_advice"]
    assert "codebase_autoencoder" not in receipt
    assert advice["training_steps"] == advice["provider_calls"] == advice["download_calls"] == 0
    assert advice["metadata_ducklake"]["status"] == advice["world_ducklake"]["status"] == "projected"
    assert advice["hydration"]["catalog_count"] == 3
    assert advice["model_registration"]["operation"] == "security.advise"
    assert advice["model_registration"]["inference_probe"]["executed"] is True
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    assert any(row["schema"] == "frozen-security-planning-summary@1" for row in loaded["summaries"])
    assert advice["summary"]["target_vocabulary"]
    assert advice["summary"]["formal_formula_heads_present"] is False
    calls = []
    def provider(prompt, **kwargs):
        calls.append(prompt)
        for identity in (checkpoint["checkpoint_sha256"], checkpoint["manifest_sha256"],
                         advice["hydration"]["world_record_cid"]):
            assert identity in prompt
        for target in advice["summary"]["target_vocabulary"]:
            assert target in prompt
        assert '"scores_are_calibrated_probabilities":false' in prompt.replace(" ", "")
        return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
    _version(monkeypatch)
    planned = prep.plan(state, provider_callable=provider)
    assert planned["qualified"] and len(calls) == 1
    before = Path(advice["output"], "inference-descriptor.json").read_bytes()
    reused = prep.context(state=state)
    assert reused["initial_indexes_reused"] is True
    assert Path(advice["output"], "inference-descriptor.json").read_bytes() == before
    validate = advisor.validate_security_advice(repository=root, expected_receipt=advice)
    assert validate["proof_authority"] is False


def test_generic_local_refresh_recomputes_current_rows_with_same_frozen_weights(tmp_path, checkpoint):
    root = tmp_path / "target"
    root.mkdir()
    source = root / "code.py"
    source.write_text("def operation(value):\n    return value + 1\n")
    hashes = {"code.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    first = advisor.prepare_security_advice(repository=root, paths=["code.py"], source_hashes=hashes,
        checkpoint=checkpoint, output=tmp_path / "advice-before")
    assert advisor.refresh_security_advice(repository=root, previous=first, output=tmp_path / "not-needed") == first
    assert not (tmp_path / "not-needed").exists()
    source.write_text("def operation(value):\n    return str(value).strip()\n")
    with pytest.raises(ValueError, match="stale"):
        advisor.validate_security_advice(repository=root, expected_receipt=first)
    second = advisor.refresh_security_advice(repository=root, previous=first, output=tmp_path / "advice-after")
    assert second["checkpoint"] == first["checkpoint"] == checkpoint
    assert second["source_hashes"] != first["source_hashes"]
    assert second["record"]["inference_sha256"] != first["record"]["inference_sha256"]
    assert second["hydration"]["world_record_cid"] != first["hydration"]["world_record_cid"]
    assert second["training_steps"] == 0 and second["proof_authority"] is False
    advisor.validate_security_advice(repository=root, expected_receipt=second)


def test_checkpoint_mode_rejects_training_and_package_overlap_before_writes(original, checkpoint):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    with pytest.raises(ValueError, match="mutually exclusive"):
        prep.initial_context(state=state, security_checkpoint=checkpoint, train_autoencoder=True)
    assert not (root / ".runtime/terminal-vectors").exists()
    with pytest.raises(ValueError, match="external security advice"):
        advisor.prepare_security_advice(repository=root, paths=["bottle.py"],
            source_hashes={"bottle.py": hashlib.sha256((root / "bottle.py").read_bytes()).hexdigest()},
            checkpoint=checkpoint, output=Path(checkpoint["output"]) / "forbidden-observation")
    assert not Path(checkpoint["output"], "forbidden-observation").exists()


def test_portable_package_is_exactly_pinned_and_relocated_in_runtime_archive(tmp_path, checkpoint):
    from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_checkpoint import PACKAGE_FILES, load_security_checkpoint
    built = deployment.build_runtime_archive(output=tmp_path / "runtime-archive", **_inputs(tmp_path),
        security_checkpoint=Path(checkpoint["output"]), security_checkpoint_manifest_sha256=checkpoint["manifest_sha256"])
    binding = built["security_checkpoint"]
    assert binding["descriptor"] == {**checkpoint, "output": deployment.ROOT + "/" + deployment.SECURITY_CHECKPOINT_PATH}
    assert binding["runtime_training_steps"] == binding["runtime_download_calls"] == 0
    assert built["security_initializer"] is built["canonical_cve_training"] is None
    offline = tmp_path / "offline-package"
    offline.mkdir()
    with tarfile.open(tmp_path / "runtime-archive/runtime.tar.gz") as archive:
        names = [name for name in archive.getnames() if name.startswith(deployment.SECURITY_CHECKPOINT_PATH + "/")]
        assert {Path(name).name for name in names} == set(PACKAGE_FILES)
        for name in names:
            (offline / Path(name).name).write_bytes(archive.extractfile(name).read())
    assert load_security_checkpoint(offline, expected_manifest_sha256=checkpoint["manifest_sha256"])["descriptor"]["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]


def test_advice_selection_drift_refused_before_provider(original, checkpoint, monkeypatch):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    prep.initial_context(state=state, security_checkpoint=checkpoint)
    selection = state / "security-checkpoint-selection.json"
    bad = json.loads(selection.read_bytes())
    bad["checkpoint_sha256"] = "a" * 64
    selection.write_text(json.dumps(bad))
    _version(monkeypatch)
    calls = []
    with pytest.raises(ValueError, match="selected model"):
        prep.plan(state, provider_callable=lambda *args, **kwargs: calls.append(args))
    assert calls == [] and not (state / "planner-invoked.json").exists()


def test_builder_resolves_pinned_hub_before_packaging_and_runtime_remains_offline(tmp_path, checkpoint, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import security_autoencoder_hub as hub
    descriptor = {"schema": hub.HUB_SCHEMA, "repository_id": "authored/security-autoencoder",
        "repository_type": "model", "revision": "a" * 40, "release_prefix": "releases/authored-fixture",
        "manifest_sha256": checkpoint["manifest_sha256"]}
    calls = []
    original_download = hub.download_security_checkpoint
    def fetch(repository, revision, path):
        calls.append((repository, revision, path))
        assert repository == descriptor["repository_id"] and revision == descriptor["revision"]
        assert path.startswith(descriptor["release_prefix"] + "/")
        return Path(checkpoint["output"], path.rsplit("/", 1)[1]).read_bytes()
    monkeypatch.setattr(hub, "download_security_checkpoint", lambda **kwargs:
        original_download(**kwargs, fetch_bytes=fetch))
    built = deployment.build_runtime_archive(output=tmp_path / "hub-runtime", **_inputs(tmp_path),
        security_checkpoint_hub_descriptor=descriptor, security_checkpoint_cache=tmp_path / "model-cache")
    assert calls and built["security_checkpoint"]["hub"] == descriptor
    assert built["security_checkpoint"]["hub_descriptor_path"] == deployment.SECURITY_CHECKPOINT_HUB
    assert built["security_training_requirements"] == []
    assert built["security_inference_requirements"] == ["numpy==1.26.4"]
    assert built["security_checkpoint"]["runtime_download_calls"] == 0
    offline = tmp_path / "runtime-local"
    offline.mkdir()
    with tarfile.open(tmp_path / "hub-runtime/runtime.tar.gz") as archive:
        for name in archive.getnames():
            if name.startswith(deployment.SECURITY_CHECKPOINT_PATH + "/"):
                (offline / Path(name).name).write_bytes(archive.extractfile(name).read())
        provenance = tmp_path / "hub.json"
        provenance.write_bytes(archive.extractfile(deployment.SECURITY_CHECKPOINT_HUB).read())
    monkeypatch.setattr(hub, "download_security_checkpoint", lambda **kwargs: pytest.fail("runtime downloaded"))
    selected = _security_runtime_inputs(security_checkpoint=offline,
        security_checkpoint_manifest_sha256=checkpoint["manifest_sha256"], security_checkpoint_hub_descriptor=provenance,
        security_initializer=None, canonical_cve_export=None, canonical_cve_manifest_sha256=None)
    assert selected["security_checkpoint_hub"] == descriptor and selected["train_autoencoder"] is False


@pytest.mark.parametrize("lake,damage", [("world-lake", "delete_catalog"),
    ("metadata-lake", "delete_catalog"), ("world-lake", "corrupt_parquet"),
    ("metadata-lake", "corrupt_parquet")])
def test_lake_drift_invalidates_advice_and_refuses_refresh(original, checkpoint, monkeypatch, lake, damage):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state)
    receipt = prep.initial_context(state=state, security_checkpoint=checkpoint)
    advice = receipt["security_autoencoder_advice"]
    assert advice["hydration"]["ducklake_verified"] is True
    output = Path(advice["output"])
    if damage == "delete_catalog":
        (output / lake / "metadata.ducklake").unlink()
    else:
        parquet = next((output / lake / "parquet").rglob("*.parquet"))
        parquet.write_bytes(b"corrupted projected observations")
    with pytest.raises(ValueError, match="inventory"):
        advisor.validate_security_advice(repository=root, expected_receipt=advice)
    with pytest.raises(ValueError, match="inventory"):
        advisor.refresh_security_advice(repository=root, previous=advice, output=state / "forbidden-refresh")
    assert not (state / "forbidden-refresh").exists()
    _version(monkeypatch)
    calls = []
    with pytest.raises(ValueError, match="inventory"):
        prep.plan(state, provider_callable=lambda *args, **kwargs: calls.append(args))
    assert calls == [] and not (state / "planner-invoked.json").exists()
