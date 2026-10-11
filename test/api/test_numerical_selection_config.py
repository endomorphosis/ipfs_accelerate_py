"""Explicit experimental configuration cannot silently select other weights."""
from copy import deepcopy
import hashlib
import json

import pytest


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _descriptor(tmp_path):
    # Structural validation only: this is deliberately not a trained model.
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding

    return dict(schema="supervisor-numerical-selection-config@1",
        profile="numpy-unary-selection-384@1", allow_experimental=True,
        checkpoint_path=str(tmp_path / "checkpoint.json"), checkpoint_sha256="a" * 64,
        embedding_snapshot=str(tmp_path / "snapshot"),
        embedding_revision="17e1f347d17fe144873b1201da91788898c639cd",
        embedding_assets=[dict(name=name, bytes=size, sha256=digest)
            for name, (size, digest) in sorted(embedding._PINNED_ASSETS.items())],
        training_steps=0, download_calls=0)


@pytest.mark.parametrize("mutation", ["opt_out", "implicit_opt_in", "foreign_profile", "extra_field",
    "training", "boolean_training", "download", "relative_checkpoint", "relative_snapshot",
    "bad_checkpoint_digest", "foreign_embedding_revision", "embedding_asset"])
def test_numerical_config_refuses_unreviewed_or_ambiguous_selection(tmp_path, mutation):
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_config as config

    value = _descriptor(tmp_path)
    assert config.validate_numerical_selection_config(value) == value
    if mutation == "opt_out": value["allow_experimental"] = False
    elif mutation == "implicit_opt_in": del value["allow_experimental"]
    elif mutation == "foreign_profile": value["profile"] = "domain-384-typed-autoencoder/v1"
    elif mutation == "extra_field": value["fallback_to_registered_security"] = True
    elif mutation == "training": value["training_steps"] = 1
    elif mutation == "boolean_training": value["training_steps"] = False
    elif mutation == "download": value["download_calls"] = 1
    elif mutation == "relative_checkpoint": value["checkpoint_path"] = "checkpoint.json"
    elif mutation == "relative_snapshot": value["embedding_snapshot"] = "snapshot"
    elif mutation == "bad_checkpoint_digest": value["checkpoint_sha256"] = "invalid"
    elif mutation == "foreign_embedding_revision": value["embedding_revision"] = "main"
    else: value["embedding_assets"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        config.validate_numerical_selection_config(value)


def test_published_numerical_descriptor_is_offline_and_does_not_choose_registered_default(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_config as config

    checkpoint = tmp_path / "not-downloaded.json"
    snapshot = tmp_path / "not-downloaded-snapshot"
    descriptor = config.published_numerical_selection_config(
        checkpoint_path=checkpoint, embedding_snapshot=snapshot)
    assert descriptor["profile"] == "numpy-unary-selection-384@1"
    assert descriptor["allow_experimental"] is True
    assert descriptor["checkpoint_sha256"] == "26ccc7cf8420e0ed6cbbe22603e09be0e3572fb32a1e489e877fc5081f506e03"
    assert descriptor["training_steps"] == descriptor["download_calls"] == 0
    assert not checkpoint.exists() and not snapshot.exists()
    assert "registered_default" not in descriptor


def test_numerical_config_load_checks_exact_file_before_assets(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_config as config

    value = _descriptor(tmp_path)
    path = tmp_path / "selection.json"
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    calls = []
    monkeypatch.setattr(config, "validate_numerical_selection_assets", lambda item: calls.append(deepcopy(item)))
    config.load_numerical_selection_config(path, expected_sha256=_sha(raw))
    assert calls == [value]
    calls.clear()
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        config.load_numerical_selection_config(path, expected_sha256=_sha(raw))
    assert calls == []
