"""Bounded authored JSON bundles; no tokenizer runtime or provider calls."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_metadata_tokenizer_bundle as bundle


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / "original-inputs"
    folder.mkdir()
    asset = folder / "original-tokenizer.json"
    asset.write_bytes(_wire(dict(version="1.0", truncation=None, padding=None, added_tokens=[],
        normalizer=None, pre_tokenizer={"type": "Whitespace"}, post_processor=None, decoder=None,
        model={"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1}, "unk_token": "[UNK]"})))
    asset_pin = _sha(asset.read_bytes())
    descriptor = dict(schema=bundle.local.SCHEMA, provider="codex_cli", model="authored-model",
        tokenizer_id="authored-local-proxy", tokenizer_revision="tokenizers/999.0/sha256:" + asset_pin,
        special_token_policy="tokenizers.encode:add_special_tokens=false", backend="tokenizers-json@1",
        tokenizers_version="999.0", asset_path=str(asset), asset_sha256=asset_pin)
    artifact = folder / "original-descriptor.json"
    artifact.write_bytes(json.dumps(descriptor, indent=2).encode())
    return dict(folder=folder, artifact=artifact, asset=asset, descriptor=descriptor,
                output=tmp_path / "private-bundle")


def stage(inputs, **changes):
    options = dict(artifact=inputs["artifact"], expected_sha256=_sha(inputs["artifact"].read_bytes()),
        output=inputs["output"], provider="codex_cli", model="authored-model")
    options.update(changes)
    return bundle.stage_metadata_tokenizer_bundle(**options)


def read(selection, **changes):
    options = dict(provider="codex_cli", model="authored-model", semantic_metadata_view="common-bindings@1")
    options.update(changes)
    return bundle.read_metadata_tokenizer_bundle(selection, **options)


def rewrite_descriptor(inputs, **changes):
    inputs["descriptor"].update(changes)
    inputs["artifact"].write_bytes(_wire(inputs["descriptor"]))


def test_stages_exact_original_and_portable_bytes_without_loading_runtime(inputs, monkeypatch):
    monkeypatch.setattr(bundle.local, "_runtime", lambda *_: pytest.fail("staging loaded tokenizer runtime"))
    original = inputs["artifact"].read_bytes()
    asset = inputs["asset"].read_bytes()
    selected = stage(inputs)
    snapshot = read(selected)
    assert snapshot.original_descriptor_bytes == original
    assert snapshot.asset_bytes == asset
    relocated = json.loads(snapshot.descriptor_bytes)
    assert relocated == {**inputs["descriptor"], "asset_path": "tokenizer.json"}
    manifest = snapshot.manifest()
    assert manifest["original_descriptor_sha256"] == _sha(original)
    assert manifest["descriptor_sha256"] == _sha(snapshot.descriptor_bytes)
    assert manifest["descriptor_sha256"] != manifest["original_descriptor_sha256"]
    assert manifest["asset_sha256"] == _sha(asset)
    assert selected["manifest_sha256"] == _sha(snapshot.manifest_bytes)
    assert manifest["tokenizers_version"] == "999.0"  # Runtime availability is a later dispatch gate.
    assert manifest["qualification"] == "estimate"
    assert manifest["tokenizer_runtime_loaded"] is False
    assert tuple(name for name, _ in snapshot.files()) == bundle.FILENAMES
    assert set(path.name for path in inputs["output"].iterdir()) == set(bundle.FILENAMES)
    assert stat.S_IMODE(inputs["output"].stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o400 for path in inputs["output"].iterdir())


def test_observation_is_content_free_detached_and_explicitly_unverified(inputs):
    snapshot = read(stage(inputs))
    observed = snapshot.observation()
    assert observed == bundle.metadata_tokenizer_observation(snapshot)
    assert observed["schema"] == bundle.OBSERVATION_SCHEMA
    assert observed["manifest_sha256"] == snapshot.selection()["manifest_sha256"]
    assert str(inputs["folder"]) not in json.dumps(observed)
    assert "[UNK]" not in json.dumps(observed)
    for key in ("tokenizer_runtime_loaded", "provider_tokenizer_equivalence_verified", "provider_usage_verified",
                "complete_run_token_savings_claimed", "completion_authority"):
        assert observed[key] is False
    observed["model"] = "changed"
    assert snapshot.observation()["model"] == "authored-model"
    snapshot.selection()["model"] = "changed"
    assert snapshot.selection()["model"] == "authored-model"
    with pytest.raises(FrozenInstanceError):
        snapshot.asset_bytes = b"changed"


def test_snapshot_remains_usable_after_original_sources_are_removed(inputs, monkeypatch):
    selected = stage(inputs)
    shutil.rmtree(inputs["folder"])
    snapshot = read(selected)
    monkeypatch.setattr(bundle.local, "_bounded_local_bytes", lambda *_a, **_k: pytest.fail("snapshot reopened source"))
    assert bundle.verify_metadata_tokenizer_bundle_snapshot(snapshot, expected_selection=selected) is snapshot
    assert len(snapshot.files()) == 4
    assert snapshot.observation()["asset_sha256"] == _sha(snapshot.asset_bytes)


def test_source_relocation_does_not_change_manifest_or_expose_host_paths(inputs, tmp_path):
    selected = stage(inputs)
    first = read(selected)
    other = stage(inputs, output=tmp_path / "second-bundle")
    assert other["bundle_path"] != selected["bundle_path"]
    assert other["manifest_sha256"] == selected["manifest_sha256"]
    assert read(other).observation() == first.observation()


@pytest.mark.parametrize("changes", [{"arm": "no-index"}, {"provider": "grok_cli"},
    {"model": "other"}, {"semantic_metadata_view": "legacy"}, {"semantic_metadata_view": "foreign"},
    {"semantic_transport_schema": "supervisor-semantic-router-input@2"}, {"metadata_census_only": True},
    {"metadata_census_only": None}, {"metadata_census_only": 0}])
def test_selection_requires_exact_full_metadata_route(inputs, changes):
    selected = stage(inputs)
    with pytest.raises(ValueError):
        read(selected, **changes)


def test_none_selection_preserves_defaults_without_inspecting_files(monkeypatch):
    monkeypatch.setattr(bundle, "_directory", lambda *_: pytest.fail("default inspected files"))
    assert bundle.validate_metadata_tokenizer_selection(None, provider=None, model=None,
        semantic_metadata_view="legacy", arm="no-index") is None
    assert bundle.read_metadata_tokenizer_bundle(None, provider=None, model=None,
        semantic_metadata_view="legacy") is None
    assert bundle.require_metadata_tokenizer_archive(None, None) is None


@pytest.mark.parametrize("field,value", [("extra", False), ("manifest_sha256", "A" * 64),
    ("manifest_sha256", None), ("bundle_path", "relative/path"), ("bundle_path", "/tmp/../bundle"),
    ("schema", "foreign"), ("model", True)])
def test_closed_selection_rejects_unknown_fields_bad_pins_and_paths(inputs, field, value):
    selected = stage(inputs)
    selected[field] = value
    with pytest.raises(ValueError):
        read(selected)


@pytest.mark.parametrize("changes", [{"extra": "hidden"}, {"backend": "from-pretrained"},
    {"special_token_policy": "guess"}, {"tokenizer_revision": "unpinned"}, {"asset_sha256": "A" * 64},
    {"asset_path": "../tokenizer.json"}, {"asset_path": "https://example.invalid/tokenizer.json"},
    {"provider": "grok_cli"}, {"model": "other"}, {"tokenizer_id": "invalid id"}])
def test_bad_original_descriptor_refuses_before_creating_output(inputs, changes):
    rewrite_descriptor(inputs, **changes)
    with pytest.raises(ValueError):
        stage(inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("raw", [b"{", b'{"schema":"a","schema":"b"}', b'{"model":NaN}', b'{"model":1e999}'])
def test_malformed_pinned_original_json_refuses_before_output(inputs, raw):
    inputs["artifact"].write_bytes(raw)
    with pytest.raises(ValueError):
        stage(inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("changes", [{"padding": {}}, {"truncation": {}}, {"padding": False}])
def test_asset_cannot_hide_padding_or_truncation(inputs, changes):
    value = json.loads(inputs["asset"].read_bytes())
    value.update(changes)
    raw = _wire(value)
    inputs["asset"].write_bytes(raw)
    pin = _sha(raw)
    rewrite_descriptor(inputs, asset_sha256=pin, tokenizer_revision="tokenizers/999.0/sha256:" + pin)
    with pytest.raises(ValueError, match="null_truncation_and_padding"):
        stage(inputs)
    assert not inputs["output"].exists()


def test_missing_explicit_padding_is_rejected(inputs):
    value = json.loads(inputs["asset"].read_bytes())
    del value["padding"]
    raw = _wire(value)
    inputs["asset"].write_bytes(raw)
    pin = _sha(raw)
    rewrite_descriptor(inputs, asset_sha256=pin, tokenizer_revision="tokenizers/999.0/sha256:" + pin)
    with pytest.raises(ValueError, match="null_truncation_and_padding"):
        stage(inputs)


def test_wrong_original_or_asset_pin_is_not_repaired(inputs):
    with pytest.raises(ValueError):
        stage(inputs, expected_sha256="0" * 64)
    inputs["asset"].write_bytes(inputs["asset"].read_bytes() + b" ")
    with pytest.raises(ValueError):
        stage(inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("which", ["descriptor", "asset", "ancestor"])
def test_symbolic_link_inputs_and_ancestors_are_refused(inputs, tmp_path, which):
    if which == "descriptor":
        link = tmp_path / "descriptor-link"
        link.symlink_to(inputs["artifact"])
        options = {"artifact": link}
    elif which == "asset":
        link = tmp_path / "asset-link"
        link.symlink_to(inputs["asset"])
        rewrite_descriptor(inputs, asset_path=str(link))
        options = {}
    else:
        link = tmp_path / "parent-link"
        link.symlink_to(inputs["folder"], target_is_directory=True)
        options = {"artifact": link / inputs["artifact"].name}
    with pytest.raises(ValueError):
        stage(inputs, **options)
    assert not inputs["output"].exists()


def test_existing_output_is_never_overwritten_or_cleaned(inputs):
    inputs["output"].mkdir()
    marker = inputs["output"] / "foreign.txt"
    marker.write_text("preserve existing data")
    with pytest.raises(ValueError, match="fresh_tokenizer_bundle"):
        stage(inputs)
    assert marker.read_text() == "preserve existing data"
    assert list(inputs["output"].iterdir()) == [marker]


def test_output_symlink_and_symlink_parent_are_not_followed(inputs, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    inputs["output"].symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError):
        stage(inputs)
    assert list(elsewhere.iterdir()) == []
    with pytest.raises(ValueError):
        stage(inputs, output=inputs["output"] / "nested")
    assert list(elsewhere.iterdir()) == []


def test_failed_write_preserves_new_directory_for_inspection(inputs, monkeypatch):
    def fail(*_):
        raise OSError("authored disk full")
    monkeypatch.setattr(bundle.os, "write", fail)
    with pytest.raises(ValueError):
        stage(inputs)
    assert inputs["output"].is_dir()
    assert (inputs["output"] / "manifest.json").exists()


@pytest.mark.parametrize("which", ["manifest.json", "original-descriptor.json", "descriptor.json", "tokenizer.json"])
def test_staged_file_changes_are_refused(inputs, which):
    selected = stage(inputs)
    path = inputs["output"] / which
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        read(selected)


def test_extra_bundle_member_and_public_directory_are_refused(inputs):
    selected = stage(inputs)
    foreign = inputs["output"] / "extra.txt"
    foreign.write_text("unexpected")
    with pytest.raises(ValueError, match="file_population"):
        read(selected)
    foreign.unlink()
    inputs["output"].chmod(0o755)
    with pytest.raises(ValueError, match="private_owned"):
        read(selected)


@pytest.mark.parametrize("field", ["manifest_bytes", "original_descriptor_bytes", "descriptor_bytes", "asset_bytes"])
def test_snapshot_mutations_do_not_bypass_independent_manifest_pin(inputs, field):
    selected = stage(inputs)
    snapshot = read(selected)
    changed = replace(snapshot, **{field: getattr(snapshot, field) + b" "})
    with pytest.raises(ValueError):
        bundle.verify_metadata_tokenizer_bundle_snapshot(changed, expected_selection=selected)


def test_snapshot_expected_selection_pin_is_independent(inputs):
    selected = stage(inputs)
    snapshot = read(selected)
    changed = {**selected, "manifest_sha256": "0" * 64}
    with pytest.raises(ValueError, match="selection_mismatch"):
        bundle.verify_metadata_tokenizer_bundle_snapshot(snapshot, expected_selection=changed)
    with pytest.raises(ValueError, match="exact_tokenizer_bundle_snapshot"):
        bundle.verify_metadata_tokenizer_bundle_snapshot(object())
    with pytest.raises(ValueError, match="immutable"):
        bundle.verify_metadata_tokenizer_bundle_snapshot(replace(snapshot, asset_bytes=bytearray(snapshot.asset_bytes)))


def test_portable_descriptor_cannot_change_business_bindings_even_if_resealed(inputs):
    selected = stage(inputs)
    snapshot = read(selected)
    descriptor = json.loads(snapshot.descriptor_bytes)
    descriptor["special_token_policy"] = "tokenizers.encode:add_special_tokens=true"
    descriptor_raw = _wire(descriptor)
    manifest = snapshot.manifest()
    manifest.update(descriptor_sha256=_sha(descriptor_raw), descriptor_bytes=len(descriptor_raw))
    manifest_raw = _wire(manifest)
    self_pinned = {**selected, "manifest_sha256": _sha(manifest_raw)}
    changed = replace(snapshot, selection_bytes=_wire(self_pinned), manifest_bytes=manifest_raw,
                      descriptor_bytes=descriptor_raw)
    with pytest.raises(ValueError, match="relocation_mismatch"):
        bundle.verify_metadata_tokenizer_bundle_snapshot(changed)


def test_selected_archive_binds_actual_shared_sources_and_new_deployment(inputs):
    selected = stage(inputs)
    root = Path(bundle.__file__).resolve().parents[3]
    rows = []
    for relative in bundle.SOURCE_PATHS:
        raw = (root / relative).read_bytes()
        rows.append(dict(path="source/" + relative, sha256=_sha(raw), bytes=len(raw)))
    assert bundle.require_metadata_tokenizer_archive({"files": rows}, selected) == selected
    assert "benchmarks/agent_supervisor/container_coding/terminal_metadata_tokenizer_deployment.py" in bundle.SOURCE_PATHS
    with pytest.raises(ValueError):
        bundle.require_metadata_tokenizer_archive({"files": rows[1:]}, selected)
    bad = [{**row, "sha256": "0" * 64} if i == 0 else row for i, row in enumerate(rows)]
    with pytest.raises(ValueError):
        bundle.require_metadata_tokenizer_archive({"files": bad}, selected)


def test_descriptor_size_bound_is_checked_before_any_output(inputs):
    inputs["artifact"].write_bytes(b" " * (bundle.local.MAX_DESCRIPTOR_BYTES + 1))
    with pytest.raises(ValueError):
        stage(inputs)
    assert not inputs["output"].exists()
