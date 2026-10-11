"""Actual authored tiny tokenizer fixtures; no provider-compatibility claims."""
import hashlib
import json
from dataclasses import replace
from importlib import metadata

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer as local
from tokenizers import Tokenizer, models, pre_tokenizers, processors

PROVIDER = "codex_cli"
MODEL = "authored-fixture-model"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_descriptor(artifact, descriptor):
    raw = json.dumps(descriptor, ensure_ascii=False, sort_keys=True).encode()
    artifact.write_bytes(raw)
    return digest(raw)


def make_binding(tmp_path, *, provider=PROVIDER, model=MODEL, add_special_tokens=False):
    """Reusable authored asset for route tests; never a real-model tokenizer."""
    tokenizer = Tokenizer(models.WordLevel({
        "[UNK]": 0, "[START]": 1, "[END]": 2,
        "hello": 3, "world": 4, "café": 5, "法律": 6,
    }, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["[START]", "[END]"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[START] $A [END]", special_tokens=[("[START]", 1), ("[END]", 2)])
    asset = tmp_path / "tokenizer.json"
    raw = tokenizer.to_str().encode()
    asset.write_bytes(raw)
    version = metadata.version("tokenizers")
    descriptor = {
        "schema": local.SCHEMA, "provider": provider, "model": model,
        "tokenizer_id": "authored-tiny-wordlevel-fixture",
        "tokenizer_revision": "tokenizers/" + version + "/sha256:" + digest(raw),
        "special_token_policy": (
            "tokenizers.encode:add_special_tokens=" + str(add_special_tokens).lower()),
        "backend": "tokenizers-json@1", "tokenizers_version": version,
        "asset_path": asset.name, "asset_sha256": digest(raw),
    }
    artifact = tmp_path / "owner-tokenizer.json"
    expected = write_descriptor(artifact, descriptor)
    return artifact, expected, descriptor


@pytest.fixture
def owner_fixture(tmp_path):
    artifact, _, descriptor = make_binding(tmp_path)
    asset = artifact.parent / descriptor["asset_path"]
    return artifact, asset, descriptor


def load(fixture, **overrides):
    artifact, _, _ = fixture
    expected = overrides.get("expected_sha256")
    if expected is None:
        expected = digest(artifact.read_bytes())
    return local.load_owner_prompt_tokenizer(**{
        "artifact": artifact, "expected_sha256": expected,
        "provider": PROVIDER, "model": MODEL, **overrides,
    })


def reseal_asset(fixture, value):
    artifact, asset, descriptor = fixture
    raw = json.dumps(value, ensure_ascii=False).encode()
    asset.write_bytes(raw)
    descriptor["asset_sha256"] = digest(raw)
    descriptor["tokenizer_revision"] = (
        "tokenizers/" + descriptor["tokenizers_version"] + "/sha256:" + digest(raw))
    write_descriptor(artifact, descriptor)


@pytest.mark.parametrize("special_tokens, expected", [(False, 4), (True, 6)])
def test_complete_unicode_text_and_exact_special_token_policy(owner_fixture, special_tokens, expected):
    artifact, _, descriptor = owner_fixture
    descriptor["special_token_policy"] = (
        "tokenizers.encode:add_special_tokens=" + str(special_tokens).lower())
    write_descriptor(artifact, descriptor)
    loaded = load(owner_fixture)
    assert loaded.count("hello world café 法律") == expected
    assert loaded.count("[START] hello [END]") == (5 if special_tokens else 3)
    assert loaded.binding() == {name: descriptor[name] for name in (
        "provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy")}
    binding = loaded.binding()
    binding["model"] = "changed-copy"
    assert loaded.binding()["model"] == MODEL
    provenance = loaded.provenance()
    assert provenance == {
        "schema": "supervisor-local-prompt-tokenizer-provenance@1",
        "backend": "tokenizers-json@1", "descriptor_path": str(artifact),
        "descriptor_sha256": digest(artifact.read_bytes()),
        "asset_path": str(owner_fixture[1]), "asset_sha256": descriptor["asset_sha256"],
        "tokenizers_version": descriptor["tokenizers_version"],
        "compatibility_basis": "owner_declaration", "completion_authority": False,
    }


def test_absolute_asset_is_an_explicit_owner_choice(owner_fixture):
    artifact, asset, descriptor = owner_fixture
    descriptor["asset_path"] = str(asset)
    write_descriptor(artifact, descriptor)
    assert load(owner_fixture).count("hello world") == 2


@pytest.mark.parametrize("change", ["provider", "model"])
def test_wrong_provider_or_model_refuses(owner_fixture, change):
    with pytest.raises(ValueError, match="pinned provider/model"):
        load(owner_fixture, **{change: "other"})


@pytest.mark.parametrize("mutation", [
    {"extra": "forbidden"}, {"backend": "from-pretrained"},
    {"special_token_policy": "guess"}, {"tokenizer_id": " "},
    {"tokenizer_revision": "sha256:old-format"}, {"asset_sha256": "F" * 64},
    {"tokenizers_version": True}, {"model": " fixture-model "},
])
def test_closed_descriptor_and_exact_fields(owner_fixture, mutation):
    artifact, _, descriptor = owner_fixture
    descriptor.update(mutation)
    write_descriptor(artifact, descriptor)
    with pytest.raises(ValueError):
        load(owner_fixture)


def test_missing_descriptor_field_refuses(owner_fixture):
    artifact, _, descriptor = owner_fixture
    del descriptor["backend"]
    write_descriptor(artifact, descriptor)
    with pytest.raises(ValueError, match="closed"):
        load(owner_fixture)


def test_runtime_version_mismatch_refuses_even_with_matching_revision(owner_fixture):
    artifact, _, descriptor = owner_fixture
    descriptor["tokenizers_version"] = "0.0.0"
    descriptor["tokenizer_revision"] = "tokenizers/0.0.0/sha256:" + descriptor["asset_sha256"]
    write_descriptor(artifact, descriptor)
    with pytest.raises(ValueError, match="runtime version"):
        load(owner_fixture)


def test_unavailable_runtime_refuses_without_loading_alternative(owner_fixture, monkeypatch):
    def unavailable(_):
        raise metadata.PackageNotFoundError("authored-runtime-missing")
    monkeypatch.setattr(local.metadata, "version", unavailable)
    with pytest.raises(ValueError, match="runtime is unavailable"):
        load(owner_fixture)


@pytest.mark.parametrize("raw", [
    b'{"schema":"one","schema":"two"}', b'{"model":NaN}', b'{"model":Infinity}',
    b"[]", b"\xff", b"{", b'{"nested":{"key":1,"key":2}}', b'{"model":1e999}',
])
def test_malformed_duplicate_or_nonfinite_descriptor_refuses(owner_fixture, raw):
    artifact, _, _ = owner_fixture
    artifact.write_bytes(raw)
    with pytest.raises(ValueError):
        load(owner_fixture)


@pytest.mark.parametrize("field, value", [
    ("truncation", {}), ("padding", {}), ("truncation", "missing"), ("padding", "missing"),
])
def test_asset_must_explicitly_disable_truncation_and_padding(owner_fixture, field, value):
    asset_object = json.loads(owner_fixture[1].read_bytes())
    if value == "missing":
        del asset_object[field]
    else:
        asset_object[field] = value
    reseal_asset(owner_fixture, asset_object)
    with pytest.raises(ValueError, match="explicit null truncation and padding"):
        load(owner_fixture)


@pytest.mark.parametrize("raw", [
    b'{"truncation":null,"padding":null,"x":NaN}',
    b'{"truncation":null,"padding":null,"x":1e999}',
    b'{"truncation":null,"padding":null,"model":{"x":1,"x":2}}',
    b'{"truncation":null,"padding":null}',
])
def test_malformed_or_incomplete_tokenizer_asset_refuses(owner_fixture, raw):
    artifact, asset, descriptor = owner_fixture
    asset.write_bytes(raw)
    descriptor["asset_sha256"] = digest(raw)
    descriptor["tokenizer_revision"] = (
        "tokenizers/" + descriptor["tokenizers_version"] + "/sha256:" + digest(raw))
    write_descriptor(artifact, descriptor)
    with pytest.raises(ValueError):
        load(owner_fixture)


@pytest.mark.parametrize("path", ["../tokenizer.json", "./tokenizer.json", "a/../tokenizer.json",
    "https://example.invalid/tokenizer.json", "file:///tmp/tokenizer.json", "a//tokenizer.json"])
def test_urls_traversal_and_noncanonical_asset_paths_refuse(owner_fixture, path):
    artifact, _, descriptor = owner_fixture
    descriptor["asset_path"] = path
    write_descriptor(artifact, descriptor)
    with pytest.raises(ValueError, match="canonical local path"):
        load(owner_fixture)


@pytest.mark.parametrize("target", ["descriptor", "asset", "ancestor"])
def test_symbolic_links_are_never_followed(owner_fixture, target):
    artifact, asset, descriptor = owner_fixture
    if target == "ancestor":
        link = artifact.parent.parent / "linked-owner-directory"
        link.symlink_to(artifact.parent, target_is_directory=True)
        with pytest.raises(ValueError, match="symbolic link"):
            load(owner_fixture, artifact=link / artifact.name)
    else:
        path = artifact if target == "descriptor" else asset
        real = path.with_suffix(".real")
        path.rename(real)
        path.symlink_to(real)
        with pytest.raises(ValueError, match="symbolic link"):
            load(owner_fixture)
    assert descriptor["provider"] == PROVIDER


@pytest.mark.parametrize("target", ["descriptor", "asset"])
def test_missing_local_input_refuses(owner_fixture, target):
    artifact, asset, _ = owner_fixture
    if target == "descriptor":
        expected = digest(artifact.read_bytes())
        artifact.unlink()
        overrides = {"expected_sha256": expected}
    else:
        asset.unlink()
        overrides = {}
    with pytest.raises(ValueError, match="unavailable"):
        load(owner_fixture, **overrides)


@pytest.mark.parametrize("target", ["descriptor", "asset"])
def test_wrong_bytes_refuse_before_load(owner_fixture, target):
    artifact, asset, _ = owner_fixture
    expected = digest(artifact.read_bytes())
    path = artifact if target == "descriptor" else asset
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load(owner_fixture, expected_sha256=expected)


@pytest.mark.parametrize("target", ["descriptor", "asset"])
def test_changed_bindings_after_load_never_return_a_count(owner_fixture, target):
    loaded = load(owner_fixture)
    path = owner_fixture[0 if target == "descriptor" else 1]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        loaded.count("hello world")


@pytest.mark.parametrize("target", ["descriptor", "asset"])
def test_binding_drift_during_encode_discards_computed_count(owner_fixture, target):
    loaded = load(owner_fixture)
    actual = loaded._tokenizer
    path = owner_fixture[0 if target == "descriptor" else 1]
    class DriftingTokenizer:
        truncation = padding = None
        def encode(self, text, *, add_special_tokens):
            result = actual.encode(text, add_special_tokens=add_special_tokens)
            path.write_bytes(path.read_bytes() + b" ")
            return result
    drifting = replace(loaded, _tokenizer=DriftingTokenizer())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        drifting.count("hello world")


@pytest.mark.parametrize("setting", ["truncation", "padding"])
def test_in_memory_tokenizer_mutation_refuses_instead_of_disabling_it(owner_fixture, setting):
    loaded = load(owner_fixture)
    if setting == "truncation":
        loaded._tokenizer.enable_truncation(max_length=1)
    else:
        loaded._tokenizer.enable_padding(length=30)
    with pytest.raises(ValueError, match="must remain disabled"):
        loaded.count("hello world")


def test_runtime_drift_after_load_refuses(owner_fixture, monkeypatch):
    loaded = load(owner_fixture)
    monkeypatch.setattr(local.metadata, "version", lambda _: "fixture-changed-runtime")
    with pytest.raises(ValueError, match="runtime version"):
        loaded.count("hello")


def test_complete_prompt_keeps_existing_utf8_byte_ceiling(owner_fixture):
    loaded = load(owner_fixture)
    assert loaded.count("é" * (local.MAX_PROMPT_BYTES // 2)) == 1
    with pytest.raises(ValueError, match="existing byte bound"):
        loaded.count("é" * (local.MAX_PROMPT_BYTES // 2 + 1))
    with pytest.raises(ValueError, match="positive untruncated"):
        loaded.count("")
    with pytest.raises(ValueError, match="existing byte bound"):
        loaded.count(b"hello")


@pytest.mark.parametrize("target", ["descriptor", "asset"])
def test_oversized_local_files_refuse_before_json_or_tokenizer_loading(owner_fixture, target):
    artifact, asset, _ = owner_fixture
    expected = digest(artifact.read_bytes())
    path = artifact if target == "descriptor" else asset
    maximum = local.MAX_DESCRIPTOR_BYTES if target == "descriptor" else local.MAX_TOKENIZER_BYTES
    with path.open("wb") as stream:
        stream.truncate(maximum + 1)
    with pytest.raises(ValueError, match="bounded nonempty regular file"):
        load(owner_fixture, expected_sha256=expected)


def test_fifo_refuses_without_blocking(owner_fixture):
    import os
    artifact, asset, _ = owner_fixture
    asset.unlink()
    os.mkfifo(asset)
    with pytest.raises(ValueError, match="regular file"):
        load(owner_fixture, expected_sha256=digest(artifact.read_bytes()))


def test_regular_file_changed_during_read_refuses(owner_fixture, monkeypatch):
    artifact = owner_fixture[0]
    expected = digest(artifact.read_bytes())
    actual = local.os.fstat
    seen = 0
    def changed(fd):
        nonlocal seen
        seen += 1
        if seen == 2:
            artifact.write_bytes(artifact.read_bytes() + b" ")
        return actual(fd)
    monkeypatch.setattr(local.os, "fstat", changed)
    with pytest.raises(ValueError, match="changed during read"):
        load(owner_fixture, expected_sha256=expected)


def test_symlink_free_nested_local_asset_loads(owner_fixture):
    artifact, asset, descriptor = owner_fixture
    nested = artifact.parent / "local-assets"
    nested.mkdir()
    asset.rename(nested / asset.name)
    descriptor["asset_path"] = "local-assets/" + asset.name
    write_descriptor(artifact, descriptor)
    assert load(owner_fixture).count("hello world") == 2
