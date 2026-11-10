"""Bounded local tokenizer bundles for an explicit Harbor metadata experiment.

Packaging validates JSON and pins bytes; it never loads a tokenizer runtime,
model, weights or remote asset. The private original descriptor remains on the
host. Only its relocated descriptor and unchanged asset need worker upload.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer as local

SELECTION_SCHEMA = "terminal-metadata-tokenizer-selection@1"
MANIFEST_SCHEMA = "terminal-metadata-tokenizer-bundle@1"
OBSERVATION_SCHEMA = "terminal-metadata-tokenizer-bundle-observation@1"
MAX_MANIFEST_BYTES = 16_384
MAX_JSON_NODES = 2_000_000
MAX_JSON_DEPTH = 64
FILENAMES = ("manifest.json", "original-descriptor.json", "descriptor.json", "tokenizer.json")
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_metadata_tokenizer_bundle.py",
    "benchmarks/agent_supervisor/container_coding/terminal_metadata_tokenizer_deployment.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/local_prompt_tokenizer.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_view.py",
)
_SELECTION_FIELDS = frozenset({"schema", "bundle_path", "manifest_sha256", "provider", "model"})
_FALSE = dict(tokenizer_runtime_loaded=False, provider_tokenizer_equivalence_verified=False,
              provider_usage_verified=False, complete_run_token_savings_claimed=False,
              completion_authority=False)
_MANIFEST_FIELDS = frozenset({"schema", "provider", "model", "tokenizer_id", "tokenizer_revision",
    "special_token_policy", "backend", "tokenizers_version", "original_descriptor_sha256",
    "original_descriptor_bytes", "descriptor_sha256", "descriptor_bytes", "asset_sha256", "asset_bytes",
    "qualification", "compatibility_basis", *_FALSE})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,255}\Z")


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(value, label):
    _need(type(value) is str and _ID.fullmatch(value), "invalid_tokenizer_" + label)
    return value


def _absolute(value):
    path = local._local_path(value)
    _need(path.is_absolute(), "absolute_tokenizer_bundle_path_required")
    return path


def _selection(value):
    _need(type(value) is dict and set(value) == _SELECTION_FIELDS
          and value["schema"] == SELECTION_SCHEMA, "closed_tokenizer_bundle_selection_required")
    _need(type(value["bundle_path"]) is str and Path(value["bundle_path"]).is_absolute(),
          "absolute_tokenizer_bundle_path_required")
    path = _absolute(value["bundle_path"])
    _need(str(path) == value["bundle_path"], "canonical_tokenizer_bundle_path_required")
    local._sha(value["manifest_sha256"], "tokenizer bundle manifest")
    _identity(value["provider"], "provider")
    _identity(value["model"], "model")
    return dict(value)


def validate_metadata_tokenizer_selection(selection, *, provider, model, semantic_metadata_view,
        arm="full", semantic_transport_schema="supervisor-semantic-router-input@1",
        metadata_census_only=False):
    """Validate the closed control and route without opening host files."""
    if selection is None:
        return None
    selected = _selection(selection)
    _need(arm == "full" and provider in {"codex_cli", "grok_cli"}
          and semantic_metadata_view in {"common-bindings@1", "common-bindings-instruction@1", "capsule-columns@1"}
          and semantic_transport_schema == "supervisor-semantic-router-input@1"
          and metadata_census_only is False, "tokenizer_bundle_requires_full_metadata_model_route")
    _need(selected["provider"] == provider and selected["model"] == model,
          "tokenizer_bundle_provider_model_mismatch")
    return selected


def _json(raw, label):
    value = local._json_object(raw, label)
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH, "tokenizer_json_structure_bound")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
    return value


def _descriptor(raw, *, provider, model):
    _need(type(raw) is bytes and 0 < len(raw) <= local.MAX_DESCRIPTOR_BYTES, "tokenizer_descriptor_byte_bound")
    descriptor = _json(raw, "tokenizer bundle descriptor")
    _need(set(descriptor) == local._FIELDS and descriptor.get("schema") == local.SCHEMA,
          "closed_local_tokenizer_descriptor_required")
    for key in local._FIELDS:
        local._text(descriptor[key], key)
    _need(descriptor["provider"] == provider and descriptor["model"] == model,
          "tokenizer_bundle_provider_model_mismatch")
    _identity(descriptor["tokenizer_id"], "id")
    _need(descriptor["backend"] == "tokenizers-json@1"
          and descriptor["special_token_policy"] in local._POLICIES,
          "explicit_local_tokenizer_backend_and_policy_required")
    local._local_path(descriptor["asset_path"], base=Path("/"))
    local._sha(descriptor["asset_sha256"], "tokenizer bundle asset")
    revision = "tokenizers/" + descriptor["tokenizers_version"] + "/sha256:" + descriptor["asset_sha256"]
    _need(descriptor["tokenizer_revision"] == revision, "tokenizer_bundle_revision_mismatch")
    # This is the existing metadata counter's identity shape, including its
    # policy hash. Runtime-version availability is checked by the live loader.
    counter_revision = revision + "/special-policy-sha256:" + _sha(descriptor["special_token_policy"].encode())
    _identity(counter_revision, "counter_revision")
    return descriptor


def _asset(raw):
    _need(type(raw) is bytes and 0 < len(raw) <= local.MAX_TOKENIZER_BYTES, "tokenizer_asset_byte_bound")
    value = _json(raw, "tokenizer bundle asset")
    _need("truncation" in value and "padding" in value
          and value["truncation"] is None and value["padding"] is None,
          "tokenizer_bundle_requires_null_truncation_and_padding")


def _manifest(original, descriptor, asset, *, provider, model):
    before = _descriptor(original, provider=provider, model=model)
    after = _descriptor(descriptor, provider=provider, model=model)
    _need(after == {**before, "asset_path": "tokenizer.json"}, "tokenizer_bundle_descriptor_relocation_mismatch")
    _asset(asset)
    _need(_sha(asset) == before["asset_sha256"], "tokenizer_bundle_asset_hash_mismatch")
    return dict(schema=MANIFEST_SCHEMA,
        **{key: before[key] for key in ("provider", "model", "tokenizer_id", "tokenizer_revision",
            "special_token_policy", "backend", "tokenizers_version")},
        original_descriptor_sha256=_sha(original), original_descriptor_bytes=len(original),
        descriptor_sha256=_sha(descriptor), descriptor_bytes=len(descriptor),
        asset_sha256=_sha(asset), asset_bytes=len(asset), qualification="estimate",
        compatibility_basis="owner_declaration", **_FALSE)


@dataclass(frozen=True, slots=True)
class MetadataTokenizerBundleSnapshot:
    """Captured bytes; this object grants no dispatch or authenticity authority."""
    selection_bytes: bytes
    manifest_bytes: bytes
    original_descriptor_bytes: bytes
    descriptor_bytes: bytes
    asset_bytes: bytes

    def selection(self):
        return json.loads(self.selection_bytes)

    def manifest(self):
        return json.loads(self.manifest_bytes)

    def files(self):
        return tuple(zip(FILENAMES, (self.manifest_bytes, self.original_descriptor_bytes,
                                    self.descriptor_bytes, self.asset_bytes)))

    def observation(self):
        return metadata_tokenizer_observation(self)


def verify_metadata_tokenizer_bundle_snapshot(snapshot, *, expected_selection=None):
    """Revalidate exact captured bytes without reopening any source files."""
    _need(type(snapshot) is MetadataTokenizerBundleSnapshot, "exact_tokenizer_bundle_snapshot_required")
    for field in ("selection_bytes", "manifest_bytes", "original_descriptor_bytes", "descriptor_bytes", "asset_bytes"):
        _need(type(getattr(snapshot, field)) is bytes, "immutable_tokenizer_bundle_bytes_required")
    _need(0 < len(snapshot.selection_bytes) <= MAX_MANIFEST_BYTES
          and 0 < len(snapshot.manifest_bytes) <= MAX_MANIFEST_BYTES, "tokenizer_bundle_manifest_byte_bound")
    selected = _selection(_json(snapshot.selection_bytes, "tokenizer bundle selection"))
    _need(_raw(selected) == snapshot.selection_bytes, "canonical_tokenizer_bundle_selection_required")
    if expected_selection is not None:
        _need(selected == _selection(expected_selection), "tokenizer_bundle_selection_mismatch")
    _need(_sha(snapshot.manifest_bytes) == selected["manifest_sha256"], "tokenizer_bundle_manifest_hash_mismatch")
    manifest = _json(snapshot.manifest_bytes, "tokenizer bundle manifest")
    _need(set(manifest) == _MANIFEST_FIELDS and manifest.get("schema") == MANIFEST_SCHEMA,
          "closed_tokenizer_bundle_manifest_required")
    expected = _manifest(snapshot.original_descriptor_bytes, snapshot.descriptor_bytes, snapshot.asset_bytes,
                         provider=selected["provider"], model=selected["model"])
    _need(_raw(expected) == snapshot.manifest_bytes, "tokenizer_bundle_manifest_regeneration_mismatch")
    return snapshot


def metadata_tokenizer_observation(snapshot):
    """Expose independent content pins and declared proxy scope, without paths."""
    verify_metadata_tokenizer_bundle_snapshot(snapshot)
    return {**snapshot.manifest(), "schema": OBSERVATION_SCHEMA,
            "manifest_sha256": snapshot.selection()["manifest_sha256"]}


@contextmanager
def _directory(path):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = None
    try:
        descriptor = os.open(path.anchor, flags)
        for part in path.parts[1:]:
            following = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    except OSError as error:
        raise ValueError("tokenizer_bundle_directory_unavailable_or_symlink") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _inventory(directory):
    names = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            _need(entry.name in FILENAMES and len(names) < len(FILENAMES),
                  "tokenizer_bundle_file_population_mismatch")
            names.add(entry.name)
    _need(names == set(FILENAMES), "tokenizer_bundle_file_population_mismatch")


def stage_metadata_tokenizer_bundle(*, artifact, expected_sha256, output, provider, model):
    """Capture authenticated inputs, then create a fresh private local bundle.

    A failed write preserves the newly created directory for inspection. An
    existing path is never overwritten, removed, repaired or cleaned up.
    """
    _identity(provider, "provider")
    _identity(model, "model")
    _need(provider in {"codex_cli", "grok_cli"}, "supported_tokenizer_bundle_provider_required")
    local._sha(expected_sha256, "original tokenizer descriptor")
    source, target = _absolute(artifact), _absolute(output)
    original = local._bounded_local_bytes(source, maximum=local.MAX_DESCRIPTOR_BYTES,
                                          expected_sha256=expected_sha256)
    parsed = _descriptor(original, provider=provider, model=model)
    asset_path = local._local_path(parsed["asset_path"], base=source.parent)
    asset = local._bounded_local_bytes(asset_path, maximum=local.MAX_TOKENIZER_BYTES,
                                       expected_sha256=parsed["asset_sha256"])
    descriptor = _raw({**parsed, "asset_path": "tokenizer.json"})
    manifest = _raw(_manifest(original, descriptor, asset, provider=provider, model=model))
    selected = dict(schema=SELECTION_SCHEMA, bundle_path=str(target), manifest_sha256=_sha(manifest),
                    provider=provider, model=model)
    snapshot = MetadataTokenizerBundleSnapshot(_raw(selected), manifest, original, descriptor, asset)
    verify_metadata_tokenizer_bundle_snapshot(snapshot, expected_selection=selected)
    with _directory(target.parent) as parent_fd:
        try:
            os.mkdir(target.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise ValueError("fresh_tokenizer_bundle_directory_required") from error
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        child = os.open(target.name, flags, dir_fd=parent_fd)
        try:
            before = os.fstat(child)
            _need(stat.S_ISDIR(before.st_mode) and before.st_uid == os.geteuid()
                  and not stat.S_IMODE(before.st_mode) & 0o077, "private_owned_tokenizer_bundle_required")
            for name, raw in snapshot.files():
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o400, dir_fd=child)
                try:
                    position = 0
                    while position < len(raw):
                        written = os.write(fd, raw[position:position + 65536])
                        _need(written > 0, "tokenizer_bundle_write_incomplete")
                        position += written
                    os.fsync(fd)
                finally:
                    os.close(fd)
            _inventory(child)
            after = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            _need((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), "tokenizer_bundle_directory_changed")
            with _directory(target) as current:
                closing = os.fstat(current)
                _need((before.st_dev, before.st_ino) == (closing.st_dev, closing.st_ino),
                      "tokenizer_bundle_directory_changed")
            os.fsync(child)
            os.fsync(parent_fd)
        finally:
            os.close(child)
    return selected


def read_metadata_tokenizer_bundle(selection, **route):
    """Capture every declared file once; uploads use only returned bytes."""
    selected = validate_metadata_tokenizer_selection(selection, **route)
    if selected is None:
        return None
    path = _absolute(selected["bundle_path"])
    with _directory(path) as directory:
        before = os.fstat(directory)
        _need(before.st_uid == os.geteuid() and not stat.S_IMODE(before.st_mode) & 0o077,
              "private_owned_tokenizer_bundle_required")
        _inventory(directory)
        manifest = local._bounded_local_bytes(path / "manifest.json", maximum=MAX_MANIFEST_BYTES,
                                              expected_sha256=selected["manifest_sha256"])
        parsed = _json(manifest, "tokenizer bundle manifest")
        _need(set(parsed) == _MANIFEST_FIELDS and parsed.get("schema") == MANIFEST_SCHEMA,
              "closed_tokenizer_bundle_manifest_required")
        captured = []
        for name, pin, bound in (("original-descriptor.json", "original_descriptor_sha256", local.MAX_DESCRIPTOR_BYTES),
            ("descriptor.json", "descriptor_sha256", local.MAX_DESCRIPTOR_BYTES),
            ("tokenizer.json", "asset_sha256", local.MAX_TOKENIZER_BYTES)):
            local._sha(parsed[pin], "tokenizer bundle file")
            captured.append(local._bounded_local_bytes(path / name, maximum=bound, expected_sha256=parsed[pin]))
        _inventory(directory)
        with _directory(path) as current:
            after = os.fstat(current)
            _need((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), "tokenizer_bundle_directory_changed")
    return verify_metadata_tokenizer_bundle_snapshot(MetadataTokenizerBundleSnapshot(
        _raw(selected), manifest, *captured), expected_selection=selected)


def require_metadata_tokenizer_archive(manifest, selection):
    """Require the selected adapter and its actual producer sources in the archive."""
    if selection is None:
        return None
    selected = _selection(selection)
    rows = manifest.get("files") if type(manifest) is dict else None
    _need(type(rows) is list and len(rows) <= 250_000, "tokenizer_bundle_requires_current_runtime_archive")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        _need(len(matches) == 1, "tokenizer_bundle_requires_current_runtime_archive")
        pin = local._sha(matches[0].get("sha256"), "tokenizer bundle producer source")
        raw = local._bounded_local_bytes(checkout / relative, maximum=2 * 1024 * 1024, expected_sha256=pin)
        _need("bytes" not in matches[0] or type(matches[0]["bytes"]) is int
              and matches[0]["bytes"] == len(raw), "tokenizer_bundle_source_size_mismatch")
    return selected
