"""Existing-CAS roundtrips for complete historical read-only code views."""

from dataclasses import asdict, replace
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_code_view_inventory as views
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as sources
from ipfs_accelerate_py.agent_supervisor.runtime import reversible_source_view as codec
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from test.api.test_repository_source_inventory import _commit_request, _git, _repository, _retain, _chunk_cid


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _manifest_sha(manifest):
    return _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())


def _no_call(*args, **kwargs):
    pytest.fail("read-only historical views activated an unadmitted source/store producer")


def _views(source, *, profile="python-whitespace@1", protected_lines=None):
    ref = views.retain_code_view_inventory(cas=source.cas,
        expected_source_inventory_cid=source.ref.inventory_cid, request=source.request,
        source_limits=source.limits, profile=profile, protected_lines=protected_lines)
    return SimpleNamespace(root=source.root, cas=source.cas, request=source.request,
        limits=source.limits, source_ref=source.ref, ref=ref, profile=profile,
        protected_lines=protected_lines)


@pytest.fixture
def retained(tmp_path):
    root = _repository(tmp_path, {b"module.py":
        b"VALUE = 1   \n# ordinary explanatory source comment\nVALUE = 2  \t\n"})
    return _views(_retain(root, tmp_path))


def _load(case, **changes):
    args = dict(cas=case.cas, expected_inventory_cid=case.ref.inventory_cid,
        expected_source_inventory_cid=case.source_ref.inventory_cid, request=case.request,
        source_limits=case.limits, profile=case.profile, protected_lines=case.protected_lines)
    args.update(changes)
    return views.load_code_view_inventory(**args)


def _record(case):
    root = case.cas.get(case.ref.inventory_cid)
    page = case.cas.get(root["entry_pages"][0])
    return root, page, page["records"][0]


def _original_source(case):
    return SimpleNamespace(root=case.root, cas=case.cas, request=case.request,
        limits=case.limits, ref=case.source_ref)


def test_full_population_keeps_original_bytes_modes_and_explicit_gitlink_frontier(tmp_path, monkeypatch):
    files = {
        b"module.py": b"VALUE = 1  \t\r\n# ordinary default comment remains  \r\n",
        b"types.pyi": b"VALUE: int   \n",
        b"notes.md": b"not Python   \r\n",
        b"broken.py": b"def broken(\n",
        b"binary.py": b"\xff\x00",
        b"empty.py": b"",
        b"invalid-\xff.txt": b"raw name source bytes\n",
        b"executable.py": b"VALUE = 42  \n",
        b"coverage/tracked.py": b"VALUE = 0   \n",
    }
    root = _repository(tmp_path, files)
    (root / "executable.py").chmod(0o755)
    first = _commit_request(root)
    (root / "alias.py").symlink_to("module.py")
    (root / "nested").mkdir()
    _git(root, "update-index", "--add", "--cacheinfo", "160000", first.expected_commit, "nested")
    source = _retain(root, tmp_path, request=_commit_request(root))
    case = _views(source)
    monkeypatch.setattr(ImmutableCAS, "__init__", _no_call)
    monkeypatch.setattr(sources, "snapshot_chunked_repository", _no_call)
    monkeypatch.setattr(sources, "read_chunked_blob_frame", _no_call)
    restored = _load(case)
    by_raw = {bytes.fromhex(e.entry.raw_path_hex): e for e in restored.entries}
    assert set(by_raw) == set(files) | {b"alias.py", b"nested"}
    for name, data in files.items():
        entry = by_raw[name]
        assert entry.original == data
        assert codec.restore_source_view(original=entry.original, view=entry.view,
            manifest=entry.manifest, expected_manifest_sha256=_manifest_sha(entry.manifest)) == data
    assert by_raw[b"module.py"].view != files[b"module.py"]
    assert b"# ordinary default comment remains  \r\n" in by_raw[b"module.py"].view
    assert by_raw[b"types.pyi"].view == b"VALUE: int\n"
    assert by_raw[b"executable.py"].entry.git_mode == "100755"
    for name in (b"notes.md", b"broken.py", b"binary.py", b"empty.py", b"invalid-\xff.txt"):
        assert by_raw[name].view == files[name]
    assert by_raw[b"alias.py"].entry.git_mode == "120000"
    assert by_raw[b"alias.py"].original == by_raw[b"alias.py"].view == b"module.py"
    assert by_raw[b"alias.py"].manifest["requested_profile"] == "identity@1"
    assert by_raw[b"nested"].original is by_raw[b"nested"].view is by_raw[b"nested"].manifest is None
    observed = restored.observation()
    assert observed["entry_count"] == len(files) + 2 and observed["gitlink_count"] == 1
    assert observed["content_complete"] is False
    assert observed["original_per_path_bytes"] == sum(map(len, files.values())) + len(b"module.py")
    assert observed["view_per_path_bytes"] < observed["original_per_path_bytes"]
    assert observed["exact_unchanged_source_restoration_verified"] is True
    assert observed["read_only"] is observed["historical_only"] is True
    assert observed["provider_token_counts"] is None
    assert all(observed[name] is False for name in ("current_source_verified", "semantic_equivalence_claimed",
                                                  "completion_authority", "execution_authority", "filesystem_restored"))
    metadata, offset = [], 0
    while offset is not None:
        page = restored.metadata_page(offset=offset, limit=3)
        assert len(page["entries"]) <= 3
        assert page["completion_authority"] is False and page["read_only"] is True
        metadata.extend(page["entries"])
        offset = page["next_offset"]
    assert [e["raw_path_hex"] for e in metadata] == [e.entry.raw_path_hex for e in restored.entries]
    assert not any(set(e) & {"original", "view", "manifest", "pieces", "data"} for e in metadata)
    assert "ordinary default comment" not in json.dumps(metadata)
    assert next(e for e in metadata if e["raw_path_hex"] == b"nested".hex())["status"] == "gitlink_requires_separate_repository"


def test_fresh_process_restores_original_and_view_after_source_repository_is_deleted(retained):
    before = _load(retained)
    expected = {e.entry.raw_path_hex: (_sha(e.original), _sha(e.view)) for e in before.entries}
    shutil.rmtree(retained.root)
    program = """
import hashlib, json, subprocess, sys
from ipfs_accelerate_py.agent_supervisor.runtime import repository_code_view_inventory as v
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as s
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
payload = json.loads(sys.stdin.read())
cas = ImmutableCAS(sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError('cold view replay tried source/Git acquisition or owner construction')
subprocess.Popen = forbidden
s.snapshot_chunked_repository = forbidden
s.preflight_committed_repository = forbidden
s.read_chunked_blob_frame = forbidden
ImmutableCAS.__init__ = forbidden
restored = v.load_code_view_inventory(cas=cas, expected_inventory_cid=payload['inventory_cid'],
    expected_source_inventory_cid=payload['source_inventory_cid'],
    request=s.SourceInventoryRequest(**payload['request']),
    source_limits=s.SourceInventoryLimits(**payload['limits']))
sha = lambda raw: hashlib.sha256(raw).hexdigest()
print(json.dumps({'hashes': {e.entry.raw_path_hex: [sha(e.original), sha(e.view)]
    for e in restored.entries}, 'observation': restored.observation()}, sort_keys=True))
"""
    result = subprocess.run([sys.executable, "-c", program, str(retained.cas.root)],
        input=json.dumps(dict(inventory_cid=retained.ref.inventory_cid,
            source_inventory_cid=retained.source_ref.inventory_cid,
            request=asdict(retained.request), limits=asdict(retained.limits))),
        text=True, capture_output=True, check=True, timeout=30,
        cwd=retained.cas.root.parent, env=os.environ.copy())
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["hashes"] == {name: list(values) for name, values in expected.items()}
    assert observed["observation"]["provider_token_counts"] is None
    assert not retained.root.exists()


@pytest.mark.parametrize("artifact", ["view", "source", "sidecar"])
def test_corrupt_view_original_chunk_or_sidecar_cannot_return_restored_inventory(retained, artifact):
    _, _, record = _record(retained)
    assert record["view_storage"] == "retained-view"
    if artifact == "view":
        path = retained.cas.path_for(record["view_cid"], source=True)
    elif artifact == "sidecar":
        path = retained.cas.path_for(record["manifest_cid"])
    else:
        source_case = SimpleNamespace(cas=retained.cas, ref=retained.source_ref)
        path = retained.cas.path_for(_chunk_cid(source_case), source=True)
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError):
        _load(retained)


@pytest.mark.parametrize("pin", ["source_inventory", "task", "profile", "protected_lines"])
def test_wrong_independent_bindings_refuse_before_raw_artifact_reads(retained, monkeypatch, pin):
    changes = {
        "source_inventory": {"expected_source_inventory_cid": cid_for_structured({"foreign": "source"})},
        "task": {"request": replace(retained.request, task_id="foreign-view-task")},
        "profile": {"profile": "python-comments@1"},
        "protected_lines": {"protected_lines": {b"module.py".hex(): (1,)}},
    }[pin]
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    with pytest.raises(sources.SourceInventoryError):
        _load(retained, **changes)


def test_rehashed_source_as_view_and_keep_map_cannot_replace_whitespace_recipe(retained):
    restored = _load(retained)
    source = restored.entries[0].original
    root, page, record = _record(retained)
    forged = deepcopy(restored.entries[0].manifest)
    forged.update(view_sha256=_sha(source), view_bytes=len(source),
                  pieces=[{"kind": "keep", "start": 0, "end": len(source)}])
    record.update(manifest_cid=retained.cas.put(forged), manifest_sha256=_manifest_sha(forged),
                  view_cid=cid_for_bytes(source), view_bytes=len(source), view_storage="original-source")
    root["entry_pages"][0] = retained.cas.put(page)
    forged_cid = retained.cas.put(root)
    with pytest.raises(sources.SourceInventoryError, match="deterministic source transform"):
        _load(retained, expected_inventory_cid=forged_cid)


def test_opt_in_comment_policy_reuses_same_source_assets_and_preserves_caller_lines(retained):
    source_ref = retained.source_ref.inventory_cid
    policy = {b"module.py".hex(): (1,)}
    case = _views(_original_source(retained), profile="python-comments@1", protected_lines=policy)
    restored = _load(case)
    assert case.source_ref.inventory_cid == source_ref
    assert case.cas is retained.cas
    entry = restored.entries[0]
    assert entry.view == b"VALUE = 1   \n\nVALUE = 2\n"
    assert entry.manifest["requested_profile"] == "python-comments@1"
    assert entry.manifest["protected_lines"] == [1]
    assert "ordinary explanatory source comment" not in json.dumps(entry.manifest)
    assert codec.restore_source_view(original=entry.original, view=entry.view,
        manifest=entry.manifest, expected_manifest_sha256=_manifest_sha(entry.manifest)) == entry.original


@pytest.mark.parametrize("record", ["request", "limits", "policy"])
def test_callback_input_drift_cannot_change_view_replay_custody(retained, monkeypatch, record):
    if record == "policy":
        retained = _views(_original_source(retained), profile="python-comments@1",
                          protected_lines={b"module.py".hex(): (1,)})
    original = ImmutableCAS.get
    calls = []

    def read_then_mutate(self, cid, **kwargs):
        value = original(self, cid, **kwargs)
        if cid == retained.ref.inventory_cid and not calls:
            calls.append(cid)
            if record == "request":
                object.__setattr__(retained.request, "task_id", "callback-view-task")
            elif record == "limits":
                object.__setattr__(retained.limits, "max_file_bytes", retained.limits.max_file_bytes + 1)
            else:
                retained.protected_lines[b"module.py".hex()] = (2,)
        return value

    monkeypatch.setattr(ImmutableCAS, "get", read_then_mutate)
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    reason = "policy changed" if record == "policy" else "request or limits changed"
    with pytest.raises(sources.SourceInventoryError, match=reason):
        _load(retained)
    assert calls == [retained.ref.inventory_cid]


def test_store_directory_replacement_during_compact_view_read_refuses(retained, monkeypatch):
    _, _, record = _record(retained)
    original = ImmutableCAS.get_bytes
    changed = []

    def read_then_replace(self, cid):
        data = original(self, cid)
        if cid == record["view_cid"] and not changed:
            changed.append(cid)
            saved = self.source_root.with_name("source-before-view-callback")
            self.source_root.rename(saved)
            shutil.copytree(saved, self.source_root)
        return data

    monkeypatch.setattr(ImmutableCAS, "get_bytes", read_then_replace)
    with pytest.raises(sources.SourceInventoryError, match="binding changed"):
        _load(retained)
    assert changed == [record["view_cid"]]


def test_codec_implementation_drift_cannot_reinterpret_retained_code_views(retained, tmp_path, monkeypatch):
    copied = tmp_path / "isolated_codec.py"
    copied.write_bytes(Path(codec.__file__).read_bytes() + b"\n# authored codec drift\n")
    monkeypatch.setattr(codec, "__file__", str(copied))
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    with pytest.raises(sources.SourceInventoryError, match="codec differs"):
        _load(retained)


def test_unknown_profile_version_is_refused_before_existing_cas_is_consulted(retained, monkeypatch):
    monkeypatch.setattr(ImmutableCAS, "get", _no_call)
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    with pytest.raises(sources.SourceInventoryError, match="unsupported code view profile"):
        _load(retained, profile="python-whitespace@2")


def test_final_inventory_publication_poisoning_view_returns_no_successful_ref(tmp_path, monkeypatch):
    root = _repository(tmp_path, {b"module.py": b"VALUE = 1   \n"})
    source = _retain(root, tmp_path)
    original = ImmutableCAS.put
    poisoned = []

    def publish_then_poison(self, value):
        cid = original(self, value)
        if value.get("schema") == views.SCHEMA:
            page = self.get(value["entry_pages"][0])
            record = page["records"][0]
            assert record["view_storage"] == "retained-view"
            path = self.path_for(record["view_cid"], source=True)
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            poisoned.append(cid)
        return cid

    monkeypatch.setattr(ImmutableCAS, "put", publish_then_poison)
    with pytest.raises(ValueError):
        _views(source)
    assert len(poisoned) == 1


def test_protected_lines_for_a_foreign_source_cannot_expand_inventory_scope(tmp_path, monkeypatch):
    source = _retain(_repository(tmp_path, {b"module.py": b"VALUE = 1   \n"}), tmp_path)
    monkeypatch.setattr(ImmutableCAS, "put", _no_call)
    monkeypatch.setattr(ImmutableCAS, "put_bytes", _no_call)
    with pytest.raises(sources.SourceInventoryError, match="regular Python source scope"):
        _views(source, protected_lines={b"foreign.py".hex(): (1,)})
