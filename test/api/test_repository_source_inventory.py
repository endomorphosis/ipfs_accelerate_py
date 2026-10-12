"""Authored Git/CAS integration for byte-exact historical source inventories."""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as inventory
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import (
    CommittedPopulationEntry, preflight_committed_repository,
)


def _git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
        stderr=subprocess.DEVNULL,
    ).decode("ascii").strip()


def _commit_request(root, *, task_id="source-inventory-task", repository_id="fixture:source-inventory"):
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "--allow-empty", "-qm", "authored inventory")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    plan = preflight_committed_repository(
        root, repository_id=repository_id, expected_commit=commit, expected_tree=tree,
    )
    return inventory.SourceInventoryRequest(task_id, repository_id, commit, tree, plan.population_cid)


def _repository(tmp_path, files=None, *, object_format="sha1"):
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-b", "main", "--object-format=" + object_format)
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "core.filemode", "true")
    if files is None:
        files = {b"module.py": b"# " + b"ordinary source comment " * 5 + b"\nVALUE = 42\n"}
    for relative, data in files.items():
        raw_path = os.fsencode(root) + b"/" + relative
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "wb") as stream:
            stream.write(data)
    return root


def _retain(root, tmp_path, request=None, limits=None):
    request = request or _commit_request(root)
    limits = limits or inventory.SourceInventoryLimits(frame_bytes=64)
    cas = ImmutableCAS(tmp_path / "cas")
    ref = inventory.retain_source_inventory(root, cas=cas, request=request, limits=limits)
    return SimpleNamespace(root=root, request=request, limits=limits, cas=cas, ref=ref)


@pytest.fixture
def retained(tmp_path):
    return _retain(_repository(tmp_path), tmp_path)


def _load(case, **changes):
    kwargs = dict(cas=case.cas, expected_inventory_cid=case.ref.inventory_cid,
                  request=case.request, limits=case.limits)
    kwargs.update(changes)
    return inventory.load_source_inventory(**kwargs)


def _chunk_cid(case, *, last=False):
    native = case.cas.get(case.ref.native_snapshot_cid)
    page = case.cas.get(native["blob_pages"][0])
    blob = case.cas.get(page["records"][0]["manifest_cid"])
    return blob["chunks"][-1 if last else 0]["source_cid"]


def _no_call(*args, **kwargs):
    pytest.fail("historical replay activated an unadmitted source/owner producer")


def test_heterogeneous_committed_bytes_and_modes_survive_exact_replay(tmp_path, monkeypatch):
    files = {
        b"module.py": b"raise RuntimeError('authored_target_must_not_execute')\n",
        b"crlf.py": "# caf\u00e9\r\nVALUE = '\u03bb'\r\n".encode(),
        b"no-newline.txt": b"no final newline",
        b"non-utf8.txt": b"\xff\xfe\x80\n",
        b"binary.bin": bytes(range(256)) + b"\x00\xff",
        b"empty.txt": b"",
        b"copy-a.txt": b"same source bytes\n",
        b"copy-b.txt": b"same source bytes\n",
        b"coverage/retained.txt": b"tracked normally excluded source\n",
        b"executable.sh": b"#!/bin/sh\nexit 7\n",
    }
    root = _repository(tmp_path, files)
    (root / "executable.sh").chmod(0o755)
    case = _retain(root, tmp_path)
    # The operation receives an already existing native store. It cannot
    # create owners or reacquire source to fill missing restoration data.
    monkeypatch.setattr(ImmutableCAS, "__init__", _no_call)
    monkeypatch.setattr(inventory, "snapshot_chunked_repository", _no_call)
    monkeypatch.setattr(inventory, "read_chunked_blob_frame", _no_call)
    restored = _load(case)
    by_raw = {bytes.fromhex(item.entry.raw_path_hex): item for item in restored.entries}
    assert set(by_raw) == set(files)
    for name, data in files.items():
        entry = by_raw[name]
        assert type(entry.entry) is CommittedPopulationEntry
        assert type(entry.data) is bytes and entry.data == data
        assert entry.source_cid == cid_for_bytes(data)
        assert entry.entry.size_bytes == len(data)
        expected_oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert entry.entry.git_object_oid == expected_oid
        assert entry.entry.git_mode == ("100755" if name == b"executable.sh" else "100644")
    assert by_raw[b"copy-a.txt"].source_cid == by_raw[b"copy-b.txt"].source_cid
    observed = restored.observation()
    assert observed["content_complete"] is True
    assert observed["unique_blob_count"] == len(set(files.values()))
    assert observed["unique_blob_bytes"] == sum(map(len, set(files.values())))
    assert observed["per_path_blob_bytes"] == sum(map(len, files.values()))
    assert observed["current_source_verified"] is False
    assert observed["filesystem_restored"] is False
    assert observed["historical_only"] is True
    assert all(observed[name] is False for name in (
        "semantic_acceptance_authority", "completion_authority", "execution_authority"))


def test_separate_process_cold_reload_uses_retained_bytes_without_original_git(retained):
    case = retained
    before = _load(case)
    expected = {item.entry.raw_path_hex: hashlib.sha256(item.data).hexdigest()
                for item in before.entries}
    shutil.rmtree(case.root)
    # Imports and the explicitly supplied existing CAS are admitted before
    # prohibiting all subprocesses and source producers in the fresh process.
    program = """
import hashlib, json, subprocess, sys
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as i
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
request = json.loads(sys.stdin.read())
cas = ImmutableCAS(sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError('cold replay attempted Git/source acquisition or a new owner')
subprocess.Popen = forbidden
i.snapshot_chunked_repository = forbidden
i.preflight_committed_repository = forbidden
i.read_chunked_blob_frame = forbidden
ImmutableCAS.__init__ = forbidden
restored = i.load_source_inventory(cas=cas, expected_inventory_cid=request['inventory_cid'],
    request=i.SourceInventoryRequest(**request['request']),
    limits=i.SourceInventoryLimits(**request['limits']))
print(json.dumps({'hashes': {e.entry.raw_path_hex: hashlib.sha256(e.data).hexdigest()
    for e in restored.entries}, 'observation': restored.observation()}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(case.cas.root)],
        input=json.dumps(dict(inventory_cid=case.ref.inventory_cid,
                             request=asdict(case.request), limits=asdict(case.limits))),
        text=True, capture_output=True, check=True, timeout=30,
        cwd=str(case.cas.root.parent), env=os.environ.copy(),
    )
    actual = json.loads(result.stdout.splitlines()[-1])
    assert actual["hashes"] == expected
    assert actual["observation"]["inventory_cid"] == case.ref.inventory_cid
    assert actual["observation"]["current_source_verified"] is False
    assert not case.root.exists()


def test_actual_git_sha256_repository_replays_full_object_identities(tmp_path):
    data = b"# exact SHA-256 Git object\r\nVALUE = 42\r\n"
    root = _repository(tmp_path, {b"module.py": data}, object_format="sha256")
    case = _retain(root, tmp_path)
    assert len(case.request.expected_commit) == len(case.request.expected_tree) == 64
    restored = _load(case)
    assert len(restored.entries) == 1
    entry = restored.entries[0]
    assert entry.data == data and entry.source_cid == cid_for_bytes(data)
    expected_oid = hashlib.sha256(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    assert entry.entry.git_object_oid == expected_oid
    assert len(entry.entry.git_object_oid) == 64
    assert restored.observation()["content_complete"] is True
    assert restored.observation()["completion_authority"] is False


@pytest.mark.parametrize("change", ["missing", "truncated", "bitflip"])
def test_missing_or_corrupt_retained_frame_cannot_restore_complete_inventory(retained, change):
    cid = _chunk_cid(retained)
    path = retained.cas.path_for(cid, source=True)
    raw = path.read_bytes()
    if change == "missing":
        path.unlink()
    elif change == "truncated":
        path.write_bytes(raw[:-1])
    else:
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(retained)


@pytest.mark.parametrize("field", [
    "task_id", "repository_id", "expected_commit", "expected_tree", "expected_population_cid",
])
def test_independent_request_mismatch_refuses_before_raw_source_reads(retained, monkeypatch, field):
    value = ("foreign-task" if field == "task_id" else "fixture:foreign" if field == "repository_id"
             else "0" * 40 if field in {"expected_commit", "expected_tree"}
             else cid_for_structured({"authored": "different complete population"}))
    wrong = replace(retained.request, **{field: value})
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    with pytest.raises(inventory.SourceInventoryError, match="independent request"):
        _load(retained, request=wrong)


def test_rehashed_native_entry_omission_cannot_shrink_independent_population(tmp_path, monkeypatch):
    retained = _retain(_repository(tmp_path, {
        b"first.py": b"VALUE = 1\n", b"second.py": b"VALUE = 2\n",
    }), tmp_path)
    root = retained.cas.get(retained.ref.inventory_cid)
    native = retained.cas.get(root["native_snapshot_cid"])
    page = retained.cas.get(native["entry_pages"][0])
    page["records"].pop()
    assert page["records"]  # A valid remaining page, rather than an empty-page schema error.
    native["entry_pages"][0] = retained.cas.put(page)
    root["native_snapshot_cid"] = retained.cas.put(native)
    forged_cid = retained.cas.put(root)
    # The caller deliberately nominates this rehashed root. The independent
    # complete population must still prevent inventory acceptance of omission.
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    with pytest.raises(ValueError):
        _load(retained, expected_inventory_cid=forged_cid)


def test_empty_committed_tree_replays_without_content_reads(tmp_path, monkeypatch):
    case = _retain(_repository(tmp_path, {}), tmp_path)
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    restored = _load(case)
    assert restored.entries == ()
    observed = restored.observation()
    assert observed["content_complete"] is True
    assert observed["entry_count"] == observed["unique_blob_count"] == observed["unique_blob_bytes"] == 0
    assert restored.metadata_page()["entries"] == []
    assert restored.metadata_page()["next_offset"] is None
    assert observed["completion_authority"] is False


@pytest.mark.parametrize("record", ["request", "limits"])
def test_caller_input_mutation_during_cas_callback_cannot_change_replay_request(retained, monkeypatch, record):
    original = ImmutableCAS.get
    callbacks = []

    def read_then_mutate(self, cid, **kwargs):
        value = original(self, cid, **kwargs)
        if cid == retained.ref.inventory_cid and not callbacks:
            callbacks.append(cid)
            if record == "request":
                object.__setattr__(retained.request, "task_id", "mutated-during-read")
            else:
                object.__setattr__(retained.limits, "max_file_bytes", retained.limits.max_file_bytes + 1)
        return value

    monkeypatch.setattr(ImmutableCAS, "get", read_then_mutate)
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    with pytest.raises(inventory.SourceInventoryError, match="request or limits changed"):
        _load(retained)
    assert callbacks == [retained.ref.inventory_cid]


def test_rehashed_duplicate_page_is_refused_before_manifest_fanout(retained, monkeypatch):
    root = retained.cas.get(retained.ref.inventory_cid)
    native = retained.cas.get(root["native_snapshot_cid"])
    native["entry_pages"] *= 2
    native_cid = retained.cas.put(native)
    root["native_snapshot_cid"] = native_cid
    root_cid = retained.cas.put(root)
    original = ImmutableCAS.get
    seen = []

    def bounded_read(self, cid, **kwargs):
        seen.append(cid)
        assert cid in {root_cid, native_cid}, "duplicate page reached unbounded manifest fanout"
        return original(self, cid, **kwargs)

    monkeypatch.setattr(ImmutableCAS, "get", bounded_read)
    monkeypatch.setattr(ImmutableCAS, "get_bytes", _no_call)
    with pytest.raises(inventory.SourceInventoryError, match="duplicate manifest pages"):
        _load(retained, expected_inventory_cid=root_cid)
    assert seen == [root_cid, native_cid]


def test_symlink_unsafe_name_and_gitlink_are_explicit_data_records(tmp_path):
    root = _repository(tmp_path, {b"module.py": b"VALUE = 1\n", b"invalid-\xff.txt": b"\xff\x00"})
    first = _commit_request(root)
    outside = tmp_path / "outside"
    outside.write_bytes(b"external target must not be captured")
    os.symlink(str(outside), root / "absolute-link")
    os.symlink("../outside", root / "escaping-link")
    (root / "nested").mkdir()
    _git(root, "update-index", "--add", "--cacheinfo", "160000", first.expected_commit, "nested")
    case = _retain(root, tmp_path, request=_commit_request(root))
    restored = _load(case)
    by_raw = {bytes.fromhex(item.entry.raw_path_hex): item for item in restored.entries}
    assert len(by_raw) == 5
    assert by_raw[b"absolute-link"].entry.git_mode == "120000"
    assert by_raw[b"absolute-link"].data == os.fsencode(outside)
    assert by_raw[b"escaping-link"].data == b"../outside"
    assert by_raw[b"invalid-\xff.txt"].data == b"\xff\x00"
    nested = by_raw[b"nested"]
    assert nested.entry.git_mode == "160000" and nested.entry.object_type == "commit"
    assert nested.entry.git_object_oid == first.expected_commit
    assert nested.source_cid is None and nested.data is None
    observed = restored.observation()
    assert observed["gitlink_count"] == 1 and observed["content_complete"] is False
    assert observed["filesystem_restored"] is False
    assert outside.read_bytes() == b"external target must not be captured"
    statuses = {bytes.fromhex(e["raw_path_hex"]): e["status"] for e in restored.metadata_page()["entries"]}
    assert statuses[b"nested"] == "gitlink_requires_separate_repository"
    assert statuses[b"absolute-link"] == statuses[b"invalid-\xff.txt"] == "retained-blob"


def test_source_drift_during_final_root_publication_returns_no_successful_ref(tmp_path, monkeypatch):
    root = _repository(tmp_path)
    request = _commit_request(root)
    cas = ImmutableCAS(tmp_path / "cas")
    real_put = ImmutableCAS.put
    root_published = []

    def publish_then_mutate(self, value):
        cid = real_put(self, value)
        if value.get("schema") == inventory.SCHEMA:
            root_published.append(cid)
            (root / "module.py").write_bytes(b"VALUE = 'changed during final publication'\n")
        return cid

    monkeypatch.setattr(ImmutableCAS, "put", publish_then_mutate)
    with pytest.raises(ValueError):
        inventory.retain_source_inventory(root, cas=cas, request=request)
    # Immutable partial artifacts are allowed; publication alone is not a
    # successful capture/current-source admission.
    assert len(root_published) == 1
    assert cas.get(root_published[0])["completion_authority"] is False


def test_final_root_publication_poisoning_a_retained_frame_returns_no_successful_ref(tmp_path, monkeypatch):
    root = _repository(tmp_path)
    request = _commit_request(root)
    cas = ImmutableCAS(tmp_path / "cas")
    original = ImmutableCAS.put
    poisoned = []

    def publish_then_poison(self, value):
        cid = original(self, value)
        if value.get("schema") == inventory.SCHEMA:
            frame = next(path for path in self.source_root.rglob("*") if path.is_file())
            raw = frame.read_bytes()
            frame.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            poisoned.append(cid)
        return cid

    monkeypatch.setattr(ImmutableCAS, "put", publish_then_poison)
    with pytest.raises(ValueError):
        inventory.retain_source_inventory(root, cas=cas, request=request)
    assert len(poisoned) == 1


@pytest.mark.parametrize("change", ["directory_replacement", "object_limit"])
def test_cas_owner_binding_drift_during_final_raw_read_refuses(retained, monkeypatch, change):
    original = ImmutableCAS.get_bytes
    chunk = _chunk_cid(retained, last=True)
    changed = []

    def read_then_change(self, cid):
        raw = original(self, cid)
        if cid == chunk and not changed:
            changed.append(True)
            if change == "object_limit":
                self.max_object_bytes -= 1
            else:
                saved = self.source_root.with_name("source-original")
                self.source_root.rename(saved)
                shutil.copytree(saved, self.source_root)
        return raw

    monkeypatch.setattr(ImmutableCAS, "get_bytes", read_then_change)
    with pytest.raises(inventory.SourceInventoryError, match="binding changed"):
        _load(retained)
    assert changed == [True]


@pytest.mark.parametrize("changes", [
    {"max_file_bytes": 1}, {"max_total_bytes": 1},
    {"max_frame_reads": 1}, {"max_rehash_bytes": 1},
])
def test_acquisition_budget_refuses_before_retained_frame_export(tmp_path, monkeypatch, changes):
    root = _repository(tmp_path)
    request = _commit_request(root)
    cas = ImmutableCAS(tmp_path / "cas")
    limits = inventory.SourceInventoryLimits(frame_bytes=64, **changes)
    monkeypatch.setattr(inventory, "snapshot_chunked_repository", _no_call)
    monkeypatch.setattr(inventory, "read_chunked_blob_frame", _no_call)
    monkeypatch.setattr(ImmutableCAS, "put_bytes", _no_call)
    with pytest.raises(ValueError):
        inventory.retain_source_inventory(root, cas=cas, request=request, limits=limits)
    assert list(cas.source_root.rglob("*")) == []


def test_metadata_pages_are_complete_bounded_detached_and_contain_no_source_bytes(tmp_path):
    files = {f"file-{number}.txt".encode(): b"ordinary source comment\n" for number in range(5)}
    case = _retain(_repository(tmp_path, files), tmp_path)
    restored = _load(case)
    page = restored.metadata_page(offset=0, limit=2)
    assert page["population_complete"] is True and page["historical_only"] is True
    assert page["completion_authority"] is False
    assert page["next_offset"] == 2
    seen = list(page["entries"])
    offset = page["next_offset"]
    while offset is not None:
        next_page = restored.metadata_page(offset=offset, limit=2)
        assert len(next_page["entries"]) <= 2
        seen.extend(next_page["entries"])
        offset = next_page["next_offset"]
    assert [entry["raw_path_hex"] for entry in seen] == [item.entry.raw_path_hex for item in restored.entries]
    assert "ordinary source comment" not in json.dumps(page)
    assert not any("data" in entry or "captured_bytes" in entry for entry in page["entries"])
    page["entries"][0]["git_mode"] = "100755"
    assert restored.entries[0].entry.git_mode == "100644"
    assert restored.metadata_page(offset=len(restored.entries))["entries"] == []
    for kwargs in ({"offset": True}, {"offset": len(restored.entries) + 1}, {"limit": 0}, {"limit": 257}):
        with pytest.raises(inventory.SourceInventoryError):
            restored.metadata_page(**kwargs)


def test_cas_inside_captured_repository_is_refused_before_source_acquisition(tmp_path, monkeypatch):
    root = _repository(tmp_path)
    request = _commit_request(root)
    cas = ImmutableCAS(root / "unadmitted-cas")
    monkeypatch.setattr(inventory, "preflight_committed_repository", _no_call)
    monkeypatch.setattr(inventory, "snapshot_chunked_repository", _no_call)
    with pytest.raises(inventory.SourceInventoryError, match="outside captured repository"):
        inventory.retain_source_inventory(root, cas=cas, request=request)


def test_foreign_cas_cannot_replace_existing_native_store(retained, monkeypatch):
    class ForeignCAS:
        def get(self, *args, **kwargs):
            pytest.fail("foreign CAS was consulted")

    monkeypatch.setattr(inventory, "snapshot_chunked_repository", _no_call)
    with pytest.raises(inventory.SourceInventoryError, match="native ImmutableCAS"):
        _load(retained, cas=ForeignCAS())
