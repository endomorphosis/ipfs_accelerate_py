"""Authored payload tests for the owner-local typed-proposal cache helpers.

These tests exercise filesystem custody and keyed integrity only. The authored
proposal is not native source evidence, a formal proof, a current-source fence,
or a qualified worker/auditor result. Actual wrapper and cold-child qualification
remain separate. The opt-out belongs to the wrapper and is tested there.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_source_context as subject


MAXIMUM = 2_000_000
CACHE_DIRECTORY = "semantic-typed-verification-cache"
KEY_FILENAME = ".owner-key"


def _canonical(value):
    return subject.codec._json(value).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _cache_directory(repository):
    return repository / ".runtime" / CACHE_DIRECTORY


def _entry(repository, binding):
    key = _key(repository).read_bytes()
    name = hashlib.sha256(b"typed-native-cache-entry@1\0" + hashlib.sha256(key).digest()
                          + _canonical(binding)).hexdigest() + ".json"
    return _cache_directory(repository) / name


def _key(repository):
    return _cache_directory(repository) / KEY_FILENAME


def _load(repository, binding, *, maximum=MAXIMUM):
    return subject._load_cached_typed_proposal(
        repository=repository, binding=binding, maximum=maximum,
    )


def _publish(repository, binding, proposal, *, maximum=MAXIMUM):
    return subject._publish_cached_typed_proposal(
        repository=repository, binding=binding, proposal=proposal, maximum=maximum,
    )


def _publish_or_refuse(repository, binding, proposal):
    # A cache publication refusal may be represented as a miss or a ValueError;
    # unanticipated exceptions are not accepted as successful fail-closed tests.
    try:
        return _publish(repository, binding, proposal)
    except ValueError:
        return None


def _replace_owned_file(path, raw):
    """Mutate only a file explicitly created within this test's tmp_path."""
    os.chmod(path, 0o600)
    path.write_bytes(raw)
    os.chmod(path, 0o400)


def _snapshot(repository):
    result = {}
    for path in [repository, *sorted(repository.rglob("*"))]:
        info = path.lstat()
        result[str(path.relative_to(repository))] = (
            info.st_dev, info.st_ino, info.st_uid, info.st_gid,
            info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns,
            path.read_bytes() if stat.S_ISREG(info.st_mode) else None,
        )
    return result


@pytest.fixture
def authored_cache(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o755)
    (repository / ".runtime").mkdir(mode=0o755)
    binding = {
        "schema": "authored-owner-local-typed-cache-binding@1",
        "repository": str(repository),
        "owner_identity": {"uid": os.geteuid(), "repository": str(repository)},
        "semantic_root_cid": "authored-root-identity",
        "blocks": [
            {"cid": "authored-root-identity", "sha256": "1" * 64, "bytes": 11},
            {"cid": "authored-omitted-block", "sha256": "2" * 64, "bytes": 22},
        ],
        "validator_sources": {"semantic_typed_source_context.py": "3" * 64},
        "producer_identity": "authored-producer",
        "parent_sha256": "4" * 64,
        "seed_symbol_ids": ["authored-seed"],
        "graph_scope": "task_closure",
        "limits": {"max_selected_symbols": 32, "max_graph_nodes": 2048,
                   "max_graph_edges": 8192, "max_report_bytes": MAXIMUM},
    }
    graph = {"nodes": [{"id": "authored-seed", "record": {"native_symbol_id": "authored-seed"}}],
             "edges": []}
    proposal = {
        "schema": "authored-owner-local-typed-proposal@1",
        "graph_scope": "task_closure",
        "graph": graph,
        "graph_sha256": _digest(graph),
        "nominee_symbol_ids": ["authored-seed"],
        "omitted_root_symbol_ids": ["authored-omitted-symbol"],
        "proof_authority": False,
        "execution_authority": False,
        "completion_authority": False,
    }
    return repository, binding, proposal


def test_owner_local_cache_exact_hit_is_detached_and_read_does_not_mutate(authored_cache):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    before = _snapshot(repository)
    original_binding, original_proposal = copy.deepcopy(binding), copy.deepcopy(proposal)
    first = _load(repository, binding)
    assert first == proposal
    assert first is not proposal
    assert first["graph"] is not proposal["graph"]
    first["graph"]["nodes"][0]["record"]["native_symbol_id"] = "mutated-caller-copy"
    first["nominee_symbol_ids"].append("mutated-caller-copy")
    assert _load(repository, binding) == original_proposal
    assert binding == original_binding
    assert proposal == original_proposal
    assert _snapshot(repository) == before


def test_owner_local_cache_reader_miss_creates_no_files(authored_cache):
    repository, binding, _ = authored_cache
    before = _snapshot(repository)
    assert _load(repository, binding) is None
    assert _snapshot(repository) == before
    assert not _cache_directory(repository).exists()


def test_owner_local_cache_publisher_protects_key_and_entry(authored_cache):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    assert stat.S_IMODE(_cache_directory(repository).stat().st_mode) == 0o700
    for path in (_key(repository), _entry(repository, binding)):
        info = path.stat(follow_symlinks=False)
        assert stat.S_ISREG(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == 0o400
        assert info.st_uid == os.geteuid()
        assert info.st_nlink == 1
    assert len(_key(repository).read_bytes()) == 32
    assert _load(repository, binding) == proposal


def test_owner_local_cache_self_consistent_unkeyed_proposal_hashes_do_not_authenticate(authored_cache):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    path = _entry(repository, binding)
    envelope = json.loads(path.read_bytes())
    old_mac = envelope["hmac_sha256"]
    changed = copy.deepcopy(proposal)
    changed["graph"]["nodes"][0]["record"]["native_symbol_id"] = "forged-selected-symbol"
    changed["graph_sha256"] = _digest(changed["graph"])
    changed["nominee_symbol_ids"] = ["forged-selected-symbol"]
    envelope["payload"]["proposal"] = changed
    assert changed["graph_sha256"] == _digest(changed["graph"])
    assert envelope["hmac_sha256"] == old_mac
    _replace_owned_file(path, _canonical(envelope))
    before = _snapshot(repository)
    assert _load(repository, binding) is None
    assert _snapshot(repository) == before


@pytest.mark.parametrize("change", ["omitted_block", "validator", "seed", "scope", "budget", "owner", "producer", "parent"])
def test_owner_local_cache_exact_binding_change_misses(authored_cache, change):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    changed = copy.deepcopy(binding)
    if change == "omitted_block":
        changed["blocks"][1]["sha256"] = "a" * 64
    elif change == "validator":
        changed["validator_sources"]["semantic_typed_source_context.py"] = "b" * 64
    elif change == "seed":
        changed["seed_symbol_ids"] = ["other-authored-seed"]
    elif change == "scope":
        changed["graph_scope"] = "full_root"
    elif change == "budget":
        changed["limits"]["max_graph_nodes"] = 1024
    elif change == "owner":
        changed["owner_identity"]["uid"] += 1
    elif change == "producer":
        changed["producer_identity"] = "other-authored-producer"
    elif change == "parent":
        changed["parent_sha256"] = "c" * 64
    before = _snapshot(repository)
    assert _load(repository, changed) is None
    assert _load(repository, binding) == proposal
    assert _snapshot(repository) == before


def test_owner_local_cache_wrong_binding_inside_correct_entry_is_rejected(authored_cache):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    path = _entry(repository, binding)
    envelope = json.loads(path.read_bytes())
    envelope["payload"]["binding"]["blocks"][1]["sha256"] = "d" * 64
    # This deliberately models a fixture with access to its own key. Even a
    # valid MAC for different input does not match the independently supplied
    # binding selected by the reader.
    envelope["hmac_sha256"] = hmac.new(
        _key(repository).read_bytes(), _canonical(envelope["payload"]), hashlib.sha256,
    ).hexdigest()
    _replace_owned_file(path, _canonical(envelope))
    assert _load(repository, binding) is None


def test_owner_local_cache_key_rotation_invalidates_old_entry(authored_cache):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    old_path = _entry(repository, binding)
    old_entry = old_path.read_bytes()
    old_key = _key(repository).read_bytes()
    new_key = bytes([old_key[0] ^ 0xFF]) + old_key[1:]
    _replace_owned_file(_key(repository), new_key)
    assert _load(repository, binding) is None
    assert old_path.read_bytes() == old_entry
    _publish(repository, binding, proposal)
    assert _load(repository, binding) == proposal
    assert _entry(repository, binding) != old_path
    assert old_path.read_bytes() == old_entry
    new_binding = copy.deepcopy(binding)
    new_binding["seed_symbol_ids"] = ["post-rotation-authored-seed"]
    _publish(repository, new_binding, proposal)
    assert _load(repository, new_binding) == proposal
    assert _load(repository, binding) == proposal


@pytest.mark.parametrize("target", ["key", "entry"])
def test_owner_local_cache_rejects_hardlinked_key_or_entry(authored_cache, tmp_path, target):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    path = _key(repository) if target == "key" else _entry(repository, binding)
    os.link(path, tmp_path / (target + "-alias"))
    before = _snapshot(repository)
    assert _load(repository, binding) is None
    _publish_or_refuse(repository, binding, proposal)
    assert _snapshot(repository) == before


@pytest.mark.parametrize("target", ["key", "entry", "cache_directory"])
def test_owner_local_cache_rejects_symlink_components(authored_cache, tmp_path, target):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    path = {"key": _key(repository), "entry": _entry(repository, binding),
            "cache_directory": _cache_directory(repository)}[target]
    moved = tmp_path / (target + "-original")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=target == "cache_directory")
    assert _load(repository, binding) is None
    _publish_or_refuse(repository, binding, proposal)
    assert path.is_symlink()
    assert path.resolve() == moved


@pytest.mark.parametrize("target,mode", [
    ("key", 0o600), ("key", 0o440), ("entry", 0o600), ("entry", 0o444),
    ("cache_directory", 0o750), ("runtime", 0o775), ("repository", 0o775),
])
def test_owner_local_cache_rejects_permissive_or_mutable_modes(authored_cache, target, mode):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    path = {"key": _key(repository), "entry": _entry(repository, binding),
            "cache_directory": _cache_directory(repository),
            "runtime": repository / ".runtime", "repository": repository}[target]
    os.chmod(path, mode)
    before = _snapshot(repository)
    assert _load(repository, binding) is None
    _publish_or_refuse(repository, binding, proposal)
    assert _snapshot(repository) == before


class _TargetedUidObservation:
    """Change the helper's observation of only one fixture-owned inode.

    Replacing the subject module's os reference leaves the real global os
    module, unrelated files, and all actual filesystem ownership unchanged.
    """

    def __init__(self, real_os, path):
        self._real_os = real_os
        info = path.stat(follow_symlinks=False)
        self._target = (info.st_dev, info.st_ino)
        self.observations = 0

    def __getattr__(self, name):
        return getattr(self._real_os, name)

    def _observed(self, info):
        if (info.st_dev, info.st_ino) != self._target:
            return info
        self.observations += 1
        names = [name for name in dir(info) if name.startswith("st_")]
        return SimpleNamespace(**{
            **{name: getattr(info, name) for name in names},
            "st_uid": self._real_os.geteuid() + 1,
        })

    def fstat(self, descriptor):
        return self._observed(self._real_os.fstat(descriptor))

    def stat(self, path, *args, **kwargs):
        return self._observed(self._real_os.stat(path, *args, **kwargs))

    def lstat(self, path, *args, **kwargs):
        return self._observed(self._real_os.lstat(path, *args, **kwargs))


@pytest.mark.parametrize("target", ["key", "entry"])
def test_owner_local_cache_rejects_wrong_owner_observation(authored_cache, monkeypatch, target):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    path = _key(repository) if target == "key" else _entry(repository, binding)
    before = _snapshot(repository)
    scoped_os = _TargetedUidObservation(os, path)
    monkeypatch.setattr(subject, "os", scoped_os)
    assert _load(repository, binding) is None
    assert scoped_os.observations > 0
    assert _snapshot(repository) == before


@pytest.mark.parametrize("damage", ["malformed_json", "noncanonical_json", "extra_envelope_field", "wrong_mac", "oversized"])
def test_owner_local_cache_malformed_entries_are_readonly_misses(authored_cache, damage):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    path = _entry(repository, binding)
    envelope = json.loads(path.read_bytes())
    if damage == "malformed_json":
        raw = b'{"schema":'
    elif damage == "noncanonical_json":
        raw = json.dumps(envelope, indent=2).encode()
    elif damage == "extra_envelope_field":
        envelope["claimed_verified"] = True
        raw = _canonical(envelope)
    elif damage == "wrong_mac":
        envelope["hmac_sha256"] = "0" * 64
        raw = _canonical(envelope)
    else:
        raw = b"x" * (MAXIMUM + 1)
    _replace_owned_file(path, raw)
    before = _snapshot(repository)
    assert _load(repository, binding) is None
    assert _snapshot(repository) == before


def test_owner_local_cache_conflicting_publication_does_not_overwrite(authored_cache):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    before = _snapshot(repository)
    changed = copy.deepcopy(proposal)
    changed["nominee_symbol_ids"] = ["conflicting-authored-symbol"]
    _publish_or_refuse(repository, binding, changed)
    assert _snapshot(repository) == before
    assert _load(repository, binding) == proposal


@pytest.mark.parametrize("change", ["key_rotation", "same_bytes_entry_replacement", "same_bytes_directory_replacement"])
def test_owner_local_cache_closing_custody_rejects_changed_fixture(authored_cache, tmp_path, monkeypatch, change):
    repository, binding, proposal = authored_cache
    _publish(repository, binding, proposal)
    original_read = subject._typed_cache_read_file
    changed = False

    def change_after_first_entry_capture(parent, name, maximum, *, exact_size=None):
        nonlocal changed
        result = original_read(parent, name, maximum, exact_size=exact_size)
        if name == _entry(repository, binding).name and not changed:
            changed = True
            if change == "key_rotation":
                key = _key(repository).read_bytes()
                _replace_owned_file(_key(repository), bytes([key[0] ^ 0xFF]) + key[1:])
            elif change == "same_bytes_entry_replacement":
                path = _entry(repository, binding)
                replacement = path.with_name("fixture-replacement")
                replacement.write_bytes(path.read_bytes())
                os.chmod(replacement, 0o400)
                replacement.replace(path)
            else:
                cache = _cache_directory(repository)
                old_cache = tmp_path / "captured-cache-directory"
                cache.rename(old_cache)
                cache.mkdir(mode=0o700)
                for original in old_cache.iterdir():
                    replacement = cache / original.name
                    replacement.write_bytes(original.read_bytes())
                    os.chmod(replacement, 0o400)
        return result

    # The mutation hook is confined to one helper and the exact tmp_path-owned
    # entry name. Other project modules and filesystem APIs remain untouched.
    monkeypatch.setattr(subject, "_typed_cache_read_file", change_after_first_entry_capture)
    assert _load(repository, binding) is None
    assert changed

