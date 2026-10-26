"""Pure config controls and real-file loader checks with authored owner proof.

The test host is not a deployed root-owned container. Only its root/namespace
ownership evidence is authored below. Bytes, hashes, regular-file safety,
private modes, path confinement and replacement detection use real files.
No scheduler contents are opened, no native capacity is acquired, and no
cross-UID or full Terminal deployment qualification is claimed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_configuration as configuration


def selected(path="/opt/owner/state/resource-scheduler.json", timeout=10):
    return {"schema": configuration.SCHEMA, "scheduler_state_path": path,
            "scheduler_policy_sha256": "a" * 64, "admission_timeout_seconds": timeout,
            "resource_scope": configuration.RESOURCE_SCOPE}


def validate(value, private=None):
    return configuration.validate_source_owner_configuration(value,
        owner_private_paths=["/opt/owner/state"] if private is None else private)


def test_pure_selection_does_not_touch_filesystem_or_mutate_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pure source resource selection accessed the filesystem")
    monkeypatch.setattr(Path, "lstat", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(configuration, "_owner_boundary", forbidden)
    value = selected()
    original = copy.deepcopy(value)
    result = validate(value)
    assert result == original and result is not value and value == original
    assert result["scheduler_state_path"] == "/opt/owner/state/resource-scheduler.json"


def test_absent_selection_preserves_legacy_without_private_path_validation():
    assert configuration.validate_source_owner_configuration(None, owner_private_paths=None) is None


@pytest.mark.parametrize("timeout", [1, 0.001, 30, 30.0])
def test_supported_bounded_timeout(timeout):
    value = validate(selected(timeout=timeout))
    assert value["admission_timeout_seconds"] == timeout
    assert type(value["admission_timeout_seconds"]) is type(timeout)


@pytest.mark.parametrize("timeout", [True, False, None, "1", 0, -1, 30.001,
                                     float("nan"), float("inf"), float("-inf"), 10 ** 500])
def test_invalid_timeout_is_refused(timeout):
    with pytest.raises(ValueError):
        validate(selected(timeout=timeout))


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign@1"), ("resource_scope", "whole_terminal_worker"),
    ("scheduler_policy_sha256", "A" * 64), ("scheduler_policy_sha256", "a" * 63),
    ("scheduler_policy_sha256", "a" * 64 + "\n"), ("scheduler_policy_sha256", None),
    ("scheduler_state_path", "/opt/owner/state"),
    ("scheduler_state_path", "/opt/owner/state-other/scheduler.json"),
    ("scheduler_state_path", "/opt/owner/state/../scheduler.json"),
    ("scheduler_state_path", "/opt/owner/state/./scheduler.json"),
    ("scheduler_state_path", "/opt/owner/state//scheduler.json"),
    ("scheduler_state_path", "//opt/owner/state/scheduler.json"),
    ("scheduler_state_path", "opt/owner/state/scheduler.json"),
    ("scheduler_state_path", "/opt/owner/state/scheduler.json/"),
    ("scheduler_state_path", "/opt/owner/state/bad\\name"),
    ("scheduler_state_path", "/opt/owner/state/bad\nname"),
    ("scheduler_state_path", "/opt/owner/state/bad\0name"),
    ("scheduler_state_path", "/opt/owner/state/bad\ud800name"),
    ("scheduler_state_path", "/opt/owner/state/" + "x" * 4096),
    ("scheduler_state_path", Path("/opt/owner/state/scheduler.json")),
    ("extra", "not admitted"),
])
def test_invalid_closed_configuration(field, value):
    altered = selected()
    altered[field] = value
    with pytest.raises(ValueError):
        validate(altered)


@pytest.mark.parametrize("field", sorted(configuration._FIELDS))
def test_missing_required_field(field):
    value = selected()
    del value[field]
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("value", [[], "", True, 1])
def test_nonobject_selection_refused(value):
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize("private", [[], (), "/opt/owner/state", ["/"], ["relative"],
    ["/opt/owner/state/"], ["/opt/owner/state", "/opt/owner/state"],
    [f"/private/{number}" for number in range(17)]])
def test_invalid_private_nominations(private):
    with pytest.raises(ValueError):
        validate(selected(), private=private)


def test_explicit_alternate_private_directory_is_allowed_without_default_selection():
    value = selected("/opt/owner/private/nested/scheduler.json")
    assert validate(value, private=["/opt/owner/state", "/opt/owner/private"]) == value


def metadata(info, **changes):
    names = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink",
             "st_size", "st_mtime_ns", "st_ctime_ns")
    values = {name: getattr(info, name) for name in names}
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "deployment"
    root.mkdir()
    private = root / "state"
    private.mkdir(mode=0o700)
    nested = private / "native"
    nested.mkdir(mode=0o700)
    state = nested / "resource-scheduler.json"
    # Deliberately not valid scheduler content: the loader must not parse it.
    state.write_bytes(b"existing scheduler contents are the native facade's responsibility\n")
    state.chmod(0o600)
    manifest = root / "container-boundary.json"
    owner = os.getuid()
    assert owner > 0, "this authored test fixture expects the nonroot admitted runner"
    value = {"schema": "supervisor-container-worker-boundary@1",
             "owner_uid": owner, "worker_uid": owner + 1,
             "namespaces": {key: os.readlink("/proc/self/ns/" + key) for key in ("pid", "mnt", "net")},
             "owner_private_paths": [str(private)],
             configuration.BOUNDARY_KEY: selected(str(state))}
    calls = []
    def write(payload=None, raw=None):
        if manifest.exists():
            manifest.chmod(0o600)
        manifest.write_bytes(raw if raw is not None else
            (json.dumps(value if payload is None else payload, sort_keys=True, indent=2) + "\n").encode())
        manifest.chmod(0o444)
    write()
    original_stat = configuration._stat
    def root_proof(path):
        info = original_stat(path)
        if path == manifest or path in manifest.parents:
            # Author only the root-deployer/host-temporary-ancestor proof.
            # Preserve inode, kind, links, size and timestamps from real files.
            return metadata(info, st_uid=0, st_mode=info.st_mode & ~0o022)
        return info
    monkeypatch.setattr(configuration, "_stat", root_proof)
    def owner_proof(path):
        calls.append(path)
        raw = configuration._read(path)
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(configuration, "_owner_boundary", owner_proof)
    return SimpleNamespace(root=root, private=private, nested=nested, state=state,
        manifest=manifest, value=value, owner=owner, calls=calls, write=write,
        stat=root_proof, owner_proof=owner_proof)


def test_loader_reads_only_boundary_and_returns_exact_selection(case, monkeypatch):
    opened = []
    original = os.open
    def observed(path, *args, **kwargs):
        opened.append(Path(path))
        assert Path(path) != case.state, "configuration loader opened active scheduler contents"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", observed)
    before = sorted(str(path) for path in case.root.rglob("*"))
    result = configuration.load_source_owner_configuration(case.manifest)
    assert result == case.value[configuration.BOUNDARY_KEY]
    assert opened and set(opened) == {case.manifest}
    assert case.calls == [case.manifest]
    assert before == sorted(str(path) for path in case.root.rglob("*"))


@pytest.mark.parametrize("explicit_null", [False, True])
def test_loader_legacy_missing_or_null_config_never_inspects_private_state(case, monkeypatch, explicit_null):
    if explicit_null:
        case.value[configuration.BOUNDARY_KEY] = None
    else:
        del case.value[configuration.BOUNDARY_KEY]
    case.value.pop("owner_private_paths")
    case.write()
    monkeypatch.setattr(configuration, "_private_chain", lambda *args, **kwargs:
        pytest.fail("legacy boundary inspected source scheduler"))
    assert configuration.load_source_owner_configuration(case.manifest) is None
    assert case.calls == [case.manifest]


def test_actual_owner_verifier_is_called_by_production_adapter(monkeypatch, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
    marker = ({"owner": "authored"}, "d" * 64)
    seen = []
    def proof(path):
        seen.append(path)
        return marker
    monkeypatch.setattr(guard, "_owner_boundary", proof)
    path = tmp_path / "boundary.json"
    assert configuration._owner_boundary(path) is marker
    assert seen == [path]


@pytest.mark.parametrize("failure", ["digest", "payload", "exception"])
def test_owner_proof_disagreement_or_failure_refuses(case, monkeypatch, failure):
    def rejected(path):
        value, digest = case.owner_proof(path)
        if failure == "exception":
            raise ValueError("authored deployed owner identity refusal")
        if failure == "digest":
            digest = "b" * 64
        else:
            value["owner_uid"] += 7
        return value, digest
    monkeypatch.setattr(configuration, "_owner_boundary", rejected)
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)


@pytest.mark.parametrize("target,change", [
    ("manifest", {"st_uid": 42}), ("root", {"st_uid": 42}),
    ("root", {"st_mode": stat.S_IFDIR | 0o775}),
    ("manifest", {"st_mode": stat.S_IFREG | 0o664}),
])
def test_manifest_full_ancestor_root_and_write_proof_is_required(case, monkeypatch, target, change):
    rejected = getattr(case, target)
    monkeypatch.setattr(configuration, "_stat", lambda path:
        metadata(case.stat(path), **change) if path == rejected else case.stat(path))
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)
    assert not case.calls


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o770, 0o711])
def test_private_root_must_be_exact_0700(case, mode):
    case.private.chmod(mode)
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)


def test_nested_scheduler_ancestor_cannot_be_worker_writable(case):
    case.nested.chmod(0o770)
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)


@pytest.mark.parametrize("target", ["private", "nested", "state"])
def test_foreign_owner_in_private_chain_refuses(case, monkeypatch, target):
    rejected = getattr(case, target)
    monkeypatch.setattr(configuration, "_stat", lambda path:
        metadata(case.stat(path), st_uid=case.owner + 999) if path == rejected else case.stat(path))
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)


@pytest.mark.parametrize("target", ["manifest", "private", "nested", "state"])
def test_symlink_anywhere_in_selected_paths_refuses(case, target):
    path = getattr(case, target)
    saved = path.with_name(path.name + ".saved")
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=saved.is_dir())
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)


@pytest.mark.parametrize("target", ["manifest", "state"])
def test_hardlinked_selected_file_refuses(case, target):
    path = getattr(case, target)
    os.link(path, path.with_name(path.name + ".hardlink"))
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)


@pytest.mark.parametrize("replacement", ["missing", "directory", "fifo"])
def test_selected_scheduler_must_already_be_regular(case, replacement):
    case.state.unlink()
    if replacement == "directory":
        case.state.mkdir()
    elif replacement == "fifo":
        os.mkfifo(case.state)
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)
    if replacement == "missing":
        assert not case.state.exists() and not case.state.with_suffix(".json.lock").exists()


@pytest.mark.parametrize("raw", [b"[]", b"{", b"{}" * 20000, b"", b'{"x":NaN}',
    b'{"repository_source_resources":null,"repository_source_resources":null}', b"\xff"])
def test_invalid_bounded_json_refuses(case, raw):
    case.write(raw=raw)
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(case.manifest)


def test_boundary_replacement_during_owner_validation_refuses(case, monkeypatch):
    def changed(path):
        result = case.owner_proof(path)
        replacement = path.with_name("replacement.json")
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o444)
        os.replace(replacement, path)
        return result
    monkeypatch.setattr(configuration, "_owner_boundary", changed)
    with pytest.raises(ValueError, match="identity changed"):
        configuration.load_source_owner_configuration(case.manifest)


def test_private_ancestor_replacement_during_validation_refuses(case, monkeypatch):
    original = configuration._private_chain
    calls = []
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            saved = case.nested.with_name("native.old")
            case.nested.rename(saved)
            case.nested.mkdir(mode=0o700)
            (case.nested / case.state.name).write_bytes(b"new unrelated state")
        return result
    monkeypatch.setattr(configuration, "_private_chain", changed)
    with pytest.raises(ValueError, match="ancestry changed"):
        configuration.load_source_owner_configuration(case.manifest)


def test_native_atomic_state_replacement_does_not_claim_content_stability(case, monkeypatch):
    original = configuration._private_chain
    calls = []
    def replaced(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            replacement = case.state.with_name("next.json")
            replacement.write_bytes(b"native-owned atomic update remains unread")
            os.replace(replacement, case.state)
        return result
    monkeypatch.setattr(configuration, "_private_chain", replaced)
    assert configuration.load_source_owner_configuration(case.manifest) == case.value[configuration.BOUNDARY_KEY]


@pytest.mark.parametrize("bad", ["relative.json", "/tmp/../boundary.json", "//tmp/boundary.json", None])
def test_noncanonical_boundary_refuses_before_verifier(case, bad):
    with pytest.raises(ValueError):
        configuration.load_source_owner_configuration(bad)
    assert not case.calls
