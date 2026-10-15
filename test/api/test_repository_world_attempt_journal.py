"""Durable private attempt custody without live or historical authority."""
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_attempt_journal as journal


def start_record():
    return dict(schema=journal.START_SCHEMA, attempt_id="case-1/baseline/0", case_id="case-1",
        arm="baseline", parent_attempt_id=None,
        observed_controls={key: "codex_cli" if key == "provider" else "authored-model" if key == "model"
            else "a" * 64 for key in journal.CONTROL_FIELDS})


def router_receipt():
    return dict(schema="router-implementation-invocation@1", invocation_id="b" * 32,
        provider="codex_cli", model="authored-model", purpose="coding", status="failed", usage=None,
        completion_authority=False,
        world_task_packet=dict(validation_count=1, dispatch_rechecked=True, provider_attempted=True),
        unknown_observation={"nullable": None, "elapsed": 1.25, "text": "café"})


@pytest.fixture
def owner(tmp_path):
    workspace = tmp_path / "workspace"
    root = tmp_path / "attempt"
    workspace.mkdir()
    root.mkdir(mode=0o700)
    value = journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start_record())
    try:
        yield value, root, workspace
    finally:
        journal.RepositoryWorldAttemptJournal.close(value)


def test_start_is_durable_before_entry_and_terminal_binds_full_receipt(owner):
    value, root, _workspace = owner
    before = value.receipt()
    assert set(p.name for p in root.iterdir()) == {"start.json"}
    assert json.loads((root / "start.json").read_bytes()) == start_record()
    original = router_receipt()
    value.retain_receipt(original)
    original["unknown_observation"]["text"] = "caller mutated"
    result = value.finish(outcome="raised", error_type="RuntimeError")
    assert result["status"] == "terminal_retained" and result["closed"] is False
    assert result["capture_error"] is False
    retained = json.loads((root / "receipt.json").read_bytes())
    assert retained == router_receipt() and retained["usage"] is None
    terminal = json.loads((root / "terminal.json").read_bytes())
    assert terminal == dict(schema=journal.TERMINAL_SCHEMA, attempt_id=start_record()["attempt_id"],
        start_sha256=before["start"]["sha256"], receipt_sha256=result["receipt"]["sha256"],
        outcome="raised", error_type="RuntimeError")
    for name in ("start", "receipt", "terminal"):
        row = result[name]
        raw = (root / row["name"]).read_bytes()
        assert row == dict(name=name + ".json", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        info = (root / row["name"]).lstat()
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_nlink == 1 and info.st_uid == os.geteuid()
    assert all(result[key] is False for key in journal.FALSE_FIELDS)
    assert "tokens" not in result and "workspace" not in result


@pytest.mark.parametrize("outcome,error_type", [("returned", None), ("raised", None),
    ("interrupted", "KeyboardInterrupt")])
def test_missing_receipt_remains_null_and_finish_does_not_invent_provider_event(owner, outcome, error_type):
    value, root, _workspace = owner
    value.mark_capture_error()
    result = value.finish(outcome=outcome, error_type=error_type)
    assert result["capture_error"] is True and result["receipt"] is None
    assert json.loads((root / "terminal.json").read_bytes())["receipt_sha256"] is None
    assert not (root / "receipt.json").exists()


def test_context_manager_closes_without_inventing_terminal_and_receipt_is_detached(owner):
    value, root, _workspace = owner
    with value as entered:
        assert entered is value
    row = value.receipt()
    assert row["closed"] is True and row["terminal"] is None and row["status"] == "started"
    row["start"]["sha256"] = "0" * 64
    assert value.receipt()["start"]["sha256"] != "0" * 64
    value.mark_capture_error()
    assert value.receipt()["capture_error"] is True
    with pytest.raises(ValueError, match="closed"):
        value.require_open()
    assert set(p.name for p in root.iterdir()) == {"start.json"}


def test_live_workspace_must_match_constructor_workspace(owner, tmp_path):
    value, root, workspace = owner
    value.require_open(workspace=workspace)
    foreign = tmp_path / "other-workspace"; foreign.mkdir()
    with pytest.raises(ValueError, match="workspace_mismatch"):
        value.require_open(workspace=foreign)
    with pytest.raises(ValueError, match="workspace_mismatch"):
        value.require_open(workspace=root.parent)
    assert value.receipt()["receipt"] is None


def test_matching_workspace_replaced_by_symlink_is_refused(owner, tmp_path):
    value, _root, workspace = owner
    original = tmp_path / "original-workspace"
    workspace.rename(original); workspace.symlink_to(original)
    with pytest.raises(OSError): value.require_open(workspace=workspace)


def test_terminal_is_exclusive_and_no_more_receipts_can_be_added(owner):
    value, root, _workspace = owner
    value.finish(outcome="returned")
    before = (root / "terminal.json").read_bytes()
    with pytest.raises(ValueError, match="already_terminal"):
        value.finish(outcome="raised")
    with pytest.raises(ValueError, match="already_terminal"):
        value.retain_receipt(router_receipt())
    assert (root / "terminal.json").read_bytes() == before


def test_receipt_is_exclusive_and_missing_usage_is_preserved(owner):
    value, root, _workspace = owner
    value.retain_receipt(router_receipt())
    before = (root / "receipt.json").read_bytes()
    with pytest.raises(ValueError, match="already_retained"):
        value.retain_receipt(router_receipt())
    assert (root / "receipt.json").read_bytes() == before


@pytest.mark.parametrize("case", ["relative", "missing", "public", "root_symlink", "parent_symlink",
    "workspace_symlink", "inside_workspace", "ancestor_workspace", "occupied"])
def test_root_requires_empty_private_canonical_disjoint_directory(tmp_path, case):
    workspace = tmp_path / "workspace"
    root = tmp_path / "attempt"
    workspace.mkdir()
    root.mkdir(mode=0o700)
    if case == "relative": root = Path("attempt")
    elif case == "missing": root = tmp_path / "missing"
    elif case == "public": root.chmod(0o755)
    elif case == "root_symlink":
        link = tmp_path / "root-link"; link.symlink_to(root); root = link
    elif case == "parent_symlink":
        link = tmp_path / "parent-link"; link.symlink_to(tmp_path); root = link / "attempt"
    elif case == "workspace_symlink":
        link = tmp_path / "workspace-link"; link.symlink_to(workspace); workspace = link
    elif case == "inside_workspace": root = workspace / "attempt"; root.mkdir(mode=0o700)
    elif case == "ancestor_workspace": root = tmp_path
    else: (root / "foreign").write_bytes(b"foreign")
    with pytest.raises((ValueError, OSError)):
        journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start_record())
    if case == "occupied": assert (root / "foreign").read_bytes() == b"foreign"


def test_foreign_owner_refused(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = tmp_path / "attempt"; root.mkdir(mode=0o700)
    monkeypatch.setattr(journal.os, "geteuid", lambda: root.stat().st_uid + 1)
    with pytest.raises(ValueError, match="root_not_private"):
        journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start_record())
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("change", ["schema", "extra", "attempt", "case", "arm", "parent", "controls",
    "upper_pin", "bool_pin", "provider", "model"])
def test_closed_start_refuses_invalid_caller_assertions_before_file_creation(tmp_path, change):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = tmp_path / "attempt"; root.mkdir(mode=0o700)
    start = start_record()
    if change == "schema": start["schema"] = "other"
    elif change == "extra": start["proof_authority"] = True
    elif change == "attempt": start["attempt_id"] = "../bad id"
    elif change == "case": start["case_id"] = True
    elif change == "arm": start["arm"] = "other"
    elif change == "parent": start["parent_attempt_id"] = start["attempt_id"]
    elif change == "controls": start["observed_controls"].pop("source_sha256")
    elif change == "upper_pin": start["observed_controls"]["source_sha256"] = "A" * 64
    elif change == "bool_pin": start["observed_controls"]["source_sha256"] = True
    elif change == "provider": start["observed_controls"]["provider"] = " codex_cli"
    else: start["observed_controls"]["model"] = "model\0"
    with pytest.raises(ValueError):
        journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start)
    assert list(root.iterdir()) == []


def test_nullable_controls_are_preserved_as_caller_observations(tmp_path):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = tmp_path / "attempt"; root.mkdir(mode=0o700)
    start = start_record()
    start["observed_controls"] = dict.fromkeys(journal.CONTROL_FIELDS)
    with journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start):
        assert json.loads((root / "start.json").read_bytes())["observed_controls"] == start["observed_controls"]


@pytest.mark.parametrize("mutation", ["schema", "invocation", "provider", "model", "status", "authority", "error_type", "purpose", "no_packet", "readonly", "returned_unattempted",
    "no_validation", "bool_count", "unrechecked"])
def test_receipt_requires_strict_router_observation_without_authenticating_it(owner, mutation):
    value, root, _workspace = owner
    row = router_receipt()
    if mutation == "schema": row["schema"] = "other"
    elif mutation == "invocation": row["invocation_id"] = "B" * 32
    elif mutation == "provider": row["provider"] = "other"
    elif mutation == "model": row["model"] = "bad model"
    elif mutation == "status": row["status"] = "unknown"
    elif mutation == "authority": row["completion_authority"] = True
    elif mutation == "error_type": row["error_type"] = "private error text"
    elif mutation == "returned_unattempted":
        row["status"] = "provider_returned"
        row["world_task_packet"].update(dispatch_rechecked=False, provider_attempted=False)
    elif mutation == "purpose": row["purpose"] = "planning"
    elif mutation == "no_packet": row["world_task_packet"] = None
    elif mutation == "readonly": row["world_task_packet"]["validation_profile"] = "read_only_observation@1"
    elif mutation == "no_validation": row["world_task_packet"]["validation_count"] = 0
    elif mutation == "bool_count": row["world_task_packet"]["validation_count"] = True
    else: row["world_task_packet"]["dispatch_rechecked"] = False
    with pytest.raises(ValueError): value.retain_receipt(row)
    assert set(p.name for p in root.iterdir()) == {"start.json"}


@pytest.mark.parametrize("mutation", ["cyclic", "nonfinite", "oversize", "deep", "object", "surrogate", "huge_int"])
def test_bounded_literal_receipt_json_rejects_before_creating_leaf(owner, mutation):
    value, root, _workspace = owner
    row = router_receipt()
    if mutation == "cyclic": row["cycle"] = row
    elif mutation == "nonfinite": row["bad"] = float("nan")
    elif mutation == "oversize": row["bad"] = "x" * (journal.RECEIPT_MAX_BYTES + 1)
    elif mutation == "deep":
        child = []
        for _ in range(journal.MAX_DEPTH + 1): child = [child]
        row["bad"] = child
    elif mutation == "object": row["bad"] = object()
    elif mutation == "surrogate": row["bad"] = "\ud800"
    else: row["bad"] = 1 << 65
    with pytest.raises(ValueError): value.retain_receipt(row)
    assert not (root / "receipt.json").exists()


@pytest.mark.parametrize("change", ["replacement", "rewrite", "hardlink", "removed_hardlink", "symlink", "chmod", "delete"])
def test_initial_inode_identity_and_contents_are_rechecked_before_join(owner, tmp_path, change):
    value, root, _workspace = owner
    leaf = root / "start.json"
    raw = leaf.read_bytes()
    if change == "replacement": leaf.rename(root / "original-start"); leaf.write_bytes(raw); leaf.chmod(0o600)
    elif change == "rewrite":
        leaf.write_bytes(raw)
        info = leaf.stat(); os.utime(leaf, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
    elif change == "hardlink": os.link(leaf, tmp_path / "alias")
    elif change == "removed_hardlink": os.link(leaf, tmp_path / "alias"); (tmp_path / "alias").unlink()
    elif change == "symlink": leaf.rename(root / "original-start"); leaf.symlink_to(root / "original-start")
    elif change == "chmod": leaf.chmod(0o400)
    else: leaf.unlink()
    with pytest.raises((ValueError, OSError)): value.finish(outcome="raised")
    assert not (root / "terminal.json").exists()


@pytest.mark.parametrize("change", ["replace", "chmod"])
def test_root_identity_is_retained_and_foreign_replacement_preserved(owner, tmp_path, change):
    value, root, _workspace = owner
    if change == "replace":
        root.rename(tmp_path / "original-attempt"); root.mkdir(mode=0o700)
        (root / "foreign").write_bytes(b"foreign")
    else: root.chmod(0o755)
    with pytest.raises(ValueError, match="root_changed"): value.require_open()
    if change == "replace": assert (root / "foreign").read_bytes() == b"foreign"


@pytest.mark.parametrize("kind", ["regular", "symlink", "directory", "hardlink"])
def test_foreign_receipt_leaf_is_never_rewritten_or_deleted(owner, tmp_path, kind):
    value, root, _workspace = owner
    foreign = tmp_path / "foreign"; foreign.write_bytes(b"foreign")
    path = root / "receipt.json"
    if kind == "regular": path.write_bytes(b"occupied")
    elif kind == "symlink": path.symlink_to(foreign)
    elif kind == "directory": path.mkdir()
    else: os.link(foreign, path)
    identity = path.lstat().st_ino
    with pytest.raises(OSError): value.retain_receipt(router_receipt())
    assert path.lstat().st_ino == identity and foreign.read_bytes() == b"foreign"
    assert value.receipt()["receipt"] is None


@pytest.mark.parametrize("fault", ["write", "zero_write", "fsync", "readback"])
def test_capture_fault_preserves_partial_evidence_without_publishing_descriptor(owner, monkeypatch, fault):
    value, root, _workspace = owner
    if fault == "write":
        write = journal.os.write
        def failed(fd, data): write(fd, data[:4]); raise OSError("PRIVATE_WRITE_FAILURE")
        monkeypatch.setattr(journal.os, "write", failed)
    elif fault == "zero_write": monkeypatch.setattr(journal.os, "write", lambda *_args: 0)
    elif fault == "fsync":
        def failed(_fd): raise OSError("PRIVATE_FSYNC_FAILURE")
        monkeypatch.setattr(journal.os, "fsync", failed)
    else:
        read = journal.os.read
        def altered(fd, size):
            raw = read(fd, size)
            # Corrupt only receipt readback; start checks must finish first.
            return b"X" + raw[1:] if raw and b'"invocation_id"' in raw else raw
        monkeypatch.setattr(journal.os, "read", altered)
    with pytest.raises((ValueError, OSError)): value.retain_receipt(router_receipt())
    assert (root / "receipt.json").exists()
    assert value.receipt()["receipt"] is None
    assert value.receipt()["terminal"] is None


@pytest.mark.parametrize("outcome,error_type", [("other", None), (True, None), ("returned", "RuntimeError"),
    ("raised", "private message"), ("raised", "a" * 129)])
def test_terminal_fields_are_closed_and_do_not_capture_exception_text(owner, outcome, error_type):
    value, root, _workspace = owner
    with pytest.raises(ValueError): value.finish(outcome=outcome, error_type=error_type)
    assert not (root / "terminal.json").exists()


def test_exact_issued_objects_and_classqualified_guards(owner):
    value, _root, _workspace = owner
    clone = object.__new__(journal.RepositoryWorldAttemptJournal)
    with pytest.raises(ValueError, match="exact_issued"):
        journal.RepositoryWorldAttemptJournal.require_open(clone)
    class Foreign(journal.RepositoryWorldAttemptJournal):
        pass
    foreign = object.__new__(Foreign)
    with pytest.raises(ValueError, match="exact_issued"):
        journal.RepositoryWorldAttemptJournal.require_open(foreign)
    with pytest.raises(AttributeError): value.require_open = lambda: None
    seal = object.__getattribute__(value, "_seal")
    object.__setattr__(value, "_seal", object())
    try:
        with pytest.raises(ValueError, match="issued_material_changed"):
            journal.RepositoryWorldAttemptJournal.require_open(value)
    finally:
        object.__setattr__(value, "_seal", seal)


def test_incomplete_start_write_is_preserved_without_issuing_a_journal(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = tmp_path / "attempt"; root.mkdir(mode=0o700)
    write = journal.os.write
    def failed(fd, data): write(fd, data[:3]); raise OSError("PRIVATE_START_FAILURE")
    monkeypatch.setattr(journal.os, "write", failed)
    with pytest.raises(OSError):
        journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start_record())
    assert (root / "start.json").read_bytes() == b'{"a'


def test_json_subclasses_never_run_conversion_hooks(owner):
    value, root, _workspace = owner
    class Foreign(dict):
        def items(self): pytest.fail("subclass callback invoked")
    row = router_receipt(); row["foreign"] = Foreign()
    with pytest.raises(ValueError, match="json_type"): value.retain_receipt(row)
    assert not (root / "receipt.json").exists()


def test_oversized_dictionary_population_is_bounded_before_alias_expansion(owner, monkeypatch):
    value, root, _workspace = owner
    shared = {str(index): [] for index in range(1024)}
    row = router_receipt(); row["large"] = shared; row["alias"] = shared
    monkeypatch.setattr(journal, "MAX_NODES", 128)
    with pytest.raises(ValueError, match="structure_bound"):
        value.retain_receipt(row)
    assert not (root / "receipt.json").exists()


def test_start_and_terminal_identity_come_from_exact_serialized_snapshot(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = tmp_path / "attempt"; root.mkdir(mode=0o700)
    start = start_record()
    original_id = start["attempt_id"]
    wire = journal._wire
    def mutate_after_snapshot(value, maximum):
        raw = wire(value, maximum)
        if value is start: start["attempt_id"] = "changed-after-capture"
        return raw
    monkeypatch.setattr(journal, "_wire", mutate_after_snapshot)
    with journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start) as value:
        result = value.finish(outcome="raised")
        assert result["attempt_id"] == original_id
    assert json.loads((root / "start.json").read_bytes())["attempt_id"] == original_id
    assert json.loads((root / "terminal.json").read_bytes())["attempt_id"] == original_id


@pytest.mark.parametrize("valid_snapshot", [True, False])
def test_receipt_validates_captured_bytes_despite_later_caller_mutation(owner, monkeypatch, valid_snapshot):
    value, root, _workspace = owner
    receipt = router_receipt()
    receipt["provider"] = "codex_cli" if valid_snapshot else "other"
    wire = journal._wire
    def mutate_after_snapshot(item, maximum):
        raw = wire(item, maximum)
        if item is receipt: receipt["provider"] = "other" if valid_snapshot else "codex_cli"
        return raw
    monkeypatch.setattr(journal, "_wire", mutate_after_snapshot)
    if valid_snapshot:
        value.retain_receipt(receipt)
        assert json.loads((root / "receipt.json").read_bytes())["provider"] == "codex_cli"
    else:
        with pytest.raises(ValueError, match="router_receipt"): value.retain_receipt(receipt)
        assert not (root / "receipt.json").exists()


def test_invalid_start_snapshot_is_not_validated_using_later_valid_caller_state(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = tmp_path / "attempt"; root.mkdir(mode=0o700)
    start = start_record(); start["attempt_id"] = "invalid id"
    wire = journal._wire
    def mutate_after_snapshot(value, maximum):
        raw = wire(value, maximum)
        if value is start: start["attempt_id"] = "now-valid"
        return raw
    monkeypatch.setattr(journal, "_wire", mutate_after_snapshot)
    with pytest.raises(ValueError, match="start_identity"):
        journal.RepositoryWorldAttemptJournal(root, workspace=workspace, start=start)
    assert list(root.iterdir()) == []
