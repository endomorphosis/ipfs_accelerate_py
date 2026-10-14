"""Private file custody and live-dispatch historical retention boundaries."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_artifact_retention as retention
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet_replay as packet_replay
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_invocation_replay as invocation_replay
from test.api.test_repository_world_dispatch import (
    TASK, dispatch_case, invoke, invocation_receipt, world_session,
)
from test.api.test_repository_context_compiler import required
from test.api.test_repository_source_index import scheduler


INVOCATION = "a" * 32


def private_root(tmp_path, name="artifacts"):
    path = tmp_path / name
    path.mkdir(mode=0o700)
    return path


def store(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = private_root(tmp_path)
    return retention.RepositoryWorldArtifactRetention(root, workspace=workspace,
        invocation_id=INVOCATION), root, workspace


def retained_receipt():
    return dict(schema=retention.SCHEMA, status="retained", reason=None,
        packet=dict(name=retention.artifact_name(INVOCATION, "packet"), bytes=5, sha256="b" * 64),
        invocation=dict(name=retention.artifact_name(INVOCATION, "invocation"), bytes=10, sha256="c" * 64),
        **dict.fromkeys(retention.FALSE_FIELDS, False))


def test_exclusive_private_regular_write_and_exact_readback(tmp_path):
    owner, root, _workspace = store(tmp_path)
    raw = b'{"literal":"caf\xc3\xa9"}'
    try:
        descriptor = owner._write("packet", raw)
        path = root / descriptor["name"]
        assert path.read_bytes() == raw
        assert descriptor == dict(name=INVOCATION + "-world-packet.json",
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_nlink == 1 and info.st_uid == os.geteuid()
        with pytest.raises(FileExistsError):
            owner._write("packet", b"replacement")
        assert path.read_bytes() == raw
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ["regular", "symlink", "directory", "hardlink"])
def test_existing_foreign_entry_is_preserved(tmp_path, kind):
    owner, root, _workspace = store(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign data")
    path = root / retention.artifact_name(INVOCATION, "packet")
    if kind == "regular": path.write_bytes(b"existing data")
    elif kind == "symlink": path.symlink_to(foreign)
    elif kind == "directory": path.mkdir()
    else: os.link(foreign, path)
    before = path.lstat()
    try:
        with pytest.raises(OSError): owner._write("packet", b"new data")
        after = path.lstat()
        assert (before.st_dev, before.st_ino, before.st_mode) == (after.st_dev, after.st_ino, after.st_mode)
        assert foreign.read_bytes() == b"foreign data"
    finally:
        owner.close()


@pytest.mark.parametrize("case", ["relative", "missing", "public", "root_symlink", "parent_symlink",
    "inside_workspace", "ancestor_workspace", "workspace_symlink"])
def test_root_requires_private_disjoint_canonical_directory(tmp_path, case):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = private_root(tmp_path)
    if case == "relative": root = Path("relative")
    elif case == "missing": root = tmp_path / "missing"
    elif case == "public": root.chmod(0o755)
    elif case == "root_symlink":
        link = tmp_path / "link"; link.symlink_to(root); root = link
    elif case == "parent_symlink":
        link = tmp_path / "link"; link.symlink_to(tmp_path); root = link / "artifacts"
    elif case == "inside_workspace": root = workspace / "artifacts"; root.mkdir(mode=0o700)
    elif case == "ancestor_workspace": root = tmp_path
    elif case == "workspace_symlink":
        link = tmp_path / "workspace-link"; link.symlink_to(workspace); workspace = link
    with pytest.raises((ValueError, OSError)):
        retention.RepositoryWorldArtifactRetention(root, workspace=workspace, invocation_id=INVOCATION)


def test_foreign_root_owner_is_refused(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = private_root(tmp_path)
    monkeypatch.setattr(retention.os, "geteuid", lambda: root.stat().st_uid + 1)
    with pytest.raises(ValueError, match="not_private"):
        retention.RepositoryWorldArtifactRetention(root, workspace=workspace, invocation_id=INVOCATION)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("bad", [None, True, "a" * 31, "A" * 32, "../" + "a" * 32])
def test_invocation_identity_is_exact(tmp_path, bad):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    root = private_root(tmp_path)
    with pytest.raises(ValueError):
        retention.RepositoryWorldArtifactRetention(root, workspace=workspace, invocation_id=bad)


def test_data_bound_precedes_file_creation(tmp_path, monkeypatch):
    owner, root, _workspace = store(tmp_path)
    monkeypatch.setattr(retention, "PACKET_MAX_BYTES", 8)
    def forbidden(*_args, **_kwargs): pytest.fail("opened file before byte refusal")
    monkeypatch.setattr(retention.os, "open", forbidden)
    try:
        with pytest.raises(ValueError, match="byte_limit"): owner._write("packet", b"x" * 9)
        assert list(root.iterdir()) == []
    finally:
        owner.close()


def test_directory_retarget_during_write_refuses_without_deleting_evidence(tmp_path, monkeypatch):
    owner, root, _workspace = store(tmp_path)
    original_write = retention.os.write
    old = tmp_path / "retained-original-root"
    moved = False
    def move(fd, data):
        nonlocal moved
        count = original_write(fd, data)
        if not moved:
            moved = True; root.rename(old); root.mkdir(mode=0o700)
        return count
    monkeypatch.setattr(retention.os, "write", move)
    try:
        with pytest.raises(ValueError, match="root_changed"): owner._write("packet", b"original")
        assert (old / retention.artifact_name(INVOCATION, "packet")).read_bytes() == b"original"
        assert list(root.iterdir()) == []
    finally:
        owner.close()


def test_readback_tamper_is_refused_and_file_preserved(tmp_path, monkeypatch):
    owner, root, _workspace = store(tmp_path)
    read = retention.os.read
    def altered(fd, size):
        raw = read(fd, size)
        return b"X" + raw[1:] if raw else raw
    monkeypatch.setattr(retention.os, "read", altered)
    try:
        with pytest.raises(ValueError, match="readback_mismatch"): owner._write("packet", b"original")
        assert (root / retention.artifact_name(INVOCATION, "packet")).read_bytes() == b"original"
    finally:
        owner.close()


def test_fsync_failure_keeps_created_file(tmp_path, monkeypatch):
    owner, root, _workspace = store(tmp_path)
    def failed(_fd): raise OSError("private fsync failure")
    monkeypatch.setattr(retention.os, "fsync", failed)
    try:
        with pytest.raises(OSError): owner._write("packet", b"original")
        assert (root / retention.artifact_name(INVOCATION, "packet")).read_bytes() == b"original"
    finally:
        owner.close()


@pytest.mark.parametrize("mutation", ["authority", "extra", "absolute_name", "foreign_id", "boolean_bytes",
    "oversize", "upper_sha", "missing_packet", "wrong_status", "free_reason"])
def test_closed_receipt_refuses_forged_bindings(mutation):
    row = retained_receipt()
    if mutation == "authority": row["proof_authority"] = True
    elif mutation == "extra": row["prompt"] = "private text"
    elif mutation == "absolute_name": row["packet"]["name"] = "/tmp/artifact.json"
    elif mutation == "foreign_id": row["packet"]["name"] = "d" * 32 + "-world-packet.json"
    elif mutation == "boolean_bytes": row["packet"]["bytes"] = True
    elif mutation == "oversize": row["packet"]["bytes"] = retention.PACKET_MAX_BYTES + 1
    elif mutation == "upper_sha": row["packet"]["sha256"] = "B" * 64
    elif mutation == "missing_packet": row["packet"] = None
    elif mutation == "wrong_status": row["status"] = []
    else: row["reason"] = "private error detail"
    with pytest.raises(ValueError): retention.validate_world_packet_artifacts(row, invocation_id=INVOCATION)


def test_receipt_copy_is_detached_and_unknown_totals_are_not_invented():
    row = retained_receipt()
    copied = retention.validate_world_packet_artifacts(row, invocation_id=INVOCATION)
    copied["packet"]["bytes"] = 9
    assert row["packet"]["bytes"] == 5
    unavailable = retention.unavailable_world_packet_artifacts(invocation_id=INVOCATION,
        reason="retention_not_requested")
    assert unavailable["packet"] is unavailable["invocation"] is None
    assert all(unavailable[key] is False for key in retention.FALSE_FIELDS)
    assert "tokens" not in unavailable


def test_live_dispatch_retains_packet_before_attempt_and_invocation_after_edit(dispatch_case, tmp_path):
    root = private_root(tmp_path)
    source, session, packet, state = dispatch_case
    def edit(_prompt, _options):
        names = list(root.iterdir())
        assert len(names) == 1 and names[0].name.endswith("-world-packet.json")
        (source / "calc.py").write_text("def add(value):\n    return value + 2\n")
        session.close()
    state["callback"] = edit
    output, receipt = invoke(dispatch_case, world_packet_artifact_dir=root)
    artifact = retention.validate_world_packet_artifacts(receipt["world_packet_artifacts"],
        invocation_id=receipt["invocation_id"])
    assert artifact["status"] == "retained" and receipt["status"] == "provider_returned"
    raw = (root / artifact["invocation"]["name"]).read_bytes()
    replayed = invocation_replay.replay_repository_world_invocation(raw,
        expected_sha256=artifact["invocation"]["sha256"])
    assert replayed.model_prompt == state["seen"][0][0]
    assert replayed.receipt()["provider_delivery_verified"] is False
    assert receipt["usage"]["prompt_tokens"] == 15
    assert output.startswith("Authored")
    assert TASK not in json.dumps(artifact) and str(root) not in json.dumps(artifact)
    with pytest.raises(ValueError): packet_replay.freeze_repository_world_packet(packet)


@pytest.mark.parametrize("failure", [TimeoutError("authored timeout"), RuntimeError("authored provider failure")])
def test_failed_attempt_retains_complete_historical_input(dispatch_case, tmp_path, capsys, failure):
    root = private_root(tmp_path)
    def failed(_prompt, _options): raise failure
    dispatch_case[3]["callback"] = failed
    with pytest.raises(type(failure)) as caught:
        invoke(dispatch_case, world_packet_artifact_dir=root)
    assert caught.value is failure
    receipt = invocation_receipt(capsys)
    assert receipt["status"] == "failed" and receipt["usage"]["prompt_tokens"] == 15
    assert receipt["world_task_packet"]["provider_attempted"] is True
    row = receipt["world_packet_artifacts"]
    assert row["status"] == "retained"
    replayed = invocation_replay.replay_repository_world_invocation((root / row["invocation"]["name"]).read_bytes(),
        expected_sha256=row["invocation"]["sha256"])
    assert replayed.model_prompt == dispatch_case[3]["seen"][0][0]


def test_pre_attempt_packet_export_failure_refuses_provider(dispatch_case, tmp_path, monkeypatch, capsys):
    root = private_root(tmp_path)
    def failed(_packet): raise OSError("PRIVATE_PACKET_EXPORT_FAILURE")
    monkeypatch.setattr(packet_replay, "freeze_repository_world_packet", failed)
    with pytest.raises(OSError): invoke(dispatch_case, world_packet_artifact_dir=root)
    assert dispatch_case[3]["seen"] == []
    row = invocation_receipt(capsys)
    assert row["world_task_packet"]["provider_attempted"] is False
    assert row["world_packet_artifacts"]["reason"] == "packet_retention_failed"
    assert "PRIVATE_PACKET_EXPORT_FAILURE" not in json.dumps(row)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("provider_failure", [False, True])
def test_post_attempt_export_failure_preserves_primary_outcome(dispatch_case, tmp_path, monkeypatch, capsys, provider_failure):
    root = private_root(tmp_path)
    def failed_export(**_kwargs): raise ValueError("PRIVATE_INVOCATION_EXPORT_FAILURE")
    monkeypatch.setattr(invocation_replay, "freeze_repository_world_invocation", failed_export)
    failure = RuntimeError("authored primary provider failure")
    if provider_failure:
        def failed(_prompt, _options): raise failure
        dispatch_case[3]["callback"] = failed
        with pytest.raises(RuntimeError) as caught: invoke(dispatch_case, world_packet_artifact_dir=root)
        assert caught.value is failure
        row = invocation_receipt(capsys)
    else:
        output, row = invoke(dispatch_case, world_packet_artifact_dir=root)
        assert output.startswith("Authored")
    assert row["status"] == ("failed" if provider_failure else "provider_returned")
    assert row["usage"]["prompt_tokens"] == 15
    assert row["world_packet_artifacts"]["reason"] == "invocation_retention_failed"
    assert row["world_packet_artifacts"]["packet"] is not None
    assert row["world_packet_artifacts"]["invocation"] is None
    assert "PRIVATE_INVOCATION_EXPORT_FAILURE" not in json.dumps(row)
    assert len(list(root.iterdir())) == 1


def test_retained_packet_tampering_blocks_join_without_rewriting(dispatch_case, tmp_path, capsys):
    root = private_root(tmp_path)
    def tamper(_prompt, _options):
        path = next(root.iterdir()); path.write_bytes(b"mutated retained packet")
    dispatch_case[3]["callback"] = tamper
    output, row = invoke(dispatch_case, world_packet_artifact_dir=root)
    assert output.startswith("Authored") and row["status"] == "provider_returned"
    assert row["world_packet_artifacts"]["reason"] == "invocation_retention_failed"
    assert next(root.iterdir()).read_bytes() == b"mutated retained packet"
    assert len(list(root.iterdir())) == 1


def test_existing_invocation_file_after_attempt_is_preserved(dispatch_case, tmp_path):
    root = private_root(tmp_path)
    foreign = []
    def occupied(_prompt, _options):
        packet = next(root.iterdir())
        target = root / packet.name.replace("-world-packet.json", "-world-invocation.json")
        target.write_bytes(b"foreign existing invocation")
        foreign.append(target)
    dispatch_case[3]["callback"] = occupied
    output, row = invoke(dispatch_case, world_packet_artifact_dir=root)
    assert output.startswith("Authored") and row["status"] == "provider_returned"
    assert row["world_packet_artifacts"]["reason"] == "invocation_retention_failed"
    assert foreign[0].read_bytes() == b"foreign existing invocation"


def test_post_attempt_close_failure_cannot_replace_provider_result(dispatch_case, tmp_path, monkeypatch):
    root = private_root(tmp_path)
    close = retention.RepositoryWorldArtifactRetention.close
    def failed_close(self):
        close(self)
        raise OSError("PRIVATE_CLOSE_FAILURE")
    monkeypatch.setattr(retention.RepositoryWorldArtifactRetention, "close", failed_close)
    output, row = invoke(dispatch_case, world_packet_artifact_dir=root)
    assert output.startswith("Authored") and row["status"] == "provider_returned"
    assert row["usage"]["prompt_tokens"] == 15
    assert row["world_packet_artifacts"]["reason"] == "invocation_retention_failed"
    assert "PRIVATE_CLOSE_FAILURE" not in json.dumps(row)
    assert len(list(root.iterdir())) == 2


def test_no_retention_requested_is_explicit_without_legacy_mutation(dispatch_case):
    _output, world = invoke(dispatch_case)
    assert world["world_packet_artifacts"]["reason"] == "retention_not_requested"
    _output, legacy = invoke(dispatch_case, world_task_packet=None)
    assert "world_task_packet" not in legacy and "world_packet_artifacts" not in legacy


def test_retention_root_without_live_packet_refuses_before_provider(dispatch_case, tmp_path):
    root = private_root(tmp_path)
    with pytest.raises(ValueError):
        invoke(dispatch_case, world_task_packet=None, world_packet_artifact_dir=root)
    assert dispatch_case[3]["seen"] == [] and list(root.iterdir()) == []
