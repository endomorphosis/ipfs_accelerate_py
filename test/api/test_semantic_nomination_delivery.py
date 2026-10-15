"""Host snapshots and actual publisher filesystem operations without containers.

The local subprocess harness adapts only OWNER_UID to the host test UID.
Unmodified production code must refuse nonroot. No root-container or provider
qualification is claimed by this authored authority adaptation.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import semantic_decision_nomination_delivery as delivery

pytest_plugins = ("test.api.test_semantic_router_integration",)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def nomination(tmp_path):
    value = {"schema": delivery.NOMINATION_SCHEMA, "policy": {
        "required_capsule_ids": ["cid:required-λ"], "optional_capsule_ids": ["cid:optional-雪"],
        "selected_optional_capsule_ids": []}}
    raw = ("  " + json.dumps(value, ensure_ascii=False, indent=2) + "\r\n").encode()
    path = tmp_path / "nomination with spaces.json"
    path.write_bytes(raw)
    path.chmod(0o644)
    return path, raw, _sha(raw)


def test_snapshot_preserves_raw_bytes_and_is_a_new_immutable_inode(nomination, tmp_path):
    source, raw, digest = nomination
    directory = tmp_path / "trial"
    directory.mkdir(mode=0o755)
    snapshot = delivery.capture_nomination_snapshot(source=source, expected_sha256=digest, directory=directory)
    assert set(snapshot) == {"schema", "artifact", "sha256", "bytes"}
    artifact = Path(snapshot["artifact"])
    assert delivery.verify_nomination_snapshot(snapshot, directory=directory) == raw
    assert artifact.read_bytes() == raw and artifact.stat().st_ino != source.stat().st_ino
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444 and artifact.stat().st_nlink == 1
    source.write_bytes(raw + b" ")
    assert delivery.verify_nomination_snapshot(snapshot, directory=directory) == raw
    with pytest.raises((ValueError, OSError)):
        delivery.capture_nomination_snapshot(source=artifact, expected_sha256=digest, directory=directory)
    assert artifact.read_bytes() == raw and not list(directory.glob(".nomination-pending-*"))


@pytest.mark.parametrize("alteration", ["symlink", "parent_symlink", "fifo", "directory", "wrong_sha", "oversized"])
def test_source_requires_pinned_bounded_nofollow_regular_bytes(nomination, tmp_path, alteration):
    source, raw, digest = nomination
    if alteration == "symlink":
        target = source.with_name("alias.json")
        target.symlink_to(source)
        source = target
    elif alteration == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        source = alias / source.name
    elif alteration == "fifo":
        source = tmp_path / "pipe"
        os.mkfifo(source)
    elif alteration == "directory":
        source = tmp_path
    elif alteration == "wrong_sha":
        digest = "0" * 64
    else:
        source.write_bytes(b"x" * (delivery.MAX_NOMINATION_BYTES + 1))
        digest = _sha(source.read_bytes())
    with pytest.raises((ValueError, OSError)):
        delivery.read_nomination_source(source=source, expected_sha256=digest)


@pytest.mark.parametrize("alteration", ["schema", "extra", "missing", "duplicate", "overlap", "selected", "nonlist", "utf8"])
def test_host_and_remote_closed_nomination_schema_agree(nomination, tmp_path, alteration):
    source, raw, _ = nomination
    value = json.loads(raw)
    if alteration == "schema":
        value["schema"] = "foreign@1"
    elif alteration == "extra":
        value["authority"] = True
    elif alteration == "missing":
        del value["policy"]["required_capsule_ids"]
    elif alteration == "overlap":
        value["policy"]["optional_capsule_ids"] = value["policy"]["required_capsule_ids"]
    elif alteration == "selected":
        value["policy"]["selected_optional_capsule_ids"] = ["foreign"]
    elif alteration == "nonlist":
        value["policy"]["optional_capsule_ids"] = "cid:optional"
    raw = json.dumps(value, ensure_ascii=False).encode()
    if alteration == "duplicate":
        raw = b'{"schema":"shadow",' + raw[1:]
    elif alteration == "utf8":
        raw = b"\xff"
    source.write_bytes(raw)
    digest = _sha(raw)
    with pytest.raises(ValueError):
        delivery.read_nomination_source(source=source, expected_sha256=digest)
    root, run_id, paths = _prepared_remote(tmp_path, raw, digest)
    Path(paths["temporary"]).write_bytes(raw)
    result = _remote("publish", root, run_id, digest, len(raw))
    assert result.returncode != 0 and not Path(paths["artifact"]).exists()


def _remote(operation, root, run_id, digest, size, *, adapt=True):
    command = delivery.nomination_delivery_command(python=sys.executable, operation=operation,
        root=str(root), run_id=run_id, expected_sha256=digest, expected_bytes=size)
    assert command[3] == delivery.NOMINATION_DELIVERY_CODE
    if adapt:
        assert command[3].count("OWNER_UID = 0") == 1
        command[3] = command[3].replace("OWNER_UID = 0", "OWNER_UID = " + str(os.geteuid()), 1)
    return subprocess.run(command, capture_output=True, text=True, timeout=10,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def _prepared_remote(tmp_path, raw, digest):
    root = tmp_path / "trusted-root"
    root.mkdir(mode=0o755)
    run_id = "a" * 32
    paths = delivery.nomination_delivery_arguments(root=str(root), run_id=run_id,
        expected_sha256=digest, expected_bytes=len(raw))
    prepared = _remote("prepare", root, run_id, digest, len(raw))
    assert prepared.returncode == 0, prepared.stderr
    receipt = json.loads(prepared.stdout)
    assert receipt["published"] is False and receipt["operation"] == "prepare"
    assert not Path(paths["artifact"]).exists()
    return root, run_id, paths


def _publisher_functions():
    """Use literal publisher functions with only the authored host UID adapted."""
    code = delivery.NOMINATION_DELIVERY_CODE
    marker = "\ntry:\n    main()\nfinally:"
    assert code.count(marker) == 1 and code.count("OWNER_UID = 0") == 1
    definitions = code.split(marker, 1)[0].replace("OWNER_UID = 0", "OWNER_UID = " + str(os.geteuid()), 1)
    namespace = {"__name__": "authored_nomination_publisher_functions"}
    exec(compile(definitions, "<authored-nomination-publisher-functions>", "exec"), namespace)
    return namespace


def test_upload_custody_normalization_uses_checked_fd_and_preserves_readonly_mode(nomination, tmp_path, monkeypatch):
    _, raw, digest = nomination
    upload = tmp_path / ".upload.json"
    upload.write_bytes(raw)
    upload.chmod(0o444)
    original = upload.stat()
    namespace = _publisher_functions()
    actual_fchown, calls = os.fchown, []

    def fchown(descriptor, uid, gid):
        observed = os.fstat(descriptor)
        calls.append((observed.st_dev, observed.st_ino, uid, gid))
        return actual_fchown(descriptor, uid, gid)

    monkeypatch.setattr(os, "fchown", fchown)
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        observed, info = namespace["read_file"](parent, upload.name, digest, len(raw), normalize_upload=True)
    finally:
        os.close(parent)
    assert observed == raw
    assert calls == [(original.st_dev, original.st_ino, os.geteuid(), os.geteuid())]
    assert info.st_uid == info.st_gid == os.geteuid()
    assert stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1


@pytest.mark.parametrize("alteration", ["hardlink", "wrong_size", "fifo", "symlink"])
def test_invalid_upload_shape_refuses_before_fd_custody_change(nomination, tmp_path, monkeypatch, alteration):
    source, raw, digest = nomination
    upload = tmp_path / ".upload.json"
    if alteration == "fifo":
        os.mkfifo(upload)
    elif alteration == "symlink":
        upload.symlink_to(source)
    else:
        upload.write_bytes(raw + b" " if alteration == "wrong_size" else raw)
        if alteration == "hardlink":
            os.link(upload, tmp_path / "second-link.json")
    namespace = _publisher_functions()
    calls = []

    def fchown(*args):
        calls.append(args)
        raise AssertionError("invalid upload must not change custody")

    monkeypatch.setattr(os, "fchown", fchown)
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises((ValueError, OSError)):
            namespace["read_file"](parent, upload.name, digest, len(raw), normalize_upload=True)
    finally:
        os.close(parent)
    assert calls == []


def test_upload_path_replacement_never_redirects_fd_normalization(nomination, tmp_path, monkeypatch):
    _, raw, digest = nomination
    upload = tmp_path / ".upload.json"
    upload.write_bytes(raw)
    upload.chmod(0o444)
    original = upload.stat()
    retained = tmp_path / "retained-upload.json"
    namespace = _publisher_functions()
    actual_fchown, calls = os.fchown, []

    def fchown(descriptor, uid, gid):
        upload.rename(retained)
        upload.write_bytes(raw)
        upload.chmod(0o640)
        opened = os.fstat(descriptor)
        calls.append((opened.st_dev, opened.st_ino))
        return actual_fchown(descriptor, uid, gid)

    monkeypatch.setattr(os, "fchown", fchown)
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(ValueError, match="changed during read"):
            namespace["read_file"](parent, upload.name, digest, len(raw), normalize_upload=True)
    finally:
        os.close(parent)
    assert calls == [(original.st_dev, original.st_ino)]
    assert upload.stat().st_ino != original.st_ino
    assert stat.S_IMODE(upload.stat().st_mode) == 0o640 and upload.read_bytes() == raw
    assert retained.stat().st_ino == original.st_ino and retained.read_bytes() == raw


def test_actual_publisher_isolates_final_from_held_upload_write_descriptor(nomination, tmp_path):
    _, raw, digest = nomination
    root, run_id, paths = _prepared_remote(tmp_path, raw, digest)
    upload = Path(paths["temporary"])
    upload.write_bytes(raw)
    held = os.open(upload, os.O_WRONLY)
    before = os.fstat(held)
    try:
        result = _remote("publish", root, run_id, digest, len(raw))
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout)
        assert receipt == {"schema": delivery.DELIVERY_SCHEMA, "operation": "publish", "published": True,
            "artifact": paths["artifact"], "sha256": digest, "bytes": len(raw),
            "provider_called": False, "completion_authority": False}
        artifact = Path(paths["artifact"])
        assert artifact.stat().st_ino != before.st_ino
        os.write(held, b"forged old upload")
        assert artifact.read_bytes() == raw
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o444 and artifact.stat().st_nlink == 1
        assert not upload.exists() and not list(artifact.parent.glob(".nomination-pending-*"))
    finally:
        os.close(held)


def test_remote_run_collision_never_reuses_or_overwrites_directory(nomination, tmp_path):
    _, raw, digest = nomination
    root, run_id, paths = _prepared_remote(tmp_path, raw, digest)
    marker = Path(paths["directory"]) / "retained"
    marker.write_bytes(b"existing")
    result = _remote("prepare", root, run_id, digest, len(raw))
    assert result.returncode != 0 and marker.read_bytes() == b"existing"


@pytest.mark.parametrize("alteration", ["parent_private", "parent_writable", "upload_symlink", "upload_hardlink", "upload_fifo", "wrong_bytes", "existing_final"])
def test_remote_publisher_refuses_unsafe_or_conflicting_inputs(nomination, tmp_path, alteration):
    source, raw, digest = nomination
    root, run_id, paths = _prepared_remote(tmp_path, raw, digest)
    upload = Path(paths["temporary"])
    parent = Path(paths["directory"]).parent
    if alteration == "upload_symlink":
        upload.symlink_to(source)
    elif alteration == "upload_hardlink":
        os.link(source, upload)
    elif alteration == "upload_fifo":
        os.mkfifo(upload)
    else:
        upload.write_bytes(raw + b" " if alteration == "wrong_bytes" else raw)
    artifact = Path(paths["artifact"])
    if alteration == "parent_private":
        parent.chmod(0o700)
    elif alteration == "parent_writable":
        parent.chmod(0o777)
    elif alteration == "existing_final":
        artifact.write_bytes(b"retained conflicting final")
        artifact.chmod(0o444)
    result = _remote("publish", root, run_id, digest, len(raw))
    assert result.returncode != 0
    if alteration == "existing_final":
        assert artifact.read_bytes() == b"retained conflicting final"
    else:
        assert not artifact.exists()


def test_unmodified_remote_publisher_requires_real_root(nomination, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("nonroot refusal requires nonroot test runner")
    _, raw, digest = nomination
    root = tmp_path / "trusted-root"
    root.mkdir(mode=0o755)
    result = _remote("prepare", root, "a" * 32, digest, len(raw), adapt=False)
    assert result.returncode != 0 and not (root / "decision-nominations").exists()


def test_host_pending_cleanup_failure_still_closes_all_directory_descriptors(nomination, tmp_path, monkeypatch):
    source, _, digest = nomination
    directory = tmp_path / "trial"
    directory.mkdir(mode=0o755)
    captured = []
    original_init, original_unlink = delivery._Directories.__init__, delivery.os.unlink
    def initialize(self, path):
        original_init(self, path)
        captured.extend(self.fds)
    def link(*args, **kwargs):
        raise OSError("authored link failure")
    def unlink(path, *args, **kwargs):
        if str(path).startswith(".nomination-pending-"):
            raise OSError("authored cleanup failure")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(delivery._Directories, "__init__", initialize)
    monkeypatch.setattr(delivery.os, "link", link)
    monkeypatch.setattr(delivery.os, "unlink", unlink)
    with pytest.raises(OSError, match="cleanup failure"):
        delivery.capture_nomination_snapshot(source=source, expected_sha256=digest, directory=directory)
    assert captured
    for descriptor in captured:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_published_nomination_reaches_real_owner_binding_and_existing_worker_loader(allocated, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_owner as owner
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_worker import load_decision_launch_policy
    repository, workspace, prompt, _ = allocated
    repository.chmod(0o755)
    (repository / ".runtime").chmod(0o755)
    parent = json.loads((repository / ".runtime/semantic/worker-context.json").read_text())
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids[1:], optional_capsule_ids=ids[:1])
    raw = (json.dumps(owner.build_decision_nomination(policy), ensure_ascii=False, indent=2) + "\n").encode()
    digest = _sha(raw)
    root, run_id, paths = _prepared_remote(tmp_path, raw, digest)
    Path(paths["temporary"]).write_bytes(raw)
    result = _remote("publish", root, run_id, digest, len(raw))
    assert result.returncode == 0, result.stderr
    task = owner.codec._prompt_parts(prompt)[0]["objective_id"]
    dispatch = owner.prepare_decision_owner_dispatch(arguments=[
        "--model", "pinned", "--semantic-repository", str(repository),
        "--semantic-decision-nomination-artifact", paths["artifact"],
        "--semantic-decision-nomination-sha256", digest, "--semantic-decision-task-id", task],
        prompt_bytes=prompt.encode(), workspace=workspace)
    assert dispatch.prompt_bytes == prompt.encode()
    assert dispatch.receipt["nomination_sha256"] == digest
    assert load_decision_launch_policy(artifact=Path(dispatch.receipt["launch_policy_artifact"]),
        expected_sha256=dispatch.receipt["launch_policy_sha256"], native_prompt=prompt,
        repository=repository, workspace=workspace) == policy
