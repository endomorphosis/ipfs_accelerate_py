"""Materialize a verified Doctor candidate in a native allocated worktree.

The owner pins the artifact digest in its admitted implementation command.
This worker never publishes a ref or completes a task. Native proposal scope,
validation, merge and owner completion gates still run after this command.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import uuid

from ..proof.formal_verification_contracts import content_identity

SCHEMA = "supervisor-doctor-candidate-handoff@1"
MAX_BYTES = 2_000_000
FIELDS = frozenset({
    "schema", "operator", "task_cid", "task_id", "task_revision", "manifest_cid",
    "repository", "base_commit", "candidate_commit", "candidate_ref", "permitted_outputs",
    "transaction_id", "transaction_receipt_id", "proof_receipt_id", "proof_scope", "edits",
    "provider_calls", "publication_authority", "completion_authority", "handoff_cid",
})


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(root, *args):
    from .candidate_execution import GIT_OWNER_ENV
    result = subprocess.run(["/usr/bin/git", "--no-replace-objects", "-c", "safe.directory=" + str(root), "-C", str(root), *args],
        env={"PATH": "/usr/bin:/bin", **GIT_OWNER_ENV}, capture_output=True, timeout=10, check=True)
    return result.stdout


def _directory(path):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("parent traversal refused")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(parent, name):
    fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_BYTES:
            raise ValueError("bounded single-link regular file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(raw) > MAX_BYTES or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError("candidate file changed during read")
        return raw, before
    finally:
        os.close(fd)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate handoff key")
        result[key] = value
    return result


def materialize_doctor_candidate(*, artifact: Path, expected_sha256: str,
                                 task_cid: str, prompt: str, workspace: Path) -> dict:
    artifact = Path(artifact).absolute()
    parent = _directory(artifact.parent)
    try:
        raw, _ = _read(parent, artifact.name)
    finally:
        os.close(parent)
    if _sha(raw) != expected_sha256:
        raise ValueError("Doctor handoff digest differs")
    handoff = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(handoff, dict) or set(handoff) != FIELDS or handoff["schema"] != SCHEMA:
        raise ValueError("closed Doctor handoff required")
    payload = {key: value for key, value in handoff.items() if key != "handoff_cid"}
    if (content_identity(payload) != handoff["handoff_cid"] or handoff["task_cid"] != task_cid
            or handoff["publication_authority"] is not False or handoff["completion_authority"] is not False
            or type(handoff["provider_calls"]) is not int or handoff["provider_calls"] != 0
            or type(handoff["task_revision"]) is not int or handoff["task_revision"] < 1
            or any(not isinstance(handoff[key], str) or not handoff[key] for key in
                   ("task_id", "transaction_id", "transaction_receipt_id", "proof_receipt_id"))):
        raise ValueError("Doctor handoff identity or authority differs")
    if len(prompt.encode()) > 256_000:
        raise ValueError("native prompt exceeds bound")
    wire, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    if not isinstance(wire, dict) or wire.get("objective_id") != handoff["task_id"]:
        raise ValueError("native task differs from Doctor candidate")
    root = Path(workspace).absolute()
    canonical = Path(handoff["repository"])
    if root.resolve(strict=True) != root or not canonical.is_absolute() or root == canonical:
        raise ValueError("a separate native allocated worktree is required")
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    common = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if top != root or common != _git(canonical, "rev-parse", "--path-format=absolute", "--git-common-dir"):
        raise ValueError("candidate worktree belongs to a foreign repository")
    base, candidate = handoff["base_commit"], handoff["candidate_commit"]
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None for value in (base, candidate)):
        raise ValueError("exact Doctor commit identities required")
    if _git(root, "rev-parse", "HEAD").decode().strip() != base:
        raise ValueError("candidate baseline drifted")
    edits = handoff["edits"]
    if not isinstance(edits, list) or len(edits) != 1:
        raise ValueError("closed Doctor operator requires exactly one edit")
    edit = edits[0]
    if not isinstance(edit, dict) or set(edit) != {"path", "before_sha256", "after_sha256", "after_bytes_base64"}:
        raise ValueError("closed Doctor edit required")
    relative = PurePosixPath(edit["path"])
    if (relative.is_absolute() or relative.as_posix() != edit["path"]
            or not relative.parts or any(part in {".", "..", ".git"} for part in relative.parts)
            or edit["path"] not in handoff["permitted_outputs"]):
        raise ValueError("Doctor edit escapes declared output scope")
    changed = _git(root, "diff", "--name-only", "-z", base, candidate).decode().split("\0")
    if [path for path in changed if path] != [edit["path"]]:
        raise ValueError("Doctor candidate has undeclared changes")
    after = base64.b64decode(edit["after_bytes_base64"], validate=True)
    if (len(after) > MAX_BYTES or _sha(after) != edit["after_sha256"]
            or _git(root, "show", candidate + ":" + edit["path"]) != after):
        raise ValueError("Doctor candidate bytes differ")
    original = _git(root, "show", base + ":" + edit["path"])
    if _sha(original) != edit["before_sha256"]:
        raise ValueError("Doctor preimage differs from baseline")
    parent = _directory(root / relative.parent)
    temporary = ".doctor-candidate-" + uuid.uuid4().hex
    try:
        current, info = _read(parent, relative.name)
        if current != original:
            raise ValueError("allocated candidate preimage drifted")
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        if info.st_uid != os.geteuid():
            # The deployed worker may edit an owner-allocated inode, but the
            # sticky allocation directory deliberately forbids replacing it
            # (and protects the read-only .git marker). Preserve that boundary.
            # Interrupted writes remain unvalidated candidates; no publication
            # or completion receipt is emitted until the exact bytes verify.
            fd = os.open(relative.name, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                now = os.fstat(fd)
                if any(getattr(now, key) != getattr(info, key) for key in fields):
                    raise ValueError("allocated candidate changed before write")
                offset = 0
                while offset < len(after):
                    count = os.write(fd, after[offset:])
                    if count <= 0:
                        raise OSError("candidate write made no progress")
                    offset += count
                os.ftruncate(fd, len(after))
                os.fsync(fd)
            finally:
                os.close(fd)
            write_mode = "preserved_owner_inode"
        else:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            with os.fdopen(fd, "wb") as stream:
                stream.write(after)
                stream.flush()
                os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode))
                os.fsync(stream.fileno())
            now = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
            if any(getattr(now, key) != getattr(info, key) for key in fields):
                raise ValueError("allocated candidate changed before write")
            os.replace(temporary, relative.name, src_dir_fd=parent, dst_dir_fd=parent)
            write_mode = "atomic_replace"
        os.fsync(parent)
        actual, written = _read(parent, relative.name)
        if actual != after or (write_mode == "preserved_owner_inode"
                and (written.st_dev, written.st_ino) != (info.st_dev, info.st_ino)):
            raise ValueError("allocated candidate differs after write")
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)
    return {"schema": "native-doctor-candidate-materialization@1", "status": "candidate_materialized",
        "handoff_cid": handoff["handoff_cid"], "task_cid": task_cid,
        "candidate_commit": candidate, "changed_paths": [edit["path"]],
        "source_after_sha256": edit["after_sha256"], "write_mode": write_mode, "provider_calls": 0,
        "publication_authority": False, "completion_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--task-cid", required=True)
    args = parser.parse_args()
    try:
        result = materialize_doctor_candidate(artifact=args.artifact, expected_sha256=args.sha256,
            task_cid=args.task_cid, prompt=sys.stdin.buffer.read(256_001).decode(), workspace=Path.cwd())
    except Exception as error:
        print(json.dumps({"schema": "native-doctor-candidate-materialization@1", "status": "refused",
            "error_type": type(error).__name__, "completion_authority": False}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
