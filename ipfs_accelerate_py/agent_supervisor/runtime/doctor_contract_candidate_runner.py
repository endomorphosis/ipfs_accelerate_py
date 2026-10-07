"""Materialize a scoped contract candidate in an allocated native worktree.

The owner pins the immutable candidate artifact. The attached proof receipt is
an operator claim, not a publication or whole-program correctness authority.
Ordinary supervisor validation, proposal, publication and completion gates must
still run. This runner neither creates commits nor advances refs.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import uuid

from ..proof.formal_verification_contracts import content_identity
from .doctor_candidate_runner import MAX_BYTES, _directory, _git, _read, _sha, _unique

SCHEMA = "supervisor-doctor-contract-candidate@1"
FIELDS = frozenset({
    "schema", "repository", "baseline_commit", "task_cid", "task_id", "task_revision",
    "manifest_cid", "proof_receipt_id", "proof_scope", "analysis_cid", "edits",
    "permitted_outputs", "provider_calls", "publication_authority", "completion_authority",
    "artifact_cid",
})
_EDIT_FIELDS = {"path", "effect", "before_sha256", "after_sha256", "after_bytes_base64"}
_META = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")


def publish_doctor_contract_candidate(repository: Path, payload: dict):
    """Expose exact candidate bytes at a pinned, owner-controlled locator.

    Evidence referenced by the artifact remains a scoped operator claim. In
    schema @1 the legacy ``proof_receipt_id`` may identify a finite native
    check; ``proof_scope`` must describe its actual evidence, not claim a
    kernel proof. The ordinary validation and publication gates still apply.
    """
    current = repository
    for name in (".runtime", "doctor-contract-candidates"):
        current = current / name
        current.mkdir(mode=0o755, exist_ok=True)
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022 or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError("candidate artifact directory is not owner-controlled and worker-readable")
    payload = {**payload, "artifact_cid": content_identity(payload)}
    raw = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    if len(raw) > MAX_BYTES:
        raise ValueError("contract candidate artifact exceeds byte bound")
    digest = _sha(raw)
    artifact = current / (digest + ".json")
    with artifact.open("xb") as stream:
        stream.write(raw)
        os.fchmod(stream.fileno(), 0o444)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return artifact, digest, payload["artifact_cid"]


def _path(value):
    if not isinstance(value, str) or not value or len(value.encode()) > 4096 or "\0" in value:
        raise ValueError("bounded relative candidate output path required")
    relative = PurePosixPath(value)
    if (relative.is_absolute() or relative.as_posix() != value or not relative.parts
            or any(part in {".", "..", ".git", ".runtime"} for part in relative.parts)):
        raise ValueError("candidate output escapes permitted source scope")
    return relative


def _absent(parent, name):
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ValueError("declared create output already exists")


def _write_bytes(fd, raw):
    offset = 0
    while offset < len(raw):
        count = os.write(fd, raw[offset:])
        if count <= 0:
            raise OSError("candidate write made no progress")
        offset += count
    os.ftruncate(fd, len(raw))
    os.fsync(fd)


def _check_parent(root, relative, parent):
    """A retained descriptor must still name this allocated output directory."""
    observed = _directory(root / relative.parent)
    try:
        expected, current = os.fstat(parent), os.fstat(observed)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("candidate output directory changed during materialization")
    finally:
        os.close(observed)


def _write(parent, relative, after, info):
    temporary = ".doctor-contract-" + uuid.uuid4().hex
    try:
        if info is None:
            fd = os.open(relative.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o644, dir_fd=parent)
            try:
                _write_bytes(fd, after)
            finally:
                os.close(fd)
            mode = "exclusive_create"
        elif info.st_uid != os.geteuid():
            # Native sticky allocation directories permit editing the owner's
            # delegated inode but deliberately forbid replacing that inode.
            fd = os.open(relative.name, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                current = os.fstat(fd)
                if any(getattr(current, key) != getattr(info, key) for key in _META):
                    raise ValueError("candidate changed before delegated write")
                _write_bytes(fd, after)
            finally:
                os.close(fd)
            mode = "preserved_owner_inode"
        else:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            try:
                os.fchmod(fd, stat.S_IMODE(info.st_mode))
                _write_bytes(fd, after)
            finally:
                os.close(fd)
            current = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
            if any(getattr(current, key) != getattr(info, key) for key in _META):
                raise ValueError("candidate changed before replacement")
            os.replace(temporary, relative.name, src_dir_fd=parent, dst_dir_fd=parent)
            mode = "atomic_replace"
        os.fsync(parent)
        actual, written = _read(parent, relative.name)
        if actual != after or (mode == "preserved_owner_inode"
                and (written.st_dev, written.st_ino) != (info.st_dev, info.st_ino)):
            raise ValueError("candidate bytes differ after materialization")
        return mode
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass


def materialize_doctor_contract_candidate(*, artifact: Path, expected_sha256: str,
        task_cid: str, prompt: str, workspace: Path) -> dict:
    """Check every preimage before writing any of at most eight outputs.

    Interrupted writes may leave a failed isolated candidate; they never emit a
    success receipt or gain publication authority. Owner-inode writes retain the
    deployed worker permission boundary instead of replacing owner-owned files.
    """
    artifact = Path(artifact).absolute()
    parent = _directory(artifact.parent)
    try:
        raw, metadata = _read(parent, artifact.name)
    finally:
        os.close(parent)
    if stat.S_IMODE(metadata.st_mode) & 0o222 or _sha(raw) != expected_sha256:
        raise ValueError("immutable pinned contract candidate required")
    candidate = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(candidate, dict) or set(candidate) != FIELDS or candidate["schema"] != SCHEMA:
        raise ValueError("closed contract candidate schema required")
    if (content_identity({key: value for key, value in candidate.items() if key != "artifact_cid"})
            != candidate["artifact_cid"] or candidate["task_cid"] != task_cid
            or type(candidate["task_revision"]) is not int or candidate["task_revision"] < 1
            or candidate["publication_authority"] is not False or candidate["completion_authority"] is not False
            or type(candidate["provider_calls"]) is not int or candidate["provider_calls"] != 0
            or any(not isinstance(candidate[key], str) or not candidate[key]
                   or len(candidate[key].encode()) > 8192 for key in (
                       "task_cid", "task_id", "manifest_cid", "proof_receipt_id", "proof_scope", "analysis_cid"))):
        raise ValueError("contract candidate identity or authority differs")
    if not isinstance(prompt, str) or len(prompt.encode()) > 256_000:
        raise ValueError("native prompt exceeds bound")
    wire, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    if not isinstance(wire, dict) or wire.get("objective_id") != candidate["task_id"]:
        raise ValueError("native task differs from contract candidate")
    root, canonical = Path(workspace).absolute(), Path(candidate["repository"])
    if (root.resolve(strict=True) != root or not canonical.is_absolute()
            or canonical.resolve(strict=True) != canonical or root == canonical
            or artifact.is_relative_to(root)):
        raise ValueError("a separate native allocated worktree and owner artifact are required")
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    common = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if top != root or common != _git(canonical, "rev-parse", "--path-format=absolute", "--git-common-dir"):
        raise ValueError("contract candidate belongs to a foreign repository")
    baseline = candidate["baseline_commit"]
    if (not isinstance(baseline, str) or re.fullmatch(r"[0-9a-f]{40}", baseline) is None
            or _git(root, "rev-parse", "HEAD").decode().strip() != baseline):
        raise ValueError("contract candidate baseline drifted")
    permissions = candidate["permitted_outputs"]
    if not isinstance(permissions, list) or not 1 <= len(permissions) <= 8:
        raise ValueError("bounded permitted output population required")
    permitted = {}
    for item in permissions:
        if (not isinstance(item, dict) or set(item) != {"path", "effect", "media_type"}
                or item["effect"] not in {"modify", "create"}
                or not isinstance(item["media_type"], str) or not item["media_type"]
                or len(item["media_type"]) > 256):
            raise ValueError("closed output permission required")
        _path(item["path"])
        if item["path"] in permitted:
            raise ValueError("duplicate output permission")
        permitted[item["path"]] = item["effect"]
    edits = candidate["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
        raise ValueError("bounded candidate edit population required")
    checked, seen, total = [], set(), 0
    try:
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != _EDIT_FIELDS:
                raise ValueError("closed candidate edit required")
            relative = _path(edit["path"])
            if edit["path"] in seen or permitted.get(edit["path"]) != edit["effect"]:
                raise ValueError("duplicate or undeclared candidate output")
            seen.add(edit["path"])
            after = base64.b64decode(edit["after_bytes_base64"], validate=True)
            total += len(after)
            if total > MAX_BYTES or _sha(after) != edit["after_sha256"]:
                raise ValueError("contract candidate bytes exceed bounds or digest differs")
            parent = _directory(root / relative.parent)
            # Retain all descriptors until every source and absence check passes.
            checked.append((parent, relative, after, None, edit))
            if edit["effect"] == "create":
                if edit["before_sha256"] is not None or _git(root, "ls-tree", "-z", baseline, "--", str(relative)):
                    raise ValueError("declared create output exists in baseline")
                _absent(parent, relative.name)
            else:
                original = _git(root, "show", baseline + ":" + str(relative))
                if _sha(original) != edit["before_sha256"]:
                    raise ValueError("contract candidate preimage differs from baseline")
                current, info = _read(parent, relative.name)
                if current != original:
                    raise ValueError("allocated candidate preimage drifted")
                checked[-1] = (parent, relative, after, info, edit)
        writes = []
        for parent, relative, after, info, edit in checked:
            _check_parent(root, relative, parent)
            mode = _write(parent, relative, after, info)
            writes.append({"path": str(relative), "effect": edit["effect"],
                           "after_sha256": edit["after_sha256"], "write_mode": mode})
        # A late concurrent edit to an earlier output must also deny success.
        for parent, relative, after, _, _ in checked:
            _check_parent(root, relative, parent)
            if _read(parent, relative.name)[0] != after:
                raise ValueError("candidate changed before final materialization receipt")
        if _git(root, "rev-parse", "HEAD").decode().strip() != baseline:
            raise ValueError("candidate baseline changed during materialization")
    finally:
        for parent, *_ in checked:
            os.close(parent)
    return {"schema": "native-doctor-contract-candidate-materialization@1",
        "status": "candidate_materialized", "artifact_cid": candidate["artifact_cid"],
        "task_cid": task_cid, "analysis_cid": candidate["analysis_cid"],
        "proof_receipt_id": candidate["proof_receipt_id"], "proof_scope": candidate["proof_scope"],
        "baseline_commit": baseline, "writes": writes, "provider_calls": 0,
        "publication_authority": False, "completion_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--task-cid", required=True)
    args = parser.parse_args()
    try:
        result = materialize_doctor_contract_candidate(artifact=args.artifact,
            expected_sha256=args.sha256, task_cid=args.task_cid,
            prompt=sys.stdin.buffer.read(256_001).decode(), workspace=Path.cwd())
    except Exception as error:
        print(json.dumps({"schema": "native-doctor-contract-candidate-materialization@1",
            "status": "refused", "error_type": type(error).__name__,
            "completion_authority": False}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
