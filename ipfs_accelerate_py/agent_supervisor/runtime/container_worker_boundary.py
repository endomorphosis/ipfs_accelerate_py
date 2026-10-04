"""Verify an explicitly deployed Docker worker boundary before dispatch.

The trusted container deployer owns the manifest. A separate Unix identity
keeps the model worker away from the supervisor owner's private authorities.
The receipt alone never authorizes access outside the container.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat


def verify_container_worker_boundary(*, artifact: Path, expected_sha256: str,
                                     workspace: Path, purpose: str = "coding") -> dict:
    if purpose not in {"coding", "validation"}:
        raise ValueError("unknown container worker purpose")
    path = Path(artifact)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("container boundary must be an exact absolute path")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("container boundary digest required")
    for parent in (path, *path.parents):
        info = parent.stat()
        if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("container boundary must be owned and protected by the deployer")
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > 16_384:
        raise ValueError("container boundary is not a bounded regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("container boundary digest mismatch")
    boundary = json.loads(raw)
    if boundary.get("schema") != "supervisor-container-worker-boundary@1":
        raise ValueError("unsupported container worker boundary")
    if (not re.fullmatch(r"[0-9a-f]{64}", str(boundary.get("container_id", "")))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(boundary.get("image_id", "")))):
        raise ValueError("container boundary lacks exact container and image identities")
    owner, worker = boundary.get("owner_uid"), boundary.get("worker_uid")
    if (type(owner) is not int or type(worker) is not int or owner <= 0
            or worker <= 0 or worker == owner or os.geteuid() != worker or os.getuid() != worker):
        raise ValueError("model worker must have its separate nonroot identity")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if any(int(status[key].strip(), 16) for key in ("CapEff", "CapPrm", "CapAmb")):
        raise ValueError("model worker must not retain privileged capabilities")
    if status.get("NoNewPrivs", "").strip() != "1":
        raise ValueError("model worker must not acquire new privileges")
    observed = {key: os.readlink(f"/proc/self/ns/{key}") for key in ("pid", "mnt", "net")}
    if boundary.get("namespaces") != observed:
        raise ValueError("container namespace differs from trusted deployment")
    roots = boundary.get("allowed_worktree_roots")
    private = boundary.get("owner_private_paths")
    validation_roots = boundary.get("validation_repository_roots", []) if purpose == "validation" else []
    if (not isinstance(roots, list) or not 1 <= len(roots) <= 8
            or not isinstance(private, list) or not 1 <= len(private) <= 16
            or not isinstance(validation_roots, list) or len(validation_roots) > 8):
        raise ValueError("container boundary requires bounded worktree and private paths")
    for value in (*roots, *private, *validation_roots):
        candidate = Path(value)
        if not candidate.is_absolute() or ".." in candidate.parts or str(candidate) == "/":
            raise ValueError("container boundary path is not scoped")
        if candidate.is_symlink() or candidate.resolve() != candidate:
            raise ValueError("container boundary path must be exact without symlinks")
        for parent in candidate.parents:
            info = parent.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, owner)
                    or os.access(parent, os.W_OK)):
                raise ValueError("container authority ancestors must be protected from the worker")
    for value in (*roots, *validation_roots):
        info = Path(value).lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, owner)
                or os.access(value, os.W_OK)):
            raise ValueError("allocated worktree parent must be controlled by the supervisor")
    cwd = workspace.resolve()
    allocated = any(cwd != Path(root).resolve() and cwd.is_relative_to(Path(root).resolve()) for root in roots)
    exact_validation = purpose == "validation" and str(cwd) in validation_roots
    if not (allocated or exact_validation):
        raise ValueError("working tree is outside the deployed worker scope")
    for value in private:
        candidate = Path(value)
        # Check an observable protected directory/file itself, not a guessed
        # inaccessible descendant. Missing or unverifiable paths fail closed.
        info = candidate.lstat()
        if (candidate.is_symlink() or candidate.resolve() != candidate
                or info.st_uid != owner or info.st_mode & 0o077
                or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
                or cwd.is_relative_to(candidate)):
            raise ValueError("supervisor private authority is not an exact protected owner path")
        if any(os.access(value, mode) for mode in (os.R_OK, os.W_OK, os.X_OK)):
            raise ValueError("model worker can access supervisor private authority")
    return {"schema": boundary["schema"], "container_id": boundary["container_id"],
            "image_id": boundary["image_id"], "namespaces": observed,
            "owner_uid": owner, "worker_uid": worker,
            "purpose": purpose, "manifest_sha256": expected_sha256, "completion_authority": False}
