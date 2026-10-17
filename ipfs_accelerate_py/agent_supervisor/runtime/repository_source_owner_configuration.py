"""Optional deployer selection of an existing native source-service scheduler.

Validation is pure and can precede installer I/O. Loading verifies the existing
owner boundary and filesystem protection, without opening scheduler contents,
creating state, selecting defaults, or acquiring native capacity. The native
existing-scheduler facade must still check the nominated policy at admission.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat


SCHEMA = "repository-source-owner-resources@1"
BOUNDARY_KEY = "repository_source_resources"
RESOURCE_SCOPE = "repository_source_service_only"
MAX_BOUNDARY_BYTES = 32768
_FIELDS = frozenset({"schema", "scheduler_state_path", "scheduler_policy_sha256",
                     "admission_timeout_seconds", "resource_scope"})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _path(value):
    _need(type(value) is str and bool(value), "exact absolute source-owner path required")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("exact absolute source-owner path required") from error
    path = PurePosixPath(value)
    _need(size <= 4096 and path.is_absolute() and value.startswith("/")
          and not value.startswith("//") and str(path) == value and value != "/"
          and ".." not in path.parts and not any(ord(c) < 32 or ord(c) == 127 or c == "\\" for c in value),
          "canonical absolute source-owner path required")
    return path


def validate_source_owner_configuration(value, *, owner_private_paths):
    """Return a detached closed configuration, or None; perform no filesystem I/O."""
    if value is None:
        return None
    _need(type(value) is dict and set(value) == _FIELDS,
          "closed source-owner resource configuration required")
    _need(value["schema"] == SCHEMA and type(value["schema"]) is str
          and value["resource_scope"] == RESOURCE_SCOPE and type(value["resource_scope"]) is str,
          "unsupported source-owner resource configuration")
    _need(type(owner_private_paths) is list and 1 <= len(owner_private_paths) <= 16,
          "bounded owner-private directory nominations required")
    private = tuple(_path(item) for item in owner_private_paths)
    _need(len(set(private)) == len(private), "duplicate owner-private directory nomination")
    state = _path(value["scheduler_state_path"])
    _need(any(state != root and state.is_relative_to(root) for root in private),
          "source scheduler must be strictly beneath an owner-private directory")
    digest = value["scheduler_policy_sha256"]
    _need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
          "exact source scheduler policy SHA-256 required")
    timeout = value["admission_timeout_seconds"]
    _need(type(timeout) in (int, float) and 0 < timeout <= 30,
          "bounded positive source admission timeout required")
    _need(math.isfinite(timeout), "finite source admission timeout required")
    return {"schema": SCHEMA, "scheduler_state_path": str(state),
            "scheduler_policy_sha256": digest, "admission_timeout_seconds": timeout,
            "resource_scope": RESOURCE_SCOPE}


def _owner_boundary(path):
    # This is the actual deployed owner identity/namespace verifier. Keep the
    # import out of the pure builder so installers need no runtime services.
    from . import source_formalization_context as guard
    return guard._owner_boundary(path)


def _read(path):
    from . import source_formalization_context as guard
    return guard._read(path, MAX_BOUNDARY_BYTES)


def _stat(path):
    return path.lstat()


def _identity(info, *, contents=False):
    result = (info.st_dev, info.st_ino, info.st_uid, info.st_gid,
              info.st_mode, info.st_nlink)
    if contents:
        result += (info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return result


def _deployer_chain(path):
    rows = []
    for candidate in reversed((path, *path.parents)):
        info = _stat(candidate)
        _need(info.st_uid == 0 and not info.st_mode & 0o022,
              "source boundary ancestry must be protected by the root deployer")
        if candidate == path:
            _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                  and 0 < info.st_size <= MAX_BOUNDARY_BYTES,
                  "source boundary must be a bounded unaliased regular file")
        else:
            _need(stat.S_ISDIR(info.st_mode), "source boundary ancestor must be an exact directory")
        rows.append((str(candidate), _identity(info, contents=candidate == path)))
    return tuple(rows)


def _private_chain(configuration, private_paths, *, owner):
    state = Path(configuration["scheduler_state_path"])
    roots = [Path(value) for value in private_paths
             if state != Path(value) and state.is_relative_to(Path(value))]
    # Every nominated enclosing directory must satisfy its promised privacy.
    # A broader unprotected nomination cannot be hidden by a nested one.
    rows = []
    for candidate in reversed(state.parents):
        info = _stat(candidate)
        _need(stat.S_ISDIR(info.st_mode) and info.st_uid in (0, owner)
              and not info.st_mode & 0o022,
              "source scheduler ancestors must be exact worker-unwritable owner directories")
        if candidate in roots:
            _need(info.st_uid == owner and stat.S_IMODE(info.st_mode) == 0o700,
                  "source scheduler requires an exact owner-private 0700 directory")
        rows.append((str(candidate), _identity(info)))
    info = _stat(state)
    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == owner,
          "source scheduler must already be an unaliased owner regular file")
    # Metadata only: the existing native facade owns content, locking, policy,
    # atomic replacement and admission. An in-flight native replacement need
    # not preserve this file's inode across a configuration read.
    return tuple(rows)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _need(key not in value, "duplicate source boundary JSON field")
        value[key] = item
    return value


def _nonfinite(_):
    raise ValueError("nonfinite source boundary JSON field")


def load_source_owner_configuration(boundary):
    """Load the optional selection from a protected, current owner boundary.

    Missing configuration returns None. A selected missing scheduler fails
    closed; no directory, policy, lock, state, or default allocator is created.
    The returned dictionary is configuration evidence, not an admission grant.
    """
    _need(isinstance(boundary, (str, Path)), "exact source owner boundary path required")
    path = Path(_path(str(boundary)))
    try:
        before = _deployer_chain(path)
        raw = _read(path)
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)
        _need(type(value) is dict, "source owner boundary must be an object")
        trusted, digest = _owner_boundary(path)
        _need(type(trusted) is dict and trusted == value
              and type(digest) is str and hashlib.sha256(raw).hexdigest() == digest,
              "source boundary changed during owner validation")
        _need(_deployer_chain(path) == before, "source boundary identity changed during validation")
        result = validate_source_owner_configuration(value.get(BOUNDARY_KEY),
                                                     owner_private_paths=value.get("owner_private_paths"))
        if result is None:
            return None
        owner, worker = value.get("owner_uid"), value.get("worker_uid")
        _need(type(owner) is int and owner > 0 and type(worker) is int and worker > 0
              and owner != worker and os.getuid() == owner and os.geteuid() == owner,
              "source configuration requires the deployed nonroot owner identity")
        private_before = _private_chain(result, value["owner_private_paths"], owner=owner)
        _need(_deployer_chain(path) == before
              and _private_chain(result, value["owner_private_paths"], owner=owner) == private_before,
              "source owner resource ancestry changed during validation")
        return result
    except (OSError, UnicodeError, RecursionError) as error:
        raise ValueError("source owner configuration is unavailable or unstable") from error
