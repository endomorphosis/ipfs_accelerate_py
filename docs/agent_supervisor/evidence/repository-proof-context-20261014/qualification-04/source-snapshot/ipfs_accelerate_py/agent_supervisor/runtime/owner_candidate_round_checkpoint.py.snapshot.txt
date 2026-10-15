"""Retain fresh advisory candidate artifacts after a failed isolated worker.

Only the owner calls this after waiting, worker cleanup and broker closure.
Nothing in this checkpoint admits another worker, publishes proofs or changes
the first worker's outcome. Native prompt bytes are never written here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys
import time

from .worker_source_views import canonical_json

SCHEMA = "supervisor-owner-candidate-checkpoint@1"
OBSERVATION_SCHEMA = "owner-candidate-refresh-observation@1"
PREFIX = "[owner-candidate-refresh] "
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
FLAGS = dict(proof_authority=False, execution_authority=False,
    completion_authority=False, retry_authority=False, settlement_authority=False,
    automatic_retry_admitted=False, model_inference_calls=0, provider_calls=0,
    source_obligations_discharged=0)


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _remaining(deadline):
    _need(type(deadline) in (int, float)
        and -sys.float_info.max <= deadline <= sys.float_info.max,
        "finite owner candidate deadline required")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("owner candidate refresh budget exhausted")
    return remaining


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _write(path, raw, created):
    """Publish only in a protected owner directory; refuse every overwrite."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    info = os.fstat(descriptor)
    created[path] = info.st_dev, info.st_ino
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fchmod(stream.fileno(), 0o400)
        os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def persist_failed_worker_candidate_context(*, source_broker, prior_returncode, absolute_deadline):
    """Build and retain one owner-private candidate checkpoint per source session.

    The exact source broker and its waited child are verified by the snapshot
    factory. Worker descendant cleanup is an independent caller prerequisite;
    these advisory records cannot stand in for that custody evidence.
    """
    from . import source_formalization_context as guard
    from .owner_candidate_round_context import prepare_candidate_round_context

    started = time.monotonic()
    _remaining(absolute_deadline)
    snapshot = prepare_candidate_round_context(source_broker=source_broker,
        prior_returncode=prior_returncode, round_number=2, absolute_deadline=absolute_deadline)
    metadata = snapshot.revalidate()
    session = metadata["prior_session_id"]
    _need(type(session) is str and re.fullmatch(r"[0-9a-f]{32}", session) is not None,
        "exact prior source session required")
    artifacts = snapshot.artifacts
    _need(set(artifacts) == {"summary", "capsules", "formal"}
        and all(type(raw) is bytes and raw for raw in artifacts.values())
        and sum(map(len, artifacts.values())) <= MAX_ARTIFACT_BYTES,
        "bounded native candidate artifacts required")
    boundary_value, boundary_sha = guard._owner_boundary(source_broker.boundary)
    _need(boundary_sha == source_broker.descriptor["boundary_manifest_sha256"]
        and boundary_value["owner_uid"] == source_broker.owner_uid == os.getuid()
        and os.geteuid() == os.getuid() and os.getgid() == source_broker.owner_uid,
        "candidate checkpoint owner boundary changed")
    parent = Path(source_broker.boundary).absolute().parent / "worker-observations"
    guard._protected_directory(parent, owner=source_broker.owner_uid, group=source_broker.owner_uid)
    parent_identity = _identity(parent)
    directory = parent / ("candidate-round-" + session)
    manifest_path = parent / (directory.name + ".json")
    created = {}
    directory_identity = None
    try:
        _remaining(absolute_deadline)
        directory.mkdir(mode=0o700)
        directory_identity = _identity(directory)
        bindings = {}
        for name in sorted(artifacts):
            _remaining(absolute_deadline)
            raw = artifacts[name]
            path = directory / (name + ".json")
            _write(path, raw, created)
            bindings[name] = dict(path=path.relative_to(parent).as_posix(), sha256=_sha(raw), bytes=len(raw))
        _sync_directory(directory)
        _need(canonical_json(snapshot.revalidate()) == canonical_json(metadata),
            "candidate context changed before checkpoint publication")
        _remaining(absolute_deadline)
        observation = dict(schema=OBSERVATION_SCHEMA, status="prepared", reason_code="candidate_context_prepared",
            **{key: metadata[key] for key in ("prior_session_id", "descriptor_sha256", "task_id",
                "native_prompt_sha256", "workspace_generation")},
            artifacts_count=len(bindings), artifact_bytes=sum(row["bytes"] for row in bindings.values()),
            elapsed_seconds=max(0., time.monotonic() - started), **FLAGS)
        manifest = dict(schema=SCHEMA, context=metadata, artifacts=bindings, observation=observation, **FLAGS)
        body = canonical_json(manifest)
        _need(len(body) <= MAX_MANIFEST_BYTES, "candidate checkpoint manifest exceeds bound")
        _need(_identity(parent) == parent_identity, "candidate checkpoint directory changed")
        _write(manifest_path, body, created)
        _sync_directory(parent)
        _need(canonical_json(snapshot.revalidate()) == canonical_json(metadata),
            "candidate context changed during checkpoint publication")
        _remaining(absolute_deadline)
        return {**observation, "checkpoint_manifest": dict(path=manifest_path.name,
            sha256=_sha(body), bytes=len(body))}
    except BaseException:
        # Remove only files this call created and still owns. A replay, a
        # pre-existing checkpoint or an unexpected replacement is left alone.
        for path, identity in reversed(tuple(created.items())):
            try:
                if _identity(path) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        if directory_identity is not None:
            try:
                if _identity(directory) == directory_identity:
                    directory.rmdir()
            except (FileNotFoundError, OSError):
                pass
        raise
