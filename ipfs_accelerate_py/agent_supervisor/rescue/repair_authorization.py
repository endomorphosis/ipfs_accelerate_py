"""Explicit finite repair epochs; locating a job never creates new authority.

The optional object lives on the configured board. This also lets the original
watchdog's cancellation hook resolve the same job without a second fleet scan.
The legacy queue remains the default and is never migrated or reset here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

SCHEMA = "agent-supervisor/finite-repair-authorization@1"
FIELDS = frozenset({"schema", "epoch_id", "objective_id", "board_id",
    "board_namespace", "candidate_policy_sha256", "max_attempts", "prior_job",
    "prior_incident_signature_sha256", "not_before_observed_at", "expires_at"})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def ordinary_bytes(path: Path, maximum: int = 8 * 1024 * 1024) -> bytes:
    if (not path.is_absolute() or ".." in path.parts
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("unaliased absolute repair metadata required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or not 0 <= before.st_size <= maximum):
            raise ValueError("bounded ordinary repair metadata required")
        raw = bytearray()
        while len(raw) <= maximum:
            part = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink,
                           s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if (len(raw) != before.st_size or len(raw) > maximum
                or stamp(before) != stamp(os.fstat(fd))
                or stamp(before) != stamp(path.lstat())):
            raise ValueError("repair metadata changed while read")
        return bytes(raw)
    finally:
        os.close(fd)


def descriptor(path: Path) -> dict:
    raw = ordinary_bytes(path)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked_descriptor(value) -> bytes:
    if (type(value) is not dict or set(value) != {"path", "bytes", "sha256"}
            or type(value["bytes"]) is not int or not 0 <= value["bytes"] <= 8 * 1024**2
            or type(value["sha256"]) is not str or not _SHA.fullmatch(value["sha256"])):
        raise ValueError("closed typed repair descriptor required")
    path = Path(value["path"])
    raw = ordinary_bytes(path)
    if len(raw) != value["bytes"] or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("repair descriptor changed")
    return raw


def json_object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate repair metadata key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("repair metadata object required")
    return value


@dataclass(frozen=True)
class AuthorizationEpoch:
    value: dict
    commitment: str
    directory: Path


def parse_authorization_epoch(state_root: Path, board: dict) -> AuthorizationEpoch | None:
    value = board.get("repair_authorization_epoch")
    if value is None:
        return None
    if type(value) is not dict or set(value) != FIELDS or value["schema"] != SCHEMA:
        raise ValueError("closed explicit repair authorization epoch required")
    value = json_object(canonical(value))  # Detach caller-owned dictionaries.
    for key in ("epoch_id", "board_id", "board_namespace"):
        if type(value[key]) is not str or not _NAME.fullmatch(value[key]):
            raise ValueError("finite repair epoch identifier required")
    if value["board_id"] != board["id"]:
        raise ValueError("repair epoch belongs to another board")
    if (type(value["objective_id"]) is not str or not value["objective_id"].strip()
            or len(value["objective_id"]) > 512):
        raise ValueError("explicit user objective required")
    for key in ("candidate_policy_sha256", "prior_incident_signature_sha256"):
        if type(value[key]) is not str or not _SHA.fullmatch(value[key]):
            raise ValueError("repair epoch commitment required")
    if type(value["max_attempts"]) is not int or not 1 <= value["max_attempts"] <= 3:
        raise ValueError("one finite repair epoch may authorize at most three attempts")
    start, end = value["not_before_observed_at"], value["expires_at"]
    if (any(type(n) not in (int, float) or not math.isfinite(n) for n in (start, end))
            or not 0 < start < end or end - start > 7 * 86400):
        raise ValueError("finite original repair epoch time boundary required")
    root = Path(state_root)
    if (not root.is_absolute() or ".." in root.parts
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError("unaliased original repair state root required")
    prior = value["prior_job"]
    if type(prior) is not dict or prior.get("path") != str(root / "repairs" / board["id"] / "job.json"):
        raise ValueError("epoch must preserve the exact legacy board job")
    checked_descriptor(prior)
    directory = root / "repair-epochs" / value["epoch_id"] / "repairs" / board["id"]
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("repair epoch job must not be aliased")
    return AuthorizationEpoch(value, digest(value), directory)


def repair_job_directory(state_root: Path, board: dict) -> Path:
    epoch = parse_authorization_epoch(state_root, board)
    return epoch.directory if epoch else Path(state_root) / "repairs" / board["id"]


def validate_source_successors(epoch: AuthorizationEpoch, policy: dict, chain,
                               attempts: int) -> None:
    """Authenticate source successors without changing the immutable epoch.

    A link cannot change roots, path sets, limits, namespace, expiry or coding
    attempts. The actual activation owner writes each link with one exact CAS;
    readers verify genuine Git objects and original external-unit completion.
    """
    if type(chain) is not list or len(chain) > min(attempts, epoch.value["max_attempts"]):
        raise ValueError("source successor chain exceeds the original spent attempt bound")
    if not chain:
        if digest(policy) != epoch.value["candidate_policy_sha256"]:
            raise ValueError("source policy differs from the immutable epoch anchor")
        return
    previous, prior_number, current = None, 0, None
    for pin in chain:
        link = json_object(checked_descriptor(pin))
        if set(link) != {"schema", "epoch_sha256", "attempt_number", "previous_link",
                "prior_policy", "successor_policy", "validation", "freeze", "activation", "git_snapshot", "external_terminal"}:
            raise ValueError("closed source successor link required")
        if (link["schema"] != "agent-supervisor/repair-source-successor@1"
                or link["epoch_sha256"] != epoch.commitment or link["previous_link"] != previous
                or type(link["attempt_number"]) is not int
                or not prior_number < link["attempt_number"] <= attempts):
            raise ValueError("forked, replayed or unspent source successor")
        before, after = link["prior_policy"], link["successor_policy"]
        if (type(before) is not dict or type(after) is not dict or set(before) != set(after)
                or any(before[k] != after[k] for k in before if k != "base_commit")
                or before["base_commit"] == after["base_commit"]):
            raise ValueError("successor may change only the selected source commit")
        if current is None:
            if digest(before) != epoch.value["candidate_policy_sha256"]:
                raise ValueError("first source successor has no epoch anchor")
        elif canonical(before) != canonical(current):
            raise ValueError("source successor policy hash chain differs")
        frozen = json_object(checked_descriptor(link["freeze"]))
        validation = json_object(checked_descriptor(link["validation"]))
        activation = json_object(checked_descriptor(link["activation"]))
        if (validation.get("schema") != "agent-supervisor/isolated-repair-candidate-validation@1"
                or validation.get("passed") is not True
                or any(validation.get(k) is not True for k in ("source_closed", "native_closed", "processes_closed", "storage_closed"))
                or validation.get("cleanup_errors") != []
                or validation.get("freeze_sha256") != frozen.get("freeze_sha256")
                or validation.get("source_commitment") != frozen.get("source_commitment")):
            raise ValueError("source successor lacks original validated frozen source")
        from .repair_candidate_owners import verify_source_successor_activation
        verify_source_successor_activation(link, frozen, activation)
        current, previous, prior_number = after, pin, link["attempt_number"]
    if canonical(current) != canonical(policy):
        raise ValueError("configured source policy is not the unique chain head")


def require_current_authorization(config: dict, board: dict, *, incident=None,
                                  job=None, now=None) -> AuthorizationEpoch | None:
    configured = next((b for b in config["boards"] if b["id"] == board["id"]), None)
    if configured is None or configured != board:
        raise ValueError("authorization must use the original configured board")
    epoch = parse_authorization_epoch(Path(config["state_dir"]), board)
    if epoch is None:
        return None
    value = epoch.value
    worker = config["repair_worker"]
    policy = worker.get("candidate_policy")
    if (type(policy) is not dict or policy.get("namespace") != value["board_namespace"]
            or policy.get("state_root") != config["state_dir"]):
        raise ValueError("epoch source policy/namespace differs")
    limit = board.get("max_repair_attempts")
    if type(limit) is not int or not value["max_attempts"] <= limit:
        raise ValueError("epoch cannot expand the configured repair cap")
    existing_job = job
    if existing_job is None:
        path = epoch.directory / "job.json"
        existing_job = json_object(ordinary_bytes(path)) if path.exists() else {}
    spent = existing_job.get("attempts", 0)
    if type(spent) is not int or not 0 <= spent <= value["max_attempts"]:
        raise ValueError("original epoch attempts are invalid or exhausted")
    validate_source_successors(epoch, policy, existing_job.get("repair_source_successors", []), spent)
    current = time.time() if now is None else now
    if type(current) not in (int, float) or not value["not_before_observed_at"] <= current < value["expires_at"]:
        raise ValueError("repair epoch is not current")
    prior = json_object(checked_descriptor(value["prior_job"]))
    if (prior.get("board_id") != board["id"] or prior.get("status") != "verified_healthy"
            or prior.get("escalation_reason") != "repair_budget_exhausted"
            or type(prior.get("attempts")) is not int or prior["attempts"] != limit):
        raise ValueError("only a known closed spent legacy job may precede a new epoch")
    old = prior.get("latest_incident", {})
    signature = old.get("signature")
    if (type(signature) is not str
            or hashlib.sha256(signature.encode()).hexdigest() != value["prior_incident_signature_sha256"]):
        raise ValueError("prior incident commitment differs")
    if job is not None:
        if job.get("repair_authorization_sha256") != epoch.commitment:
            raise ValueError("job is not bound to this explicit epoch")
        incident = job.get("latest_incident")
    if incident is not None:
        if type(incident) is not dict:
            raise ValueError("original incident object required")
        observed = incident.get("observed_at") if type(incident) is dict else None
        fresh_signature = incident.get("signature") if type(incident) is dict else None
        if (incident.get("board_id") != board["id"]
                or type(observed) not in (int, float) or not math.isfinite(observed)
                or not value["not_before_observed_at"] <= observed <= current
                or type(fresh_signature) is not str or fresh_signature == signature):
            raise ValueError("new epoch requires a distinct incident after its sealed boundary")
        if job is None:
            state = json_object(ordinary_bytes(Path(config["state_dir"]) / board["id"] / "state.json"))
            if state.get("incident_signature") != fresh_signature or state.get("observed_at") != observed:
                raise ValueError("incident is not the current original observer sample")
    return epoch
