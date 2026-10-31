"""Isolated, retained candidates for the existing infrastructure repair worker.

This module allocates no host resources and never adopts a runtime.  The parent
supplies its original checkout/source custody and live repair-resource guards.
Provider children receive the existing inherited read-only-state boundary;
only their candidate checkout is implementation scratch.  Qualification and
adoption are distinct, hash-bound observations, never an agent's final answer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from ..merge.checkout_lock import (
    CheckoutMutationLease, checkout_mutation_lease_state,
    read_checkout_mutation_lease, checkout_lock_repository_matches,
)
from ..merge.worktree_lifecycle import (
    WorktreeLifecycleStore, WorkspaceLifecycleRecord, current_process_birth,
)
from ..todo_daemon.engine import run_command

POLICY_SCHEMA = "agent-supervisor/isolated-repair-candidate-policy@1"
STATE_SCHEMA = "agent-supervisor/isolated-repair-candidate@1"
FREEZE_SCHEMA = "agent-supervisor/isolated-repair-candidate-freeze@1"
VALIDATION_SCHEMA = "agent-supervisor/isolated-repair-candidate-validation@1"
ADOPTION_SCHEMA = "agent-supervisor/isolated-repair-candidate-adoption@1"
PROTECTED_STATE_ENV = "IPFS_ACCELERATE_AGENT_PROTECTED_STATE_ROOT"
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RELATIVE = re.compile(r"[A-Za-z0-9_. /-]+\Z")


class CandidateError(ValueError):
    """Candidate authority, finite scope, or retained evidence disagrees."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CandidateError(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _integer(value: Any, name: str, maximum: int) -> int:
    _require(type(value) is int and 0 < value <= maximum, name + " must be a bounded plain positive integer")
    return value


def _text(value: Any, name: str, maximum: int = 4096) -> str:
    _require(type(value) is str and 0 < len(value) <= maximum and "\x00" not in value,
             name + " must be bounded nonempty text")
    return value


def _path(value: Any) -> Path:
    path = Path(_text(str(value), "absolute path"))
    _require(path.is_absolute() and ".." not in path.parts, "absolute canonical path required")
    for parent in (path, *path.parents):
        _require(not parent.is_symlink(), "candidate path traverses a symlink")
    return path


def _relative(value: Any) -> str:
    value = _text(value, "source path", 512)
    path = Path(value)
    _require(bool(_RELATIVE.fullmatch(value)) and not path.is_absolute()
             and all(part not in ("", ".", "..", ".git") for part in value.split("/")),
             "unsafe or nonliteral source path")
    return value


def _paths(value: Any, name: str) -> tuple[str, ...]:
    _require(type(value) is list and 0 < len(value) <= 4096, name + " must be a bounded file list")
    result = tuple(_relative(item) for item in value)
    _require(len(result) == len(set(result)), name + " contains duplicate paths")
    return result


@dataclass(frozen=True)
class CandidatePolicy:
    repository_root: Path
    state_root: Path
    namespace: str
    base_commit: str
    source_paths: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    max_source_bytes: int
    max_candidate_bytes: int
    max_files: int
    git_timeout_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return dict(schema=POLICY_SCHEMA, repository_root=str(self.repository_root),
                    state_root=str(self.state_root), namespace=self.namespace,
                    base_commit=self.base_commit, source_paths=list(self.source_paths),
                    allowed_paths=list(self.allowed_paths), max_source_bytes=self.max_source_bytes,
                    max_candidate_bytes=self.max_candidate_bytes, max_files=self.max_files,
                    git_timeout_seconds=self.git_timeout_seconds)


def parse_candidate_policy(worker: Mapping[str, Any]) -> CandidatePolicy | None:
    """Validate the entire opt-in policy before queue or filesystem effects."""
    if "candidate_policy" not in worker:
        return None
    value = worker["candidate_policy"]
    keys = {"schema", "repository_root", "state_root", "namespace", "base_commit",
            "source_paths", "allowed_paths", "max_source_bytes", "max_candidate_bytes",
            "max_files", "git_timeout_seconds"}
    _require(type(value) is dict and set(value) == keys and value["schema"] == POLICY_SCHEMA,
             "closed isolated candidate policy required")
    repository, state = _path(value["repository_root"]), _path(value["state_root"])
    _require(repository.is_dir() and state.is_dir(), "existing repository and state roots required")
    _require(repository != state and repository not in state.parents and state not in repository.parents,
             "candidate state must be disjoint from source")
    namespace = _text(value["namespace"], "namespace", 64)
    _require(re.fullmatch(r"ptr_lane_\d+", namespace) is not None, "explicit repair lane namespace required")
    _require(type(value["base_commit"]) is str and bool(_OID.fullmatch(value["base_commit"])),
             "exact base commit required")
    sources, allowed = _paths(value["source_paths"], "source_paths"), _paths(value["allowed_paths"], "allowed_paths")
    source_bytes = _integer(value["max_source_bytes"], "max_source_bytes", 512 * 1024 * 1024)
    candidate_bytes = _integer(value["max_candidate_bytes"], "max_candidate_bytes", 1024 * 1024 * 1024)
    files = _integer(value["max_files"], "max_files", 4096)
    _require(candidate_bytes >= source_bytes and files >= len(sources), "candidate bounds do not cover selected source")
    timeout = _integer(value["git_timeout_seconds"], "git_timeout_seconds", 120)
    return CandidatePolicy(repository, state, namespace, value["base_commit"], sources,
                           allowed, source_bytes, candidate_bytes, files, timeout)


@dataclass(frozen=True)
class OwnerGuardReceipt:
    """Observation returned only after a parent-owned live guard executes.

    The callbacks, rather than this serializable evidence, carry authority.
    A caller must wrap its actual original source/resource owner, not an agent
    report, subprocess exit, configuration boolean, or historic receipt.
    """
    kind: str
    reference: str
    binding_sha256: str


@dataclass(frozen=True)
class CandidateAuthorization:
    stage: str
    binding_sha256: str
    source_owner_reference: str
    resource_owner_reference: str


class CandidateAuthority(ABC):
    @abstractmethod
    def require_current(self, stage: str, binding: Mapping[str, Any]) -> CandidateAuthorization:
        """Execute original live guards; returning evidence does not acquire a lease."""

    @abstractmethod
    def require_receipt(self, stage: str, binding: Mapping[str, Any],
                        receipt: Mapping[str, Any]) -> None:
        """Independently verify an original validation/adoption result, or refuse."""


class ExistingRepairAuthority(CandidateAuthority):
    """Production adapter over an existing checkout lease and retained owners.

    Construct inside the parent's original admitted scope. ``source_guard``
    checks its real original source-owner handle; ``resource_guard`` checks the
    existing bounded repair cgroup/storage owner. Neither callback may dispatch
    a provider or allocate a replacement owner. The exact checkout lease stays
    live across source transitions. Missing/unknown guards fail closed.
    """
    def __init__(self, *, checkout_lease: CheckoutMutationLease,
                 source_guard: Callable[[str, Mapping[str, Any]], OwnerGuardReceipt],
                 resource_guard: Callable[[str, Mapping[str, Any]], OwnerGuardReceipt],
                 validation_guard: Callable[[Mapping[str, Any], Mapping[str, Any]], OwnerGuardReceipt] | None = None,
                 adoption_guard: Callable[[Mapping[str, Any], Mapping[str, Any]], OwnerGuardReceipt] | None = None):
        _require(type(checkout_lease) is CheckoutMutationLease, "original checkout lease required")
        _require(callable(source_guard) and callable(resource_guard), "live original owner guards required")
        self.checkout_lease = checkout_lease
        self.source_guard, self.resource_guard = source_guard, resource_guard
        _require(validation_guard is None or callable(validation_guard), "invalid validation owner guard")
        _require(adoption_guard is None or callable(adoption_guard), "invalid adoption owner guard")
        self.validation_guard, self.adoption_guard = validation_guard, adoption_guard

    def require_current(self, stage: str, binding: Mapping[str, Any]) -> CandidateAuthorization:
        _require(checkout_mutation_lease_state(self.checkout_lease) == "current"
                 and read_checkout_mutation_lease(self.checkout_lease.lock_path) == self.checkout_lease,
                 "original checkout lease is no longer exact/current")
        _require(checkout_lock_repository_matches(dict(self.checkout_lease.metadata),
                 Path(binding["repository_root"])) is True,
                 "original checkout lease belongs to a different repository")
        receipts = []
        for kind, guard in (("source", self.source_guard), ("resource", self.resource_guard)):
            value = guard(stage, binding)
            _require(type(value) is OwnerGuardReceipt and value.kind == kind
                     and value.binding_sha256 == digest(binding), "original " + kind + " owner guard refused")
            _text(value.reference, kind + " owner reference")
            receipts.append(value.reference)
        return CandidateAuthorization(stage, digest(binding), *receipts)

    def require_receipt(self, stage: str, binding: Mapping[str, Any],
                        receipt: Mapping[str, Any]) -> None:
        self.require_current(stage, binding)
        guard = {"validate": self.validation_guard, "adopt": self.adoption_guard}.get(stage)
        _require(callable(guard), "original " + stage + " owner verifier is not configured")
        value = guard(binding, receipt)
        _require(type(value) is OwnerGuardReceipt and value.kind == stage
                 and value.binding_sha256 == digest(dict(binding=binding, receipt=receipt)),
                 "original " + stage + " receipt was not independently verified")
        _text(value.reference, "original " + stage + " reference")


@dataclass
class CandidateSession:
    policy: CandidatePolicy
    binding: dict[str, Any]
    authority: CandidateAuthority
    lifecycle: WorktreeLifecycleStore
    record: WorkspaceLifecycleRecord
    baseline: dict[str, Any]
    active_baseline: dict[str, Any]
    git_marker: dict[str, Any]
    runner: Callable[..., Any] = field(repr=False)
    stage: str = "prepared"
    freeze: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None

    @property
    def candidate_root(self) -> Path:
        return Path(self.binding["candidate_root"])

    @property
    def metadata_root(self) -> Path:
        return Path(self.binding["metadata_root"])


def _admit(authority: CandidateAuthority, stage: str, binding: Mapping[str, Any]) -> CandidateAuthorization:
    _require(isinstance(authority, CandidateAuthority), "typed original candidate authority required")
    value = authority.require_current(stage, binding)
    _require(type(value) is CandidateAuthorization and value.stage == stage
             and value.binding_sha256 == digest(binding), "current candidate authorization refused")
    _text(value.source_owner_reference, "source owner reference")
    _text(value.resource_owner_reference, "resource owner reference")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _file(path: Path, maximum: int) -> dict[str, Any]:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1,
             "candidate file must be an owned, unaliased regular file")
    _require(before.st_size <= maximum, "candidate file exceeds byte bound")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "candidate file identity changed")
        content_hash, used = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, maximum + 1 - used))
            if not chunk:
                break
            content_hash.update(chunk); used += len(chunk)
            _require(used <= maximum, "candidate file grew beyond byte bound")
        after = os.fstat(fd)
        _require((opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) ==
                 (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "candidate file changed during observation")
    finally:
        os.close(fd)
    named = path.lstat()
    _require((named.st_dev, named.st_ino) == (before.st_dev, before.st_ino), "named candidate file changed")
    return dict(bytes=used, sha256=content_hash.hexdigest(), executable=bool(before.st_mode & 0o111))


def _selected(root: Path, paths: Sequence[str], maximum: int) -> dict[str, Any]:
    result, used = {}, 0
    for name in sorted(paths):
        path = _path(str(root / name))
        value = _file(path, maximum - used)
        result[name] = value; used += value["bytes"]
        _require(used <= maximum, "selected source exceeds finite byte bound")
    return result


def _inventory(root: Path, policy: CandidatePolicy) -> dict[str, Any]:
    _path(str(root))
    root_stat = root.lstat()
    _require(stat.S_ISDIR(root_stat.st_mode) and root_stat.st_uid == os.getuid(), "candidate root is not an owned directory")
    result, used, entries = {}, 0, 0
    for directory, names, files in os.walk(root, followlinks=False):
        _require(len(Path(directory).relative_to(root).parts) <= 32, "candidate directory depth bound exceeded")
        names.sort(); files.sort()
        for name in (*names, *files):
            entries += 1
            _require(entries <= policy.max_files * 8 + 64, "candidate directory entry bound exceeded")
            path = Path(directory) / name
            _path(str(path))
            mode = path.lstat()
            _require(mode.st_uid == os.getuid() and (stat.S_ISDIR(mode.st_mode) or stat.S_ISREG(mode.st_mode)),
                     "candidate contains a foreign or special entry")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git":
                continue
            _require(len(result) < policy.max_files, "candidate file count exceeded")
            value = _file(path, policy.max_candidate_bytes - used)
            result[relative] = value; used += value["bytes"]
    named_root = root.lstat()
    _require((root_stat.st_dev, root_stat.st_ino) == (named_root.st_dev, named_root.st_ino), "candidate root identity changed")
    return result


def _git(policy: CandidatePolicy, runner: Callable[..., Any], cwd: Path, *args: str,
         stdin: str | None = None) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    result = runner(["git", "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false", *args],
                    cwd=cwd, timeout_seconds=policy.git_timeout_seconds,
                    stdin=stdin, preserve_output_newlines=True, environment=environment)
    _require(result.returncode == 0 and type(result.stdout) is str, "bounded original Git operation failed: " + args[0])
    _require(len(result.stdout.encode()) <= 2 * 1024 * 1024, "Git observation exceeds finite metadata bound")
    return result.stdout


def prepare_candidate(*, policy: CandidatePolicy, board_id: str, attempt_id: str,
                      attempt_number: int, active_root: Path, authority: CandidateAuthority,
                      run_command_fn: Callable[..., Any] = run_command) -> CandidateSession:
    """Retain an exactly pinned sparse worktree; never reuse or delete a path.

    Caller enters the original source WRITE and admitted repair owner before
    invoking this function. Failure preserves every created path and lifecycle
    claim for explicit original-owner recovery; there is no cleanup/retry.
    """
    _require(type(policy) is CandidatePolicy, "validated candidate policy required")
    _require(parse_candidate_policy({"candidate_policy": policy.to_dict()}) == policy,
             "candidate policy is not the validated current policy")
    _require(type(board_id) is str and bool(_NAME.fullmatch(board_id))
             and type(attempt_id) is str and bool(_NAME.fullmatch(attempt_id)), "bounded board/attempt identity required")
    _integer(attempt_number, "attempt_number", 1000000)
    active = _path(str(active_root))
    _require(active.is_dir(), "active source root unavailable")
    candidate = policy.state_root / "worktrees" / policy.namespace / attempt_id
    metadata = policy.state_root / "repair-candidates" / board_id / attempt_id
    _path(str(candidate)); _path(str(metadata))
    _require(candidate != active and active not in candidate.parents and candidate not in active.parents,
             "candidate must be disjoint from active source")
    _require(not candidate.exists() and not metadata.exists(), "candidate or retained attempt already exists")
    binding = dict(policy_sha256=digest(policy.to_dict()), repository_root=str(policy.repository_root),
                   namespace=policy.namespace, board_id=board_id, attempt_id=attempt_id,
                   attempt_number=attempt_number, base_commit=policy.base_commit,
                   candidate_root=str(candidate), metadata_root=str(metadata), active_root=str(active))
    _admit(authority, "prepare", binding)
    head = _git(policy, run_command_fn, policy.repository_root, "rev-parse", "--verify", policy.base_commit + "^{commit}").strip()
    _require(head == policy.base_commit, "base commit changed")
    _require(_git(policy, run_command_fn, active, "rev-parse", "--verify", "HEAD").strip() == head,
             "active checkout is not the selected base commit")
    active_baseline = _selected(active, policy.source_paths, policy.max_source_bytes)
    branch = "rescue/repair/" + policy.namespace + "/" + board_id + "/" + attempt_id
    lifecycle = WorktreeLifecycleStore(repo_root=policy.repository_root)
    record = lifecycle.begin_preparing(task_id="infrastructure-repair:" + board_id,
        canonical_task_cid="", attempt=attempt_number, lane_id=policy.namespace,
        workspace_path=candidate, branch=branch, merge_target=policy.base_commit,
        state_dir=str(metadata), allow_replace_stale=False)
    candidate.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_new(metadata / "preparing.json", dict(schema=STATE_SCHEMA, stage="preparing", binding=binding,
                                                policy=policy.to_dict(), lifecycle=record.to_dict()))
    _admit(authority, "prepare", binding)
    _git(policy, run_command_fn, policy.repository_root, "worktree", "add", "--no-checkout", "-b", branch,
         str(candidate), policy.base_commit)
    _git(policy, run_command_fn, candidate, "read-tree", policy.base_commit)
    _git(policy, run_command_fn, candidate, "sparse-checkout", "set", "--no-cone", "--no-sparse-index", "--stdin",
         stdin="".join("/" + name + "\n" for name in policy.source_paths))
    _git(policy, run_command_fn, candidate, "checkout", "--force", branch)
    baseline = _inventory(candidate, policy)
    _require(set(baseline) == set(policy.source_paths), "sparse checkout differs from exact selected file set")
    _require(baseline == active_baseline, "selected active source differs from the pinned base")
    git_marker = _file(candidate / ".git", 16384)
    record = lifecycle.mark_active(candidate, lease_id=record.lease_id, expected_fence=record.fence,
                                   expected_record=record)
    session = CandidateSession(policy, binding, authority, lifecycle, record, baseline,
                               active_baseline, git_marker, run_command_fn)
    _write_new(metadata / "prepared.json", dict(schema=STATE_SCHEMA, stage="prepared", binding=binding,
        policy=policy.to_dict(), lifecycle=record.to_dict(), baseline=baseline,
        active_baseline=active_baseline, git_marker=git_marker, coding_attempts_incremented=False))
    return session


def _current(session: CandidateSession, stage: str) -> None:
    _admit(session.authority, stage, session.binding)
    _require(session.lifecycle.load_workspace(session.candidate_root) == session.record,
             "candidate lifecycle changed")
    _require(session.record.owner == current_process_birth(), "candidate owner changed; original recovery required")
    _require(_file(session.candidate_root / ".git", 16384) == session.git_marker, "candidate Git binding changed")
    _require(_selected(Path(session.binding["active_root"]), session.policy.source_paths,
                       session.policy.max_source_bytes) == session.active_baseline,
             "active source changed during candidate repair")


def provider_environment(session: CandidateSession) -> dict[str, str]:
    """Arm the EXISTING provider runner's inherited filesystem boundary.

    The parent must use this exact environment and session.candidate_root for
    both provider commands. The runner refuses if its boundary cannot apply;
    no unprotected or active-checkout fallback is allowed.
    """
    _require(session.stage == "prepared", "provider may only enter a fresh prepared candidate")
    _current(session, "code")
    return {PROTECTED_STATE_ENV: str(session.policy.state_root), "IPFS_PROOF_REUSE_STATE_ROOT": "",
            "GIT_OPTIONAL_LOCKS": "0", "PYTHONDONTWRITEBYTECODE": "1"}


def freeze_candidate(session: CandidateSession) -> dict[str, Any]:
    """Freeze actual file changes, not provider claims; retain candidate_ready."""
    _require(session.stage == "prepared", "candidate was already frozen or held")
    _current(session, "freeze")
    current = _inventory(session.candidate_root, session.policy)
    changed = sorted(name for name in set(session.baseline) | set(current)
                     if session.baseline.get(name) != current.get(name))
    _require(bool(changed), "provider produced no candidate source change")
    _require(set(changed) <= set(session.policy.allowed_paths), "candidate changed a path outside its admitted scope")
    _require(set(session.baseline) <= set(current), "candidate source deletion requires separate explicit policy")
    value = dict(schema=FREEZE_SCHEMA, stage="candidate_ready", binding=session.binding,
                 baseline_sha256=digest(session.baseline), manifest=current, changed_paths=changed,
                 source_commitment=digest(dict(base_commit=session.policy.base_commit, manifest=current)),
                 validation_passed=False, runtime_adopted=False, completion_authority=False,
                 coding_attempts_incremented=False)
    value["freeze_sha256"] = digest(value)
    session.record = session.lifecycle.mark_settling(session.candidate_root,
        lease_id=session.record.lease_id, expected_fence=session.record.fence, expected_record=session.record)
    _write_new(session.metadata_root / "candidate-ready.json", value)
    session.freeze, session.stage = value, "candidate_ready"
    return value


def _frozen(session: CandidateSession, stage: str) -> None:
    _current(session, stage)
    _require(session.freeze is not None and _inventory(session.candidate_root, session.policy) == session.freeze["manifest"],
             "frozen candidate source changed")


def _positive_closeout(receipt: Mapping[str, Any]) -> None:
    for name in ("source_closed", "native_closed", "processes_closed", "storage_closed"):
        _require(receipt.get(name) is True, "positive original " + name + " receipt required")
    _require(receipt.get("cleanup_errors") == [], "original cleanup error or unknown closure")


def record_validation(session: CandidateSession, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Record independently owned checks; never run tests from agent JSON.

    The parent-owned authority must independently verify the actual original
    result descriptors and closeout before approving the ``validate`` stage.
    A persisted JSON flag by itself cannot call this capability.
    """
    _require(session.stage == "candidate_ready", "candidate is not waiting for validation")
    _frozen(session, "validate")
    _require(type(receipt) is dict and receipt.get("schema") == VALIDATION_SCHEMA,
             "original candidate validation receipt required")
    _require(receipt.get("freeze_sha256") == session.freeze["freeze_sha256"]
             and receipt.get("source_commitment") == session.freeze["source_commitment"]
             and receipt.get("passed") is True, "validation failed or belongs to stale source")
    _positive_closeout(receipt)
    _text(receipt.get("original_receipt_reference"), "original validation receipt")
    session.authority.require_receipt("validate", session.binding, receipt)
    value = dict(schema=STATE_SCHEMA, stage="adoption_pending", binding=session.binding,
                 freeze_sha256=session.freeze["freeze_sha256"], validation=dict(receipt),
                 validation_sha256=digest(receipt), runtime_adopted=False,
                 coding_attempts_incremented=False, completion_authority=False)
    _write_new(session.metadata_root / "adoption-pending.json", value)
    session.validation, session.stage = dict(receipt), "adoption_pending"
    return value


def record_adoption(session: CandidateSession, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Observe an actual original-owner cutover; never perform or retry it."""
    _require(session.stage == "adoption_pending", "candidate is not waiting for adoption")
    _frozen(session, "adopt")
    _require(type(receipt) is dict and receipt.get("schema") == ADOPTION_SCHEMA
             and receipt.get("freeze_sha256") == session.freeze["freeze_sha256"]
             and receipt.get("validation_sha256") == digest(session.validation)
             and receipt.get("adopted") is True, "adoption failed, stale, or unknown")
    _positive_closeout(receipt)
    for name in ("original_receipt_reference", "runtime_release", "loaded_source_commitment"):
        _text(receipt.get(name), name)
    _require(receipt["loaded_source_commitment"] == session.freeze["source_commitment"], "adopted source differs")
    session.authority.require_receipt("adopt", session.binding, receipt)
    value = dict(schema=STATE_SCHEMA, stage="adopted_pending_probe", binding=session.binding,
                 adoption=dict(receipt), adoption_sha256=digest(receipt),
                 fresh_observer_probe_required=True, observer_verified=False,
                 completion_authority=False, coding_attempts_incremented=False)
    _write_new(session.metadata_root / "adopted-pending-probe.json", value)
    session.stage = "adopted_pending_probe"
    return value


def _read_evidence(path: Path, maximum: int = 8 * 1024 * 1024) -> dict[str, Any]:
    descriptor = _file(_path(str(path)), maximum)
    payload = path.read_bytes()
    _require(len(payload) == descriptor["bytes"] and hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
             "retained evidence changed during read")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "retained evidence contains duplicate keys")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(CandidateError("nonfinite retained evidence")))
    _require(type(value) is dict, "retained candidate evidence must be an object")
    return value


def resume_candidate(*, policy: CandidatePolicy, board_id: str, attempt_id: str,
                     attempt_number: int, active_root: Path, authority: CandidateAuthority,
                     run_command_fn: Callable[..., Any] = run_command) -> CandidateSession:
    """Resume ONLY frozen validation/adoption work without another provider call.

    A changed process requires the existing original lifecycle owner's exact
    dead-owner proof and atomic transfer. Alive/unknown owners, stale policies,
    changed files or incomplete receipts retain all old evidence. This function
    neither replenishes coding attempts nor resumes a partially executed agent.
    """
    _require(type(policy) is CandidatePolicy and
             parse_candidate_policy({"candidate_policy": policy.to_dict()}) == policy,
             "validated current candidate policy required")
    _require(type(board_id) is str and bool(_NAME.fullmatch(board_id))
             and type(attempt_id) is str and bool(_NAME.fullmatch(attempt_id)), "bounded board/attempt identity required")
    _integer(attempt_number, "attempt_number", 1000000)
    active = _path(str(active_root))
    candidate = _path(str(policy.state_root / "worktrees" / policy.namespace / attempt_id))
    metadata = _path(str(policy.state_root / "repair-candidates" / board_id / attempt_id))
    _require(candidate.is_dir() and metadata.is_dir(), "retained frozen candidate is unavailable")
    _require(candidate != active and active not in candidate.parents and candidate not in active.parents,
             "candidate must remain disjoint from active source")
    binding = dict(policy_sha256=digest(policy.to_dict()), repository_root=str(policy.repository_root),
        namespace=policy.namespace, board_id=board_id, attempt_id=attempt_id,
        attempt_number=attempt_number, base_commit=policy.base_commit,
        candidate_root=str(candidate), metadata_root=str(metadata), active_root=str(active))
    _admit(authority, "resume", binding)
    names = list(metadata.iterdir())
    _require(len(names) <= 64, "retained candidate metadata entry bound exceeded")
    _require(not (metadata / "adopted-pending-probe.json").exists(),
             "runtime adoption already recorded; use its fresh-probe path without another adoption")
    prepared = _read_evidence(metadata / "prepared.json")
    ready = _read_evidence(metadata / "candidate-ready.json")
    _require(prepared.get("schema") == STATE_SCHEMA and prepared.get("stage") == "prepared"
             and prepared.get("binding") == binding and prepared.get("policy") == policy.to_dict(),
             "prepared candidate policy or binding changed")
    _require(ready.get("schema") == FREEZE_SCHEMA and ready.get("stage") == "candidate_ready"
             and ready.get("binding") == binding and ready.get("validation_passed") is False
             and ready.get("runtime_adopted") is False and ready.get("completion_authority") is False
             and ready.get("coding_attempts_incremented") is False, "frozen candidate scope changed")
    unsigned = dict(ready)
    frozen_digest = unsigned.pop("freeze_sha256", None)
    _require(type(frozen_digest) is str and bool(_SHA.fullmatch(frozen_digest))
             and digest(unsigned) == frozen_digest, "frozen candidate receipt digest changed")
    baseline = prepared.get("baseline")
    _require(type(baseline) is dict and prepared.get("active_baseline") == baseline
             and set(baseline) == set(policy.source_paths)
             and _selected(active, policy.source_paths, policy.max_source_bytes) == baseline,
             "retained baseline or active source changed")
    _require(_file(candidate / ".git", 16384) == prepared.get("git_marker"), "retained candidate Git binding changed")
    current_manifest = _inventory(candidate, policy)
    changed = sorted(name for name in set(baseline) | set(current_manifest)
                     if baseline.get(name) != current_manifest.get(name))
    _require(ready.get("manifest") == current_manifest and ready.get("changed_paths") == changed
             and bool(changed) and set(changed) <= set(policy.allowed_paths)
             and set(baseline) <= set(current_manifest)
             and ready.get("baseline_sha256") == digest(baseline)
             and ready.get("source_commitment") == digest(dict(base_commit=policy.base_commit, manifest=current_manifest)),
             "retained frozen candidate source or allowed changes disagree")
    stage, validation_receipt = "candidate_ready", None
    pending_path = metadata / "adoption-pending.json"
    if pending_path.exists():
        pending = _read_evidence(pending_path)
        validation_receipt = pending.get("validation")
        _require(pending.get("schema") == STATE_SCHEMA and pending.get("stage") == "adoption_pending"
                 and pending.get("binding") == binding and pending.get("freeze_sha256") == frozen_digest
                 and pending.get("runtime_adopted") is False and pending.get("completion_authority") is False
                 and pending.get("coding_attempts_incremented") is False
                 and type(validation_receipt) is dict
                 and pending.get("validation_sha256") == digest(validation_receipt),
                 "retained validation/adoption-pending receipt changed")
        _require(validation_receipt.get("schema") == VALIDATION_SCHEMA
                 and validation_receipt.get("freeze_sha256") == frozen_digest
                 and validation_receipt.get("source_commitment") == ready["source_commitment"]
                 and validation_receipt.get("passed") is True, "retained validation is stale or failed")
        _positive_closeout(validation_receipt)
        authority.require_receipt("validate", binding, validation_receipt)
        stage = "adoption_pending"
    lifecycle = WorktreeLifecycleStore(repo_root=policy.repository_root)
    record = lifecycle.load_workspace(candidate)
    _require(type(record) is WorkspaceLifecycleRecord, "original candidate lifecycle is missing")
    _require(type(prepared.get("lifecycle")) is dict, "original lifecycle evidence is unavailable")
    original = WorkspaceLifecycleRecord.from_dict(prepared["lifecycle"])
    expected_branch = "rescue/repair/" + policy.namespace + "/" + board_id + "/" + attempt_id
    _require(record.record_id == original.record_id and record.task_id == "infrastructure-repair:" + board_id
             and record.canonical_task_cid == "" and record.attempt == attempt_number
             and record.workspace_path == str(candidate) and record.branch == expected_branch
             and record.merge_target == policy.base_commit and record.state_dir == str(metadata)
             and record.repo_root == original.repo_root and record.state.value == "settling",
             "original frozen lifecycle binding disagrees")
    captured = record
    owner_changed = record.owner != current_process_birth()
    if owner_changed:
        expected = dict(expected_record_id=record.record_id, expected_fence=record.fence,
            expected_lease_id=record.lease_id, expected_task_id=record.task_id,
            expected_canonical_task_cid=record.canonical_task_cid, expected_attempt=record.attempt,
            expected_branch=record.branch, expected_merge_target=record.merge_target,
            expected_repo_root=record.repo_root, expected_state_dir=record.state_dir)
        lifecycle.require_exact_dead_owner(candidate, **expected)
        _admit(authority, "resume", binding)
        record = lifecycle.adopt_dead_owner(candidate, lane_id=policy.namespace, **expected)
        record = lifecycle.mark_settling(candidate, lease_id=record.lease_id,
            expected_fence=record.fence, expected_record=record)
    session = CandidateSession(policy, binding, authority, lifecycle, record, baseline,
        prepared["active_baseline"], prepared["git_marker"], run_command_fn,
        stage=stage, freeze=ready, validation=validation_receipt)
    _current(session, "resume")
    receipt = dict(schema=STATE_SCHEMA, stage=stage, binding=binding,
        original_lifecycle=captured.to_dict(), resumed_lifecycle=record.to_dict(),
        dead_owner_transferred=owner_changed, freeze_sha256=frozen_digest,
        validation_sha256=digest(validation_receipt) if validation_receipt is not None else None,
        provider_dispatched=False, coding_attempts_incremented=False,
        runtime_adopted=False, completion_authority=False)
    resume_path = metadata / ("resumed-fence-" + str(record.fence) + "-from-" + str(captured.fence)
                              + "-pid-" + str(record.owner.pid) + ".json")
    if resume_path.exists():
        _require(_read_evidence(resume_path) == receipt, "existing resume receipt differs")
    else:
        _write_new(resume_path, receipt)
    return session


def observe_verified_deferred_adoption(*, policy: CandidatePolicy, board_id: str,
        attempt_id: str, attempt_number: int, authority: CandidateAuthority,
        deferred_receipt: Mapping[str, Any], activation_receipt: Mapping[str, Any],
        source_successor: Mapping[str, Any]) -> dict[str, Any]:
    """Record an already activated, original-owner-verified source successor.

    This API does not resume a provider, validate again, mutate a checkout,
    transfer lifecycle custody, adopt a source or touch services. It observes
    the exact completed external operation. Ordinary resume/_current keep
    their original active-baseline refusal without any exception.
    """
    from .repair_candidate_owners import verify_source_successor_activation
    from .repair_authorization import checked_descriptor, json_object
    _require(type(policy) is CandidatePolicy, "validated current source policy required")
    _integer(attempt_number, "attempt_number", 3)
    _require(bool(_NAME.fullmatch(board_id)) and bool(_NAME.fullmatch(attempt_id)), "original board/token required")
    metadata = policy.state_root / "repair-candidates" / board_id / attempt_id
    prepared = _read_evidence(metadata / "prepared.json")
    ready = _read_evidence(metadata / "candidate-ready.json")
    pending = _read_evidence(metadata / "adoption-pending.json")
    binding = prepared["binding"]
    old = parse_candidate_policy({"candidate_policy": prepared["policy"]})
    _require(old is not None and {k: v for k, v in old.to_dict().items() if k != "base_commit"}
             == {k: v for k, v in policy.to_dict().items() if k != "base_commit"},
             "deferred observation cannot widen source policy or resource bounds")
    _require(binding == ready["binding"] == pending["binding"]
             and binding["board_id"] == board_id and binding["attempt_id"] == attempt_id
             and binding["attempt_number"] == attempt_number
             and binding["base_commit"] == old.base_commit
             and binding["policy_sha256"] == digest(old.to_dict()), "original frozen binding differs")
    unsigned = dict(ready); freeze_sha = unsigned.pop("freeze_sha256", None)
    _require(freeze_sha == digest(unsigned) and ready["source_commitment"]
             == digest(dict(base_commit=old.base_commit, manifest=ready["manifest"])), "original freeze changed")
    candidate_root = Path(binding["candidate_root"])
    _require(candidate_root == policy.state_root / "worktrees" / policy.namespace / attempt_id
             and _inventory(candidate_root, old) == ready["manifest"]
             and _file(candidate_root / ".git", 16384) == prepared["git_marker"], "retained candidate changed")
    validation = pending["validation"]
    _require(pending["stage"] == "adoption_pending" and pending["freeze_sha256"] == freeze_sha
             and pending["validation_sha256"] == digest(validation)
             and validation["schema"] == VALIDATION_SCHEMA and validation["passed"] is True
             and validation["freeze_sha256"] == freeze_sha, "original accepted validation differs")
    _positive_closeout(validation)
    _require(deferred_receipt["known_deferred"] is True and deferred_receipt["adopted"] is False
             and deferred_receipt["freeze_sha256"] == freeze_sha
             and activation_receipt["schema"] == ADOPTION_SCHEMA and activation_receipt["adopted"] is True
             and activation_receipt["freeze_sha256"] == freeze_sha
             and activation_receipt["validation_sha256"] == digest(validation)
             and activation_receipt["loaded_source_commitment"] == ready["source_commitment"],
             "actual deferred adoption binding differs")
    _positive_closeout(activation_receipt)
    link = dict(source_successor)
    _require(link["prior_policy"] == old.to_dict() and link["successor_policy"] == policy.to_dict()
             and link["attempt_number"] == attempt_number, "source successor is not this original attempt")
    actual = json_object(checked_descriptor(link["activation"]))
    verify_source_successor_activation(link, ready, actual)
    _require(_selected(Path(binding["active_root"]), policy.source_paths, policy.max_source_bytes)
             == ready["manifest"], "current activated source is not the validated manifest")
    lifecycle = WorktreeLifecycleStore(repo_root=policy.repository_root)
    record = lifecycle.load_workspace(candidate_root)
    original = WorkspaceLifecycleRecord.from_dict(prepared["lifecycle"])
    _require(type(record) is WorkspaceLifecycleRecord and record.record_id == original.record_id
             and record.task_id == "infrastructure-repair:" + board_id and record.attempt == attempt_number
             and record.workspace_path == str(candidate_root) and record.merge_target == old.base_commit
             and record.state_dir == str(metadata) and record.state.value == "settling",
             "original retained lifecycle differs")
    lifecycle.require_exact_dead_owner(candidate_root, expected_record_id=record.record_id,
        expected_fence=record.fence, expected_lease_id=record.lease_id, expected_task_id=record.task_id,
        expected_canonical_task_cid=record.canonical_task_cid, expected_attempt=record.attempt,
        expected_branch=record.branch, expected_merge_target=record.merge_target,
        expected_repo_root=record.repo_root, expected_state_dir=record.state_dir)
    authority.require_receipt("adopt", binding, activation_receipt)
    value = dict(schema=STATE_SCHEMA, stage="adopted_pending_probe", binding=binding,
        adoption=dict(activation_receipt), adoption_sha256=digest(activation_receipt),
        source_successor_sha256=digest(link), deferred_original_owner_observation=True,
        fresh_observer_probe_required=True, observer_verified=False,
        provider_or_validation_or_adoption_replayed=False, completion_authority=False,
        coding_attempts_incremented=False, lifecycle_mutations=0)
    path = metadata / "adopted-pending-probe.json"
    if path.exists():
        _require(_read_evidence(path) == value, "existing deferred observation differs; no replay")
    else:
        _write_new(path, value)
    return value
