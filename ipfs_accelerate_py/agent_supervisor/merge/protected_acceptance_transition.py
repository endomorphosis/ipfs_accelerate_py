"""Hardened Git construction for prompt-v3 protected acceptance phases.

The builder writes blobs and a tree through an alternate index, creates a
direct one-parent commit object, validates while the canonical repository-wide
checkout mutation lease is held, and publishes only by checked ``update-ref``.
It deliberately has no dependency on the entrypoint composition layer.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core.protected_acceptance_contracts import (
    CandidatePlan,
    EvidenceHandle,
    GitFileIdentity,
    PhaseCandidateRequest,
    PhaseEvidenceResult,
    ProtectedAcceptanceDenied,
    ProtectedAcceptanceError,
    PublicationResult,
    RejectionResult,
    ValidatedCandidate,
    canonical_json_bytes,
    content_id,
)
from .checkout_lock import (
    DEFAULT_CHECKOUT_MUTATION_LOCK_NAME,
    CheckoutMutationLease,
    acquire_checkout_mutation_lease,
    read_checkout_mutation_lease,
    release_checkout_mutation_lease,
)

_MAX_GIT_OUTPUT = 4 * 1024 * 1024
_FORBIDDEN_GIT_CONFIG_PREFIXES = (
    "filter.",
    "diff.",
    "include.",
    "credential.",
    "url.",
    "http.",
    "gpg.",
)
_FORBIDDEN_GIT_CONFIG_KEYS = frozenset(
    {
        "commit.gpgsign",
        "tag.gpgsign",
        "user.signingkey",
        "core.hookspath",
        "core.attributesfile",
        "core.sshcommand",
        "core.gitproxy",
        "core.fsmonitor",
        "core.worktree",
        "core.alternaterefscommand",
    }
)
_FORBIDDEN_PROCESS_ENV = frozenset(
    {
        "LD_PRELOAD",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    }
)
_HARMLESS_CLEARED_GIT_ENV = frozenset({"GIT_PAGER"})


class ProtectedTransitionGitError(ProtectedAcceptanceError):
    """A Git or repository invariant failed closed."""


class ProtectedTransitionRace(ProtectedTransitionGitError):
    """The checkout lease or target ref changed during the transition."""


@dataclass(frozen=True)
class TransitionHooks:
    """Testable crash/race boundaries; production callers normally omit it."""

    before_tree: Callable[[], None] | None = None
    after_commit: Callable[[str], None] | None = None
    before_cas: Callable[[CandidatePlan], None] | None = None
    after_cas: Callable[[CandidatePlan], None] | None = None

    def __post_init__(self) -> None:
        for callback in (
            self.before_tree,
            self.after_commit,
            self.before_cas,
            self.after_cas,
        ):
            if callback is not None and not callable(callback):
                raise TypeError("transition hook must be callable")


_NO_TRANSITION_HOOKS = TransitionHooks()


def _reject_ambient_git_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    source = os.environ if environ is None else environ
    if not isinstance(source, Mapping):
        raise TypeError("environment inspection requires a mapping")
    forbidden = sorted(
        key
        for key in source
        if (key.startswith("GIT_") and key not in _HARMLESS_CLEARED_GIT_ENV)
        or key in _FORBIDDEN_PROCESS_ENV
        or key.startswith("DYLD_")
    )
    if forbidden:
        # Names are safe to disclose; values may be credentials or paths and
        # are deliberately never copied into the exception.
        raise ProtectedAcceptanceDenied(
            "ambient Git/loader environment is forbidden: " + ", ".join(forbidden)
        )


def _absolute_git() -> str:
    executable = shutil.which("git", path="/usr/bin:/bin")
    if not executable or not os.path.isabs(executable):
        raise ProtectedTransitionGitError("an absolute Git executable is unavailable")
    metadata = os.stat(executable, follow_symlinks=True)
    if not stat.S_ISREG(metadata.st_mode):
        raise ProtectedTransitionGitError("Git executable is not a regular file")
    return executable


def _git_environment(
    *,
    index_path: Path | None = None,
    timestamp: str | None = None,
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "HOME": "/nonexistent/protected-acceptance",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_COUNT": "6",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": "/dev/null",
        "GIT_CONFIG_KEY_1": "commit.gpgSign",
        "GIT_CONFIG_VALUE_1": "false",
        "GIT_CONFIG_KEY_2": "tag.gpgSign",
        "GIT_CONFIG_VALUE_2": "false",
        "GIT_CONFIG_KEY_3": "core.attributesFile",
        "GIT_CONFIG_VALUE_3": "/dev/null",
        "GIT_CONFIG_KEY_4": "core.fsmonitor",
        "GIT_CONFIG_VALUE_4": "false",
        "GIT_CONFIG_KEY_5": "protocol.file.allow",
        "GIT_CONFIG_VALUE_5": "never",
    }
    if index_path is not None:
        environment["GIT_INDEX_FILE"] = os.fspath(index_path)
    if timestamp is not None:
        environment.update(
            {
                "GIT_AUTHOR_NAME": "Agent Supervisor Protected Acceptance",
                "GIT_AUTHOR_EMAIL": "protected-acceptance@example.invalid",
                "GIT_AUTHOR_DATE": timestamp,
                "GIT_COMMITTER_NAME": "Agent Supervisor Protected Acceptance",
                "GIT_COMMITTER_EMAIL": "protected-acceptance@example.invalid",
                "GIT_COMMITTER_DATE": timestamp,
            }
        )
    return environment


def _run_git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    index_path: Path | None = None,
    timestamp: str | None = None,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> bytes:
    if type(arguments) not in {tuple, list} or any(
        type(item) is not str for item in arguments
    ):
        raise TypeError("Git arguments must be a sequence of strings")
    command = [_absolute_git(), *arguments]
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=_git_environment(index_path=index_path, timestamp=timestamp),
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtectedTransitionGitError("Git invocation failed") from exc
    if len(result.stdout) > _MAX_GIT_OUTPUT or len(result.stderr) > _MAX_GIT_OUTPUT:
        raise ProtectedTransitionGitError("Git output exceeded its closed bound")
    if result.returncode not in allowed_returncodes:
        raise ProtectedTransitionGitError(
            "Git command failed without exposing untrusted stderr"
        )
    return bytes(result.stdout)


def _open_owned_directory(path: Path, *, deny_group_other_write: bool = True) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtectedTransitionGitError(
            "protected directory cannot be opened safely"
        ) from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or (deny_group_other_write and metadata.st_mode & 0o022)
    ):
        os.close(descriptor)
        raise ProtectedTransitionGitError("protected directory owner or mode is unsafe")
    return descriptor


def _stable_owned_file(
    path: Path,
    *,
    maximum_bytes: int,
    required_mode: int | None = None,
) -> bytes:
    parent_fd = _open_owned_directory(path.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or before.st_size < 0
            or before.st_size > maximum_bytes
            or (
                required_mode is not None
                and stat.S_IMODE(before.st_mode) != required_mode
            )
        ):
            raise ProtectedTransitionGitError(
                "protected file owner, link, mode, or size is unsafe"
            )
        chunks = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if len(data) > maximum_bytes or identity_before != identity_after:
            raise ProtectedTransitionRace(
                "protected file changed during its stable read"
            )
        return data
    except OSError as exc:
        raise ProtectedTransitionGitError(
            "protected file cannot be read safely"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _lexical_no_symlinks(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ProtectedTransitionGitError("repository path is not absolute and lexical")
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise ProtectedTransitionGitError(
                "repository path component is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProtectedTransitionGitError("repository path cannot contain symlinks")
    return path


def _common_git_directory(repo_root: Path) -> Path:
    raw = (
        _run_git(
            repo_root,
            ("rev-parse", "--path-format=absolute", "--git-common-dir"),
        )
        .decode("utf-8", "strict")
        .strip()
    )
    common = _lexical_no_symlinks(Path(raw))
    descriptor = _open_owned_directory(common)
    os.close(descriptor)
    return common


def _entry_lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ProtectedTransitionGitError(
            "repository metadata cannot be inspected"
        ) from exc
    return True


def _validate_repository_policy(repo_root: Path, common_dir: Path) -> None:
    replacements = _run_git(
        repo_root, ("for-each-ref", "--format=%(refname)", "refs/replace/")
    )
    if replacements.strip():
        raise ProtectedAcceptanceDenied("Git replacement refs are forbidden")
    for path in (
        common_dir / "info" / "grafts",
        common_dir / "objects" / "info" / "alternates",
    ):
        if _entry_lexists(path):
            raise ProtectedAcceptanceDenied(
                "Git grafts and object alternates are forbidden"
            )

    configured = _run_git(
        repo_root,
        ("config", "--local", "--null", "--name-only", "--get-regexp", ".*"),
        allowed_returncodes=(0, 1),
    )
    for raw_key in configured.split(b"\0"):
        if not raw_key:
            continue
        try:
            key = raw_key.decode("utf-8", "strict").casefold()
        except UnicodeDecodeError as exc:
            raise ProtectedAcceptanceDenied("Git config key is not UTF-8") from exc
        if key in _FORBIDDEN_GIT_CONFIG_KEYS or key.startswith(
            _FORBIDDEN_GIT_CONFIG_PREFIXES
        ):
            raise ProtectedAcceptanceDenied(
                "repository Git config enables a forbidden extension point"
            )
        if key.startswith("submodule.") and key.endswith(".ignore"):
            raise ProtectedAcceptanceDenied("submodule ignore policy is forbidden")

    hooks = common_dir / "hooks"
    if _entry_lexists(hooks):
        hooks_fd = _open_owned_directory(hooks)
        try:
            for name in os.listdir(hooks_fd):
                if name.startswith(".") or name.endswith(".sample"):
                    continue
                metadata = os.stat(name, dir_fd=hooks_fd, follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode) or metadata.st_mode & 0o111:
                    raise ProtectedAcceptanceDenied(
                        "active or redirected Git hooks are forbidden"
                    )
        finally:
            os.close(hooks_fd)


def _current_ref(repo_root: Path) -> str:
    output = (
        _run_git(
            repo_root,
            ("symbolic-ref", "-q", "HEAD"),
            allowed_returncodes=(0, 1),
        )
        .decode("ascii", "strict")
        .strip()
    )
    return output


def _resolve_one(repo_root: Path, revision: str) -> str:
    output = _run_git(repo_root, ("rev-parse", "--verify", f"{revision}^{{commit}}"))
    lines = output.decode("ascii", "strict").splitlines()
    if len(lines) != 1 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", lines[0]):
        raise ProtectedTransitionGitError("Git revision did not resolve exactly once")
    return lines[0]


def _resolve_tree(repo_root: Path, revision: str) -> str:
    output = _run_git(repo_root, ("rev-parse", "--verify", f"{revision}^{{tree}}"))
    lines = output.decode("ascii", "strict").splitlines()
    if len(lines) != 1 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", lines[0]):
        raise ProtectedTransitionGitError("Git tree did not resolve exactly once")
    return lines[0]


def resolve_prompt_v3_birth_target_once(repo_root: Path, target_ref: str) -> str:
    """Resolve the configured birth target once, never ``HEAD HEAD``."""

    _reject_ambient_git_environment()
    return _resolve_one(repo_root, target_ref)


def _require_clean_contents(repo_root: Path, parent_commit: str) -> None:
    expected_tree = _resolve_tree(repo_root, parent_commit)
    actual_index_tree = (
        _run_git(repo_root, ("write-tree",)).decode("ascii", "strict").strip()
    )
    if actual_index_tree != expected_tree:
        raise ProtectedAcceptanceDenied(
            "checkout real index does not equal the exact expected tree"
        )
    temporary = Path(tempfile.mkdtemp(prefix="protected-acceptance-clean-index-"))
    os.chmod(temporary, 0o700)
    clean_index = temporary / "clean.index"
    try:
        # A controlled umask ensures Git cannot create a group-writable index.
        previous_umask = os.umask(0o077)
        try:
            _run_git(
                repo_root,
                ("read-tree", f"{parent_commit}^{{tree}}"),
                index_path=clean_index,
            )
        finally:
            os.umask(previous_umask)
        os.chmod(clean_index, 0o600, follow_symlinks=False)
        status = _run_git(
            repo_root,
            (
                f"--work-tree={repo_root}",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ),
            index_path=clean_index,
        )
        ignored = _run_git(
            repo_root,
            (
                f"--work-tree={repo_root}",
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ),
            index_path=clean_index,
        )
        if status or ignored:
            raise ProtectedAcceptanceDenied(
                "isolated checkout must contain no modified, untracked, ignored, or submodule drift"
            )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _target_worktree(repo_root: Path, target_ref: str) -> Path | None:
    raw = _run_git(repo_root, ("worktree", "list", "--porcelain", "-z"))
    records = []
    current: dict[str, str] = {}
    for token in raw.split(b"\0"):
        if not token:
            if current:
                records.append(current)
                current = {}
            continue
        try:
            line = token.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise ProtectedTransitionGitError(
                "worktree inventory is not UTF-8"
            ) from exc
        key, separator, value = line.partition(" ")
        if separator:
            current[key] = value
    if current:
        records.append(current)
    matches = [
        Path(item["worktree"]) for item in records if item.get("branch") == target_ref
    ]
    if len(matches) > 1:
        raise ProtectedTransitionRace(
            "protected target ref is held by multiple worktrees"
        )
    return matches[0] if matches else None


def _require_clean_target_worktree(
    target_root: Path, target_ref: str, parent_commit: str
) -> None:
    target_root = _lexical_no_symlinks(target_root)
    descriptor = _open_owned_directory(target_root)
    os.close(descriptor)
    if _current_ref(target_root) != target_ref:
        raise ProtectedTransitionRace("target worktree branch binding changed")
    if _resolve_one(target_root, "HEAD") != parent_commit:
        raise ProtectedTransitionRace("target worktree HEAD changed")
    _require_clean_contents(target_root, parent_commit)


def _require_clean_exact_checkout(request: PhaseCandidateRequest) -> None:
    repo_root = Path(request.repository.root)
    if _current_ref(repo_root):
        raise ProtectedAcceptanceDenied(
            "protected construction requires a detached exact checkout"
        )
    if _resolve_one(repo_root, request.repository.target_ref) != request.parent_commit:
        raise ProtectedTransitionRace(
            "target ref no longer names the exact candidate parent"
        )
    if _resolve_one(repo_root, "HEAD") != request.parent_commit:
        raise ProtectedTransitionRace(
            "checkout HEAD no longer names the exact candidate parent"
        )
    _require_clean_contents(repo_root, request.parent_commit)
    target_root = _target_worktree(repo_root, request.repository.target_ref)
    if target_root is not None:
        _require_clean_target_worktree(
            target_root, request.repository.target_ref, request.parent_commit
        )


def _process_start_identity(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        end = raw.rfind(")")
        fields = raw[end + 2 :].split()
        return fields[19] if end > 0 and len(fields) > 19 else ""
    except (OSError, UnicodeError):
        return ""


def _lease_owner_active(metadata: Mapping[str, Any]) -> bool:
    """Return False only for a conclusively dead, compatible lease owner."""

    if (
        not isinstance(metadata, Mapping)
        or metadata.get("kind") != "merge"
        or not isinstance(metadata.get("pid"), int)
        or int(metadata.get("pid", 0)) <= 0
    ):
        return True
    pid = int(metadata["pid"])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    expected_start = str(metadata.get("process_start_identity") or "")
    if expected_start:
        observed_start = _process_start_identity(pid)
        if not observed_start:
            return True
        if observed_start != expected_start:
            return False
    return True


def _lease_metadata(request: PhaseCandidateRequest) -> dict[str, Any]:
    nonce = secrets.token_bytes(32)
    lease_id = (
        "sha256:"
        + hashlib.sha256(
            request.parent_commit.encode("ascii")
            + request.policy.phase.value.encode("ascii")
            + nonce
        ).hexdigest()
    )
    return {
        "kind": "merge",
        "pid": os.getpid(),
        "owner_script": "protected_acceptance_transition",
        "repo_root": "",
        "worktree_root": request.repository.root,
        "repository_id": "",
        "task_id": "ASE3-033",
        "attempt": 0,
        "branch": request.repository.target_ref,
        "operation": "protected_acceptance_transition",
        "lease_id": lease_id,
        "process_start_identity": _process_start_identity(os.getpid()),
    }


def _acquire_canonical_lease(
    request: PhaseCandidateRequest, common_dir: Path
) -> CheckoutMutationLease:
    if request.repository.lease_name != DEFAULT_CHECKOUT_MUTATION_LOCK_NAME:
        raise ProtectedAcceptanceDenied(
            "transition must use the canonical checkout mutation lock"
        )
    lock_path = common_dir / DEFAULT_CHECKOUT_MUTATION_LOCK_NAME
    lease, reason, _owner, _waited = acquire_checkout_mutation_lease(
        lock_path,
        _lease_metadata(request),
        owner_active=_lease_owner_active,
        timeout_seconds=1.0,
    )
    if lease is None:
        raise ProtectedTransitionRace(
            f"canonical checkout mutation lease unavailable: {reason}"
        )
    directory_fd = _open_owned_directory(common_dir)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return lease


def _validate_lease(plan: CandidatePlan, common_dir: Path) -> CheckoutMutationLease:
    lock_path = common_dir / DEFAULT_CHECKOUT_MUTATION_LOCK_NAME
    try:
        data = _stable_owned_file(
            lock_path, maximum_bytes=64 * 1024, required_mode=0o600
        )
    except ProtectedTransitionGitError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            raise ProtectedTransitionRace("checkout mutation lease is absent") from exc
        raise
    try:
        metadata = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtectedTransitionRace("checkout mutation lease is malformed") from exc
    observed = read_checkout_mutation_lease(lock_path)
    if (
        observed is None
        or observed.lease_id != plan.lease_id
        or observed.device != plan.lease_device
        or observed.inode != plan.lease_inode
        or metadata != dict(observed.metadata)
        or metadata.get("operation") != "protected_acceptance_transition"
        or metadata.get("worktree_root") != plan.request.repository.root
        or metadata.get("branch") != plan.request.repository.target_ref
    ):
        raise ProtectedTransitionRace("checkout mutation lease was lost or replaced")
    return observed


def _release_lease(lease: CheckoutMutationLease, common_dir: Path) -> bool:
    released = release_checkout_mutation_lease(lease)
    if released:
        descriptor = _open_owned_directory(common_dir)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return released


def _checked_update_ref(
    repo_root: Path,
    ref_name: str,
    new_object: str,
    old_object: str,
) -> None:
    _run_git(repo_root, ("update-ref", "--no-deref", ref_name, new_object, old_object))


def _checked_delete_ref(repo_root: Path, ref_name: str, old_object: str) -> None:
    _run_git(repo_root, ("update-ref", "--no-deref", "-d", ref_name, old_object))


def _optional_ref(repo_root: Path, ref_name: str) -> str | None:
    raw = _run_git(
        repo_root,
        ("rev-parse", "--verify", "--quiet", ref_name),
        allowed_returncodes=(0, 1),
    )
    if not raw:
        return None
    lines = raw.decode("ascii", "strict").splitlines()
    if len(lines) != 1 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", lines[0]):
        raise ProtectedTransitionRace("rescue ref did not resolve exactly once")
    return lines[0]


def _rescue_ref(request: PhaseCandidateRequest) -> str:
    phase = request.policy.phase.value.replace("/", "-").lower()
    return (
        "refs/agent-supervisor/protected-acceptance-rescue/"
        f"{phase}-{request.authority.authority_id.removeprefix('sha256:')[:24]}"
    )


def _verify_direct_commit(
    repo_root: Path, commit_id: str, parent: str, tree: str
) -> None:
    raw = _run_git(repo_root, ("cat-file", "commit", commit_id))
    headers = raw.split(b"\n\n", 1)[0].splitlines()
    trees = [line[5:].decode("ascii") for line in headers if line.startswith(b"tree ")]
    parents = [
        line[7:].decode("ascii") for line in headers if line.startswith(b"parent ")
    ]
    if trees != [tree] or parents != [parent]:
        raise ProtectedTransitionGitError(
            "candidate is not the exact direct one-parent commit"
        )


def build_phase_candidate(
    request: PhaseCandidateRequest,
    *,
    authority_validator: Callable[[Any, int], bool],
    hooks: TransitionHooks = _NO_TRANSITION_HOOKS,
) -> CandidatePlan:
    """Build a direct candidate through a fresh alternate index."""

    if (
        not isinstance(request, PhaseCandidateRequest)
        or not callable(authority_validator)
        or not isinstance(hooks, TransitionHooks)
    ):
        raise TypeError(
            "build_phase_candidate requires typed request, authority validator, and hooks"
        )
    now_ns = time.time_ns()
    if not (
        request.authority.issued_at_ns <= now_ns < request.authority.expires_at_ns
        and authority_validator(request.authority, now_ns) is True
    ):
        raise ProtectedAcceptanceDenied(
            "phase authority is not fresh and verifier-authenticated"
        )
    _reject_ambient_git_environment()
    repo_root = _lexical_no_symlinks(Path(request.repository.root))
    root_fd = _open_owned_directory(repo_root)
    os.close(root_fd)
    common_dir = _common_git_directory(repo_root)
    _validate_repository_policy(repo_root, common_dir)
    _require_clean_exact_checkout(request)
    lease = _acquire_canonical_lease(request, common_dir)
    rescue_ref = _rescue_ref(request)
    rescue_created = False
    temporary: Path | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix="protected-acceptance-index-"))
        os.chmod(temporary, 0o700)
        index_path = temporary / "candidate.index"
        _require_clean_exact_checkout(request)
        previous_umask = os.umask(0o077)
        try:
            _run_git(
                repo_root,
                ("read-tree", f"{request.parent_commit}^{{tree}}"),
                index_path=index_path,
            )
        finally:
            os.umask(previous_umask)
        os.chmod(index_path, 0o600, follow_symlinks=False)
        identities = []
        for artifact in sorted(request.artifacts, key=lambda item: item.path):
            blob_id = (
                _run_git(
                    repo_root,
                    ("hash-object", "-w", "--stdin"),
                    input_bytes=artifact.data,
                )
                .decode("ascii", "strict")
                .strip()
            )
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", blob_id):
                raise ProtectedTransitionGitError(
                    "hash-object returned an invalid blob ID"
                )
            _run_git(
                repo_root,
                (
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    "100644",
                    blob_id,
                    artifact.path,
                ),
                index_path=index_path,
            )
            os.chmod(index_path, 0o600, follow_symlinks=False)
            identities.append(
                GitFileIdentity(
                    path=artifact.path,
                    mode="100644",
                    blob_id=blob_id,
                    raw_content_id=content_id(artifact.data),
                    byte_length=len(artifact.data),
                )
            )
        if hooks.before_tree is not None:
            hooks.before_tree()
        tree_id = (
            _run_git(repo_root, ("write-tree",), index_path=index_path)
            .decode("ascii", "strict")
            .strip()
        )
        os.chmod(index_path, 0o600, follow_symlinks=False)
        _stable_owned_file(index_path, maximum_bytes=16 * 1024 * 1024)
        index_fd = os.open(index_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(index_fd)
        finally:
            os.close(index_fd)
        temporary_fd = _open_owned_directory(temporary)
        try:
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        commit_id = (
            _run_git(
                repo_root,
                (
                    "commit-tree",
                    tree_id,
                    "-p",
                    request.parent_commit,
                    "-m",
                    request.commit_message,
                ),
                timestamp=request.commit_timestamp,
            )
            .decode("ascii", "strict")
            .strip()
        )
        _verify_direct_commit(repo_root, commit_id, request.parent_commit, tree_id)
        if not request.dry_run:
            existing_rescue = _optional_ref(repo_root, rescue_ref)
            if existing_rescue is None:
                _checked_update_ref(
                    repo_root,
                    rescue_ref,
                    commit_id,
                    "0" * len(request.parent_commit),
                )
                rescue_created = True
            elif (
                existing_rescue != commit_id
                or _resolve_one(repo_root, request.repository.target_ref)
                != request.parent_commit
            ):
                raise ProtectedTransitionRace(
                    "stale rescue ref is not safely reconcilable"
                )
        if hooks.after_commit is not None:
            hooks.after_commit(commit_id)
        return CandidatePlan(
            request=request,
            tree_id=tree_id,
            commit_id=commit_id,
            rescue_ref=rescue_ref,
            file_identities=tuple(identities),
            lease_id=lease.lease_id,
            lease_device=lease.device,
            lease_inode=lease.inode,
        )
    except Exception as exc:
        cleanup_failed = False
        if rescue_created:
            try:
                _checked_delete_ref(repo_root, rescue_ref, commit_id)
            except ProtectedTransitionGitError:
                cleanup_failed = True
        if not _release_lease(lease, common_dir):
            cleanup_failed = True
        if cleanup_failed:
            raise ProtectedTransitionRace(
                "candidate construction failed and exact cleanup was fenced"
            ) from exc
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _verify_bound_evidence(
    candidate: CandidatePlan,
    handle: EvidenceHandle,
    loader: Callable[[CandidatePlan, EvidenceHandle], bytes],
) -> None:
    if not isinstance(handle, EvidenceHandle):
        raise TypeError("phase evidence loader requires an EvidenceHandle")
    raw = loader(candidate, handle)
    if (
        type(raw) is not bytes
        or len(raw) != handle.byte_length
        or content_id(raw) != handle.content_id
    ):
        raise ProtectedAcceptanceDenied(
            "phase evidence bytes differ from their bounded content handle"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtectedAcceptanceDenied(
            "phase evidence binding is not canonical JSON"
        ) from exc
    expected = {
        "schema": "ipfs_accelerate_py.agent_supervisor.phase-evidence-binding@1",
        "candidate_commit": candidate.commit_id,
        "authority_id": candidate.request.authority.authority_id,
        "kind": handle.kind,
    }
    if payload != expected or raw != canonical_json_bytes(expected):
        raise ProtectedAcceptanceDenied(
            "phase evidence is not bound to the exact candidate and authority"
        )


def run_phase_evidence(
    candidate: CandidatePlan,
    runner: Callable[[CandidatePlan], tuple[EvidenceHandle, ...]],
    evidence_loader: Callable[[CandidatePlan, EvidenceHandle], bytes],
) -> PhaseEvidenceResult:
    if (
        not isinstance(candidate, CandidatePlan)
        or not callable(runner)
        or not callable(evidence_loader)
    ):
        raise TypeError(
            "phase evidence requires a typed candidate, runner, and bytes loader"
        )
    try:
        handles = runner(candidate)
        if type(handles) is not tuple:
            raise TypeError("phase evidence runner must return a tuple")
        for handle in handles:
            _verify_bound_evidence(candidate, handle, evidence_loader)
        return PhaseEvidenceResult(candidate=candidate, handles=handles)
    except Exception:
        reject_phase_candidate(candidate)
        raise


def validate_phase_candidate(
    candidate: CandidatePlan,
    evidence: PhaseEvidenceResult,
    validator: Callable[
        [CandidatePlan, PhaseEvidenceResult], tuple[EvidenceHandle, ...]
    ],
    evidence_loader: Callable[[CandidatePlan, EvidenceHandle], bytes],
) -> ValidatedCandidate:
    if (
        not isinstance(candidate, CandidatePlan)
        or not isinstance(evidence, PhaseEvidenceResult)
        or evidence.candidate != candidate
        or not callable(validator)
        or not callable(evidence_loader)
    ):
        raise TypeError("candidate validation inputs are not strictly bound")
    try:
        handles = validator(candidate, evidence)
        if type(handles) is not tuple:
            raise TypeError("phase validator must return a tuple")
        for handle in handles:
            _verify_bound_evidence(candidate, handle, evidence_loader)
        return ValidatedCandidate(evidence=evidence, validation_handles=handles)
    except Exception:
        reject_phase_candidate(candidate)
        raise


def _publish_phase_candidate_held(
    validated: ValidatedCandidate,
    *,
    authority_validator: Callable[[Any, int], bool],
    pre_cas_validator: Callable[[ValidatedCandidate], bool],
    hooks: TransitionHooks = _NO_TRANSITION_HOOKS,
) -> PublicationResult:
    """Revalidate lease/ref/callback and CAS the exact target ref."""

    if (
        not isinstance(validated, ValidatedCandidate)
        or not callable(authority_validator)
        or not callable(pre_cas_validator)
    ):
        raise TypeError("publication requires a validated candidate and callback")
    candidate = validated.evidence.candidate
    request = candidate.request
    now_ns = time.time_ns()
    if not (
        request.authority.issued_at_ns <= now_ns < request.authority.expires_at_ns
        and authority_validator(request.authority, now_ns) is True
    ):
        raise ProtectedAcceptanceDenied(
            "phase authority expired, rotated, or was revoked before publication"
        )
    _reject_ambient_git_environment()
    repo_root = Path(request.repository.root)
    common_dir = _common_git_directory(repo_root)
    lease = _validate_lease(candidate, common_dir)
    _require_clean_exact_checkout(request)
    if hooks.before_cas is not None:
        hooks.before_cas(candidate)
    if pre_cas_validator(validated) is not True:
        raise ProtectedAcceptanceDenied(
            "pre-CAS candidate validation did not return literal True"
        )
    lease = _validate_lease(candidate, common_dir)
    _require_clean_exact_checkout(request)
    if request.dry_run:
        if not _release_lease(lease, common_dir):
            raise ProtectedTransitionRace("dry-run lease was replaced before release")
        return PublicationResult(
            candidate=candidate,
            old_commit=request.parent_commit,
            new_commit=candidate.commit_id,
            published=False,
            dry_run=True,
        )
    if _resolve_one(repo_root, candidate.rescue_ref) != candidate.commit_id:
        raise ProtectedTransitionRace("checked rescue ref was lost before publication")
    target_root = _target_worktree(repo_root, request.repository.target_ref)
    if target_root is None:
        _checked_update_ref(
            repo_root,
            request.repository.target_ref,
            candidate.commit_id,
            request.parent_commit,
        )
    else:
        _require_clean_target_worktree(
            target_root, request.repository.target_ref, request.parent_commit
        )
        _run_git(
            target_root,
            ("merge", "--ff-only", "--no-edit", candidate.commit_id),
        )
        if (
            _resolve_one(target_root, request.repository.target_ref)
            != candidate.commit_id
            or _resolve_one(target_root, "HEAD") != candidate.commit_id
            or _resolve_tree(target_root, "HEAD") != candidate.tree_id
        ):
            raise ProtectedTransitionRace(
                "checked-out target fast-forward was incomplete"
            )
        _require_clean_contents(target_root, candidate.commit_id)
    if hooks.after_cas is not None:
        hooks.after_cas(candidate)
    _checked_delete_ref(repo_root, candidate.rescue_ref, candidate.commit_id)
    if not _release_lease(lease, common_dir):
        raise ProtectedTransitionRace(
            "publication succeeded but lease release was fenced"
        )
    return PublicationResult(
        candidate=candidate,
        old_commit=request.parent_commit,
        new_commit=candidate.commit_id,
        published=True,
        dry_run=False,
    )


def publish_phase_candidate(
    validated: ValidatedCandidate,
    *,
    authority_validator: Callable[[Any, int], bool],
    pre_cas_validator: Callable[[ValidatedCandidate], bool],
    hooks: TransitionHooks = _NO_TRANSITION_HOOKS,
) -> PublicationResult:
    """Publish transactionally, settling ordinary callback and Git failures."""

    if (
        not isinstance(validated, ValidatedCandidate)
        or not callable(authority_validator)
        or not callable(pre_cas_validator)
    ):
        raise TypeError("publication requires a validated candidate and callback")
    candidate = validated.evidence.candidate
    request = candidate.request
    repo_root = Path(request.repository.root)
    common_dir = _common_git_directory(repo_root)
    if _resolve_one(repo_root, request.repository.target_ref) == candidate.commit_id:
        if _resolve_one(repo_root, candidate.rescue_ref) != candidate.commit_id:
            raise ProtectedTransitionRace(
                "published candidate lacks its exact pending-settlement rescue"
            )
        target_root = _target_worktree(repo_root, request.repository.target_ref)
        if target_root is not None:
            _require_clean_target_worktree(
                target_root, request.repository.target_ref, candidate.commit_id
            )
        try:
            lease = _validate_lease(candidate, common_dir)
        except ProtectedTransitionRace:
            lease = _acquire_canonical_lease(request, common_dir)
        _checked_delete_ref(repo_root, candidate.rescue_ref, candidate.commit_id)
        if not _release_lease(lease, common_dir):
            raise ProtectedTransitionRace(
                "published candidate settlement lost its exact lease"
            )
        return PublicationResult(
            candidate=candidate,
            old_commit=request.parent_commit,
            new_commit=candidate.commit_id,
            published=True,
            dry_run=False,
            settlement_pending=True,
        )
    try:
        return _publish_phase_candidate_held(
            validated,
            authority_validator=authority_validator,
            pre_cas_validator=pre_cas_validator,
            hooks=hooks,
        )
    except Exception as exc:
        if (
            _resolve_one(repo_root, request.repository.target_ref)
            == candidate.commit_id
        ):
            if _resolve_one(repo_root, candidate.rescue_ref) != candidate.commit_id:
                raise ProtectedTransitionRace(
                    "publication completed but settlement rescue was lost"
                ) from exc
            target_root = _target_worktree(repo_root, request.repository.target_ref)
            if target_root is not None:
                _require_clean_target_worktree(
                    target_root, request.repository.target_ref, candidate.commit_id
                )
            lease = _validate_lease(candidate, common_dir)
            _checked_delete_ref(repo_root, candidate.rescue_ref, candidate.commit_id)
            if not _release_lease(lease, common_dir):
                raise ProtectedTransitionRace(
                    "publication completed but settlement lost its exact lease"
                ) from exc
            return PublicationResult(
                candidate=candidate,
                old_commit=request.parent_commit,
                new_commit=candidate.commit_id,
                published=True,
                dry_run=False,
                settlement_pending=True,
            )
        try:
            reject_phase_candidate(candidate)
        except Exception as cleanup_exc:
            raise ProtectedTransitionRace(
                "publication failed and exact transaction settlement was fenced"
            ) from cleanup_exc
        raise


def _reject_phase_candidate_held(candidate: CandidatePlan) -> RejectionResult:
    """Reject or roll back only through checked rescue-ref CAS semantics."""

    if not isinstance(candidate, CandidatePlan):
        raise TypeError("rejection requires CandidatePlan")
    _reject_ambient_git_environment()
    request = candidate.request
    repo_root = Path(request.repository.root)
    common_dir = _common_git_directory(repo_root)
    try:
        lease = _validate_lease(candidate, common_dir)
    except ProtectedTransitionRace:
        # Publication normally releases its construction lease.  A later
        # operator rejection must reacquire the same canonical merge namespace
        # before using the retained rescue ref; replacement/live ownership is
        # still fail-closed in _acquire_canonical_lease.
        lease = _acquire_canonical_lease(request, common_dir)
    target = _resolve_one(repo_root, request.repository.target_ref)
    rolled_back = False
    rescue_deleted = False
    if target == candidate.commit_id:
        raise ProtectedAcceptanceDenied(
            "published candidates are terminal and cannot be rejected or rolled back"
        )
    elif target != request.parent_commit:
        raise ProtectedTransitionRace(
            "target ref is neither candidate nor checked parent"
        )
    if not request.dry_run:
        _checked_delete_ref(repo_root, candidate.rescue_ref, candidate.commit_id)
        rescue_deleted = True
    released = _release_lease(lease, common_dir)
    if not released:
        raise ProtectedTransitionRace(
            "candidate rejected but checkout lease was replaced"
        )
    return RejectionResult(
        candidate=candidate,
        target_rolled_back=rolled_back,
        rescue_ref_deleted=rescue_deleted,
        lease_released=True,
    )


def reject_phase_candidate(candidate: CandidatePlan) -> RejectionResult:
    """Settle rejection while never abandoning a lease on ordinary failure."""

    if not isinstance(candidate, CandidatePlan):
        raise TypeError("rejection requires CandidatePlan")
    repo_root = Path(candidate.request.repository.root)
    if (
        _resolve_one(repo_root, candidate.request.repository.target_ref)
        == candidate.commit_id
    ):
        raise ProtectedAcceptanceDenied(
            "published candidates are terminal and cannot be rejected or rolled back"
        )
    try:
        return _reject_phase_candidate_held(candidate)
    except Exception as exc:
        common_dir = _common_git_directory(Path(candidate.request.repository.root))
        lock_path = common_dir / DEFAULT_CHECKOUT_MUTATION_LOCK_NAME
        observed = read_checkout_mutation_lease(lock_path)
        if observed is not None:
            metadata = observed.metadata
            if not (
                observed.lease_id == candidate.lease_id
                and observed.device == candidate.lease_device
                and observed.inode == candidate.lease_inode
                and metadata.get("operation") == "protected_acceptance_transition"
                and metadata.get("worktree_root") == candidate.request.repository.root
                and metadata.get("branch") == candidate.request.repository.target_ref
            ):
                raise ProtectedTransitionRace(
                    "rejection failed under a foreign checkout lease"
                ) from exc
            if not _release_lease(observed, common_dir):
                raise ProtectedTransitionRace(
                    "rejection failed and exact lease settlement was fenced"
                ) from exc
        raise


__all__ = (
    "ProtectedTransitionGitError",
    "ProtectedTransitionRace",
    "TransitionHooks",
    "build_phase_candidate",
    "publish_phase_candidate",
    "reject_phase_candidate",
    "resolve_prompt_v3_birth_target_once",
    "run_phase_evidence",
    "validate_phase_candidate",
)
