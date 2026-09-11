"""Bounded, same-user physical checkout admission; never task-state authority.

The retained kernel flock serializes cooperating native checkout creators. It
neither reserves disk blocks nor controls arbitrary hooks, tests or shell copies.
"""
from __future__ import annotations

import fcntl
import math
import os
import re
import selectors
import stat
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence


class CheckoutStorageDeferred(RuntimeError):
    def __init__(self, reason: str, **details: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class CheckoutStoragePolicy:
    minimum_available_bytes: int = 8 * 1024 ** 3
    minimum_available_inodes: int = 50_000
    lock_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in ("minimum_available_bytes", "minimum_available_inodes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CheckoutStorageDeferred("checkout_storage_policy_invalid", field=name)
        if (type(self.lock_timeout_seconds) not in (int, float)
                or not math.isfinite(self.lock_timeout_seconds) or self.lock_timeout_seconds <= 0):
            raise CheckoutStorageDeferred("checkout_storage_policy_invalid", field="lock_timeout_seconds")

    @classmethod
    def from_environment(cls) -> "CheckoutStoragePolicy":
        values = {}
        for field, variable in (("minimum_available_bytes", "IPFS_ACCELERATE_WORKTREE_MIN_FREE_BYTES"),
                                ("minimum_available_inodes", "IPFS_ACCELERATE_WORKTREE_MIN_FREE_INODES")):
            value = os.environ.get(variable)
            if value is not None:
                if not re.fullmatch(r"[0-9]+", value):
                    raise CheckoutStorageDeferred("checkout_storage_policy_invalid", field=variable)
                values[field] = int(value)
        return cls(**values)


_THREAD_LOCK = threading.RLock()
_LOCK_FD: int | None = None
_LOCK_DIRECTORY_FD: int | None = None
_LOCK_DEPTH = 0
_LOCK_ROOT = Path("/tmp") / f"ipfs-agent-supervisor-allocation-{os.getuid()}"


def _after_fork_child() -> None:
    global _THREAD_LOCK, _LOCK_FD, _LOCK_DIRECTORY_FD, _LOCK_DEPTH
    # Do not LOCK_UN: fork shares the parent's open-file description. Merely
    # close the child's copies so it cannot prolong or revoke parent custody.
    for descriptor in (_LOCK_FD, _LOCK_DIRECTORY_FD):
        if descriptor is not None:
            os.close(descriptor)
    _THREAD_LOCK = threading.RLock()
    _LOCK_FD = _LOCK_DIRECTORY_FD = None
    _LOCK_DEPTH = 0


os.register_at_fork(after_in_child=_after_fork_child)


def _validate_lock_identity() -> None:
    assert _LOCK_FD is not None and _LOCK_DIRECTORY_FD is not None
    directory = os.fstat(_LOCK_DIRECTORY_FD)
    current_directory = os.stat(_LOCK_ROOT, follow_symlinks=False)
    inode = os.fstat(_LOCK_FD)
    current = os.stat("allocation.lock", dir_fd=_LOCK_DIRECTORY_FD, follow_symlinks=False)
    if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.getuid()
            or stat.S_IMODE(directory.st_mode) != 0o700
            or (directory.st_dev, directory.st_ino) != (current_directory.st_dev, current_directory.st_ino)
            or not stat.S_ISREG(inode.st_mode) or inode.st_uid != os.getuid()
            or stat.S_IMODE(inode.st_mode) != 0o600 or inode.st_nlink != 1
            or (inode.st_dev, inode.st_ino) != (current.st_dev, current.st_ino)):
        raise CheckoutStorageDeferred("checkout_allocation_lock_identity_unavailable")


@contextmanager
def allocation_exclusion(policy: CheckoutStoragePolicy) -> Iterator[None]:
    """Retain one fixed per-user flock through nested preparation, with a deadline."""
    global _LOCK_FD, _LOCK_DIRECTORY_FD, _LOCK_DEPTH
    started = time.monotonic()
    if not _THREAD_LOCK.acquire(timeout=policy.lock_timeout_seconds):
        raise CheckoutStorageDeferred("checkout_allocation_busy")
    outer = _LOCK_DEPTH == 0
    try:
        try:
            if outer:
                _LOCK_ROOT.mkdir(mode=0o700, exist_ok=True)
                _LOCK_DIRECTORY_FD = os.open(_LOCK_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
                _LOCK_FD = os.open("allocation.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                                   0o600, dir_fd=_LOCK_DIRECTORY_FD)
                _validate_lock_identity()
                while True:
                    try:
                        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() - started >= policy.lock_timeout_seconds:
                            raise CheckoutStorageDeferred("checkout_allocation_busy")
                        time.sleep(0.05)
        except OSError as exc:
            raise CheckoutStorageDeferred("checkout_allocation_lock_unavailable", error=str(exc)) from exc
        _validate_lock_identity()
        _LOCK_DEPTH += 1
        try:
            yield
        finally:
            _LOCK_DEPTH -= 1
    finally:
        if outer:
            # Closing our last descriptor releases the kernel lock. Never
            # unlink the persistent inode or use PID/JSON text as lock custody.
            for descriptor in (_LOCK_FD, _LOCK_DIRECTORY_FD):
                if descriptor is not None:
                    os.close(descriptor)
            _LOCK_FD = _LOCK_DIRECTORY_FD = None
        _THREAD_LOCK.release()


def _git(repo: Path, *args: str, optional: bool = False, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False, input=input_bytes,
                                timeout=60, env={**os.environ, "GIT_NO_LAZY_FETCH": "1",
                                                 "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckoutStorageDeferred("checkout_estimate_unavailable", source=str(repo), error=str(exc)) from exc
    if optional and result.returncode == 1:
        return b""
    if result.returncode:
        raise CheckoutStorageDeferred("checkout_estimate_unavailable", source=str(repo),
                                      command=list(args), returncode=result.returncode)
    return result.stdout


def _commit(repo: Path, ref: str) -> str:
    value = _git(repo, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}").strip().decode("ascii")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise CheckoutStorageDeferred("checkout_commit_unavailable", source=str(repo))
    return value


def _common_dir(repo: Path) -> Path:
    path = Path(os.fsdecode(_git(repo, "rev-parse", "--git-common-dir").strip()))
    return (path if path.is_absolute() else repo / path).resolve(strict=True)


@contextmanager
def verified_branch_reference(repo: Path, branch: str, commit: str) -> Iterator[None]:
    """Hold Git's own verify/prepare lock; no reference value is changed.

    An existing-branch checkout must use --no-checkout then read-tree of the
    admitted commit: ordinary worktree add tries to rewrite the locked branch.
    """
    reference = branch if branch.startswith("refs/heads/") else "refs/heads/" + branch
    _git(repo, "check-ref-format", reference)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise CheckoutStorageDeferred("checkout_commit_unavailable", source=str(repo))
    try:
        process = subprocess.Popen(["git", "update-ref", "--stdin"], cwd=repo,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=0)
    except OSError as exc:
        raise CheckoutStorageDeferred("checkout_branch_reference_unavailable", reference=reference) from exc
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            for command, response in ((b"start\n", b"start: ok\n"),
                (f"verify {reference} {commit}\nprepare\n".encode(), b"prepare: ok\n")):
                try:
                    process.stdin.write(command)
                    process.stdin.flush()
                    if not selector.select(timeout=5) or os.read(process.stdout.fileno(), 64) != response:
                        raise CheckoutStorageDeferred("checkout_branch_reference_unavailable", reference=reference)
                except OSError as exc:
                    raise CheckoutStorageDeferred("checkout_branch_reference_unavailable", reference=reference) from exc
        yield
    finally:
        # This owned child received only start/verify/prepare. EOF aborts its
        # transaction; it cannot execute a callback or change a ref value.
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


@dataclass(frozen=True)
class GitCheckoutEstimate:
    source: Path
    destination: Path
    commit: str
    common_directory: Path
    blob_sizes: tuple[int, ...]
    directory_count: int
    gitlinks: tuple[tuple[str, str], ...]
    checkout_expansion: int = 1

    @property
    def blob_bytes(self) -> int:
        return sum(self.blob_sizes)

    @property
    def file_count(self) -> int:
        return len(self.blob_sizes)


def estimate_checkout(repo: Path, destination: Path, ref: str) -> GitCheckoutEstimate:
    repo = repo.resolve(strict=True)
    commit = _commit(repo, ref)
    sizes = []
    file_paths = []
    directories = {"."}
    gitlinks = []
    # A smudge/process filter, custom encoding or executable checkout hook can
    # allocate far more than the stored blob. Do not call that a known estimate
    # or silently disable the repository's behavior to get an admission.
    hook_path = Path(os.fsdecode(_git(repo, "rev-parse", "--git-path", "hooks/post-checkout").strip()))
    hook = hook_path if hook_path.is_absolute() else repo / hook_path
    if hook.is_file() and os.access(hook, os.X_OK):
        raise CheckoutStorageDeferred("checkout_expansion_unbounded", source=str(repo))
    expansion = 1
    for setting in ("core.autocrlf", "core.eol"):
        if _git(repo, "config", "--get", setting, optional=True).strip().lower() in {b"true", b"crlf"}:
            expansion = 2
    for record in _git(repo, "ls-tree", "-r", "-l", "-z", commit).split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, object_id, size = metadata.split()
            path = PurePosixPath(os.fsdecode(raw_path))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe tree path")
            directories.update(str(parent) for parent in path.parents)
            if kind == b"blob" and mode in {b"100644", b"100755", b"120000"}:
                value = int(size)
                if value < 0:
                    raise ValueError("negative blob size")
                sizes.append(value)
                file_paths.append(raw_path)
            elif kind == b"commit" and mode == b"160000":
                directories.add(str(path))
                gitlinks.append((str(path), object_id.decode("ascii")))
            else:
                raise ValueError("unrecognized Git tree entry")
        except (ValueError, UnicodeError) as exc:
            raise CheckoutStorageDeferred("checkout_estimate_unavailable", source=str(repo), error=str(exc)) from exc
    if file_paths:
        # Ask Git about the exact immutable tree, including its real global and
        # info attributes. Parsing .gitattributes ourselves would miss macros,
        # overrides and configured attribute files. Unsupported Git must refuse.
        attributes = _git(repo, "check-attr", "--source=" + commit, "-z", "--stdin",
                          "filter", "working-tree-encoding", "eol", "ident",
                          input_bytes=b"\0".join(file_paths) + b"\0").split(b"\0")
        if attributes[-1] != b"" or len(attributes[:-1]) != len(file_paths) * 12:
            raise CheckoutStorageDeferred("checkout_attributes_unavailable", source=str(repo))
        for offset in range(0, len(attributes) - 1, 3):
            file_path, attribute, value = attributes[offset:offset + 3]
            if attribute in {b"filter", b"working-tree-encoding"} and value not in {b"unspecified", b"unset"}:
                raise CheckoutStorageDeferred("checkout_expansion_unbounded", source=str(repo),
                                              path=os.fsdecode(file_path), attribute=os.fsdecode(attribute))
            if attribute == b"eol" and value == b"crlf":
                expansion = max(expansion, 2)
            if attribute == b"ident" and value == b"set":
                expansion = max(expansion, (len(commit) + 10) // 4)
    return GitCheckoutEstimate(repo, destination, commit, _common_dir(repo), tuple(sizes),
                               len(directories), tuple(gitlinks), expansion)


def _dependency_source(parent: GitCheckoutEstimate, relative: str, commit: str) -> Path:
    candidates = [parent.source / relative]
    modules = _git(parent.source, "config", "-z", "--blob", parent.commit + ":.gitmodules",
                   "--get-regexp", r"^submodule\..*\.path$", optional=True)
    for record in modules.split(b"\0"):
        if not record:
            continue
        key, _, value = record.partition(b"\n")
        if os.fsdecode(value) == relative:
            name = os.fsdecode(key)[len("submodule."):-len(".path")]
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise CheckoutStorageDeferred("checkout_dependency_source_unavailable", path=relative)
            candidates.append(parent.common_directory / "modules" / name)
    for record in _git(parent.source, "worktree", "list", "--porcelain", "-z").split(b"\0"):
        if record.startswith(b"worktree "):
            candidates.append(Path(os.fsdecode(record[len(b"worktree "):])) / relative)
    for candidate in dict.fromkeys(candidates):
        try:
            if (candidate.is_dir() and _common_dir(candidate) != parent.common_directory
                    and _commit(candidate, commit) == commit):
                return candidate
        except (CheckoutStorageDeferred, OSError):
            continue
    raise CheckoutStorageDeferred("checkout_dependency_source_unavailable", path=relative, commit=commit)


def estimate_configured_checkouts(repo: Path, destination: Path, ref: str,
                                  dependencies: Sequence[str] = ()) -> tuple[GitCheckoutEstimate, ...]:
    primary = estimate_checkout(repo, destination, ref)
    grouped: dict[str, list[str]] = {}
    links = dict(primary.gitlinks)
    for raw in dependencies:
        path = PurePosixPath(raw)
        if not raw or path.is_absolute() or ".." in path.parts:
            raise CheckoutStorageDeferred("checkout_dependency_path_invalid", path=raw)
        relative = str(path)
        matches = [name for name in links if relative == name or relative.startswith(name + "/")]
        if not matches:
            raise CheckoutStorageDeferred("checkout_dependency_gitlink_unavailable", path=relative, commit=primary.commit)
        name = max(matches, key=len)
        grouped.setdefault(name, [])
        if relative != name:
            grouped[name].append(relative[len(name) + 1:])
    estimates = [primary]
    for relative, nested in sorted(grouped.items()):
        source = _dependency_source(primary, relative, links[relative])
        estimates.extend(estimate_configured_checkouts(source, destination / relative,
                         links[relative], nested))
    return tuple(estimates)


def _sample(path: Path) -> dict[str, int | str]:
    ancestor = path.absolute()
    try:
        while True:
            try:
                fd = os.open(ancestor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                break
            except FileNotFoundError:
                if ancestor == ancestor.parent:
                    raise
                ancestor = ancestor.parent
        try:
            device = os.fstat(fd).st_dev
            sample = os.fstatvfs(fd)
        finally:
            os.close(fd)
        if sample.f_frsize <= 0 or sample.f_bavail < 0 or sample.f_favail < 0:
            raise ValueError("unknown filesystem measurement")
        return {"path": str(path), "ancestor": str(ancestor), "device": device,
                "block_size": sample.f_frsize, "available_bytes": sample.f_bavail * sample.f_frsize,
                "available_inodes": sample.f_favail}
    except (OSError, ValueError) as exc:
        raise CheckoutStorageDeferred("checkout_storage_unavailable", path=str(path), error=str(exc)) from exc


def require_capacity(estimates: Sequence[GitCheckoutEstimate], policy: CheckoutStoragePolicy) -> None:
    allocations: dict[int, dict] = {}
    for estimate in estimates:
        for role, path in (("checkout", estimate.destination), ("git_store", estimate.common_directory)):
            sample = _sample(path)
            block = int(sample["block_size"])
            if role == "checkout":
                # Exact stored blob bytes plus filesystem block rounding,
                # directories and bounded Git administrative-file allowance.
                required_bytes = sum(max(1, (size * estimate.checkout_expansion + block - 1) // block) * block
                                     for size in estimate.blob_sizes)
                required_inodes = estimate.file_count + estimate.directory_count + 16
                required_bytes += (estimate.directory_count + 16) * block
            else:
                required_bytes = 8 * 1024 ** 2 + estimate.file_count * 512
                required_inodes = 128 + estimate.file_count // 100
            entry = allocations.setdefault(int(sample["device"]), {
                "available_bytes": int(sample["available_bytes"]), "available_inodes": int(sample["available_inodes"]),
                "required_bytes": policy.minimum_available_bytes, "required_inodes": policy.minimum_available_inodes,
                "paths": []})
            entry["available_bytes"] = min(entry["available_bytes"], int(sample["available_bytes"]))
            entry["available_inodes"] = min(entry["available_inodes"], int(sample["available_inodes"]))
            entry["required_bytes"] += required_bytes
            entry["required_inodes"] += required_inodes
            entry["paths"].append({"role": role, **sample})
    for device, entry in allocations.items():
        if entry["available_bytes"] < entry["required_bytes"]:
            raise CheckoutStorageDeferred("checkout_storage_bytes_low", device=device, **entry)
        if entry["available_inodes"] < entry["required_inodes"]:
            raise CheckoutStorageDeferred("checkout_storage_inodes_low", device=device, **entry)


@contextmanager
def checkout_allocation(*, repo_root: Path, destination: Path, ref: str,
                        dependency_paths: Sequence[str] = (),
                        policy: CheckoutStoragePolicy | None = None) -> Iterator[tuple[GitCheckoutEstimate, ...]]:
    policy = policy or CheckoutStoragePolicy.from_environment()
    with allocation_exclusion(policy):
        try:
            estimates = estimate_configured_checkouts(repo_root, destination, ref, dependency_paths)
            _validate_lock_identity()
            require_capacity(estimates, policy)
        except (OSError, ValueError) as exc:
            raise CheckoutStorageDeferred("checkout_estimate_unavailable", error=str(exc)) from exc
        yield estimates
