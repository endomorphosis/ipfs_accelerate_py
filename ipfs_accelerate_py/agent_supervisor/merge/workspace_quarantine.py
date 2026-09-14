"""Freeze an ambiguous board workspace area; independent work uses a fresh area.

The scope is explicitly the whole configured root, not an inferred association
between a missing Portal state and one historical workspace. Native mutation
readers share a repository lock; installing the freeze takes it exclusively.
"""

from __future__ import annotations

import fcntl
from fnmatch import fnmatchcase
import hashlib
from itertools import islice
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .quarantine_validation import require, strict_json, QuarantineDenied
from ..task_sources.control_plane_contracts import (
    content_identity,
    canonical_json_bytes,
)

SCHEMA = "ipfs_accelerate_py/agent-supervisor/workspace-root-quarantine@1"
MAX_FILES = 8192
MAX_BYTES = 16_777_216
MAX_RECORDS = 256


def registry(repo_root: Path) -> Path:
    from .worktree_lifecycle import lifecycle_store_dir

    return lifecycle_store_dir(Path(repo_root)).parent / "agent-workspace-quarantine"


@contextmanager
def directory_guard(directory: Path, *, exclusive: bool = False):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.geteuid()
        and not info.st_mode & 0o022,
        "workspace_quarantine_registry_unowned",
    )
    path = directory / "mutation.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.geteuid()
            and info.st_nlink == 1,
            "workspace_quarantine_lock_unowned",
        )
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(
                    fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
                )
                break
            except BlockingIOError:
                require(
                    time.monotonic() < deadline, "workspace_quarantine_mutation_busy"
                )
                time.sleep(0.01)
        yield directory
    finally:
        os.close(fd)


@contextmanager
def guard(repo_root: Path, *, exclusive: bool = False):
    with directory_guard(registry(repo_root), exclusive=exclusive) as directory:
        yield directory


CLAIM_DIRS = {"implementation-resource-claims", "implementation-task-claims"}


@contextmanager
def claim_update(lock_path: Path):
    """Compose the existing claim CAS with the repository custody lock."""
    path = Path(lock_path).parent.resolve() / Path(lock_path).name
    if path.parent.name not in CLAIM_DIRS:
        yield
        return
    with directory_guard(
        path.parent.parent / "agent-workspace-quarantine"
    ) as directory:
        for record in records(directory):
            require(
                not any(
                    row["path"] == str(path) for row in record["snapshot"]["files"]
                ),
                "workspace_claim_quarantined",
            )
        yield


def claim_retained(repo_root: Path, metadata: dict[str, Any]) -> bool:
    """A frozen native claim remains occupied after its owner dies."""
    with guard(repo_root) as directory:
        for record in records(directory):
            for row in record["snapshot"]["files"]:
                path = Path(row["path"])
                if path.parent.name not in CLAIM_DIRS:
                    continue
                raw, info = read_regular(path)
                require(
                    hashlib.sha256(raw).hexdigest() == row["sha256"]
                    and info.st_dev == row["device"]
                    and info.st_ino == row["inode"],
                    "workspace_quarantine_claim_changed",
                )
                if strict_json(raw.decode()) == metadata:
                    return True
    return False


def read_regular(path: Path, *, bound: int = MAX_BYTES) -> tuple[bytes, Any]:
    # Opening a FIFO must not block before fstat can reject its type.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.geteuid()
            and before.st_nlink == 1
            and 0 <= before.st_size <= bound,
            "workspace_quarantine_file_invalid",
        )
        raw = b""
        while len(raw) <= bound:
            chunk = os.read(fd, min(65536, bound + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(fd)
        require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            and len(raw) == before.st_size,
            "workspace_quarantine_file_changed",
        )
        return raw, before
    finally:
        os.close(fd)


def bounded_entries(directory: Path, *, bound: int, reason: str) -> list[Path]:
    # Path.glob/iterdir may materialize a whole directory internally. Bound the
    # native scanner itself, including unmatched names, before filtering.
    with os.scandir(directory) as entries:
        paths = [Path(entry.path) for entry in islice(entries, bound + 1)]
    require(len(paths) <= bound, reason)
    return paths


def records(directory: Path) -> list[dict[str, Any]]:
    paths = [
        path
        for path in bounded_entries(
            directory,
            bound=MAX_RECORDS + 1,  # The native mutation.lock has its own slot.
            reason="workspace_quarantine_registry_bound",
        )
        if fnmatchcase(path.name, "*.json")
    ]
    require(len(paths) <= MAX_RECORDS, "workspace_quarantine_registry_bound")
    result = []
    for path in sorted(paths):
        raw, _ = read_regular(path, bound=262144)
        value = strict_json(raw.decode())
        require(
            type(value) is dict
            and set(value) == {"schema", "root", "fresh_root", "snapshot", "cid"}
            and value["schema"] == SCHEMA
            and value["cid"]
            == content_identity({key: value[key] for key in value if key != "cid"})
            and type(value["root"]) is str
            and type(value["fresh_root"]) is str
            and path.stem == hashlib.sha256(value["root"].encode()).hexdigest(),
            "workspace_quarantine_record_invalid",
        )
        root, fresh = Path(value["root"]), Path(value["fresh_root"])
        require(
            root.is_absolute()
            and root.resolve() == root
            and fresh
            == root.parent
            / (
                "independent-workspaces-"
                + hashlib.sha256(str(root).encode()).hexdigest()[:24]
            )
            and fresh.resolve() == fresh,
            "workspace_quarantine_root_invalid",
        )
        snapshot = value["snapshot"]
        require(
            type(snapshot) is dict
            and set(snapshot) == {"schema", "root", "files"}
            and snapshot["schema"] == SCHEMA + "/census"
            and snapshot["root"] == str(root)
            and type(snapshot["files"]) is list
            and len(snapshot["files"]) <= MAX_FILES,
            "workspace_quarantine_snapshot_invalid",
        )
        seen, total = set(), 0
        for row in snapshot["files"]:
            require(
                type(row) is dict
                and set(row) == {"path", "sha256", "size", "device", "inode"}
                and type(row["path"]) is str
                and Path(row["path"]).is_absolute()
                and row["path"] not in seen
                and type(row["sha256"]) is str
                and len(row["sha256"]) == 64
                and all(char in "0123456789abcdef" for char in row["sha256"])
                and all(
                    type(row[key]) is int and row[key] >= 0
                    for key in ("size", "device", "inode")
                ),
                "workspace_quarantine_snapshot_invalid",
            )
            seen.add(row["path"])
            total += row["size"]
        require(total <= MAX_BYTES, "workspace_quarantine_snapshot_bound")
        result.append(value)
    return result


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_unfenced(directory: Path, workspace: Path) -> None:
    target = workspace.resolve()
    for record in records(directory):
        retained = Path(record["root"])
        # A recursive mutation of an ancestor can also destroy retained state.
        require(
            not within(target, retained) and not within(retained, target),
            "workspace_root_quarantined",
        )


@contextmanager
def mutation(repo_root: Path, workspace: Path):
    with repository_guards(repo_root) as directories:
        for directory in directories:
            require_unfenced(directory, Path(workspace))
        yield


def census(repo_root: Path, worktree_root: Path) -> dict[str, Any]:
    from .worktree_lifecycle import (
        WorktreeLifecycleStore, WorkspaceLifecycleRecord, WorktreeLifecycleError,
        _canonical_json_bytes,
    )

    root = worktree_root.resolve()
    pool = root / ".pool-state"
    lifecycle = WorktreeLifecycleStore(repo_root=repo_root)
    installed = (
        [item for item in records(registry(repo_root)) if item["root"] == str(root)]
        if registry(repo_root).exists()
        else []
    )
    require(len(installed) <= 1, "workspace_quarantine_scope_duplicate")
    held_claims = (
        {
            row["path"]
            for row in installed[0]["snapshot"]["files"]
            if Path(row["path"]).parent.name in CLAIM_DIRS
        }
        if installed
        else None
    )
    candidates = []
    remaining_files, remaining_bytes = MAX_FILES, MAX_BYTES
    claim_directories = tuple(
        (registry(repo_root).parent / name, "*.lock") for name in sorted(CLAIM_DIRS)
    )
    for directory, pattern in (
        (pool, "*"),
        (lifecycle.store_dir, "*.json"),
        *claim_directories,
    ):
        try:
            info = directory.lstat()
        except FileNotFoundError:
            continue
        require(stat.S_ISDIR(info.st_mode), "workspace_quarantine_directory_invalid")
        paths = bounded_entries(
            directory,
            bound=remaining_files,
            reason="workspace_quarantine_population_bound",
        )
        remaining_files -= len(paths)
        for path in paths:
            if not fnmatchcase(path.name, pattern):
                continue
            if path.name.startswith(".") and path.name.endswith(".update.lock"):
                continue
            raw, info = read_regular(path, bound=remaining_bytes)
            remaining_bytes -= len(raw)
            if directory == pool:
                # Pool metadata and existing ownership sidecars are retained.
                require(
                    path.suffix in {".json", ".lock"},
                    "workspace_quarantine_pool_entry_unknown",
                )
                value = strict_json(raw.decode())
                require(type(value) is dict, "workspace_quarantine_pool_entry_invalid")
                if path.suffix == ".json":
                    require(
                        value.get("lease_token") == path.stem
                        and type(value.get("path")) is str
                        and within(Path(value["path"]).resolve(), root),
                        "workspace_quarantine_pool_binding_invalid",
                    )
            elif directory.name in CLAIM_DIRS:
                value = strict_json(raw.decode())
                require(type(value) is dict, "workspace_quarantine_claim_invalid")
                owner_root = value.get("worktree_root") or value.get("repo_root")
                require(
                    type(owner_root) is str and bool(owner_root),
                    "workspace_quarantine_claim_root_unknown",
                )
                if Path(owner_root).resolve() != Path(repo_root).resolve():
                    continue
                require(
                    type(value.get("lease_id")) is str
                    and bool(value["lease_id"])
                    and type(value.get("canonical_task_cid")) is str
                    and bool(value["canonical_task_cid"]),
                    "workspace_quarantine_claim_binding_unknown",
                )
                if directory.name == "implementation-resource-claims":
                    require(
                        type(value.get("resource_path")) is str
                        and bool(value["resource_path"]),
                        "workspace_quarantine_claim_resource_unknown",
                    )
                if held_claims is not None and str(path) not in held_claims:
                    continue
            else:
                value = strict_json(raw.decode())
                if path.name.startswith("quarantine-"):
                    try:
                        receipt = lifecycle._load_strict_quarantine_payload(
                            None, receipt_path=path,
                        )
                    except WorktreeLifecycleError as exc:
                        raise QuarantineDenied(
                            "workspace_quarantine_lifecycle_quarantine_invalid"
                        ) from exc
                    require(
                        receipt is not None
                        and raw == _canonical_json_bytes(receipt),
                        "workspace_quarantine_lifecycle_quarantine_changed",
                    )
                    verified_raw, verified_info = read_regular(path, bound=len(raw))
                    require(
                        verified_raw == raw
                        and all(
                            getattr(verified_info, field) == getattr(info, field)
                            for field in (
                                "st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
                                "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns",
                            )
                        ),
                        "workspace_quarantine_lifecycle_quarantine_changed",
                    )
                    value = receipt["lifecycle_record"]
                require(
                    type(value) is dict and type(value.get("workspace_path")) is str,
                    "workspace_quarantine_lifecycle_entry_invalid",
                )
                if not within(Path(value["workspace_path"]).resolve(), root):
                    continue
                if path.name.startswith("ws-"):
                    record = WorkspaceLifecycleRecord.from_dict(value)
                    require(
                        record.record_id == record.compute_record_id(),
                        "workspace_quarantine_lifecycle_identity_invalid",
                    )
            candidates.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                }
            )
    require(
        len(candidates) <= MAX_FILES
        and sum(row["size"] for row in candidates) <= MAX_BYTES,
        "workspace_quarantine_population_bound",
    )
    return {
        "schema": SCHEMA + "/census",
        "root": str(root),
        "files": sorted(candidates, key=lambda item: item["path"]),
    }


def plan(repo_root: Path, worktree_root: Path) -> dict[str, Any]:
    root = worktree_root.resolve()
    key = hashlib.sha256(str(root).encode()).hexdigest()
    fresh = root.parent / ("independent-workspaces-" + key[:24])
    value = {
        "schema": SCHEMA,
        "root": str(root),
        "fresh_root": str(fresh),
        "snapshot": census(repo_root, root),
    }
    value["cid"] = content_identity(value)
    return value


def freeze(
    repo_root: Path, worktree_root: Path, *, expected: dict[str, Any]
) -> dict[str, Any]:
    root = worktree_root.resolve()
    key = hashlib.sha256(str(root).encode()).hexdigest()
    with guard(repo_root, exclusive=True) as directory:
        current = census(repo_root, root)
        require(current == expected, "workspace_quarantine_census_changed")
        installed = records(directory)
        prior = [record for record in installed if record["root"] == str(root)]
        if prior:
            require(
                len(prior) == 1 and prior[0]["snapshot"] == current,
                "workspace_quarantine_retained_scope_changed",
            )
            return prior[0]
        require(len(installed) < MAX_RECORDS, "workspace_quarantine_registry_bound")
        fresh = root.parent / ("independent-workspaces-" + key[:24])
        try:
            fresh.lstat()
        except FileNotFoundError:
            pass
        else:
            raise QuarantineDenied("workspace_quarantine_fresh_root_preexists")
        value = {
            "schema": SCHEMA,
            "root": str(root),
            "fresh_root": str(fresh),
            "snapshot": current,
        }
        value["cid"] = content_identity(value)
        raw = canonical_json_bytes(value) + b"\n"
        require(len(raw) <= 262144, "workspace_quarantine_record_bound")
        fd, name = tempfile.mkstemp(prefix=".freeze-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, directory / (key + ".json"))
            parent_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            Path(name).unlink(missing_ok=True)
        return value


def verify(repo_root: Path, worktree_root: Path) -> dict[str, Any]:
    with guard(repo_root) as directory:
        matches = [
            record
            for record in records(directory)
            if record["root"] == str(worktree_root.resolve())
        ]
        require(len(matches) == 1, "workspace_quarantine_scope_missing")
        require(
            matches[0]["snapshot"] == census(repo_root, worktree_root),
            "workspace_quarantine_custody_changed",
        )
        return matches[0]


def mutation_boundary(*workspace_names: str, pool: bool = False):
    """Guard an existing native operation, including its Git/sidecar effects."""
    import functools
    import inspect
    from contextlib import ExitStack

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def protected(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            owner = bound.arguments.get("self")
            repo_root = (
                getattr(owner, "repo_root", None)
                if owner is not None
                else bound.arguments.get("repo_root")
            )
            if repo_root is None and owner is not None:
                repo_root = getattr(getattr(owner, "config", None), "repo_root", None)
            require(repo_root is not None, "workspace_mutation_root_unbound")
            if pool:
                paths = [owner.worktree_root]
            else:
                paths = [
                    bound.arguments[name]
                    for name in workspace_names
                    if bound.arguments.get(name) is not None
                ]
            if not paths:
                with repository_guards(Path(repo_root)) as directories:
                    require(
                        not any(records(directory) for directory in directories),
                        "workspace_mutation_path_unbound",
                    )
                    return function(*args, **kwargs)
            with ExitStack() as stack:
                for path in paths:
                    stack.enter_context(mutation(Path(repo_root), Path(path)))
                return function(*args, **kwargs)

        return protected

    return decorate


@contextmanager
def repository_guards(repo_root: Path):
    """Retain each enclosing Git store's custody lock, outermost first.

    A nested supervisor may mutate a workspace inside a root frozen by its
    superproject. All mutation paths must share that parent freeze lock, even
    when their own registry has no retained workspaces.
    """
    from contextlib import ExitStack

    common = registry(Path(repo_root)).parent.resolve()
    roots = [
        common,
        *(
            parent
            for parent in common.parents
            if (parent / "HEAD").is_file() and (parent / "objects").is_dir()
        ),
    ]
    with ExitStack() as stack:
        directories = [
            stack.enter_context(directory_guard(root / "agent-workspace-quarantine"))
            for root in sorted(roots, key=lambda item: len(item.parts))
        ]
        yield directories


@contextmanager
def maintenance(repo_root: Path):
    """Hold root and submodule custody guards throughout a global Git operation."""
    with repository_guards(repo_root) as directories:
        yield not any(records(directory) for directory in directories)


def maintenance_boundary(*, count_result: bool = False):
    """Defer repository-wide maintenance under any ancestor repository custody.

    Submodule Git common directories live below their superproject common
    directory. Pruning them must participate in the same root freeze lock.
    """
    import functools

    def decorate(function):
        @functools.wraps(function)
        def protected(self, *args, **kwargs):
            repo_root = getattr(self, "repo_root", None)
            if repo_root is None:
                repo_root = getattr(getattr(self, "config", None), "repo_root", None)
            require(repo_root is not None, "workspace_maintenance_root_unbound")
            with maintenance(Path(repo_root)) as allowed:
                if not allowed:
                    return (
                        0
                        if count_result
                        else {
                            "attempted": False,
                            "skipped": True,
                            "reason": "retained_workspace_scope",
                        }
                    )
                return function(self, *args, **kwargs)

        return protected

    return decorate
