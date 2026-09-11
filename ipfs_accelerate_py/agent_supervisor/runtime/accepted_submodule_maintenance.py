"""Native preflight maintenance for accepted, non-runtime dependency gitlinks.

Ordinary maintenance never quarantines locks or settles callbacks. The configured
operator holds its resume/owner/mutation guards around this call. Every actual
source transition additionally takes the canonical checkout merge lease. An
owner need not be running: no owner, task, or callback admission is performed;
normal resume retains its later owner recovery and authentication boundaries.
"""

from __future__ import annotations
import json
import os
import stat
from pathlib import Path
import time
import uuid

from ..merge import accepted_submodule_sync as sync


def public_custody_census(dataset, *, own_pid=None):
    """Positive canonical holders veto; unrelated unavailable /proc stays unknown."""
    own_pid = os.getpid() if own_pid is None else own_pid
    positives, unavailable = [], []
    prefix = str(dataset) + "/"
    expected_owned_git = [
        "git",
        "-C",
        str(dataset),
        "update-ref",
        "--stdin",
        "-m",
        "accepted submodule fast-forward",
    ]

    def canonical(value):
        return value == str(dataset) or value.startswith(prefix)

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            before = (entry / "stat").read_text()
            fields = before[before.rfind(")") + 2 :].split()
            if fields[0] in ("Z", "X"):
                continue
            argv = [
                x.decode() for x in (entry / "cmdline").read_bytes().split(b"\0") if x
            ]
            # Only our own prepared Git HEAD transaction is expected inside the
            # canonical checkout during revalidation; no worker child is exempt.
            if int(fields[1]) == own_pid and argv == expected_owned_git:
                continue
            held = []
            if canonical(os.readlink(entry / "cwd")):
                held.append("cwd")
            try:
                for descriptor in (entry / "fd").iterdir():
                    try:
                        if canonical(os.readlink(descriptor)):
                            held.append("fd:" + descriptor.name)
                    except FileNotFoundError:
                        continue
            except PermissionError:
                unavailable.append(int(entry.name))
            # A Git process naming this exact canonical checkout also vetoes,
            # even before it has changed cwd or opened an index descriptor.
            if Path(argv[0]).name == "git" and str(dataset) in argv:
                held.append("git_argv")
            after = (entry / "stat").read_text()
            current = after[after.rfind(")") + 2 :].split()
            if current[19] != fields[19]:
                unavailable.append(int(entry.name))
                continue
            if held:
                positives.append(
                    {
                        "pid": int(entry.name),
                        "birth": int(fields[19]),
                        "references": held,
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError, ValueError, UnicodeError, IndexError):
            unavailable.append(int(entry.name))
    return {
        "positive_holders": positives,
        "unavailable_unrelated_pids": sorted(set(unavailable)),
        "global_absence_claimed": False,
        "historical_callback_closure_claimed": False,
    }


def _private_directory(path, *, repo_root):
    path = Path(path).absolute()
    sync.require(
        ".." not in path.parts and not path.is_relative_to(repo_root),
        "maintenance_archive_inside_checkout",
    )
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
                os.fsync(fd)
            except FileExistsError:
                pass
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=fd,
            )
            os.close(fd)
            fd = child
        sync.require(
            not Path(os.readlink(f"/proc/self/fd/{fd}")).is_relative_to(repo_root),
            "maintenance_archive_inside_checkout",
        )
    finally:
        os.close(fd)


def _journal(path, report):
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    with sync.directory(path.parent) as parent:
        temporary = path.name + ".tmp-" + uuid.uuid4().hex
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)


def assess_accepted_configured_submodule(board):
    """Assess one ordinary accepted dependency mismatch without filesystem effects.

    This trusted configured-operator entry constructs its own fresh preflight;
    no supplied status/projection grants maintenance authority. All unrelated
    admission failures deny before archive or source effects. Multiple mismatches
    remain explicit rather than weakening the single-checkout preservation gate.
    This source operation does not admit an owner. The native caller retains
    its ordinary post-maintenance owner recovery/authentication gates, including
    when the incumbent owner is legitimately stopped.
    """
    from .configured_board_scheduler import preflight_configured_board

    root = Path(board.repo_root)
    before = preflight_configured_board(board)
    sync.require(
        before.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/configured-board-preflight@1"
        and before.get("repo_root") == str(root)
        and before.get("config_path") == str(board.config_path)
        and before.get("board_namespace") == board.board_namespace,
        "maintenance_preflight_scope_changed",
    )
    if before.get("valid") is True:
        return {
            "needed": False,
            "reason": "configured_source_already_aligned",
            "preflight": before,
        }
    checks = before.get("checks", [])
    failed = [item for item in checks if item.get("passed") is not True]
    sync.require(
        failed
        and {item.get("name") for item in failed}
        == {"checkout_clean", "configured_submodules"}
        and before.get("errors")
        == [f"{item['name']}: {item['detail']}" for item in failed],
        "non_submodule_preflight_failure",
    )
    source_binding = board.payload.get("source_binding", {})
    allowed = {
        source_binding.get(key + "_submodule_path")
        for key in ("ipfs_datasets", "ipfs_kit")
    }
    allowed.discard(None)
    sync.require(
        source_binding.get("ipfs_accelerate_submodule_path") not in allowed,
        "executing_runtime_in_maintenance_scope",
    )
    submodules = next(
        item["detail"] for item in failed if item["name"] == "configured_submodules"
    )
    drift = [item for item in submodules if item.get("valid") is not True]
    sync.require(len(drift) == 1, "single_dependency_mismatch_required")
    row = drift[0]
    relative = row.get("path")
    sync.require(
        relative in allowed
        and relative in board.worktree_submodule_paths
        and row.get("exact_worktree") is True
        and row.get("planning_revision_is_ancestor") is True
        and row.get("dirty") == []
        and row.get("head") != row.get("gitlink"),
        "dependency_mismatch_not_admitted",
    )
    parent = sync.git_text(root, "rev-parse", "HEAD")
    snapshot = sync.accepted_submodule_snapshot(
        root, relative, expected_parent=parent, expected_old=row["head"]
    )
    sync.require(
        snapshot["target"] == row["gitlink"], "accepted_dependency_target_changed"
    )
    return {
        "needed": True,
        "parent": parent,
        "row": row,
        "snapshot": snapshot,
        "config_sha256": sync.digest(sync.read_regular(board.config_path)[0]),
    }


def maintain_accepted_configured_submodule(board, *, archive_root, custody_guard):
    """Reassess freshly under the caller's native guards before source effects."""
    from .configured_board_scheduler import preflight_configured_board

    assessment = assess_accepted_configured_submodule(board)
    if not assessment["needed"]:
        return {
            "changed": False,
            "reason": assessment["reason"],
            "preflight": assessment["preflight"],
        }
    root = Path(board.repo_root)
    row = assessment["row"]
    relative = row["path"]
    parent = assessment["parent"]
    snapshot = assessment["snapshot"]
    config_hash = assessment["config_sha256"]

    def guard():
        sync.require(
            sync.digest(sync.read_regular(board.config_path)[0]) == config_hash,
            "configured_maintenance_config_changed",
        )
        custody_guard()
        observed = public_custody_census(root / relative)
        sync.require(
            not observed["positive_holders"], "canonical_dependency_holder_present"
        )
        return observed

    observation = guard()
    sync.require(
        sync.accepted_submodule_snapshot(
            root, relative, expected_parent=parent, expected_old=row["head"]
        )
        == snapshot,
        "assessment_source_changed",
    )
    archive = (
        Path(archive_root)
        / board.board_namespace
        / ("accepted-submodule-" + uuid.uuid4().hex)
    )
    _private_directory(archive.parent, repo_root=root)
    # Never reuse a preexisting leaf, even if its random name collides. Ancestors
    # are no-follow locators; this exact new uid/mode/inode owns trial artifacts.
    with sync.directory(archive.parent) as parent_fd:
        os.mkdir(archive.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    with sync.directory(archive) as archive_fd:
        info = os.fstat(archive_fd)
        sync.require(
            info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700,
            "maintenance_archive_ownership_invalid",
        )
        archive_identity = (info.st_dev, info.st_ino)
    report = {
        "schema": "native/accepted-submodule-maintenance@1",
        "changed": False,
        "success": False,
        "parent": parent,
        "config_sha256": config_hash,
        "snapshot": snapshot,
        "phases": [],
        "initial_custody_observation": observation,
        "historical_callback_closure_claimed": False,
    }

    def phase(name, **detail):
        with sync.directory(archive) as archive_fd:
            info = os.fstat(archive_fd)
            sync.require(
                (info.st_dev, info.st_ino) == archive_identity
                and info.st_uid == os.geteuid()
                and stat.S_IMODE(info.st_mode) == 0o700,
                "maintenance_archive_ownership_changed",
            )
        report["phases"].append({"name": name, "at": time.time(), **detail})
        _journal(archive / "journal.json", report)

    try:
        phase("native_dependency_maintenance_admitted")
        sync.require(
            sync.accepted_submodule_snapshot(
                root, relative, expected_parent=parent, expected_old=row["head"]
            )
            == snapshot,
            "assessment_source_changed",
        )
        result = sync.synchronize_accepted_submodule(
            root,
            relative,
            expected_parent=parent,
            expected_old=row["head"],
            archive=archive,
            phase=phase,
            custody_guard=guard,
        )
        report["changed"] = True
        report["final_snapshot"] = result
        after = preflight_configured_board(board)
        sync.require(after.get("valid") is True, "post_maintenance_preflight_denied")
        sync.require(
            sync.accepted_submodule_snapshot(
                root, relative, expected_parent=parent, expected_old=row["gitlink"]
            )
            == result,
            "source_changed_during_maintenance_preflight",
        )
        guard()
        report["preflight"] = after
        report["success"] = True
        phase("native_dependency_maintenance_complete")
        return {
            "changed": True,
            "preflight": after,
            "journal": str(archive / "journal.json"),
        }
    except BaseException as error:
        report["error_type"] = type(error).__name__
        if isinstance(error, sync.Refused):
            report["reason"] = str(error)
        phase("maintenance_stopped_preserving_recorded_state")
        raise
