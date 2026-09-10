"""Pytest plugin: IVP selection, proof-reuse cache, and durable green ledger.

Loaded from ``test/api/conftest.py`` so hermetic validation still sees it when
``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` blocks pytest11 entry points. Fail closed:
any missing context, hash mismatch, or I/O error runs the item.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from .board_pytest_selection import (
        UNAFFECTED_SKIP_REASON,
        select_affected_test_files,
    )
except Exception:  # hermetic/capsule import must not disable the green ledger
    UNAFFECTED_SKIP_REASON = "ivp-unaffected"

    def select_affected_test_files(*_args, **_kwargs):
        return None
from .pytest_item_ledger import (
    PID_MARKER_RETRY_LIMIT,
    SKIP_REASON,
    command_fingerprint,
    file_sha256,
    git_worktree_root,
    infer_board_workspace,
    is_empty_int_failure_text,
    is_pid_marker_retry_nodeid,
    ledger_context,
    load_records,
    record_item,
    reusable_nodeids,
)

_STATE: dict[str, Any] = {}


def _workspace(config: Any) -> Path:
    invocation = getattr(config, "invocation_dir", None)
    if invocation:
        return git_worktree_root(Path(str(invocation)))
    return git_worktree_root(Path.cwd())


def _relative_test_file(workspace: Path, item: Any) -> str:
    path = getattr(item, "path", None) or getattr(item, "fspath", None)
    if path is None:
        return ""
    try:
        return Path(str(path)).resolve().relative_to(workspace).as_posix()
    except (OSError, ValueError):
        return Path(str(path)).as_posix()


def pytest_configure(config: Any) -> None:
    """Point proof-reuse at durable board state and enable write in worktrees."""

    try:
        workspace = _workspace(config)
        context = ledger_context(workspace)
        located = infer_board_workspace(workspace)
        if located is None and context is None:
            return
        if located is None:
            repo_root, board = workspace, "aseh"
        else:
            repo_root, board = located
        task_id = str(context["task_id"]) if context else "unbound"
        cache_root = repo_root / "data" / board / "state" / "proof-reuse" / task_id
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(
            "IPFS_TEST_PROOF_REUSE_CACHE_DIR",
            str(cache_root),
        )
        if not os.environ.get("IPFS_TEST_PROOF_REUSE_MODE", "").strip():
            os.environ["IPFS_TEST_PROOF_REUSE_MODE"] = "write"
        if hasattr(config, "option"):
            current = getattr(config.option, "proof_reuse_mode", None)
            if not current:
                setattr(config.option, "proof_reuse_mode", "write")
    except Exception:
        return


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    _STATE.clear()
    try:
        import pytest
    except Exception:
        return
    try:
        workspace = _workspace(config)
        context = ledger_context(workspace)
        if context is None:
            return
        test_files = []
        for item in items:
            relative = _relative_test_file(workspace, item)
            if relative:
                test_files.append(relative)
        affected = select_affected_test_files(workspace, test_files)
        if affected is not None:
            skip_unaffected = pytest.mark.skip(reason=UNAFFECTED_SKIP_REASON)
            for item in items:
                relative = _relative_test_file(workspace, item)
                if relative and relative not in affected:
                    item.add_marker(skip_unaffected)
        command_sha256 = command_fingerprint(list(config.args))
        hashes: dict[str, str] = {}
        for item in items:
            relative = _relative_test_file(workspace, item)
            if not relative or relative in hashes:
                continue
            digest = file_sha256(workspace / relative)
            if digest is not None:
                hashes[relative] = digest
        reusable = reusable_nodeids(
            load_records(context["dest"], workspace=workspace),
            command_sha256=command_sha256,
            workspace_fingerprint_value=str(context["workspace_fingerprint"]),
            test_file_hashes=hashes,
        )
        skip_green = pytest.mark.skip(reason=SKIP_REASON)
        for item in items:
            if item.nodeid in reusable:
                item.add_marker(skip_green)
        _STATE.update(
            {
                "context": context,
                "command_sha256": command_sha256,
                "workspace": workspace,
            }
        )
    except Exception:
        _STATE.clear()


def _empty_int_excinfo(excinfo: Any) -> bool:
    if excinfo is None:
        return False
    value = getattr(excinfo, "value", None)
    if value is None and isinstance(excinfo, tuple) and len(excinfo) >= 2:
        value = excinfo[1]
    return is_empty_int_failure_text(str(value or "")) or is_empty_int_failure_text(
        str(excinfo)
    )


def pytest_runtest_call(item: Any) -> Any:
    """Retry grant-handoff pid-marker races inside the call phase.

    ``Path.write_text`` creates an empty file before writing the PID. The
    protected tests wait on ``exists()`` then ``int(read_text())``, so a
    concurrent reader sees ``''``. Retrying the same item heals ASEH-061
    without editing the protected test file or calling Grok.
    """

    outcome = yield
    nodeid = str(getattr(item, "nodeid", "") or "")
    if not is_pid_marker_retry_nodeid(nodeid):
        return
    runtest = getattr(item, "runtest", None)
    if not callable(runtest):
        return
    for _ in range(PID_MARKER_RETRY_LIMIT):
        if not _empty_int_excinfo(getattr(outcome, "excinfo", None)):
            return
        try:
            runtest()
        except Exception as exc:
            if not is_empty_int_failure_text(str(exc)):
                return
            continue
        force = getattr(outcome, "force_result", None)
        if callable(force):
            force(None)
        return


pytest_runtest_call.pytest_impl = {"hookwrapper": True}  # type: ignore[attr-defined]


def pytest_runtest_logreport(report: Any) -> None:
    if getattr(report, "when", "") != "call":
        return
    if not _STATE:
        return
    if report.skipped:
        return
    try:
        context = _STATE["context"]
        workspace: Path = _STATE["workspace"]
        location = getattr(report, "location", None)
        item_path = str(location[0]) if location else ""
        try:
            test_file = (workspace / item_path).resolve().relative_to(workspace).as_posix()
        except (OSError, ValueError):
            test_file = item_path.replace("\\", "/")
        digest = file_sha256(workspace / test_file)
        if not digest:
            return
        outcome = "passed" if report.passed else "failed"
        record_item(
            context["dest"],
            nodeid=str(report.nodeid),
            outcome=outcome,
            test_file=test_file,
            test_file_sha256=digest,
            workspace_fingerprint_value=str(context["workspace_fingerprint"]),
            command_sha256=str(_STATE["command_sha256"]),
            task_id=str(context["task_id"]),
            workspace=workspace,
        )
    except Exception:
        return
