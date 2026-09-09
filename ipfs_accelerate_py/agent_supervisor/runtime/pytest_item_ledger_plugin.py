"""Pytest plugin: skip matching greens from the durable item ledger.

Loaded from ``test/api/conftest.py`` so hermetic validation still sees it when
``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` blocks pytest11 entry points. Fail closed:
any missing context, hash mismatch, or I/O error runs the item.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pytest_item_ledger import (
    SKIP_REASON,
    command_fingerprint,
    file_sha256,
    ledger_context,
    load_records,
    record_item,
    reusable_nodeids,
)

_STATE: dict[str, Any] = {}


def _workspace(config: Any) -> Path:
    invocation = getattr(config, "invocation_dir", None)
    if invocation:
        return Path(str(invocation)).resolve()
    return Path.cwd().resolve()


def _relative_test_file(workspace: Path, item: Any) -> str:
    path = getattr(item, "path", None) or getattr(item, "fspath", None)
    if path is None:
        return ""
    try:
        return Path(str(path)).resolve().relative_to(workspace).as_posix()
    except (OSError, ValueError):
        return Path(str(path)).as_posix()


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
            load_records(context["dest"]),
            command_sha256=command_sha256,
            workspace_fingerprint_value=str(context["workspace_fingerprint"]),
            test_file_hashes=hashes,
        )
        skip = pytest.mark.skip(reason=SKIP_REASON)
        for item in items:
            if item.nodeid in reusable:
                item.add_marker(skip)
        _STATE.update(
            {
                "context": context,
                "command_sha256": command_sha256,
                "workspace": workspace,
            }
        )
    except Exception:
        _STATE.clear()


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
        )
    except Exception:
        return
