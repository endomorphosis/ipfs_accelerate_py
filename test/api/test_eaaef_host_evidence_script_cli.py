"""Cold CLI contracts for the EAAEF host-evidence writer scripts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECT_RECEIPTS = REPO_ROOT / "scripts/collect_eaaef_host_admission_receipts.py"
ISSUE_EVIDENCE = REPO_ROOT / "scripts/issue_eaaef_host_evidence.py"
SCRIPTS = (COLLECT_RECEIPTS, ISSUE_EVIDENCE)
CAMPAIGN = REPO_ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
EAAEF_DATA = (
    REPO_ROOT / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
)


def _load_script(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _snapshot_tree(root: Path) -> tuple[tuple[object, ...], ...]:
    """Capture content and write-sensitive metadata without recording atime."""

    if not root.exists() and not root.is_symlink():
        return (("missing",),)
    entries: list[tuple[object, ...]] = []
    paths = [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]
    for path in paths:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        common = (
            relative,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if path.is_symlink():
            entries.append(("symlink", *common, os.readlink(path)))
        elif path.is_dir():
            entries.append(("directory", *common))
        elif path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(("file", *common, digest))
        else:
            entries.append(("other", *common))
    return tuple(entries)


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_is_cold_when_the_script_isolated_from_the_repository(
    tmp_path: Path,
    script: Path,
) -> None:
    isolated = tmp_path / script.name
    shutil.copy2(script, isolated)
    before = _snapshot_tree(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(isolated), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
    assert script.name in completed.stdout
    assert completed.stderr == ""
    assert _snapshot_tree(tmp_path) == before


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_preserves_receipt_authority_and_operator_paths(
    tmp_path: Path,
    script: Path,
) -> None:
    operator_home = tmp_path / "home"
    invocation_cwd = tmp_path / "cwd"
    operator_home.mkdir()
    invocation_cwd.mkdir()
    relevant_paths = (
        CAMPAIGN / "receipts/host_admission",
        EAAEF_DATA / "authority",
        operator_home,
        invocation_cwd,
    )
    before = {path: _snapshot_tree(path) for path in relevant_paths}
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(operator_home),
            "XDG_CACHE_HOME": str(operator_home / "cache"),
            "XDG_CONFIG_HOME": str(operator_home / "config"),
            "XDG_DATA_HOME": str(operator_home / "data"),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(script), "--help"],
        cwd=invocation_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
    assert completed.stderr == ""
    assert {path: _snapshot_tree(path) for path in relevant_paths} == before


def test_collect_no_argument_behavior_still_runs_the_existing_writer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(COLLECT_RECEIPTS, "tested_collect_eaaef_receipts")
    expected = {
        "written": ["receipt.json"],
        "decisions": {"EAAEF-180": "inventory"},
    }
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_collect_host_admission",
        lambda: calls.append("collect_and_write") or expected,
    )

    assert vars(module._parse_args([])) == {}
    assert module.main([]) == 0
    assert calls == ["collect_and_write"]
    assert json.loads(capsys.readouterr().out) == expected


def test_issue_no_argument_behavior_still_materializes_then_collects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(ISSUE_EVIDENCE, "tested_issue_eaaef_evidence")
    calls: list[str] = []

    def materialize() -> dict[str, object]:
        calls.append("materialize")
        return {"decisions": {"EAAEF-185": "typed_missing"}}

    def collect() -> dict[str, object]:
        calls.append("collect")
        return {"decisions": {"EAAEF-191": "no_go"}}

    monkeypatch.setattr(
        module,
        "_host_evidence_entrypoints",
        lambda: (materialize, collect),
    )

    assert vars(module._parse_args([])) == {}
    assert module.main([]) == 0
    assert calls == ["materialize", "collect"]
    assert json.loads(capsys.readouterr().out) == {
        "materialize": {"decisions": {"EAAEF-185": "typed_missing"}},
        "collection": {"EAAEF-191": "no_go"},
        "process_started": False,
        "configured_board_launch": False,
        "live_launch_allowed": False,
    }
