"""Exercise nested-project Node preloads using real local package resolution."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is required for package-scope regressions")


def _package(directory: Path, name: str, label: str) -> None:
    package = directory / "node_modules" / name
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({
        "name": name, "type": "module", "exports": {".": "./preload.mjs", "./register": "./preload.mjs"},
    }))
    (package / "preload.mjs").write_text(
        "import fs from 'node:fs';\n"
        f"globalThis.fixturePreload = {label!r};\n"
        f"fs.appendFileSync('preload-evaluations.txt', {label!r} + '\\n');\n",
    )


def _project(
    root: Path, *, package: str = "fixture-preload", declared: bool = True,
    installed: bool = True, expected: str = "project", project_name: str = "nested project",
) -> tuple[Path, Path]:
    project = root / project_name
    project.mkdir()
    (project / "package.json").write_text(json.dumps({
        "name": "fixture-project", "type": "module",
        "devDependencies": {package: "1.0.0"} if declared else {},
    }))
    if installed:
        _package(project, package, "project")
    script = project / "scope.test.mjs"
    script.write_text(
        "import test from 'node:test';\nimport assert from 'node:assert/strict';\nimport fs from 'node:fs';\n"
        "test('project preload and workspace arguments keep their meaning', () => {\n"
        f"  assert.equal(globalThis.fixturePreload, {expected!r});\n"
        f"  assert.equal(process.cwd(), {str(root)!r});\n"
        "  fs.writeFileSync('observation.json', JSON.stringify({loaded: globalThis.fixturePreload, cwd: process.cwd()}));\n"
        "});\n",
    )
    return project, script


def _run(root: Path, argv: list[str], **kwargs: Any) -> int:
    return lane_resume._run_argv(
        argv, repo_root=root, timeout=10, log_path=root / "lane.log", heartbeat=lambda: None, **kwargs,
    )


@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize("package,subpath", [("fixture-preload", ""), ("@fixture/preload", "/register")])
def test_declared_project_preload_runs_from_workspace_without_changing_test_paths(
    tmp_path: Path, equals: bool, package: str, subpath: str,
) -> None:
    _project(tmp_path, package=package)
    specifier = package + subpath
    option = [f"--import={specifier}"] if equals else ["--import", specifier]
    argv = [NODE, *option, "--test", "nested project/scope.test.mjs"]
    normalized = lane_resume._normalize_node_preloads(argv, repo_root=tmp_path, cwd=tmp_path, timeout=10)
    assert normalized[-2:] == argv[-2:]
    assert "file://" in normalized[1 if equals else 2]
    assert not (tmp_path / "preload-evaluations.txt").exists(), "resolution executed the package"
    assert _run(tmp_path, argv) == 0, (tmp_path / "lane.log").read_text()
    assert json.loads((tmp_path / "observation.json").read_text()) == {"loaded": "project", "cwd": str(tmp_path)}
    assert "# cwd=" + str(tmp_path) in (tmp_path / "lane.log").read_text()


@pytest.mark.parametrize("declared,installed", [(False, True), (True, False)])
def test_missing_or_undeclared_preload_retains_module_not_found_failure(
    tmp_path: Path, declared: bool, installed: bool,
) -> None:
    project, _script = _project(tmp_path, declared=declared, installed=installed)
    argv = [NODE, "--import", "fixture-preload", "--test", "nested project/scope.test.mjs"]
    assert lane_resume._normalize_node_preloads(argv, repo_root=tmp_path, cwd=tmp_path, timeout=10) == argv
    assert _run(tmp_path, argv) != 0
    assert "ERR_MODULE_NOT_FOUND" in (tmp_path / "lane.log").read_text()
    assert not (tmp_path / "observation.json").exists()
    assert not (tmp_path / "node_modules").exists(), "normalization installed a root dependency"
    if not installed:
        assert not (project / "node_modules").exists()


def test_existing_workspace_resolution_is_preserved_over_nested_project_version(tmp_path: Path) -> None:
    _project(tmp_path, expected="root")
    _package(tmp_path, "fixture-preload", "root")
    argv = [NODE, "--import", "fixture-preload", "--test", "nested project/scope.test.mjs"]
    assert lane_resume._normalize_node_preloads(argv, repo_root=tmp_path, cwd=tmp_path, timeout=10) == argv
    assert _run(tmp_path, argv) == 0, (tmp_path / "lane.log").read_text()
    assert json.loads((tmp_path / "observation.json").read_text())["loaded"] == "root"


def test_multiple_declaring_projects_do_not_choose_an_arbitrary_preload(tmp_path: Path) -> None:
    _project(tmp_path, project_name="one")
    _project(tmp_path, project_name="two")
    argv = [NODE, "--import", "fixture-preload", "--test", "one/scope.test.mjs", "two/scope.test.mjs"]
    assert lane_resume._normalize_node_preloads(argv, repo_root=tmp_path, cwd=tmp_path, timeout=10) == argv
    assert _run(tmp_path, argv) != 0
    assert "ERR_MODULE_NOT_FOUND" in (tmp_path / "lane.log").read_text()


def test_preload_file_url_and_script_argument_named_import_are_not_rewritten(tmp_path: Path) -> None:
    project, script = _project(tmp_path)
    preload = (project / "node_modules/fixture-preload/preload.mjs").as_uri()
    argv = [NODE, "--import", preload, str(script), "--import", "other-preload"]
    assert lane_resume._normalize_node_preloads(argv, repo_root=tmp_path, cwd=tmp_path, timeout=10) == argv
    assert _run(tmp_path, argv) == 0, (tmp_path / "lane.log").read_text()


def test_declared_preload_symlink_outside_repo_is_not_admitted(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project, _script = _project(root, installed=False)
    _package(tmp_path, "fixture-preload", "external")
    (project / "node_modules").mkdir()
    (project / "node_modules/fixture-preload").symlink_to(tmp_path / "node_modules/fixture-preload", target_is_directory=True)
    # The workspace itself can resolve its parent's dependency, which must be
    # preserved, rather than replaced by the nested project's symlink.
    argv = [NODE, "--import", "fixture-preload", "--test", "nested project/scope.test.mjs"]
    assert lane_resume._normalize_node_preloads(argv, repo_root=root, cwd=root, timeout=10) == argv
    assert not (root / "preload-evaluations.txt").exists()


def test_resolution_timeout_does_not_block_the_lane(tmp_path: Path) -> None:
    _project(tmp_path)
    executable = tmp_path / "node"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
    executable.chmod(0o755)
    argv = [str(executable), "--import", "fixture-preload", "--test", "nested project/scope.test.mjs"]
    started = time.monotonic()
    normalized = lane_resume._normalize_node_preloads(argv, repo_root=tmp_path, cwd=tmp_path, timeout=0.15)
    assert normalized == argv
    assert time.monotonic() - started < 1.5


def test_preload_runtime_repair_changes_contract_without_clock_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path)
    command = [NODE, "--import", "fixture-preload", "--test", "nested project/scope.test.mjs"]
    before = lane_resume._command_contract({}, command, None, repo_root=tmp_path)
    assert lane_resume._command_contract({}, command, None, repo_root=tmp_path) == before
    original = lane_resume._normalize_node_preloads
    relocated = FunctionType(original.__code__.replace(
        co_firstlineno=original.__code__.co_firstlineno + 100, co_filename="different-source-path", co_linetable=b"",
    ), original.__globals__)
    monkeypatch.setattr(lane_resume, "_normalize_node_preloads", relocated)
    assert lane_resume._command_contract({}, command, None, repo_root=tmp_path) == before
    def changed(*args: Any, **kwargs: Any) -> list[str]:
        return original(*args, **kwargs)
    monkeypatch.setattr(lane_resume, "_normalize_node_preloads", changed)
    after = lane_resume._command_contract({}, command, None, repo_root=tmp_path)
    assert after != before
    assert lane_resume._command_contract({}, command, None, repo_root=tmp_path) == after


def test_repaired_preload_runtime_rearms_exhausted_task_without_counter_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project(tmp_path)
    producer = tmp_path / "producer.mjs"
    producer.write_text("import fs from 'node:fs'; fs.writeFileSync('result.json', JSON.stringify({produced:true}));\n")
    board = tmp_path / "board.md"
    board.write_text(
        "## TEST-001 Run repaired preload validation\n- Status: pending\n- Depends on:\n"
        "- Deterministic command: node producer.mjs\n"
        "- Validation: node --import fixture-preload --test 'nested project/scope.test.mjs'\n"
        "- Predicted files: result.json\n",
    )
    from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import parse_todo_task_fields
    fields = dict(parse_todo_task_fields(board.read_text(), task_header_prefix="## TEST-"))["TEST-001"]
    command = lane_resume.parse_deterministic_command(fields["deterministic_command"], repo_root=tmp_path)
    validation = lane_resume.parse_deterministic_command(fields["validation"], repo_root=tmp_path)
    prior = lane_resume._command_contract(fields, command, validation, repo_root=tmp_path)
    attempts = tmp_path / "attempts.json"
    attempts.write_text(json.dumps({
        "TEST-001": 3, "_retry_after_ms": {"TEST-001": int(time.time() * 1000) + 900_000},
        "_command_evidence": {"TEST-001": {"contract": prior, "before": {}, "after": {}, "exit_code": 1}},
    }))
    original = lane_resume._normalize_node_preloads
    def repaired(*args: Any, **kwargs: Any) -> list[str]:
        return original(*args, **kwargs)
    monkeypatch.setattr(lane_resume, "_normalize_node_preloads", repaired)
    monkeypatch.setattr(lane_resume, "_refill_module", lambda _root: None)
    result = lane_resume.run_resume_cycle(
        repo_root=tmp_path, todo_path=board, task_header_prefix="## TEST-", attempt_path=attempts,
        log_path=tmp_path / "lane.log", heartbeat=lambda **_kwargs: None,
    )
    assert result["advanced_task_id"] == "TEST-001", result
    assert json.loads((tmp_path / "result.json").read_text())["produced"] is True
    assert "retry rearmed TEST-001: command contract changed" in (tmp_path / "lane.log").read_text()
    assert "Status: completed" in board.read_text()
