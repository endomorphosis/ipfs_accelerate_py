from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
    parse_todo_blocks,
    split_csv,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module(
    "ips_inventory_gate_under_test",
    "scripts/validate_incremental_proof_sealer_board.py",
)
capture = _load_module(
    "ips_capture_for_inventory_gate_test",
    "scripts/capture_incremental_proof_sealer_baselines.py",
)


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write(
        path,
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
    )


def _init_repository(path: Path, initial: dict[str, str]) -> tuple[str, str]:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "inventory-gate@example.invalid")
    _git(path, "config", "user.name", "Inventory Gate Test")
    for relative, content in initial.items():
        _write(path / relative, content)
    _git(path, "add", "--all")
    _git(path, "commit", "-q", "-m", "planning")
    return _git(path, "rev-parse", "HEAD"), _git(path, "rev-parse", "HEAD^{tree}")


def _commit(path: Path, message: str) -> tuple[str, str]:
    _git(path, "add", "--all")
    _git(path, "commit", "-q", "-m", message)
    return _git(path, "rev-parse", "HEAD"), _git(path, "rev-parse", "HEAD^{tree}")


def _taskboard_text_with_inventory_todos() -> str:
    """Copy the live board with IPS-001/002/003 forced back to todo."""
    text = (ROOT / "docs/architecture/incremental_proof_sealer.todo.md").read_text(
        encoding="utf-8"
    )
    for task_id in ("IPS-001", "IPS-002", "IPS-003"):
        start = text.index(f"## {task_id} ")
        end = text.find("\n## ", start + 1)
        if end < 0:
            end = len(text)
        block = text[start:end]
        for status in ("completed", "in_progress", "blocked"):
            block = block.replace(f"- Status: {status}", "- Status: todo", 1)
        text = text[:start] + block + text[end:]
    return text


def _complete_task(root: Path, task_id: str) -> str:
    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    path = root / relative
    text = path.read_text(encoding="utf-8")
    start = text.index(f"## {task_id} ")
    end = text.find("\n## ", start + 1)
    if end < 0:
        end = len(text)
    block = text[start:end]
    assert "- Status: todo" in block, block[:200]
    text = (
        text[:start]
        + block.replace("- Status: todo", "- Status: completed", 1)
        + text[end:]
    )
    _write(path, text)
    _git(root, "config", "user.email", "implementation-daemon@example.invalid")
    _git(root, "config", "user.name", "Implementation Daemon")
    _git(root, "add", "--", relative)
    _git(root, "commit", "-q", "-m", f"{task_id}: mark todo completed")
    return _git(root, "rev-parse", "HEAD")


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _mock_runner_source_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = {
        "accelerate": {
            "revision": "a" * 40,
            "tree": "b" * 40,
            "content_digest": "sha256:" + "c" * 64,
        }
    }
    monkeypatch.setattr(
        gate,
        "_capture_runner_source_binding",
        lambda _runner, _errors: copy.deepcopy(binding),
    )

    def materialize(
        _runner: str,
        _binding: dict[str, dict[str, str]],
        _errors: list[str],
    ) -> tuple[Path, Path, dict[str, str]]:
        root = gate.REPO_ROOT / gate.RELEASE_WORK_ROOT / "materialized"
        source = root / "source"
        stage = root / "staged"
        source.mkdir(parents=True, exist_ok=True)
        stage.mkdir(parents=True, exist_ok=True)
        return source, stage, {"accelerate": "sha256:" + "d" * 64}

    monkeypatch.setattr(gate, "_materialize_runner_source", materialize)
    monkeypatch.setattr(
        gate,
        "_verify_materialized_source",
        lambda *_args, **_kwargs: {"accelerate": "sha256:" + "d" * 64},
    )


def _set_test_runner_repositories(
    monkeypatch: pytest.MonkeyPatch,
    repositories: dict[str, Path],
) -> None:
    monkeypatch.setattr(gate, "REPOSITORY_PATHS", repositories)
    monkeypatch.setattr(
        gate,
        "RUNNER_UNMATERIALIZED_GITLINKS",
        {name: {} for name in repositories},
    )


def _pre_capture_config() -> dict[str, Any]:
    return {
        "schema": "test-scheduler@1",
        "immutable": {"provider_shell_authority": False},
        "operator_baseline_receipts": {},
        "protected_paths": list(gate.BASE_PROTECTED_PATHS),
    }


def _run_relevance(
    repository: Path,
    source_revision: str,
    parent_revision: str,
    *,
    pins: dict[str, Any] | None = None,
) -> list[str]:
    previous_root = gate.REPO_ROOT
    previous_paths = gate.REPOSITORY_PATHS
    previous_gitlinks = gate.RUNNER_UNMATERIALIZED_GITLINKS
    gate.REPO_ROOT = repository
    gate.REPOSITORY_PATHS = {"accelerate": Path(".")}
    fixture_gitlinks: dict[str, str] = {}
    for line in _git(repository, "ls-files", "--stage").splitlines():
        metadata, relative = line.split("\t", 1)
        mode, oid, _stage = metadata.split(" ", 2)
        if mode == "160000":
            fixture_gitlinks[relative] = oid
    gate.RUNNER_UNMATERIALIZED_GITLINKS = {"accelerate": fixture_gitlinks}
    try:
        errors: list[str] = []
        gate._validate_inventory_source_relevance(
            task_id="IPS-001",
            spec={
                "repository": "accelerate",
                "inventory": next(
                    path
                    for path in gate.ACCELERATE_INVENTORY_OUTPUTS
                    if path.endswith(".json")
                ),
                "report": next(
                    path
                    for path in gate.ACCELERATE_INVENTORY_OUTPUTS
                    if path.endswith(".md")
                ),
            },
            receipt={"source_revision": source_revision},
            parent_revision=parent_revision,
            configured_receipts=pins
            or {
                "IPS-001": {
                    "path": f"{gate.BASELINE_RECEIPT_ROOT}/accelerate.json",
                    "retained_log_paths": [],
                }
            },
            errors=errors,
        )
        return errors
    finally:
        gate.REPO_ROOT = previous_root
        gate.REPOSITORY_PATHS = previous_paths
        gate.RUNNER_UNMATERIALIZED_GITLINKS = previous_gitlinks


def test_post_capture_allowed_bundle_is_inventory_relevant(tmp_path: Path) -> None:
    repository = tmp_path / "allowed"
    _init_repository(
        repository,
        {
            gate.POST_CAPTURE_PIN_CONFIG_PATH: json.dumps(_pre_capture_config()),
            "source.py": "VALUE = 1\n",
        },
    )
    source_revision = _git(repository, "rev-parse", "HEAD")
    evidence_pins = {
        "IPS-001": {
            "path": f"{gate.BASELINE_RECEIPT_ROOT}/accelerate.json",
            "retained_log_paths": [
                f"{gate.BASELINE_LOG_ROOT}/accelerate-proof-focused-core-15-capture.log"
            ],
        },
        "IPS-002": {
            "path": f"{gate.BASELINE_RECEIPT_ROOT}/datasets.json",
            "retained_log_paths": [
                f"{gate.BASELINE_LOG_ROOT}/datasets-zkp-focused-current-capture.log"
            ],
        },
        "IPS-003": {
            "path": f"{gate.BASELINE_RECEIPT_ROOT}/kit.json",
            "retained_log_paths": [
                f"{gate.BASELINE_LOG_ROOT}/kit-proof-certificate-capture.log"
            ],
        },
    }
    after = _pre_capture_config()
    after["operator_baseline_receipts"] = evidence_pins
    all_evidence_paths = [
        path
        for pin in evidence_pins.values()
        for path in (pin["path"], *pin["retained_log_paths"])
    ]
    after["protected_paths"] = [*gate.BASE_PROTECTED_PATHS, *all_evidence_paths]
    _write_json(repository / gate.POST_CAPTURE_PIN_CONFIG_PATH, after)
    for pin in evidence_pins.values():
        _write_json(repository / pin["path"], {"operator": "capture"})
        for retained_log in pin["retained_log_paths"]:
            _write(repository / retained_log, "pytest retained transcript\n")
    parent_revision, _ = _commit(repository, "operator receipt and pin bundle")
    for relative in gate.ACCELERATE_INVENTORY_OUTPUTS:
        _write(repository / relative, "{}\n" if relative.endswith(".json") else "inventory\n")

    errors = _run_relevance(
        repository,
        source_revision,
        parent_revision,
        pins=evidence_pins,
    )
    assert errors == []


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "src/module.py",
        "requirements.lock",
        ".gitignore",
        "docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md",
        "docs/architecture/incremental_proof_sealer.objectives.md",
        "docs/architecture/incremental_proof_sealer.todo.md",
        "config/incremental_proof_sealer_baseline_suite_registry.json",
        "scripts/validate_incremental_proof_sealer_board.py",
        "scripts/capture_incremental_proof_sealer_baselines.py",
        "test/api/test_incremental_proof_sealer_inventory_gate.py",
    ],
)
def test_post_capture_rejects_every_unreviewed_control_or_source_change(
    tmp_path: Path,
    forbidden_path: str,
) -> None:
    repository = tmp_path / "forbidden"
    _init_repository(
        repository,
        {gate.POST_CAPTURE_PIN_CONFIG_PATH: json.dumps(_pre_capture_config())},
    )
    source_revision = _git(repository, "rev-parse", "HEAD")
    _write(repository / forbidden_path, "changed after capture\n")
    parent_revision, _ = _commit(repository, "unreviewed change")

    errors = _run_relevance(repository, source_revision, parent_revision)
    if forbidden_path == "docs/architecture/incremental_proof_sealer.todo.md":
        assert any("taskboard" in error for error in errors), errors
    else:
        assert any("relevance-changing" in error for error in errors), errors


def test_post_capture_rejects_unrelated_scheduler_change(tmp_path: Path) -> None:
    repository = tmp_path / "config-tamper"
    _init_repository(
        repository,
        {gate.POST_CAPTURE_PIN_CONFIG_PATH: json.dumps(_pre_capture_config())},
    )
    source_revision = _git(repository, "rev-parse", "HEAD")
    changed = _pre_capture_config()
    changed["immutable"]["provider_shell_authority"] = True
    _write_json(repository / gate.POST_CAPTURE_PIN_CONFIG_PATH, changed)
    parent_revision, _ = _commit(repository, "unrelated config change")

    errors = _run_relevance(repository, source_revision, parent_revision)
    assert any("not limited to receipt pins/protected paths" in error for error in errors)


def test_inventory_parent_cannot_skip_non_inventory_commit(tmp_path: Path) -> None:
    repository = tmp_path / "stale-parent"
    first, _ = _init_repository(
        repository,
        {gate.POST_CAPTURE_PIN_CONFIG_PATH: json.dumps(_pre_capture_config())},
    )
    _write(repository / "src/module.py", "new source\n")
    _commit(repository, "advance")

    errors = _run_relevance(repository, first, first)
    assert any(
        "direct output-only candidate" in error
        or "relevance-changing" in error
        for error in errors
    ), errors


@pytest.mark.parametrize("inventory_only", [True, False])
def test_nested_gitlink_delta_is_recursively_limited_to_inventory_documents(
    tmp_path: Path,
    inventory_only: bool,
) -> None:
    origin = tmp_path / "datasets-origin"
    _init_repository(origin, {"README.md": "nested source\n"})
    repository = tmp_path / "outer"
    _init_repository(
        repository,
        {gate.POST_CAPTURE_PIN_CONFIG_PATH: json.dumps(_pre_capture_config())},
    )
    _git(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        "ipfs_datasets_py",
    )
    source_revision, _ = _commit(repository, "captured gitlink")
    nested = repository / "ipfs_datasets_py"
    _git(nested, "config", "user.email", "inventory-gate@example.invalid")
    _git(nested, "config", "user.name", "Inventory Gate Test")
    if inventory_only:
        for relative in gate.NESTED_INVENTORY_OUTPUTS["ipfs_datasets_py"]:
            _write(nested / relative, "{}\n" if relative.endswith(".json") else "inventory\n")
    else:
        _write(nested / "src/changed.py", "CHANGED = True\n")
    _commit(nested, "nested delta")
    parent_revision, _ = _commit(repository, "advance gitlink")

    previous_root = gate.REPO_ROOT
    gate.REPO_ROOT = repository
    try:
        errors: list[str] = []
        gate._validate_nested_gitlink_inventory_delta(
            submodule="ipfs_datasets_py",
            source_revision=source_revision,
            parent_revision=parent_revision,
            errors=errors,
        )
    finally:
        gate.REPO_ROOT = previous_root
    if inventory_only:
        assert errors == []
    else:
        assert any("gitlink delta contains non-inventory paths" in error for error in errors)


def test_accelerate_inventory_accepts_candidate_no_ff_merge_and_status_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "accelerate-publication"
    parent, _ = _init_repository(
        root,
        {
            taskboard: _taskboard_text_with_inventory_todos(),
            "source.py": "captured\n",
        },
    )
    target = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    for relative in gate.ACCELERATE_INVENTORY_OUTPUTS:
        _write(root / relative, "{}\n" if relative.endswith(".json") else "inventory\n")
    candidate, _ = _commit(root, "IPS-001 inventory candidate")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate IPS-001")
    merged = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    missing_status_errors: list[str] = []
    gate._validate_accelerate_inventory_lifecycle(
        task_id="IPS-001",
        parent_revision=parent,
        current_revision=merged,
        outputs=set(gate.ACCELERATE_INVENTORY_OUTPUTS),
        require_published=True,
        errors=missing_status_errors,
    )
    assert any(
        "publication lineage lacks its daemon status commit" in error
        for error in missing_status_errors
    ), missing_status_errors
    completion = _complete_task(root, "IPS-001")
    errors: list[str] = []

    gate._validate_accelerate_inventory_lifecycle(
        task_id="IPS-001",
        parent_revision=parent,
        current_revision=completion,
        outputs=set(gate.ACCELERATE_INVENTORY_OUTPUTS),
        require_published=True,
        errors=errors,
    )

    assert candidate != completion
    assert errors == []


@pytest.mark.parametrize(
    ("task_id", "submodule"),
    (("IPS-002", "ipfs_datasets_py"), ("IPS-003", "ipfs_kit_py")),
)
def test_nested_inventory_accepts_child_outer_no_ff_merge_and_status_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    submodule: str,
) -> None:
    origin = tmp_path / f"{submodule}-origin"
    nested_parent, _ = _init_repository(origin, {"source.py": "captured\n"})
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / f"{submodule}-publication"
    _init_repository(
        root,
        {taskboard: _taskboard_text_with_inventory_todos()},
    )
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        submodule,
    )
    control_captured, _ = _commit(root, "capture nested source")
    nested = root / submodule
    _git(nested, "config", "user.email", "inventory-gate@example.invalid")
    _git(nested, "config", "user.name", "Inventory Gate Test")
    for relative in gate.NESTED_INVENTORY_OUTPUTS[submodule]:
        _write(nested / relative, "{}\n" if relative.endswith(".json") else "inventory\n")
    nested_candidate, _ = _commit(nested, f"{task_id} nested inventory")
    target = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    _git(root, "add", "--", submodule)
    _git(root, "commit", "-q", "-m", f"{task_id} outer candidate")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", f"integrate {task_id}")
    completion = _complete_task(root, task_id)
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_nested_inventory_lifecycle(
        task_id=task_id,
        submodule=submodule,
        parent_revision=nested_parent,
        current_nested_revision=nested_candidate,
        control_captured_revision=control_captured,
        control_current_revision=completion,
        outputs=set(gate.NESTED_INVENTORY_OUTPUTS[submodule]),
        require_published=True,
        errors=errors,
    )

    assert errors == []



def test_control_transition_admits_sibling_inventory_second_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling nested inventory candidates must not look like side-branch laundering."""

    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "sibling-candidate"
    captured, _ = _init_repository(
        root,
        {
            taskboard: _taskboard_text_with_inventory_todos(),
            "source.py": "captured\n",
        },
    )
    target = _git(root, "branch", "--show-current")
    origin = tmp_path / "datasets-origin"
    _init_repository(origin, {"source.py": "captured\n"})
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        "ipfs_datasets_py",
    )
    control_captured, _ = _commit(root, "capture nested")
    nested = root / "ipfs_datasets_py"
    _git(nested, "config", "user.email", "inventory-gate@example.invalid")
    _git(nested, "config", "user.name", "Inventory Gate Test")
    for relative in gate.NESTED_INVENTORY_OUTPUTS["ipfs_datasets_py"]:
        _write(
            nested / relative,
            "{}\n" if relative.endswith(".json") else "inventory\n",
        )
    _commit(nested, "IPS-002 nested inventory")
    _git(root, "checkout", "-q", "-b", "candidate")
    _git(root, "add", "--", "ipfs_datasets_py")
    _git(root, "commit", "-q", "-m", "IPS-002 outer candidate")
    candidate = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate IPS-002")
    completion = _complete_task(root, "IPS-002")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []
    gate._validate_accelerate_control_transition(
        task_id="IPS-001",
        captured_revision=control_captured,
        current_revision=completion,
        configured_receipts={},
        errors=errors,
    )
    assert not any("untrusted merged side branch" in error for error in errors), errors
    assert candidate in set(
        _git(root, "rev-list", f"{control_captured}..{completion}").splitlines()
    )


def test_control_transition_skips_dead_competing_nested_inventory_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing concurrent nested inventory tips must not fail first-parent control.

    When a no-ff merge keeps the first-parent gitlink (TREESAME / -s ours), the
    candidate remains reachable in the DAG but never rewrote first-parent content.
    """

    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "dead-competing-tip"
    _init_repository(
        root,
        {
            taskboard: _taskboard_text_with_inventory_todos(),
            "source.py": "captured\n",
        },
    )
    target = _git(root, "branch", "--show-current")
    origin = tmp_path / "kit-origin"
    _init_repository(origin, {"source.py": "captured\n"})
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        "ipfs_kit_py",
    )
    control_captured, _ = _commit(root, "capture nested")
    nested = root / "ipfs_kit_py"
    _git(nested, "config", "user.email", "inventory-gate@example.invalid")
    _git(nested, "config", "user.name", "Inventory Gate Test")

    # Winning tip lands on first-parent history.
    for relative in gate.NESTED_INVENTORY_OUTPUTS["ipfs_kit_py"]:
        _write(
            nested / relative,
            "{}\n" if relative.endswith(".json") else "winning inventory\n",
        )
    _commit(nested, "IPS-003 winning nested inventory")
    _git(root, "add", "--", "ipfs_kit_py")
    _git(root, "commit", "-q", "-m", "IPS-003 first-parent inventory tip")
    winning_gitlink = _git(root, "rev-parse", f"HEAD:ipfs_kit_py")

    # Losing concurrent tip on a side branch from capture (before winning tip).
    capture_gitlink = _git(root, "rev-parse", f"{control_captured}:ipfs_kit_py")
    _git(root, "checkout", "-q", "-b", "losing-candidate", control_captured)
    nested_losing = root / "ipfs_kit_py"
    _git(nested_losing, "checkout", "-q", "-f", capture_gitlink)
    for relative in gate.NESTED_INVENTORY_OUTPUTS["ipfs_kit_py"]:
        _write(
            nested_losing / relative,
            "{}\n" if relative.endswith(".json") else "losing inventory\n",
        )
    _commit(nested_losing, "IPS-003 losing nested inventory")
    _git(root, "add", "--", "ipfs_kit_py")
    _git(root, "commit", "-q", "-m", "IPS-003 losing outer candidate")
    losing_candidate = _git(root, "rev-parse", "HEAD")
    losing_gitlink = _git(root, "rev-parse", "HEAD:ipfs_kit_py")
    assert losing_gitlink != winning_gitlink

    # Merge keeps first-parent gitlink (dead competing tip).
    _git(root, "checkout", "-q", target)
    _git(
        root,
        "merge",
        "-q",
        "-s",
        "ours",
        "--no-ff",
        "losing-candidate",
        "-m",
        "merge losing IPS-003 without taking tip",
    )
    assert _git(root, "rev-parse", f"HEAD:ipfs_kit_py") == winning_gitlink
    completion = _complete_task(root, "IPS-001")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []
    gate._validate_accelerate_control_transition(
        task_id="IPS-001",
        captured_revision=control_captured,
        current_revision=completion,
        configured_receipts={},
        errors=errors,
    )
    assert not any("untrusted merged side branch" in error for error in errors), errors
    assert losing_candidate in set(
        _git(root, "rev-list", f"{control_captured}..{completion}").splitlines()
    )
    assert gate._is_dead_competing_inventory_tip(
        repository=root,
        commit=losing_candidate,
        first_parent_commits=set(
            _git(
                root,
                "rev-list",
                "--first-parent",
                f"{control_captured}..{completion}",
            ).splitlines()
        ),
        outputs={"ipfs_kit_py"},
    )


def test_operational_residual_board_appendix_is_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "residual-appendix"
    parent, _ = _init_repository(
        root, {taskboard: _taskboard_text_with_inventory_todos()}
    )
    residual = """
## IPS-057 Resolve merge retry-budget failure for IPS-003

- Status: todo
- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1
- Retry repair source: IPS-003
- Retry failure kind: merge
- Depends on: IPS-000
- Outputs: ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json
- Validation: python scripts/validate_incremental_proof_sealer_board.py --check-artifact IPS-003
- Acceptance: fix merge then complete.
"""
    path = root / taskboard
    _write(path, path.read_text(encoding="utf-8") + residual)
    _git(root, "add", "--", taskboard)
    _git(root, "commit", "-q", "-m", "Agent: record retry-budget guardrail outputs")
    current = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    assert gate._is_operational_residual_board_appendix(parent, current)


def test_accelerate_inventory_rejects_full_dag_rewrite_then_revert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Side-branch rewrite/revert must not launder a final-tree-looking publication."""

    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "accelerate-dag-launder"
    parent, _ = _init_repository(
        root, {taskboard: _taskboard_text_with_inventory_todos()}
    )
    target = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    for relative in gate.ACCELERATE_INVENTORY_OUTPUTS:
        _write(
            root / relative,
            "{}\n" if relative.endswith(".json") else "inventory\n",
        )
    _commit(root, "IPS-001 inventory candidate")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate IPS-001")
    _complete_task(root, "IPS-001")
    # Extra side branch rewrites inventory then reverts to the admitted blobs.
    _git(root, "checkout", "-q", "-b", "launder")
    rewritten = next(iter(gate.ACCELERATE_INVENTORY_OUTPUTS))
    _write(root / rewritten, "temporary rewrite\n")
    _commit(root, "rewrite inventory on side branch")
    _write(
        root / rewritten,
        "{}\n" if rewritten.endswith(".json") else "inventory\n",
    )
    _commit(root, "revert inventory rewrite on side branch")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "launder", "-m", "merge laundered history")
    current = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_accelerate_inventory_lifecycle(
        task_id="IPS-001",
        parent_revision=parent,
        current_revision=current,
        outputs=set(gate.ACCELERATE_INVENTORY_OUTPUTS),
        require_published=True,
        errors=errors,
    )

    assert any(
        "reachable Git DAG rewrites inventory outputs" in error for error in errors
    ), errors


def test_accelerate_inventory_rejects_first_parent_rewrite_then_revert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "accelerate-first-parent-launder"
    parent, _ = _init_repository(
        root, {taskboard: _taskboard_text_with_inventory_todos()}
    )
    target = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    for relative in gate.ACCELERATE_INVENTORY_OUTPUTS:
        _write(
            root / relative,
            "{}\n" if relative.endswith(".json") else "inventory\n",
        )
    _commit(root, "IPS-001 inventory candidate")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate IPS-001")
    _complete_task(root, "IPS-001")
    rewritten = next(iter(gate.ACCELERATE_INVENTORY_OUTPUTS))
    _write(root / rewritten, "temporary rewrite\n")
    _commit(root, "rewrite inventory on first parent")
    _write(
        root / rewritten,
        "{}\n" if rewritten.endswith(".json") else "inventory\n",
    )
    current, _ = _commit(root, "revert inventory rewrite on first parent")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_accelerate_inventory_lifecycle(
        task_id="IPS-001",
        parent_revision=parent,
        current_revision=current,
        outputs=set(gate.ACCELERATE_INVENTORY_OUTPUTS),
        require_published=True,
        errors=errors,
    )

    assert any(
        "reachable Git DAG rewrites inventory outputs" in error for error in errors
    ), errors


def test_accelerate_inventory_rejects_side_branch_taskboard_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "accelerate-side-board"
    parent, _ = _init_repository(
        root, {taskboard: _taskboard_text_with_inventory_todos()}
    )
    target = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    for relative in gate.ACCELERATE_INVENTORY_OUTPUTS:
        _write(
            root / relative,
            "{}\n" if relative.endswith(".json") else "inventory\n",
        )
    _commit(root, "IPS-001 inventory candidate")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate IPS-001")
    _complete_task(root, "IPS-001")
    # Side branch touches the taskboard (even if later merged without final drift).
    _git(root, "checkout", "-q", "-b", "board-side")
    text = (root / taskboard).read_text(encoding="utf-8")
    _write(
        root / taskboard,
        text.replace(
            "- Validation: python scripts/validate_incremental_proof_sealer_board.py "
            "--check-artifact IPS-002",
            "- Validation: python forged.py",
            1,
        ),
    )
    _git(root, "add", "--", taskboard)
    _git(root, "commit", "-q", "-m", "side board mutation")
    # Revert board text so final tree matches first-parent content.
    _write(root / taskboard, text)
    _git(root, "add", "--", taskboard)
    _git(root, "commit", "-q", "-m", "revert side board mutation")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "board-side", "-m", "merge board side")
    current = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_accelerate_inventory_lifecycle(
        task_id="IPS-001",
        parent_revision=parent,
        current_revision=current,
        outputs=set(gate.ACCELERATE_INVENTORY_OUTPUTS),
        require_published=True,
        errors=errors,
    )

    assert any(
        "taskboard was modified on an untrusted merged side branch" in error
        for error in errors
    ), errors


def test_accelerate_inventory_rejects_post_merge_blob_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "accelerate-rewrite"
    parent, _ = _init_repository(
        root, {taskboard: _taskboard_text_with_inventory_todos()}
    )
    target = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    for relative in gate.ACCELERATE_INVENTORY_OUTPUTS:
        _write(root / relative, "{}\n" if relative.endswith(".json") else "inventory\n")
    _commit(root, "IPS-001 inventory candidate")
    _git(root, "checkout", "-q", target)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate IPS-001")
    _complete_task(root, "IPS-001")
    rewritten = next(iter(gate.ACCELERATE_INVENTORY_OUTPUTS))
    _write(root / rewritten, "rewritten\n")
    current, _ = _commit(root, "rewrite accepted inventory")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_accelerate_inventory_lifecycle(
        task_id="IPS-001",
        parent_revision=parent,
        current_revision=current,
        outputs=set(gate.ACCELERATE_INVENTORY_OUTPUTS),
        require_published=True,
        errors=errors,
    )

    assert any("blobs changed after" in error for error in errors), errors


def test_completed_inventory_without_outputs_is_not_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    root = tmp_path / "missing-completed-output"
    _init_repository(
        root, {taskboard: _taskboard_text_with_inventory_todos()}
    )
    _complete_task(root, "IPS-001")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_published_inventory_artifacts(errors)

    assert any(
        "IPS-001 is completed but committed outputs are missing" in error
        for error in errors
    ), errors


def test_inventory_contract_revision_changes_all_three_canonical_task_cids() -> None:
    prior = {
        "IPS-001": "baguqeeraaumb2lxdl4znxw2fcpafbupteyul5nocu3jefmldcaouzwuadfoq",
        "IPS-002": "baguqeerazjmoazoowy2plo62xoeivdni4cmh3yaifs6ijg3ady2kpdjdspaq",
        "IPS-003": "baguqeerarwmbfdov5nxfz4zu64cls4m2lhzs2ysmmmx77iiyrgiua5uo7z6q",
    }
    path = ROOT / "docs/architecture/incremental_proof_sealer.todo.md"
    current: dict[str, str] = {}
    for task_id, title, _line, fields in parse_todo_blocks(
        path.read_text(encoding="utf-8"), task_header_prefix="## IPS-"
    ):
        if task_id not in prior:
            continue
        current[task_id] = canonical_task_identity(
            {
                "task_id": task_id,
                "title": title,
                "outputs": split_csv(fields.get("outputs", "")),
                "acceptance": fields.get("acceptance", ""),
                "metadata": fields,
            },
            board_namespace=fields.get("board_namespace", "") or path.name,
            source_path=path,
        ).canonical_task_cid

    assert set(current) == set(prior)
    assert all(current[task_id] != old for task_id, old in prior.items())


def test_reference_only_namespace_allows_static_classification() -> None:
    payload = {
        "baseline_evidence": {
            "path": "fixed.json",
            "receipt_digest": _digest("receipt"),
            "required_command_ids": ["fixed-suite"],
            "evidence_origin": "operator_capture",
            "assurance": "process_observed_only",
            "nonclaim": "pytest_execution_not_cryptographically_proven",
        },
        "classifications": [
            {
                "classification_method": "static source inventory",
                "surface": "tests/test_proof_store.py",
                "surfaces_found": 1,
            }
        ],
    }
    errors: list[str] = []
    gate._validate_reference_only_inventory_namespace("IPS-001", payload, errors)
    assert errors == []


def test_registry_validation_never_executes_modified_capture_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "registry-code-executed"
    repository = tmp_path / "repository"
    _write(
        repository / gate.BASELINE_CAPTURE_SCRIPT,
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
    )
    _write(
        repository / gate.BASELINE_SUITE_REGISTRY,
        (ROOT / gate.BASELINE_SUITE_REGISTRY).read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(gate, "REPO_ROOT", repository)
    errors: list[str] = []
    reviewed = gate._reviewed_suite_registry(errors)
    assert reviewed
    assert errors == []
    assert not marker.exists()


def test_registry_safe_argv_mutation_is_rejected_by_independent_digest_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "registry-digest"
    payload = json.loads((ROOT / gate.BASELINE_SUITE_REGISTRY).read_text(encoding="utf-8"))
    payload["repositories"]["accelerate"][0]["argv_template"][-1] = "tests/evil.py"
    _write_json(repository / gate.BASELINE_SUITE_REGISTRY, payload)
    monkeypatch.setattr(gate, "REPO_ROOT", repository)
    errors: list[str] = []

    gate._reviewed_suite_registry(errors)

    assert any("registry digest differs" in error for error in errors)


@pytest.mark.parametrize("hostile_name", ["GIT_DIR", "GIT_INDEX_FILE"])
def test_gate_git_checks_ignore_hostile_inherited_git_redirection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_name: str,
) -> None:
    repository = tmp_path / "fixed-git-environment"
    expected, _ = _init_repository(repository, {"tracked.txt": "bound\n"})
    monkeypatch.setenv(hostile_name, str(tmp_path / "attacker-controlled"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.repositoryformatversion")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "999")

    result = gate._git("rev-parse", "HEAD", cwd=repository)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
    assert gate._fixed_git_environment()["GIT_OPTIONAL_LOCKS"] == "0"
    assert hostile_name not in gate._fixed_git_environment()


def test_gate_marks_module_collection_skip_incomplete() -> None:
    log = """============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 0 items / 1 skipped
=========================== short test summary info ============================
SKIPPED [1] tests/test_optional.py:7: optional module unavailable
============================== 1 skipped in 0.01s ==============================
"""
    assert gate._collection_count(log) == 0
    assert gate._nonpass_nodes(log) == [
        {
            "status": "skipped",
            "node_id": "tests/test_optional.py:7",
            "detail": "SKIPPED [1] tests/test_optional.py:7: optional module unavailable",
        }
    ]
    assert gate._collection_complete(log) is False


@pytest.mark.parametrize(
    "collected_line",
    (
        "collecting ... collected 0 items / 1 error",
        "collected 7 items / 2 errors / 1 deselected",
    ),
)
def test_gate_marks_collected_error_summary_incomplete(
    collected_line: str,
) -> None:
    log = (
        "============================= test session starts "
        "==============================\n"
        f"{collected_line}\n"
        "=========================== short test summary info "
        "============================\n"
        "==== 1 error in 0.01s ===="
    )

    assert gate._collection_count(log) in {0, 7}
    assert gate._collection_complete(log) is False


def test_gate_parses_decorated_collection_error_node() -> None:
    log = (
        "============================= test session starts "
        "==============================\n"
        "collected 0 items / 1 error\n"
        "____________ ERROR collecting tests/test_broken.py ____________\n"
        "=========================== short test summary info "
        "============================\n"
        "ERROR tests/test_broken.py\n"
        "==== 1 error in 0.01s ===="
    )

    assert gate._collection_complete(log) is False
    nodes = gate._nonpass_nodes(log)
    assert any(
        item["status"] == "error"
        and item["node_id"] == "collecting tests/test_broken.py"
        for item in nodes
    )


@pytest.mark.parametrize("error_count", (6, 40, 3, 52, 1))
def test_gate_collection_abort_nodes_match_capture_parser(
    error_count: int,
) -> None:
    paths = [f"tests/test_collection_{index:03d}.py" for index in range(error_count)]
    transcript = "\n".join(
        (
            "============================= test session starts ==============================",
            f"collected 0 items / {error_count} errors",
            *(
                f"________ ERROR collecting {path} ________"
                for path in paths
            ),
            "=========================== short test summary info ============================",
            *(f"ERROR {path}" for path in paths),
            f"================ {error_count} errors in 0.10s ================",
        )
    )

    captured = capture.parse_pytest_log(transcript.encode("utf-8"))

    assert gate._collection_complete(transcript) is False
    assert gate._nonpass_nodes(transcript) == captured["non_pass_nodes"]
    assert len(gate._nonpass_nodes(transcript)) == error_count


def test_gate_keeps_ordinary_skipped_item_collection_complete() -> None:
    log = """============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 1 item
tests/test_optional.py::test_optional SKIPPED (optional)                       [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_optional.py:7: optional
============================== 1 skipped in 0.01s ==============================
"""
    nonpass = gate._nonpass_nodes(log)
    assert [(item["status"], item["node_id"]) for item in nonpass] == [
        ("skipped", "tests/test_optional.py::test_optional")
    ]
    assert gate._collection_complete(log) is True


@pytest.mark.parametrize(
    "shadow",
    [
        {"nested": {"passed_count": 999, "failed_count": 0}},
        {"nested": [{"retained_transcript": "invented"}]},
        {"nested": {"pytest_result": "passed"}},
        {"execution_status": "success"},
        {"claim": "all test cases were green and succeeded"},
        {"details": {"command_line": ["pytest", "tests"]}},
        {"provider_summary": {"tests_green_total": 999, "tests_red_total": 0}},
        {"successful_cases_total": 999},
        {"execution_ok_total": 999},
        {"pytest": {"green": 999, "red": 0}},
        {"test_summary": {"green": 999, "red": 0}},
    ],
)
def test_reference_only_namespace_rejects_recursive_execution_shadows(
    shadow: dict[str, Any],
) -> None:
    payload = {
        "baseline_evidence": {
            "path": "fixed.json",
            "receipt_digest": _digest("receipt"),
        },
        **shadow,
    }
    errors: list[str] = []
    gate._validate_reference_only_inventory_namespace("IPS-001", payload, errors)
    assert errors, shadow


@dataclass
class CapturedBundle:
    root: Path
    spec: dict[str, Any]
    planning_revisions: dict[str, str]
    planning_trees: dict[str, str]
    source_revisions: dict[str, str]
    source_trees: dict[str, str]
    reviewed_suites: dict[str, dict[str, Any]]
    receipt_path: Path
    config_path: Path
    inventory_path: Path
    report_path: Path
    pristine_receipt: dict[str, Any]
    pristine_config: dict[str, Any]
    pristine_inventory: dict[str, Any]
    pristine_report: str
    pristine_logs: dict[str, bytes]

    def restore(self) -> None:
        _write_json(self.receipt_path, copy.deepcopy(self.pristine_receipt))
        _write_json(self.config_path, copy.deepcopy(self.pristine_config))
        _write_json(self.inventory_path, copy.deepcopy(self.pristine_inventory))
        _write(self.report_path, self.pristine_report)
        for relative, raw in self.pristine_logs.items():
            path = self.root / relative
            if path.is_symlink():
                path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

    def write_receipt(
        self,
        mutation: Callable[[dict[str, Any]], None],
        *,
        synchronize_pin: bool = True,
    ) -> dict[str, Any]:
        receipt = copy.deepcopy(self.pristine_receipt)
        mutation(receipt)
        receipt["receipt_digest"] = capture.receipt_digest(receipt)
        _write_json(self.receipt_path, receipt)
        if synchronize_pin:
            config = copy.deepcopy(self.pristine_config)
            pin = config["operator_baseline_receipts"]["IPS-003"]
            pin["receipt_digest"] = receipt["receipt_digest"]
            pin["source_revision"] = receipt.get("source_revision")
            pin["source_tree"] = receipt.get("source_tree")
            pin["suite_definition_digests"] = {
                command["id"]: command["suite_definition_digest"]
                for command in receipt.get("commands", [])
            }
            pin["retained_log_paths"] = [
                command["log"]["relative_path"]
                for command in receipt.get("commands", [])
            ]
            config["protected_paths"] = [
                *gate.BASE_PROTECTED_PATHS,
                pin["path"],
                *pin["retained_log_paths"],
            ]
            _write_json(self.config_path, config)
        return receipt


@pytest.fixture(scope="module")
def captured_bundle(tmp_path_factory: pytest.TempPathFactory) -> CapturedBundle:
    root = tmp_path_factory.mktemp("inventory-gate-bundle")
    planning_revisions: dict[str, str] = {}
    planning_trees: dict[str, str] = {}
    source_revisions: dict[str, str] = {}
    source_trees: dict[str, str] = {}

    datasets = root / "ipfs_datasets_py"
    planning_revisions["datasets"], planning_trees["datasets"] = _init_repository(
        datasets, {"README.md": "datasets planning\n"}
    )
    _write(datasets / "source.py", "DATASET = 1\n")
    source_revisions["datasets"], source_trees["datasets"] = _commit(
        datasets, "tested datasets"
    )

    kit = root / "ipfs_kit_py"
    planning_revisions["kit"], planning_trees["kit"] = _init_repository(
        kit,
        {
            ".gitignore": (
                "build/\n*.key\nsitecustomize.py\n.pytest_cache/\n__pycache__/\n"
            ),
            "README.md": "kit planning\n",
        },
    )
    _write(
        kit / "tests/test_proof_certificate_store.py",
        "def test_proof_certificate_store():\n    assert True\n",
    )
    _write(
        kit / "tests/test_reuse_capabilities.py",
        "def test_reuse_capabilities():\n    assert True\n",
    )
    source_revisions["kit"], source_trees["kit"] = _commit(kit, "tested kit")

    planning_revisions["accelerate"], planning_trees["accelerate"] = _init_repository(
        root,
        {
            ".gitignore": (
                "artifacts/\n"
                f"/{gate.RELEASE_WORK_ROOT}/\n"
            ),
            "README.md": "accelerate planning\n",
        },
    )
    base_config = json.loads(
        (ROOT / "config/agent_supervisor_incremental_proof_sealer_scheduler.json").read_text(
            encoding="utf-8"
        )
    )
    base_config["source_binding"]["accelerator_required_ancestor"] = planning_revisions[
        "accelerate"
    ]
    base_config["source_binding"]["accelerator_planning_revision"] = planning_revisions[
        "accelerate"
    ]
    base_config["source_binding"]["ipfs_datasets_planning_revision"] = planning_revisions[
        "datasets"
    ]
    base_config["source_binding"]["ipfs_kit_planning_revision"] = planning_revisions[
        "kit"
    ]
    base_config["operator_baseline_receipts"] = {}
    base_config["protected_paths"] = list(gate.BASE_PROTECTED_PATHS)
    config_path = root / "config/agent_supervisor_incremental_proof_sealer_scheduler.json"
    _write_json(config_path, base_config)
    source_revisions["accelerate"], source_trees["accelerate"] = _commit(
        root, "tested controls"
    )

    capture_id = "20260811T130000.000000Z-4242"
    python_info = capture._python_metadata()
    pytest_info = capture._pytest_metadata()
    suites = [
        capture.SUITES_BY_ID["kit-proof-certificate"],
        capture.SUITES_BY_ID["kit-reuse-capabilities"],
    ]
    commands = [
        capture._capture_command(
            root, root, suite, capture_id, python_info, pytest_info
        )
        for suite in suites
    ]
    reviewed_suites: dict[str, dict[str, Any]] = {}
    for suite in suites:
        payload = capture.suite_definition_payload(suite)
        payload["suite_definition_digest"] = capture.suite_definition_digest(suite)
        reviewed_suites[suite.id] = payload

    receipt: dict[str, Any] = {
        "schema_version": capture.SCHEMA_VERSION,
        "operator_origin": capture.OPERATOR_ORIGIN,
        "repository": "kit",
        "task_id": "IPS-003",
        "capture_id": capture_id,
        "captured_at": "2026-08-11T13:00:02.000000Z",
        "required_command_ids": [suite.id for suite in suites],
        "planning_revision": planning_revisions["kit"],
        "planning_tree": planning_trees["kit"],
        "source_revision": source_revisions["kit"],
        "source_tree": source_trees["kit"],
        "execution_head": source_revisions["accelerate"],
        "execution_tree": source_trees["accelerate"],
        "source_revisions": dict(source_revisions),
        "source_trees": dict(source_trees),
        "source_clean_before": {name: True for name in source_revisions},
        "source_clean_after": {name: True for name in source_revisions},
        "ignored_sensitive_inputs": {
            "policy_id": capture.IGNORED_INPUT_POLICY_ID,
            "repositories": {
                name: capture._ignored_sensitive_binding({})
                for name in source_revisions
            },
        },
        "git_environment_policy_id": gate.BASELINE_GIT_ENVIRONMENT_POLICY,
        "commands": commands,
        "assurance": capture._assurance_payload(process_observed=True, aggregate=True),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = capture.receipt_digest(receipt)
    receipt_path = root / gate.BASELINE_RECEIPT_ROOT / "kit.json"
    _write_json(receipt_path, receipt)

    spec = {
        "repository": "kit",
        "revision": planning_revisions["kit"],
        "receipt": f"{gate.BASELINE_RECEIPT_ROOT}/kit.json",
        "inventory": (
            "ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json"
        ),
        "report": (
            "ipfs_kit_py/docs/architecture/INCREMENTAL_PROOF_SEALER_INVENTORY.md"
        ),
        "command_ids": tuple(suite.id for suite in suites),
        "cwd": "ipfs_kit_py",
        "timeouts": tuple(suite.timeout_seconds for suite in suites),
    }
    pin = {
        "path": spec["receipt"],
        "receipt_digest": receipt["receipt_digest"],
        "planning_revision": planning_revisions["kit"],
        "source_revision": source_revisions["kit"],
        "source_tree": source_trees["kit"],
        "required_command_ids": list(spec["command_ids"]),
        "suite_definition_digests": {
            command["id"]: command["suite_definition_digest"] for command in commands
        },
        "retained_log_paths": [command["log"]["relative_path"] for command in commands],
    }
    pinned_config = copy.deepcopy(base_config)
    pinned_config["operator_baseline_receipts"] = {"IPS-003": pin}
    pinned_config["protected_paths"] = [
        *gate.BASE_PROTECTED_PATHS,
        spec["receipt"],
        *pin["retained_log_paths"],
    ]
    _write_json(config_path, pinned_config)
    _git(root, "add", "-f", spec["receipt"], *pin["retained_log_paths"])
    _commit(root, "pin operator receipt")

    required_surfaces = [
        "profile_d_policy.py",
        "mcplusplus/artifacts.py",
        "iroh/release.py",
        "test_joined_release_receipt.py",
        "install_lotus.py",
        "proof_certificate_store.py",
        "event_dag.py",
    ]
    baseline_evidence = {
        "path": spec["receipt"],
        "receipt_digest": receipt["receipt_digest"],
        "required_command_ids": list(spec["command_ids"]),
        "evidence_origin": gate.BASELINE_OPERATOR_ORIGIN,
        "assurance": "process_observed_only",
        "nonclaim": "pytest_execution_not_cryptographically_proven",
    }
    inventory = {
        "planning_revision": planning_revisions["kit"],
        "inventory_worktree_parent_revision": source_revisions["kit"],
        "baseline_evidence": baseline_evidence,
        "classifications": [
            {
                "classification_method": "static source inventory",
                "surface": surface,
                "surfaces_found": 1,
            }
            for surface in required_surfaces
        ],
    }
    inventory_path = root / spec["inventory"]
    report_path = root / spec["report"]
    _write_json(inventory_path, inventory)
    report = "\n".join(
        [
            "# Inventory",
            spec["receipt"],
            receipt["receipt_digest"],
            *spec["command_ids"],
            "process_observed_only",
            "pytest_execution_not_cryptographically_proven",
        ]
    )
    _write(report_path, report + "\n")
    pristine_logs = {
        command["log"]["relative_path"]: (
            root / command["log"]["relative_path"]
        ).read_bytes()
        for command in commands
    }
    return CapturedBundle(
        root=root,
        spec=spec,
        planning_revisions=planning_revisions,
        planning_trees=planning_trees,
        source_revisions=source_revisions,
        source_trees=source_trees,
        reviewed_suites=reviewed_suites,
        receipt_path=receipt_path,
        config_path=config_path,
        inventory_path=inventory_path,
        report_path=report_path,
        pristine_receipt=receipt,
        pristine_config=pinned_config,
        pristine_inventory=inventory,
        pristine_report=report + "\n",
        pristine_logs=pristine_logs,
    )


@pytest.fixture
def bound_gate(
    captured_bundle: CapturedBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> CapturedBundle:
    bundle = captured_bundle
    bundle.restore()
    monkeypatch.setattr(gate, "REPO_ROOT", bundle.root)
    monkeypatch.setattr(gate, "CONFIG_PATH", bundle.config_path)
    monkeypatch.setattr(
        gate,
        "TASKBOARD_PATH",
        bundle.root / "docs/architecture/incremental_proof_sealer.todo.md",
    )
    monkeypatch.setattr(
        gate,
        "OBJECTIVES_PATH",
        bundle.root / "docs/architecture/incremental_proof_sealer.objectives.md",
    )
    monkeypatch.setattr(
        gate,
        "PLAN_PATH",
        bundle.root / "docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md",
    )
    monkeypatch.setattr(
        gate,
        "REPOSITORY_PATHS",
        {
            "accelerate": Path("."),
            "datasets": Path("ipfs_datasets_py"),
            "kit": Path("ipfs_kit_py"),
        },
    )
    monkeypatch.setattr(
        gate,
        "RUNNER_UNMATERIALIZED_GITLINKS",
        {"accelerate": {}, "datasets": {}, "kit": {}},
    )
    monkeypatch.setattr(gate, "ACCELERATE_REVISION", bundle.planning_revisions["accelerate"])
    monkeypatch.setattr(gate, "DATASETS_REVISION", bundle.planning_revisions["datasets"])
    monkeypatch.setattr(gate, "KIT_REVISION", bundle.planning_revisions["kit"])
    monkeypatch.setattr(gate, "PLANNING_TREES", bundle.planning_trees)
    monkeypatch.setattr(gate, "BASELINE_RECEIPT_SPECS", {"IPS-003": bundle.spec})
    monkeypatch.setattr(
        gate,
        "_reviewed_suite_registry",
        lambda errors: copy.deepcopy(bundle.reviewed_suites),
    )
    return bundle


def test_capture_pin_and_artifact_gate_end_to_end(bound_gate: CapturedBundle) -> None:
    result = gate.validate_artifact("IPS-003")
    assert result["valid"], result["errors"]


def _reanchor_receipt_command_paths(
    receipt: dict[str, Any], old_root: str, new_root: str, *, command_indexes: set[int]
) -> None:
    for index, command in enumerate(receipt["commands"]):
        if index not in command_indexes:
            continue
        command["argv"] = [token.replace(old_root, new_root) for token in command["argv"]]
        variables = command["environment"]["variables"]
        command["environment"]["variables"] = {
            key: value.replace(old_root, new_root) for key, value in variables.items()
        }
        command["command_digest"] = "sha256:" + hashlib.sha256(
            gate._canonical_json_bytes(
                {
                    "id": command["id"],
                    "argv": command["argv"],
                    "cwd": command["cwd"],
                    "environment": command["environment"],
                }
            )
        ).hexdigest()


def test_gate_accepts_pinned_receipt_from_deleted_historical_checkout(
    bound_gate: CapturedBundle,
) -> None:
    historical_root = str(bound_gate.root.parent / "deleted-capture-checkout")
    bound_gate.write_receipt(
        lambda receipt: _reanchor_receipt_command_paths(
            receipt,
            str(bound_gate.root),
            historical_root,
            command_indexes=set(range(len(receipt["commands"]))),
        )
    )

    errors: list[str] = []
    receipt = gate._validate_baseline_receipt("IPS-003", bound_gate.spec, errors)
    config = json.loads(bound_gate.config_path.read_text(encoding="utf-8"))

    assert historical_root != str(bound_gate.root)
    assert not Path(historical_root).exists()
    assert errors == []
    assert config["operator_baseline_receipts"]["IPS-003"] == (
        gate._expected_receipt_pin(bound_gate.spec, receipt)
    )


@pytest.mark.parametrize("mutation", ("mixed", "swapped", "relative", "traversal"))
def test_gate_rejects_ambiguous_or_unsafe_historical_capture_anchors(
    bound_gate: CapturedBundle,
    mutation: str,
) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        old_root = str(bound_gate.root)
        _reanchor_receipt_command_paths(
            receipt,
            old_root,
            "/historical/operator/one",
            command_indexes=set(range(len(receipt["commands"]))),
        )
        if mutation == "mixed":
            _reanchor_receipt_command_paths(
                receipt,
                "/historical/operator/one",
                "/historical/operator/two",
                command_indexes={1},
            )
        elif mutation == "swapped":
            first, second = receipt["commands"][:2]
            first_cache = next(
                token for token in first["argv"] if token.startswith("cache_dir=")
            )
            second_cache = next(
                token for token in second["argv"] if token.startswith("cache_dir=")
            )
            first["argv"][first["argv"].index(first_cache)] = second_cache
        elif mutation == "relative":
            receipt["commands"][0]["environment"]["variables"]["HOME"] = (
                "relative/capture/home"
            )
        else:
            command = receipt["commands"][0]
            workspace = command["workspace_relative_path"]
            command["environment"]["variables"]["HOME"] = (
                f"/historical/operator/one/../escape/{workspace}/home"
            )
        for command in receipt["commands"]:
            command["command_digest"] = "sha256:" + hashlib.sha256(
                gate._canonical_json_bytes(
                    {
                        "id": command["id"],
                        "argv": command["argv"],
                        "cwd": command["cwd"],
                        "environment": command["environment"],
                    }
                )
            ).hexdigest()

    bound_gate.write_receipt(mutate)
    errors: list[str] = []
    gate._validate_baseline_receipt("IPS-003", bound_gate.spec, errors)

    assert any(
        "historical capture root" in error
        or "canonical absolute capture path" in error
        or "exact capture-local suffix" in error
        for error in errors
    ), errors


@pytest.mark.parametrize(
    "schema",
    (
        "incremental-proof-sealer-baseline-receipt@1",
        "incremental-proof-sealer-baseline-receipt@2",
        "incremental-proof-sealer-baseline-receipt@3",
    ),
)
def test_gate_rejects_old_schema_even_with_recomputed_self_digest(
    bound_gate: CapturedBundle,
    schema: str,
) -> None:
    bound_gate.write_receipt(
        lambda receipt: receipt.__setitem__("schema_version", schema)
    )
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("schema_version" in error for error in result["errors"])


def test_gate_rejects_wrong_argv_with_recomputed_command_and_receipt_digests(
    bound_gate: CapturedBundle,
) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        command = receipt["commands"][0]
        command["argv"][-1] = "tests/evil.py"
        command["command_digest"] = "sha256:" + hashlib.sha256(
            gate._canonical_json_bytes(
                {
                    "id": command["id"],
                    "argv": command["argv"],
                    "cwd": command["cwd"],
                    "environment": command["environment"],
                }
            )
        ).hexdigest()

    bound_gate.write_receipt(mutate)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("protected reviewed suite" in error for error in result["errors"])


def test_gate_rejects_arbitrary_suite_digest_even_when_every_self_digest_matches(
    bound_gate: CapturedBundle,
) -> None:
    bound_gate.write_receipt(
        lambda receipt: receipt["commands"][0].__setitem__(
            "suite_definition_digest", _digest("forged-suite")
        )
    )
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("reviewed registry" in error for error in result["errors"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_gate_rejects_missing_duplicate_or_reordered_suites(
    bound_gate: CapturedBundle,
    mutation: str,
) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        commands = receipt["commands"]
        if mutation == "missing":
            receipt["commands"] = commands[:1]
        elif mutation == "duplicate":
            receipt["commands"] = [commands[0], copy.deepcopy(commands[0])]
        else:
            receipt["commands"] = list(reversed(commands))

    bound_gate.write_receipt(mutate)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any(
        "ordered command set" in error
        or "fixed ordered command set" in error
        or "repeats retained log" in error
        for error in result["errors"]
    ), result["errors"]


def test_gate_rejects_stale_nested_tested_revision(bound_gate: CapturedBundle) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        receipt["source_revision"] = bound_gate.planning_revisions["kit"]
        receipt["source_tree"] = bound_gate.planning_trees["kit"]
        receipt["source_revisions"]["kit"] = bound_gate.planning_revisions["kit"]
        receipt["source_trees"]["kit"] = bound_gate.planning_trees["kit"]

    bound_gate.write_receipt(mutate)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any(
        "does not match the captured control gitlink" in error
        for error in result["errors"]
    ), result["errors"]


def test_nested_inventory_parent_must_equal_receipt_tested_source(
    bound_gate: CapturedBundle,
) -> None:
    inventory = copy.deepcopy(bound_gate.pristine_inventory)
    inventory["inventory_worktree_parent_revision"] = (
        bound_gate.planning_revisions["kit"]
    )
    _write_json(bound_gate.inventory_path, inventory)

    result = gate.validate_artifact("IPS-003")

    assert not result["valid"]
    assert any(
        "nested inventory_worktree_parent_revision must equal the "
        "receipt-tested source revision" in error
        for error in result["errors"]
    ), result["errors"]


def test_nested_gate_rejects_post_capture_control_source_commit(
    bound_gate: CapturedBundle,
) -> None:
    prior = _git(bound_gate.root, "rev-parse", "HEAD")
    changed = bound_gate.root / "source_after_capture.py"
    _write(changed, "CHANGED = True\n")
    _git(bound_gate.root, "add", "source_after_capture.py")
    _git(bound_gate.root, "commit", "-q", "-m", "unreviewed control source")
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        _git(bound_gate.root, "checkout", "-q", "--detach", prior)
    assert not result["valid"]
    assert any("relevance-changing" in error for error in result["errors"])


def _isolate_structural_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_read", lambda *args: "")
    monkeypatch.setattr(gate, "_validate_tasks", lambda *args: {})
    monkeypatch.setattr(gate, "_validate_goals", lambda *args: None)
    monkeypatch.setattr(gate, "_validate_plan", lambda *args: None)
    monkeypatch.setattr(gate, "_validate_git_state", lambda *args: None)


def test_check_all_rejects_post_capture_source_before_synthesis(
    bound_gate: CapturedBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_structural_preflight(monkeypatch)
    prior = _git(bound_gate.root, "rev-parse", "HEAD")
    changed = bound_gate.root / "preflight_source_change.py"
    _write(changed, "CHANGED = True\n")
    _git(bound_gate.root, "add", changed.name)
    _git(bound_gate.root, "commit", "-q", "-m", "unreviewed preflight source")
    try:
        result = gate.validate(check_all=True)
    finally:
        _git(bound_gate.root, "checkout", "-q", "--detach", prior)
    assert not result["valid"]
    assert any("relevance-changing" in error for error in result["errors"])


def test_check_all_rejects_tampered_and_repinned_log_before_synthesis(
    bound_gate: CapturedBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_structural_preflight(monkeypatch)
    relative = next(iter(bound_gate.pristine_logs))
    tampered = bound_gate.pristine_logs[relative] + b"operator-tamper\n"
    (bound_gate.root / relative).write_bytes(tampered)

    def mutate(receipt: dict[str, Any]) -> None:
        command = next(
            item for item in receipt["commands"] if item["log"]["relative_path"] == relative
        )
        command["log"]["bytes"] = len(tampered)
        command["log"]["sha256"] = "sha256:" + hashlib.sha256(tampered).hexdigest()

    bound_gate.write_receipt(mutate)
    result = gate.validate(check_all=True)
    assert not result["valid"]
    assert any("final retained output line" in error for error in result["errors"])


def test_check_all_rejects_repinned_wrong_receipt_source(
    bound_gate: CapturedBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_structural_preflight(monkeypatch)

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["source_revision"] = bound_gate.planning_revisions["kit"]
        receipt["source_tree"] = bound_gate.planning_trees["kit"]
        receipt["source_revisions"]["kit"] = bound_gate.planning_revisions["kit"]
        receipt["source_trees"]["kit"] = bound_gate.planning_trees["kit"]

    bound_gate.write_receipt(mutate)
    result = gate.validate(check_all=True)
    assert not result["valid"]
    assert any("preflight kit delta" in error for error in result["errors"])


def test_terminal_mode_does_not_reapply_historical_source_relevance(
    bound_gate: CapturedBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_structural_preflight(monkeypatch)
    monkeypatch.setattr(gate, "_validated_baseline_synthesis", lambda *args: True)

    def forbidden_current_check(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("terminal mode reapplied historical source relevance")

    monkeypatch.setattr(gate, "_validate_current_baseline_sources", forbidden_current_check)
    result = gate.validate(check_all=True, check_terminal=True)
    assert result["valid"], result["errors"]


def _write_single_repository_synthesis_candidate(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    inventory = "docs/architecture/incremental_proof_sealer_inventory/accelerate.json"
    report = "docs/architecture/incremental_proof_sealer_inventory/accelerate.md"
    _init_repository(root, {"README.md": "source\n"})
    _write_json(root / inventory, {"static": "inventory"})
    _write(root / report, "static inventory\n")
    completion, _ = _commit(root, "inventory completion")
    spec = {
        "repository": "accelerate",
        "receipt": f"{gate.BASELINE_RECEIPT_ROOT}/accelerate.json",
        "inventory": inventory,
        "report": report,
    }
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(gate, "REPOSITORY_PATHS", {"accelerate": Path(".")})
    monkeypatch.setattr(gate, "BASELINE_RECEIPT_SPECS", {"IPS-001": spec})
    receipt_digest = _digest("operator-accelerate-receipt")
    pins = {
        "IPS-001": {
            "path": spec["receipt"],
            "receipt_digest": receipt_digest,
        }
    }
    matrix = {
        "schema_version": gate.BASELINE_SYNTHESIS_SCHEMA,
        "synthesis_worktree_parent_revision": completion,
        "baseline_receipts": pins,
        "inventory_artifacts": {
            "IPS-001": {
                "inventory": inventory,
                "report": report,
                "completion_revision": completion,
            }
        },
        "repository_authorities": gate.TRUST_BASELINE_AUTHORITIES,
        "proof_class_decisions": gate.TRUST_BASELINE_PROOF_CLASS_DECISIONS,
        "aggregation_decision": gate.TRUST_BASELINE_AGGREGATION_DECISION,
        "backend_decisions": gate.TRUST_BASELINE_BACKEND_DECISIONS,
        "trust_nonclaims": list(gate.TRUST_BASELINE_NONCLAIMS),
    }
    matrix_path = root / gate.BASELINE_SYNTHESIS_JSON
    _write_json(matrix_path, matrix)
    _write(
        root / gate.BASELINE_SYNTHESIS_REPORT,
        "\n".join(
            (
                "IPS-001",
                str(spec["receipt"]),
                receipt_digest,
                "pytest_execution_not_cryptographically_proven",
                *(
                    f"{key}={str(value).lower() if isinstance(value, bool) else value}"
                    for mapping in (
                        gate.TRUST_BASELINE_AUTHORITIES,
                        gate.TRUST_BASELINE_PROOF_CLASS_DECISIONS,
                        gate.TRUST_BASELINE_AGGREGATION_DECISION,
                        gate.TRUST_BASELINE_BACKEND_DECISIONS,
                    )
                    for key, value in mapping.items()
                ),
                *gate.TRUST_BASELINE_NONCLAIMS,
            )
        )
        + "\n",
    )
    return (
        {"operator_baseline_receipts": pins},
        {"IPS-001": {"receipt_digest": receipt_digest}},
        matrix_path,
    )


def test_ips004_candidate_semantics_accept_exact_current_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, receipts, _ = _write_single_repository_synthesis_candidate(
        tmp_path / "candidate-positive", monkeypatch
    )
    errors: list[str] = []

    accepted = gate._validated_baseline_synthesis(
        config, receipts, errors, candidate=True
    )

    assert accepted, errors
    assert errors == []


def test_ips004_candidate_semantics_reject_arbitrary_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, receipts, matrix_path = _write_single_repository_synthesis_candidate(
        tmp_path / "candidate-negative", monkeypatch
    )
    _write_json(matrix_path, {"arbitrary": "provider-owned assertion"})
    errors: list[str] = []

    accepted = gate._validated_baseline_synthesis(
        config, receipts, errors, candidate=True
    )

    assert not accepted
    assert any("trust baseline matrix" in error for error in errors)


def test_ips004_candidate_rejects_backend_or_claim_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, receipts, matrix_path = _write_single_repository_synthesis_candidate(
        tmp_path / "candidate-decision-drift", monkeypatch
    )
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["backend_decisions"]["existing_recursive_backend"] = "supported"
    matrix["trust_nonclaims"].remove("recursive_proof_verification_available")
    _write_json(matrix_path, matrix)
    errors: list[str] = []

    accepted = gate._validated_baseline_synthesis(
        config, receipts, errors, candidate=True
    )

    assert not accepted
    assert any("backend decisions" in error for error in errors)
    assert any("trust nonclaims" in error for error in errors)


@pytest.mark.parametrize("retained", ["matrix", "report"])
def test_ips004_partial_candidate_is_never_treated_as_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retained: str,
) -> None:
    root = tmp_path / f"partial-{retained}"
    _init_repository(root, {"README.md": "source\n"})
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    relative = (
        gate.BASELINE_SYNTHESIS_JSON
        if retained == "matrix"
        else gate.BASELINE_SYNTHESIS_REPORT
    )
    _write(root / relative, "{}\n" if retained == "matrix" else "partial\n")
    errors: list[str] = []

    accepted = gate._validated_baseline_synthesis({}, {}, errors, candidate=True)

    assert not accepted
    assert errors == ["IPS-004 baseline synthesis artifacts are only partially present"]


def test_committed_ips004_cannot_launder_a_precursor_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "historical-laundering"
    inventory = "docs/architecture/incremental_proof_sealer_inventory/accelerate.json"
    inventory_report = "docs/architecture/incremental_proof_sealer_inventory/accelerate.md"
    receipt_path = f"{gate.BASELINE_RECEIPT_ROOT}/accelerate.json"
    pre_capture = _pre_capture_config()
    captured, _ = _init_repository(
        root,
        {
            gate.POST_CAPTURE_PIN_CONFIG_PATH: json.dumps(pre_capture),
            "README.md": "captured source\n",
        },
    )
    receipt_digest = _digest("historical-receipt")
    pin = {
        "path": receipt_path,
        "receipt_digest": receipt_digest,
        "retained_log_paths": [],
    }
    pinned = copy.deepcopy(pre_capture)
    pinned["operator_baseline_receipts"] = {"IPS-001": pin}
    pinned["protected_paths"] = [*gate.BASE_PROTECTED_PATHS, receipt_path]
    _write_json(root / gate.POST_CAPTURE_PIN_CONFIG_PATH, pinned)
    _write_json(root / receipt_path, {"operator": "fixed"})
    _commit(root, "pin evidence")
    _write(root / "source.py", "UNREVIEWED = True\n")
    _commit(root, "unreviewed source between capture and inventory")
    _write_json(root / inventory, {"classification": "static"})
    _write(root / inventory_report, "static inventory\n")
    completion, _ = _commit(root, "inventory-only completion")
    matrix = {
        "schema_version": gate.BASELINE_SYNTHESIS_SCHEMA,
        "synthesis_worktree_parent_revision": completion,
        "baseline_receipts": {"IPS-001": pin},
        "inventory_artifacts": {
            "IPS-001": {
                "inventory": inventory,
                "report": inventory_report,
                "completion_revision": completion,
            }
        },
    }
    _write_json(root / gate.BASELINE_SYNTHESIS_JSON, matrix)
    _write(
        root / gate.BASELINE_SYNTHESIS_REPORT,
        f"IPS-001\n{receipt_path}\n{receipt_digest}\n"
        "pytest_execution_not_cryptographically_proven\n",
    )
    _commit(root, "committed synthesis")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(gate, "REPOSITORY_PATHS", {"accelerate": Path(".")})
    monkeypatch.setattr(
        gate,
        "BASELINE_RECEIPT_SPECS",
        {
            "IPS-001": {
                "repository": "accelerate",
                "receipt": receipt_path,
                "inventory": inventory,
                "report": inventory_report,
            }
        },
    )
    errors: list[str] = []

    accepted = gate._validated_baseline_synthesis(
        {"operator_baseline_receipts": {"IPS-001": pin}},
        {"IPS-001": {"execution_head": captured, "receipt_digest": receipt_digest}},
        errors,
    )

    assert not accepted
    assert any(
        "relevance-changing" in error and "source.py" in error
        for error in errors
    ), errors


def test_ips004_artifact_route_requires_candidate_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(gate, "_load_json", lambda *args: {"pins": "fixed"})
    monkeypatch.setattr(gate, "_validate_config", lambda *args: None)
    monkeypatch.setattr(
        gate,
        "_validate_operator_baseline_bundle",
        lambda config, errors, *, enforce_current_sources: (
            calls.append((enforce_current_sources, False))
            or {task_id: {} for task_id in gate.BASELINE_RECEIPT_SPECS}
        ),
    )

    def validate_candidate(config: Any, receipts: Any, errors: Any, *, candidate: bool) -> bool:
        calls.append((False, candidate))
        return True

    monkeypatch.setattr(gate, "_validated_baseline_synthesis", validate_candidate)

    result = gate.validate_artifact("IPS-004")

    assert result["valid"], result["errors"]
    assert calls == [(True, False), (False, True)]


def test_no_argument_cli_runs_phase_aware_full_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_validate(*, check_all: bool, check_terminal: bool = False) -> dict[str, Any]:
        calls.append((check_all, check_terminal))
        return {"valid": True, "errors": []}

    monkeypatch.setattr(gate, "validate", fake_validate)

    assert gate.main([]) == 0
    assert calls == [(True, False)]
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_bootstrap_config_accepts_only_exact_empty_pin_phase() -> None:
    config = json.loads(
        (ROOT / "config/agent_supervisor_incremental_proof_sealer_scheduler.json").read_text(
            encoding="utf-8"
        )
    )
    config["operator_baseline_receipts"] = {}
    config["protected_paths"] = list(gate.BASE_PROTECTED_PATHS)
    bootstrap_errors: list[str] = []
    preflight_errors: list[str] = []

    gate._validate_config(config, bootstrap_errors, bootstrap=True)
    gate._validate_config(config, preflight_errors)

    assert bootstrap_errors == []
    assert any("operator_baseline_receipts task ids" in error for error in preflight_errors)
    forged = copy.deepcopy(config)
    forged["operator_baseline_receipts"] = {"IPS-001": {}}
    forged_errors: list[str] = []
    gate._validate_config(forged, forged_errors, bootstrap=True)
    assert "bootstrap operator_baseline_receipts must be exactly empty" in forged_errors


def test_check_bootstrap_cli_dispatches_phase_specific_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        gate,
        "validate_bootstrap",
        lambda: calls.append("bootstrap") or {"valid": True, "errors": []},
    )

    assert gate.main(["--check-bootstrap"]) == 0
    assert calls == ["bootstrap"]
    assert json.loads(capsys.readouterr().out)["valid"] is True


@pytest.mark.parametrize("unexpected", [".capture.lock", ".accelerate.json.partial"])
def test_bootstrap_rejects_stale_capture_lock_and_orphan_receipt_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unexpected: str,
) -> None:
    root = tmp_path / "bootstrap-root"
    receipt_root = root / gate.BASELINE_RECEIPT_ROOT
    (receipt_root / "logs").mkdir(parents=True)
    (receipt_root / "work").mkdir()
    _write(receipt_root / unexpected, "ambiguous\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_bootstrap_receipt_root(errors)

    assert any("unexpected entry" in error for error in errors)


def test_all_post_capture_gates_reject_stale_capture_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "post-capture-lock"
    lock = root / gate.BASELINE_RECEIPT_ROOT / ".capture.lock"
    _write(lock, "ambiguous\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._validate_no_capture_lock(errors)

    assert errors == [
        "operator baseline evidence is ambiguous while stale .capture.lock exists"
    ]


def test_operator_bundle_rejects_mixed_capture_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = {
        "IPS-001": {"repository": "accelerate"},
        "IPS-002": {"repository": "datasets"},
    }
    receipts = {
        "IPS-001": {
            "capture_id": "20260811T130000.000000Z-1",
            "source_revisions": {"accelerate": "a", "datasets": "b", "kit": "c"},
            "ignored_sensitive_inputs": {"policy_id": "fixed"},
            "commands": [],
        },
        "IPS-002": {
            "capture_id": "20260811T130000.000000Z-2",
            "source_revisions": {"accelerate": "a", "datasets": "b", "kit": "c"},
            "ignored_sensitive_inputs": {"policy_id": "fixed"},
            "commands": [],
        },
    }
    monkeypatch.setattr(gate, "BASELINE_RECEIPT_SPECS", specs)

    def validate_receipt(
        task_id: str,
        _spec: dict[str, Any],
        _errors: list[str],
        *,
        bundle_capture_roots: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if bundle_capture_roots is not None:
            bundle_capture_roots[task_id] = gate.PurePosixPath("/historical")
        return receipts[task_id]

    monkeypatch.setattr(
        gate,
        "_validate_baseline_receipt",
        validate_receipt,
    )
    monkeypatch.setattr(
        gate,
        "_expected_receipt_pin",
        lambda spec, receipt: {"capture_id": receipt["capture_id"]},
    )
    config = {
        "operator_baseline_receipts": {
            task_id: {"capture_id": receipt["capture_id"]}
            for task_id, receipt in receipts.items()
        }
    }
    errors: list[str] = []

    gate._validate_operator_baseline_bundle(
        config, errors, enforce_current_sources=False
    )

    assert errors == ["operator baseline receipts do not share one exact capture id"]


def test_gate_rejects_pytest_no_tests_collected_exit_code(
    bound_gate: CapturedBundle,
) -> None:
    bound_gate.write_receipt(
        lambda receipt: receipt["commands"][0].__setitem__("exit_code", 5)
    )
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("closed pytest exit code" in error for error in result["errors"])


def test_gate_rejects_inconsistent_complete_collection_count(
    bound_gate: CapturedBundle,
) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        command = receipt["commands"][0]
        command["collected_count"] = 99

    bound_gate.write_receipt(mutate)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any(
        "collection and outcome counts disagree" in error
        or "collected_count does not match" in error
        for error in result["errors"]
    )


def test_gate_rejects_log_tampering(bound_gate: CapturedBundle) -> None:
    relative = next(iter(bound_gate.pristine_logs))
    with (bound_gate.root / relative).open("ab") as handle:
        handle.write(b"tampered\n")
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("full retained log" in error for error in result["errors"])


def test_gate_rejects_oversized_retained_log_before_reading_it(
    bound_gate: CapturedBundle,
) -> None:
    relative = next(iter(bound_gate.pristine_logs))
    with (bound_gate.root / relative).open("wb") as handle:
        handle.truncate(gate.BASELINE_MAX_LOG_BYTES + 1)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("fixed 64-MiB bound" in error for error in result["errors"])


def test_gate_rejects_receipt_digest_tampering(bound_gate: CapturedBundle) -> None:
    receipt = copy.deepcopy(bound_gate.pristine_receipt)
    receipt["receipt_digest"] = _digest("tampered-receipt")
    _write_json(bound_gate.receipt_path, receipt)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("canonical receipt content" in error for error in result["errors"])


def test_gate_rejects_oversized_receipt_before_reading_it(
    bound_gate: CapturedBundle,
) -> None:
    with bound_gate.receipt_path.open("wb") as handle:
        handle.truncate(gate.BASELINE_MAX_RECEIPT_BYTES + 1)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("fixed two-MiB bound" in error for error in result["errors"])


def test_gate_rejects_config_pin_tampering(bound_gate: CapturedBundle) -> None:
    config = copy.deepcopy(bound_gate.pristine_config)
    config["operator_baseline_receipts"]["IPS-003"]["receipt_digest"] = _digest(
        "wrong-pin"
    )
    _write_json(bound_gate.config_path, config)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("receipt pin is not exact" in error for error in result["errors"])


def test_gate_rejects_registry_digest_config_tampering(
    bound_gate: CapturedBundle,
) -> None:
    config = copy.deepcopy(bound_gate.pristine_config)
    config["baseline_suite_registry_digest"] = _digest("attacker-registry")
    _write_json(bound_gate.config_path, config)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("baseline_suite_registry_digest" in error for error in result["errors"])


def test_gate_rejects_ignored_input_binding_tamper_with_recomputed_receipt_digest(
    bound_gate: CapturedBundle,
) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        receipt["ignored_sensitive_inputs"]["repositories"]["kit"] = {
            "count": 0,
            "digest": _digest("forged ignored-input binding"),
        }

    bound_gate.write_receipt(mutate)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("captured zero-input binding" in error for error in result["errors"])


def test_gate_rejects_unreviewed_git_environment_policy(
    bound_gate: CapturedBundle,
) -> None:
    bound_gate.write_receipt(
        lambda receipt: receipt.__setitem__(
            "git_environment_policy_id", "attacker-selected-git-environment@1"
        )
    )
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("git_environment_policy_id is not reviewed" in error for error in result["errors"])


def test_gate_rejects_current_ignored_trust_sensitive_input(
    bound_gate: CapturedBundle,
) -> None:
    ignored = bound_gate.root / "ipfs_kit_py/build/proving_key.bin"
    ignored.parent.mkdir(parents=True, exist_ok=True)
    ignored.write_bytes(b"not a proving key; adversarial fixture only\n")
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        ignored.unlink()
        ignored.parent.rmdir()
    assert not result["valid"]
    assert any("undeclared ignored regular input" in error for error in result["errors"])


def test_gate_rejects_current_ignored_trust_sensitive_symlink(
    bound_gate: CapturedBundle,
    tmp_path: Path,
) -> None:
    target = tmp_path / "test-only.key"
    target.write_bytes(b"test-only external target\n")
    ignored = bound_gate.root / "ipfs_kit_py/verification.key"
    ignored.symlink_to(target)
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        ignored.unlink()
    assert not result["valid"]
    assert any("undeclared ignored symlink input" in error for error in result["errors"])


def test_gate_rejects_current_ignored_trust_sensitive_fifo(
    bound_gate: CapturedBundle,
) -> None:
    fifo = bound_gate.root / "ipfs_kit_py/build/proving_key.bin"
    fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(fifo)
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        fifo.unlink()
        fifo.parent.rmdir()
    assert not result["valid"]
    assert any("undeclared ignored special input" in error for error in result["errors"])


def test_gate_rejects_ignored_sitecustomize_startup_hook(
    bound_gate: CapturedBundle,
) -> None:
    hook = bound_gate.root / "ipfs_kit_py/sitecustomize.py"
    hook.write_text("raise RuntimeError('must never load')\n", encoding="utf-8")
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        hook.unlink()
    assert not result["valid"]
    assert any("sitecustomize.py" in error for error in result["errors"])


def test_gate_allows_redirected_regular_pytest_cache(
    bound_gate: CapturedBundle,
) -> None:
    cache = bound_gate.root / "ipfs_kit_py/.pytest_cache/cache.bin"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"unavailable to the fixed redirected pytest cache\n")
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        cache.unlink()
        cache.parent.rmdir()
    assert result["valid"], result["errors"]


@pytest.mark.parametrize(
    "relative",
    (
        "conftest.py",
        "target/release/nargo",
        "target/release/provekit",
        "build/rapidsnark",
        "build/snarkjs",
        "trusted_setup.ptau",
        "circuit.wasm",
        "build/backend.so",
    ),
)
def test_gate_deny_by_default_rejects_future_ignored_execution_inputs(
    bound_gate: CapturedBundle,
    relative: str,
) -> None:
    repository = bound_gate.root / "ipfs_kit_py"
    exclude = repository / ".git/info/exclude"
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write(f"/{relative}\n")
    candidate = repository / relative
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"adversarial ignored execution input\n")
    try:
        errors: list[str] = []
        gate._validate_current_trust_sensitive_ignored_inputs(errors)
    finally:
        candidate.unlink()
        parent = candidate.parent
        while parent != repository and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    assert any(relative in error and "undeclared ignored" in error for error in errors)


@pytest.mark.parametrize(
    "relative",
    (
        ".hypothesis/examples/case",
        "__pycache__/module.cpython-312.pyc",
        ".ruff_cache/entry",
    ),
)
def test_gate_allows_only_explicit_redirected_cache_roots(
    bound_gate: CapturedBundle,
    relative: str,
) -> None:
    repository = bound_gate.root / "ipfs_kit_py"
    candidate = repository / relative
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"irrelevant redirected cache fixture\n")
    try:
        errors: list[str] = []
        gate._validate_current_trust_sensitive_ignored_inputs(errors)
    finally:
        candidate.unlink()
        parent = candidate.parent
        while parent != repository and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    assert errors == []


def _init_opaque_gitlink_fixture(root: Path) -> tuple[Path, str]:
    origin = root.parent / f"{root.name}-reviewed-origin"
    reviewed_oid, _ = _init_repository(origin, {"reviewed.py": "VALUE = 1\n"})
    _init_repository(root, {"README.md": "outer\n"})
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        "vendor/reviewed",
    )
    _commit(root, "bind reviewed opaque gitlink")
    return origin, reviewed_oid


@pytest.mark.parametrize("materialization", ("absent", "clean"))
def test_current_scan_prunes_only_exact_index_bound_opaque_gitlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    materialization: str,
) -> None:
    root = tmp_path / f"opaque-{materialization}"
    _origin, reviewed_oid = _init_opaque_gitlink_fixture(root)
    if materialization == "absent":
        _git(root, "submodule", "deinit", "-f", "--", "vendor/reviewed")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(gate, "REPOSITORY_PATHS", {"accelerate": Path(".")})
    monkeypatch.setattr(
        gate,
        "RUNNER_UNMATERIALIZED_GITLINKS",
        {"accelerate": {"vendor/reviewed": reviewed_oid}},
    )
    errors: list[str] = []

    gate._validate_current_trust_sensitive_ignored_inputs(errors)

    assert errors == []


@pytest.mark.parametrize(
    "drift",
    ("unknown_gitlink", "oid", "dirty_materialization", "nonrepo_materialization"),
)
def test_current_scan_rejects_opaque_gitlink_trust_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    root = tmp_path / f"opaque-drift-{drift}"
    _origin, reviewed_oid = _init_opaque_gitlink_fixture(root)
    expected_oid = reviewed_oid
    if drift == "unknown_gitlink":
        unknown_origin = tmp_path / "unknown-origin"
        _init_repository(unknown_origin, {"unknown.py": "VALUE = 2\n"})
        _git(
            root,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(unknown_origin),
            "vendor/unknown",
        )
        _commit(root, "add unknown opaque gitlink")
    elif drift == "oid":
        expected_oid = "0" * 40
    elif drift == "dirty_materialization":
        _write(root / "vendor/reviewed/untracked.py", "UNTRUSTED = True\n")
    else:
        _git(root, "submodule", "deinit", "-f", "--", "vendor/reviewed")
        _write(root / "vendor/reviewed/untrusted.bin", "not a repository\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(gate, "REPOSITORY_PATHS", {"accelerate": Path(".")})
    monkeypatch.setattr(
        gate,
        "RUNNER_UNMATERIALIZED_GITLINKS",
        {"accelerate": {"vendor/reviewed": expected_oid}},
    )
    errors: list[str] = []

    gate._validate_current_trust_sensitive_ignored_inputs(errors)

    assert errors
    assert any(
        token in error
        for error in errors
        for token in (
            "unknown opaque gitlinks",
            "index mode/OID drifted",
            "worktree contains staged, unstaged, untracked, or ignored drift",
            "non-repository directory",
        )
    ), errors


def test_current_scan_rejects_unknown_nested_git_admin_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "nested-git-admin"
    _init_repository(root, {"README.md": "source\n"})
    _write(root / "ordinary/.git", "gitdir: /untrusted\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(gate, "REPOSITORY_PATHS", {"accelerate": Path(".")})
    monkeypatch.setattr(
        gate, "RUNNER_UNMATERIALIZED_GITLINKS", {"accelerate": {}}
    )
    errors: list[str] = []

    gate._validate_current_trust_sensitive_ignored_inputs(errors)

    assert any("unknown nested Git administration entry" in error for error in errors)


def test_gate_rejects_log_traversal(bound_gate: CapturedBundle) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        receipt["commands"][0]["log"]["relative_path"] = (
            f"{gate.BASELINE_LOG_ROOT}/../escaped.log"
        )

    bound_gate.write_receipt(mutate)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("canonical repository-relative path" in error for error in result["errors"])


def test_gate_rejects_symlinked_retained_log(
    bound_gate: CapturedBundle,
    tmp_path: Path,
) -> None:
    relative = next(iter(bound_gate.pristine_logs))
    path = bound_gate.root / relative
    path.unlink()
    external = tmp_path / "external.log"
    external.write_bytes(bound_gate.pristine_logs[relative])
    path.symlink_to(external)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("symlink" in error for error in result["errors"])


def test_gate_rejects_final_file_swap_immediately_before_secure_open(
    bound_gate: CapturedBundle,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    relative = next(iter(bound_gate.pristine_logs))
    path = bound_gate.root / relative
    external = tmp_path / "swapped.log"
    external.write_bytes(bound_gate.pristine_logs[relative])
    real_open = os.open
    swapped = False

    def swap_then_open(
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and str(target) == path.name and dir_fd is not None:
            swapped = True
            path.unlink()
            path.symlink_to(external)
        return real_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(gate.os, "open", swap_then_open)
    result = gate.validate_artifact("IPS-003")
    assert swapped
    assert not result["valid"]
    assert any("symlink or unsafe component" in error for error in result["errors"])


def test_gate_rejects_ancestor_swap_immediately_before_secure_open(
    bound_gate: CapturedBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = bound_gate.root / gate.BASELINE_LOG_ROOT
    real_logs = logs.with_name("retained-logs-swap-target")
    real_open = os.open
    swapped = False

    def swap_then_open(
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and str(target) == "logs" and dir_fd is not None:
            swapped = True
            logs.rename(real_logs)
            logs.symlink_to(real_logs, target_is_directory=True)
        return real_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(gate.os, "open", swap_then_open)
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        if logs.is_symlink():
            logs.unlink()
        if real_logs.exists():
            real_logs.rename(logs)
    assert swapped
    assert not result["valid"]
    assert any("symlink or unsafe component" in error for error in result["errors"])


def test_gate_rejects_intermediate_log_directory_symlink_to_inside_repository(
    bound_gate: CapturedBundle,
) -> None:
    logs = bound_gate.root / gate.BASELINE_LOG_ROOT
    real_logs = logs.with_name("retained-logs-real")
    logs.rename(real_logs)
    logs.symlink_to(real_logs, target_is_directory=True)
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        logs.unlink()
        real_logs.rename(logs)
    assert not result["valid"]
    assert any("symlink or unsafe component" in error for error in result["errors"])


def test_gate_rejects_intermediate_receipt_directory_symlink_to_inside_repository(
    bound_gate: CapturedBundle,
) -> None:
    receipt_root = bound_gate.root / gate.BASELINE_RECEIPT_ROOT
    real_receipt_root = receipt_root.with_name("baseline-receipts-real")
    receipt_root.rename(real_receipt_root)
    receipt_root.symlink_to(real_receipt_root, target_is_directory=True)
    try:
        result = gate.validate_artifact("IPS-003")
    finally:
        receipt_root.unlink()
        real_receipt_root.rename(receipt_root)
    assert not result["valid"]
    assert any("symlink or unsafe component" in error for error in result["errors"])


def test_gate_rejects_inventory_execution_aliases(bound_gate: CapturedBundle) -> None:
    inventory = copy.deepcopy(bound_gate.pristine_inventory)
    inventory["provider_summary"] = {
        "passed_count": 999,
        "failed_count": 0,
        "claim": "all test cases were green and succeeded",
    }
    _write_json(bound_gate.inventory_path, inventory)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("provider-owned" in error or "copies protected" in error for error in result["errors"])


def test_gate_rejects_future_completion_commit_self_embedding(
    bound_gate: CapturedBundle,
) -> None:
    inventory = copy.deepcopy(bound_gate.pristine_inventory)
    inventory["final_task_commit"] = bound_gate.source_revisions["kit"]
    _write_json(bound_gate.inventory_path, inventory)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("future completion commit" in error for error in result["errors"])


def test_gate_rejects_markdown_execution_success_claim(bound_gate: CapturedBundle) -> None:
    _write(
        bound_gate.report_path,
        bound_gate.pristine_report + "All test cases were green and succeeded.\n",
    )
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("Markdown inventory restates operator outcomes" in error for error in result["errors"])


def test_gate_rejects_markdown_outcome_alias_counts(bound_gate: CapturedBundle) -> None:
    _write(
        bound_gate.report_path,
        bound_gate.pristine_report + "999 successful cases; zero red cases.\n",
    )
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("Markdown inventory" in error for error in result["errors"])


def test_gate_rejects_receipt_directory_in_protected_paths(
    bound_gate: CapturedBundle,
) -> None:
    config = copy.deepcopy(bound_gate.pristine_config)
    config["protected_paths"].append(gate.BASELINE_RECEIPT_ROOT)
    _write_json(bound_gate.config_path, config)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("launch-invalid receipt directory" in error for error in result["errors"])


def test_gate_rejects_duplicate_protected_file_path(bound_gate: CapturedBundle) -> None:
    config = copy.deepcopy(bound_gate.pristine_config)
    config["protected_paths"].append(config["protected_paths"][-1])
    _write_json(bound_gate.config_path, config)
    result = gate.validate_artifact("IPS-003")
    assert not result["valid"]
    assert any("unique exact-file list" in error for error in result["errors"])


def _benchmark_fixture_payload() -> dict[str, Any]:
    revisions = {repository: character * 40 for repository, character in zip(("accelerate", "datasets", "kit"), "abc")}
    trees = {repository: character * 40 for repository, character in zip(("accelerate", "datasets", "kit"), "def")}
    transitions: list[dict[str, Any]] = []
    prior_root: str | None = None
    for index, scenario in enumerate(gate.BENCHMARK_SCENARIOS):
        full = index in gate.BENCHMARK_FULL_TRANSITIONS
        required = 10
        invalidated = 10 if full else 2
        added = 0
        reused = required - invalidated
        full_cost = 100.0
        incremental_cost = 100.0 if full else 20.0
        root = _digest(f"benchmark-root-{index}")
        metrics = {
            "leaf_proving_seconds": incremental_cost / 10.0,
            "aggregation_seconds": 1.0,
            "prover_cpu_seconds": incremental_cost,
            "prover_gpu_seconds": 0.0,
            "peak_memory_bytes": 1024.0,
            "proof_size_bytes": 2048.0,
            "seal_size_bytes": 4096.0,
            "storage_growth_bytes": 512.0,
            "seal_verification_seconds": 0.1,
            "wall_clock_seconds": incremental_cost / 2.0,
            "full_proof_cost": full_cost,
            "incremental_proof_cost": incremental_cost,
        }
        transitions.append(
            {
                "index": index,
                "scenario": scenario,
                "repository_revision": hashlib.sha1(f"commit-{index}".encode()).hexdigest(),
                "parent_seal": prior_root,
                "seal_status": "sealed_full" if full else "sealed_incremental",
                "required_units": required,
                "reused_units": reused,
                "invalidated_units": invalidated,
                "added_units": added,
                "removed_units": 0,
                "newly_proved_units": invalidated + added,
                "unit_count_provenance": "observed_planner_output",
                "cache_hit_rate": reused / required,
                **metrics,
                "metric_provenance": {metric: "measured" for metric in gate.BENCHMARK_METRICS},
                "measurement_provenance": "measured",
                "compute_saved_percent": (full_cost - incremental_cost) / full_cost * 100.0,
                "chain_depth": 0 if full else 1,
                "fallback_reason": f"reviewed full checkpoint at {index}" if full else None,
                "full_seal_root": root,
                "incremental_seal_root": root,
                "deterministic_roots_match": True,
                "simulated_required_units": 0,
                "rejected_attempts": (
                    [{"kind": "wrong_parent", "terminal_status": "stale_parent"}]
                    if index == 37
                    else []
                ),
            }
        )
        prior_root = root
    return {
        "schema_version": gate.BENCHMARK_SCHEMA,
        "benchmark_id": gate.BENCHMARK_ID,
        "seed": gate.BENCHMARK_SEED,
        "transition_count": gate.BENCHMARK_TRANSITION_COUNT,
        "benchmark_worktree_parent_revision": revisions["accelerate"],
        "source_revisions": revisions,
        "source_trees": trees,
        "execution_context": {
            "runner_id": "protected-board-benchmark-runner@1",
            "argv": gate._benchmark_workload_argv(),
            "process_observed": True,
            "test_execution_cryptographically_proven": False,
            "claim": "benchmark_process_observed_metrics_retain_per_metric_provenance",
        },
        "capabilities": {
            "real_prover_available": False,
            "recursive_verification_available": False,
            "gpu_available": False,
            "notes": "Fixture measurements are process observations, not production proof claims.",
        },
        "transitions": transitions,
    }


def _write_benchmark_fixture(root: Path, payload: dict[str, Any]) -> None:
    _write_json(root / gate.BENCHMARK_JSON, payload)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=gate.BENCHMARK_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in payload["transitions"]:
        writer.writerow(
            {
                field: (
                    ""
                    if row[field] is None
                    else str(row[field]).lower()
                    if isinstance(row[field], bool)
                    else row[field]
                )
                for field in gate.BENCHMARK_CSV_FIELDS
            }
        )
    _write(root / gate.BENCHMARK_CSV, stream.getvalue())


def test_ips053_closed_benchmark_and_csv_accept_exact_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _benchmark_fixture_payload()
    root = tmp_path / "benchmark-positive"
    _write_benchmark_fixture(root, payload)
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate,
        "_validate_parent_bound_output_lifecycle",
        lambda **kwargs: None,
    )
    errors: list[str] = []

    gate._validate_benchmark_artifacts(errors)

    assert errors == []


def test_ips053_rejects_recomputed_savings_and_csv_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _benchmark_fixture_payload()
    payload["transitions"][1]["compute_saved_percent"] = 99.0
    root = tmp_path / "benchmark-negative"
    _write_benchmark_fixture(root, payload)
    csv_path = root / gate.BENCHMARK_CSV
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("localized private source edit", "forged scenario", 1), encoding="utf-8")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate,
        "_validate_parent_bound_output_lifecycle",
        lambda **kwargs: None,
    )
    errors: list[str] = []

    gate._validate_benchmark_artifacts(errors)

    assert any("compute_saved_percent arithmetic" in error for error in errors)
    assert any("CSV row 01 field scenario" in error for error in errors)


def test_ips054_summary_must_derive_from_raw_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _benchmark_fixture_payload()
    root = tmp_path / "summary-negative"
    _write_benchmark_fixture(root, payload)
    benchmark_raw = (root / gate.BENCHMARK_JSON).read_bytes()
    expected = gate._derived_benchmark_summary(
        payload, "sha256:" + hashlib.sha256(benchmark_raw).hexdigest()
    )
    summary = {**expected, "limitations": ["Fixture data is not production evidence."]}
    summary["average_reuse_rate"]["value_percent"] = 99.0
    _write_json(root / gate.BENCHMARK_SUMMARY_JSON, summary)
    _write(root / gate.BENCHMARK_REPORT, "intentionally incomplete\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate,
        "_validate_parent_bound_output_lifecycle",
        lambda **kwargs: None,
    )
    errors: list[str] = []

    gate._validate_benchmark_summary(errors)

    assert any("average_reuse_rate does not derive exactly" in error for error in errors)
    assert any("benchmark report omits" in error for error in errors)


def test_validation_metadata_rejects_shell_control_operators() -> None:
    errors: list[str] = []
    argv = gate._validation_argv(
        "IPS-test", "python fixed.py && python forged.py", errors
    )
    assert argv
    assert errors == [
        "IPS-test validation contains a forbidden shell control operator"
    ]


def test_fixed_runner_bounds_subprocess_output(tmp_path: Path) -> None:
    status, exit_code, _, output = gate._run_observed_process(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 65536)"],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=10,
        maximum_output_bytes=1024,
    )
    assert status == "output_limit"
    assert exit_code is None
    assert len(output) == 1024


def _assert_process_terminated(pid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists():
            return
        if stat_path.read_text(encoding="ascii").split()[2] == "Z":
            return
        time.sleep(0.01)
    pytest.fail(f"protected runner left residual process {pid} alive")


def test_fixed_runner_terminates_and_rejects_residual_child(
    tmp_path: Path,
) -> None:
    program = """
import subprocess
import sys

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import os,time; os.close(1); os.close(2); time.sleep(30)",
])
print(child.pid, flush=True)
"""
    status, exit_code, _, output = gate._run_observed_process(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=5,
    )
    child_pid = int(output.splitlines()[0])
    assert status == "residual_process_terminated"
    assert exit_code is None
    _assert_process_terminated(child_pid)


def test_fixed_runner_terminates_detached_double_fork_descendant(
    tmp_path: Path,
) -> None:
    program = """
import os
import time

first = os.fork()
if first:
    os._exit(0)
os.setsid()
second = os.fork()
if second:
    os._exit(0)
print(os.getpid(), flush=True)
os.close(1)
os.close(2)
time.sleep(30)
"""
    status, exit_code, _, output = gate._run_observed_process(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=5,
    )
    descendant_pid = int(output.splitlines()[0])

    assert status == "residual_process_terminated"
    assert exit_code is None
    _assert_process_terminated(descendant_pid)


def test_fixed_runner_contains_descendants_after_bound_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "RUNNER_MAX_DESCENDANT_PROCESSES", 1)
    program = """
import subprocess
import sys

children = [
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(3)
]
print(" ".join(str(child.pid) for child in children), flush=True)
"""
    status, exit_code, _, output = gate._run_observed_process(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=5,
    )
    child_pids = [int(value) for value in output.split()]

    assert status == "cleanup_failed"
    assert exit_code is None
    assert len(child_pids) == 3
    for child_pid in child_pids:
        _assert_process_terminated(child_pid)


def test_task_gate_enforces_ordered_same_submodule_writers() -> None:
    text = (ROOT / "docs/architecture/incremental_proof_sealer.todo.md").read_text(
        encoding="utf-8"
    )
    text = text.replace(
        "- Depends on: IPS-047, IPS-026, IPS-044",
        "- Depends on: IPS-047, IPS-026",
        1,
    )
    config = json.loads(
        (ROOT / "config/agent_supervisor_incremental_proof_sealer_scheduler.json").read_text(
            encoding="utf-8"
        )
    )
    errors: list[str] = []

    gate._validate_tasks(text, config, errors)

    assert any(
        "same-submodule writers IPS-044 and IPS-050 are unordered" in error
        for error in errors
    )


def test_task_gate_enforces_outputs_equal_predicted_files() -> None:
    text = (ROOT / "docs/architecture/incremental_proof_sealer.todo.md").read_text(
        encoding="utf-8"
    )
    text = text.replace(
        "- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/forest_codec.py, ipfs_datasets_py/tests/unit/logic/zkp/incremental_sealing/test_forest_codec.py, ipfs_datasets_py/tests/fixtures/incremental_proof_sealer/forest_vectors.json",
        "- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/zkp/incremental_sealing/forest_codec.py",
        1,
    )
    config = json.loads(
        (ROOT / "config/agent_supervisor_incremental_proof_sealer_scheduler.json").read_text(
            encoding="utf-8"
        )
    )
    errors: list[str] = []

    gate._validate_tasks(text, config, errors)

    assert "IPS-011 Outputs must exactly equal Predicted files" in errors


def test_materialization_tasks_declare_exact_bounded_text_envelopes() -> None:
    text = (ROOT / "docs/architecture/incremental_proof_sealer.todo.md").read_text(
        encoding="utf-8"
    )
    config = json.loads(
        (
            ROOT
            / "config/agent_supervisor_incremental_proof_sealer_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    errors: list[str] = []

    gate._validate_tasks(text, config, errors)

    assert errors == []
    for envelope in (
        gate.BENCHMARK_PROPOSAL_ENVELOPE,
        gate.RELEASE_PROPOSAL_ENVELOPE,
    ):
        assert envelope["schema"].endswith("task-artifact-envelope@1")
        assert "allow_binary" not in envelope
        canonical = json.dumps(
            envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        assert f"- Proposal artifact envelope: {canonical}" in text
        assert envelope["max_file_bytes"] <= 16_000_000
        assert envelope["max_patch_bytes"] <= 16_000_000
        assert envelope["max_output_bytes"] <= 24_000_000
    assert (
        gate.RELEASE_MAX_LOG_BYTES
        + gate.BASELINE_MAX_RECEIPT_BYTES
        + gate.RELEASE_MAX_REPORT_BYTES
        < gate.RELEASE_PROPOSAL_ENVELOPE["max_patch_bytes"]
    )


def test_validation_paths_are_bootstrap_present_or_dependency_produced() -> None:
    task_text = (
        ROOT / "docs/architecture/incremental_proof_sealer.todo.md"
    ).read_text(encoding="utf-8")
    goal_text = (
        ROOT / "docs/architecture/incremental_proof_sealer.objectives.md"
    ).read_text(encoding="utf-8")
    config = json.loads(
        (
            ROOT
            / "config/agent_supervisor_incremental_proof_sealer_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    setup_errors: list[str] = []
    dependencies = gate._validate_tasks(task_text, config, setup_errors)
    assert setup_errors == []
    errors: list[str] = []

    gate._validate_validation_path_closure(
        task_text, goal_text, dependencies, errors
    )

    assert errors == []
    bad_goals = goal_text.replace(
        "test/api/incremental_sealing/test_full_checkpoint.py",
        "test/api/incremental_sealing/test_unproduced_seals.py",
        1,
    )
    goal_errors: list[str] = []
    gate._validate_validation_path_closure(
        task_text, bad_goals, dependencies, goal_errors
    )
    assert any(
        "IPS-G080 validation path is neither bootstrap-present nor dependency-produced"
        in error
        for error in goal_errors
    )
    bad_tasks = task_text.replace(
        "python -m pytest -q test/api/incremental_sealing/test_full_checkpoint.py",
        "python -m pytest -q test/api/incremental_sealing/test_unproduced_task.py",
        1,
    )
    task_errors: list[str] = []
    gate._validate_validation_path_closure(
        bad_tasks, goal_text, dependencies, task_errors
    )
    assert any(
        "IPS-038 validation path is neither bootstrap-present nor dependency-produced"
        in error
        for error in task_errors
    )


@pytest.mark.parametrize(
    "outputs",
    [
        {gate.BENCHMARK_JSON, gate.BENCHMARK_CSV},
        {
            gate.RELEASE_REPORT,
            gate.RELEASE_VALIDATION_JSON,
            gate.RELEASE_VALIDATION_LOG,
        },
    ],
)
def test_parent_bound_evidence_accepts_exact_output_only_completion_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outputs: set[str],
) -> None:
    root = tmp_path / "lifecycle"
    taskboard = "docs/architecture/incremental_proof_sealer.todo.md"
    parent, parent_tree = _init_repository(
        root,
        {
            "source.py": "bound\n",
            taskboard: _taskboard_text_with_inventory_todos(),
        },
    )
    datasets_revision, datasets_tree = _init_repository(
        root / "ipfs_datasets_py", {"source.py": "datasets\n"}
    )
    kit_revision, kit_tree = _init_repository(
        root / "ipfs_kit_py", {"source.py": "kit\n"}
    )
    revisions = {
        "accelerate": parent,
        "datasets": datasets_revision,
        "kit": kit_revision,
    }
    trees = {
        "accelerate": parent_tree,
        "datasets": datasets_tree,
        "kit": kit_tree,
    }
    target_branch = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    for relative in outputs:
        _write(root / relative, f"evidence:{relative}\n")
    _git(root, "add", "--", *sorted(outputs))
    _git(root, "commit", "-q", "-m", "candidate evidence")
    _git(root, "checkout", "-q", target_branch)
    _write(root / "pre-merge-lane.py", "independently admitted before merge\n")
    _git(root, "add", "--", "pre-merge-lane.py")
    _git(root, "commit", "-q", "-m", "interleaved lane before merge")
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate evidence")
    integrated = _git(root, "rev-parse", "HEAD")
    _write(root / "interleaved-lane.py", "independently admitted lane output\n")
    _git(root, "add", "--", "interleaved-lane.py")
    _git(root, "commit", "-q", "-m", "interleaved authorized lane")
    completed_board = (root / taskboard).read_text(encoding="utf-8").replace(
        "- Status: todo", "- Status: completed", 1
    )
    _write(root / taskboard, completed_board)
    _git(root, "config", "user.email", "implementation-daemon@example.invalid")
    _git(root, "config", "user.name", "Implementation Daemon")
    _git(root, "add", "--", taskboard)
    _git(root, "commit", "-q", "-m", "IPS-001: mark todo completed")
    completion = _git(root, "rev-parse", "HEAD")
    completion_tree = _git(root, "rev-parse", "HEAD^{tree}")
    assert _git(root, "rev-parse", f"{integrated}^2")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate, "_task_output_exists_at_control_revision", lambda *_args: True
    )
    monkeypatch.setattr(
        gate,
        "_current_repository_bindings",
        lambda errors: (
            {**revisions, "accelerate": completion},
            {**trees, "accelerate": completion_tree},
        ),
    )
    errors: list[str] = []

    gate._validate_parent_bound_output_lifecycle(
        label="test evidence",
        completion_task_id="IPS-001",
        parent_revision=parent,
        source_revisions=revisions,
        source_trees=trees,
        completion_outputs=outputs,
        errors=errors,
    )

    assert errors == []


def test_parent_bound_evidence_rejects_completion_commit_with_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bad-lifecycle"
    parent, parent_tree = _init_repository(root, {"source.py": "bound\n"})
    datasets_revision, datasets_tree = _init_repository(
        root / "ipfs_datasets_py", {"source.py": "datasets\n"}
    )
    kit_revision, kit_tree = _init_repository(
        root / "ipfs_kit_py", {"source.py": "kit\n"}
    )
    revisions = {
        "accelerate": parent,
        "datasets": datasets_revision,
        "kit": kit_revision,
    }
    trees = {
        "accelerate": parent_tree,
        "datasets": datasets_tree,
        "kit": kit_tree,
    }
    target_branch = _git(root, "branch", "--show-current")
    _git(root, "checkout", "-q", "-b", "candidate")
    _write(root / gate.BENCHMARK_JSON, "{}\n")
    _write(root / gate.BENCHMARK_CSV, "header\n")
    _write(root / "source.py", "tampered\n")
    _git(root, "add", "--all")
    _git(root, "commit", "-q", "-m", "laundered candidate")
    _git(root, "checkout", "-q", target_branch)
    _git(root, "merge", "-q", "--no-ff", "candidate", "-m", "integrate bad evidence")
    completion = _git(root, "rev-parse", "HEAD")
    completion_tree = _git(root, "rev-parse", "HEAD^{tree}")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate,
        "_current_repository_bindings",
        lambda errors: (
            {**revisions, "accelerate": completion},
            {**trees, "accelerate": completion_tree},
        ),
    )
    errors: list[str] = []

    gate._validate_parent_bound_output_lifecycle(
        label="test evidence",
        completion_task_id="IPS-053",
        parent_revision=parent,
        source_revisions=revisions,
        source_trees=trees,
        completion_outputs={gate.BENCHMARK_JSON, gate.BENCHMARK_CSV},
        errors=errors,
    )

    assert any("candidate must change exactly" in error for error in errors)


def test_task_completion_output_requires_a_regular_git_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "symlink-output"
    _init_repository(root, {"target.txt": "target\n"})
    (root / "declared.txt").symlink_to("target.txt")
    _git(root, "add", "--", "declared.txt")
    _git(root, "commit", "-q", "-m", "symlink output")
    revision = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(gate, "REPO_ROOT", root)

    assert not gate._task_output_exists_at_control_revision(
        revision, "declared.txt"
    )


def test_runtime_taskboard_allows_only_backed_monotonic_completion_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    original = (ROOT / relative).read_text(encoding="utf-8")
    root = tmp_path / "runtime-board"
    captured, _ = _init_repository(root, {relative: original})
    completed = original.replace("- Status: todo", "- Status: completed", 3)
    _write(root / relative, completed)
    _git(root, "config", "user.email", "implementation-daemon@example.invalid")
    _git(root, "config", "user.name", "Implementation Daemon")
    _git(root, "add", "--", relative)
    _git(root, "commit", "-q", "-m", "IPS-001: mark todo completed")
    current = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate, "_task_output_exists_at_control_revision", lambda *_args: True
    )
    errors: list[str] = []

    gate._validate_taskboard_status_transition(captured, current, errors)

    assert errors == []
    tampered = completed.replace(
        "- Validation: python scripts/validate_incremental_proof_sealer_board.py "
        "--check-artifact IPS-001",
        "- Validation: python forged.py",
        1,
    )
    _write(root / relative, tampered)
    _git(root, "add", "--", relative)
    _git(root, "commit", "-q", "-m", "IPS-001: mark todo completed")
    tampered_revision = _git(root, "rev-parse", "HEAD")
    tamper_errors: list[str] = []

    gate._validate_taskboard_status_transition(
        captured, tampered_revision, tamper_errors
    )

    assert any("changed immutable metadata for IPS-001" in error for error in tamper_errors)


def test_runtime_taskboard_replay_rejects_definition_change_then_revert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "docs/architecture/incremental_proof_sealer.todo.md"
    original = (ROOT / relative).read_text(encoding="utf-8")
    root = tmp_path / "runtime-board-laundering"
    captured, _ = _init_repository(root, {relative: original})
    _git(root, "config", "user.email", "implementation-daemon@example.invalid")
    _git(root, "config", "user.name", "Implementation Daemon")
    forged = original.replace(
        "- Validation: python scripts/validate_incremental_proof_sealer_board.py "
        "--check-artifact IPS-001",
        "- Validation: python forged.py",
        1,
    )
    _write(root / relative, forged)
    _git(root, "add", "--", relative)
    _git(root, "commit", "-q", "-m", "IPS-001: mark todo completed")
    completed = original.replace("- Status: todo", "- Status: completed", 1)
    _write(root / relative, completed)
    _git(root, "add", "--", relative)
    _git(root, "commit", "-q", "-m", "IPS-001: mark todo completed")
    current = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate, "_task_output_exists_at_control_revision", lambda *_args: True
    )
    errors: list[str] = []

    gate._validate_taskboard_status_transition(captured, current, errors)

    assert any(
        "bytes other than exact Status" in error
        or "changed without a completed task transition" in error
        for error in errors
    ), errors


def test_ips055_rejects_skeletal_trust_and_migration_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _write(
        tmp_path / "docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_MODEL.md",
        "Integrity commitment\nSigned execution receipt\n",
    )
    _write(
        tmp_path / "docs/architecture/INCREMENTAL_PROOF_SEALER_MIGRATION.md",
        "accept adapt reverify reject simulated\n",
    )
    errors: list[str] = []

    gate._validate_trust_and_migration_docs(errors)

    assert any("does not establish correct execution" in error for error in errors)
    assert any("no assurance upgrade" in error for error in errors)


def _release_report_text(receipt_digest: str, revisions: dict[str, str]) -> str:
    terms = (
        gate.RELEASE_VALIDATION_SCHEMA,
        receipt_digest,
        *revisions.values(),
        "existing ZK systems",
        "real proving",
        "simulated",
        "structural validation",
        "direct execution proof",
        "proof-unit granularity",
        "complete cache key",
        "invalidation rules",
        "full-proof fallback",
        "Merkle manifest aggregation",
        "40-transition benchmark",
        "average proof reuse rate",
        "average proving-compute reduction",
        "best incremental case",
        "worst incremental case",
        "proof size",
        "seal size",
        "verification latency",
        "storage overhead",
        "crash-recovery results",
        "tamper-test results",
        "trusted signed receipts",
        "integrity commitments",
        "remaining work before production use",
        gate.RELEASE_PUBLIC_LOG_POLICY,
        "live IPFS was refused before release suite execution",
        "three new incremental-sealing suites require fully green execution",
        "pytest process outputs were observed but test execution was not cryptographically proven",
        "Repository verification was decomposed into content-addressed proof units",
        "stale or simulated evidence",
    )
    return "\n".join(terms) + "\n"


def _write_release_fixture(
    root: Path,
    revisions: dict[str, str],
    trees: dict[str, str],
    suite: dict[str, Any],
    *,
    pytest_output: bytes | None = None,
) -> dict[str, Any]:
    terminal_output = b'{"valid": true}\n'
    if pytest_output is None:
        pytest_output = (
            b"============================= test session starts ==============================\n"
            b"collected 1 item\n"
            b"test_example.py::test_example PASSED\n"
            b"============================== 1 passed in 0.01s ===============================\n"
        )
    raw_log = terminal_output + pytest_output
    terminal = {
        "id": "terminal-board-gate",
        "argv": [sys.executable, "scripts/validate_incremental_proof_sealer_board.py", "--check-terminal"],
        "cwd": ".",
        "timeout_seconds": 120,
        "duration_ns": 1,
        "capture_status": "completed",
        "exit_code": 0,
        "log_offset": 0,
        "log_bytes": len(terminal_output),
        "log_sha256": "sha256:" + hashlib.sha256(terminal_output).hexdigest(),
    }
    text = pytest_output.decode()
    counts = gate._summary_counts(text.splitlines()[-1])
    collected_count = gate._collection_count(text)
    collection_complete = gate._collection_complete(text)
    non_pass_nodes = gate._nonpass_nodes(text)
    command = {
        **{
            key: suite[key]
            for key in ("id", "suite_origin", "argv", "cwd", "timeout_seconds")
        },
        "duration_ns": 2,
        "capture_status": "completed",
        "exit_code": 1 if counts["failed"] or counts["errors"] else 0,
        "collected_count": collected_count,
        "collection_complete": collection_complete,
        "outcome_counts": counts,
        "non_pass_nodes": non_pass_nodes,
        "summary_line": text.splitlines()[-1],
        "log_offset": len(terminal_output),
        "log_bytes": len(pytest_output),
        "log_sha256": "sha256:" + hashlib.sha256(pytest_output).hexdigest(),
        "assurance": "process_observed_only",
    }
    if (
        suite["suite_origin"] == "reviewed_existing_zk_suite"
        and "baseline_observation" not in suite
    ):
        suite["baseline_observation"] = {
            "receipt_digest": "sha256:" + "a" * 64,
            "capture_status": command["capture_status"],
            "exit_code": command["exit_code"],
            "collected_count": command["collected_count"],
            "collection_complete": command["collection_complete"],
            "outcome_counts": copy.deepcopy(command["outcome_counts"]),
            "non_pass_nodes": copy.deepcopy(command["non_pass_nodes"]),
        }
    command["acceptance_status"] = gate._release_acceptance_status(suite, command)
    body = {
        "schema_version": gate.RELEASE_VALIDATION_SCHEMA,
        "runner_id": gate.RELEASE_RUNNER_ID,
        "validation_worktree_parent_revision": revisions["accelerate"],
        "source_revisions": revisions,
        "source_trees": trees,
        "environment_policy_id": gate.RELEASE_ENVIRONMENT_POLICY,
        "terminal_gate": terminal,
        "pytest_commands": [command],
        "retained_log": {
            "path": gate.RELEASE_VALIDATION_LOG,
            "bytes": len(raw_log),
            "sha256": "sha256:" + hashlib.sha256(raw_log).hexdigest(),
        },
        "assurance": gate._release_assurance(),
    }
    body["receipt_digest"] = "sha256:" + hashlib.sha256(
        gate._canonical_json_bytes(body)
    ).hexdigest()
    _write(root / gate.RELEASE_VALIDATION_LOG, raw_log.decode())
    _write_json(root / gate.RELEASE_VALIDATION_JSON, body)
    _write(root / gate.RELEASE_REPORT, _release_report_text(body["receipt_digest"], revisions))
    return body


def test_ips056_release_receipt_accepts_exact_observed_log_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release-positive"
    revisions = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "abc")}
    trees = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "def")}
    suite = {
        "id": "one-fixed-suite",
        "suite_origin": "reviewed_existing_zk_suite",
        "argv": [sys.executable, "-m", "pytest", "-q", "test_example.py"],
        "cwd": ".",
        "timeout_seconds": 120,
    }
    _write_release_fixture(root, revisions, trees, suite)
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate, "_validate_parent_bound_output_lifecycle", lambda **kwargs: None
    )
    monkeypatch.setattr(gate, "_release_suite_specs", lambda errors: [suite])
    errors: list[str] = []

    gate._validate_release_validation(errors)

    assert errors == []


def test_ips056_completed_report_rejects_surviving_request_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release-marker"
    revisions = {
        name: value * 40
        for name, value in zip(("accelerate", "datasets", "kit"), "abc")
    }
    trees = {
        name: value * 40
        for name, value in zip(("accelerate", "datasets", "kit"), "def")
    }
    suite = {
        "id": "one-fixed-suite",
        "suite_origin": "reviewed_existing_zk_suite",
        "argv": [sys.executable, "-m", "pytest", "-q", "test_example.py"],
        "cwd": ".",
        "timeout_seconds": 120,
    }
    _write_release_fixture(root, revisions, trees, suite)
    with (root / gate.RELEASE_REPORT).open("a", encoding="utf-8") as report:
        report.write(gate.RELEASE_REPORT_REQUEST_MARKER + "\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate, "_validate_parent_bound_output_lifecycle", lambda **kwargs: None
    )
    monkeypatch.setattr(gate, "_release_suite_specs", lambda errors: [suite])
    errors: list[str] = []

    gate._validate_release_validation(errors)

    assert any("retains its materialization request marker" in error for error in errors)


def test_ips056_accepts_exact_retained_baseline_red_but_never_calls_it_green(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release-baseline-red"
    revisions = {
        name: value * 40
        for name, value in zip(("accelerate", "datasets", "kit"), "abc")
    }
    trees = {
        name: value * 40
        for name, value in zip(("accelerate", "datasets", "kit"), "def")
    }
    suite = {
        "id": "baseline-red-suite",
        "suite_origin": "reviewed_existing_zk_suite",
        "argv": [sys.executable, "-m", "pytest", "-q", "test_example.py"],
        "cwd": ".",
        "timeout_seconds": 120,
    }
    output = (
        b"============================= test session starts ==============================\n"
        b"collected 2 items\n"
        b"test_example.py::test_ok PASSED\n"
        b"test_example.py::test_red FAILED\n"
        b"FAILED test_example.py::test_red - AssertionError\n"
        b"========================= 1 failed, 1 passed in 0.01s ==========================\n"
    )
    body = _write_release_fixture(
        root,
        revisions,
        trees,
        suite,
        pytest_output=output,
    )
    assert body["pytest_commands"][0]["acceptance_status"] == (
        "baseline_compatible_non_green"
    )
    with (root / gate.RELEASE_REPORT).open("a", encoding="utf-8") as report:
        report.write("baseline-red-suite baseline_compatible_non_green remains explicit\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate, "_validate_parent_bound_output_lifecycle", lambda **kwargs: None
    )
    monkeypatch.setattr(gate, "_release_suite_specs", lambda errors: [suite])
    errors: list[str] = []

    gate._validate_release_validation(errors)

    assert errors == []
    regressed = copy.deepcopy(body["pytest_commands"][0])
    regressed["outcome_counts"]["failed"] = 2
    regressed["outcome_counts"]["selected"] = 3
    regressed["non_pass_nodes"].append(
        {"status": "failed", "node_id": "test_example.py::new", "detail": "new"}
    )
    assert gate._release_acceptance_status(suite, regressed) == "regressed"


@pytest.mark.parametrize("outcome", ["skipped", "xfailed", "deselected"])
def test_new_release_suites_reject_every_non_green_outcome(outcome: str) -> None:
    counts = {field: 0 for field in gate.BASELINE_OUTCOME_FIELDS}
    counts["passed"] = 1
    counts[outcome] = 1
    counts["selected"] = 1 if outcome == "deselected" else 2
    observation = {
        "capture_status": "completed",
        "exit_code": 0,
        "collected_count": counts["selected"] + counts["deselected"],
        "collection_complete": True,
        "outcome_counts": counts,
        "non_pass_nodes": [],
    }
    spec = {"suite_origin": "incremental_proof_sealer_current_tree_suite"}

    assert gate._release_acceptance_status(spec, observation) == "regressed"


def test_existing_release_suite_cannot_turn_a_failure_into_deselection() -> None:
    baseline_counts = {field: 0 for field in gate.BASELINE_OUTCOME_FIELDS}
    baseline_counts.update({"passed": 1, "failed": 1, "selected": 2})
    baseline_node = {
        "status": "failed",
        "node_id": "test_example.py::test_red",
        "detail": "test_example.py::test_red FAILED",
    }
    spec = {
        "suite_origin": "reviewed_existing_zk_suite",
        "baseline_observation": {
            "capture_status": "completed",
            "exit_code": 1,
            "collected_count": 2,
            "collection_complete": True,
            "outcome_counts": baseline_counts,
            "non_pass_nodes": [baseline_node],
        },
    }
    current_counts = copy.deepcopy(baseline_counts)
    current_counts.update({"failed": 0, "deselected": 1, "selected": 1})
    observation = {
        "capture_status": "completed",
        "exit_code": 0,
        "collected_count": 2,
        "collection_complete": True,
        "outcome_counts": current_counts,
        "non_pass_nodes": [],
    }

    assert gate._release_acceptance_status(spec, observation) == "regressed"


def test_existing_retained_skip_is_explicitly_non_green() -> None:
    counts = {field: 0 for field in gate.BASELINE_OUTCOME_FIELDS}
    counts.update({"passed": 1, "skipped": 1, "selected": 2})
    node = {
        "status": "skipped",
        "node_id": "test_example.py::test_optional",
        "detail": "test_example.py::test_optional SKIPPED",
    }
    observation = {
        "capture_status": "completed",
        "exit_code": 0,
        "collected_count": 2,
        "collection_complete": True,
        "outcome_counts": counts,
        "non_pass_nodes": [node],
    }
    spec = {
        "suite_origin": "reviewed_existing_zk_suite",
        "baseline_observation": {
            "receipt_digest": "sha256:" + "a" * 64,
            **copy.deepcopy(observation),
        },
    }

    assert gate._release_acceptance_status(spec, observation) == (
        "baseline_compatible_non_green"
    )


def test_release_runner_refuses_live_ipfs_before_starting_any_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "abc")}
    trees = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "def")}
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _write(
        tmp_path / gate.RELEASE_REPORT,
        "Substantive release report\n" + gate.RELEASE_REPORT_REQUEST_MARKER + "\n",
    )
    (tmp_path / gate.RELEASE_VALIDATION_JSON).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / gate.RELEASE_VALIDATION_JSON).write_bytes(gate.RELEASE_REQUEST_JSON)
    (tmp_path / gate.RELEASE_VALIDATION_LOG).write_bytes(gate.RELEASE_REQUEST_LOG)
    monkeypatch.setattr(gate, "_current_repository_bindings", lambda errors: (revisions, trees))
    monkeypatch.setattr(gate, "_release_suite_specs", lambda errors: [])
    monkeypatch.setattr(gate.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/ipfs")
    _mock_runner_source_binding(monkeypatch)

    def forbidden_process(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("release runner started a process despite live IPFS preflight")

    monkeypatch.setattr(gate, "_run_observed_process", forbidden_process)

    result = gate.run_release_validation()

    assert result["valid"] is False
    assert any("fixed PATH resolves ipfs" in error for error in result["errors"])


def test_release_runner_never_publishes_secret_bearing_process_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "abc")}
    trees = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "def")}
    suite = {
        "id": "new-suite",
        "suite_origin": "incremental_proof_sealer_current_tree_suite",
        "argv": [sys.executable, "-m", "pytest", "-q", "test_example.py"],
        "cwd": ".",
        "timeout_seconds": 120,
    }
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _write(
        tmp_path / gate.RELEASE_REPORT,
        "Substantive release report\n" + gate.RELEASE_REPORT_REQUEST_MARKER + "\n",
    )
    (tmp_path / gate.RELEASE_VALIDATION_JSON).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / gate.RELEASE_VALIDATION_JSON).write_bytes(gate.RELEASE_REQUEST_JSON)
    (tmp_path / gate.RELEASE_VALIDATION_LOG).write_bytes(gate.RELEASE_REQUEST_LOG)
    monkeypatch.setattr(gate, "_current_repository_bindings", lambda errors: (revisions, trees))
    monkeypatch.setattr(gate, "_release_suite_specs", lambda errors: [suite])
    monkeypatch.setattr(gate, "_validate_release_ipfs_preflight", lambda errors: None)
    _mock_runner_source_binding(monkeypatch)
    outputs = iter(
        (
            ("completed", 0, 1, b'{"valid": true}\n'),
            (
                "completed",
                1,
                2,
                b"authorization: abcdefghijklmnopqrstuvwxyz1234\n1 failed in 0.01s\n",
            ),
        )
    )
    monkeypatch.setattr(gate, "_run_observed_process", lambda *_args, **_kwargs: next(outputs))
    published: list[str] = []
    monkeypatch.setattr(
        gate,
        "_atomic_write_artifact",
        lambda relative, raw: published.append(relative),
    )

    result = gate.run_release_validation()

    assert result["valid"] is False
    assert any("secret scan" in error for error in result["errors"])
    assert published == []


@pytest.mark.parametrize("raw", (b"invalid-utf8:\xff\n", b"nul:\x00\n"))
def test_release_public_log_rejects_binary_or_non_utf8_bytes(raw: bytes) -> None:
    errors: list[str] = []

    gate._validate_release_public_log(raw, errors)

    assert errors


def test_benchmark_artifact_reader_rejects_oversize_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    path = tmp_path / gate.BENCHMARK_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(gate.BENCHMARK_MAX_ARTIFACT_BYTES + 1)
    errors: list[str] = []

    payload = gate._artifact_json(
        gate.BENCHMARK_JSON,
        errors,
        maximum_bytes=gate.BENCHMARK_MAX_ARTIFACT_BYTES,
        bound_label="one-MiB",
    )

    assert payload == {}
    assert any("one-MiB" in error for error in errors)


def test_benchmark_validation_materializes_once_then_is_hash_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repository(
        tmp_path,
        {
            ".gitignore": f"/{gate.RELEASE_WORK_ROOT}/\n",
            gate.BENCHMARK_CLI: "# fixed benchmark CLI\n",
            "source.py": "VALUE = 1\n",
        },
    )
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(monkeypatch, {"accelerate": Path(".")})
    (tmp_path / gate.BENCHMARK_JSON).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / gate.BENCHMARK_JSON).write_bytes(gate.BENCHMARK_REQUEST_JSON)
    (tmp_path / gate.BENCHMARK_CSV).write_bytes(gate.BENCHMARK_REQUEST_CSV)
    process_calls = 0
    real_observed = gate._run_observed_process

    def observed(
        argv: list[str], *_args: Any, **kwargs: Any
    ) -> tuple[str, int, int, bytes]:
        nonlocal process_calls
        if argv[0] == "git":
            return real_observed(argv, *_args, **kwargs)
        process_calls += 1
        cwd = Path(kwargs["cwd"])
        json_output = cwd / argv[argv.index("--json-output") + 1]
        csv_output = cwd / argv[argv.index("--csv-output") + 1]
        _write_json(json_output.resolve(), {"materialized": True})
        _write(csv_output.resolve(), "materialized\n")
        return "completed", 0, 1, b"benchmark observed\n"

    def artifact(task_id: str) -> dict[str, Any]:
        assert task_id == "IPS-053"
        valid = (tmp_path / gate.BENCHMARK_JSON).read_bytes() != gate.BENCHMARK_REQUEST_JSON
        return {"valid": valid, "errors": [] if valid else ["request remains"]}

    monkeypatch.setattr(gate, "_run_observed_process", observed)
    monkeypatch.setattr(gate, "validate_artifact", artifact)

    first = gate.run_benchmark_validation()
    hashes = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in (gate.BENCHMARK_JSON, gate.BENCHMARK_CSV)
    }
    second = gate.run_benchmark_validation()

    assert first["valid"] and first["materialization"] == "materialized_once"
    assert second["valid"] and second["materialization"] == "already_complete_read_only"
    assert process_calls == 1
    assert hashes == {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in (gate.BENCHMARK_JSON, gate.BENCHMARK_CSV)
    }


def _init_runner_binding_repository(root: Path) -> None:
    _init_repository(
        root,
        {
            ".gitignore": (
                f"/{gate.RELEASE_WORK_ROOT}/\n"
                "/ignored-execution-input.py\n"
            ),
            gate.BENCHMARK_CLI: "# fixed benchmark CLI\n",
            "source.py": "VALUE = 1\n",
        },
    )


@pytest.mark.parametrize("mutation", ("unstaged", "staged", "assume-unchanged"))
def test_runner_source_binding_rejects_tracked_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _init_runner_binding_repository(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(monkeypatch, {"accelerate": Path(".")})
    original_info = (tmp_path / "source.py").stat()
    if mutation == "assume-unchanged":
        _git(tmp_path, "update-index", "--assume-unchanged", "--", "source.py")
    _write(tmp_path / "source.py", "VALUE = 2\n")
    if mutation == "assume-unchanged":
        os.utime(
            tmp_path / "source.py",
            ns=(original_info.st_atime_ns, original_info.st_mtime_ns),
        )
    if mutation == "staged":
        _git(tmp_path, "add", "--", "source.py")
    errors: list[str] = []

    gate._capture_runner_source_binding("benchmark", errors)

    assert errors
    if mutation == "assume-unchanged":
        assert any("index flag" in error for error in errors), errors
    else:
        assert any(
            "index differs from HEAD" in error or "worktree has staged" in error
            for error in errors
        ), errors


def test_runner_source_binding_rejects_ignored_execution_input_but_allows_work_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_runner_binding_repository(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(monkeypatch, {"accelerate": Path(".")})
    _write(
        tmp_path / gate.RELEASE_WORK_ROOT / "suite" / "sitecustomize.py",
        "# runner-owned transient\n",
    )
    clean_errors: list[str] = []

    binding = gate._capture_runner_source_binding("benchmark", clean_errors)

    assert clean_errors == []
    assert set(binding) == {"accelerate"}
    _write(tmp_path / "ignored-execution-input.py", "raise RuntimeError('forged')\n")
    dirty_errors: list[str] = []

    gate._capture_runner_source_binding("benchmark", dirty_errors)

    assert any("ignored execution-relevant mutations" in error for error in dirty_errors)


@pytest.mark.parametrize("replacement_kind", ("replace-ref", "grafts"))
def test_runner_source_binding_rejects_git_object_replacement_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    _init_runner_binding_repository(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(monkeypatch, {"accelerate": Path(".")})
    if replacement_kind == "replace-ref":
        original_commit = _git(tmp_path, "rev-parse", "HEAD")
        original_tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
        _write(tmp_path / "source.py", "VALUE = 2\n")
        _git(tmp_path, "add", "--", "source.py")
        _git(tmp_path, "commit", "-q", "-m", "replacement tree")
        replacement_tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")
        _git(tmp_path, "checkout", "-q", original_commit)
        _git(tmp_path, "replace", original_tree, replacement_tree)
        _git(tmp_path, "read-tree", "--reset", "-u", "HEAD")
    else:
        _write(tmp_path / ".git" / "info" / "grafts", "0" * 40 + "\n")
    errors: list[str] = []

    gate._capture_runner_source_binding("benchmark", errors)

    assert any(
        "replacement refs" in error or "grafts file" in error for error in errors
    ), errors
    assert gate._fixed_git_environment()["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_runner_unmaterialized_gitlink_allowlist_rejects_unknown_or_oid_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_oid = "1" * 40
    monkeypatch.setattr(
        gate,
        "RUNNER_UNMATERIALIZED_GITLINKS",
        {"accelerate": {"vendor/reviewed": expected_oid}},
    )
    valid_entries = {
        "vendor/reviewed": ("160000", "commit", expected_oid),
    }
    valid_errors: list[str] = []

    gate._validate_unmaterialized_gitlinks(
        "accelerate", valid_entries, "benchmark", valid_errors
    )

    assert valid_errors == []
    bad_entries = {
        "vendor/reviewed": ("160000", "commit", "2" * 40),
        "vendor/unknown": ("160000", "commit", "3" * 40),
    }
    bad_errors: list[str] = []
    gate._validate_unmaterialized_gitlinks(
        "accelerate", bad_entries, "benchmark", bad_errors
    )
    assert any("allowlist drifted" in error for error in bad_errors)


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_runner_source_binding_rejects_unsafe_declared_artifact_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _init_runner_binding_repository(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(monkeypatch, {"accelerate": Path(".")})
    artifact = tmp_path / gate.BENCHMARK_JSON
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if kind == "symlink":
        artifact.symlink_to(tmp_path / "source.py")
    else:
        os.mkfifo(artifact)
    errors: list[str] = []

    gate._capture_runner_source_binding("benchmark", errors)

    assert any("regular non-symlink file" in error for error in errors), errors


def test_runner_source_binding_rejects_dirty_nested_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_runner_binding_repository(tmp_path)
    nested = tmp_path / "nested"
    _init_repository(nested, {"nested_source.py": "VALUE = 1\n"})
    _commit(tmp_path, "bind nested repository")
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(
        monkeypatch,
        {"accelerate": Path("."), "nested": Path("nested")},
    )
    _write(nested / "nested_source.py", "VALUE = 2\n")
    errors: list[str] = []

    gate._capture_runner_source_binding("benchmark", errors)

    assert any("nested worktree has staged" in error for error in errors), errors


def test_benchmark_runner_rejects_source_mutation_during_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_runner_binding_repository(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(monkeypatch, {"accelerate": Path(".")})
    (tmp_path / gate.BENCHMARK_JSON).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / gate.BENCHMARK_JSON).write_bytes(gate.BENCHMARK_REQUEST_JSON)
    (tmp_path / gate.BENCHMARK_CSV).write_bytes(gate.BENCHMARK_REQUEST_CSV)
    real_observed = gate._run_observed_process

    def mutate_source(
        argv: list[str], *_args: Any, **kwargs: Any
    ) -> tuple[str, int, int, bytes]:
        if argv[0] == "git":
            return real_observed(argv, *_args, **kwargs)
        cwd = Path(kwargs["cwd"])
        isolated_source = cwd / "source.py"
        isolated_source.chmod(0o644)
        _write(isolated_source, "VALUE = 999\n")
        json_output = cwd / argv[argv.index("--json-output") + 1]
        csv_output = cwd / argv[argv.index("--csv-output") + 1]
        _write_json(json_output.resolve(), {"materialized": True})
        _write(csv_output.resolve(), "materialized\n")
        return "completed", 0, 1, b"benchmark observed\n"

    monkeypatch.setattr(gate, "_run_observed_process", mutate_source)
    monkeypatch.setattr(
        gate,
        "validate_artifact",
        lambda _task_id: pytest.fail("dirty execution was validated as evidence"),
    )

    result = gate.run_benchmark_validation()

    assert result["valid"] is False
    assert any("materialized source bytes changed" in error for error in result["errors"])


def test_release_runner_rejects_source_mutation_during_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_runner_binding_repository(tmp_path)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _set_test_runner_repositories(monkeypatch, {"accelerate": Path(".")})
    _write(
        tmp_path / gate.RELEASE_REPORT,
        "Substantive release report\n" + gate.RELEASE_REPORT_REQUEST_MARKER + "\n",
    )
    (tmp_path / gate.RELEASE_VALIDATION_JSON).parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_path / gate.RELEASE_VALIDATION_JSON).write_bytes(gate.RELEASE_REQUEST_JSON)
    (tmp_path / gate.RELEASE_VALIDATION_LOG).write_bytes(gate.RELEASE_REQUEST_LOG)
    monkeypatch.setattr(gate, "_release_suite_specs", lambda _errors: [])
    monkeypatch.setattr(gate, "_validate_release_ipfs_preflight", lambda _errors: None)
    real_observed = gate._run_observed_process

    def mutate_source(
        argv: list[str], *_args: Any, **kwargs: Any
    ) -> tuple[str, int, int, bytes]:
        if argv[0] == "git":
            return real_observed(argv, *_args, **kwargs)
        isolated_source = Path(kwargs["cwd"]) / "source.py"
        isolated_source.chmod(0o644)
        _write(isolated_source, "VALUE = 999\n")
        return "completed", 0, 1, b'{"valid":true}\n'

    monkeypatch.setattr(gate, "_run_observed_process", mutate_source)
    monkeypatch.setattr(
        gate,
        "validate_artifact",
        lambda _task_id: pytest.fail("dirty release execution published evidence"),
    )

    result = gate.run_release_validation()

    assert result["valid"] is False
    assert any("materialized source bytes changed" in error for error in result["errors"])


def test_convergent_validation_rejects_partial_materialization_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    (tmp_path / gate.BENCHMARK_JSON).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / gate.BENCHMARK_JSON).write_bytes(gate.BENCHMARK_REQUEST_JSON)
    monkeypatch.setattr(
        gate,
        "_run_observed_process",
        lambda *_args, **_kwargs: pytest.fail("partial request executed benchmark"),
    )

    result = gate.run_benchmark_validation()

    assert result["valid"] is False
    assert any("partial" in error for error in result["errors"])


@pytest.mark.parametrize("runner", ("benchmark", "release"))
def test_convergent_validation_never_materializes_an_absent_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        gate,
        "_run_observed_process",
        lambda *_args, **_kwargs: pytest.fail("absent request started protected work"),
    )
    if runner == "benchmark":
        result = gate.run_benchmark_validation()
    else:
        _write(
            tmp_path / gate.RELEASE_REPORT,
            "Substantive final report\n" + gate.RELEASE_REPORT_REQUEST_MARKER + "\n",
        )
        result = gate.run_release_validation()

    assert result["valid"] is False
    assert any("request is absent" in error for error in result["errors"])


def test_release_work_is_ignored_but_final_evidence_is_not() -> None:
    transient = f"{gate.RELEASE_WORK_ROOT}/suite/pytest-cache/nodeids"
    ignored = subprocess.run(
        ("git", "check-ignore", "--no-index", "-q", transient),
        cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0
    for relative in (
        gate.BENCHMARK_JSON,
        gate.BENCHMARK_CSV,
        gate.RELEASE_VALIDATION_JSON,
        gate.RELEASE_VALIDATION_LOG,
    ):
        visible = subprocess.run(
            ("git", "check-ignore", "--no-index", "-q", relative),
            cwd=ROOT,
            check=False,
        )
        assert visible.returncode == 1, relative


def test_accelerate_inventory_json_has_one_narrow_gitignore_exception() -> None:
    exact = "docs/architecture/incremental_proof_sealer_inventory/accelerate.json"
    neighbor = "docs/architecture/incremental_proof_sealer_inventory/unreviewed.json"
    exact_result = subprocess.run(
        ("git", "check-ignore", "--no-index", "-q", "--", exact),
        cwd=ROOT,
        check=False,
    )
    neighbor_result = subprocess.run(
        ("git", "check-ignore", "--no-index", "-q", "--", neighbor),
        cwd=ROOT,
        check=False,
    )

    assert exact_result.returncode == 1
    assert neighbor_result.returncode == 0


def test_release_work_cleanup_unlinks_hardlink_without_chmodding_external_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    external = tmp_path / "external.txt"
    _write(external, "external bytes\n")
    external.chmod(0o644)
    work_root = root / gate.RELEASE_WORK_ROOT
    work_root.mkdir(parents=True)
    os.link(external, work_root / "hardlink.txt")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    errors: list[str] = []

    gate._clean_release_work_root(errors)

    assert errors == []
    assert not work_root.exists()
    assert external.read_text(encoding="utf-8") == "external bytes\n"
    assert stat.S_IMODE(external.stat().st_mode) == 0o644


def test_materialized_read_only_walk_rejects_regular_hardlink_without_chmod(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "external.txt"
    _write(external, "external bytes\n")
    external.chmod(0o640)
    os.link(external, source / "tracked.txt")
    errors: list[str] = []

    gate._make_materialized_source_read_only(source, errors)

    assert any("regular leaf is hardlinked" in error for error in errors)
    assert stat.S_IMODE(external.stat().st_mode) == 0o640


def test_atomic_artifact_publication_replaces_hardlink_without_external_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    output_parent = root / "artifacts"
    output_parent.mkdir(parents=True)
    external = tmp_path / "external.txt"
    _write(external, "external bytes\n")
    external.chmod(0o644)
    os.link(external, output_parent / "result.txt")
    monkeypatch.setattr(gate, "REPO_ROOT", root)

    gate._atomic_write_artifact("artifacts/result.txt", b"published bytes\n")

    assert (output_parent / "result.txt").read_bytes() == b"published bytes\n"
    assert external.read_text(encoding="utf-8") == "external bytes\n"
    assert stat.S_IMODE(external.stat().st_mode) == 0o644


def test_atomic_artifact_publication_rejects_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / "artifacts").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(gate, "REPO_ROOT", root)

    with pytest.raises(OSError):
        gate._atomic_write_artifact("artifacts/result.txt", b"forbidden\n")

    assert not (external / "result.txt").exists()


def test_release_environment_routes_all_writes_outside_materialized_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "materialized" / "source"
    workspace = tmp_path / "materialized" / "runtime" / "suite"
    source_root.mkdir(parents=True)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)

    environment = gate._release_environment(workspace, source_root=source_root)

    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(source_root.resolve())
    assert environment["PYTEST_ADDOPTS"] == (
        f"--benchmark-storage=file://{workspace / 'pytest-benchmark'}"
    )
    for name in (
        "home",
        "hypothesis",
        "ipfs-repo",
        "pycache",
        "pytest-benchmark",
        "pytest-cache",
        "pytest-tmp",
        "tmp",
    ):
        assert (workspace / name).is_dir()


def test_release_validation_materializes_report_binding_then_is_hash_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions = {
        name: value * 40
        for name, value in zip(("accelerate", "datasets", "kit"), "abc")
    }
    trees = {
        name: value * 40
        for name, value in zip(("accelerate", "datasets", "kit"), "def")
    }
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    _write(
        tmp_path / gate.RELEASE_REPORT,
        "Substantive final report\n" + gate.RELEASE_REPORT_REQUEST_MARKER + "\n",
    )
    (tmp_path / gate.RELEASE_VALIDATION_JSON).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / gate.RELEASE_VALIDATION_JSON).write_bytes(gate.RELEASE_REQUEST_JSON)
    (tmp_path / gate.RELEASE_VALIDATION_LOG).write_bytes(gate.RELEASE_REQUEST_LOG)
    monkeypatch.setattr(
        gate, "_current_repository_bindings", lambda errors: (revisions, trees)
    )
    monkeypatch.setattr(gate, "_release_suite_specs", lambda errors: [])
    monkeypatch.setattr(gate, "_validate_release_ipfs_preflight", lambda errors: None)
    _mock_runner_source_binding(monkeypatch)
    process_calls = 0

    def observed(*_args: Any, **_kwargs: Any) -> tuple[str, int, int, bytes]:
        nonlocal process_calls
        process_calls += 1
        return "completed", 0, 1, b'{"valid":true}\n'

    monkeypatch.setattr(gate, "_run_observed_process", observed)

    def artifact(task_id: str) -> dict[str, Any]:
        assert task_id == "IPS-056"
        report = (tmp_path / gate.RELEASE_REPORT).read_text(encoding="utf-8")
        valid = (
            gate.RELEASE_REPORT_REQUEST_MARKER not in report
            and "receipt_digest: sha256:" in report
            and (tmp_path / gate.RELEASE_VALIDATION_JSON).read_bytes()
            != gate.RELEASE_REQUEST_JSON
        )
        return {"valid": valid, "errors": [] if valid else ["not materialized"]}

    monkeypatch.setattr(gate, "validate_artifact", artifact)

    first = gate.run_release_validation()
    paths = (
        gate.RELEASE_REPORT,
        gate.RELEASE_VALIDATION_JSON,
        gate.RELEASE_VALIDATION_LOG,
    )
    hashes = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in paths
    }
    second = gate.run_release_validation()

    assert first["valid"] and first["materialization"] == "materialized_once"
    assert second["valid"] and second["materialization"] == "already_complete_read_only"
    assert process_calls == 1
    assert hashes == {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in paths
    }


def test_ips056_release_receipt_rejects_retained_log_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release-tamper"
    revisions = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "abc")}
    trees = {name: value * 40 for name, value in zip(("accelerate", "datasets", "kit"), "def")}
    suite = {
        "id": "one-fixed-suite",
        "suite_origin": "reviewed_existing_zk_suite",
        "argv": [sys.executable, "-m", "pytest", "-q", "test_example.py"],
        "cwd": ".",
        "timeout_seconds": 120,
    }
    _write_release_fixture(root, revisions, trees, suite)
    with (root / gate.RELEASE_VALIDATION_LOG).open("ab") as handle:
        handle.write(b"forged trailing success\n")
    monkeypatch.setattr(gate, "REPO_ROOT", root)
    monkeypatch.setattr(
        gate, "_validate_parent_bound_output_lifecycle", lambda **kwargs: None
    )
    monkeypatch.setattr(gate, "_release_suite_specs", lambda errors: [suite])
    errors: list[str] = []

    gate._validate_release_validation(errors)

    assert any("retained_log binding" in error for error in errors)
    assert any("unbound trailing" in error for error in errors)
