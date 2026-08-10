"""Contract tests for the sealed logic-family parser Wave-2 board."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    configured_board_launch_plan,
    load_configured_board,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    REPO_ROOT / "scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py"
)
CONFIG_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json"
)


def _validator_module():
    spec = importlib.util.spec_from_file_location("lfp2_board_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _closeout_tasks(
    *, status_049: str = "todo", status_050: str = "todo", open_task: str = ""
) -> list[SimpleNamespace]:
    tasks = []
    for index in range(51):
        task_id = f"LFP2-{index:03d}"
        status = "completed"
        if task_id == "LFP2-049":
            status = status_049
        elif task_id == "LFP2-050":
            status = status_050
        if task_id == open_task:
            status = "todo"
        tasks.append(SimpleNamespace(task_id=task_id, status=status))
    return tasks


def test_live_wave2_board_is_valid_at_any_progress_state() -> None:
    module = _validator_module()
    report = module.validate_all()
    assert report["valid"] is True, report["errors"]
    assert report["seed_task_count"] == 51
    assert set(report["completed_task_ids"]) | set(report["open_task_ids"]) == set(
        module.TASK_IDS
    )
    assert report["terminal_task_id"] == "LFP2-050"


def test_initial_projection_is_validated_without_freezing_live_progress() -> None:
    module = _validator_module()
    source = module.TODO_PATH.read_text(encoding="utf-8")
    initial = source.replace("- Status: completed", "- Status: todo")
    initial = initial.replace("- Status: todo", "- Status: completed", 1)
    errors: list[str] = []
    report = module._validate_tasks(initial, errors)
    assert errors == []
    assert report["completed_task_ids"] == ["LFP2-000"]
    assert report["ready_task_ids"] == [
        "LFP2-001",
        "LFP2-002",
        "LFP2-003",
        "LFP2-004",
    ]


def test_seed_digest_normalizes_status_only() -> None:
    module = _validator_module()
    source = module.TODO_PATH.read_text(encoding="utf-8")
    completed = source.replace("- Status: todo", "- Status: completed")
    assert module._seed_digest(completed) == module._seed_digest(source)

    semantic_mutation = source.replace(
        "- Interfaces: LogicClaimRuntimeAudit@1",
        "- Interfaces: LogicClaimRuntimeAudit@999",
        1,
    )
    assert module._seed_digest(semantic_mutation) != module._seed_digest(source)


def test_required_interface_owners_reject_missing_and_duplicate_owners() -> None:
    module = _validator_module()
    source = module.TODO_PATH.read_text(encoding="utf-8")

    missing = source.replace("ParseArtifact@2", "ParseArtifact@999", 1)
    missing_errors: list[str] = []
    module._validate_tasks(missing, missing_errors)
    assert (
        "ParseArtifact@2 must be owned exactly by LFP2-006; got []"
        in missing_errors
    )

    duplicate = source.replace(
        "LogicObligation@2, BackendRequest@2",
        "LogicObligation@2, BackendRequest@2, ParseArtifact@2",
        1,
    )
    duplicate_errors: list[str] = []
    module._validate_tasks(duplicate, duplicate_errors)
    assert (
        "ParseArtifact@2 must be owned exactly by LFP2-006; "
        "got ['LFP2-006', 'LFP2-007']"
        in duplicate_errors
    )


def test_closeout_tasks_require_deterministic_materializers_before_validation() -> None:
    module = _validator_module()
    source = module.TODO_PATH.read_text(encoding="utf-8")

    missing_role = source.replace("- Provider role: deterministic-only\n", "", 1)
    role_errors: list[str] = []
    module._validate_tasks(missing_role, role_errors)
    assert "LFP2-049 Provider role must be deterministic-only" in role_errors

    wrong_materializer = source.replace(
        "python -m ipfs_datasets_py.logic.conformance.release_v2 materialize",
        "python -m ipfs_datasets_py.logic.conformance.release_v2 inspect",
        1,
    )
    validation_errors: list[str] = []
    module._validate_tasks(wrong_materializer, validation_errors)
    assert (
        "LFP2-050 Validation must run its deterministic materializer before the "
        "board validator"
        in validation_errors
    )


def test_merge_target_worktree_parser_rejects_missing_and_duplicate_branches() -> None:
    module = _validator_module()
    target = f"refs/heads/{module.MERGE_TARGET_BRANCH}"
    missing = "worktree /candidate\0HEAD 1\0branch refs/heads/candidate\0\0"
    duplicate = (
        f"worktree /one\0HEAD 1\0branch {target}\0\0"
        f"worktree /two\0HEAD 2\0branch {target}\0\0"
    )
    with pytest.raises(RuntimeError, match="got 0"):
        module._merge_target_worktree_from_porcelain(missing)
    with pytest.raises(RuntimeError, match="got 2"):
        module._merge_target_worktree_from_porcelain(duplicate)


def test_predecessor_runtime_anchors_use_the_merge_target_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _validator_module()
    main = tmp_path / "main"
    candidate = tmp_path / "candidate"
    main.mkdir()
    _run_git(main, "init", "-b", module.MERGE_TARGET_BRANCH)
    _run_git(main, "config", "user.email", "validator@example.invalid")
    _run_git(main, "config", "user.name", "Board Validator")
    (main / ".gitignore").write_text("runtime/\n", encoding="utf-8")
    (main / "tracked.txt").write_text("main source\n", encoding="utf-8")
    _run_git(main, "add", ".gitignore", "tracked.txt")
    _run_git(main, "commit", "-m", "seed")
    _run_git(main, "worktree", "add", "-b", "candidate", str(candidate))

    (candidate / "tracked.txt").write_text("candidate source\n", encoding="utf-8")
    runtime_anchor = main / "runtime/anchor.json"
    runtime_anchor.parent.mkdir(parents=True)
    runtime_anchor.write_text('{"sealed":true}\n', encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", candidate)
    monkeypatch.setattr(
        module,
        "PREDECESSOR_FILE_DIGESTS",
        {"tracked.txt": module._sha256(candidate / "tracked.txt")},
    )
    runtime_digests = {"runtime/anchor.json": module._sha256(runtime_anchor)}
    monkeypatch.setattr(module, "PREDECESSOR_RUNTIME_ARTIFACT_DIGESTS", runtime_digests)
    scheduler = {"predecessor_runtime_artifact_digests": runtime_digests}

    assert module._canonical_main_worktree(main) == main.resolve()
    assert module._canonical_main_worktree(candidate) == main.resolve()
    assert not (candidate / "runtime/anchor.json").exists()
    errors: list[str] = []
    module._validate_predecessor_artifacts(scheduler, errors)
    assert errors == []

    runtime_anchor.write_text('{"sealed":false}\n', encoding="utf-8")
    drift_errors: list[str] = []
    module._validate_predecessor_artifacts(scheduler, drift_errors)
    assert drift_errors == [
        "Wave-1 predecessor runtime artifact changed: runtime/anchor.json"
    ]

    runtime_anchor.unlink()
    missing_errors: list[str] = []
    module._validate_predecessor_artifacts(scheduler, missing_errors)
    assert missing_errors == [
        "Wave-1 predecessor runtime artifact changed: runtime/anchor.json"
    ]

    (candidate / "tracked.txt").write_text("candidate drift\n", encoding="utf-8")
    source_errors: list[str] = []
    module._validate_predecessor_artifacts(scheduler, source_errors)
    assert source_errors[0] == "Wave-1 predecessor artifact changed: tracked.txt"


@pytest.mark.parametrize(
    ("status_049", "status_050"),
    (("todo", "todo"), ("completed", "todo")),
)
def test_fixed_point_artifacts_are_validated_before_and_after_049_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_049: str,
    status_050: str,
) -> None:
    module = _validator_module()
    fixed_path = tmp_path / "fixed.json"
    ledger_path = tmp_path / "ledger.jsonl"
    fixed_path.write_text("{}\n", encoding="utf-8")
    ledger_path.write_text("{}\n{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "FIXED_POINT_PATH", fixed_path)
    monkeypatch.setattr(module, "GAP_LEDGER_PATH", ledger_path)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def validate(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"receipt_id": "sha256:validated"}

    fake_module = SimpleNamespace(validate_fixed_point_artifacts=validate)
    monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake_module)
    tasks = _closeout_tasks(status_049=status_049, status_050=status_050)
    errors: list[str] = []
    assert module._validate_fixed_point_artifacts(tasks, errors) is True
    assert errors == []
    assert calls == [
        (
            (fixed_path, ledger_path),
            {"repo_root": tmp_path, "tasks": tasks},
        )
    ]


def test_fixed_point_gate_fails_closed_on_partial_invalid_or_open_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _validator_module()
    fixed_path = tmp_path / "fixed.json"
    ledger_path = tmp_path / "ledger.jsonl"
    fixed_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "FIXED_POINT_PATH", fixed_path)
    monkeypatch.setattr(module, "GAP_LEDGER_PATH", ledger_path)

    def invalid(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("digest mismatch")

    fake_module = SimpleNamespace(validate_fixed_point_artifacts=invalid)
    monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake_module)
    errors: list[str] = []
    valid = module._validate_fixed_point_artifacts(
        _closeout_tasks(open_task="LFP2-048"), errors
    )
    assert valid is False
    assert any("must both exist or neither exist" in error for error in errors)
    assert any("open: ['LFP2-048']" in error for error in errors)
    assert any("digest mismatch" in error for error in errors)

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing validator")),
    )
    unavailable_errors: list[str] = []
    module._validate_fixed_point_artifacts(
        _closeout_tasks(), unavailable_errors
    )
    assert any("validator is unavailable" in error for error in unavailable_errors)


def test_release_artifacts_validate_with_050_excluded_from_terminal_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _validator_module()
    markdown_path = tmp_path / "release.md"
    json_path = tmp_path / "release.json"
    markdown_path.write_text("# Release\n", encoding="utf-8")
    json_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "RELEASE_MARKDOWN_PATH", markdown_path)
    monkeypatch.setattr(module, "RELEASE_JSON_PATH", json_path)
    markdown_relative = Path("docs/release.md")
    json_relative = Path("data/release.json")
    monkeypatch.setattr(
        module, "RELEASE_MARKDOWN_RELATIVE_PATH", markdown_relative
    )
    monkeypatch.setattr(module, "RELEASE_JSON_RELATIVE_PATH", json_relative)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def validate(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"release_id": "sha256:validated"}

    fake_module = SimpleNamespace(validate_release_artifacts=validate)
    monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake_module)
    tasks = _closeout_tasks(status_049="completed", status_050="todo")
    errors: list[str] = []
    module._validate_release_artifacts(tasks, fixed_point_valid=True, errors=errors)
    assert errors == []
    assert calls == [
        (
            (markdown_relative, json_relative),
            {"repo_root": tmp_path},
        )
    ]

    prerequisite_errors: list[str] = []
    module._validate_release_artifacts(
        _closeout_tasks(status_049="todo", status_050="todo"),
        fixed_point_valid=False,
        errors=prerequisite_errors,
    )
    assert any("completed LFP2-049" in error for error in prerequisite_errors)
    assert any("current LFP2-049" in error for error in prerequisite_errors)

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing validator")),
    )
    unavailable_errors: list[str] = []
    module._validate_release_artifacts(
        tasks,
        fixed_point_valid=True,
        errors=unavailable_errors,
    )
    assert any("validator is unavailable" in error for error in unavailable_errors)


def test_launch_plan_is_dynamic_grok_first_and_static_goal_refill() -> None:
    board = load_configured_board(CONFIG_PATH, repo_root=REPO_ROOT)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        duration_seconds=300,
        stamp="20260809T000000Z",
    )
    common = [
        value.removeprefix("--common-arg=")
        for value in plan["argv"]
        if value.startswith("--common-arg=")
    ]
    assert plan["lanes"] == 4
    assert plan["strict_task_sharding"] is False
    assert "--implementation-supervisor-strict-task-sharding" not in plan["argv"]
    assert "--strict-task-sharding" not in common
    assert "--objective-refill-scan" in common
    assert "--no-objective-goal-refinement" in common
    assert "--codebase-refill-scan" not in common
    assert plan["environment"] == {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
            "primary_quota_exhausted"
        ),
        "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.5",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
    }
