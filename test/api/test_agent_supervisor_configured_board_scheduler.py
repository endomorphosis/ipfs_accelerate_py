"""Tests for the sealed scheduler-config runtime adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_bootstrap_seal import (
    build_bootstrap_seal_payload,
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    ConfiguredBoardError,
    configured_board_launch_plan,
    load_configured_board,
    preflight_configured_board,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KITA_CONFIG = (
    REPO_ROOT / "config/agent_supervisor_ipfs_kit_runtime_readiness_scheduler.json"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def _configure_git(repo: Path) -> None:
    _git(repo, "config", "user.name", "Configured Board Test")
    _git(repo, "config", "user.email", "configured-board@example.invalid")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_configured_repo(tmp_path: Path) -> tuple[Path, Path]:
    child = tmp_path / "dependency-source"
    child.mkdir()
    _git(child, "init", "-b", "main")
    _configure_git(child)
    _write(child / "dependency.txt", "dependency\n")
    _git(child, "add", "dependency.txt")
    _git(child, "commit", "-m", "seed dependency")
    child_revision = _git(child, "rev-parse", "HEAD").stdout.strip()

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _configure_git(repo)
    _write(repo / "README.md", "configured board fixture\n")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "dependency",
    )
    _git(repo, "add", "README.md", ".gitmodules", "dependency")
    _git(repo, "commit", "-m", "seed repository")
    ancestor = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _write(repo / "docs/plan.md", "plan\n")
    _write(repo / "docs/objectives.md", "# Objectives\n")
    _write(repo / "docs/tasks.md", "# Tasks\n")
    _write(
        repo / "scripts/validate_board.py",
        (
            "import json\n"
            "print(json.dumps({'valid': True, 'errors': []}, sort_keys=True))\n"
        ),
    )
    _write(
        repo / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
        "raise SystemExit(0)\n",
    )
    config_path = repo / "config/scheduler.json"
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "configured_board_test.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "plan_path": "docs/plan.md",
        "validator_path": "scripts/validate_board.py",
        "task_prefix": "TEST-",
        "goal_prefix": "TEST-G",
        "board_namespace": "configured-board-test",
        "merge_target_branch": "main",
        "source_binding": {
            "accelerator_required_ancestor": ancestor,
            "accelerator_required_branch": "main",
            "dependency_submodule_path": "dependency",
            "dependency_planning_revision": child_revision,
        },
        "max_lanes": 2,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "poll_interval_seconds": 5,
        "daemon_interval_seconds": 60,
        "check_interval_seconds": 30,
        "stale_seconds": 1800,
        "watchdog_startup_grace_seconds": 300,
        "max_restarts": 3,
        "max_task_attempts": 3,
        "implementation_retry_budget": 3,
        "validation_retry_budget": 3,
        "merge_retry_budget": 3,
        "implementation_timeout_seconds": 7200,
        "implementation_max_timeout_seconds": 21600,
        "implementation_log_stall_seconds": 1200,
        "worktree_submodule_paths": ["dependency"],
        "protected_paths": [
            "config/scheduler.json",
            "docs/plan.md",
            "docs/objectives.md",
            "docs/tasks.md",
            "scripts/validate_board.py",
        ],
        "runtime_paths": {
            "root": "data/configured-board",
            "state": "data/configured-board/state",
            "worktrees": "data/configured-board/worktrees",
            "merge_queue": "data/configured-board/merge-queue",
            "logs": "data/configured-board/logs",
        },
        "lanes": [
            {
                "index": 0,
                "name": "test-lane-0",
                "strict_shard_remainder": 0,
            },
            {
                "index": 1,
                "name": "test-lane-1",
                "strict_shard_remainder": 1,
            },
        ],
        "provider": {
            "provider_id": "codex",
            "model_id": "test-model",
            "max_concurrency": 2,
        },
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(
        repo,
        "add",
        "config/scheduler.json",
        "docs/plan.md",
        "docs/objectives.md",
        "docs/tasks.md",
        "scripts/validate_board.py",
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
    )
    _git(repo, "commit", "-m", "add configured board")
    return repo, config_path


def _common_args(plan: dict[str, object]) -> list[str]:
    prefix = "--common-arg="
    return [
        item[len(prefix) :]
        for item in plan["argv"]
        if isinstance(item, str) and item.startswith(prefix)
    ]


def test_kita_config_maps_to_four_strict_existing_supervisor_lanes() -> None:
    board = load_configured_board(KITA_CONFIG, repo_root=REPO_ROOT)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260801T000000Z",
    )
    args = plan["argv"]
    common = _common_args(plan)

    lane_flag = args.index("--implementation-supervisor-lanes-per-track")
    assert args[lane_flag + 1] == "4"
    assert "--implementation-supervisor-strict-task-sharding" in args
    assert "--exit-when-all-tracks-terminal" in args
    assert "--detach" in args
    assert "--implement" in common
    assert "--strict-task-sharding" in common
    assert "--objective-refill-scan" not in common
    assert "--codebase-refill-scan" not in common
    assert "--no-objective-task-janitor" in common
    assert common.count("--worktree-submodule-path") == 2
    assert set(board.worktree_submodule_paths).issubset(common)
    assert common.count("--implementation-protected-path") == len(board.protected_paths)
    assert plan["environment"] == {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
            "primary_quota_exhausted"
        ),
        "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.5",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "medium",
    }


def test_ordered_provider_contract_requires_complete_unambiguous_fields(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "max_concurrency": 2,
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        ConfiguredBoardError,
        match="fallback_model_id",
    ):
        load_configured_board(config_path, repo_root=repo)

    payload["provider"]["fallback_model_id"] = "gpt-5.6-terra"
    payload["provider"]["fallback_trigger"] = "primary_quota_exhausted"
    payload["provider"]["fallback_reasoning_effort"] = "high"
    payload["provider"]["provider_id"] = "auto"
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        ConfiguredBoardError,
        match="cannot be mixed",
    ):
        load_configured_board(config_path, repo_root=repo)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("primary_provider_id", "grok"),
        ("primary_model_id", "grok-4"),
        ("fallback_provider_id", "openai"),
        ("fallback_model_id", "gpt-5.6"),
        ("fallback_trigger", "primary_unavailable"),
        ("fallback_reasoning_effort", "low"),
    ),
)
def test_ordered_provider_contract_seals_fallback_authority(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
    }
    payload["provider"][field] = value
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ConfiguredBoardError, match=field):
        load_configured_board(config_path, repo_root=repo)


@pytest.mark.parametrize("reasoning_effort", ("medium", "high"))
def test_ordered_provider_contract_accepts_only_supported_reasoning_efforts(
    tmp_path: Path,
    reasoning_effort: str,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": reasoning_effort,
        "max_concurrency": 2,
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    board = load_configured_board(config_path, repo_root=repo)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260803T000000Z",
    )

    assert plan["environment"][scheduler_module.CODEX_REASONING_EFFORT_ENV] == (
        reasoning_effort
    )


def test_legacy_provider_launch_environment_remains_backward_compatible(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    board = load_configured_board(config_path, repo_root=repo)

    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260803T000000Z",
    )

    assert plan["environment"] == {
        scheduler_module.PROVIDER_ENV: "codex",
        scheduler_module.CODEX_MODEL_ENV: "test-model",
    }


def test_launch_config_overrides_ambient_provider_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
    }
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "configure ordered provider route")
    observed: dict[str, str] = {}
    controlled_names = (
        scheduler_module.PROVIDER_ENV,
        scheduler_module.FALLBACK_PROVIDER_ENV,
        scheduler_module.FALLBACK_TRIGGER_ENV,
        scheduler_module.GROK_MODEL_ENV,
        scheduler_module.CODEX_MODEL_ENV,
        scheduler_module.CODEX_REASONING_EFFORT_ENV,
    )
    for name in controlled_names:
        monkeypatch.setenv(name, "ambient-value")

    def fake_multi_supervisor_main(_argv: list[str]) -> int:
        observed.update(
            {name: scheduler_module.os.environ[name] for name in controlled_names}
        )
        return 0

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner.main",
        fake_multi_supervisor_main,
    )

    result = scheduler_module.main(
        [
            "--repo-root",
            str(repo),
            "--config",
            str(config_path),
            "launch",
            "--implement",
            "--foreground",
            "--duration-seconds",
            "1",
        ]
    )

    assert result == 0
    assert observed == {
        scheduler_module.PROVIDER_ENV: "grok_cli",
        scheduler_module.FALLBACK_PROVIDER_ENV: "codex",
        scheduler_module.FALLBACK_TRIGGER_ENV: "primary_quota_exhausted",
        scheduler_module.GROK_MODEL_ENV: "grok-4.5",
        scheduler_module.CODEX_MODEL_ENV: "gpt-5.6-terra",
        scheduler_module.CODEX_REASONING_EFFORT_ENV: "high",
    }


def test_sparse_legacy_launch_clears_stale_ordered_route_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["provider"] = {"max_concurrency": 2}
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "use sparse legacy provider config")
    for name in scheduler_module.SCHEDULER_PROVIDER_ENV_NAMES:
        monkeypatch.setenv(name, "stale-ordered-value")
    observed: dict[str, str | None] = {}

    def fake_multi_supervisor_main(_argv: list[str]) -> int:
        observed.update(
            {
                name: scheduler_module.os.environ.get(name)
                for name in scheduler_module.SCHEDULER_PROVIDER_ENV_NAMES
            }
        )
        return 0

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner.main",
        fake_multi_supervisor_main,
    )

    result = scheduler_module.main(
        [
            "--repo-root",
            str(repo),
            "--config",
            str(config_path),
            "launch",
            "--implement",
            "--foreground",
            "--duration-seconds",
            "1",
        ]
    )

    assert result == 0
    assert observed == {
        name: None for name in scheduler_module.SCHEDULER_PROVIDER_ENV_NAMES
    }


def test_preflight_accepts_exact_committed_binding_then_rejects_drift(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    board = load_configured_board(config_path, repo_root=repo)

    report = preflight_configured_board(board)
    assert report["valid"] is True, report["errors"]

    _write(repo / "docs/plan.md", "dirty plan\n")
    dirty_report = preflight_configured_board(board)
    assert dirty_report["valid"] is False
    assert any(error.startswith("checkout_clean:") for error in dirty_report["errors"])

    _write(repo / "docs/plan.md", "plan\n")
    assert not _git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    _git(repo, "submodule", "deinit", "-f", "--", "dependency")
    uninitialized_report = preflight_configured_board(board)
    assert uninitialized_report["valid"] is False
    submodule_check = next(
        check
        for check in uninitialized_report["checks"]
        if check["name"] == "configured_submodules"
    )
    assert submodule_check["passed"] is False
    assert submodule_check["detail"][0]["exact_worktree"] is False


def test_preflight_accepts_only_descendant_submodule_progress(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    child_worktree = repo / "dependency"
    child_source = Path(
        _git(child_worktree, "remote", "get-url", "origin").stdout.strip()
    )

    _write(child_source / "dependency.txt", "advanced dependency\n")
    _git(child_source, "add", "dependency.txt")
    _git(child_source, "commit", "-m", "advance dependency")
    advanced_revision = _git(
        child_source,
        "rev-parse",
        "HEAD",
    ).stdout.strip()
    _git(child_worktree, "fetch", "origin")
    _git(child_worktree, "checkout", advanced_revision)
    _git(repo, "add", "dependency")
    _git(repo, "commit", "-m", "record dependency progress")

    board = load_configured_board(config_path, repo_root=repo)
    advanced_report = preflight_configured_board(board)
    assert advanced_report["valid"] is True, advanced_report["errors"]
    advanced_check = next(
        check
        for check in advanced_report["checks"]
        if check["name"] == "configured_submodules"
    )
    assert advanced_check["detail"][0]["planning_revision_is_ancestor"] is True

    _git(child_source, "checkout", "--orphan", "divergent")
    _write(child_source / "dependency.txt", "divergent dependency\n")
    _git(child_source, "add", "dependency.txt")
    _git(child_source, "commit", "-m", "diverge dependency")
    divergent_revision = _git(
        child_source,
        "rev-parse",
        "HEAD",
    ).stdout.strip()
    _git(child_worktree, "fetch", "origin", "divergent")
    _git(child_worktree, "checkout", divergent_revision)
    _git(repo, "add", "dependency")
    _git(repo, "commit", "-m", "record divergent dependency")

    divergent_report = preflight_configured_board(board)
    assert divergent_report["valid"] is False
    divergent_check = next(
        check
        for check in divergent_report["checks"]
        if check["name"] == "configured_submodules"
    )
    assert divergent_check["passed"] is False
    assert divergent_check["detail"][0]["planning_revision_is_ancestor"] is False


def test_preflight_rejects_missing_submodule_planning_revision(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    del payload["source_binding"]["dependency_planning_revision"]
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "remove dependency planning revision")

    board = load_configured_board(config_path, repo_root=repo)
    report = preflight_configured_board(board)
    submodule_check = next(
        check for check in report["checks"] if check["name"] == "configured_submodules"
    )

    assert report["valid"] is False
    assert submodule_check["passed"] is False
    assert submodule_check["detail"][0]["planning_revision"] == ""
    assert submodule_check["detail"][0]["planning_revision_is_ancestor"] is False


def test_preflight_rejects_submodule_head_gitlink_mismatch(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    child_worktree = repo / "dependency"
    child_source = Path(
        _git(child_worktree, "remote", "get-url", "origin").stdout.strip()
    )
    _write(child_source / "dependency.txt", "unrecorded advance\n")
    _git(child_source, "add", "dependency.txt")
    _git(child_source, "commit", "-m", "unrecorded dependency advance")
    revision = _git(child_source, "rev-parse", "HEAD").stdout.strip()
    _git(child_worktree, "fetch", "origin")
    _git(child_worktree, "checkout", revision)

    board = load_configured_board(config_path, repo_root=repo)
    report = preflight_configured_board(board)
    submodule = next(
        check for check in report["checks"] if check["name"] == "configured_submodules"
    )["detail"][0]

    assert report["valid"] is False
    assert submodule["valid"] is False
    assert submodule["head"] != submodule["gitlink"]


def test_preflight_rejects_dirty_submodule_worktree(tmp_path: Path) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    _write(repo / "dependency" / "dependency.txt", "dirty dependency\n")

    board = load_configured_board(config_path, repo_root=repo)
    report = preflight_configured_board(board)
    submodule = next(
        check for check in report["checks"] if check["name"] == "configured_submodules"
    )["detail"][0]

    assert report["valid"] is False
    assert submodule["valid"] is False
    assert submodule["dirty"]


def test_preflight_requires_exact_content_addressed_bootstrap_seal(
    tmp_path: Path,
) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    seal_relative = "config/bootstrap-seal.json"
    payload["source_binding"].update(
        {
            "bootstrap_seal_path": seal_relative,
            "record_recursive_repository_forest_at_launch": True,
            "changed_revision_requires_fresh_inventory_and_baseline": True,
        }
    )
    payload["provider"] = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
    }
    payload["protected_paths"].append(seal_relative)
    _write(config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "require bootstrap seal")

    missing_board = load_configured_board(config_path, repo_root=repo)
    missing = preflight_configured_board(missing_board)
    assert missing["valid"] is False
    assert any(error.startswith("bootstrap_seal:") for error in missing["errors"])

    validator_report = {"valid": True, "errors": []}
    seal = build_bootstrap_seal_payload(
        repo_root=repo,
        board_namespace=payload["board_namespace"],
        source_binding=payload["source_binding"],
        worktree_submodule_paths=payload["worktree_submodule_paths"],
        protected_paths=payload["protected_paths"],
        seal_path=seal_relative,
        taskboard_path=payload["taskboard_path"],
        task_header_prefix=payload["task_prefix"],
        validator_report=validator_report,
    )
    (repo / seal_relative).write_bytes(canonical_json_bytes(seal) + b"\n")
    _git(repo, "add", seal_relative)
    _git(repo, "commit", "-m", "seal configured board")

    board = load_configured_board(config_path, repo_root=repo)
    report = preflight_configured_board(board)
    assert report["valid"] is True, report["errors"]
    seal_check = next(
        check for check in report["checks"] if check["name"] == "bootstrap_seal"
    )
    assert seal_check["passed"] is True
    assert seal_check["detail"]["seal_id"] == seal["seal_id"]
    assert (
        seal_check["detail"]["authoring_board_id"]
        == seal["authoring_board"]["authoring_board_id"]
    )
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260808T000000Z",
        preflight_report=report,
    )
    assert (
        plan["environment"]["IPFS_ACCELERATE_CONFIGURED_BOARD_BOOTSTRAP_SEAL_ID"]
        == seal["seal_id"]
    )
    assert (
        plan["environment"]["IPFS_ACCELERATE_CONFIGURED_BOARD_AUTHORING_BOARD_ID"]
        == seal["authoring_board"]["authoring_board_id"]
    )
    assert (
        plan["authoring_launch"]["launch_id"]
        == plan["environment"]["IPFS_ACCELERATE_CONFIGURED_BOARD_LAUNCH_ID"]
    )
    assert (
        plan["authoring_launch"]["launch_head"]
        == _git(repo, "rev-parse", "HEAD^{commit}").stdout.strip()
    )
    assert (
        plan["authoring_launch"]["launch_tree"]
        == _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()
    )

    tampered_seal = dict(seal)
    tampered_seal["seal_id"] = "sha256:" + "0" * 64
    (repo / seal_relative).write_bytes(canonical_json_bytes(tampered_seal) + b"\n")
    with pytest.raises(ConfiguredBoardError, match="changed after preflight"):
        configured_board_launch_plan(
            board,
            implement=True,
            detach=True,
            stamp="20260808T000001Z",
            preflight_report=report,
        )
    (repo / seal_relative).write_bytes(canonical_json_bytes(seal) + b"\n")

    _write(repo / "docs/plan.md", "changed after seal\n")
    stale = preflight_configured_board(board)
    stale_seal = next(
        check for check in stale["checks"] if check["name"] == "bootstrap_seal"
    )
    assert stale["valid"] is False
    assert stale_seal["passed"] is False


def test_loader_rejects_runtime_path_escape(tmp_path: Path) -> None:
    repo, config_path = _seed_configured_repo(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["runtime_paths"]["state"] = "../escaped-state"
    payload["protected_paths"][0] = "config/unsafe.json"
    unsafe = repo / "config/unsafe.json"
    _write(unsafe, json.dumps(payload))

    with pytest.raises(ConfiguredBoardError, match="unsafe relative path"):
        load_configured_board(unsafe, repo_root=repo)
