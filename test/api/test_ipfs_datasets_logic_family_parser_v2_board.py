"""Contract tests for the sealed logic-family parser Wave-2 board."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_live_wave2_board_is_valid_and_has_four_initial_tasks() -> None:
    module = _validator_module()
    report = module.validate_all()
    assert report["valid"] is True, report["errors"]
    assert report["seed_task_count"] == 51
    assert report["completed_task_ids"] == ["LFP2-000"]
    assert report["ready_task_ids"] == [
        "LFP2-001",
        "LFP2-002",
        "LFP2-003",
        "LFP2-004",
    ]
    assert report["terminal_task_id"] == "LFP2-050"


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
