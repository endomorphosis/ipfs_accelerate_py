"""Supervisor-ingestion tests for the IPFS Kit kernel-VFS/FUSE board."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file
from scripts.validate_ipfs_kit_fuse_vfs_board import (
    GOAL_IDS,
    INITIAL_READY,
    INITIAL_SHARDS,
    TASK_IDS,
)

from scripts import validate_ipfs_kit_fuse_vfs_board as board_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.todo.md"
OBJECTIVE_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.objectives.md"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json"
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_ipfs_kit_fuse_vfs_board.py"


def _replace_in_task_block(
    text: str,
    task_id: str,
    needle: str,
    replacement: str,
    *,
    count: int = 1,
) -> str:
    header = f"## {task_id} "
    start = text.index(header)
    end = text.find("\n## KVFS-", start + len(header))
    if end < 0:
        end = len(text)
    block = text[start:end]
    assert needle in block
    updated = block.replace(needle, replacement, count)
    return f"{text[:start]}{updated}{text[end:]}"


def _blocked_kvfs814_board(text: str) -> str:
    """Project the completed live guardrail back to an unanchored active card."""

    text = _replace_in_task_block(
        text,
        "KVFS-814",
        "- Status: completed",
        "- Status: blocked",
    )
    header = "## KVFS-814 "
    start = text.index(header)
    block = text[start:]
    anchor = next(
        line
        for line in block.splitlines()
        if line.startswith("- Resolution receipt digest: ")
    )
    return _replace_in_task_block(
        text,
        "KVFS-814",
        f"{anchor}\n",
        "",
    )


def _kvfs814_fields() -> dict[str, str]:
    return next(
        fields
        for task_id, _title, fields in board_validator.parse_tasks()
        if task_id == "KVFS-814"
    )


def _resolution_receipt(fields: dict[str, str]) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": board_validator.RECONCILIATION_RESOLUTION_SCHEMA,
        "task_id": "KVFS-814",
        "reconciliation_fingerprint": fields["reconciliation fingerprint"],
        "kind": fields["reconciliation kind"],
        "reason": fields["reconciliation reason"],
        "resolved": True,
        "resolved_at": "2026-08-10T23:45:00+00:00",
        "resolution_method": "isolated_gitlink_reconciliation",
        "postconditions": {
            "candidate_count_before": 1,
            "candidate_count_after": 0,
            "active_blocker_present_after": False,
            "dirty_worktree_group_count_after": 0,
            "cleanup_skip_count_after": 0,
        },
        "evidence": {"result": "candidate reconciled in an isolated worktree"},
    }
    receipt["receipt_digest"] = board_validator._resolution_receipt_digest(
        receipt
    )
    return receipt


def _receipt_section(receipt: dict[str, object]) -> str:
    return (
        "## Resolution Receipt\n\n```json\n"
        f"{json.dumps(receipt, indent=2, sort_keys=True)}\n```\n"
    )


def test_declared_validator_accepts_the_sealed_projection() -> None:
    result = subprocess.run(
        (sys.executable, str(VALIDATOR_PATH), "--check-all"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["task_count"] == 40
    assert report["parsed_task_count"] == 43
    assert report["operational_task_ids"] == [
        "KVFS-812",
        "KVFS-813",
        "KVFS-814",
    ]
    assert report["pending_operational_task_ids"] == []
    assert report["goal_count"] == 9
    assert report["initial_ready_task_ids"] == list(INITIAL_READY)
    assert report["ready_task_ids"] == ["KVFS-811"]
    assert report["initial_shards"] == {str(key): value for key, value in INITIAL_SHARDS.items()}


def test_production_parsers_consume_exact_tasks_goals_and_dependencies() -> None:
    parsed_tasks = parse_task_file(TODO_PATH, task_header_prefix="## KVFS-")
    goals = parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))
    tasks = tuple(
        task
        for task in parsed_tasks
        if task.metadata.get("canonical board task") != "false"
    )
    operational = tuple(task for task in parsed_tasks if task not in tasks)
    assert tuple(task.task_id for task in tasks) == TASK_IDS
    assert tuple(task.task_id for task in operational) == (
        "KVFS-812",
        "KVFS-813",
        "KVFS-814",
    )
    assert tuple(goal.goal_id for goal in goals) == GOAL_IDS
    assert all(task.metadata["goal id"] in GOAL_IDS for task in tasks)
    by_id = {task.task_id: task for task in tasks}
    completed = {task_id for task_id, task in by_id.items() if task.status == "completed"}
    ready = tuple(
        task_id for task_id in TASK_IDS
        if by_id[task_id].status == "todo"
        and all(dependency in completed for dependency in by_id[task_id].depends_on)
    )
    assert ready == ("KVFS-811",)


def test_native_scheduler_loader_and_strict_shards_match() -> None:
    board = load_configured_board(CONFIG_PATH, repo_root=REPO_ROOT)
    assert board.max_lanes == 4
    assert board.strict_task_sharding is True
    assert board.worktree_submodule_paths == ("ipfs_kit_py",)
    assert board.payload["initial_projection"]["ready_task_ids"] == list(INITIAL_READY)
    for task_id in INITIAL_READY:
        shard = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 4
        assert INITIAL_SHARDS[shard] == task_id


@pytest.mark.parametrize(
    ("needle", "replacement", "expected_error"),
    (
        (
            "- Generated by: ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1",
            "- Generated by: forged.retry-repair@1",
            "lacks exact operational provenance",
        ),
        (
            "- Retry repair source: KVFS-101",
            "- Retry repair source: KVFS-999",
            "is not a recognized retry repair",
        ),
        (
            "- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/host_contracts.py, ipfs_kit_py/tests/kernel_vfs/contracts/test_host_contracts.py\n- Validation: test -f",
            "- Outputs: ipfs_kit_py/ipfs_kit_py/core/vfs/host_contracts.py, outside-source.py\n- Validation: test -f",
            "output scope differs from source KVFS-101",
        ),
        (
            "- Canonical board task: false\n\n- Acceptance: Retry-budget guardrail filed this from repeated validation failures in KVFS-101.",
            "- Canonical board task: false\n- Scope paths: ipfs_kit_py, outside-source\n\n- Acceptance: Retry-budget guardrail filed this from repeated validation failures in KVFS-101.",
            "declared scope differs from source KVFS-101",
        ),
        (
            "## KVFS-812 Resolve validation retry-budget failure for KVFS-101\n\n- Status: completed",
            "## KVFS-812 Resolve validation retry-budget failure for KVFS-101\n\n- Status: todo",
            "is pending after source KVFS-101 completed",
        ),
    ),
)
def test_operational_retry_appendix_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
    replacement: str,
    expected_error: str,
) -> None:
    text = TODO_PATH.read_text(encoding="utf-8")
    assert needle in text
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(expected_error in error for error in report["errors"])


@pytest.mark.parametrize(
    ("needle", "replacement", "expected_error"),
    (
        (
            "- Generated by: ipfs_accelerate_py.agent_supervisor.reconciliation-guardrail@1",
            "- Generated by: forged.reconciliation-guardrail@1",
            "lacks exact reconciliation provenance",
        ),
        (
            "- Status: completed",
            "- Status: todo",
            "unsafe reconciliation status",
        ),
        (
            "- Is schedulable: false",
            "- Is schedulable: true",
            "reconciliation authority gate mismatch",
        ),
        (
            "- Review only: true",
            "- Review only: false",
            "reconciliation authority gate mismatch",
        ),
        (
            "- Blocked reason: operator_reconciliation_required",
            "- Blocked reason: operator_approval_optional",
            "reconciliation authority gate mismatch",
        ),
        (
            "- Canonical board task: false",
            "- Canonical board task: true",
            "lacks exact reconciliation provenance",
        ),
        (
            "- Reconciliation fingerprint: 32e5200646c95fe450e24df57f17c145ce6f4ad4",
            "- Reconciliation fingerprint: 32e5200646c95fe450e24df57f17c145ce6f4ad5",
            "reconciliation fingerprint mismatch",
        ),
        (
            "- Dedupe key: reconciliation_guardrail:preflight_merge_conflict",
            "- Dedupe key: reconciliation_guardrail:main_checkout_dirty",
            "reconciliation dedupe key mismatch",
        ),
        (
            "- Outputs: data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery, docs/architecture/ipfs_kit_fuse_vfs.todo.md",
            "- Outputs: data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery, docs/architecture/ipfs_kit_fuse_vfs.todo.md, ipfs_kit_py",
            "reconciliation output scope mismatch",
        ),
        (
            "- Depends on:\n- Outputs:",
            "- Depends on: KVFS-811\n- Outputs:",
            "reconciliation appendix must not alter the sealed DAG",
        ),
        (
            "- Outputs: data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery, docs/architecture/ipfs_kit_fuse_vfs.todo.md\n- Board namespace:",
            "- Outputs: data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery, docs/architecture/ipfs_kit_fuse_vfs.todo.md\n- Scope paths: ipfs_kit_py\n- Board namespace:",
            "reconciliation scope authority is unsafe",
        ),
        (
            "- Board namespace: ipfs-kit-kernel-vfs-fuse-v1",
            "- Board namespace: forged-board",
            "reconciliation board namespace mismatch",
        ),
        (
            "- Validation: test -f ",
            "- Validation: sh -c test -f ",
            "reconciliation validation is not fail-closed",
        ),
        (
            "This task is intentionally operator-gated",
            "This task can run unattended",
            "reconciliation acceptance/evidence mismatch",
        ),
    ),
)
def test_operational_reconciliation_appendix_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
    replacement: str,
    expected_error: str,
) -> None:
    text = TODO_PATH.read_text(encoding="utf-8")
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(
        _replace_in_task_block(text, "KVFS-814", needle, replacement),
        encoding="utf-8",
    )
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(expected_error in error for error in report["errors"])


def test_reconciliation_discovery_must_belong_to_a_sibling_repo_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = TODO_PATH.read_text(encoding="utf-8")
    task = next(item for item in board_validator.parse_tasks() if item[0] == "KVFS-814")
    discovery = task[2]["reconciliation discovery"]
    forged = (
        "/tmp/forged/data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery/"
        f"{Path(discovery).name}"
    )
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(
        _replace_in_task_block(
            text,
            "KVFS-814",
            f"- Reconciliation discovery: {discovery}",
            f"- Reconciliation discovery: {forged}",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(
        "invalid reconciliation discovery provenance" in error
        for error in report["errors"]
    )


def test_reconciliation_discovery_must_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = TODO_PATH.read_text(encoding="utf-8")
    discovery = _kvfs814_fields()["reconciliation discovery"]
    missing = str(Path(discovery).with_name(
        Path(discovery).name.replace("2026-08-10", "2099-12-31")
    ))
    assert not Path(missing).exists()
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(
        _replace_in_task_block(
            text,
            "KVFS-814",
            discovery,
            missing,
            count=-1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(
        "KVFS-814 discovery evidence is unavailable" in error
        for error in report["errors"]
    )


def test_reconciliation_discovery_rejects_a_nested_git_directory_as_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = TODO_PATH.read_text(encoding="utf-8")
    discovery = _kvfs814_fields()["reconciliation discovery"]
    suffix = Path(board_validator.RECONCILIATION_OUTPUTS[0])
    repo_root = Path(discovery).parent
    for _part in suffix.parts:
        repo_root = repo_root.parent
    nested_candidate_root = repo_root / "data"
    assert nested_candidate_root.is_dir()
    assert board_validator._git_toplevel(nested_candidate_root) == repo_root
    nested = str(nested_candidate_root / suffix / Path(discovery).name)
    assert not board_validator._supervisor_owned_discovery_path(nested)

    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(
        _replace_in_task_block(
            text,
            "KVFS-814",
            discovery,
            nested,
            count=-1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(
        "invalid reconciliation discovery provenance" in error
        for error in report["errors"]
    )


def test_reconciliation_manifest_rejects_synchronized_board_fingerprint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = TODO_PATH.read_text(encoding="utf-8")
    fingerprint = _kvfs814_fields()["reconciliation fingerprint"]
    drifted = f"{fingerprint[:-1]}{'0' if fingerprint[-1] != '0' else '1'}"
    assert drifted[:12] == fingerprint[:12]
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(
        _replace_in_task_block(
            text,
            "KVFS-814",
            fingerprint,
            drifted,
            count=-1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(
        "reconciliation manifest binding mismatch" in error
        for error in report["errors"]
    )


def test_reconciliation_appendix_rejects_concurrent_duplicate_guardrails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _blocked_kvfs814_board(TODO_PATH.read_text(encoding="utf-8"))
    start = text.index("## KVFS-814 ")
    duplicate = text[start:].replace("KVFS-814", "KVFS-815").replace(
        "kvfs-814", "kvfs-815"
    )
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(f"{text.rstrip()}\n\n{duplicate}", encoding="utf-8")
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(
        "concurrent duplicate operational reconciliation task" in error
        for error in report["errors"]
    )


def test_completed_reconciliation_card_requires_a_receipt_and_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _blocked_kvfs814_board(TODO_PATH.read_text(encoding="utf-8"))
    fields = _kvfs814_fields()
    discovery_path = Path(fields["reconciliation discovery"])
    discovery_without_receipt = discovery_path.read_text(encoding="utf-8").split(
        "\n## Resolution Receipt\n",
        1,
    )[0]
    original_reader = board_validator._read_bounded_regular_file

    def read_fixture(
        task_id: str,
        path: Path,
        *,
        errors: list[str],
    ) -> str | None:
        if task_id == "KVFS-814":
            return discovery_without_receipt
        return original_reader(task_id, path, errors=errors)

    monkeypatch.setattr(
        board_validator,
        "_read_bounded_regular_file",
        read_fixture,
    )
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(
        _replace_in_task_block(
            text,
            "KVFS-814",
            "- Status: blocked",
            "- Status: completed",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert report["pending_operational_task_ids"] == []
    assert any(
        "must have one machine-readable resolution receipt" in error
        for error in report["errors"]
    )
    assert any(
        "resolution receipt anchor mismatch" in error
        for error in report["errors"]
    )


def test_completed_reconciliation_card_accepts_an_anchored_receipt() -> None:
    report = board_validator.validate()

    assert report["valid"] is True
    assert report["pending_operational_task_ids"] == []


def test_reconciliation_resolution_receipt_rejects_unanchored_and_tampered_data(
) -> None:
    fields = _kvfs814_fields()
    receipt = _resolution_receipt(fields)
    anchored_fields = {
        **fields,
        "resolution receipt digest": str(receipt["receipt_digest"]),
    }

    errors: list[str] = []
    board_validator._validate_reconciliation_resolution_receipt(
        task_id="KVFS-814",
        fields=anchored_fields,
        discovery_text=_receipt_section(receipt),
        candidate_count=1,
        errors=errors,
    )
    assert errors == []

    unanchored_errors: list[str] = []
    board_validator._validate_reconciliation_resolution_receipt(
        task_id="KVFS-814",
        fields=fields,
        discovery_text=_receipt_section(receipt),
        candidate_count=1,
        errors=unanchored_errors,
    )
    assert any(
        "resolution receipt anchor mismatch" in error
        for error in unanchored_errors
    )

    tampered = json.loads(json.dumps(receipt))
    tampered["evidence"] = {"result": "changed after hashing"}
    digest_errors: list[str] = []
    board_validator._validate_reconciliation_resolution_receipt(
        task_id="KVFS-814",
        fields=anchored_fields,
        discovery_text=_receipt_section(tampered),
        candidate_count=1,
        errors=digest_errors,
    )
    assert any(
        "resolution receipt digest mismatch" in error
        for error in digest_errors
    )

    tampered["receipt_digest"] = board_validator._resolution_receipt_digest(
        tampered
    )
    anchor_errors: list[str] = []
    board_validator._validate_reconciliation_resolution_receipt(
        task_id="KVFS-814",
        fields=anchored_fields,
        discovery_text=_receipt_section(tampered),
        candidate_count=1,
        errors=anchor_errors,
    )
    assert any(
        "resolution receipt anchor mismatch" in error
        for error in anchor_errors
    )

    duplicate_errors: list[str] = []
    section = _receipt_section(receipt)
    board_validator._validate_reconciliation_resolution_receipt(
        task_id="KVFS-814",
        fields=anchored_fields,
        discovery_text=f"{section}\n{section}",
        candidate_count=1,
        errors=duplicate_errors,
    )
    assert any(
        "must have one machine-readable resolution receipt" in error
        for error in duplicate_errors
    )


def test_reconciliation_evidence_must_be_regular_and_bounded(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.md"
    target.write_text("evidence", encoding="utf-8")
    symlink = tmp_path / "symlink.md"
    symlink.symlink_to(target)
    symlink_errors: list[str] = []

    assert board_validator._read_bounded_regular_file(
        "KVFS-814",
        symlink,
        errors=symlink_errors,
    ) is None
    assert any("regular non-symlink" in error for error in symlink_errors)

    oversized = tmp_path / "oversized.md"
    oversized.write_bytes(
        b"x" * (board_validator.MAX_DISCOVERY_EVIDENCE_BYTES + 1)
    )
    oversized_errors: list[str] = []
    assert board_validator._read_bounded_regular_file(
        "KVFS-814",
        oversized,
        errors=oversized_errors,
    ) is None
    assert any("exceeds 1 MiB" in error for error in oversized_errors)


def test_operational_appendices_do_not_change_canonical_ownership_or_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(board_validator, "TODO_PATH", TODO_PATH)

    report = board_validator.validate()

    assert report["valid"] is True
    assert report["task_count"] == len(TASK_IDS)
    assert report["parsed_task_count"] == len(TASK_IDS) + 3
    assert report["completed_task_ids"] == sorted(set(TASK_IDS) - {"KVFS-811"})
    assert report["ready_task_ids"] == ["KVFS-811"]


def test_validator_rejects_initial_completion_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = TODO_PATH.read_text(encoding="utf-8")
    needle = (
        "## KVFS-000 Seal the architecture, objective heap, task DAG, "
        "and scheduler controls\n\n- Status: completed"
    )
    assert needle in text
    mutated = tmp_path / "mutated.todo.md"
    mutated.write_text(
        text.replace(needle, needle.replace("completed", "todo"), 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(board_validator, "TODO_PATH", mutated)

    report = board_validator.validate()

    assert report["valid"] is False
    assert any(
        "sealed initial completions regressed" in error
        for error in report["errors"]
    )
