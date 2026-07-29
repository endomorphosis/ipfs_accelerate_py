"""Tests for the frozen SwissKnife / IPFS VFS pilot (VFS-037 / VFS-G131)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.vfs_symbolic_pilot import (
    PILOT_OBJECTIVE_ID,
    PILOT_TASK_ID,
    SWISS_KNIFE_VFS_PILOT_SCHEMA,
    PilotArtifactSet,
    PilotConclusion,
    PilotConfig,
    PilotMode,
    PilotStage,
    PilotVerificationError,
    SwissKnifeVfsPilotReport,
    VfsSymbolicPilotError,
    admitted_entries_for_pilot,
    build_hermetic_pilot_forest_policy,
    dry_run_pilot,
    freeze_repository_descriptors,
    is_vfs_relevant_path,
    main,
    scan_inventory,
    verify_pilot,
    verify_pilot_report,
)


def _config(tmp_path: Path, *, seed_broken: bool = True, seed_inconclusive: bool = True) -> PilotConfig:
    policy, swiss, accel = build_hermetic_pilot_forest_policy(
        tmp_path / "forest",
        seed_broken=seed_broken,
        seed_inconclusive=seed_inconclusive,
    )
    return PilotConfig(
        accelerator_root=accel,
        swissknife_root=swiss,
        forest_policy=policy,
        artifact_dir=tmp_path / "artifacts",
        findings_board_path=tmp_path / "findings.todo.md",
        write_artifacts=True,
        write_findings_board=True,
    )


def test_is_vfs_relevant_path_admits_swissknife_and_vfs_closure() -> None:
    assert is_vfs_relevant_path("src/any.ts", repository_alias="swissknife")
    assert is_vfs_relevant_path(
        "ipfs_accelerate_py/agent_supervisor/vfs_surface.py",
        repository_alias="ipfs_accelerate_py",
    )
    assert is_vfs_relevant_path(
        "ipfs_kit_py/vfs/manager.py",
        repository_alias="ipfs_kit_py",
    )
    assert not is_vfs_relevant_path(
        "unrelated/module.py",
        repository_alias="ipfs_accelerate_py",
    )


def test_dry_run_freezes_scans_publishes_and_writes_board(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)

    assert report.schema == SWISS_KNIFE_VFS_PILOT_SCHEMA
    assert report.objective_id == PILOT_OBJECTIVE_ID
    assert report.task_id == PILOT_TASK_ID
    assert report.mode is PilotMode.DRY_RUN
    assert report.conclusion is PilotConclusion.PASSED
    assert report.provider_calls == 0
    assert report.source_mutations == 0
    assert report.authorizes_repair is False
    assert report.is_completion_evidence is False
    assert report.admitted_file_count >= 5
    assert report.swissknife_file_count >= 3
    assert report.vfs_closure_file_count >= 1
    assert report.finding_count >= 1
    assert report.executable_task_count >= 1
    assert report.inconclusive_count >= 1
    assert report.artifacts is not None

    for field_name in (
        "forest_cid",
        "manifest_cid",
        "coverage_cid",
        "inventory_cid",
        "graph_cid",
        "cache_cid",
        "proof_cid",
        "zk_shadow_cid",
        "finding_ledger_cid",
        "taskboard_cid",
    ):
        assert getattr(report.artifacts, field_name)

    stage_names = {stage.stage for stage in report.stages}
    assert stage_names == set(PilotStage)

    artifact_dir = config.resolved_artifact_dir()
    assert (artifact_dir / "report.json").is_file()
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "coverage.json").is_file()
    assert (artifact_dir / "taskboard.json").is_file()

    board = config.resolved_findings_board_path().read_text(encoding="utf-8")
    assert "vfs/swissknife-vfs-pilot@1" in board
    assert "authorizes_repair: `false`" in board
    assert report.artifacts.manifest_cid in board
    assert report.artifacts.taskboard_cid in board


def test_verify_recomputes_without_provider_or_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    verified = verify_pilot_report(report, config=config, recompute=True)

    assert verified.mode is PilotMode.VERIFY
    assert verified.conclusion is PilotConclusion.PASSED
    assert verified.provider_calls == 0
    assert verified.source_mutations == 0
    assert verified.artifacts is not None
    assert verified.artifacts.inventory_cid == report.artifacts.inventory_cid
    assert verified.artifacts.graph_cid == report.artifacts.graph_cid
    assert verified.artifacts.taskboard_cid == report.artifacts.taskboard_cid


def test_verify_fails_on_changed_trees(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)

    # Mutate SwissKnife tree after freeze.
    swiss = config.swissknife_root
    (swiss / "src" / "drift.ts").write_text("export const x = 1;\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "."],
        cwd=swiss,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["git", "commit", "-m", "drift"],
        cwd=swiss,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with pytest.raises(PilotVerificationError) as excinfo:
        verify_pilot_report(report, config=config, recompute=True)
    assert excinfo.value.reason_code in {"changed_trees", "stale_evidence"}


def test_verify_fails_on_noncanonical_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    payload = report.to_dict()
    payload["report_cid"] = "baguqeera" + "0" * 52

    with pytest.raises(PilotVerificationError) as excinfo:
        verify_pilot_report(payload, config=None, recompute=False)
    assert excinfo.value.reason_code == "stale_evidence"


def test_verify_fails_on_forged_authority(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    payload = report.to_dict()
    payload.pop("report_cid", None)
    payload["authorizes_repair"] = True
    with pytest.raises(VfsSymbolicPilotError):
        SwissKnifeVfsPilotReport.from_dict(payload)


def test_inconclusive_findings_are_non_executable(tmp_path: Path) -> None:
    config = _config(tmp_path, seed_broken=False, seed_inconclusive=True)
    report = dry_run_pilot(config)
    assert report.inconclusive_count >= 1
    assert report.executable_task_count == 0
    assert report.review_count >= 1


def test_board_is_bounded_deduplicated_and_goal_backed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    board_json = json.loads(
        (config.resolved_artifact_dir() / "taskboard.json").read_text(encoding="utf-8")
    )
    assert board_json.get("board_namespace") or board_json.get("goal_id")
    # Projections never authorize repair.
    assert board_json.get("authorizes_repair", False) is False
    assert board_json.get("is_completion_evidence", False) is False

    board_md = config.resolved_findings_board_path().read_text(encoding="utf-8")
    assert "VFS-G" in board_md or "goal" in board_md.lower()
    assert report.board_namespace == "ipfs-kit-vfs-symbolic-assurance-v1"


def test_inventory_accounts_for_every_admitted_swissknife_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    forest = freeze_repository_descriptors(config)
    index = scan_inventory(forest)
    admitted = admitted_entries_for_pilot(index)
    swiss = [entry for entry in admitted if entry.repository_alias == "swissknife"]
    assert swiss
    # Every included SwissKnife parser-eligible entry is admitted.
    for entry in index.entries:
        if entry.repository_alias != "swissknife":
            continue
        if entry.inclusion == "included" and entry.parser_eligible:
            assert any(
                item.entry_cid == entry.entry_cid for item in swiss
            ), entry.relative_path


def test_report_round_trip_is_canonical(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = dry_run_pilot(config)
    restored = SwissKnifeVfsPilotReport.from_dict(json.loads(report.to_json()))
    assert restored.report_cid == report.report_cid
    assert restored.artifacts is not None
    assert PilotArtifactSet.from_dict(restored.artifacts.to_dict()).manifest_cid == (
        report.artifacts.manifest_cid
    )


def test_cli_verify_hermetic_exits_zero(tmp_path: Path) -> None:
    code = main(
        [
            "--verify",
            "--hermetic",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--findings-board",
            str(tmp_path / "findings.todo.md"),
            "--json",
        ]
    )
    assert code == 0


def test_cli_dry_run_hermetic_exits_zero(tmp_path: Path) -> None:
    code = main(
        [
            "--dry-run",
            "--hermetic",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--findings-board",
            str(tmp_path / "findings.todo.md"),
        ]
    )
    assert code == 0
    assert (tmp_path / "artifacts" / "report.json").is_file()


def test_verify_entry_self_check(tmp_path: Path) -> None:
    config = _config(tmp_path)
    verified = verify_pilot(config)
    assert verified.mode is PilotMode.VERIFY
    assert verified.conclusion is PilotConclusion.PASSED
