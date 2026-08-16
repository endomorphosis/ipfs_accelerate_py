"""DCR-011 deterministic multi-root forest observation tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_repair_forest import (
    DCR_REQUIRED_ROOT_IDS,
    DeterministicRepairForestError,
    capture_deterministic_repair_forest,
    non_authoritative_snapshot,
    verify_deterministic_repair_forest,
)

_ROOTS = (
    ("swissknife", "swissknife", "consumer"),
    ("mcp-plus-plus", "Mcp-Plus-Plus", "consumer"),
    ("ipfs-accelerate", "external/ipfs_accelerate", "provider"),
    ("ipfs-datasets", "external/ipfs_datasets", "provider"),
    ("ipfs-kit", "external/ipfs_kit", "provider"),
)


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def _commit(path: Path) -> None:
    _git(path, "add", ".")
    _git(
        path, "-c", "user.name=DCR", "-c", "user.email=dcr@example.test", "commit", "-m", "initial"
    )


def _forest(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "forest"
    workspace.mkdir()
    for _root_id, relative, _role in _ROOTS:
        root = workspace / relative
        root.mkdir(parents=True)
        _git(root, "init")
        (root / "tracked.txt").write_text(relative, encoding="utf-8")
        _commit(root)
    _git(workspace, "init")
    (workspace / "orchestration.txt").write_text("orchestration", encoding="utf-8")
    _git(workspace, "add", "orchestration.txt")
    _git(
        workspace,
        "-c",
        "user.name=DCR",
        "-c",
        "user.email=dcr@example.test",
        "commit",
        "-m",
        "orchestration",
    )
    _git(workspace, "add", *(relative for _id, relative, _role in _ROOTS))
    _git(
        workspace,
        "-c",
        "user.name=DCR",
        "-c",
        "user.email=dcr@example.test",
        "commit",
        "-m",
        "gitlinks",
    )
    policy = workspace / "config" / "roots.json"
    policy.parent.mkdir()
    policy.write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/deterministic-repair-roots@1",
                "interface": "RepairRootOwnership@1",
                "roots": [
                    {
                        "id": "orchestration",
                        "relative_path": ".",
                        "role": "orchestration_only",
                        "allowed_write_prefixes": [],
                        "pin_path": "",
                    },
                    *[
                        {
                            "id": root_id,
                            "relative_path": relative,
                            "role": role,
                            "allowed_write_prefixes": ["."],
                            "pin_path": relative,
                        }
                        for root_id, relative, role in _ROOTS
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )
    config = workspace / "config" / "policy.json"
    config.write_text('{"policy":"dcr-011"}', encoding="utf-8")
    return workspace, policy, config


def test_capture_binds_six_roots_gitlinks_overlays_and_relocates(tmp_path: Path) -> None:
    workspace, policy, config = _forest(tmp_path)
    manifest = capture_deterministic_repair_forest(
        workspace_root=workspace,
        root_policy_path=policy,
        config_paths=(config,),
        exclusions=(".cache",),
    )
    assert manifest["authoritative"] is True
    assert tuple(item["root_id"] for item in manifest["portable"]["roots"]) == DCR_REQUIRED_ROOT_IDS
    assert all("realpath" not in item for item in manifest["portable"]["roots"])
    assert all(
        item["content_digest"].startswith("sha256:") for item in manifest["portable"]["roots"]
    )
    assert all(item["planning_gitlink_revision"] for item in manifest["portable"]["roots"][1:])
    assert set(manifest["host"]["roots"]) == set(DCR_REQUIRED_ROOT_IDS)
    assert all(item["realpath"] for item in manifest["host"]["roots"].values())

    relocated = tmp_path / "relocated"
    shutil.copytree(workspace, relocated)
    replay = capture_deterministic_repair_forest(
        workspace_root=relocated,
        root_policy_path=relocated / "config/roots.json",
        config_paths=(relocated / "config/policy.json",),
        exclusions=(".cache",),
    )
    assert replay["portable"] == manifest["portable"]
    assert replay["portable_identity"] == manifest["portable_identity"]
    assert replay["host"]["workspace_realpath"] != manifest["host"]["workspace_realpath"]


def test_overlay_or_missing_root_invalidates_verification(tmp_path: Path) -> None:
    workspace, policy, config = _forest(tmp_path)
    manifest = capture_deterministic_repair_forest(
        workspace_root=workspace, root_policy_path=policy, config_paths=(config,)
    )
    verify_deterministic_repair_forest(
        manifest, workspace_root=workspace, root_policy_path=policy, config_paths=(config,)
    )
    (workspace / "external/ipfs_accelerate" / "tracked.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(DeterministicRepairForestError, match="dirty overlay changed"):
        verify_deterministic_repair_forest(
            manifest, workspace_root=workspace, root_policy_path=policy, config_paths=(config,)
        )
    shutil.rmtree(workspace / "external/ipfs_kit")
    with pytest.raises(DeterministicRepairForestError, match="every required root"):
        capture_deterministic_repair_forest(
            workspace_root=workspace, root_policy_path=policy, config_paths=(config,)
        )


def test_non_authoritative_snapshot_cannot_be_verified(tmp_path: Path) -> None:
    workspace, policy, config = _forest(tmp_path)
    captured = capture_deterministic_repair_forest(
        workspace_root=workspace, root_policy_path=policy, config_paths=(config,)
    )
    snapshot = non_authoritative_snapshot(captured["portable"], note="active overlays")
    with pytest.raises(DeterministicRepairForestError, match="non-authoritative"):
        verify_deterministic_repair_forest(
            snapshot, workspace_root=workspace, root_policy_path=policy, config_paths=(config,)
        )
