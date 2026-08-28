"""Recovery admission for dependency-preflight bounded-store artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)
from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import (
    BoundedArtifactStore,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    canonical_project_dependency_preflight_receipt_bytes,
    project_dependency_preflight_error_receipt,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "user.name=Recovery Fixture",
            "-c",
            "user.email=recovery-fixture@example.invalid",
            "-C",
            str(repo),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _seed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("recovery fixture\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "data/agent_supervisor/recovery-fixture/run-v1/\n",
        encoding="utf-8",
    )
    _git(repo, "add", "README.md", ".gitignore")
    _git(repo, "commit", "-q", "-m", "seed recovery fixture")
    runtime_root = (
        repo / "data/agent_supervisor/recovery-fixture/run-v1"
    )
    for directory in (
        runtime_root,
        runtime_root / "state",
        runtime_root / "worktrees",
        runtime_root / "merge-queue",
    ):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    return repo


def _runtime_root(repo: Path) -> Path:
    return repo / "data/agent_supervisor/recovery-fixture/run-v1"


def _runtime_contract(
    repo: Path,
    *,
    include_selected_lane: bool = True,
) -> dict[str, Any]:
    runtime_root = _runtime_root(repo)
    state_root = runtime_root / "state"
    bindings = [
        {
            "slice_id": "slice-0",
            "lane_index": 0,
            "lane_id": "lane-0",
            "active_task_id": "TEST-A",
            "task_ids": ("TEST-A",),
            "attempt": 1,
        },
    ]
    if include_selected_lane:
        bindings.append(
            {
                "slice_id": "slice-1",
                "lane_index": 1,
                "lane_id": "lane-1",
                "active_task_id": "TEST-B",
                "task_ids": ("TEST-B",),
                "attempt": 1,
            }
        )
    return {
        "runtime_roots": (
            state_root,
            runtime_root / "worktrees",
            runtime_root / "merge-queue",
        ),
        "owner_bound_artifacts": (),
        "runtime_bindings": tuple(bindings),
        "slice_id": "slice-1",
        "lane_id": "lane-1",
        "state_dir": state_root / "lane-1",
        "state_prefix": "fixture_lane_1",
    }


def _put_preflight_blob(
    repo: Path,
    *,
    kind: str = "validation_project_dependency_preflight_receipt",
) -> Path:
    store_root = (
        _runtime_root(repo)
        / "state/lane-1/dependency-preflight-artifacts"
    )
    receipt = project_dependency_preflight_error_receipt(
        repo,
        ["python -m pytest -q"],
        RuntimeError("fixture dependency failure"),
    )
    canonical = canonical_project_dependency_preflight_receipt_bytes(receipt)
    with BoundedArtifactStore(store_root) as store:
        reference = store.put_blob(
            canonical,
            kind=kind,
            retention_class="checkpoint",
            media_type="application/json",
        )
    digest = reference.digest.removeprefix("sha256:")
    return store_root / "blobs/sha256" / digest[:2] / f"{digest}.blob"


def _snapshot(
    repo: Path,
    *,
    include_selected_lane: bool = True,
) -> tuple[dict[str, Any], ...]:
    return runner._snapshot_plan_bound_recovery_artifacts(
        root=repo,
        **_runtime_contract(repo, include_selected_lane=include_selected_lane),
    )


def test_recovery_snapshots_exact_active_lane_preflight_bounded_store(
    tmp_path: Path,
) -> None:
    repo = _seed_repo(tmp_path)
    blob_path = _put_preflight_blob(repo)

    evidence = _snapshot(repo)

    store_relative = Path(
        "data/agent_supervisor/recovery-fixture/run-v1/"
        "state/lane-1/dependency-preflight-artifacts"
    )
    expected = {
        (store_relative / ".bounded-store.lock").as_posix(),
        (store_relative / "manifest.json").as_posix(),
        (store_relative / "manifest.previous.json").as_posix(),
        blob_path.relative_to(repo).as_posix(),
    }
    assert {item["path"] for item in evidence} == expected
    assert {item["kind"] for item in evidence} == {"file"}


def test_recovery_rejects_preflight_store_without_selected_lane_binding(
    tmp_path: Path,
) -> None:
    repo = _seed_repo(tmp_path)
    _put_preflight_blob(repo)

    with pytest.raises(ValueError, match="selected recovery lane is absent"):
        _snapshot(repo, include_selected_lane=False)


def test_recovery_rejects_generic_bounded_store_under_preflight_path(
    tmp_path: Path,
) -> None:
    repo = _seed_repo(tmp_path)
    _put_preflight_blob(repo, kind="generic_runtime_artifact")

    with pytest.raises(ValueError, match="blob metadata is mixed"):
        _snapshot(repo)


def test_recovery_rejects_tampered_manifest_bound_preflight_blob(
    tmp_path: Path,
) -> None:
    repo = _seed_repo(tmp_path)
    blob_path = _put_preflight_blob(repo)
    blob_path.write_bytes(blob_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="blob custody is unsafe"):
        _snapshot(repo)
