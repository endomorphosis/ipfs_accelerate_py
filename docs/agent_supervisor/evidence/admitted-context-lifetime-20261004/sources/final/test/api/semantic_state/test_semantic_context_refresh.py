"""Real producer reconstruction and task-bound refresh across edited sources."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("ipfs_datasets_py.logic.software_contracts.semantic_state")

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    load_semantic_worker_context,
    prepare_semantic_context,
    resolve_semantic_worker_context,
)


@pytest.fixture
def nomination(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("from dependency import add\ndef target(x): return add(x, 1)\n")
    (repo / "dependency.py").write_text("def add(a, b): return a + b\n")
    prepared = prepare_semantic_context(
        repository=repo,
        paths=["target.py", "dependency.py"],
        required_raw_paths=["target.py"],
        objective="Maintain addition",
        task_id="REFRESH-001",
        output=repo / ".semantic/initial",
    )
    return dict(
        repository=repo,
        artifact=".semantic/initial/worker-context.json",
        expected_sha256=prepared["worker_payload_sha256"],
        task_id="REFRESH-001",
        refresh_output=repo / ".semantic/retries",
        attempt_id="attempt:2",
    )


def test_refresh_reconstructs_changed_source_and_keeps_prior_nomination(nomination):
    args = nomination
    repo = args["repository"]
    original = (repo / args["artifact"]).read_bytes()
    old = json.loads(original)
    assert old["reconstruction"]["nomination_matched"] is True
    changed = "def add(a, b):\n    return a - b\n"
    (repo / "dependency.py").write_text(changed)
    resolved = resolve_semantic_worker_context(**args)
    new = json.loads(resolved["text"])
    assert resolved["refreshed"] is True
    assert (repo / args["artifact"]).read_bytes() == original
    assert new["semantic_root_cid"] != old["semantic_root_cid"]
    assert new["scope_cid"] != old["scope_cid"]
    assert (
        new["manifest"]["dependency.py"]["sha256"] == hashlib.sha256(changed.encode()).hexdigest()
    )
    assert new["required_raw_paths"] == old["required_raw_paths"]
    assert new["raw_sources"]["target.py"] == (repo / "target.py").read_text()
    lineage = new["refresh_lineage"]
    assert lineage["previous_payload_sha256"] == args["expected_sha256"]
    assert lineage["previous_semantic_root_cid"] == old["semantic_root_cid"]
    assert lineage["semantic_root_cid"] == new["semantic_root_cid"]
    assert set(lineage["source_delta"]) == {"dependency.py"}
    assert lineage["source_delta"]["dependency.py"]["before"] == old["manifest"]["dependency.py"]
    assert lineage["source_delta"]["dependency.py"]["after"] == new["manifest"]["dependency.py"]
    assert lineage["semantic_acceptance_authority"] is False
    assert new["reconstruction"]["nomination_matched"] is True
    assert new["reconstruction"]["full_repository"] is False
    assert new["doctor_snapshot_id"] != old["doctor_snapshot_id"]
    # Reopen actual durable world observations; the reconstruction is bound.
    from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_database import (
        ProgramWorldDatabase,
    )

    output = repo / Path(resolved["artifact"]).parent
    records = ProgramWorldDatabase(
        output / "world.duckdb", output / "world-lake"
    ).records_for_decision(task_id="REFRESH-001")
    assert records["n"] == 1
    assert (
        load_semantic_worker_context(
            repository=repo,
            artifact=resolved["artifact"],
            expected_sha256=resolved["sha256"],
            task_id=args["task_id"],
        )
        == resolved["text"]
    )


def test_fresh_nomination_does_not_write_refresh_artifacts(nomination):
    result = resolve_semantic_worker_context(**nomination)
    assert result["refreshed"] is False
    assert result["artifact"] == nomination["artifact"]
    assert not nomination["refresh_output"].exists()


@pytest.mark.parametrize("defect", ["digest", "task", "deleted", "symlink", "output_escape"])
def test_refresh_cannot_bypass_nomination_and_scope_checks(nomination, defect):
    args = dict(nomination)
    repo = args["repository"]
    (repo / "dependency.py").write_text("def add(a,b): return a-b\n")
    if defect == "digest":
        args["expected_sha256"] = "0" * 64
    elif defect == "task":
        args["task_id"] = "OTHER"
    elif defect == "deleted":
        (repo / "target.py").unlink()
    elif defect == "symlink":
        original = (repo / "target.py").read_text()
        (repo.parent / "outside.py").write_text(original)
        (repo / "target.py").unlink()
        (repo / "target.py").symlink_to(repo.parent / "outside.py")
    elif defect == "output_escape":
        args["refresh_output"] = repo.parent / "outside"
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_semantic_worker_context(**args)
    assert not nomination["refresh_output"].exists()


def test_refresh_detects_second_edit_after_preparation(nomination):
    (nomination["repository"] / "target.py").write_text("def target(x): return x+1\n")
    refreshed = resolve_semantic_worker_context(**nomination)
    (nomination["repository"] / "target.py").write_text("def target(x): return x+2\n")
    with pytest.raises(ValueError, match="stale"):
        load_semantic_worker_context(
            repository=nomination["repository"],
            artifact=refreshed["artifact"],
            expected_sha256=refreshed["sha256"],
            task_id=nomination["task_id"],
        )


def test_cold_reconstruction_disagreement_is_not_persisted(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_context_runtime as runtime

    (tmp_path / "target.py").write_text("def target(x): return x+1\n")
    scan = runtime._scan_scoped_sources
    calls = []

    def disagree(sources, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            sources = {"target.py": b"def target(x): return x-1\n"}
        return scan(sources, **kwargs)

    monkeypatch.setattr(runtime, "_scan_scoped_sources", disagree)
    with pytest.raises(ValueError, match="reconstruction differs"):
        prepare_semantic_context(
            repository=tmp_path,
            paths=["target.py"],
            required_raw_paths=["target.py"],
            objective="Preserve addition",
            task_id="REFRESH-001",
            output=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()


def test_native_scanner_exclusions_cannot_hide_explicit_scope(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "target.py").write_text("def target(x): return x+1\n")
    with pytest.raises(ValueError, match="omitted a scoped source"):
        prepare_semantic_context(
            repository=tmp_path,
            paths=["__pycache__/target.py"],
            required_raw_paths=["__pycache__/target.py"],
            objective="Preserve addition",
            task_id="REFRESH-001",
            output=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()
