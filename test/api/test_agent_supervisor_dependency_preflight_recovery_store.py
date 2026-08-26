"""Plan-bound recovery validation for dependency-preflight artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    artifact_store as artifact_store_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.artifact_store import (
    ArtifactQuotaPolicy,
    BoundedArtifactStore,
)
from ipfs_accelerate_py.agent_supervisor.objectives.scan_receipts import (
    ScanMode,
    ScanTerminalReason,
    build_scan_result,
    persist_scan_receipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    canonical_project_dependency_preflight_receipt_bytes,
    project_dependency_preflight_error_receipt,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _seed_store(tmp_path: Path) -> tuple[Path, Path, Path]:
    state_root = tmp_path / "repo" / "state"
    store_root = state_root / "lane-0" / "dependency-preflight-artifacts"
    receipt = project_dependency_preflight_error_receipt(
        tmp_path / "workspace",
        ("python -m pytest -q",),
        RuntimeError("fixture dependency preflight failure"),
    )
    canonical = canonical_project_dependency_preflight_receipt_bytes(receipt)
    with BoundedArtifactStore(store_root) as store:
        reference = store.put_blob(
            canonical,
            kind="validation_project_dependency_preflight_receipt",
            retention_class="checkpoint",
            media_type="application/json",
        )
    digest = reference.digest.removeprefix("sha256:")
    blob_path = store_root / "blobs" / "sha256" / digest[:2] / f"{digest}.blob"
    return state_root, store_root, blob_path


def _seed_store_at(store_root: Path, tmp_path: Path) -> None:
    receipt = project_dependency_preflight_error_receipt(
        tmp_path / "workspace",
        ("python -m pytest -q",),
        RuntimeError("fixture dependency preflight failure"),
    )
    prior_umask = os.umask(0o077)
    try:
        with BoundedArtifactStore(store_root) as store:
            store.put_blob(
                canonical_project_dependency_preflight_receipt_bytes(receipt),
                kind="validation_project_dependency_preflight_receipt",
                retention_class="checkpoint",
                media_type="application/json",
            )
    finally:
        os.umask(prior_umask)


def _ignored_runtime_repo(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "recovery@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Recovery Test"],
        cwd=repo,
        check=True,
    )
    ignored_root = "data/agent_supervisor/recovery-concurrency/run-v1"
    (repo / ".gitignore").write_text(ignored_root + "/\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("accepted\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "tracked.txt"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    runtime_root = repo / ignored_root
    state_root = runtime_root / "state"
    worktree_root = runtime_root / "worktrees"
    merge_root = runtime_root / "merge-queue"
    for directory in (state_root, worktree_root, merge_root):
        directory.mkdir(parents=True, mode=0o700)
    for directory in (runtime_root, state_root, worktree_root, merge_root):
        directory.chmod(0o700)
    return repo, state_root, worktree_root, merge_root


def _store_artifacts(store_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in store_root.rglob("*")
                if path.is_file() or path.is_symlink()
            ),
            key=str,
        )
    )


def _validate_store(
    state_root: Path,
    store_root: Path,
    artifacts: tuple[Path, ...] | None = None,
) -> tuple[dict[str, object], ...]:
    del artifacts
    return multi_runner_module._validate_plan_bound_dependency_preflight_store(
        accepted_tree_root=state_root.parent,
        store_root=store_root,
    )


def _classify(state_root: Path, artifact: Path) -> str:
    lane_root = state_root / "lane-0"
    return multi_runner_module._plan_bound_recovery_runtime_kind(
        artifact,
        directory_projection=False,
        runtime_roots=(
            state_root,
            state_root.parent / "worktrees",
            state_root.parent / "merge-queue",
        ),
        owner_bound_artifacts=(),
        runtime_bindings=(
            {
                "lane_index": 0,
                "lane_id": "lane-0",
                "active_task_id": "LGCVF-080",
                "task_ids": ("LGCVF-080",),
                "attempt": 1,
            },
        ),
        state_dir=lane_root,
        state_prefix="lgcvf_lane_0",
    )


def _reseal_manifest(path: Path, manifest: dict[str, object]) -> None:
    digest_body = dict(manifest)
    digest_body.pop("manifest_digest", None)
    manifest["manifest_digest"] = "sha256:" + hashlib.sha256(
        _canonical_json_bytes(digest_body)
    ).hexdigest()
    path.write_bytes(_canonical_json_bytes(manifest) + b"\n")
    path.chmod(0o600)


def test_bound_lane_accepts_one_aggregate_bounded_store(tmp_path: Path) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    artifacts = _store_artifacts(store_root)

    assert artifacts
    assert {
        _classify(state_root, artifact) for artifact in artifacts
    } == {"dependency-preflight-store"}
    evidence = _validate_store(state_root, store_root, artifacts)
    assert len(evidence) == len(artifacts)
    assert {item["path"] for item in evidence} == {
        artifact.relative_to(state_root.parent).as_posix()
        for artifact in artifacts
    }
    assert {item["kind"] for item in evidence} == {"file"}


def test_bounded_store_creates_private_intermediate_blob_directory(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    prior_umask = os.umask(0)
    try:
        with BoundedArtifactStore(store_root):
            pass
    finally:
        os.umask(prior_umask)

    assert stat.S_IMODE((store_root / "blobs").stat().st_mode) == 0o700


def test_aggregate_uses_each_semantic_read_as_its_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    artifacts = _store_artifacts(store_root)
    reads: Counter[Path] = Counter()
    stable_read = multi_runner_module._read_stable_regular_bytes

    def counted_read(path: Path, *, max_bytes: int = 1_048_576):
        reads[Path(path)] += 1
        return stable_read(path, max_bytes=max_bytes)

    monkeypatch.setattr(
        multi_runner_module,
        "_read_stable_regular_bytes",
        counted_read,
    )
    monkeypatch.setattr(
        multi_runner_module,
        "_plan_bound_recovery_artifact_evidence",
        lambda *_args, **_kwargs: pytest.fail("aggregate performed a generic reread"),
    )

    evidence = _validate_store(state_root, store_root, artifacts)
    assert len(evidence) == len(artifacts)
    assert reads == Counter(
        {
            artifact: 1
            for artifact in artifacts
            if artifact.name != ".bounded-store.lock"
        }
    )


def test_aggregate_holds_shared_store_lock_across_semantic_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    manifest_reader = (
        multi_runner_module._read_plan_bound_dependency_preflight_manifest
    )
    exclusive_attempt_was_blocked = False

    def read_while_probing_lock(accepted_tree_root: Path, artifact: Path):
        nonlocal exclusive_attempt_was_blocked
        result = manifest_reader(accepted_tree_root, artifact)
        if artifact.name == "manifest.json" and not exclusive_attempt_was_blocked:
            descriptor = os.open(store_root / ".bounded-store.lock", os.O_RDWR)
            try:
                with pytest.raises((BlockingIOError, OSError)):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                exclusive_attempt_was_blocked = True
            finally:
                os.close(descriptor)
        return result

    monkeypatch.setattr(
        multi_runner_module,
        "_read_plan_bound_dependency_preflight_manifest",
        read_while_probing_lock,
    )

    _validate_store(state_root, store_root)
    assert exclusive_attempt_was_blocked is True


def test_bound_lane_rejects_tampered_dependency_preflight_blob(
    tmp_path: Path,
) -> None:
    state_root, store_root, blob_path = _seed_store(tmp_path)
    blob_path.write_bytes(blob_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="blob custody is unsafe"):
        _validate_store(state_root, store_root)


def test_bound_lane_rejects_orphan_and_foreign_store_files(
    tmp_path: Path,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    orphan_receipt = project_dependency_preflight_error_receipt(
        tmp_path / "other-workspace",
        ("python -m pytest -q",),
        RuntimeError("orphan dependency preflight receipt"),
    )
    orphan_bytes = canonical_project_dependency_preflight_receipt_bytes(
        orphan_receipt
    )
    orphan_digest = hashlib.sha256(orphan_bytes).hexdigest()
    orphan_path = (
        store_root
        / "blobs"
        / "sha256"
        / orphan_digest[:2]
        / f"{orphan_digest}.blob"
    )
    orphan_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    orphan_path.write_bytes(orphan_bytes)
    orphan_path.chmod(0o600)

    with pytest.raises(ValueError, match="live blobs differ"):
        _validate_store(state_root, store_root)

    orphan_path.unlink()
    foreign_path = store_root / "foreign.tmp"
    foreign_path.write_bytes(b"")
    foreign_path.chmod(0o600)
    with pytest.raises(ValueError, match="noncanonical projection"):
        _validate_store(state_root, store_root)


@pytest.mark.parametrize(
    ("metadata_case", "expected_error"),
    (
        ("malformed", "blob metadata is malformed"),
        ("mixed", "blob metadata is mixed"),
    ),
)
def test_bound_lane_rejects_malformed_or_mixed_manifest_metadata(
    tmp_path: Path,
    metadata_case: str,
    expected_error: str,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    manifest_path = store_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = next(iter(manifest["blobs"].values()))
    if metadata_case == "malformed":
        metadata.pop("references")
    else:
        metadata["kind"] = "foreign_artifact"
    _reseal_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=expected_error):
        _validate_store(state_root, store_root)


def test_manifest_derived_blob_rejects_symlinked_shard_parent(
    tmp_path: Path,
) -> None:
    state_root, store_root, blob_path = _seed_store(tmp_path)
    artifacts = _store_artifacts(store_root)
    outside_shard = tmp_path / "outside-shard"
    blob_path.parent.rename(outside_shard)
    blob_path.parent.symlink_to(outside_shard, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        _validate_store(state_root, store_root, artifacts)


def test_store_scan_validates_empty_shards_and_empty_projections(
    tmp_path: Path,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    canonical_empty_shard = store_root / "blobs" / "sha256" / "aa"
    canonical_empty_shard.mkdir(mode=0o700)
    assert _validate_store(state_root, store_root)

    foreign_empty_shard = store_root / "blobs" / "sha256" / "zz"
    foreign_empty_shard.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="blob shard is noncanonical"):
        _validate_store(state_root, store_root)

    foreign_empty_shard.rmdir()
    projection = store_root / "projections" / "foreign.json"
    projection.write_bytes(b"{}\n")
    projection.chmod(0o600)
    with pytest.raises(ValueError, match="noncanonical projection"):
        _validate_store(state_root, store_root)


def test_snapshot_expands_exact_ignored_run_and_rejects_foreign_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "recovery@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Recovery Test"],
        cwd=repo,
        check=True,
    )
    ignored_root = (
        "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
    )
    (repo / ".gitignore").write_text(
        ignored_root + "run-v*/\n",
        encoding="utf-8",
    )
    (repo / "tracked.txt").write_text("accepted\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "tracked.txt"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )

    runtime_root = repo / ignored_root / "run-v39"
    state_root = runtime_root / "state"
    worktree_root = runtime_root / "worktrees"
    merge_root = runtime_root / "merge-queue"
    lane_root = state_root / "lane-0"
    portal_root = lane_root / "lgcvf_lane_0_database_portal_attempts"
    store_root = portal_root / "dependency-preflight-artifacts"
    worktree_root.mkdir(parents=True, mode=0o700)
    merge_root.mkdir(parents=True, mode=0o700)
    receipt = project_dependency_preflight_error_receipt(
        tmp_path / "workspace",
        ("python -m pytest -q",),
        RuntimeError("fixture dependency preflight failure"),
    )
    with BoundedArtifactStore(store_root) as store:
        store.put_blob(
            canonical_project_dependency_preflight_receipt_bytes(receipt),
            kind="validation_project_dependency_preflight_receipt",
            retention_class="checkpoint",
            media_type="application/json",
        )
    for private_runtime_directory in (
        runtime_root,
        state_root,
        lane_root,
        portal_root,
    ):
        private_runtime_directory.chmod(0o700)
    historical = repo / ignored_root / "run-v38" / "foreign.bin"
    historical.parent.mkdir(parents=True, mode=0o700)
    historical.write_bytes(b"historical ignored runtime\n")
    external_database = lane_root / "lgcvf_lane_0_database_execution.duckdb"
    external_database.write_bytes(b"typed external database authority\n")
    external_database.chmod(0o600)
    external_wal = lane_root / "lgcvf_lane_0_database_execution.duckdb.wal"
    external_wal.write_bytes(b"typed external WAL authority\n")
    external_wal.chmod(0o600)
    supervisor_pid_update_lock = (
        lane_root / ".lgcvf_lane_0_supervisor.pid.update.lock"
    )
    supervisor_pid_update_lock.write_bytes(b"")
    supervisor_pid_update_lock.chmod(0o600)
    scan_receipt_root = lane_root / "lgcvf_lane_0_scan_receipts"
    scan_result = build_scan_result(
        ScanTerminalReason.DISABLED,
        ScanMode.EXHAUSTIVE,
        "recovery-test/v1",
        repo,
        datetime.now(timezone.utc),
        identity={
            "repository_id": "repository:sha256:" + "a" * 64,
            "tree_id": "sha256:" + "b" * 64,
        },
    )
    persist_scan_receipt(
        scan_result,
        scan_receipt_root,
        scan_kind="fixture",
        relative_to=lane_root,
    )
    scan_receipt = next(scan_receipt_root.glob("*.json"))
    scan_receipt.chmod(0o600)
    attempt_root = portal_root / ("a" * 24)
    attempt_root.mkdir(mode=0o700)
    task_projection = attempt_root / "task-projection.md"
    task_projection.write_text("# sealed Portal projection\n", encoding="utf-8")
    task_projection.chmod(0o600)
    implementation_logs = attempt_root / "implementation-logs"
    implementation_logs.mkdir(mode=0o700)
    reconciliation_log = (
        implementation_logs
        / "lgcvf-080-reconciliation-validation-0123456789ab-"
        "0123456789abcdef.log"
    )
    reconciliation_log.write_bytes(b"reconciliation validation\n")
    reconciliation_log.chmod(0o600)
    seed_recovery_dir = implementation_logs / "seed_recovery"
    seed_recovery_dir.mkdir(mode=0o700)
    seed_recovery = seed_recovery_dir / "lgcvf-080-attempt-1-seed-recovery.md"
    seed_recovery.write_bytes(b"seed recovery\n")
    seed_recovery.chmod(0o600)
    implementation_lock = attempt_root / "implementation.lock"
    implementation_lock.write_bytes(b"{}\n")
    implementation_lock.chmod(0o600)
    context_receipt = implementation_logs / "lgcvf-080-base-context-receipt.json"
    context_receipt.write_bytes(b"{}\n")
    context_receipt.chmod(0o600)
    portal_identity_checks: list[Path] = []

    def record_portal_identity(**kwargs):
        portal_identity_checks.append(Path(kwargs["attempt_root"]))
        return "LGCVF-080", {}

    monkeypatch.setattr(
        multi_runner_module,
        "_validate_plan_bound_portal_attempt_identity",
        record_portal_identity,
    )
    generic_evidence = (
        multi_runner_module._plan_bound_recovery_artifact_evidence
    )

    def reject_external_generic_read(root, artifact, *, workspace):
        if artifact in {external_database, supervisor_pid_update_lock}:
            pytest.fail("external authority was content-bound as recovery evidence")
        return generic_evidence(root, artifact, workspace=workspace)

    monkeypatch.setattr(
        multi_runner_module,
        "_plan_bound_recovery_artifact_evidence",
        reject_external_generic_read,
    )
    runtime_bindings = (
        {
            "slice_id": "slice-0",
            "lane_index": 0,
            "lane_id": "lane-0",
            "active_task_id": "LGCVF-080",
            "task_ids": ("LGCVF-080",),
            "attempt": 1,
        },
    )
    evidence = multi_runner_module._snapshot_plan_bound_recovery_artifacts(
        root=repo,
        runtime_roots=(state_root, worktree_root, merge_root),
        owner_bound_artifacts=(supervisor_pid_update_lock,),
        runtime_bindings=runtime_bindings,
        slice_id="slice-0",
        lane_id="lane-0",
        state_dir=lane_root,
        state_prefix="lgcvf_lane_0",
    )
    assert evidence
    external_database_path = external_database.relative_to(repo).as_posix()
    assert all(item["path"] != external_database_path for item in evidence)
    external_wal_path = external_wal.relative_to(repo).as_posix()
    assert all(item["path"] != external_wal_path for item in evidence)
    supervisor_pid_update_lock_path = (
        supervisor_pid_update_lock.relative_to(repo).as_posix()
    )
    assert all(
        item["path"] != supervisor_pid_update_lock_path for item in evidence
    )
    evidence_paths = {str(item["path"]) for item in evidence}
    assert scan_receipt.relative_to(repo).as_posix() in evidence_paths
    assert task_projection.relative_to(repo).as_posix() in evidence_paths
    assert implementation_lock.relative_to(repo).as_posix() in evidence_paths
    assert reconciliation_log.relative_to(repo).as_posix() in evidence_paths
    assert seed_recovery.relative_to(repo).as_posix() in evidence_paths
    assert context_receipt.relative_to(repo).as_posix() in evidence_paths
    assert portal_identity_checks == [attempt_root]

    foreign = lane_root / "foreign.bin"
    foreign.write_bytes(b"unknown current runtime\n")
    with pytest.raises(ValueError, match="noncanonical runtime projection"):
        multi_runner_module._snapshot_plan_bound_recovery_artifacts(
            root=repo,
            runtime_roots=(state_root, worktree_root, merge_root),
            owner_bound_artifacts=(),
            runtime_bindings=runtime_bindings,
            slice_id="slice-0",
            lane_id="lane-0",
            state_dir=lane_root,
            state_prefix="lgcvf_lane_0",
        )

    foreign.unlink()
    foreign_lane = state_root / "lane-999"
    foreign_lane.mkdir(mode=0o700)
    foreign_database = (
        foreign_lane / "lgcvf_lane_999_database_execution.duckdb"
    )
    foreign_database.write_bytes(b"foreign lane database\n")
    foreign_database.chmod(0o600)
    with pytest.raises(ValueError, match="noncanonical runtime projection"):
        multi_runner_module._snapshot_plan_bound_recovery_artifacts(
            root=repo,
            runtime_roots=(state_root, worktree_root, merge_root),
            owner_bound_artifacts=(),
            runtime_bindings=runtime_bindings,
            slice_id="slice-0",
            lane_id="lane-0",
            state_dir=lane_root,
            state_prefix="lgcvf_lane_0",
        )


def test_snapshot_locks_only_the_selected_lane_dependency_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, state_root, worktree_root, merge_root = _ignored_runtime_repo(tmp_path)
    own_state = state_root / "lane-0"
    sibling_state = state_root / "lane-1"
    own_store = own_state / "dependency-preflight-artifacts"
    sibling_store = sibling_state / "dependency-preflight-artifacts"
    _seed_store_at(own_store, tmp_path / "own")
    _seed_store_at(sibling_store, tmp_path / "sibling")
    for directory in (own_state, sibling_state):
        directory.chmod(0o700)
    bindings = (
        {
            "slice_id": "slice-0",
            "lane_index": 0,
            "lane_id": "lane-0",
            "task_ids": ("LGCVF-080",),
        },
        {
            "slice_id": "slice-1",
            "lane_index": 1,
            "lane_id": "lane-1",
            "task_ids": ("LGCVF-070",),
        },
    )
    stable_read = multi_runner_module._read_stable_regular_bytes

    def reject_sibling_content_read(path: Path, *, max_bytes: int = 1_048_576):
        candidate = Path(path)
        if candidate.is_relative_to(sibling_state):
            pytest.fail("a sealed sibling artifact was opened for content evidence")
        return stable_read(candidate, max_bytes=max_bytes)

    monkeypatch.setattr(
        multi_runner_module,
        "_read_stable_regular_bytes",
        reject_sibling_content_read,
    )
    common = {
        "root": repo,
        "runtime_roots": (state_root, worktree_root, merge_root),
        "owner_bound_artifacts": (),
        "runtime_bindings": bindings,
        "slice_id": "slice-0",
        "lane_id": "lane-0",
        "state_dir": own_state,
        "state_prefix": "fixture_lane_0",
    }

    with (sibling_store / ".bounded-store.lock").open("rb") as sibling_lock:
        fcntl.flock(sibling_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = multi_runner_module._snapshot_plan_bound_recovery_artifacts(
            **common,
        )
    own_relative = own_store.relative_to(repo)
    assert evidence
    assert all(
        Path(str(item["path"])).is_relative_to(own_relative)
        for item in evidence
    )

    with (own_store / ".bounded-store.lock").open("rb") as own_lock:
        fcntl.flock(own_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="store lock is busy"):
            multi_runner_module._snapshot_plan_bound_recovery_artifacts(
                **common,
            )

    foreign_state = state_root / "lane-2"
    foreign_state.mkdir(mode=0o700)
    foreign = foreign_state / "fixture_lane_2_status.json"
    foreign.write_bytes(b"{}\n")
    foreign.chmod(0o600)
    with pytest.raises(ValueError, match="noncanonical runtime projection"):
        multi_runner_module._snapshot_plan_bound_recovery_artifacts(**common)


def test_reassigned_lane_snapshot_uses_supplied_state_scope(tmp_path: Path) -> None:
    repo, state_root, worktree_root, merge_root = _ignored_runtime_repo(tmp_path)
    lane_id = "recovery-1-0123456789ab"
    state_prefix = "recovery_1_0123456789ab"
    state_dir = state_root / lane_id
    store_root = state_dir / "dependency-preflight-artifacts"
    _seed_store_at(store_root, tmp_path / "reassigned")
    state_dir.chmod(0o700)
    status_path = state_dir / f"{state_prefix}_status.json"
    status_path.write_bytes(b"{}\n")
    status_path.chmod(0o600)
    binding = {
        "slice_id": "slice-0",
        "lane_index": 0,
        "lane_id": lane_id,
        "task_ids": ("LGCVF-080",),
    }
    common = {
        "directory_projection": False,
        "runtime_roots": (state_root, worktree_root, merge_root),
        "owner_bound_artifacts": (),
        "runtime_bindings": (binding,),
        "state_dir": state_dir,
        "state_prefix": state_prefix,
    }
    assert multi_runner_module._plan_bound_recovery_runtime_kind(
        store_root / "manifest.json",
        **common,
    ) == "dependency-preflight-store"
    assert multi_runner_module._plan_bound_recovery_runtime_kind(
        status_path,
        **common,
    ) == "file"

    evidence = multi_runner_module._snapshot_plan_bound_recovery_artifacts(
        root=repo,
        runtime_roots=(state_root, worktree_root, merge_root),
        owner_bound_artifacts=(),
        runtime_bindings=(binding,),
        slice_id="slice-0",
        lane_id=lane_id,
        state_dir=state_dir,
        state_prefix=state_prefix,
    )
    evidence_paths = {str(item["path"]) for item in evidence}
    assert status_path.relative_to(repo).as_posix() in evidence_paths
    assert (store_root / "manifest.json").relative_to(repo).as_posix() in (
        evidence_paths
    )

    with pytest.raises(ValueError, match="reassigned recovery prefix is mixed"):
        multi_runner_module._snapshot_plan_bound_recovery_artifacts(
            root=repo,
            runtime_roots=(state_root, worktree_root, merge_root),
            owner_bound_artifacts=(),
            runtime_bindings=(binding,),
            slice_id="slice-0",
            lane_id=lane_id,
            state_dir=state_dir,
            state_prefix="recovery_1_ffffffffffff",
        )


@pytest.mark.parametrize(
    ("relative_name", "expected"),
    (
        ("implementation-protected-path-incident.json", "portal-file"),
        ("database-portal-protected-path-recovery-intent.json", "portal-file"),
        ("database-portal-protected-path-recovery.json", "portal-file"),
        (
            "database-portal-external-protected-checkout-recovery.json",
            "portal-file",
        ),
        ("database-portal-inflight-process-recovery.json", "portal-file"),
        (
            "database-portal-validation-retry-seed-conflict-recovery.json",
            "portal-file",
        ),
        ("database-portal-pooled-worktree-create-recovery.json", "portal-file"),
        (
            "implementation-protected-path-auto-clearance-0123456789abcdef.json",
            "portal-file",
        ),
        ("portal-task-state.worktree-context.json", "portal-file"),
        ("submodule-merge-rollback-guardrail.json", "portal-file"),
        (
            "implementation-task-claim-release-receipts/"
            "canonical-task-0123456789abcdef01234567-a1.json",
            "portal-file",
        ),
        (
            "post-merge-declared-output-repair/"
            "lgcvf-080-attempt-1.log",
            "portal-file",
        ),
        (
            "post-merge-declared-output-requalification/" + "a" * 64 + ".json",
            "portal-file",
        ),
        (
            "implementation-logs/post-merge-declared-output-requalification/"
            "lgcvf-080-0123456789abcdef.log",
            "portal-file",
        ),
        (
            "provider_route_receipts/lgcvf-080-0123456789ab/"
            "provider-route-attempt-1.json",
            "portal-file",
        ),
        (
            "provider_route_receipts/lgcvf-080-0123456789ab/"
            "provider-route-1.json",
            "portal-file",
        ),
        (
            "provider_route_receipts/lgcvf-080-0123456789ab/"
            "provider-filesystem-boundary-attempt-1.json",
            "portal-file",
        ),
        (
            "provider_route_receipts/lgcvf-080-0123456789ab/"
            "provider-filesystem-boundary-1.json",
            "portal-file",
        ),
        ("implementation-protected-path-incident.json.bak", ""),
        ("portal-task-state.worktree-context.json.bak", ""),
        ("submodule-merge-rollback-guardrail.json.bak", ""),
        ("database-portal-protected-path-recovery-intent.json.bak", ""),
        (
            "implementation-protected-path-auto-clearance-0123456789abcde.json",
            "",
        ),
        (
            "implementation-protected-path-auto-clearance-0123456789abcdeg.json",
            "",
        ),
        (
            "implementation-task-claim-release-receipts/"
            "canonical-task-0123456789abcdef01234567-a0.json",
            "",
        ),
        (
            "post-merge-declared-output-repair/"
            "foreign-task-attempt-1.log",
            "",
        ),
        (
            "post-merge-declared-output-requalification/" + "a" * 63 + ".json",
            "",
        ),
        (
            "implementation-logs/post-merge-declared-output-requalification/"
            "foreign-task-0123456789abcdef.log",
            "",
        ),
        (
            "provider_route_receipts/foreign-task-0123456789ab/"
            "provider-route-attempt-1.json",
            "",
        ),
        (
            "provider_route_receipts/lgcvf-080-0123456789ab/"
            "provider-route-attempt-0.json",
            "",
        ),
    ),
)
def test_portal_recovery_receipt_names_are_exact(
    tmp_path: Path,
    relative_name: str,
    expected: str,
) -> None:
    state_root = tmp_path / "state"
    artifact = (
        state_root
        / "lane-0"
        / "lgcvf_lane_0_database_portal_attempts"
        / ("a" * 24)
        / relative_name
    )
    observed = multi_runner_module._plan_bound_portal_or_scan_runtime_kind(
        state_root=state_root,
        artifact=artifact,
        state_prefix="lgcvf_lane_0",
        directory_projection=False,
        binding={
            "lane_index": 0,
            "task_ids": ("LGCVF-080",),
        },
    )
    assert observed == expected

    if expected == "portal-file":
        assert multi_runner_module._plan_bound_portal_task_descendant_matches(
            Path(relative_name),
            task_name="lgcvf-080",
        )


@pytest.mark.parametrize(
    ("relative_name", "expected"),
    (
        (
            "train/post-merge-recovery-cursors/" + "a" * 64 + ".json",
            "external-authority",
        ),
        (
            "train/post-merge-recovery-cursors/" + "a" * 63 + ".json",
            "",
        ),
        (
            "train/post-merge-recovery-cursors/" + "g" * 64 + ".json",
            "",
        ),
        (
            "train/post-merge-recovery-cursors/" + "a" * 64 + ".json.bak",
            "",
        ),
    ),
)
def test_merge_recovery_cursor_names_are_exact(
    tmp_path: Path,
    relative_name: str,
    expected: str,
) -> None:
    state_root = tmp_path / "state"
    observed = multi_runner_module._plan_bound_recovery_runtime_kind(
        tmp_path / "merge-queue" / relative_name,
        directory_projection=False,
        runtime_roots=(
            state_root,
            tmp_path / "worktrees",
            tmp_path / "merge-queue",
        ),
        owner_bound_artifacts=(),
        runtime_bindings=(
            {
                "lane_index": 0,
                "lane_id": "lane-0",
                "task_ids": ("LGCVF-080",),
            },
        ),
        state_dir=state_root / "lane-0",
        state_prefix="lgcvf_lane_0",
    )
    assert observed == expected


@pytest.mark.parametrize(
    ("quota_name", "quota_value", "expected_error"),
    (
        ("DEFAULT_ARTIFACT_STORE_MAX_BLOBS", 0, "blob-count quota"),
        ("DEFAULT_ARTIFACT_STORE_MAX_BYTES", 1, "byte quota"),
    ),
)
def test_dependency_preflight_store_enforces_aggregate_quotas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quota_name: str,
    quota_value: int,
    expected_error: str,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    monkeypatch.setattr(artifact_store_module, quota_name, quota_value)

    with pytest.raises(ValueError, match=expected_error):
        _validate_store(state_root, store_root)


def test_dependency_preflight_store_enforces_per_blob_quota(
    tmp_path: Path,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    manifest_path = store_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = next(iter(manifest["blobs"].values()))
    metadata["size_bytes"] = (
        artifact_store_module.DEFAULT_ARTIFACT_BLOB_MAX_BYTES + 1
    )
    _reseal_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="blob exceeds its size quota"):
        _validate_store(state_root, store_root)


def test_compacted_store_accepts_evictions_and_absent_previous_blob(
    tmp_path: Path,
) -> None:
    state_root, store_root, blob_path = _seed_store(tmp_path)
    quotas = ArtifactQuotaPolicy(max_bytes=1, max_blob_bytes=1)
    store = BoundedArtifactStore(store_root, quotas=quotas)
    try:
        result = store.compact(max_items=1, force_quota=True)
        assert result.evicted == 1
        assert not blob_path.exists()
        previous = json.loads(
            (store_root / "manifest.previous.json").read_text(encoding="utf-8")
        )
        assert previous["blobs"]
        assert (store_root / "evictions.jsonl").is_file()

        evidence = _validate_store(state_root, store_root)
        assert any(item["path"].endswith("/evictions.jsonl") for item in evidence)
    finally:
        store.close()


def test_dependency_preflight_store_rejects_malformed_eviction_log(
    tmp_path: Path,
) -> None:
    state_root, store_root, _blob_path = _seed_store(tmp_path)
    eviction_log = store_root / "evictions.jsonl"
    eviction_log.write_bytes(b"{}\n")
    eviction_log.chmod(0o600)

    with pytest.raises(ValueError, match="eviction record is malformed"):
        _validate_store(state_root, store_root)
