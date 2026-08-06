"""Tests for WorkerEvidenceFactory@1 / WorkerEvidenceView@1 (WPD-012)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.worker_evidence_factory import (
    DEFAULT_REQUIRED_QUERIES,
    QueryCoverageStatus,
    WORKER_EVIDENCE_FACTORY_EVIDENCE,
    WORKER_EVIDENCE_FACTORY_INTERFACE,
    WORKER_EVIDENCE_VIEW_INTERFACE,
    WorkerEvidenceFactory,
    WorkerEvidencePathEscapeError,
    WorkerEvidenceQueryKind,
    build_single_repository_forest,
    build_worker_evidence_factory,
    build_worker_evidence_view,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Evidence Factory")
    _git(repository, "config", "user.email", "evidence@example.invalid")
    _write(
        repository / "src" / "service.py",
        "class Service:\n    def dispatch(self, request):\n        return request\n",
    )
    _write(repository / "README.md", "fixture\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    # Dirty overlay: modified tracked file + untracked source.
    _write(
        repository / "src" / "service.py",
        "class Service:\n    def dispatch(self, request):\n        return transform(request)\n",
    )
    _write(repository / "src" / "extra.py", "def extra():\n    return 1\n")
    return repository


def test_interfaces_and_evidence_key_are_stable() -> None:
    assert WORKER_EVIDENCE_FACTORY_INTERFACE == "WorkerEvidenceFactory@1"
    assert WORKER_EVIDENCE_VIEW_INTERFACE == "WorkerEvidenceView@1"
    assert WORKER_EVIDENCE_FACTORY_EVIDENCE == "wpd/evidence-factory@1"
    assert {item.value for item in DEFAULT_REQUIRED_QUERIES} == {
        "forest_binding",
        "dirty_overlay",
        "graph_index",
    }


def test_evidence_view_is_content_addressed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = build_worker_evidence_factory()
    graph_cid = content_identity({"fixture": "graph"})
    index_cid = content_identity({"fixture": "index"})

    first = factory.build(
        repository,
        graph_cid=graph_cid,
        index_cid=index_cid,
        paths=("src/service.py",),
    )
    second = factory.build(
        repository,
        graph_cid=graph_cid,
        index_cid=index_cid,
        paths=("src/service.py",),
    )

    assert first.view_cid
    assert first.view_cid == second.view_cid
    assert first.view_cid == content_identity(first.to_dict())
    assert first.forest_binding.repository_forest_cid
    assert first.forest_binding.dirty_overlay_cid
    assert first.forest_binding.dirty is True
    assert first.coverage_complete is True
    assert first.graph_cid == graph_cid
    assert first.index_cid == index_cid
    assert "src/service.py" in first.admitted_paths
    # Body-free durable projection.
    payload = first.to_dict()
    blob = str(payload).casefold()
    assert "source_text" not in blob
    assert "source_body" not in blob
    assert "class Service" not in blob


def test_content_identity_changes_with_dirty_overlay(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = WorkerEvidenceFactory()
    graph_cid = content_identity({"fixture": "graph-a"})
    index_cid = content_identity({"fixture": "index-a"})

    dirty_view = factory.build(
        repository,
        graph_cid=graph_cid,
        index_cid=index_cid,
    )
    # Restore clean tree so overlay digest changes.
    _git(repository, "checkout", "--", "src/service.py")
    (repository / "src" / "extra.py").unlink()
    clean_view = factory.build(
        repository,
        graph_cid=graph_cid,
        index_cid=index_cid,
    )

    assert dirty_view.forest_binding.dirty is True
    assert clean_view.forest_binding.dirty is False
    assert (
        dirty_view.forest_binding.dirty_overlay_cid
        != clean_view.forest_binding.dirty_overlay_cid
    )
    assert dirty_view.view_cid != clean_view.view_cid


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = build_worker_evidence_factory()
    graph_cid = content_identity({"fixture": "graph"})
    index_cid = content_identity({"fixture": "index"})

    with pytest.raises(WorkerEvidencePathEscapeError, match="escape"):
        factory.build(
            repository,
            graph_cid=graph_cid,
            index_cid=index_cid,
            paths=("../outside.py",),
        )

    with pytest.raises(WorkerEvidencePathEscapeError, match="escape"):
        factory.build(
            repository,
            graph_cid=graph_cid,
            index_cid=index_cid,
            paths=("/etc/passwd",),
        )

    with pytest.raises(WorkerEvidencePathEscapeError, match="escape"):
        factory.resolve_path("src/../../etc/passwd", checkout_root=repository)


def test_incomplete_required_queries_mark_coverage_false_without_inventing_facts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    factory = build_worker_evidence_factory()

    # Default portfolio requires graph_index; without CIDs coverage is incomplete.
    view = factory.build(repository)

    assert view.coverage_complete is False
    assert view.graph_cid == ""
    assert view.index_cid == ""
    assert "graph_index" in view.query_coverage.incomplete_kinds
    assert "forest_binding" in view.query_coverage.satisfied_kinds
    assert "dirty_overlay" in view.query_coverage.satisfied_kinds

    graph_record = next(
        item
        for item in view.query_coverage.records
        if item.kind is WorkerEvidenceQueryKind.GRAPH_INDEX
    )
    assert graph_record.status is QueryCoverageStatus.INCOMPLETE
    assert graph_record.result_cid == ""
    assert "missing_" in graph_record.reason_code
    # Incomplete coverage must not invent a synthetic graph/index identity.
    assert not any(
        token in (view.graph_cid + view.index_cid)
        for token in ("invented", "synthetic", "placeholder")
    )


def test_partial_graph_or_index_does_not_satisfy_required_pair(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = build_worker_evidence_factory()
    only_graph = content_identity({"fixture": "graph-only"})

    view = factory.build(repository, graph_cid=only_graph)

    assert view.coverage_complete is False
    # Partial claim is cleared so incomplete coverage cannot look complete.
    assert view.graph_cid == ""
    assert view.index_cid == ""
    graph_record = next(
        item
        for item in view.query_coverage.records
        if item.kind is WorkerEvidenceQueryKind.GRAPH_INDEX
    )
    assert graph_record.status is QueryCoverageStatus.INCOMPLETE
    assert "index_cid" in graph_record.reason_code


def test_explicit_graph_and_index_complete_coverage(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    graph_cid = content_identity({"fixture": "graph-complete"})
    index_cid = content_identity({"fixture": "index-complete"})
    view = build_worker_evidence_view(
        repository,
        graph_cid=graph_cid,
        index_cid=index_cid,
    )

    assert view.coverage_complete is True
    assert view.graph_cid == graph_cid
    assert view.index_cid == index_cid
    assert view.query_coverage.incomplete_kinds == ()
    assert set(view.query_coverage.satisfied_kinds) == {
        "forest_binding",
        "dirty_overlay",
        "graph_index",
    }


def test_adapter_providers_bind_identities_without_source_bodies(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    graph_cid = content_identity({"fixture": "provider-graph"})
    index_cid = content_identity({"fixture": "provider-index"})
    doctor_cid = content_identity({"fixture": "provider-doctor"})

    factory = WorkerEvidenceFactory(
        graph_provider=lambda _forest, _desc: {"graph_cid": graph_cid},
        index_provider=lambda _forest, _desc: type(
            "Index", (), {"index_id": index_cid}
        )(),
        doctor_snapshot_provider=lambda _forest, _desc: {
            "snapshot_cid": doctor_cid
        },
        default_required_queries=(
            WorkerEvidenceQueryKind.FOREST_BINDING,
            WorkerEvidenceQueryKind.DIRTY_OVERLAY,
            WorkerEvidenceQueryKind.GRAPH_INDEX,
            WorkerEvidenceQueryKind.DOCTOR_SNAPSHOT,
        ),
    )
    view = factory.build(repository)

    assert view.coverage_complete is True
    assert view.graph_cid == graph_cid
    assert view.index_cid == index_cid
    assert view.doctor_snapshot_cid == doctor_cid
    assert factory.last_view is view


def test_failed_provider_does_not_invent_facts(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    def _boom(_forest, _desc):  # noqa: ANN001
        raise RuntimeError("adapter offline")

    factory = WorkerEvidenceFactory(
        graph_provider=_boom,
        index_provider=_boom,
    )
    view = factory.build(repository)

    assert view.coverage_complete is False
    assert view.graph_cid == ""
    assert view.index_cid == ""


def test_prebuilt_forest_binding_is_honored(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    forest = build_single_repository_forest(repository, alias="worker")
    graph_cid = content_identity({"fixture": "forest-graph"})
    index_cid = content_identity({"fixture": "forest-index"})

    view = WorkerEvidenceFactory().build(
        forest=forest,
        graph_cid=graph_cid,
        index_cid=index_cid,
    )

    assert view.repository_forest_cid == forest.forest_id
    assert view.forest_binding.git_tree_id == forest.write_descriptor().tree
    roots = view.to_implementation_forest_roots()
    assert roots["repository_forest_cid"] == forest.forest_id
    assert roots["git_tree_id"] == forest.write_descriptor().tree
    assert roots["dirty_overlay_cid"] == view.dirty_overlay_cid


def test_required_path_scope_incomplete_when_no_paths(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = WorkerEvidenceFactory(
        default_required_queries=(
            WorkerEvidenceQueryKind.FOREST_BINDING,
            WorkerEvidenceQueryKind.DIRTY_OVERLAY,
            WorkerEvidenceQueryKind.PATH_SCOPE,
        )
    )
    view = factory.build(repository)

    assert view.coverage_complete is False
    assert "path_scope" in view.query_coverage.incomplete_kinds
    path_record = next(
        item
        for item in view.query_coverage.records
        if item.kind is WorkerEvidenceQueryKind.PATH_SCOPE
    )
    assert path_record.status is QueryCoverageStatus.INCOMPLETE
    assert path_record.result_cid == ""
