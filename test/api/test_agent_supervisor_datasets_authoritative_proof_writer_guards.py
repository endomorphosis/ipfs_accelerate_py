"""Fail-closed tests for legacy proof and symbolic semantic writers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.semantic_truth_authority import (
    DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    DATASETS_SEMANTIC_TRUTH_AUTHORITY,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    STATE_SCHEMA_REVISION_ENV,
    AcceleratorSemanticTruthWriterProhibitedError,
)
from ipfs_accelerate_py.agent_supervisor.planning.database_symbolic_planning import (
    DatabaseSymbolicPlanner,
    PlanDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.database_evidence_store import (
    DatabaseEvidenceStore,
    EvidenceKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.database_repair_evidence import (
    DatabaseRepairEvidenceStore,
    FixedPointStatus,
)

WriterFactory = Callable[[Path], Any]


@pytest.fixture(
    params=(
        DatabaseEvidenceStore,
        DatabaseRepairEvidenceStore,
        DatabaseSymbolicPlanner,
    )
)
def writer_factory(request: pytest.FixtureRequest) -> WriterFactory:
    return request.param


def _clear_authority_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STATE_SCHEMA_REVISION_ENV, raising=False)
    monkeypatch.delenv(SEMANTIC_TRUTH_AUTHORITY_ENV, raising=False)


def test_datasets_schema_profile_refuses_side_store_before_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer_factory: WriterFactory,
) -> None:
    database = tmp_path / "not-created" / "semantic-writer.duckdb"
    _clear_authority_markers(monkeypatch)
    monkeypatch.setenv(
        STATE_SCHEMA_REVISION_ENV,
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    )

    with pytest.raises(
        AcceleratorSemanticTruthWriterProhibitedError,
        match="ipfs_datasets_py is the sole",
    ):
        writer_factory(database)

    assert not database.exists()
    assert not database.parent.exists()


def test_datasets_authority_marker_refuses_side_store_before_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer_factory: WriterFactory,
) -> None:
    database = tmp_path / "not-created" / "semantic-writer.duckdb"
    _clear_authority_markers(monkeypatch)
    monkeypatch.setenv(
        SEMANTIC_TRUTH_AUTHORITY_ENV,
        DATASETS_SEMANTIC_TRUTH_AUTHORITY,
    )

    with pytest.raises(AcceleratorSemanticTruthWriterProhibitedError):
        writer_factory(database)

    assert not database.exists()
    assert not database.parent.exists()


def test_open_rechecks_authority_before_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer_factory: WriterFactory,
) -> None:
    database = tmp_path / "not-created" / "semantic-writer.duckdb"
    _clear_authority_markers(monkeypatch)
    writer = writer_factory(database)
    monkeypatch.setenv(
        STATE_SCHEMA_REVISION_ENV,
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    )

    with pytest.raises(AcceleratorSemanticTruthWriterProhibitedError):
        writer.open()

    assert not database.exists()
    assert not database.parent.exists()


def test_pure_contract_values_remain_available_under_datasets_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_authority_markers(monkeypatch)
    monkeypatch.setenv(
        STATE_SCHEMA_REVISION_ENV,
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    )

    assert EvidenceKind.PROOF.value == "proof"
    assert PlanDisposition.ADMITTED.value == "admitted"
    assert FixedPointStatus.REACHED.value == "reached"


def test_default_profile_preserves_existing_store_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer_factory: WriterFactory,
) -> None:
    database = tmp_path / "legacy-profile" / "side-store.duckdb"
    _clear_authority_markers(monkeypatch)

    with writer_factory(database) as store:
        assert store.is_open

    assert database.is_file()

