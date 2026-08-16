from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.database_impact_graph import (
    DatabaseImpactGraph,
)
from ipfs_accelerate_py.agent_supervisor.analysis.database_repository_indexer import (
    DatabaseRepositoryIndexer,
)
from ipfs_accelerate_py.agent_supervisor.analysis.duckdb_ast_index import DuckDBASTIndex
from ipfs_accelerate_py.agent_supervisor.analysis.mutation_ledger import MutationLedger
from ipfs_accelerate_py.agent_supervisor.analysis.semantic_truth_authority import (
    DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    DATASETS_SEMANTIC_TRUTH_AUTHORITY,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    STATE_SCHEMA_REVISION_ENV,
    AcceleratorSemanticTruthWriterProhibitedError,
)

WriterFactory = Callable[[Path], Any]


@pytest.fixture(params=(DuckDBASTIndex, DatabaseRepositoryIndexer, MutationLedger, DatabaseImpactGraph))
def writer_factory(request: pytest.FixtureRequest) -> WriterFactory:
    return request.param


def test_datasets_authority_refuses_writer_before_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer_factory: WriterFactory,
) -> None:
    database = tmp_path / "not-created" / "semantic-writer.duckdb"
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


def test_open_rechecks_authority_before_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer_factory: WriterFactory,
) -> None:
    database = tmp_path / "not-created" / "semantic-writer.duckdb"
    monkeypatch.delenv(STATE_SCHEMA_REVISION_ENV, raising=False)
    writer = writer_factory(database)
    monkeypatch.setenv(
        STATE_SCHEMA_REVISION_ENV,
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION,
    )

    with pytest.raises(AcceleratorSemanticTruthWriterProhibitedError):
        writer.open()

    assert not database.exists()
    assert not database.parent.exists()


def test_task_scoped_provider_authority_marker_refuses_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writer_factory: WriterFactory,
) -> None:
    database = tmp_path / "not-created" / "semantic-writer.duckdb"
    monkeypatch.delenv(STATE_SCHEMA_REVISION_ENV, raising=False)
    monkeypatch.setenv(
        SEMANTIC_TRUTH_AUTHORITY_ENV,
        DATASETS_SEMANTIC_TRUTH_AUTHORITY,
    )

    with pytest.raises(AcceleratorSemanticTruthWriterProhibitedError):
        writer_factory(database)

    assert not database.exists()
    assert not database.parent.exists()
