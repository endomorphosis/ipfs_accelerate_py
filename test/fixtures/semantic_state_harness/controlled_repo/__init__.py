"""Controlled Python fixture repository for the semantic-compression harness.

Interface: ``ControlledSemanticRepository@1``

This package describes a small, deterministic target tree and mutation matrix.
Loading the fixture API never imports or executes target-tree modules; target
sources exist only as path->bytes recipes that materializers write to disk.

SCH-014 / sch/fixture@1
"""

from __future__ import annotations

from .controlled_repository import (
    CONTROLLED_REPO_INTERFACE,
    CONTROLLED_REPO_SCHEMA,
    CORPUS_ID,
    ControlledSemanticRepository,
)
from .mutation_case import (
    ChangedSymbolOracle,
    ConfidenceOracle,
    FixtureOracle,
    InvalidationOracle,
    MerkleOracle,
    MutationCase,
    PathOperation,
    ReceiptFreshnessOracle,
)

__all__ = [
    "CONTROLLED_REPO_INTERFACE",
    "CONTROLLED_REPO_SCHEMA",
    "CORPUS_ID",
    "ChangedSymbolOracle",
    "ConfidenceOracle",
    "ControlledSemanticRepository",
    "FixtureOracle",
    "InvalidationOracle",
    "MerkleOracle",
    "MutationCase",
    "PathOperation",
    "ReceiptFreshnessOracle",
]
