"""Partitioned fixture repositories for the Semantic Compression Governor.

Interface: ``SemanticGovernorFixtureCorpus@1``
Evidence: ``scg/partitioned-corpus@1``
Task: SCG-040

This package describes a deterministic base tree and calibration / development /
held-out case matrix. Loading the fixture API never imports or executes target
tree modules; target sources exist only as path->bytes recipes that
materialisers write to disk.
"""

from __future__ import annotations

from .case_record import (
    ADVERSARIAL_SCENARIOS,
    PARTITIONS,
    TASK_FAMILIES,
    FixtureCase,
    FixtureCorpusError,
    OmissionOracle,
    OutcomeOracle,
    PathOperation,
    ScannerView,
)
from .corpus import (
    CORPUS_ID,
    EVIDENCE_ID,
    FIXTURE_CORPUS_INTERFACE,
    FIXTURE_CORPUS_SCHEMA,
    TASK_ID,
    SemanticGovernorFixtureCorpus,
    apply_operations,
    changed_paths,
    content_digest,
    read_tree_bytes,
    tree_digest,
    write_tree,
)

__all__ = [
    "ADVERSARIAL_SCENARIOS",
    "CORPUS_ID",
    "EVIDENCE_ID",
    "FIXTURE_CORPUS_INTERFACE",
    "FIXTURE_CORPUS_SCHEMA",
    "PARTITIONS",
    "TASK_FAMILIES",
    "TASK_ID",
    "FixtureCase",
    "FixtureCorpusError",
    "OmissionOracle",
    "OutcomeOracle",
    "PathOperation",
    "ScannerView",
    "SemanticGovernorFixtureCorpus",
    "apply_operations",
    "changed_paths",
    "content_digest",
    "read_tree_bytes",
    "tree_digest",
    "write_tree",
]
