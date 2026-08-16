"""Fail-closed boundary for accelerator-local semantic-truth writers.

The database program already propagates its sealed schema revision through the
process environment.  When that revision selects the datasets-authoritative
operational profile, legacy accelerator AST, mutation, repository-index and
impact stores must not open.  They are compatibility implementations for other
profiles, not fallbacks for canonical ``ipfs_datasets_py`` semantic artifacts.

This lower-level module performs no I/O.  Callers invoke the assertion before
creating directories, database files, or DDL.
"""

from __future__ import annotations

import os
from typing import Final

STATE_SCHEMA_REVISION_ENV: Final[str] = (
    "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
)
SEMANTIC_TRUTH_AUTHORITY_ENV: Final[str] = (
    "IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY"
)
DATASETS_SEMANTIC_TRUTH_AUTHORITY: Final[str] = "ipfs_datasets_py"
DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION: Final[str] = (
    "datasets-authoritative-operational-v1"
)


class AcceleratorSemanticTruthWriterProhibitedError(RuntimeError):
    """An accelerator-local semantic writer was selected under datasets authority."""


def assert_accelerator_semantic_writer_permitted(*, writer: str) -> None:
    """Refuse a local semantic writer under the sealed datasets profile."""

    revision = str(os.environ.get(STATE_SCHEMA_REVISION_ENV) or "").strip()
    authority = str(os.environ.get(SEMANTIC_TRUTH_AUTHORITY_ENV) or "").strip()
    if (
        revision == DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION
        or authority == DATASETS_SEMANTIC_TRUTH_AUTHORITY
    ):
        raise AcceleratorSemanticTruthWriterProhibitedError(
            f"{writer} is prohibited by datasets semantic authority "
            f"(schema_revision={revision!r}, authority={authority!r}): "
            "ipfs_datasets_py is the sole AST, semantic dependency, mutation "
            "impact, proof-obligation, and counterexample authority; consume "
            "verified datasets CIDs instead of opening a local writer"
        )


__all__ = [
    "AcceleratorSemanticTruthWriterProhibitedError",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION",
    "DATASETS_SEMANTIC_TRUTH_AUTHORITY",
    "SEMANTIC_TRUTH_AUTHORITY_ENV",
    "STATE_SCHEMA_REVISION_ENV",
    "assert_accelerator_semantic_writer_permitted",
]
