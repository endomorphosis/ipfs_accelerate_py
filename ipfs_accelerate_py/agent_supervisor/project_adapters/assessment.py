"""Typed project support matrix (EAAEF-043)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .base import GenericProjectAdapter, ProjectSupport
from .python import PythonProjectAdapter


ASSESSMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/project-support-assessment@1"
)
OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "preview_only",
        "unsupported_language",
        "unsupported_build_system",
        "unsafe_repository",
        "insufficient_validation",
        "human_configuration_required",
        "mutation_not_admitted",
        "supported_inventory",
    }
)


class AssessmentError(ValueError):
    """Support assessment is malformed."""


def assess_repository(root: Path | str) -> Mapping[str, Any]:
    support = GenericProjectAdapter().inspect(root)
    outcome = getattr(support, "outcome", None)
    if hasattr(outcome, "value"):
        outcome = outcome.value
    outcome = str(outcome or "preview_only")
    if outcome not in OUTCOMES:
        raise AssessmentError(f"unknown support outcome: {outcome}")
    python = PythonProjectAdapter()
    mutation = python.mutation_admitted(support)
    return MappingProxyType(
        {
            "schema": ASSESSMENT_SCHEMA,
            "outcome": outcome,
            "languages": list(getattr(support, "languages", ())),
            "mutation_admitted": bool(mutation),
        }
    )
