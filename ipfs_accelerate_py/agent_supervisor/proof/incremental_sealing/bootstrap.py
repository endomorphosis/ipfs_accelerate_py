"""Hermetic IncrementalProofSealer bootstrap (IPS-044).

Explicit dependency injection only.  Importing this module creates no files,
keys, subprocesses, installs, network access, or daemon handles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

BOOTSTRAP_EVIDENCE: Final[str] = "ips/import-hermeticity@1"
BOOTSTRAP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "bootstrap@1"
)


class BootstrapError(ValueError):
    """Fail-closed bootstrap contract violation."""


@dataclass(frozen=True, slots=True)
class IncrementalSealingBootstrap:
    """Bound adapters for one process.  All collaborators are injected."""

    schema: str = BOOTSTRAP_SCHEMA
    evidence_subset: str = BOOTSTRAP_EVIDENCE
    datasets_classify: Callable[[Mapping[str, Any]], Any] | None = None
    kit_cache_lookup: Callable[[str], Any] | None = None
    admit: Callable[[Any], Any] | None = None

    def has_datasets_adapter(self) -> bool:
        return self.datasets_classify is not None

    def has_kit_adapter(self) -> bool:
        return self.kit_cache_lookup is not None

    def has_admission(self) -> bool:
        return self.admit is not None

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "datasets_adapter": self.has_datasets_adapter(),
            "kit_adapter": self.has_kit_adapter(),
            "admission": self.has_admission(),
            "import_side_effects": False,
        }


def bind_bootstrap(
    *,
    datasets_classify: Callable[[Mapping[str, Any]], Any] | None = None,
    kit_cache_lookup: Callable[[str], Any] | None = None,
    admit: Callable[[Any], Any] | None = None,
) -> IncrementalSealingBootstrap:
    """Return a bootstrap bound only to the provided callables."""

    return IncrementalSealingBootstrap(
        datasets_classify=datasets_classify,
        kit_cache_lookup=kit_cache_lookup,
        admit=admit,
    )


__all__ = (
    "BOOTSTRAP_EVIDENCE",
    "BOOTSTRAP_SCHEMA",
    "BootstrapError",
    "IncrementalSealingBootstrap",
    "bind_bootstrap",
)
