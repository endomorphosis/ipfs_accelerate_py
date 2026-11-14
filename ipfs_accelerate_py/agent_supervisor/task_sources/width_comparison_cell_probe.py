"""Measure the three genuine width comparison cells.

Each probe calls the published producer and returns None unless that result
equals the locked dict. A match is cached. A miss is not. The cell modules
import torch inside the call; this probe does not treat that import as a
remote fetch or a training run.
"""

from __future__ import annotations

from .dependency_block_resolution import (
    EXPECTED_NATIVE384_PROBE,
    EXPECTED_NATIVE768_PROBE,
    EXPECTED_SPARSE8_PROBE,
)

_cached_sparse8: dict[str, object] | None = None
_cached_native384: dict[str, object] | None = None
_cached_native768: dict[str, object] | None = None


def measure_sparse8_comparison_probe() -> dict[str, object] | None:
    """Return the locked canonical 8D sparse projection, or None."""

    global _cached_sparse8
    if _cached_sparse8 is not None:
        return dict(_cached_sparse8)
    try:
        from ipfs_datasets_py.logic.autoformal.legal_width_comparison_cells_v1 import (
            run_sparse8_cell,
        )

        measured = run_sparse8_cell()
    except Exception:
        return None
    if measured != EXPECTED_SPARSE8_PROBE:
        return None
    _cached_sparse8 = measured
    return dict(_cached_sparse8)


def measure_native384_comparison_probe() -> dict[str, object] | None:
    """Return the locked pinned 384D settings receipt, or None."""

    global _cached_native384
    if _cached_native384 is not None:
        return dict(_cached_native384)
    try:
        from ipfs_datasets_py.logic.autoformal.legal_width_comparison_cells_v1 import (
            run_native384_cell,
        )

        measured = run_native384_cell()
    except Exception:
        return None
    if measured != EXPECTED_NATIVE384_PROBE:
        return None
    _cached_native384 = measured
    return dict(_cached_native384)


def measure_native768_comparison_probe() -> dict[str, object] | None:
    """Return the locked complete 768D encoder identity, or None."""

    global _cached_native768
    if _cached_native768 is not None:
        return dict(_cached_native768)
    try:
        from ipfs_datasets_py.logic.autoformal.legal_width_comparison_cells_v1 import (
            run_native768_cell,
        )

        measured = run_native768_cell()
    except Exception:
        return None
    if measured != EXPECTED_NATIVE768_PROBE:
        return None
    _cached_native768 = measured
    return dict(_cached_native768)
