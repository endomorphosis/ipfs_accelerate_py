"""Compatibility package for result aggregator modules.

Some tests import `result_aggregator.*` but the implementation currently lives
under `test/distributed_testing/result_aggregator`.
"""

from __future__ import annotations

from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_impl_dir = _repo_root / "test" / "distributed_testing" / "result_aggregator"

if _impl_dir.is_dir():
    __path__.append(str(_impl_dir))  # type: ignore[name-defined]
