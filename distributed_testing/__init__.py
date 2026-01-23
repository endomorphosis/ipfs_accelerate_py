"""Compatibility package for distributed testing modules.

The code under `test/distributed_testing` is imported by tests as
`distributed_testing.*`. Expose it as a top-level package by extending this
package's path.
"""

from __future__ import annotations

from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_impl_dir = _repo_root / "test" / "distributed_testing"

if _impl_dir.is_dir():
    __path__.append(str(_impl_dir))  # type: ignore[name-defined]
