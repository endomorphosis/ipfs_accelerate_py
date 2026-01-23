"""Compatibility package for test imports.

Some tests import `common.*` from repo root. The actual test helpers live under
`test/common`, so we extend this package's search path to include that directory.
"""

from __future__ import annotations

from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_test_common = _repo_root / "test" / "common"

if _test_common.is_dir():
    __path__.append(str(_test_common))  # type: ignore[name-defined]
