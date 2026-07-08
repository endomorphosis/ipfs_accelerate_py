"""Test-suite shim for performance trend analyzer.

The real implementation lives at `test/distributed_testing/performance_trend_analyzer.py`.
That module requires optional data-science dependencies (notably `pandas`).

This shim makes imports in the unit test suite predictable:
- If optional deps are missing, importing this module will skip the dependent tests.
- If deps are present, it re-exports the implementation symbols.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from ..performance_trend_analyzer import *  # noqa: F401,F403
