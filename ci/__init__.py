"""Compatibility shim for legacy imports.

Some distributed testing modules use absolute imports like `from ci.api_interface ...`.
The reference CI implementation lives under `test/distributed_testing/ci`.

This package extends its module search path to include that directory so those
imports keep working during tests.
"""

from __future__ import annotations

import os
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_test_ci_pkg = os.path.join(_repo_root, "test", "distributed_testing", "ci")
if os.path.isdir(_test_ci_pkg):
    __path__.append(_test_ci_pkg)  # type: ignore[attr-defined]
