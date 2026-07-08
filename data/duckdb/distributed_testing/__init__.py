"""distributed_testing module for IPFS Accelerate.

This repo also vendors a much larger reference implementation under
`test/duckdb_api/distributed_testing`. The unit tests import modules like
`duckdb_api.distributed_testing.load_balancer`, so we extend the package path to
include that test tree when present.
"""

from __future__ import annotations

import os
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_repo_root = os.path.abspath(
	os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
_test_pkg = os.path.join(_repo_root, "test", "duckdb_api", "distributed_testing")
if os.path.isdir(_test_pkg):
	__path__.append(_test_pkg)  # type: ignore[attr-defined]
