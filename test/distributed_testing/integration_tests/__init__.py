"""Integration test package for distributed testing."""

from __future__ import annotations

import sys as _sys

# Allow integration tests to import sibling helpers via single-dot relative paths.
from .. import model_sharding as _model_sharding
from .. import resource_pool_bridge as _resource_pool_bridge

_sys.modules[__name__ + ".model_sharding"] = _model_sharding
_sys.modules[__name__ + ".resource_pool_bridge"] = _resource_pool_bridge
