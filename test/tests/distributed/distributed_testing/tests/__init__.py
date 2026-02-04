"""
Test modules for the Distributed Testing Framework

This package contains various test modules for different components
of the Distributed Testing Framework, including:

- CI/CD system integrations
- Fault tolerance and recovery
- Coordinator redundancy
"""

from __future__ import annotations

import sys as _sys

# Provide package-level aliases so tests can use legacy single-dot relative imports
# (e.g. from test.tests.distributed.distributed_testing.ci import ...) while the implementation lives one level up.
from test.tests.distributed.distributed_testing import ci as _ci
from test.tests.distributed.distributed_testing import coordinator as _coordinator
from test.tests.distributed.distributed_testing import dynamic_resource_manager as _dynamic_resource_manager
from test.tests.distributed.distributed_testing import integration_mode as _integration_mode
from test.tests.distributed.distributed_testing import performance_trend_analyzer as _performance_trend_analyzer
from test.tests.distributed.distributed_testing import worker as _worker

_sys.modules[__name__ + ".ci"] = _ci
_sys.modules[__name__ + ".coordinator"] = _coordinator
_sys.modules[__name__ + ".dynamic_resource_manager"] = _dynamic_resource_manager
_sys.modules[__name__ + ".integration_mode"] = _integration_mode
_sys.modules[__name__ + ".performance_trend_analyzer"] = _performance_trend_analyzer
_sys.modules[__name__ + ".worker"] = _worker
_sys.modules["worker"] = _worker