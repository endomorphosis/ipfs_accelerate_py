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
# (e.g. from .ci import ...) while the implementation lives one level up.
from .. import ci as _ci
from .. import coordinator as _coordinator
from .. import dynamic_resource_manager as _dynamic_resource_manager
from .. import integration_mode as _integration_mode
from .. import performance_trend_analyzer as _performance_trend_analyzer
from .. import worker as _worker

_sys.modules[__name__ + ".ci"] = _ci
_sys.modules[__name__ + ".coordinator"] = _coordinator
_sys.modules[__name__ + ".dynamic_resource_manager"] = _dynamic_resource_manager
_sys.modules[__name__ + ".integration_mode"] = _integration_mode
_sys.modules[__name__ + ".performance_trend_analyzer"] = _performance_trend_analyzer
_sys.modules[__name__ + ".worker"] = _worker