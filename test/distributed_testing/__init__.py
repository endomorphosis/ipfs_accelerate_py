"""
Distributed Testing Framework

This package provides functionality for distributed execution and testing
of models across heterogeneous environments.

Major components:
- CircuitBreaker: Prevents cascading failures
- StateManager: Manages distributed state
- WorkerRegistry: Manages worker registration and health
- TransactionLog: Logs operations for recovery
- PluginSystem: Extensible plugin architecture
"""

from __future__ import annotations

__version__ = "1.0.0"

import sys as _sys

# Ensure the test package is also visible as the top-level `distributed_testing`
# module so patching paths in tests resolve to the same module objects.
_sys.modules["distributed_testing"] = _sys.modules[__name__]

try:
	from . import browser_recovery_strategies as _browser_recovery_strategies
	_sys.modules["distributed_testing.browser_recovery_strategies"] = _browser_recovery_strategies
except Exception:
	pass