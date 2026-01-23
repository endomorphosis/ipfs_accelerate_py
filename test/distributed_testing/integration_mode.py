"""Integration-test mode flags.

We distinguish between:
- Simulated integration tests: use mocks/offline stubs and should be safe in CI.
- Real integration tests: may require external services, network, or hardware.

Env vars:
- IPFS_ACCEL_RUN_INTEGRATION_TESTS_SIMULATED=1  -> enable simulated integration tests
- IPFS_ACCEL_RUN_INTEGRATION_TESTS_REAL=1       -> enable real integration tests

Back-compat:
- IPFS_ACCEL_RUN_INTEGRATION_TESTS=1 is treated as SIMULATED.

Note: these helpers are imported at test module import time, so they must rely
on environment variables (not pytest options set later).
"""

from __future__ import annotations

import os
from typing import Literal


IntegrationMode = Literal["simulated", "real"]


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_integration_mode() -> IntegrationMode | None:
    """Return the enabled integration mode, if any."""
    if _env_truthy(os.environ.get("IPFS_ACCEL_RUN_INTEGRATION_TESTS_REAL")):
        return "real"

    if _env_truthy(os.environ.get("IPFS_ACCEL_RUN_INTEGRATION_TESTS_SIMULATED")):
        return "simulated"

    # Back-compat: old flag implies simulated.
    if os.environ.get("IPFS_ACCEL_RUN_INTEGRATION_TESTS") == "1":
        return "simulated"

    return None


def integration_enabled() -> bool:
    return get_integration_mode() is not None


def simulated_integration_enabled() -> bool:
    return get_integration_mode() == "simulated"


def real_integration_enabled() -> bool:
    return get_integration_mode() == "real"


def integration_opt_in_message() -> str:
    return (
        "Integration tests are opt-in; set IPFS_ACCEL_RUN_INTEGRATION_TESTS_SIMULATED=1 "
        "(simulated) or IPFS_ACCEL_RUN_INTEGRATION_TESTS_REAL=1 (real)."
    )
