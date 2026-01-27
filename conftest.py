import os
import sys
import warnings
from pathlib import Path

# Ensure integration/long tests are enabled before module imports during collection.
os.environ.setdefault("TEST_MODE", "development")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("IPFS_ACCEL_RUN_INTEGRATION_TESTS_SIMULATED", "1")
os.environ.setdefault("IPFS_ACCEL_RUN_INTEGRATION_TESTS", "1")
os.environ.setdefault("RUN_LONG_TESTS", "1")


def pytest_configure() -> None:
    # Preserve previous intent of PYTHONPATH=.
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Suppress known third-party deprecation warnings during tests
    warnings.filterwarnings(
        "ignore",
        message=r"websockets\.legacy is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"websockets\.server\.WebSocketServerProtocol is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"websockets\.client\.WebSocketClientProtocol is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPy.* has no __module__ attribute",
        category=DeprecationWarning,
    )
