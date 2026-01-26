import os
import sys
import warnings
from pathlib import Path


def pytest_configure() -> None:
    # Preserve previous intent of pytest-env without requiring the plugin.
    os.environ.setdefault("TEST_MODE", "development")
    os.environ.setdefault("MPLBACKEND", "Agg")

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
