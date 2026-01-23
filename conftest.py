import os
import sys
from pathlib import Path


def pytest_configure() -> None:
    # Preserve previous intent of pytest-env without requiring the plugin.
    os.environ.setdefault("TEST_MODE", "development")
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Preserve previous intent of PYTHONPATH=.
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
