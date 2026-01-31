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

warnings.filterwarnings(
    "ignore",
    message=r"Can't initialize NVML",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*result_aggregator\.log.*",
    category=ResourceWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*result_aggregator_integration\.log.*",
    category=ResourceWarning,
)


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
    warnings.filterwarnings(
        "ignore",
        message=r"Can't initialize NVML",
        category=UserWarning,
    )


def _close_result_aggregator_file_handlers() -> None:
    import logging

    targets = {"result_aggregator.log", "result_aggregator_integration.log"}

    def _close_from_logger(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                filename = os.path.basename(getattr(handler, "baseFilename", ""))
                if filename in targets:
                    handler.close()
                    logger.removeHandler(handler)

    root_logger = logging.getLogger()
    _close_from_logger(root_logger)

    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            _close_from_logger(logger)


def pytest_sessionfinish(session, exitstatus):
    _close_result_aggregator_file_handlers()
