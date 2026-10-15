"""Shared execution budget for fresh benchmark configurations and receipts."""

DEFAULT_AGENT_TIMEOUT_SECONDS = 600
MIN_AGENT_TIMEOUT_SECONDS = 120
MAX_AGENT_TIMEOUT_SECONDS = 600
MIN_DRIVER_TIMEOUT_SECONDS = 90
MAX_DRIVER_TIMEOUT_SECONDS = 600
DEFAULT_DRIVER_TIMEOUT_SECONDS = DEFAULT_AGENT_TIMEOUT_SECONDS - 30
LEGACY_AGENT_TIMEOUT_SECONDS = 300


def validate_agent_timeout(seconds: int) -> int:
    if type(seconds) is not int or not MIN_AGENT_TIMEOUT_SECONDS <= seconds <= MAX_AGENT_TIMEOUT_SECONDS:
        raise ValueError("agent timeout must be an integer from 120 to 600 seconds")
    return seconds


def validate_driver_timeout(seconds: int) -> int:
    if type(seconds) is not int or not MIN_DRIVER_TIMEOUT_SECONDS <= seconds <= MAX_DRIVER_TIMEOUT_SECONDS:
        raise ValueError("driver timeout must be an integer from 90 to 600 seconds")
    return seconds
