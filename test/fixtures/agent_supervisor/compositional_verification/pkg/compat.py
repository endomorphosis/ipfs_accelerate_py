"""Compatibility aliases for a one-step producer API migration."""

from .module_a import produce as produce_v2


def produce_v1(limit: int) -> int:
    """Previous producer entry point; v2 is the current admitted API."""

    return produce_v2(limit)
