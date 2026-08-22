"""Transitive consumer of module B."""

from .module_b import consume


def present(limit: int) -> int:
    value = consume(limit)
    assert 1 <= value <= 21
    return value * 2
