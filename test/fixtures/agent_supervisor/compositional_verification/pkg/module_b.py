"""Consumer whose assumption is supplied by module A."""

from .module_a import produce
from .schema import MAX_PRODUCED_VALUE


def consume(limit: int) -> int:
    value = produce(limit)
    assert 0 <= value <= MAX_PRODUCED_VALUE
    return value + 1
