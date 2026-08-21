"""Producer with a finite result range and an exceptional path."""


def produce(limit: int) -> int:
    """Return a bounded value, rejecting a negative caller limit."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    return 10
