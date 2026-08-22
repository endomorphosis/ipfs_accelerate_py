"""Security policy for caller limits admitted by the producer."""

MAXIMUM_CALLER_LIMIT = 1_000


def admit_limit(limit: int) -> bool:
    """Return whether the caller limit is inside the security policy."""

    return 0 <= limit <= MAXIMUM_CALLER_LIMIT
