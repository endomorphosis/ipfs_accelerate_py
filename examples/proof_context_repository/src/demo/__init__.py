"""Tiny ordinary-Python package used by the PCCE walkthrough."""


def increment(value: int) -> int:
    """Return the next integer."""

    return value + 1


__all__ = ["increment"]
