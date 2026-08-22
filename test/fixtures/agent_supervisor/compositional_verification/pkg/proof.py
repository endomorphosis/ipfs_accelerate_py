"""Compact proof obligation text restored by proof-repair cases."""

OBLIGATION = "producer-upper-bound <= schema.MAX_PRODUCED_VALUE"


def obligation_text() -> str:
    """Return the current proof obligation sentence."""

    return OBLIGATION
