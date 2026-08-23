"""Shared lock used to keep consume/present from interfering."""

from threading import Lock

CONSUMER_LOCK = Lock()
PRESENTER_LOCK = Lock()


def ordered_locks() -> tuple[Lock, Lock]:
    """Return the documented lock order: consumer then presenter."""

    return CONSUMER_LOCK, PRESENTER_LOCK
