"""Closed supervisor lifecycle state machine for CASF."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from .contracts import (
    FederationAuthorityError,
    FederationLifecycleState,
    _integer,
)

_TRANSITIONS: Final[Mapping[FederationLifecycleState, frozenset[FederationLifecycleState]]] = (
    MappingProxyType(
        {
            FederationLifecycleState.DECLARED: frozenset(
                {
                    FederationLifecycleState.ADMITTED,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.ADMITTED: frozenset(
                {
                    FederationLifecycleState.STARTING,
                    FederationLifecycleState.PAUSED,
                    FederationLifecycleState.DRAINING,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.STARTING: frozenset(
                {
                    FederationLifecycleState.IDLE,
                    FederationLifecycleState.ACTIVE,
                    FederationLifecycleState.RECOVERING,
                    FederationLifecycleState.QUARANTINED,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.IDLE: frozenset(
                {
                    FederationLifecycleState.ACTIVE,
                    FederationLifecycleState.PAUSED,
                    FederationLifecycleState.DRAINING,
                    FederationLifecycleState.RECOVERING,
                    FederationLifecycleState.QUARANTINED,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.ACTIVE: frozenset(
                {
                    FederationLifecycleState.IDLE,
                    FederationLifecycleState.PAUSED,
                    FederationLifecycleState.DRAINING,
                    FederationLifecycleState.RECOVERING,
                    FederationLifecycleState.QUARANTINED,
                    FederationLifecycleState.FAILED,
                }
            ),
            FederationLifecycleState.PAUSED: frozenset(
                {
                    FederationLifecycleState.STARTING,
                    FederationLifecycleState.IDLE,
                    FederationLifecycleState.DRAINING,
                    FederationLifecycleState.RECOVERING,
                    FederationLifecycleState.QUARANTINED,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.DRAINING: frozenset(
                {
                    FederationLifecycleState.IDLE,
                    FederationLifecycleState.RECOVERING,
                    FederationLifecycleState.QUARANTINED,
                    FederationLifecycleState.COMPLETED,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.RECOVERING: frozenset(
                {
                    FederationLifecycleState.STARTING,
                    FederationLifecycleState.IDLE,
                    FederationLifecycleState.ACTIVE,
                    FederationLifecycleState.PAUSED,
                    FederationLifecycleState.QUARANTINED,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.QUARANTINED: frozenset(
                {
                    FederationLifecycleState.RECOVERING,
                    FederationLifecycleState.FAILED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.COMPLETED: frozenset({FederationLifecycleState.STOPPED}),
            FederationLifecycleState.FAILED: frozenset(
                {
                    FederationLifecycleState.RECOVERING,
                    FederationLifecycleState.QUARANTINED,
                    FederationLifecycleState.STOPPED,
                }
            ),
            FederationLifecycleState.STOPPED: frozenset(),
        }
    )
)


def legal_transitions(
    state: FederationLifecycleState | str,
) -> frozenset[FederationLifecycleState]:
    return _TRANSITIONS[FederationLifecycleState(state)]


def assert_transition(
    current: FederationLifecycleState | str,
    requested: FederationLifecycleState | str,
    *,
    active_effects: int = 0,
    active_attempts: int = 0,
) -> FederationLifecycleState:
    """Validate a lifecycle transition and completion safety invariants."""

    source = FederationLifecycleState(current)
    target = FederationLifecycleState(requested)
    effects = _integer(active_effects, "active_effects")
    attempts = _integer(active_attempts, "active_attempts")
    if target not in _TRANSITIONS[source]:
        raise FederationAuthorityError(
            f"illegal supervisor lifecycle transition {source.value}->{target.value}"
        )
    if target is FederationLifecycleState.COMPLETED and (effects != 0 or attempts != 0):
        raise FederationAuthorityError(
            "a supervisor with active effects or attempts cannot complete"
        )
    return target


def admits_new_work(state: FederationLifecycleState | str) -> bool:
    return FederationLifecycleState(state) in {
        FederationLifecycleState.IDLE,
        FederationLifecycleState.ACTIVE,
    }


def is_terminal(state: FederationLifecycleState | str) -> bool:
    return FederationLifecycleState(state) in {
        FederationLifecycleState.COMPLETED,
        FederationLifecycleState.FAILED,
        FederationLifecycleState.STOPPED,
    }


__all__ = [
    "admits_new_work",
    "assert_transition",
    "is_terminal",
    "legal_transitions",
]
